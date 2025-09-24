# train.py

import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from Trainer import Trainer  # Ensure Trainer.py is in the same directory or adjust the import path
from BertRegression import BertRegression
from SequenceDatasetRegression import SequenceDatasetRegression

def setup(rank, world_size):
    """
    Initializes the default process group.
    """
    os.environ['MASTER_ADDR'] = 'localhost'  # Modify if running on multiple nodes
    os.environ['MASTER_PORT'] = '12355'      # Modify if running on multiple nodes
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    Destroys the default process group.
    """
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    # Initialize the model and compile it
    model = BertRegression(args.model_name, num_labels=1000).to(device)
    compiled_model = torch.compile(model, backend="cudagraphs")  # You can experiment with different backends
    
    # Wrap the compiled model with DDP
    ddp_model = DDP(compiled_model, device_ids=[rank], output_device=rank)
    
    # Initialize the dataset
    dataset = SequenceDatasetRegression(
        args.fasta,
        args.labels,
        args.model_name,
        args.max_length,
    )
    
    # Use DistributedSampler
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Shuffle is handled by DistributedSampler
        num_workers=8,  # Adjust based on CPU cores
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True
    )
    
    # Initialize the optimizer
    optimizer = AdamW(ddp_model.parameters(), lr=args.learning_rate)
    
    # Define the loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Initialize TensorBoard writer only on rank 0
    writer = SummaryWriter(log_dir=f"runs/{args.run_name}") if rank == 0 else None
    
    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()
    
    # Initialize the Trainer
    trainer = Trainer(
        model=ddp_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=None,  # Add validation loader if needed
        device=device,
        writer=writer,
        scaler=scaler,
        profiler=None,  # Add profiler if needed
        accumulation_steps=args.accumulation_steps,
    )
    
    # Start training
    trainer.train(num_epochs=args.num_epochs)
    
    # Close the TensorBoard writer
    if writer:
        writer.close()
    
    cleanup()

def main():
    """If your script expects `--local-rank` argument to be set, please
    change it to read from `os.environ['LOCAL_RANK']` instead. See 
    https://pytorch.org/docs/stable/distributed.html#launch-utility for 
    further instructions"""
    local_rank = int(os.environ['LOCAL_RANK'])
    
    parser = argparse.ArgumentParser(description='Training script with torch.compile and DDP.')
    
    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to the dataset (FASTA format).')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to the labels file.')
    parser.add_argument('--model_name', type=str,
                        default='/path/to/DNABERT-2-117M',
                        help='Pre-trained model name or path.')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--run_name', type=str, default='multi_gpu_run',
                        help='Name for the training run (used in logging).')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU.')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use.')
    
    args = parser.parse_args()
    
    world_size = args.world_size
    
    if world_size > torch.cuda.device_count():
        raise ValueError(f"Requested world_size {world_size} exceeds available GPUs {torch.cuda.device_count()}")
    
    # Spawn one process per GPU
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
