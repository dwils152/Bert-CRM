import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
import os
import sys
from utils import setup, tqdm_wrapper, custom_collate_fn, cleanup
from SupervisedDataset import SupervisedDataset
from UnsupervisedDataset import UnsupervisedDataset
from EarlyStopping import EarlyStopping
from BertCRFForTokenClassification import BertCRFForTokenClassification
from BertForTokenClassification import BertForTokenClassification

def main():

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    device = torch.device("cuda:{}".format(
        rank) if torch.cuda.is_available() else "cpu")

    # Initialize Dataset 
    if args.mode == 'supervised':
        dataset = SupervisedDataset(args.data_path, max_length=args.max_length)
    else:
        dataset = UnsupervisedDataset(args.data_path, max_length=args.max_length)

    # Initialize Model
    if args.use_crf:
        model = BertCRFForTokenClassification(
            args.model_name, num_labels=args.num_labels).to(device)
    else:
        model = BertForTokenClassification(
            args.model_name, num_labels=args.num_labels).to(device)

    model = DDP(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = int(0.05 * total_size)
    test_size = total_size - train_size - val_size

    # Generate indices and split them
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]

    # Create samplers with specific indices
    train_sampler = DistributedSampler(
        Subset(dataset, train_indices), shuffle=True)
    val_sampler = DistributedSampler(
        Subset(dataset, val_indices), shuffle=False)
    test_sampler = DistributedSampler(
        Subset(dataset, test_indices), shuffle=False)

    # DataLoaders
    trainloader = DataLoader(Subset(dataset, train_indices), batch_size=128,
                             sampler=train_sampler, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    valloader = DataLoader(Subset(dataset, val_indices), batch_size=128,
                           sampler=val_sampler, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    testloader = DataLoader(Subset(dataset, test_indices), batch_size=128,
                            sampler=test_sampler, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    num_epochs = 3
    # Training and Validation Loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        valid_loss = model.train_model(trainloader, valloader,
                           optimizer, device, rank)
        if rank == 0:
            print(f"Validation Loss: {valid_loss}")

    #print(f"Process {dist.get_rank()} reaching the barrier.")
    dist.barrier()
    #print(f"Process {dist.get_rank()} passed the barrier.")

    # Call the test function and get predictions and labels for each process
    model.test_model(testloader, device, rank, world_size)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for DNABERT token classification.')

    parser.add_argument('--mode', type=str, choices=['supervised', 'unsupervised'], required=True,
                        help='Training mode: supervised or unsupervised.')
    parser.add_argument('--use_crf', action='store_true',
                        help='Use CRF on top of BERT. Without this flag, CRF will not be used.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data.')
    parser.add_argument('--model_name', type=str, default='zhihan1996/DNABERT-2-117M',
                        help='Model name or path to be used with Hugging Face Transformers.')
    parser.add_argument('--num_labels', type=int, required=True,
                        help='Number of labels for token classification.')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer.')

    args = parser.parse_args()
    main(args)
