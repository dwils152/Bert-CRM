import argparse
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from Trainer import Trainer
from SequenceDataset import SequenceDataset
from transformers import AutoModelForTokenClassification
from peft import LoraConfig, TaskType, get_peft_model

#------- Multi
# from NtTransformerMulti import NtTransformerMulti
# from SeqDatasetMulti import SeqDatasetMulti
# from TrainerMulti import Trainer

def main(args):

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
        target_modules=["attention.self.query", "attention.self.value"],
        modules_to_save=["classifier"]
    )

    model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=1,
            trust_remote_code=True
    ).to(device)

    model = get_peft_model(model, lora_config)
    
    dataset = SequenceDataset(
        args.fasta,
        args.labels,
        args.model_name,
        2048,
    )

    # Create a subset of the data (currently using 100% of the data)
    total_indices = list(range(len(dataset)))
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Shuffle indices
    total_indices = torch.randperm(len(dataset)).tolist()
    subset_size = int(len(dataset) * 1.0)  # Use 100% of the data
    subset_indices = total_indices[:subset_size]
    subset = torch.utils.data.Subset(dataset, subset_indices)

    # Calculate split sizes
    total_size = len(subset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_indices = subset_indices[:train_size]
    val_indices = subset_indices[train_size:train_size + val_size]
    test_indices = subset_indices[train_size + val_size:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=2,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=4,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=4,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        writer=writer,
    )

    # Start training
    trainer.train(num_epochs=args.num_epochs)
    trainer.test(test_loader)
    model.save_pretrained('model.pt')
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script'
    )

    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to the dataset (FASTA format).')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to the labels file.')
    parser.add_argument('--model_name', type=str,
                        default='InstaDeepAI/nucleotide-transformer-v2-250m-multi-species',
                        help='Pre-trained model name or path.')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--run_name', type=str, default='default_run',
                        help='Name for the training run (used in logging).')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients.')

    args = parser.parse_args()
    main(args)
