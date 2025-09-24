import argparse
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset, Sampler
import os
import bert_crm
from bert_crm.utils.utils import (
    setup,
    supervised_collate_fn,
    unsupervised_collate_fn,
    cleanup
)
from bert_crm.models.BertCRMClassification import BertCRMClassification
from bert_crm.models.SequenceDataset import SequenceDataset


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertCRMClassification('zhihan1996/DNABERT-2-117M', 2).to(device)
    #model = DDP(model, device_ids=[rank])
    dataset = SequenceDataset(args.fasta, max_length=args.max_length)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Generate indices and split them
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]

    # Create subsets for each split
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoader for each split
    trainloader = DataLoader(train_subset, batch_size=16, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=16, shuffle=False)
    testloader = DataLoader(test_subset, batch_size=16, shuffle=False)





    num_epochs = 10
    # Training and Validation Loop
    for epoch in range(num_epochs):
        #train_sampler.set_epoch(epoch)
        valid_loss = model.train_model(trainloader, valloader,
                                              optimizer, device)
        print(f"Validation Loss: {valid_loss}")

    model.test_model(testloader, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script for DNABERT token classification.')

    parser.add_argument('--fasta', type=str,
                        help='Path to the  dataset.')
    parser.add_argument('--model_name', type=str, default='zhihan1996/DNABERT-2-117M',
                        help='Model name or path to be used with Hugging Face Transformers.')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels for token classification.')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')

    args = parser.parse_args()
    main(args)
