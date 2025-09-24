import argparse
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
import os
import bert_crm
from bert_crm.utils.utils import (
    setup,
    supervised_collate_fn,
    unsupervised_collate_fn,
    cleanup
)
from bert_crm.models.SupervisedDataset import SupervisedDataset
from bert_crm.models.BertCRFForTokenClassification import BertCRFForTokenClassification
from bert_crm.models.BertForTokenClassification import BertForTokenClassification


def main(args):

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    device = torch.device("cuda:{}".format(
        rank) if torch.cuda.is_available() else "cpu")

    # Initialize Model
    if args.use_crf:
        model = BertCRFForTokenClassification(
            args.model_name, num_labels=args.num_labels).to(device)
    else:
        model = BertForTokenClassification(
            args.model_name, num_labels=args.num_labels).to(device)

    model = DDP(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    if args.use_splits:

        collate_fn = supervised_collate_fn

        train_set = SupervisedDataset(
            args.train_split, max_length=args.max_length)
        val_set = SupervisedDataset(args.val_split, max_length=args.max_length)
        test_set = SupervisedDataset(
            args.test_split, max_length=args.max_length)

        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        test_sampler = DistributedSampler(test_set, shuffle=False)

        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=128, num_workers=8, sampler=train_sampler, shuffle=False, collate_fn=collate_fn)
        valloader = torch.utils.data.DataLoader(
            val_set, batch_size=128, num_workers=8, sampler=val_sampler, shuffle=False, collate_fn=collate_fn)
        testloader = torch.utils.data.DataLoader(
            test_set, batch_size=128, num_workers=8, sampler=test_sampler, shuffle=False, collate_fn=collate_fn)

    else:

        # Initialize Dataset
        dataset = SupervisedDataset(args.data_path, max_length=args.max_length)
        collate_fn = supervised_collate_fn

        # Calculate split sizes
        total_size = len(dataset) // 100
        train_size = int(0.9 * total_size)
        val_size = int(0.05 * total_size)
        test_size = total_size - train_size - val_size

        # Generate indices and split them
        indices = torch.randperm(total_size).tolist()[:total_size]
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
        trainloader = DataLoader(Subset(dataset, train_indices), batch_size=16,
                                 sampler=train_sampler, shuffle=False, num_workers=16, collate_fn=collate_fn)
        valloader = DataLoader(Subset(dataset, val_indices), batch_size=16,
                               sampler=val_sampler, shuffle=False, num_workers=16, collate_fn=collate_fn)
        testloader = DataLoader(Subset(dataset, test_indices), batch_size=16,
                                sampler=test_sampler, shuffle=False, num_workers=16, collate_fn=collate_fn)

    num_epochs = 3
    # Training and Validation Loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        valid_loss = model.module.train_model(trainloader, valloader,
                                              optimizer, device, rank)
        if rank == 0:
            print(f"Validation Loss: {valid_loss}")

    # print(f"Process {dist.get_rank()} reaching the barrier.")
    dist.barrier()
    # print(f"Process {dist.get_rank()} passed the barrier.")

    # Call the test function and get predictions and labels for each process
    model.module.test_model(testloader, device, rank)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')

    cleanup()


def debug():

    print(sys.path)
    data_path = '../../supervised_results_0.05/mouse/labels_chunk/chunk_98.fa.csv'
    dataset = SupervisedDataset(data_path, max_length=args.max_length)
    for i in range(1):
        coords = dataset[i][3]
        if coords[0].split('-')[0] == 'chr13':
            print(coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script for DNABERT token classification.')

    parser.add_argument('--use_crf', action='store_true',
                        help='Use CRF on top of BERT. Without this flag, CRF will not be used.')
    parser.add_argument('--data_path', type=str,
                        help='Path to the training data.')
    parser.add_argument('--use_splits', action='store_true',
                        help='Use splits for training, validation, and testing.')
    parser.add_argument('--train_split', type=str,
                        help='Path to the training split file.')
    parser.add_argument('--val_split', type=str,
                        help='Path to the validation split file.')
    parser.add_argument('--test_split', type=str,
                        help='Path to the testing split file.')
    parser.add_argument('--model_name', type=str, default='/users/dwils152/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/25abaf0bd247444fcfa837109f12088114898d98',
                        help='Model name or path to be used with Hugging Face Transformers.')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels for token classification.')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer.')

    args = parser.parse_args()
    main(args)
    # debug()
