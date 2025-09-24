import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import tqdm as tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from bert_crm.utils.utils import (
    setup,
    unsupervised_collate_fn,
    cleanup
)
from bert_crm.models.UnsupervisedDataset import UnsupervisedDataset
from bert_crm.models.BertCRMClassification import BertCRMClassification


def main(args):

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    device = torch.device("cuda:{}".format(
        rank) if torch.cuda.is_available() else "cpu")

    model = BertCRMClassification(
            args.model_name, num_labels=2).to(device)
    model = DDP(model, device_ids=[rank])

    # Initialize Dataset
    dataset = UnsupervisedDataset(args.data_path, max_length=args.max_length)
    collate_fn = unsupervised_collate_fn
    sampler = DistributedSampler(dataset, shuffle=True)

    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, 
                            shuffle=False, num_workers=16, collate_fn=collate_fn)

    model.eval()
    local_embeddings = []
    if rank == 0:
        #data_iter = tqdm.tqdm(dataloader, desc="Generating embeddings",
        #                 total=len(dataloader))
        data_iter = dataloader
    else:
        data_iter = dataloader

    with torch.no_grad():
        for batch in data_iter:
            inputs, attention_mask, _ = batch
            inputs.to(device)
            attention_mask.to(device)
            outputs = model(inputs, attention_mask)
            local_embeddings.append(outputs)
            print(f'outputs shape: {outputs.shape}')

    # Concatenate all embeddings
    local_embeddings = torch.cat(local_embeddings, dim=0)
    print(f'local_embeddings shape: {local_embeddings.shape}')

    gathered_embeddings = [torch.zeros_like(local_embeddings)
                            for _ in range(world_size)]
    print(f'gathered_embeddings shape: {gathered_embeddings[0].shape}') 
    dist.all_gather(gathered_embeddings, local_embeddings)

    if rank == 0:
        all_embeddings = torch.cat(gathered_embeddings, dim=0)
        all_embeddings_npy = all_embeddings.cpu().numpy()
        np.save("embeddings.npy", all_embeddings_npy)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating DNA sequence embeddings.')

    parser.add_argument('--data_path', type=str,
                        help='Path to the training data.')
    parser.add_argument('--model_name', type=str, default='zhihan1996/DNABERT-2-117M',
                        help='Model name or path to be used with Hugging Face Transformers.')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length for the inputs.')

    args = parser.parse_args()
    main(args)
