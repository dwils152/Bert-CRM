import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from transformers import AutoTokenizer
from BertForTokenClassification import BertForTokenClassification
from SupervisedDataset import SupervisedDataset, UnsupervisedDataset
from tqdm import tqdm
import argparse
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import sys


def setup_ddp(rank, world_size):
    # Sets up the environment for distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    # Cleanup the distributed process group
    dist.destroy_process_group()


def tqdm_wrapper(iterable, rank, *args, **kwargs):
    # Only prints the progress bar for one of the GPUs
    if rank == 0:
        return tqdm(iterable, *args, **kwargs)
    else:
        return iterable


def custom_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    coordinates = [item[3] for item in batch]  # List of coordinate lists

    return input_ids, attention_mask, labels, coordinates


def custom_collate_fn_unsupervised(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    coordinates = [item[2] for item in batch]  # List of coordinate lists

    return input_ids, attention_mask, coordinates


def setup_model(model_path, use_ddp=False):
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertForTokenClassification(
        'zhihan1996/DNABERT-2-117M', num_labels=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if use_ddp:
        new_state_dict = {
            'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}
    else:
        new_state_dict = {k.replace('module.', '')
                                    : v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)

    return tokenizer, model


'''def predict(model, data_loader, device, rank, world_size):
    model.eval()
    all_predictions, all_true_labels, all_coordinates = [], [], []

    with torch.no_grad():
        for batch in tqdm_wrapper(data_loader, rank, desc="Predicting"):
            input_ids, attention_mask, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            predictions, logits = model(input_ids, attention_mask)
            predicted_labels = torch.tensor(
                [item for sublist in predictions for item in sublist], device=device)

            flat_predictions = predicted_labels.reshape(-1)
            flat_attention_mask = attention_mask.reshape(-1)
            flat_input_ids = input_ids.reshape(-1)
            flat_coordinates = [
                item for sublist in coordinates for item in sublist]

            # Mask for active positions (not padded)
            active_positions = flat_attention_mask == 1

            # Exclude specific tokens (CLS, SEP, padding)
            exclude_tokens = (flat_input_ids != 1) & (
                flat_input_ids != 2) & (flat_input_ids != 3)
            final_mask = active_positions & exclude_tokens

            active_predictions = flat_predictions[final_mask]
            active_coordinates = flat_coordinates

            all_predictions.extend(active_predictions.tolist())
            all_coordinates.extend(active_coordinates)

    # Write results to a file
    file_name = f'predictions_with_coordinates_rank_{rank}.txt'
    with open(file_name, 'w+') as f:
        for coord, pred in zip(all_coordinates, all_predictions):
            f.write(f"{coord}: {pred}\n")

import torch'''


def predict(model, data_loader, device, rank, world_size):
    model.eval()
    file_name = f'predictions_with_coordinates_rank_{rank}.txt'

    with torch.no_grad(), open(file_name, 'w+') as f:
        for batch in tqdm_wrapper(data_loader, rank, desc="Predicting"):
            input_ids, attention_mask, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            predictions, _ = model(input_ids, attention_mask)
            # print(len(predictions[0]))

            predictions = torch.tensor(
                [item for sublist in predictions for item in sublist], device=device)

            # Flatten coordinates to match predictions' shape
            coordinates = [item for sublist in coordinates for item in sublist]

            # print(f'length of predictions: {len(predictions)}')
            # print(f'length of coordinates: {len(coordinates)}')

            # Flatten input_ids and attention_mask for processing
            flat_input_ids = input_ids.view(-1)
            flat_attention_mask = attention_mask.view(-1)

            # Mask for active positions (not padded)
            active_positions = flat_attention_mask == 1

            # Create a mask to exclude specific tokens. Adjust token values as necessary.
            exclude_tokens = (flat_input_ids != 1) & (
                flat_input_ids != 2) & (flat_input_ids != 3)

            # Combine masks for final filtering
            final_mask = active_positions & exclude_tokens

            # Apply final mask to predictions and coordinates
            active_predictions = predictions[final_mask]
            active_coordinates = [coordinates[i]
                                  for i in range(len(coordinates)) if final_mask[i]]

            # print(f'length of active predictions: {len(active_predictions)}')
            # print(f'length of active coordinates: {len(active_coordinates)}')
            # sys.exit()

            # Write results directly to file
            for coord, pred in zip(active_coordinates, active_predictions.tolist()):
                f.write(f"{coord}: {pred}\n")


def predict_w_metrics(model, data_loader, device, rank, world_size):
    model.eval()
    all_predictions, all_true_labels, all_coordinates = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            predictions, logits = model(input_ids, attention_mask)
            predicted_labels = torch.tensor(
                [item for sublist in predictions for item in sublist], device=device)

            flat_true_labels = labels.reshape(-1)
            flat_predictions = predicted_labels.reshape(-1)
            flat_attention_mask = attention_mask.reshape(-1)
            flat_input_ids = input_ids.reshape(-1)
            flat_coordinates = [
                item for sublist in coordinates for item in sublist]

            # Mask for active positions (not padded)
            active_positions = flat_attention_mask == 1

            # Exclude specific tokens (CLS, SEP, padding)
            exclude_tokens = (flat_input_ids != 1) & (
                flat_input_ids != 2) & (flat_input_ids != 3)
            final_mask = active_positions & exclude_tokens

            active_labels = flat_true_labels[final_mask]
            active_predictions = flat_predictions[final_mask]
            active_coordinates = [flat_coordinates[i]
                                  for i in range(len(final_mask)) if final_mask[i]]

            all_predictions.extend(active_predictions.tolist())
            all_true_labels.extend(active_labels.tolist())
            all_coordinates.extend(active_coordinates)

    # Convert lists to tensors and coordinates to a flat list
    all_predictions_tensor = torch.tensor(
        all_predictions, dtype=torch.long, device=device)
    all_true_labels_tensor = torch.tensor(
        all_true_labels, dtype=torch.long, device=device)
    all_coordinates_flat = sum(all_coordinates, [])

    # Gather all predictions, labels, and coordinates from different processes
    gathered_predictions = [torch.zeros_like(
        all_predictions_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(
        all_true_labels_tensor) for _ in range(world_size)]
    gathered_coordinates = [torch.zeros_like(torch.tensor(
        all_coordinates_flat)) for _ in range(world_size)]

    dist.all_gather(gathered_predictions, all_predictions_tensor)
    dist.all_gather(gathered_labels, all_true_labels_tensor)
    dist.all_gather(gathered_coordinates, torch.tensor(
        all_coordinates_flat, device=device))

    if rank == 0:
        # Concatenate results from all processes on master process
        all_predictions_tensor = torch.cat(gathered_predictions)
        all_true_labels_tensor = torch.cat(gathered_labels)

        return all_predictions_tensor, all_true_labels_tensor


def calculate_metrics(predictions, true_labels):

    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)
    return f1, mcc, accuracy, cm, auc


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument("--local_rank", type=int,
                           help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = argparser.parse_args()

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_ddp(rank, world_size)

    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    tokenizer, model = setup_model(args.model_path)
    model = model.to(f'cuda:{rank}')
    model = DDP(model, device_ids=[rank])
    data_path = args.data_path
    dataset = UnsupervisedDataset(data_path, 1024)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=8, sampler=sampler, collate_fn=custom_collate_fn_unsupervised)
    predict(model, dataloader, device, rank, world_size)

    cleanup()


def debug():
    dataset = SupervisedDataset("data/hg38_all_chunks.shuf.head.csv", 1024)
    for i in range(len(dataset)):
        item = dataset[i]
        print(item)


if __name__ == "__main__":
    main()
    # debug()
