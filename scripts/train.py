from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
from torch.optim import AdamW
from pathlib import Path
from torchcrf import CRF
import regex as re
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
import os
import gc
import psutil
import datetime
import sys


def print_memory_usage():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"Memory Usage: {memory.percent}% | Total: {memory.total / (1024**3):.2f} GB | Used: {memory.used / (1024**3):.2f} GB | Free: {memory.available / (1024**3):.2f} GB")
    print(f"Swap Usage: {swap.percent}% | Total: {swap.total / (1024**3):.2f} GB | Used: {swap.used / (1024**3):.2f} GB | Free: {swap.free / (1024**3):.2f} GB")


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        print(
            f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(
            f"Reserved GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. No GPU memory usage info.")


def setup(rank, world_size):
    # Sets up the environment for distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=10_000))


def cleanup():
    # Cleanup the distributed process group
    dist.destroy_process_group()


def tqdm_wrapper(iterable, rank, *args, **kwargs):
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


class SupervisedDataset(Dataset):
    def __init__(self, data_path, max_length: int):
        super(SupervisedDataset, self).__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            'zhihan1996/DNABERT-2-117M',
            max_length=max_length,
            padding_side="right",
            padding="max_length",
            trust_remote_code=True,
        )
        self.sequences, self.labels, self.additional_infos = self._load_data()

    def _load_data(self):
        sequences = []
        labels = []
        additional_infos = []
        with open(self.data_path, "r") as fin:
            for line in fin:
                parts = re.split(r"[:|]", line.strip())
                if len(parts) == 3:  # Ensure each line has the correct format
                    seq, label_str, additional_info = parts
                    sequences.append(seq)
                    labels.append(label_str)
                    additional_infos.append(additional_info)
                else:
                    raise ValueError(f'Invalid line: {line.strip()}')
        return sequences, labels, additional_infos

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        seq = self.sequences[i]
        label_str = self.labels[i]
        additional_info = self.additional_infos[i]

        inputs = self.tokenizer(
            seq, max_length=self.max_length, padding="max_length", return_tensors="pt")

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Efficiently handle 'N' replacement and mask update
        n_token_id = self.tokenizer.convert_tokens_to_ids('N')
        mask_token_id = 4  # Assuming 4 is the mask token ID
        input_ids = torch.where(input_ids == n_token_id,
                                torch.tensor(mask_token_id), input_ids)

        mask_token_ids = [4]
        should_mask = torch.isin(input_ids, torch.tensor(mask_token_ids))
        attention_mask = torch.where(
            should_mask, torch.tensor(0), attention_mask)

        # Prepare labels
        label_int = [int(x) for x in label_str.split(',')]
        labels = torch.zeros(self.max_length, dtype=torch.long)
        labels[0] = 0
        labels[1:len(label_int)+1] = torch.tensor(label_int)

        # Calculating coordinates
        id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        coordinates = self._calculate_coordinates(
            input_ids, additional_info, id_to_token)

        if 0 in input_ids:
            print(input_ids.tolist())
            print(seq)
            print(additional_info)
            sys.exit()

        return input_ids, attention_mask, labels, coordinates

    def _calculate_coordinates(self, input_ids, additional_info, id_to_token):
        coordinates = []
        position = 0
        for input_id in input_ids.tolist():
            if input_id in {1, 2, 3}:
                coordinates.append(('XXX', 'XXX', 'XXX'))
                continue
            else:
                length = 1 if input_id == 4 else len(id_to_token[input_id])
                start = position
                end = start + length
                coordinates.append((additional_info, start, end))
                position = end
        return coordinates


class BertForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForTokenClassification, self).__init__()
        self.num_labels = num_labels

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True)
        self.classifier = nn.Linear(768, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.freeze_except_lora()

    def freeze_except_lora(self):
        for name, submodule in self.bert.named_children():
            if name == 'encoder':
                for layer in submodule.layer:
                    for param_name, param in layer.named_parameters():
                        if not any(lora_component in param_name for lora_component in ['Lora_A_Q', 'Lora_B_Q', 'Lora_A_K', 'Lora_B_K', 'Lora_A_V', 'Lora_B_V']):
                            param.requires_grad = False
            else:
                # For other submodules (like pooler),  all parameters
                for param in submodule.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        # Compute the CRF loss
        if labels is not None:
            # The CRF layer returns negative log likelihood
            crf_loss = -self.crf(logits, labels, mask=attention_mask.byte())
            # Averaging the loss over the batch
            loss = crf_loss.mean()
            return loss
        else:
            return self.crf.decode(logits), outputs


def train(model, train_loader, val_loader, optimizer, device, rank):
    model.train()
    for batch in tqdm_wrapper(train_loader, rank, desc="Training"):
        input_ids, attention_mask, labels, _ = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

        # labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.clone().detach()
        loss = model(input_ids, attention_mask, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(f"Process {dist.get_rank()} reached the barrier.")
    dist.barrier()
    # print(f"Process {dist.get_rank()} passed the barrier.")

    model.eval()
    val_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm_wrapper(val_loader, rank, desc="Validating"):
            input_ids, attention_mask, labels, _ = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            # labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.clone().detach()
            batch_loss = model(input_ids, attention_mask, labels)

            # Aggregate loss
            val_loss += batch_loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

     # Convert to torch tensor for distributed reduction
    val_loss_tensor = torch.tensor(val_loss).to(device)
    total_samples_tensor = torch.tensor(total_samples).to(device)

    # Reduce (sum) the losses and total samples across all GPUs
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    # Calculate average validation loss across all GPUs
    average_val_loss = val_loss_tensor.item() / total_samples_tensor.item()

    return average_val_loss


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def test(model, data_loader, device, rank, world_size):
    model.eval()
    # all_predictions, all_true_labels, all_coordinates = [], [], []
    all_predictions, all_true_labels = [], []
    prediction_name = f'predictions_rank_{rank}.npy'
    label_name = f'true_labels_rank_{rank}.npy'

    with torch.no_grad():
        for batch in tqdm_wrapper(data_loader, rank, desc="Testing"):
            input_ids, attention_mask, labels, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            predictions, logits = model(input_ids, attention_mask)
            predicted_labels = torch.tensor(
                [item for sublist in predictions for item in sublist], device=device)

            print(len(logits))
            print(logits[0].shape)

            flat_true_labels = labels.view(-1)
            flat_predictions = predicted_labels.view(-1)
            flat_attention_mask = attention_mask.view(-1)

            # Select only the non-padded elements
            active_positions = flat_attention_mask == 1
            active_labels = flat_true_labels[active_positions]
            active_predictions = flat_predictions[active_positions]

            all_predictions.extend(active_predictions.cpu().tolist())
            all_true_labels.extend(active_labels.cpu().tolist())

    np.save(prediction_name, np.array(all_predictions))
    np.save(label_name, np.array(all_true_labels))


def calculate_metrics(predictions, true_labels):

    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)
    return f1, mcc, accuracy, cm, auc


def main():

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    # Set device
    device = torch.device("cuda:{}".format(
        rank) if torch.cuda.is_available() else "cpu")

    # Initialize Dataset and DataLoader
    data_path = sys.argv[1]
    dataset = SupervisedDataset(data_path, max_length=1024)

    # rint(len(dataset))

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
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    # Initialize Model and Optimizer
    model = BertForTokenClassification(
        'zhihan1996/DNABERT-2-117M', num_labels=2).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Training and Validation Loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        valid_loss = train(model, trainloader, valloader,
                           optimizer, device, rank)
        if rank == 0:
            print(f"Validation Loss: {valid_loss}")

    print(f"Process {dist.get_rank()} reaching the barrier.")
    dist.barrier()
    print(f"Process {dist.get_rank()} passed the barrier.")

    # Early Stopping check
    # early_stopper(valid_loss)
    # if early_stopper.early_stop:
    #    print("Early stopping")
    #    break

    # Call the test function and get predictions and labels for each process
    test(model, testloader, device, rank, world_size)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')


def debug():
    data_path = Path('test_label_create/all_chunks.fasta.csv')
    dataset = SupervisedDataset(data_path, max_length=1024)

    vocab = dataset.tokenizer.get_vocab()
    rev_vocab = {v: k for k, v in vocab.items()}
    print(rev_vocab[4])
    print(f'sequence three')
    print(f'input_ids: {dataset[0][0]}')
    print(f'mask: {dataset[0][1]}')
    print(f'labels: {dataset[0][2]}')
    print(f'coordinates: {dataset[0][3]}')

    ids = dataset[0][0].tolist()
    print(ids)

if __name__ == "__main__":
    main()
    # debug()
