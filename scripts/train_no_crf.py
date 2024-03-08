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
                    # raise ValueError(f'Invalid line: {line.strip()}')
                    pass
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

        # Define mask token IDs based on your context (this is just an example)
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

        self.bert.pooler = None
        self.classifier = nn.Linear(768, self.num_labels)
        # self.crf = CRF(self.num_labels, batch_first=True)
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

        return logits

        # Compute the CRF loss
        if labels is not None:
            # The CRF layer returns negative log likelihood
            crf_loss = -self.crf(logits, labels, mask=attention_mask.byte())
            # Averaging the loss over the batch
            loss = crf_loss.mean()
            return loss
        else:
            return self.crf.decode(logits), outputs


def train(model, train_loader, val_loader, optimizer, device, rank, loss_fn):
    model.train()
    for batch in tqdm_wrapper(train_loader, rank, desc="Training"):
        input_ids, attention_mask, labels, _ = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

        # labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.clone().detach()
        logits = model(input_ids, attention_mask, labels)
        logits = logits.view(-1, 2)
        labels = labels.view(-1)

        active_loss = attention_mask.view(-1) == 1
        active_outputs = logits[active_loss]
        active_labels = labels[active_loss]
        loss = loss_fn(active_outputs, active_labels.long())
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
            logits = model(input_ids, attention_mask, labels)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

            active_loss = attention_mask.view(-1) == 1
            active_outputs = logits[active_loss]
            active_labels = labels[active_loss]
            loss = loss_fn(active_outputs, active_labels.long())
            val_loss += loss.item()

    return val_loss / len(val_loader)


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
    predictions, true_labels, max_proba = [], [], []
    prediction_name = f'predictions_rank_{rank}.npy'
    label_name = f'true_labels_rank_{rank}.npy'
    proba_name = f'proba_rank_{rank}.npy'

    with torch.no_grad():

        for batch in tqdm_wrapper(data_loader, rank, desc="Testing"):
            input_ids, attention_mask, labels, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            logits = model(input_ids, attention_mask)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

            active_loss = attention_mask.view(-1) == 1
            active_logits = logits[active_loss]
            active_labels = labels[active_loss]

            preds = torch.argmax(active_logits, dim=1)
            probabilities = torch.softmax(active_logits, dim=1)
            pos_proba = probabilities[:, 1]

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(active_labels.cpu().numpy())
            # Storing max probability of each prediction
            max_proba.extend(pos_proba.cpu().numpy())

    # Save the results as NumPy arrays
    np.save(prediction_name, np.array(predictions))
    np.save(label_name, np.array(true_labels))
    np.save(proba_name, np.array(max_proba))


def main():

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    torch.autograd.set_detect_anomaly(True)

    # Set device
    device = torch.device("cuda:{}".format(
        rank) if torch.cuda.is_available() else "cpu")

    # Initialize Dataset and DataLoader
    data_path = sys.argv[1]
    dataset = SupervisedDataset(data_path, max_length=1024)

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
    trainloader = DataLoader(Subset(dataset, train_indices), batch_size=32,
                             sampler=train_sampler, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    valloader = DataLoader(Subset(dataset, val_indices), batch_size=32,
                           sampler=val_sampler, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    testloader = DataLoader(Subset(dataset, test_indices), batch_size=32,
                            sampler=test_sampler, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    num_epochs = 3
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    # Initialize Model and Optimizer
    model = BertForTokenClassification(
        'zhihan1996/DNABERT-2-117M', num_labels=2).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Training and Validation Loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        valid_loss = train(model, trainloader, valloader,
                           optimizer, device, rank, loss_fn)
        if rank == 0:
            print(f"Validation Loss: {valid_loss}")

    # print(f"Process {dist.get_rank()} reaching the barrier.")
    dist.barrier()
    # print(f"Process {dist.get_rank()} passed the barrier.")

    # Early Stopping check
    # early_stopper(valid_loss)
    # if early_stopper.early_stop:
    #    print("Early stopping")
    #    break

    # Call the test function and get predictions and labels for each process
    test(model, testloader, device, rank, world_size)
    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')



if __name__ == "__main__":
    main()
