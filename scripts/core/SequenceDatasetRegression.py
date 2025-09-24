import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Bio import SeqIO
import numpy as np
import random
from functools import partial
import sys

class SequenceDatasetRegression(Dataset):
    def __init__(
            self,
            fasta_path,
            labels_path,
            base_model,
            max_length,
            smoothing=None,
            transform=None
        ):

        super(SequenceDatasetRegression, self).__init__()
        self.fasta_path = fasta_path
        self.fasta_list = self._read_fasta()

        self.mmap_labels = np.memmap(
            labels_path,
            dtype='int32', 
            mode='r', 
            shape=(5945, 2000))
            #shape=(len(self.fasta_list), len(self.fasta_list[0])))

        self.max_length = max_length
        self.smoothing = smoothing
        self.transform = transform

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            max_length=max_length,
            return_tensors="pt",
            padding_side="right",
            truncation=True,
            padding="max_length",
            trust_remote_code=True
        )

    def _read_fasta(self):
        records = list(SeqIO.parse(self.fasta_path, 'fasta'))
        return records
            
    def __len__(self):
        return len(self.fasta_list)

    def __getitem__(self, idx):
        record = self.fasta_list[idx]
        sequence = str(record.seq)
        
        # Extract pad value
        pad_str = record.id.split("|")[-1].split("=")[-1]
        pad = int(pad_str)
        if pad < 0 or pad > self.max_length:
            raise ValueError(f"Pad value {pad} exceeds max_length {self.max_length} at index {idx}")
        
        # Must check if the pad value is greater 0, otherwise the mask will be all zeros
        nt_mask = torch.ones(self.max_length)
        if pad > 0:
            nt_mask[-pad:] = 0

        label = self.mmap_labels[idx, :]
        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze(0)  # Remove the batch dimension
        attention_mask = inputs["attention_mask"].squeeze(0)

        if self.smoothing:
            label = self.smoothing(label)

        if self.transform:
            return input_ids, attention_mask, self.transform(torch.tensor(label))
        else:
            return input_ids, attention_mask, torch.tensor(label)
