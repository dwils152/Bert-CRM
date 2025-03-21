import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Bio import SeqIO
import numpy as np
import random
from functools import partial
import sys

class SequenceDataset(Dataset):
    def __init__(
            self,
            fasta_path,
            labels_path,
            base_model,
            max_length,
        ):

        super(SequenceDataset, self).__init__()
        self.fasta_path = fasta_path
        self.fasta_list = self._read_fasta()

        self.mmap_labels = np.memmap(
            labels_path,
            dtype='int32', 
            mode='r', 
            shape=(len(self.fasta_list), 2047))
            #shape=(len(self.fasta_list), len(self.fasta_list[0])))
        self.max_length = max_length


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

        label = self.mmap_labels[idx, :]
        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()  # Remove the batch dimension
        attention_mask = inputs["attention_mask"].squeeze()
        label = torch.tensor(label).squeeze().float()

        # print(f'input: {input_ids.shape}')
        # print(f'mask: {attention_mask.shape}')
        # print(f'label: {label.shape}')

        return input_ids, attention_mask, label
