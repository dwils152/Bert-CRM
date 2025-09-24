import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Bio import SeqIO
import numpy as np
import random
from functools import partial
import sys

class SeqDatasetMulti(Dataset):
    def __init__(
            self,
            fasta_path,
            labels_path,
            base_model,
            max_length,
            smoothing=None,
            transform=None
        ):

        super(SeqDatasetMulti, self).__init__()
        self.fasta_path = fasta_path
        self.fasta_list = self._read_fasta()

        self.mmap_labels = np.memmap(
            labels_path,
            dtype='int32', 
            mode='r', 
            shape=(1177643, 2000))
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
        
        # Tokenization
        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze(0)  # shape: [max_length]
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Labels
        label = torch.from_numpy(self.mmap_labels[idx].copy())  # shape: [2000]
        
        # Mask operations (if needed)
        single_set = torch.tensor([4102, 4103, 4104, 4105, 4106])
        single_mask = torch.isin(input_ids, single_set).bool()
        n_mask = (input_ids == 4106).bool()
        
        # Apply transforms if any
        if self.transform:
            input_ids = self.transform(input_ids)
        
        return input_ids, attention_mask, label, single_mask, n_mask


if __name__ == "__main__":
    fasta_path = "/projects/zcsu_research1/dwils152/Bert-CRM/results/Human/hg38_masked_no_scaffolds_2000_0_1.0_right.fa"