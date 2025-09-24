import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import numpy as np


class SequenceDatasetEvo(Dataset):
    def __init__(
            self,
            fasta_path,
            labels_path
        ):

        super(SequenceDatasetEvo, self).__init__()
        self.fasta_path = fasta_path
        self.fasta_list = self._read_fasta()

        self.mmap_labels = np.memmap(
            labels_path,
            dtype='int8', 
            mode='r', 
            shape=(len(self.fasta_list), 8192))

    def _read_fasta(self):
        records = list(SeqIO.parse(self.fasta_path, 'fasta'))
        return records
            
    def __len__(self):
        return len(self.fasta_list)

    def __getitem__(self, idx):
        record = self.fasta_list[idx]
        sequence = str(record.seq)
        label = self.mmap_labels[idx, :]
        return sequence, label
