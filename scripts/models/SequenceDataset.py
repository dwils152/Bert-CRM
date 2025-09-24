import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SequenceDataset(Dataset):
    def __init__(self, fasta_path: str, max_length: int):
        super(SequenceDataset, self).__init__()
        self.fasta_path = fasta_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            'zhihan1996/DNABERT-2-117M',
            max_length=max_length,
            padding_side="right",
            truncation=True,
            padding="max_length",
            trust_remote_code=True
        )
        self.positions_labels = self._get_positions(self.fasta_path)

    def _get_positions(self, fasta_path):
        positions_labels = []
        with open(fasta_path, "r") as file:
            pos = file.tell()
            line = file.readline()
            while line:
                if line.startswith(">"):
                    # Check if the header contains '_pos' or '_neg'
                    label = 1 if '_pos' in line else 0
                    sequence_pos = file.tell()  # Position of the sequence following the header
                    positions_labels.append((sequence_pos, label))
                pos = file.tell()
                line = file.readline()
        return positions_labels

    def __len__(self):
        return len(self.positions_labels)

    def _read_sequence(self, position):
        with open(self.fasta_path, "r") as fin:
            fin.seek(position)
            return fin.readline().strip()

    def __getitem__(self, idx):
        position, label = self.positions_labels[idx]
        sequence = self._read_sequence(position)

        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)  # Remove the batch dimension
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        return input_ids, attention_mask, torch.tensor(label)



        

        

        

