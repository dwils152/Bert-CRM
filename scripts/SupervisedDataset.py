import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
        self.line_offsets = []  # Store line offsets for random access

        with open(data_path, "r") as fin:
            offset = 0
            for line in fin:
                self.line_offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, i):
        with open(self.data_path, "r") as fin:
            # Read the line corresponding to index i
            fin.seek(self.line_offsets[i])
            line = fin.readline()

            # Extract sequence, labels, and additional info (like chromosome)
            parts = re.split(r"[:|]", line)
            seq, label_str, additional_info = parts[0].strip(
            ), parts[1].strip(), parts[2].strip()

            # Tokenization
            inputs = self.tokenizer(
                seq, max_length=self.max_length, padding="max_length", return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]

            # Prepare labels
            label_int = [int(x) for x in label_str.split(',')]
            label_int.extend([0] * (self.max_length - len(label_int)))
            labels = torch.Tensor(label_int)

            # Reverse mapping from token IDs to strings
            id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}

            # Calculate token lengths and map to coordinates
            token_lengths = [len(id_to_token.get(token_id, ""))
                             for token_id in input_ids.tolist() if token_id not in {1, 2, 3}]
            coordinates = []
            position = 0
            for length in token_lengths:
                start = position
                end = start + length
                # Using additional_info as chromosome placeholder
                coordinates.append((additional_info, start, end))
                position = end

            return input_ids, attention_mask, labels, coordinates


class UnsupervisedDataset(Dataset):
    def __init__(self, data_path, max_length: int):
        super(UnsupervisedDataset, self).__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            'zhihan1996/DNABERT-2-117M',
            max_length=max_length,
            padding_side="right",
            padding="max_length",
            trust_remote_code=True,
        )

        self.sequence_data = []  # Store sequences and headers
        current_sequence = ""
        current_header = ""
        with open(data_path, "r") as fin:
            for line in fin:
                if line.startswith(">"):  # FASTA header line
                    if current_sequence:
                        self.sequence_data.append(
                            (current_header, current_sequence))
                        current_sequence = ""
                    # Remove '>' and store header
                    current_header = line.strip()[1:]
                else:
                    current_sequence += line.strip()
            if current_sequence:  # Add the last sequence if exists
                self.sequence_data.append((current_header, current_sequence))

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, i):
        header, seq = self.sequence_data[i]

        # Tokenization
        inputs = self.tokenizer(
            seq, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Reverse mapping from token IDs to strings
        id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        id_to_token_len = {key: len(value)
                           for key, value in id_to_token.items()}

        # print(input_ids)

        # Calculate token lengths and map to coordinates
        token_lengths = [len(id_to_token.get(token_id, ""))
                         for token_id in input_ids.tolist() if token_id not in {1, 2, 3}]

        coordinates = []
        position = 0

        for token_id in input_ids.tolist():
            if token_id in {1, 2, 3}:
                coordinates.append(('XXX', 'XXX', 'XXX'))
            else:
                start = position
                end = start + id_to_token_len[token_id]
                coordinates.append((header, start, end))
                position = end

        '''for length in token_lengths:
            start = position
            end = start + length
            # Use header instead of chromosome info
            coordinates.append((header, start, end))
            position = end'''

        return input_ids, attention_mask, coordinates


def debug():
    dataset = UnsupervisedDataset(
        "../results/split_genomes/panTro4_1000_0_1.0.fa", 1024)

    input_ids, attention_mask, coordinates = dataset[0]

    print(f'input_ids: {len(input_ids)}')
    print(f'attention_mask: {len(attention_mask)}')
    print(f'coordinates: {len(coordinates)}')


# debug()
