from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sys


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
        # print(f'the header is {header}')

        # Tokenization
        inputs = self.tokenizer(
            seq, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Alter the attention mask to ignore the [UNK] token
        # This is necessary because the [UNK] token is not counted
        # in the attention mask.
        for i in range(len(input_ids)):
            if input_ids[i] == 0:
                attention_mask[i] = 0

        # Reverse mapping from token IDs to strings
        id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        id_to_token_len = {key: len(value)
                           for key, value in id_to_token.items()}

        coordinates = []
        position = 0

        for token_id in input_ids.tolist():
            # CLS, SEP, and PAD tokens
            if token_id in {1, 2, 3}:
                coordinates.append(('XXX', 'XXX', 'XXX'))
            # UNK token, For example N will get token_id 0
            # because it's not in the vocabulary.
            elif token_id == 0:
                start = position
                end = start + 1
                coordinates.append((header, start, end))
                position = end
            else:
                start = position
                end = start + id_to_token_len[token_id]
                coordinates.append((header, start, end))
                position = end

        return input_ids, attention_mask, coordinates


if __name__ == '__main__':
    data_path = sys.argv[1]
    max_length = 1024
    dataset = UnsupervisedDataset(data_path, max_length)
    token_dict = dataset.tokenizer.get_vocab()
    tokenizer = dataset.tokenizer

    # inputs = tokenizer('NNNNNNATA', max_length=20,
    #                   padding="max_length", return_tensors="pt")
    # print(inputs)
    # print(len(dataset))
    # print(dataset[0])
