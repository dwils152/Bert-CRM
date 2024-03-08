import re
import sys
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

        #if 0 in input_ids:
        #    print(input_ids.tolist())
        #    print(seq)
        #    print(additional_info)
        #    sys.exit()

        return input_ids, attention_mask, labels, coordinates
