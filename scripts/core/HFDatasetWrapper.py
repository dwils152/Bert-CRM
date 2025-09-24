from SequenceDataset import SequenceDataset

class HFDatasetWrapper(SequenceDataset):
    def __init__(self, seq_dataset):
        
        self.seq_dataset = seq_dataset

    def __len__(self):
        return len(self.seq_dataset)

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.seq_dataset[idx]
        # Return a dict that Trainer expects:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
