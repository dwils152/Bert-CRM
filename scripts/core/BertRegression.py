# BertRegression.py

from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F

class BertRegression(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertRegression, self).__init__()
        self.num_labels = num_labels

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        self.classifier = nn.Linear(768, num_labels)

        self.return_logits = True
        self.global_step = 0

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

    # def forward(self, input_ids, attention_mask):
    #     encoder_output, _ = self.bert(
    #         input_ids=input_ids, attention_mask=attention_mask
    #     )

    #     cls_token = encoder_output[:, 0, :]  # CLS token
    #     return self.classifier(cls_token)  

    def forward(self, input_ids, attention_mask):
        # Get all token embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # shape: [batch_size, seq_len, hidden_size]
        
        # Apply the classifier to every token in the sequence
        logits = self.classifier(sequence_output)  # shape: [batch_size, seq_len, num_labels]
        
        return logits
