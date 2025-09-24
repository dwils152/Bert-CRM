import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
)

class NtModel(nn.Module):
    def __init__(self, num_labels=1):  # For binary classification
        super().__init__()

        self.base_model = AutoModelForTokenClassification.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            num_labels=num_labels,
            trust_remote_code=True
        )
        
        #self._freeze_all_except_last_n(2)

    # def _freeze_all_except_last_n(self, n: int):
    #     """Freeze all parameters except last n transformer layers and classifier"""
    #     # Freeze entire base model first
    #     for param in self.base_model.parameters():
    #         param.requires_grad = False
            
    #     # Unfreeze last n transformer layers
    #     total_layers = len(self.base_model.encoder.layer)
    #     layers_to_unfreeze = self.base_model.encoder.layer[-n:]
        
    #     print(f"Unfreezing last {n}/{total_layers} layers:")
    #     for layer in layers_to_unfreeze:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    #     # Always keep classifier trainable
    #     for param in self.classifier.parameters():
    #         param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use last hidden state [batch_size, seq_len, hidden_size]
        #hidden_states = outputs.hidden_states[-1]
        #logits = self.classifier(hidden_states)
        #return logits  # [batch_size, seq_len, num_labels]
        return outputs