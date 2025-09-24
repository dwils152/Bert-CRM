import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForTokenClassification,
    PreTrainedModel,
    AutoConfig
)

class CRMTokenClassifier(PreTrainedModel):
    def __init__(self, model_name):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = 1

        super().__init__(config)
        # Rename the attribute to avoid conflict
        self._base_model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config
        )

        self.post_init()

    @property
    def base_model(self):
        # Return the underlying model when the base_model property is accessed.
        return self._base_model

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):

        # Use the underlying model for inference.
        outputs = self._base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        
        logits = outputs.logits
        logits = logits.squeeze(2)[:, 1:]

        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fn(logits, labels).mean()

        return {"logits": logits,
                "loss": loss}
