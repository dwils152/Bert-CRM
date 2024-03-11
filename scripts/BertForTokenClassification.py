from transformers import AutoModel
import torch
from torch import nn
from utils import tqdm_wrapper
import numpy as np
import torch.distributed as dist

class BertForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForTokenClassification, self).__init__()
        self.num_labels = num_labels

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True)

        self.bert.pooler = None
        self.classifier = nn.Linear(768, self.num_labels)
        self.freeze_except_lora()
        self.loss_fn = nn.CrossEntropyLoss()

    def freeze_except_lora(self):
        for name, submodule in self.bert.named_children():
            if name == 'encoder':
                for layer in submodule.layer:
                    for param_name, param in layer.named_parameters():
                        if not any(lora_component in param_name for lora_component in 
                                   ['Lora_A_Q', 'Lora_B_Q', 'Lora_A_K', 'Lora_B_K', 'Lora_A_V', 'Lora_B_V']):
                            param.requires_grad = False
            else:
                # For other submodules (like pooler),  all parameters
                for param in submodule.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits

    def train_model(self, train_loader, val_loader, optimizer, device, rank):
        self.train()
        for batch in tqdm_wrapper(train_loader, rank, desc="Training"):
            input_ids, attention_mask, labels, _ = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            # labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.clone().detach()
            logits = self.forward(input_ids, attention_mask, labels)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

            active_loss = attention_mask.view(-1) == 1
            active_outputs = logits[active_loss]
            active_labels = labels[active_loss]
            loss = self.loss_fn(active_outputs, active_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f"Process {dist.get_rank()} reached the barrier.")
        dist.barrier()
        # print(f"Process {dist.get_rank()} passed the barrier.")

        self.eval()
        val_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm_wrapper(val_loader, rank, desc="Validating"):
                input_ids, attention_mask, labels, _ = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                # labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.clone().detach()
                logits = self.forward(input_ids, attention_mask, labels)
                logits = logits.view(-1, 2)
                labels = labels.view(-1)

                active_loss = attention_mask.view(-1) == 1
                active_outputs = logits[active_loss]
                active_labels = labels[active_loss]
                loss = self.loss_fn(active_outputs, active_labels.long())
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def test_model(self, data_loader, device, rank):
        self.eval()
        predictions, true_labels, max_proba = [], [], []
        prediction_name = f'predictions_rank_{rank}.npy'
        label_name = f'true_labels_rank_{rank}.npy'
        proba_name = f'proba_rank_{rank}.npy'

        with torch.no_grad():

            for batch in tqdm_wrapper(data_loader, rank, desc="Testing"):
                input_ids, attention_mask, labels, coordinates = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                logits = self.forward(input_ids, attention_mask)
                logits = logits.view(-1, 2)
                labels = labels.view(-1)

                active_loss = attention_mask.view(-1) == 1
                active_logits = logits[active_loss]
                active_labels = labels[active_loss]

                preds = torch.argmax(active_logits, dim=1)
                probabilities = torch.softmax(active_logits, dim=1)
                pos_proba = probabilities[:, 1]

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(active_labels.cpu().numpy())
                # Storing max probability of each prediction
                max_proba.extend(pos_proba.cpu().numpy())

        # Save the results as NumPy arrays
        np.save(prediction_name, np.array(predictions))
        np.save(label_name, np.array(true_labels))
        np.save(proba_name, np.array(max_proba))