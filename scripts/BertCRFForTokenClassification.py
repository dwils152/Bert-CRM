from transformers import AutoModel
import torch
from torch import nn
from torchcrf import CRF
from utils import tqdm_wrapper
import numpy as np
import torch.distributed as dist

class BertCRFForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertCRFForTokenClassification, self).__init__()
        self.num_labels = num_labels

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True)
        self.classifier = nn.Linear(768, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.freeze_except_lora()

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

        # Compute the CRF loss
        if labels is not None:
            # The CRF layer returns negative log likelihood
            crf_loss = -self.crf(logits, labels, mask=attention_mask.byte())
            # Averaging the loss over the batch
            loss = crf_loss.mean()
            return loss
        else:
            return self.crf.decode(logits), outputs

    def train_model(self, train_loader, val_loader, optimizer, device, rank):
        self.train()
        for batch in tqdm_wrapper(train_loader, rank, desc="Training"):
            input_ids, attention_mask, labels, _ = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            # labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.clone().detach()
            loss = self.forward(input_ids, attention_mask, labels)

            # Backward pass and optimize
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
                batch_loss = self.forward(input_ids, attention_mask, labels)

                # Aggregate loss
                val_loss += batch_loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        # Convert to torch tensor for distributed reduction
        val_loss_tensor = torch.tensor(val_loss).to(device)
        total_samples_tensor = torch.tensor(total_samples).to(device)

        # Reduce (sum) the losses and total samples across all GPUs
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        # Calculate average validation loss across all GPUs
        average_val_loss = val_loss_tensor.item() / total_samples_tensor.item()

        return average_val_loss
    
    def test_model(self, data_loader, device, rank):
        self.eval()
        # all_predictions, all_true_labels, all_coordinates = [], [], []
        all_predictions, all_true_labels = [], []
        prediction_name = f'predictions_rank_{rank}.npy'
        label_name = f'true_labels_rank_{rank}.npy'

        with torch.no_grad():
            for batch in tqdm_wrapper(data_loader, rank, desc="Testing"):
                input_ids, attention_mask, labels, coordinates = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                predictions, logits = self.forward(input_ids, attention_mask)
                predicted_labels = torch.tensor(
                    [item for sublist in predictions for item in sublist], device=device)

                print(len(logits))
                print(logits[0].shape)

                flat_true_labels = labels.view(-1)
                flat_predictions = predicted_labels.view(-1)
                flat_attention_mask = attention_mask.view(-1)

                # Select only the non-padded elements
                active_positions = flat_attention_mask == 1
                active_labels = flat_true_labels[active_positions]
                active_predictions = flat_predictions[active_positions]

                all_predictions.extend(active_predictions.cpu().tolist())
                all_true_labels.extend(active_labels.cpu().tolist())

        np.save(prediction_name, np.array(all_predictions))
        np.save(label_name, np.array(all_true_labels))