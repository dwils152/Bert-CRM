from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import torch


class NtModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        config.token_dropout = False
        self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", config=config)
        for param in self.model.parameters():
            param.requires_grad = False

        hidden_size = 2560

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", "value"],
        )

        self.transformer = get_peft_model(
            self.model, 
            lora_config
        )
         # Project CLS token to 100 features (1 per token)
        self.cls_projector = nn.Linear(hidden_size, 100)
        
        # Token classifier
        self.token_classifier = nn.Linear(hidden_size + 1, 1)  # 2560 (local) + 1 (global)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs['hidden_states'][-1]
        cls_embedding = last_hidden_state[:, 0, :]     # (batch_size, hidden_size)
        token_embeddings = last_hidden_state[:, 1:, :] # (batch_size, 100, hidden_size)

        # Process CLS features
        cls_features = self.cls_projector(cls_embedding)  # (batch_size, 100)
        
       # Project CLS to 100 features
        cls_features = self.cls_projector(cls_embedding)  # (batch, 100)
        cls_features = cls_features.unsqueeze(-1)  # (batch, 100, 1)
        
        # Combine with token embeddings
        combined = torch.cat([token_embeddings, cls_features], dim=-1)  # (batch, 100, hidden_size + 1)
        
        # Predictions
        token_preds = self.token_classifier(combined).squeeze(-1)  # (batch, 100)
        return token_preds

def train_model(self, train_loader, optimizer, device, writer, lambda_val=0.5):
    """
    Trains the model for one epoch with corrected loss calculation and multi-task handling.
    """
    self.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()  # Better numerical stability than BCELoss + sigmoid

    with open("train_log.txt", "a") as train_log:
        for batch in tqdm(train_loader, total=len(train_loader), desc="Training"):
            input_ids, attention_mask, token_labels, seq_labels, nt_mask = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]

            optimizer.zero_grad()
            
            # Forward pass - assuming model returns token_preds and cls_preds
            token_preds, cls_preds = self.forward(input_ids, attention_mask)
            
            # --- Token-level loss ---
            # Flatten and mask relevant positions
            token_labels_flat = token_labels.view(-1).float()  # (batch*seq_len,)
            token_preds_flat = token_preds.view(-1)            # (batch*seq_len,)
            mask = nt_mask.view(-1).bool()                     # (batch*seq_len,)
            
            # Calculate masked BCE loss
            token_loss = bce_loss_fn(
                token_preds_flat[mask],
                token_labels_flat[mask]
            )

            # --- Sequence-level loss (optional) ---
            # Only if you have sequence-level labels
            seq_loss = bce_loss_fn(cls_preds, seq_labels.float())
            
            # Combined loss
            total_loss = lambda_val * token_loss + (1 - lambda_val) * seq_loss

            # --- Metrics ---
            with torch.no_grad():
                # Calculate token-level AUROC
                masked_preds = torch.sigmoid(token_preds_flat[mask])
                try:
                    token_auroc = roc_auc_score(
                        token_labels_flat[mask].cpu().numpy(),
                        masked_preds.cpu().numpy()
                    )
                except ValueError:
                    token_auroc = 0.0

                # Calculate sequence-level AUROC if applicable
                if seq_labels is not None:
                    seq_auroc = roc_auc_score(
                        seq_labels.cpu().numpy(),
                        torch.sigmoid(cls_preds).cpu().numpy()
                    )

            # Backprop
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            # Logging
            writer.add_scalars("Loss", {
                "total": total_loss.item(),
                "token": token_loss.item(),
                "sequence": seq_loss.item() if seq_labels is not None else 0
            }, self.global_step)

            writer.add_scalars("AUROC", {
                "token": token_auroc,
                "sequence": seq_auroc if seq_labels is not None else 0
            }, self.global_step)

            self.global_step += 1

            # File logging
            log_str = (
                f"Step {self.global_step}: "
                f"Total Loss = {total_loss.item():.4f}, "
                f"Token AUROC = {token_auroc:.4f}"
            )
            if seq_labels is not None:
                log_str += f", Seq AUROC = {seq_auroc:.4f}"
            train_log.write(log_str + "\n")