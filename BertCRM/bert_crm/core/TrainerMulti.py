import gc
import torch
import csv
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Don't forget this import:
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        writer: SummaryWriter = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = writer
        self.train_step = 0
        self.val_step = 0

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            progress_bar = tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

            for batch in progress_bar:
                # Unpack mask from batch
                input_ids, attention_mask, labels, single_mask, n_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                single_mask = single_mask.to(self.device)
                labels = labels.to(self.device, dtype=torch.float32)

                print(labels)
                
                # Forward pass with mask
                logits = self.model(input_ids, attention_mask, single_mask)

                loss = self.loss_fn(logits, labels)
                
                # Optimization
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Calculate metrics
                probs = torch.sigmoid(logits)  # Use sigmoid instead of softmax
                preds = (probs > 0.5).long()  # Threshold at 0.5
                acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

                # Logging
                if self.writer and (self.train_step % 8 == 0):
                    self.writer.add_scalar("Train/Loss", loss.item(), self.train_step)
                    self.writer.add_scalar("Train/Accuracy", acc, self.train_step)
                self.train_step += 1

            # Validation
            self.validate(epoch)
            
            # Cleanup
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids, attention_mask, labels, single_mask, n_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                single_mask = single_mask.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids, attention_mask, single_mask)
                logits = logits[:, 1:-1, :]
                labels = labels[:, :logits.shape[1]]

                outputs_reshaped = logits.reshape(-1, logits.size(-1))
                labels_reshaped = labels.reshape(-1).long()

                loss = self.loss_fn(outputs_reshaped, labels_reshaped)
                probs = torch.softmax(outputs_reshaped, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                acc = accuracy_score(labels_reshaped.cpu(), preds.cpu())

                total_loss += loss.item()
                total_acc += acc

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        
        if self.writer:
            self.writer.add_scalar("Val/Loss", avg_loss, epoch)
            self.writer.add_scalar("Val/Accuracy", avg_acc, epoch)
            
        print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
        self.model.train()

    def test(self, test_loader, output_file="test_predictions.csv"):
        self.model.eval()
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["prob", "pred", "label"])
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing"):
                    input_ids, attention_mask, labels, single_mask, n_mask = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    single_mask = single_mask.to(self.device)
                    labels = labels.to(self.device)
                    
                    logits = self.model(input_ids, attention_mask, single_mask)
                    logits = logits[:, 1:-1, :]
                    labels = labels[:, :logits.shape[1]]

                    outputs_reshaped = logits.reshape(-1, logits.size(-1))
                    labels_reshaped = labels.reshape(-1).long()

                    probs = torch.softmax(outputs_reshaped, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                    for p, y_pred, y_true in zip(probs.cpu(), preds.cpu(), labels_reshaped.cpu()):
                        writer.writerow([
                            f"{p[y_pred]:.4f}",  # Probability of predicted class
                            y_pred.item(),
                            y_true.item()
                        ])
        return output_file