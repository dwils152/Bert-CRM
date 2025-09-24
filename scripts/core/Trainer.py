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
        """
        Train loop.
        """
        self.model.train()
        for epoch in range(num_epochs):

            progress_bar = tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

            for batch in progress_bar:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                logits = self.model(input_ids, attention_mask).logits
                logits = logits[:, 1:, :]
                logits = logits.squeeze()

                # Compute loss
                loss = self.loss_fn(logits, labels)
                
                # Backprop and optimizer step
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Compute training accuracy
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                acc = accuracy_score(labels.cpu().view(-1), preds.cpu().view(-1))

                # Logging to TensorBoard
                if self.writer and (self.train_step % 8 == 0):
                    self.writer.add_scalar("Train/Loss", loss.item(), self.train_step)
                    self.writer.add_scalar("Train/Accuracy", acc, self.train_step)
                self.train_step += 1

            # Validate after each epoch
            self.validate(epoch)

            # Garbage collection
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def validate(self, epoch):
        """
        Validation loop. Compute average loss and accuracy over all val data.
        """
        self.model.eval()


        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"Validation Epoch {epoch + 1}"
            )

            for batch in progress_bar:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask).logits
                logits = logits[:, 1:, :]
                logits = logits.squeeze()

                loss = self.loss_fn(logits, labels)

                # Compute training accuracy
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                acc = accuracy_score(labels.cpu().view(-1), preds.cpu().view(-1))

                if self.writer:
                    self.writer.add_scalar("Val/Loss", loss, self.val_step)
                    self.writer.add_scalar("Val/Accuracy", acc, self.val_step)
                self.val_step += 1

        self.model.train()

    def test(self, test_loader, output_file="test_predictions.csv"):

        self.model.eval()

        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prob", "pred", "label"])

            with torch.no_grad():
                for i, batch in tqdm(enumerate(test_loader),
                                     total=len(test_loader),
                                     desc="Testing..."):
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    
                    logits = self.model(input_ids, attention_mask).logits
                    logits = logits[:, 1:, :]
                    logits = logits.squeeze()

                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()

                    # Write each prediction to CSV
                    for p, y_pred, y_true in zip(probs.cpu(), preds.cpu(), labels.cpu()):
                        writer.writerow([f"{p:.6f}", int(y_pred), int(y_true)])

        print(f"Predictions written to {output_file}")
        return output_file
