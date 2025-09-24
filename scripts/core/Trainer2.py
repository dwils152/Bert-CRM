import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


class TrainerV2:
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
        self.scaler = scaler
        self.profiler = profiler
        self.train_step = 0
        self.val_step = 0

    def train(self, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )

            for idx, batch in progress_bar:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(input_ids, attention_mask).logits
                logits = logits[:, 1:-1, :]

                outputs_reshaped = logits.reshape(-1)
                labels_reshaped = labels.reshape(-1)
                loss = self.loss_fn(outputs_reshaped, labels_reshaped.float())
                loss.backward()
                self.optimizer.step()

                probs = torch.sigmoid(outputs_reshaped)
                preds = (probs > 0.5).float()

                acc = accuracy_score(labels, preds)
                if self.writer and (self.train_step % 10 == 0):
                    self.writer.add_scalar("Train/Loss", loss.item(), self.train_step)
                    self.writer.add_scalar("Train/Accuracy", acc, self.train_step)
                self.train_step += 1

            # Validation
            if self.val_loader is not None:
                self.validate(epoch)

            gc.collect()
            torch.cuda.empty_cache()


    def validate(self, epoch):
        self.model.eval()

        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc=f"Validation Epoch {epoch + 1}"
            )

            for idx, batch in progress_bar:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(input_ids, attention_mask).logits
                logits = logits[:, 1:-1, :]

                outputs_reshaped = logits.reshape(-1)
                labels_reshaped = labels.reshape(-1)
                loss = self.loss_fn(outputs_reshaped, labels_reshaped.float())
                loss.backward()

                probs = torch.sigmoid(outputs_reshaped)
                preds = (probs > 0.5).float()

                if self.writer:
                    self.writer.add_scalar("Validation/Loss", loss.item(), epoch)
                    self.writer.add_scalar("Validation/Accuracy", avg_accuracy, epoch)

        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    def test(self, test_loader, model=None, dataset_name="test"):
        """Test loop with metrics."""
        model = model or self.model
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        all_probs = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"Testing ({dataset_name})"
            )
            for i, batch in progress_bar:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits[:, 1:-1, :]
                    labels_sliced = labels[:, 1:-1]

                # Reshape
                batch_size, seq_len, num_labels = logits.size()
                outputs_reshaped = logits.reshape(-1, num_labels)
                labels_reshaped = labels_sliced.reshape(-1)

                # Probabilities
                if num_labels == 1:
                    probs = torch.sigmoid(outputs_reshaped)
                else:
                    probs = F.softmax(outputs_reshaped, dim=-1)

                # Loss
                loss = self.loss_fn(outputs_reshaped, labels_reshaped.float())
                test_loss += loss.item()
                num_batches += 1

                # For final metrics
                all_probs.append(probs.cpu())
                all_labels.append(labels_reshaped.cpu())

                # Accuracy
                if num_labels == 1:
                    preds = (probs > 0.5).long()
                else:
                    preds = probs.argmax(dim=-1)
                test_accuracy += (preds == labels_reshaped).float().mean().item()

                progress_bar.set_postfix({
                    "Loss": test_loss / num_batches,
                    "Accuracy": test_accuracy / num_batches
                })

        # Final metrics
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        if num_labels == 1:
            final_preds = (all_probs > 0.5).long()
        else:
            final_preds = all_probs.argmax(dim=-1)

        metrics = {
            "loss": test_loss / num_batches,
            "accuracy": test_accuracy / num_batches,
            "f1": f1_score(all_labels, final_preds, average='macro' if num_labels > 1 else 'binary'),
            "precision": precision_score(all_labels, final_preds, average='macro' if num_labels > 1 else 'binary'),
            "recall": recall_score(all_labels, final_preds, average='macro' if num_labels > 1 else 'binary')
        }

        # AUROC
        try:
            if num_labels == 1:
                metrics["auroc"] = roc_auc_score(all_labels, all_probs)
            else:
                metrics["auroc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Couldn't calculate AUROC: {e}")
            metrics["auroc"] = float('nan')

        print(f"Test metrics ({dataset_name}): {metrics}")
        return metrics
