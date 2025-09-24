import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import gc
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        writer=None,
        scaler=None,
        profiler=None,
        accumulation_steps=1,
        swa_start_epoch=10,             # <-- New param: epoch to start SWA
        swa_lr=1e-5                    # <-- New param: SWA learning rate
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The DDP-wrapped model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_fn (callable): Loss function.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader, optional): DataLoader for validation data.
            device (torch.device, optional): Device to train on.
            writer (SummaryWriter, optional): TensorBoard SummaryWriter.
            scaler (GradScaler, optional): For mixed precision training.
            profiler (torch.profiler.profile, optional): PyTorch profiler instance.
            accumulation_steps (int, optional): Number of steps to accumulate gradients.
            swa_start_epoch (int, optional): Epoch at which to start updating SWA.
            swa_lr (float, optional): Learning rate used by SWA scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = writer
        self.scaler = scaler
        self.profiler = profiler
        self.accumulation_steps = accumulation_steps
        self.train_step = 0
        self.val_step = 0

        # SWA components
        self.swa_start_epoch = swa_start_epoch
        # Wrap your model with AveragedModel
        self.swa_model = AveragedModel(self.model)
        # Create an SWA scheduler (optional but typical)
        # Use your optimizer here; if you already have a separate scheduler,
        # you may incorporate it differently.
        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=swa_lr, 
            anneal_epochs=5,         # adjust as needed
            anneal_strategy='linear'    # can be 'linear' or 'cos'
        )

    def train(self, num_epochs):
        """
        Executes the training loop.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0

            # If your train_loader uses a DistributedSampler, set the epoch 
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            progress_bar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for i, batch in progress_bar:
                input_ids, attention_mask, labels, nt_mask = [
                    b.to(self.device, non_blocking=True) if isinstance(b, torch.Tensor) else b 
                    for b in batch
                ]

                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    labels = labels.float()

                    print(outputs.size())
                    print(labels.size())

                    # Flatten tensors
                    labels_flat = labels.view(-1)
                    outputs_flat = outputs.view(-1)

                    loss = self.loss_fn(outputs_flat, labels_flat)
                    loss = loss / self.accumulation_steps

                # Gradient scaling for mixed precision
                self.scaler.scale(loss).backward()
                epoch_loss += loss.item() * self.accumulation_steps

                # Compute accuracy
                probs = torch.sigmoid(outputs_flat).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(labels_flat.detach().cpu().numpy().astype(int), preds)
                try:
                    auc = roc_auc_score(labels_flat.detach().cpu().numpy().astype(int), probs)
                except:
                    pass
            
                epoch_acc += acc

                # Step optimizer (with gradient accumulation)
                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Logging to TensorBoard
                if self.writer:
                    self.writer.add_scalar("Metrics/Train_Accuracy", acc, self.train_step)
                    self.writer.add_scalar("Metrics/Train_Loss", loss.item(), self.train_step)
                    self.writer.add_scalar("Metrics/Train_AUC", auc, self.train_step)
                self.train_step += 1

                # Profiler step
                if self.profiler:
                    self.profiler.step()

                progress_bar.set_postfix({
                    "Loss": epoch_loss / (i + 1), 
                    "Acc": epoch_acc / (i + 1)
                })

            # If you've reached SWA start epoch, update the SWA model
            if epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                # Step the SWA scheduler (if being used)
                self.swa_scheduler.step()
            else:
                # If you're using a different scheduler or
                # want to step your original scheduler, do it here
                pass

            # Validation after each epoch (optional)
            self.validate()

            # Clear cache to free memory
            gc.collect()
            torch.cuda.empty_cache()

        # ----------------------------
        #  Post-Training SWA Updates
        # ----------------------------
        print("Updating Batch Normalization statistics for SWA model...")
        update_bn(self.train_loader, self.swa_model, device=self.device)
        
        # After updating BN, you can evaluate or test using self.swa_model
        # (You can do additional validation or just save the model.)

        # Save the final SWA model
        final_swa_path = "final_swa_model.pt"
        torch.save(self.swa_model.state_dict(), final_swa_path)
        print(f"Final SWA model saved to {final_swa_path}")

    def validate(self):
        """
        Executes the validation loop.

        Returns:
            tuple: Average loss and accuracy on the validation set.
        """
        if self.val_loader is None:
            return

        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, total=len(self.val_loader), desc="Validation"):
                input_ids, attention_mask, labels, nt_mask = [
                    b.to(self.device, non_blocking=True) if isinstance(b, torch.Tensor) else b 
                    for b in batch
                ]

                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    labels = labels.float()

                    # Flatten tensors
                    labels_flat = labels.view(-1)
                    outputs_flat = outputs.view(-1)

                    # Compute loss
                    loss = self.loss_fn(outputs_flat, labels_flat)
                    val_loss += loss.item()

                # Compute accuracy
                probs = torch.sigmoid(outputs_flat).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(labels_flat.detach().cpu().numpy().astype(int), preds)
                val_acc += acc

                if self.writer:
                    self.writer.add_scalar("Metrics/Val_Accuracy", acc, self.val_step)
                    self.writer.add_scalar("Metrics/Val_Loss", loss.item(), self.val_step)
                self.val_step += 1

        self.model.train()
        return val_loss / len(self.val_loader), val_acc / len(self.val_loader)

    def test(self, test_loader):
        """
        Executes the testing loop.

        Args:
            test_loader (DataLoader): DataLoader for testing data.

        Returns:
            dict: Metrics computed on the test set.
        """
        # If you want to test the SWA model, use self.swa_model.eval()
        # or if you want the original model, use self.model.eval()
        
        # Example: test using SWA model
        self.swa_model.eval()

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
                input_ids, attention_mask, labels, nt_mask = [
                    b.to(self.device, non_blocking=True) if isinstance(b, torch.Tensor) else b 
                    for b in batch
                ]

                with torch.cuda.amp.autocast():
                    outputs = self.swa_model(input_ids, attention_mask)
                    labels = labels.float()

                    labels_flat = labels.view(-1)
                    outputs_flat = outputs.view(-1)

                # Compute predictionsW
                probs = torch.sigmoid(outputs_flat).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels_flat.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy (SWA model): {accuracy:.4f}")

        if self.writer:
            self.writer.add_scalar("Metrics/Test_Accuracy_SWA", accuracy, 0)

        return {
            "accuracy": accuracy,
        }
