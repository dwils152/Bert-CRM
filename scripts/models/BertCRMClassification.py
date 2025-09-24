from transformers import AutoModel
import torch
from torch import nn
import bert_crm
from bert_crm.utils.utils import tqdm_wrapper
import numpy as np
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, accuracy_score


class BertCRMClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertCRMClassification, self).__init__()
        self.num_labels = num_labels

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True)
        self.classifier = nn.Linear(768, 1)
        self.return_logits = True
        #self.freeze_except_lora()

    def freeze_except_lora(self):
        for name, submodule in self.bert.named_children():
            if name == 'encoder':
                for layer in submodule.layer:
                    for param_name, param in layer.named_parameters():
                        if not any(lora_component in param_name for lora_component in ['Lora_A_Q', 'Lora_B_Q', 'Lora_A_K', 'Lora_B_K', 'Lora_A_V', 'Lora_B_V']):
                            param.requires_grad = False
            else:
                # For other submodules (like pooler),  all parameters
                for param in submodule.parameters():
                    param.requires_grad = False


    def forward(self, input_ids, attention_mask):
        encoder_output, pooling_output, attention_probs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)

        print(f'encoder_output shape: {encoder_output.shape}')
        print(f'pooling_output shape: {pooling_output.shape}')

        
        cls_token = encoder_output[:, 0, :]
        print(f'cls_token shape: {cls_token.shape}')
        return self.classifier(pooling_output)
        #return cls_token

    def get_embeddings(self, input_ids, attention_mask):
        encoder_output, pooling_output, attention_probs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)
        
        cls_token = encoder_output[:, 0, :]
        return cls_token


    def train_model(self, train_loader, val_loader, optimizer, device):
        self.train()

        with open(f"train_log.txt", "a") as train_log:
            
            for batch in train_loader:
                input_ids, attention_mask, labels = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                optimizer.zero_grad()
                logits = self.forward(input_ids, attention_mask)
                logits = logits.view(-1)

                # Compute loss
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                loss.backward()
                optimizer.step()

                predictions = torch.sigmoid(logits).round()
                acc = accuracy_score(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
                #roc_auc = roc_auc_score(labels.cpu().detach().numpy(), torch.sigmoid(logits).cpu().detach().numpy())

                train_log.write(f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}\n")#, ROC-AUC: {roc_auc:.4f}\n")
                train_log.flush()

        # print(f"Process {dist.get_rank()} reached the barrier.")
        #dist.barrier()
        # print(f"Process {dist.get_rank()} passed the barrier.")

        self.eval()

        with torch.no_grad():
            with open(f"val_log.txt", "a") as val_log:
                for batch in val_loader:
                    input_ids, attention_mask, labels = [
                        b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                    logits = self.forward(input_ids, attention_mask)
                    logits = logits.view(-1)

                    # Compute loss
                    loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                    
                    predictions = torch.sigmoid(logits).round()
                    acc = accuracy_score(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
                    #roc_auc = roc_auc_score(labels.cpu().detach().numpy(), torch.sigmoid(logits).cpu().detach().numpy())
                    val_log.write(f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}\n")#, ROC-AUC: {roc_auc:.4f}\n")
                    val_log.flush()
                
       

    def test_model(self, data_loader, device):
        self.eval()
        test_loss = []
        test_accuracy = []
        test_roc_auc = []

        with torch.no_grad():

            with open(f"test_log.txt", "a") as test_log:
                for batch in data_loader:
                    input_ids, attention_mask, labels = [
                        b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                    logits = self.forward(input_ids, attention_mask)
                    logits = logits.view(-1)

                    # Compute loss
                    loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                    # Convert logits to predictions for accuracy and ROC computation
                    predictions = torch.sigmoid(logits).round()
                    acc = accuracy_score(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
                    roc_auc = roc_auc_score(labels.cpu().detach().numpy(), torch.sigmoid(logits).cpu().detach().numpy())

                    test_log.write(f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}\n")#, ROC-AUC: {roc_auc:.4f}\n")
                    test_log.flush()

