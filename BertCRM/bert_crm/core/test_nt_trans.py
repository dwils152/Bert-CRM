from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import torch

# tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

class NtModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        config.token_dropout = False
        self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", config=config)
        for param in self.model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", "value"],
        )

        self.nt_transformer = get_peft_model(
            self.model, 
            lora_config
        )
        self.classifier = nn.Linear(2560, 100)
        self.global_step = 0

    def forward(self, input_ids, attention_mask):
        outputs = self.nt_transformer(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      encoder_attention_mask=attention_mask,
                                      output_hidden_states=True)
        embeddings = outputs['hidden_states'][-1]
        #print(f'Embeddings shape: {embeddings.shape}')
        logits = torch.sigmoid(self.classifier(embeddings[:, 0, :]))
        return logits

    def train_model(self, train_loader, optimizer, device, writer):
        self.train()

        with open(f"train_log.txt", "a") as train_log:
            for batch in tqdm(train_loader, total=len(train_loader)):
                input_ids, attention_mask, labels, nt_mask = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                optimizer.zero_grad()
                # Get both outputs from the model
                classifier_output = self.forward(input_ids, attention_mask)
                labels = labels.float()

                # Flatten tensors and apply mask
                nt_mask_flat = nt_mask.view(-1).bool()
                labels_flat = labels.view(-1)[nt_mask_flat]
                classifier_output_flat = classifier_output.view(-1)[nt_mask_flat]

                # Create binary targets for zero inflation component
                zero_targets = (labels_flat == 0).float()  # 1 if label == 0, else 0

                print(zero_targets)
                print(classifier_output_flat)

                # Compute BCE loss for zero inflation component
                bce_loss_fn = nn.BCELoss()
                bce_loss = bce_loss_fn(classifier_output_flat, zero_targets)
                try:
                    auroc = roc_auc_score(zero_targets.detach().cpu().numpy(), classifier_output_flat.detach().cpu().numpy())
                except:
                    pass

                # Backward pass and optimization
                bce_loss.backward()
                optimizer.step()

                writer.add_scalar("Metrics/AUROC", auroc, self.global_step)
                self.global_step += 1

# tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
# model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

# Choose the length to which the input sequences are padded. By default, the 
# model max length is chosen, but feel free to decrease it as the time taken to 
# obtain the embeddings increases significantly with it.
# max_length = tokenizer.model_max_length

# # Create a dummy dna sequence and tokenize it
# sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
# tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

# # Compute the embeddings
# attention_mask = tokens_ids != tokenizer.pad_token_id
# torch_outs = model(
#     tokens_ids,
#     attention_mask=attention_mask,
#     #encoder_attention_mask=attention_mask,
#     output_hidden_states=True
# )

# # Compute sequences embeddings
# embeddings = torch_outs['hidden_states'][-1].detach().numpy()
# print(f"Embeddings shape: {embeddings.shape}")
# print(f"Embeddings per token: {embeddings}")