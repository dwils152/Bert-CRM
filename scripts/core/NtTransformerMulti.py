import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import torch
import torch.nn.functional as F

class NtTransformerMulti(nn.Module):
    def __init__(self, num_labels=1):  # For binary classification
        super().__init__()

        self.base_model = AutoModelForTokenClassification.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            num_labels=num_labels,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            return_tensors="pt",
            padding_side="right",
            truncation=True,
            padding="max_length",
            trust_remote_code=True
        )

        self.classifier_1mer = nn.Linear(768, 1)
        self.classifier_6mer = nn.Linear(768, 6)
        
        #self._freeze_all_except_last_n(2)

    # def _freeze_all_except_last_n(self, n: int):
    #     """Freeze all parameters except last n transformer layers and classifier"""
    #     # Freeze entire base model first
    #     for param in self.base_model.parameters():
    #         param.requires_grad = False
            
    #     # Unfreeze last n transformer layers
    #     total_layers = len(self.base_model.encoder.layer)
    #     layers_to_unfreeze = self.base_model.encoder.layer[-n:]
        
    #     print(f"Unfreezing last {n}/{total_layers} layers:")
    #     for layer in layers_to_unfreeze:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    #     # Always keep classifier trainable
    #     for param in self.classifier.parameters():
    #         param.requires_grad = True
    def forward(self, input_ids, attention_mask, single_nucleotide_mask):
        # 1) Run the base model
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]  # (B, L, 768)

        # 2) Create masks
        attention_mask = attention_mask.bool()
        single_nucleotide_mask = single_nucleotide_mask.bool()
        cls_mask = (input_ids == self.tokenizer.cls_token_id)

        mask_1mer = single_nucleotide_mask & attention_mask & ~cls_mask
        mask_6mer = ~single_nucleotide_mask & attention_mask & ~cls_mask

        # 3) Gather hidden states
        single_hidden = last_hidden[mask_1mer]  # (N_single, 768)
        six_hidden = last_hidden[mask_6mer]     # (N_six, 768)

        # 4) Compute logits (keep gradients!)
        single_logits = self.classifier_1mer(single_hidden).squeeze(-1)  # (N_single,)
        six_logits = self.classifier_6mer(six_hidden)                     # (N_six, 6)

        # 5) Build expanded sequences using tensor operations
        B, L = input_ids.shape
        expanded_sequences = []

        # Track indices for slicing
        single_idx = 0
        six_idx = 0

        for b in range(B):
            # Get number of 1mer/6mer tokens in this batch
            num_single = mask_1mer[b].sum().item()
            num_six = mask_6mer[b].sum().item()

            # Slice logits for this example
            example_single = single_logits[single_idx:single_idx + num_single]
            example_six = six_logits[six_idx:six_idx + num_six].flatten()  # (num_six * 6,)

            # Combine and add to list
            combined = torch.cat([example_single, example_six])
            expanded_sequences.append(combined)

            # Update indices
            single_idx += num_single
            six_idx += num_six

        # 6) Pad sequences while preserving gradients
        padded = torch.nn.utils.rnn.pad_sequence(
            expanded_sequences, 
            batch_first=True, 
            padding_value=0.0
        )
        return padded  # (B, max_len)


    def forward_(self, input_ids, attention_mask, single_nucleotide_mask):
        """
        Args:
            input_ids: (B, L)        token IDs
            attention_mask: (B, L)
            single_nucleotide_mask: (B, L), True for single-nucleotide tokens, False for 6-mer tokens

        Returns:
            base_expanded_outputs: List of length B,
               where each element is a tensor of shape (M_i,) containing
               the expanded per-base outputs for the i-th example in the batch.
               M_i = sum of expansions (1 or 6) across all tokens in sequence i.
        """
        # 1) Run the base model to get hidden states
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # shape = (B, L, 768)
        last_hidden = outputs.hidden_states[-1]  

        # Convert to bool
        attention_mask = attention_mask.bool()
        single_nucleotide_mask = single_nucleotide_mask.bool()

        # (Optional) Exclude special tokens if needed
        cls_id = self.tokenizer.cls_token_id
        cls_mask = (input_ids == cls_id)
        
        # Create separate masks for 1-mer vs. 6-mer
        mask_1mer = single_nucleotide_mask & attention_mask & ~cls_mask
        mask_6mer = ~single_nucleotide_mask & attention_mask & ~cls_mask

        # 2) Gather hidden states for each type
        single_hidden = last_hidden[mask_1mer]   # shape (N_single, 768)
        six_hidden    = last_hidden[mask_6mer]   # shape (N_six, 768)

        # 3) Classify
        single_logits = self.classifier_1mer(single_hidden)  # (N_single, 1)
        six_logits    = self.classifier_6mer(six_hidden)     # (N_six, 6)

        # 4) We'll reconstruct outputs per example in the batch
        B, L = input_ids.shape
        expanded_sequences = []

        # Convert them to CPU if you plan to do Python list manipulations
        single_logits_cpu = single_logits.squeeze(-1).detach().cpu()  # shape (N_single,)
        six_logits_cpu = six_logits.detach().cpu()  # shape (N_six, 6)

        # We need to know which positions in single_logits/six_logits
        # correspond to which tokens. PyTorch "fancy indexing" is flattening them
        # in row-major order. We'll do it example-by-example to keep track:

        single_idx = 0
        six_idx = 0

        for b_idx in range(B):
            expanded_list = []

            # For each token in the b_idx-th sequence:
            for t_idx in range(L):
                if not attention_mask[b_idx, t_idx]:
                    # outside actual tokens (padding) or masked
                    continue

                if cls_mask[b_idx, t_idx]:
                    # Could skip or put a special value
                    continue

                if single_nucleotide_mask[b_idx, t_idx]:
                    # 1-mer token
                    # single_logits_cpu is a 1D array for *all* single tokens across entire batch
                    # single_idx points to the next single-token logit
                    val = single_logits_cpu[single_idx].item()
                    expanded_list.append(val)
                    single_idx += 1
                else:
                    # 6-mer token
                    # six_logits_cpu is shape (N_six, 6)
                    # six_idx points to the next row
                    vals_6 = six_logits_cpu[six_idx].tolist()  # 6 floats
                    expanded_list.extend(vals_6)
                    six_idx += 1

            # Now we've expanded the b_idx-th sequence into base-level outputs
            expanded_tensor = torch.tensor(expanded_list, device=last_hidden.device)
            expanded_sequences.append(expanded_tensor)

        # Find max length
        max_len = max(seq.size(0) for seq in expanded_sequences)

        padded_tensors = []
        for seq in expanded_sequences:
            length_diff = max_len - seq.size(0)
            # pad on the right with 0.0 (or another pad_value)
            padded_seq = F.pad(seq, (0, length_diff), value=0.0)  
            padded_tensors.append(padded_seq)

        # Stack into final 2D tensor (B, max_len)
        final_tensor = torch.stack(padded_tensors, dim=0)
        return final_tensor



        

if __name__ == "__main__":

    model = NtTransformerMulti()

    print(model.tokenizer.get_vocab())

    test_seq = "ATATATNCCTACCTCGNNNN"
    inputs = model.tokenizer(test_seq, max_length=21, padding="max_length")
    input_ids = torch.tensor(inputs['input_ids'])
    attention_mask = torch.tensor(inputs['attention_mask'])

    single_set = torch.tensor([4102, 4103, 4104, 4105, 4106])
    single_mask = torch.isin(input_ids, single_set).bool()
    n_mask = (input_ids == 4106).bool()

    #unsqueeze the batch dimension
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    single_mask = single_mask.unsqueeze(0)

    #model(input_ids, attention_mask, single_mask)
    print(model(input_ids, attention_mask, single_mask))


