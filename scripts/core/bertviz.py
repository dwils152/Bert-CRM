import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, utils
import matplotlib.pyplot as plt
import torch
import os
import glob
import re
from bert_crm.models.BertForTokenClassification import BertForTokenClassification
from bert_crm.models.SupervisedDataset import SupervisedDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from bert_crm.utils.utils import supervised_collate_fn
from collections import OrderedDict
import sys    
import numpy as np

def main():
    
    #from_dataloader()
    from_seq()
    
def from_seq():
    
    seq = 'AGGAATATAAAGATAAATGAGACATATCTTTAGAGAGAGAGAGAGAGAGAGAGCAAGAGAGCTGATACCTAAAAGATGAGTGGGAGTTAGCTAAGCGGAGAATTTATAAATGGAGTGGACATTTCTAATAAACACCTTTAACATTCAATACGCTAGAGCTAATTCTCAGATCAATCTTCTTGCTAACAATTGTCTTGGTTCCTTTTCTTGAGGTTGTTCACATAAGAAACATTTTGAGGTCTTTGTGACAAGCTGCTTATTCTGGGCTCTAAATTCAAATAAAACATTGGCATAGAGAAATAAATCATTTCCATAGATGTCAACATCTTTGTGAGACACACAAGTGTATTATAATCTCACCCGTTACTGTTTCACCTAGGTCGTTAGCATGAATACCCATTTGAGATACCCCATCATTACACTCTAGGCCAGAAGCTTTCAAATTCTGAAATGTCCACAGACAAGACTTTCTATTTCACTCTGTCCAGAAACCTTGAGTAAAAGAAAAGGAGGTAGGAAAAGGAAATGAGTCAGGGAACTGAAAACACCAGAGAGAGCCCGGTCTGTGTAAGGGATATTTTCACCATCCCCACCCTGTTAACTCAGTACTTCGTACGTTCACAAAAAGACATTCCCAAGTTCGTCATGCTATTTTATGCAGCACCTTCCCTTGGCCCGCTTCTTCAGCTAATTAAATTCTCATTCCTAAAAACTCAGTTCTGGCATGTTATCCTCTTACCCCACTGTCCCAGTCTAATTTAGATACACCTCCTTTGTGATCTCATAGCTTGCTACGCTTTTATATCTTATGGCTGTATAATAATAATTGATTTACTTATCTAACTTATCTTCTTCTTGAGAGAAGTAGTTCCTTGAGGGCAAGGCTCATGTTTTAATCATTTTTGTACCCCTCATTTCTAGCAGAGTACTTAAAAAGAATCCTTTCAATGACTTCATATTCATGCATGATAGTTACTTTTCCTTCCTTAGCCCCTAGAGA'
    model, tokenizer = load_model()
    model.return_logits = False
    tokenizer_dict = tokenizer.get_vocab()
    inputs = tokenizer.encode(seq, return_tensors='pt')

    ids_list = inputs.tolist()[0]
    kmer_to_id = {value: key for key, value in tokenizer_dict.items()}
    kmer_split = [kmer_to_id[i] for i in ids_list]
    kmer_lengths = [len(kmer) for kmer in kmer_split] #1024
    
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=inputs.device)
    length = attention_mask.squeeze().sum().item() #190
    kmer_lengths = kmer_lengths[:length][1:-1] #188

    # Process Self-Attention Matrices
    output, attn = model.forward(input_ids=inputs, attention_mask=attention_mask)

    attention_heads = attn[-1]
    for i in range(12):
        print(f'Head {i+1}')
        attn_head = attention_heads[i]
        attn_head = attn_head[0].squeeze()
        attn_head  = expand_matrix(attn_head.detach().numpy(), kmer_lengths)
        print(attn_head.shape)
        #attn_head = attn_head[:len(seq), :len(seq)]
        plt.matshow(attn_head)
        plt.savefig(f'head_{i+1}.png', dpi=300)
    
    


def from_dataloader():
    #print entire tensor
    torch.set_printoptions(profile="full")
    datapath = "/projects/zcsu_research1/dwils152/Bert-CRM/all_results/supervised_results_bug_fix/human/subset_10_000_no_N.csv"
    dataset = SupervisedDataset(datapath, 1024)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=supervised_collate_fn)
    
    model, tokenizer = load_model()
    tokenizer_dict = tokenizer.get_vocab()
    model.return_logits = False

    samples = len(dataset) * 12
    # Initialize a 360 x 1,000,000 matrix to store the attention weights
    #attention_weights = torch.zeros((samples, 1_000_000))
    
    for idx, batch in enumerate(dataloader):

        # Process Input Tokens
        input_ids, attention_mask, labels, coordinates = batch
        print(input_ids.shape)
        print(attention_mask.shape)
        ids_list = input_ids.tolist()[0]
        #print(ids_list)
        #print(len(ids_list))
        kmer_to_id = {value: key for key, value in tokenizer_dict.items()}
        kmer_split = [kmer_to_id[i] for i in ids_list]
        kmer_lengths = [len(kmer) for kmer in kmer_split] #1024
        
        #print(kmer_lengths)

        length = attention_mask.squeeze().sum().item() #190
        kmer_lengths = kmer_lengths[:length][1:-1] #188

        # Process Self-Attention Matrices
        outputs, attn = model.forward(input_ids=input_ids,  attention_mask=attention_mask)
        attention_heads = attn[-1]

        for i in range(12):
            print(f'Sequence {idx+1}, Head {i+1}')

            
            attn_head = attention_heads[i]
            attn_head = attn_head[0].squeeze() # Remove the batch dimension
            attn_head = attn_head[:length, :length]
            expanded_head = expand_matrix(attn_head.detach().numpy(), kmer_lengths)
            # Flatten the matrix and store it in the attention_weights matrix
            attention_weights[idx * 12 + i, :] = torch.from_numpy(expanded_head.flatten())
            

            #plt.matshow(expanded_head)
            #plt.savefig(f'test.png', dpi=300) 

        

        if idx == NUM_SEQS - 1:
            break

    np.save('attention_weights.npy', attention_weights) 
        


def load_model():
    #model_path = '/projects/zcsu_research1/dwils152/Bert-CRM/all_results/results_crm_ncrm_n_no_nsampled_10_000_with_alibi/human/train_no_crf/model.pth'
    model_path = '/projects/zcsu_research1/dwils152/Bert-CRM/results/human/train_no_crf/model.pth'
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertForTokenClassification('zhihan1996/DNABERT-2-117M', num_labels=2)

    # Unwrap the model from DDP
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    
    # Load the trained model
    model.load_state_dict(new_state_dict)
    return model, tokenizer

        
def expand_matrix(matrix, expansion_lengths):
    """
    Expands a given matrix into a larger matrix based on the expansion lengths.

    Parameters:
    matrix (numpy.ndarray): The original matrix to be expanded.
    expansion_lengths (list): A list of integers representing the expansion lengths for each row and column.

    Returns:
    numpy.ndarray: The expanded matrix.
    """

    matrix = matrix[1:-1, 1:-1] # Trim the CLS and SEP tokens
    
    num_rows, num_cols = matrix.shape
    total_rows = sum(expansion_lengths)
    total_cols = sum(expansion_lengths)

    # Prepare an empty matrix of the desired size
    expanded_matrix = np.zeros((total_rows, total_cols))

    # Define the starting indices for each submatrix
    row_indices = [0]
    col_indices = [0]

    for exp_len in expansion_lengths:
        row_indices.append(row_indices[-1] + exp_len)
        col_indices.append(col_indices[-1] + exp_len)

    # Populate the expanded matrix
    for i in range(num_rows):
        for j in range(num_cols):
            row_start, row_end = row_indices[i], row_indices[i + 1]
            col_start, col_end = col_indices[j], col_indices[j + 1]
            expanded_matrix[row_start:row_end, col_start:col_end] = matrix[i, j]

    return expanded_matrix

import numpy as np


def expand_matrix_fast(matrix, expansion_lengths):
    matrix = matrix[1:-1, 1:-1]  # Trim the CLS and SEP tokens
    
    num_rows, num_cols = matrix.shape
    row_expand = np.repeat(matrix, expansion_lengths, axis=0)
    
    col_expansion_lengths = np.repeat(expansion_lengths, expansion_lengths)
    expanded_matrix = np.repeat(row_expand, col_expansion_lengths, axis=1)

    return expanded_matrix




if __name__ == '__main__':
    main()
    