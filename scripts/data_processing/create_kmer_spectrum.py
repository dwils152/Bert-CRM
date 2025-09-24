import argparse
import os
import sys
from typing import List, Tuple, Generator
import json
from collections import Counter
import regex
import re
from transformers import AutoTokenizer
from torch import Tensor
import numpy as np
from tqdm import tqdm
import random
import ahocorasick

def parse_tokenizer(tokenizer: str) -> List[str]:
    with open(tokenizer) as fin:
        tokenizer = json.load(fin)
    vocab = tokenizer['model']['vocab'].keys()
    return list(vocab)
    
def init_specturm(vocab: List[str]) -> Counter:
    return Counter({k: 0 for k in vocab})

def search_kmers(seq: str, vocab: List[str], counter: Counter) -> None:
    for kmer in vocab:
        matches = regex.findall(kmer, seq, overlapped=True)
        counter[kmer] += len(matches)

def build_automaton(vocab):
    A = ahocorasick.Automaton()
    for idx, kmer in enumerate(vocab):
        A.add_word(kmer, (idx, kmer))
    A.make_automaton()
    return A

def search_kmers_aho(seq: str, vocab: List[str], counter: Counter) -> None:
    A = build_automaton(vocab)
    for end_index, (idx, kmer) in A.iter(seq):
        counter[kmer] += 1

def seqs_gen(split_genome: str) -> Generator[Tuple[str, str, str], None, None]:
    delims = r'[:|]'
    with open(split_genome) as fin:
        for line in fin:
            seq, labels, coords = re.split(delims, line.strip())
            yield (seq.strip(), labels.strip(), coords.strip())

def filter_by_labels(labels: str) -> bool:
    labels_list = labels.split(',')
    labels_list_int = [int(label) for label in labels_list]
    if all(label == 0 for label in labels_list_int) or all(label == 1 for label in labels_list_int):
        return True
    return False

def all_seqs_list(split_genome:str) -> List[Tuple[str, str, str]]:
    all_seqs = []
    delims = r'[:|]'
    with open(split_genome) as fin:
        for line in tqdm(fin, desc='Reading sequences', total=3088298):
            seq, labels, coords = re.split(delims, line.strip())
            seq = seq.strip()
            labels = labels.strip()
            coords = coords.strip()
            all_seqs.append((seq, labels, coords))
    return all_seqs

def filter_by_n(seq: str) -> bool:
    return "N" not in seq

def tokenize_seq(seq: str) -> Tensor:
    tokenizer = AutoTokenizer.from_pretrained(
            'zhihan1996/DNABERT-2-117M',
            max_length=1000,
            padding_side="right",
            padding="max_length",
            trust_remote_code=True,
        )
    inputs = tokenizer(seq, max_length=1000, padding="max_length", return_tensors="pt")
    return inputs["input_ids"][0]

def create_kmer_spec_data_labels_random_subset() -> None:
    vocab = parse_tokenizer(args.tokenizer)
    seqs = all_seqs_list(args.split_genome)

    pos_list = []
    neg_list = []
    
    for seq, labels, _ in tqdm(seqs, total=len(seqs), desc='Filtering sequences'):
        if filter_by_labels(labels) and filter_by_n(seq):
            if labels[0] == '1':
                pos_list.append(seq)
            else:
                neg_list.append(seq)
                
    keep_seqs = pos_list + neg_list
    print(len(pos_list))
    print(len(neg_list))
    
    kmer_dataset = np.zeros((len(keep_seqs), len(vocab)))
    for idx, seq in tqdm(enumerate(keep_seqs), total=len(keep_seqs), desc='Creating kmer dataset'):
        counter = init_specturm(vocab)
        search_kmers_aho(seq, vocab, counter)
        kmer_dataset[idx, :] = np.array(list(counter.values()))

    np.save('kmer_dataset.npy', kmer_dataset) 
    
    
    
    
    
def create_kmer_spec_data_labels() -> None:
    vocab = parse_tokenizer(args.tokenizer)
    num_pos = 0
    num_neg = 0
    keep_seqs = []
    labels_list = []

    for seq, labels, _ in seqs_gen(args.split_genome):
        if filter_by_labels(labels) and filter_by_n(seq):
            if labels[0] == '1':
                num_pos += 1
                labels_list.append(1)
            else:
                num_neg += 1
                labels_list.append(0)
            keep_seqs.append(seq)

    np.save('labels.npy', np.array(labels_list))
    print(f'Number of positive sequences: {num_pos}')
    print(f'Number of negative sequences: {num_neg}')

    kmer_dataset = np.zeros((len(keep_seqs), len(vocab)))
    for idx, seq in tqdm(enumerate(keep_seqs), total=len(keep_seqs)):
        counter = init_specturm(vocab)
        #search_kmers_aho(seq, vocab, counter)
        search_kmers(seq, vocab, counter)
        kmer_dataset[idx, :] = np.array(list(counter.values()))

    np.save('kmer_dataset.npy', kmer_dataset)

def create_tok_spec_data_labels() -> None:

    token_ids = list(range(5, 4096))
    counter = Counter({k: 0 for k in token_ids})

    num_pos = 0
    num_neg = 0
    keep_seqs = []
    labels_list = []
    for seq, labels, _ in seqs_gen(args.split_genome):
        if filter_by_labels(labels) and filter_by_n(seq):
            if labels[0] == '1':
                num_pos += 1
                labels_list.append(1)
            else:
                num_neg += 1
                labels_list.append(0)
            keep_seqs.append(seq)

    np.save('labels.npy', np.array(labels_list))
    print(f'Number of positive sequences: {num_pos}')
    print(f'Number of negative sequences: {num_neg}')

    kmer_dataset = np.zeros((len(keep_seqs), len(token_ids)))
    for idx, seq in tqdm(enumerate(keep_seqs), total=len(keep_seqs)):
        tokenized_seq = tokenize_seq(seq).tolist()
        for token in tokenized_seq:
            if token in token_ids:
                counter[token] += 1

    np.save('token_dataset.npy', kmer_dataset)


    


def main(args) -> None:
    create_kmer_spec_data_labels_random_subset()
    #create_kmer_spec_data_labels()
    #create_tok_spec_data_labels()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create kmer spectrum for training seqs.')
    parser.add_argument('--tokenizer', type=str, help='Tokenizer to use.')
    parser.add_argument('--split_genome', type=str, help='Training sequences.')
    args = parser.parse_args()
    main(args)