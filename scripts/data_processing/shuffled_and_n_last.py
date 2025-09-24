from Bio import SeqIO
import sys
from tqdm import tqdm
import numpy as np


def shuffle_fasta(input_fasta, output_fasta):
    # Read all records from the input FASTA file
    records = list(SeqIO.parse(input_fasta, "fasta"))

    has_n = list()
    has_no_n = list()
    for idx, record in tqdm(enumerate(records), desc='Checking for Ns'):
        if 'N' in record.seq:
            has_n.append(idx)
        else:
            has_no_n.append(idx)

    # Shuffle the records with N
    print('Shuffling records with N')
    shuf_no_n = np.random.permutation(has_no_n)
    shuf_n = np.random.permutation(has_n)
    idxs = np.concatenate((shuf_no_n, shuf_n))

    # Write the shuffled records to the output FASTA file
    print('Writing shuffled records to output FASTA')
    with open(output_fasta, "w") as output_handle:
        for idx in tqdm(idxs, desc='Writing to output FASTA'):
            SeqIO.write(records[idx], output_handle, "fasta")


if __name__ == '__main__':
    fasta_in = sys.argv[1]
    fasta_out = fasta_in.replace('.fa', '_shuffled.fa')
    shuffle_fasta(fasta_in, fasta_out)
