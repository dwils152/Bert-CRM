import argparse
from Bio import SeqIO
import re

def extract_chunk_id(record):
    chunk_id = (
        record.id.split(":")[1]
    )
    return chunk_id

def main(args):

    fasta_out = args.fasta.replace(
        '.fa', f'.lt_{args.proportion_n}_n.fa')
    bed_out = args.bed.replace(
        '.bed', f'.lt_{args.proportion_n}_n.bed')

    id_set = set()
    
    with open(fasta_out, 'w') as fout:
        for record in SeqIO.parse(args.fasta, 'fasta'):
            if record.seq.count("N") / len(record.seq) < float(args.proportion_n):
                id_set.add(extract_chunk_id(record))
                SeqIO.write(record, fout, 'fasta')

    with open(args.bed, 'r') as fin, open(bed_out, 'w') as fout:
        for line in fin:
            chunk_id = line.strip().split('\t')[-1]
            if chunk_id in id_set:
                fout.write(line)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta')
    parser.add_argument('--bed')
    parser.add_argument('--proportion_n')
    args = parser.parse_args()
    main(args)