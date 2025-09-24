import sys
from Bio import SeqIO
import random

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <fasta>")
        sys.exit(1)

    fasta = sys.argv[1]
    records = SeqIO.parse(fasta, "fasta")
    
    #randomly shuffle the sequences and split into two files
    records = list(records)
    random.shuffle(records)
    pos_records = records[:len(records)//2]
    neg_records = records[len(records)//2:]
    
    # Add _pos and _neg to the headers and write both sets to one file
    with open('split_crm_pos_neg.fa', 'w') as fout:
        for record in pos_records:
            record.id += '_pos'
            SeqIO.write(record, fout, 'fasta')
        for record in neg_records:
            record.id += '_neg'
            SeqIO.write(record, fout, 'fasta')
        
        
        
if __name__ == "__main__":
    main()