from Bio import SeqIO
import sys

reference_genome = sys.argv[1]

with open('all_Ns.bed', 'w') as fout:
    with open(reference_genome) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            chrom = record.id
            for idx, char in enumerate(record.seq):
                if char == "N":
                   fout.write(f'{chrom}\t{idx}\t{idx+1}\n') 