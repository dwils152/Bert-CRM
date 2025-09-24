from Bio import SeqIO
import sys

def main() -> None:
    
    with open(sys.argv[2], 'w') as fout:
        with open(sys.argv[1], 'r') as fin:
            # convert the sequences to uppercase and write them to the output file
            for record in SeqIO.parse(fin, "fasta"):
                fout.write(f">{record.id}\n{record.seq.upper()}\n")
                
if __name__ == "__main__":
    main()