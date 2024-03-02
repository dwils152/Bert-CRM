import argparse
from Bio import SeqIO

class ChromosomeSegmentor:
    def __init__(self, ref_genome, length, overlap, n_threshold):
        self.ref_genome = ref_genome
        self.ref_split_name = self.ref_genome.split(".")[0] + f"_{length}_{overlap}_{n_threshold}.fa"
        self.length = length
        self.overlap = overlap
        self.n_threshold = n_threshold

    def _middle_pad_sequence(self, seq):
        pad_left = (self.length - len(seq)) // 2
        pad_right = self.length - len(seq) - pad_left
        return "N" * pad_left + seq + "N" * pad_right

    def _right_pad_sequence(self, seq):
        pad_right = self.length - len(seq)
        return seq + "N" * pad_right

    def _break_chromosome(self, chromosome, length, overlap):
        seq = str(chromosome.seq).upper()  # Convert sequence to uppercase
        chrom_segments = []
        for start in range(0, len(seq), length - overlap):
            end = start + length
            # Check to see of the end of the interval runs off the end of the chromosome
            if end <= len(seq):
                segment = seq[start:end]
            else:
                # If it does, pad the sequence with Ns
                segment = self._right_pad_sequence(seq[start:])
            n_count = segment.count("N")
            segment_length = len(segment)
            n_percent = n_count / segment_length
            if n_percent <= self.n_threshold:
                chrom_segments.append((start, segment))
        return chrom_segments
    
    def process_fasta(self):
        with open(self.ref_spliti_name, "w") as fout:
                    
            for record in SeqIO.parse(self.ref_genome, "fasta"):
                chrom_segments = self._break_chromosome(record, self.length, self.overlap)
                for start, segment in chrom_segments:
                    fout.write(f">{record.id}:{start}-{start+len(segment)}\n{segment}\n")
        return self.ref_split

def main():
    parser = argparse.ArgumentParser(description='Process fasta sequences.')
    parser.add_argument('ref_genome', type=str, help='Input fasta file')
    parser.add_argument('--length', type=int, default=1000, help='Length of segments')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between segments')
    parser.add_argument('--n_threshold', type=float, default=1.0, help='Maximum fraction of N allowed in a segment')
    args = parser.parse_args()

    # Splits the genome into segments
    segmentor = ChromosomeSegmentor(args.ref_genome, args.length, args.overlap, args.n_threshold)
    segmentor.process_fasta()
    
    
if __name__ == "__main__":
    main()
    


