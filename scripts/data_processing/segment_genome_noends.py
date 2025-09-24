import argparse
from typing import List, Tuple
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

class ChromosomeSegmentor:
    def __init__(
        self,
        ref_genome: str,
        length: int,
        overlap: int,
        n_threshold: float,
        padding_type: str = 'right'
    ) -> None:
        
        self.ref_genome: str = ref_genome
        self.ref_split_name: str = (
            self.ref_genome.split(".")[0]
            + f"_{length}_{overlap}_{n_threshold}.fa"
        )
        self.length: int = length
        self.overlap: int = overlap
        self.n_threshold: float = n_threshold
        self.padding_type: str = padding_type

    def _middle_pad_sequence(self, seq: str) -> Tuple[str, int, int]:
        """
        Pads the sequence equally on both sides with Ns.
        Returns (padded_seq, pad_left, pad_right).
        """
        pad_left: int = (self.length - len(seq)) // 2
        pad_right: int = self.length - len(seq) - pad_left
        padded_seq: str = "N" * pad_left + seq + "N" * pad_right
        return padded_seq, pad_left, pad_right

    def _right_pad_sequence(self, seq: str) -> Tuple[str, int, int]:
        """
        Pads the sequence on the right with Ns.
        Returns (padded_seq, pad_left, pad_right).
        """
        pad_left: int = 0
        pad_right: int = self.length - len(seq)
        padded_seq: str = seq + "N" * pad_right
        return padded_seq, pad_left, pad_right

    def _pad_sequence(self, seq: str) -> Tuple[str, int, int]:
        """
        Chooses which padding function to use based on self.padding_type.
        Returns (padded_seq, pad_left, pad_right).
        """
        if self.padding_type == 'middle':
            return self._middle_pad_sequence(seq)
        else:  # default to right padding
            return self._right_pad_sequence(seq)

    def _break_chromosome(
        self,
        chromosome: SeqRecord,
        length: int,
        overlap: int
    ) -> List[Tuple[int, str, int, int]]:
        """
        Breaks a chromosome into segments of specified length with overlap,
        pads if necessary, and applies the N-threshold filter.
        
        Returns:
            List of tuples: (start_index, sequence_segment, pad_left, pad_right)
        """
        seq = str(chromosome.seq).upper()
        chrom_segments: List[Tuple[int, str, int, int]] = []

        # Only go up to len(seq) - length + 1 to avoid partial (padded) segments
        for start in range(0, len(seq) - length + 1, length - overlap):
            end = start + length
            segment = seq[start:end]
            
            # No padding needed, since you never exceed the chromosome length
            pad_left: int = 0
            pad_right: int = 0
            
            n_count = segment.count("N")
            n_percent = n_count / len(segment)
            
            if n_percent <= self.n_threshold:
                chrom_segments.append((start, segment, pad_left, pad_right))

        return chrom_segments
        
    def process_fasta(self) -> str:
        """
        Processes the fasta file, writes segmented records to a new file, and
        returns the output filename.
        """
        with open(self.ref_split_name, "w") as fout:
            for record in SeqIO.parse(self.ref_genome, "fasta"):
                chrom_segments = self._break_chromosome(record, self.length, self.overlap)
                for start, segment, pad_left, pad_right in chrom_segments:
                    total_padding = pad_left + pad_right
                    fout.write(
                        f">{record.id}:{start}-{start+len(segment)}\n"
                        f"{segment}\n"
                    )
        return self.ref_split_name

    def write_bed(self):

        bedname = self.ref_split_name.replace(".fa", ".bed")

        with open(bedname, "w") as fout:
            for record in SeqIO.parse(self.ref_genome, "fasta"):
                chrom_segments = self._break_chromosome(record, self.length, self.overlap)
                for start, segment, pad_left, pad_right in chrom_segments:
                    fout.write(
                        f"{record.id}\t{start}\t{start+len(segment)}\n"
                    )

def main() -> None:
    """
    Main function to parse arguments and run ChromosomeSegmentor.
    """
    parser = argparse.ArgumentParser(description='Process fasta sequences.')
    parser.add_argument('ref_genome', type=str, help='Input fasta file')
    parser.add_argument('--length', type=int, default=8192, help='Length of segments')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between segments')
    parser.add_argument(
        '--n_threshold',
        type=float,
        default=0.5,
        help='Maximum fraction of N allowed in a segment'
    )
    parser.add_argument(
        '--padding_type',
        type=str,
        default='right',
        choices=['right', 'middle'],
        help='Type of padding to apply (default: right)'
    )

    args = parser.parse_args()

    cs = ChromosomeSegmentor(
        ref_genome=args.ref_genome,
        length=args.length,
        overlap=args.overlap,
        n_threshold=args.n_threshold,
        padding_type=args.padding_type
    )

    cs.process_fasta()
    cs.write_bed()

if __name__ == "__main__":
    main()
