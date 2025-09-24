#!/usr/bin/env python
import os
import sys
import argparse
import torch
from typing import Tuple, List
from contextlib import ExitStack

from Bio import SeqIO
from transformers import AutoTokenizer

###############################################################################
# 1) Worker function to process a single chromosome
###############################################################################
def _process_one_chrom(
    chrom_id: str,
    seq_str: str,
    tokenizer: AutoTokenizer,  # Changed: tokenizer instance instead of model name
    chunk_size: int
) -> Tuple[List[str], List[str]]:
    """
    Tokenizes an entire chromosome at once, splits into chunks,
    and returns lists of FASTA lines and BED lines.
    """
    # Tokenize entire chromosome at once
    with torch.no_grad():
        input_ids = tokenizer(seq_str, return_tensors="pt")["input_ids"].squeeze()

    # Remove the special 'CLS' token at the start
    input_ids = input_ids[1:]

    total_tokens = input_ids.shape[0]
    num_chunks = total_tokens // chunk_size

    if num_chunks == 0:
        print(f"Chromosome {chrom_id} too short for chunk_size={chunk_size}",
              file=sys.stderr)
        return [], []

    input_ids_full = input_ids[: num_chunks * chunk_size]
    chunks = torch.split(input_ids_full, chunk_size)

    fasta_lines, bed_lines = [], []
    global_start, chunk_id = 0, 0

    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(chunk)
        tokens_cleaned = [t.replace(" ", "") for t in tokens]

        joined_tokens = "".join(tokens_cleaned)
        chunk_length = len(joined_tokens)

        start = global_start
        stop = start + chunk_length

        fasta_lines.append(
            f">{chrom_id}:{chrom_id}_{chunk_id:07d}:{start}-{stop}\n{joined_tokens}\n"
        )

        local_pos = start
        for t in tokens_cleaned:
            tlen = len(t)
            bed_lines.append(
                f"{chrom_id}\t{local_pos}\t{local_pos + tlen}\t{chrom_id}_{chunk_id:07d}\n"
            )
            local_pos += tlen

        global_start = stop
        chunk_id += 1

    print(f"Processed chromosome: {chrom_id}", file=sys.stderr)
    return fasta_lines, bed_lines

###############################################################################
# 2) GenomeTokenizer class -- now serial
###############################################################################
class GenomeTokenizer:
    def __init__(
        self,
        fasta_in: str,
        model_name: str,
        fasta_out: str,
        bed_out: str,
        chunk_size: int = 2047,
    ):
        self.fasta_in = fasta_in
        self.model_name = model_name
        self.fasta_out = fasta_out
        self.bed_out = bed_out
        self.chunk_size = chunk_size

    def _read_fasta(self):
        """Yield SeqRecord objects for each chromosome/contig in the FASTA."""
        return SeqIO.parse(self.fasta_in, "fasta")

    def process_chromosomes(self):
        """
        Processes all chromosomes serially and writes results.
        """
        # Load tokenizer once for all chromosomes
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        with ExitStack() as stack:
            fasta_out_file = stack.enter_context(open(self.fasta_out, 'w'))
            bed_out_file = stack.enter_context(open(self.bed_out, 'w'))

            for chrom in self._read_fasta():
                fasta_lines, bed_lines = _process_one_chrom(
                    chrom_id=chrom.id,
                    seq_str=str(chrom.seq),
                    tokenizer=tokenizer,  # Pass tokenizer instance
                    chunk_size=self.chunk_size
                )
                fasta_out_file.write(''.join(fasta_lines))
                bed_out_file.write(''.join(bed_lines))

###############################################################################
# 3) Main driver for CLI usage
###############################################################################
def main(args):
    basename = os.path.basename(args.fasta_in).split('.')[0]
    fasta_out = f'{basename}.chunks.fa'
    bed_out = f'{basename}.chunks.bed'

    tokenizer = GenomeTokenizer(
        fasta_in=args.fasta_in,
        model_name=args.model,
        fasta_out=fasta_out,
        bed_out=bed_out,
        chunk_size=args.chunk_size
    )
    tokenizer.process_chromosomes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fasta_in', required=True,
                        help='Input FASTA file')
    parser.add_argument('-m', '--model',
                        default="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
                        help='Transformer model to use for tokenization')
    parser.add_argument('-c', '--chunk_size', type=int, default=2047,
                        help='Chunk size for splitting sequences (in tokens, not bases)')
    args = parser.parse_args()
    main(args)