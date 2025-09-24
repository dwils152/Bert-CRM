import os
import sys
import argparse
import torch
import multiprocessing as mp
from torch import Tensor
from typing import Tuple, List
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, as_completed

from Bio import SeqIO
from transformers import AutoTokenizer

###############################################################################
# 1) Worker function to process a single chromosome
###############################################################################
def _process_one_chrom(
    chrom_id: str,
    seq_str: str,
    model_name: str,
    chunk_size: int
) -> Tuple[List[str], List[str]]:
    """
    Tokenizes an entire chromosome at once, splits into chunks,
    and returns lists of FASTA lines and BED lines.
    """
    # Create tokenizer inside worker
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 1) Tokenize entire chromosome at once
    with torch.no_grad():
        input_ids = tokenizer(seq_str, return_tensors="pt")["input_ids"].squeeze()

    # Remove the special 'CLS' token at the start (token at index 0)
    input_ids = input_ids[1:]

    # 2) Split the entire chromosome's tokens into fixed-sized chunks
    total_tokens = input_ids.shape[0]
    num_chunks   = total_tokens // chunk_size
    leftover     = total_tokens % chunk_size  # will be ignored

    # If no chunk fits, just return empty results
    if num_chunks == 0:
        print(f"[Worker PID {os.getpid()}] Chromosome {chrom_id} is too short for chunk_size={chunk_size}", 
              file=sys.stderr)
        return [], []

    # We only take the full-sized portion
    input_ids_full = input_ids[: num_chunks * chunk_size]

    # 3) For each chunk, decode and write FASTA/BED lines
    chunks = torch.split(input_ids_full, chunk_size)
    fasta_lines = []
    bed_lines   = []

    # We'll track a global start coordinate across the chromosome
    global_start = 0
    chunk_id = 0

    for chunk in chunks:
        # Convert chunk of IDs to string tokens
        tokens = tokenizer.convert_ids_to_tokens(chunk)
        # Remove any spaces from tokens (often nucleotides won't have them)
        tokens_cleaned = [t.replace(" ", "") for t in tokens]

        joined_tokens = "".join(tokens_cleaned)
        chunk_length  = len(joined_tokens)

        start = global_start
        stop  = start + chunk_length

        # FASTA line: >chr_id:start-stop plus the chunked sequence
        fasta_lines.append(f">{chrom_id}:{chrom_id}_{chunk_id:07d}:{start}-{stop}\n{joined_tokens}\n")

        # BED lines: each token is individually placed
        local_pos = start
        for t in tokens_cleaned:
            tlen = len(t)
            bed_lines.append(f"{chrom_id}\t{local_pos}\t{local_pos + tlen}\t{chrom_id}_{chunk_id:07d}\n")
            local_pos += tlen

        # Update global position for next chunk
        global_start = stop
        chunk_id += 1

    print(f"[Worker PID {os.getpid()}] Processed chromosome: {chrom_id}", file=sys.stderr)

    return fasta_lines, bed_lines

###############################################################################
# 2) GenomeTokenizer class with parallel chromosome processing
###############################################################################
class GenomeTokenizer:
    def __init__(
        self,
        fasta_in: str,
        model_name: str,
        fasta_out: str,
        bed_out: str,
        chunk_size: int = 2047,
        workers: int = None
    ):
        self.fasta_in = fasta_in
        self.model_name = model_name
        self.fasta_out = fasta_out
        self.bed_out = bed_out
        self.chunk_size = chunk_size

        # Default to number of CPU cores if not specified
        self.workers = workers or mp.cpu_count()

    def _read_fasta(self):
        """Yield SeqRecord objects for each chromosome/contig in the FASTA."""
        return SeqIO.parse(self.fasta_in, "fasta")

    def process_chromosomes(self):
        """
        Processes all chromosomes in parallel using ProcessPoolExecutor.
        Writes results to self.fasta_out and self.bed_out.
        """
        with ExitStack() as stack:
            # Open output files once in the main process
            fasta_out_file = stack.enter_context(open(self.fasta_out, 'w'))
            bed_out_file   = stack.enter_context(open(self.bed_out, 'w'))

            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for chrom in self._read_fasta():
                    # Submit one job per chromosome
                    future = executor.submit(
                        _process_one_chrom,
                        chrom_id=chrom.id,
                        seq_str=str(chrom.seq),
                        model_name=self.model_name,
                        chunk_size=self.chunk_size
                    )
                    futures.append(future)

                # Collect each result as it finishes and write to disk
                for future in as_completed(futures):
                    fasta_lines, bed_lines = future.result()
                    fasta_out_file.write(''.join(fasta_lines))
                    bed_out_file.write(''.join(bed_lines))

###############################################################################
# 3) Main driver for CLI usage
###############################################################################
def main(args):
    basename = os.path.basename(args.fasta_in).split('.')[0]
    fasta_out = f'{basename}.chunks.fa'
    bed_out   = f'{basename}.chunks.bed'

    tokenizer = GenomeTokenizer(
        fasta_in=args.fasta_in,
        model_name=args.model,
        fasta_out=fasta_out,
        bed_out=bed_out,
        chunk_size=args.chunk_size,
        workers=args.workers
    )
    tokenizer.process_chromosomes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fasta_in', required=True, help='Input FASTA file')
    parser.add_argument('-m', '--model', default="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
                        help='Transformer model to use for tokenization')
    parser.add_argument('-c', '--chunk_size', type=int, default=2047,
                        help='Chunk size for splitting sequences (in tokens, not bases)')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of parallel processes (default: CPU count)')
    args = parser.parse_args()
    main(args)
