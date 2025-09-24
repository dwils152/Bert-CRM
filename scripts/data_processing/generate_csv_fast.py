from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import torch
from torch import Tensor
import pybedtools
from tqdm import tqdm
import sys
import pandas as pd


@dataclass
class Coordinate:
    chrom: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.chrom}-{self.start}-{self.end}"


class CsvGenerator():
    def __init__(self, fasta: Path, crm: Path):
        self.fasta = fasta
        self.crm = crm
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            model_max_length=1024,
            padding='max_length',
            padding_size='right',
            trust_remote_code=True
        )


    def _parse_fasta(self) -> Tuple:
        """
        Parse a fasta file and return a tuple of SeqRecord objects
        """
        return tuple(SeqIO.parse(self.fasta, "fasta"))


    def _tokenize_seq(self, record) -> Tuple[Coordinate, Tensor, Tensor]:
        """
        Tokenize a sequence using BPE from the pretrained DNA-BERT 2 model.
        Token IDs are returned from the tokenizer and will need to be mapped back to 
        the original sequence using the tokenizer's vocab.
        """
        seq = str(record.seq)
        seq = seq.replace('N', '[MASK]')
        coords = record.id
        chrom, start_end = coords.split(':')
        start, end = start_end.split('-')
        coords = Coordinate(chrom=chrom, start=int(start), end=int(end))

        inputs = self.tokenizer(seq,
                                padding='max_length',
                                truncation=True,
                                return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return coords, input_ids


    def _map_id_to_seq(self, tokenized_seq) -> Tuple[Coordinate, List[str]]:
        """
        Map token IDs back to the original sequence
        """
        coords, inputs = tokenized_seq

        id_to_seq = {v: k for k, v in self.tokenizer.get_vocab().items()}
        tokens_list = list()
        for i in range(len(inputs[0])):
            tokens_list.append(id_to_seq[inputs[0][i].item()])

        return coords, tokens_list


    def _get_token_coords(self, coord, token_list) -> List[Coordinate]:
        """
        Get the coordinates of each token in the original sequence
        """
        start = coord.start
        coord_list = list()
        for word in token_list:
            if word != '[CLS]' and word != '[SEP]' and word != '[PAD]':
                if word == '[MASK]':
                    coord_list.append(Coordinate(
                        chrom=coord.chrom, start=start, end=start+1))
                    start += 1
                else:
                    coord_list.append(Coordinate(
                        chrom=coord.chrom, start=start, end=start+len(word)))
                    start += len(word)

        return coord_list

    def _get_token_labels(self, coord_list) -> List[int]:
        """
        Get the labels for each token. 1 for CRM, 0 for NCRM. The threshold for CRM is 0.5.
        """
        bed_format = "\n".join(
            f"{coord.chrom}\t{coord.start}\t{coord.end}" for coord in coord_list)

        #print(bed_format)

        token_coords = pybedtools.BedTool(bed_format, from_string=True)

        crm_coords = pybedtools.BedTool(self.crm)
        intersect_crm = token_coords.intersect(crm_coords, u=True, f=0.5)

        # Create a set of CRM intersected intervals
        crm_intervals = set()
        for interval in intersect_crm:
            crm_intervals.add((interval.chrom, interval.start, interval.end))

        # Determine labels
        labels = []
        for coord in coord_list:
            interval = (coord.chrom, coord.start, coord.end)
            label = 1 if interval in crm_intervals else 0
            labels.append(label)

        return labels

    def _intersect_token_crm(self, token_bed_string, crm_file):
        
        # Create pybedtools object from the token_df and crm_file
        token_coords = pybedtools.BedTool(token_bed_string, from_string=True)
        crm_coords = pybedtools.BedTool(self.crm)
        intersect_crm = token_coords.intersect(crm_coords, u=True, f=0.5)
        return intersect_crm


    def generate_csv(self, output: Path):

        seqs = self._parse_fasta()
        
        seqs_dict = {}
        # Create a dictionary that maps the sequence id to the sequence
        for idx, record in enumerate(seqs):
            seqs_dict[record.id] = str(record.seq)
        
        all_token_coords = {}

        for idx, record in tqdm(enumerate(seqs), total=len(seqs), desc="Generating CSV"):

                tokenized = self._tokenize_seq(record)
                coords, words = self._map_id_to_seq(tokenized)
                token_coords = self._get_token_coords(coords, words)
                all_token_coords[record.id] = token_coords

        transformed = [(value, key) for key, values in all_token_coords.items() for value in values]
        coord_df = pd.DataFrame(transformed, columns=['coords', 'seq_id'], dtype=str)

        # Split the coords column on : and - to get chrom, start, and end
        coord_df[['chrom', 'start', 'end']] = coord_df['coords'].str.split('[:-]', expand=True)
        
        # Drop the coords column
        coord_df.drop(columns=['coords'], inplace=True)
        coord_df = coord_df[['chrom', 'start', 'end', 'seq_id']]

        # Convert DataFrame to BED formatted string
        bed_string = '\n'.join(coord_df.apply(lambda row: 
            f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['seq_id']}", axis=1))
        
        intersection = self._intersect_token_crm(bed_string, self.crm).to_dataframe()

        if intersection.empty:
            coord_df['label'] = 0
            coord_df['label'] = coord_df['label'].astype(str)
        else:
            

            # Change the start and end columns to integers
            coord_df['start'] = coord_df['start'].astype(int)
            coord_df['end'] = coord_df['end'].astype(int)

            # For the rows in the dataframe, if the entire row is in the intersection, set the label to 1
            # If not, set the label to 0
            coord_df = coord_df.merge(intersection, on=['chrom', 'start', 'end'], how='left')

            # If name is NaN then the label is 0, else the label is 1
            coord_df['label'] = coord_df['name'].apply(lambda x: 0 if pd.isna(x) else 1)
            coord_df['label'] = coord_df['label'].astype(str)

        collapsed_df = coord_df.groupby('seq_id')['label'].apply(list).reset_index()
        collapsed_df['label'] = collapsed_df['label'].apply(lambda x: ','.join(x))

        # Add the sequence to the dataframe using the seq_id
        collapsed_df['seq'] = collapsed_df['seq_id'].apply(lambda x: seqs_dict[x])
        collapsed_df['seq_id'] = collapsed_df['seq_id'].str.replace(":", "-")
        with open(output, 'w') as fout:
            # Iterate over the df
            for idx, row in collapsed_df.iterrows():
                fout.write(f"{row['seq']}: {row['label']}| {row['seq_id']}\n")
            


def main():

    torch.set_printoptions(threshold=10_000)
    csv_generator = CsvGenerator(
        Path(sys.argv[1]),  # fasta
        Path(sys.argv[2]),  # crm
    )

    csv_generator.generate_csv(Path(sys.argv[1] + ".csv"))


def debug():

    torch.set_printoptions(threshold=10_000)

    csv_generator = CsvGenerator(
        Path(sys.argv[1]),
        Path(sys.argv[2])
    )

    parsed_fasta = csv_generator._parse_fasta()
    test_seq = parsed_fasta[5]
    test_tokenized = csv_generator._tokenize_seq(test_seq)
    csv_generator._map_id_to_seq(test_tokenized)


if __name__ == "__main__":
    main()
