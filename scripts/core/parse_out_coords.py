import re
import sys
from tqdm import tqdm


def parse_coords(file_path: str) -> str:

    remove_chars = r"[',():-]"

    with open('parsed_coords_python.bed', 'w') as fout:
        with open(file_path, 'r') as fin:
            for line in fin:
                result = re.sub(remove_chars, " ", line)
                tab_delimited = re.sub(r"\s+", "\t", result)
                # print(tab_delimited)
                (chrom,
                 start_interval,
                 end_interval,
                 start_pos, end_pos,
                 prediction,
                 prob) = tab_delimited.strip().split("\t")

                new_start = int(start_interval) + int(start_pos)
                new_stop = int(start_interval) + int(end_pos)

                yield f"{chrom}\t{new_start}\t{new_stop}\t{prediction}\t{prob}"


def write_parsed_coords(line_generator) -> None:
    with open('parsed_coords_python.bed', 'w') as fout:
        for line in tqdm(line_generator):
            fout.write(f"{line}\n")


def main() -> None:
    write_parsed_coords(parse_coords(sys.argv[1]))


if __name__ == "__main__":
    main()
