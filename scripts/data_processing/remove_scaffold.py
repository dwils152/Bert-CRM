import sys

def is_scaffold(header):
    # Define keywords that identify scaffold sequences
    scaffold_keywords = ["_random", "_alt", "chrUn"]
    return any(keyword in header for keyword in scaffold_keywords)

def filter_scaffolds(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        write_sequence = True
        for line in infile:
            if line.startswith('>'):  # Header line
                write_sequence = not is_scaffold(line)
            if write_sequence:
                outfile.write(line)

def main():
    genome = sys.argv[1]
    filter_scaffolds(genome, genome.replace('.fa', '_no_scaffolds.fa'))

if __name__ == '__main__':
    main()
