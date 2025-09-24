import sys

def main():

    chunks = sys.argv[1]
    token_size = int(sys.argv[2])

    with open(chunks, 'r') as fin:
        with open(f'{chunks}.tokens', 'w') as fout:
            for line in fin:
                line_parts = line.strip().split('\t')
                chrom, start, end, crm_id = line_parts
                start = int(start)
                end = int(end)
                for i in range(start, end, token_size):

                    end = start + token_size
                    token_line = f'{chrom}\t{str(start)}\t{str(end)}\t{crm_id}\n'
                    fout.write(token_line)

                    start += token_size
                   

if __name__ == "__main__":
    main()