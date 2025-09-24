import sys
import re

delims = r'[|:-]'

with open(sys.argv[1], 'r') as fin:
    with open(f'{sys.argv[1]}.pad_int', 'w') as fout:
        for line in fin:
            line = line.strip()
            if line.startswith(">"):
                line = line.replace(">", "")
                chrom, start, end = re.split(delims, line)
                pad_int = 0
                if int(end) - int(start) < 1000:
                    pad_int = 1000 - (int(end) - int(start))

                if pad_int < 0:
                    raise ValueError('Padding int should not be negative')
                
                line = f'>{chrom}-{start}:{end}|{pad_int}'
            fout.write(line + '\n')


        
