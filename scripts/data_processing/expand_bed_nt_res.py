#!/usr/bin/env python3

def expand_bed_to_nucleotide(input_file, output_file):
    """
    Expands BED file intervals to nucleotide resolution, copying the score for each position.
    
    Parameters:
    input_file (str): Path to input BED file
    output_file (str): Path to output BED file
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse the BED file fields
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            segment_id = fields[3]
            
            # Write each position within the interval
            for pos in range(start, end):
                fout.write(f"{chrom}\t{pos}\t{pos+1}\t{segment_id}\n")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py input.bed output.bed")
        sys.exit(1)
        
    expand_bed_to_nucleotide(sys.argv[1], sys.argv[2])