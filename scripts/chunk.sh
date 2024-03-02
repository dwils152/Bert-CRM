#!/bin/bash

# Usage: ./split_fasta.sh input.fasta 100
# This will split input.fasta into multiple files, each with 100 records

input_file=$1  # Input FASTA file
chunk_size=$2  # Number of records per chunk

# Check if input file and chunk size are provided
if [ -z "$input_file" ] || [ -z "$chunk_size" ]; then
    echo "Usage: $0 <input_file> <chunk_size>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File $input_file not found"
    exit 1
fi

# Create output directory
output_dir="fasta_chunks"
mkdir -p "$output_dir"

# Split the file
awk -v chunk_size="$chunk_size" -v output_dir="$output_dir" '
    BEGIN { record_count = 0; file_count = 1; }
    /^>/ { 
        if (record_count % chunk_size == 0 && record_count > 0) {
            close(output_file);
            file_count++;
        }
        output_file = output_dir "/chunk_" file_count ".fa";
        record_count++;
    }
    { print > output_file; }
' "$input_file"

echo "FASTA file split into chunks in $output_dir"
