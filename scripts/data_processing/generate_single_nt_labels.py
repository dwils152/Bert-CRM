import numpy as np
import sys
import pybedtools
import os

def create_labels_matrix(bin_bed_file, overlap_bed_file, output_memmap_file='labels_matrix.dat', interval_length=8192):
    # Load BED files
    bins = list(pybedtools.BedTool(bin_bed_file))
    overlaps = pybedtools.BedTool(overlap_bed_file).sort().merge()

    num_bins = len(bins)
    
    # Create memmap array: shape (num_bins, interval_length)
    labels_matrix = np.memmap(
        output_memmap_file,
        dtype='uint8',
        mode='w+',
        shape=(num_bins, interval_length)
    )

    # For each bin interval, fill in labels
    for idx, bin_interval in enumerate(bins):
        # Initialize current row to zeros
        labels_matrix[idx, :] = 0

        # Intersect current bin with overlaps
        current_bin = pybedtools.BedTool([bin_interval])
        intersected = current_bin.intersect(overlaps)

        for overlap in intersected:
            # Calculate overlap coordinates relative to bin start
            rel_start = max(overlap.start, bin_interval.start) - bin_interval.start
            rel_end = min(overlap.end, bin_interval.end) - bin_interval.start
            labels_matrix[idx, rel_start:rel_end] = 1

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{num_bins} intervals.")

    # Ensure data is written to disk
    labels_matrix.flush()
    print(f"Labels matrix successfully created at: {output_memmap_file}")

# Example usage:
create_labels_matrix(sys.argv[1], sys.argv[2], 'labels_matrix.dat', interval_length=8192)
