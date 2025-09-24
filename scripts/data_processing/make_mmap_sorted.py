#!/usr/bin/env python3
"""
Create a memmap (num_ids, 1000) from a BED file already sorted by ID.

Columns: chrom, start, end, ID, label

Usage:
    python create_memmap_sorted.py <input_bed> <output_mmap> <id_list.txt>

This script:
  1) Reads the BED file in a first pass to collect unique IDs (row order).
  2) Creates a memmap array of shape (num_ids, 1000).
  3) Reads the BED file again and populates each row with 1000 labels.

Assumptions:
  - The BED file is already sorted by the 4th column (ID).
  - Each ID appears in exactly 1000 consecutive lines.
  - The label is in the 5th column, interpretable as an integer.
"""

import sys
import numpy as np


def create_memmap_sorted(
    bed_file: str,
    mmap_file: str,
    id_list_file: str,
    dtype = np.int32
):
    """Create memmap (num_ids, 1000) from a sorted BED file."""

    # ---------- PASS 1: Identify unique IDs in sorted order ----------
    print(f"PASS 1: Collecting unique IDs from {bed_file}")

    unique_ids = []
    current_id = None

    with open(bed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            arr = line.split('\t')
            if len(arr) < 5:
                raise ValueError(f"Line {line_num} has fewer than 5 columns.")

            bed_id = arr[3]
            if bed_id != current_id:
                unique_ids.append(bed_id)
                current_id = bed_id

    num_ids = len(unique_ids)
    print(f"Found {num_ids} unique IDs.")

    # Create and save the ID list
    with open(id_list_file, "w") as outf:
        for uid in unique_ids:
            outf.write(uid + "\n")
    print(f"Wrote ID list to {id_list_file}")

    # ---------- Create the memmap array ----------
    print(f"Creating memmap of shape ({num_ids}, 1000) at '{mmap_file}' ...")
    X = np.memmap(
        mmap_file,
        mode="w+",
        dtype=dtype,
        shape=(num_ids, 1000)
    )
    # We'll track how many labels we've assigned per ID (row)
    counters = np.zeros(num_ids, dtype=np.int64)

    # ---------- PASS 2: Fill the memmap array ----------
    print("PASS 2: Populating memmap...")
    current_id = None
    row_idx = -1  # Will move to 0 at the first new ID

    with open(bed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            arr = line.split('\t')
            if len(arr) < 5:
                raise ValueError(f"Line {line_num} has fewer than 5 columns.")

            bed_id = arr[3]
            label_str = arr[4]

            if bed_id != current_id:
                # Moved to a new ID => increment row_idx
                row_idx += 1
                current_id = bed_id

            # Convert the label to int (or float) as needed
            label_val = int(label_str)

            # Insert the label
            col_idx = counters[row_idx]
            if col_idx >= 1000:
                raise ValueError(
                    f"ID '{bed_id}' has more than 1000 lines (line {line_num})."
                )

            X[row_idx, col_idx] = label_val
            counters[row_idx] += 1

    # Flush/close the memmap
    print("Flushing memmap to disk...")
    X.flush()
    del X

    # Optional: verify each row got exactly 1000 entries
    print("Checking that each ID has exactly 1000 labels...")
    bad_rows = np.where(counters != 1000)[0]
    if len(bad_rows) > 0:
        for r in bad_rows:
            print(f"ID {unique_ids[r]} has {counters[r]} != 1000 labels.")
        raise ValueError("Some IDs do not have exactly 1000 labels.")
    else:
        print("All IDs have 1000 labels as expected.")

    print("Done!")
    print(f"Memmap file: {mmap_file}")
    print(f"ID list file: {id_list_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input_bed> <output_mmap> <id_list.txt>")
        sys.exit(1)

    bed_file = sys.argv[1]
    mmap_file = sys.argv[2]
    id_list_file = sys.argv[3]

    create_memmap_sorted(
        bed_file=bed_file,
        mmap_file=mmap_file,
        id_list_file=id_list_file,
        dtype=np.int32  # Or np.int8, np.float32, etc. depending on your label range
    )
