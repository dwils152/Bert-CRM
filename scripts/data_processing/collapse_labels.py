#!/usr/bin/env python3
import argparse
import numpy as np
from collections import defaultdict

def main(args):
    # First pass: read the file and store labels in a dictionary
    # dict_of_lists[id_str] = [label1, label2, ..., label1000]
    dict_of_lists = defaultdict(list)

    print("Reading input...")
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Line has unexpected format: {line}")
            id_str, label_str = parts
            dict_of_lists[id_str].append(int(label_str))

    # Validate that each ID has exactly 1000 labels
    ids = sorted(dict_of_lists.keys())
    for i in ids:
        if len(dict_of_lists[i]) != 1000:
            raise ValueError(
                f"ID {i} has {len(dict_of_lists[i])} labels, but expected 1000."
            )

    num_ids = len(ids)

    print(f"Found {num_ids} IDs. Creating memmap of shape: ({num_ids}, 1000)")

    # Create the memory map array
    # Adjust dtype as needed, e.g. 'int8', 'int16', or 'float32'
    arr = np.memmap(
        args.output,
        dtype='int8',
        mode='w+',
        shape=(num_ids, 1000)
    )

    # Fill the memmap array with label data
    print("Writing labels to memmap...")
    for idx, id_str in enumerate(ids):
        labels = dict_of_lists[id_str]
        arr[idx, :] = labels

    # Flush changes to disk
    arr.flush()
    print(f"Done. Labels saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group by ID, collect 1000 labels for each, and write to a (num_ids x 1000) memmap."
    )
    parser.add_argument("--input", required=True, help="Path to the input text file.")
    parser.add_argument("--output", required=True, help="Path to the output memmap file.")
    args = parser.parse_args()

    main(args)
