#!/usr/bin/env python3
"""
Example: Using Dask to read and group a large BED file on an HPC cluster with SLURM.
Then create a memmap (num_ids, 1000) with the grouped labels.
"""

import argparse
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Dask to group a BED file by ID and create a memmap."
    )
    parser.add_argument("--bed_file", required=True, help="Path to the input BED file")
    parser.add_argument("--memmap_file", required=True, help="Output memmap file")
    parser.add_argument("--idlist_file", required=True, help="Output ID list file")
    parser.add_argument("--npartitions", type=int, default=32, 
                        help="Number of Dask partitions to read the file into")
    parser.add_argument("--threads_per_worker", type=int, default=8,
                        help="Number of threads per Dask worker (if using LocalCluster)")
    return parser.parse_args()

def main():
    args = parse_args()

    bed_file = args.bed_file
    memmap_file = args.memmap_file
    idlist_file = args.idlist_file
    npartitions = args.npartitions
    threads_per_worker = args.threads_per_worker

    # 1) Start a local Dask cluster on this node
    #    (If you're using multiple nodes, you'd do something more advanced with dask_jobqueue or dask-mpi.)
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=threads_per_worker,
        memory_limit="0",  # Let the job's cgroup limit memory usage
        scheduler_port=0,  # random free port
    )
    client = Client(cluster)
    print(client)

    # 2) Read the BED file with Dask
    #    We assume columns: chrom, start, end, id, label
    #    If your file has no header, we use "header=None" and manually name columns
    print(f"Reading BED file into Dask DataFrame: {bed_file}")
    ddf = dd.read_csv(
        bed_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "id", "label"],
        dtype={
            "chrom": str,
            "start": "int64",
            "end": "int64",
            "id": str,
            "label": "int64"
        },
        blocksize=None,       # Let dask figure out chunk sizes
        # OR we can specify "blocksize='256MB'", etc.
    )
    ddf = ddf.repartition(npartitions=npartitions)
    print("Dask DataFrame created with:", ddf.npartitions, "partitions")

    # 3) Group by 'id' and collect labels into a list
    #    We'll use a function that aggregates the 'label' column.
    #    Dask groupby-apply for collecting lists can be done with:
    #        .apply(lambda x: x.tolist())
    #    But there's also a built-in "aggregate" approach with .agg
    #    We'll do something straightforward here.
    print("Grouping by ID and collecting labels. This can be memory-intensive!")
    grouped_ddf = ddf.groupby("id")["label"].apply(lambda x: list(x), meta=('label','object'))

    # 4) Trigger computation to get a Pandas Series with index=ID, value=list_of_labels
    #    This operation will bring the grouped data to the driver node (this node).
    #    Make sure you have enough RAM to handle the final result in memory.
    grouped_series = grouped_ddf.compute()
    print("Groupby -> list collection is done. Result type:", type(grouped_series))
    print("Number of unique IDs =", len(grouped_series))

    # 5) Convert the Dask result to a normal Pandas Series (it already should be one)
    #    The index is each unique ID, the value is a list of labels
    #    We'll get the index (sorted or not) and the data
    unique_ids = grouped_series.index.tolist()
    lists_of_labels = grouped_series.values  # each entry is a Python list

    num_ids = len(unique_ids)
    print(f"Found {num_ids} unique IDs. Creating memmap of shape ({num_ids}, 1000).")

    # 6) Create the memmap array
    #    We'll assume each ID has EXACTLY 1000 labels.
    #    Adjust dtype as needed.
    X = np.memmap(
        memmap_file,
        mode="w+",
        dtype=np.int32,  # or int8, etc., depending on your label range
        shape=(num_ids, 1000)
    )

    # 7) Write the ID list to a file (row -> ID)
    with open(idlist_file, "w") as f:
        for uid in unique_ids:
            f.write(f"{uid}\n")

    # 8) Fill the memmap array
    print("Filling memmap array with labels...")
    for row_idx, (uid, label_list) in enumerate(zip(unique_ids, lists_of_labels)):
        if len(label_list) != 1000:
            raise ValueError(f"ID '{uid}' has {len(label_list)} labels, expected 1000!")
        X[row_idx, :] = label_list

    # 9) Flush to disk
    X.flush()
    del X
    print("Memmap creation finished!")

    # Cleanly shut down the Dask cluster
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
