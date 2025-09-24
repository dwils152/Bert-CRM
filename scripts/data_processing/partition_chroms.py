import argparse
import pandas as pd
import heapq
import seaborn as sns
import matplotlib.pyplot as plt

def chroms_sizes(chrom_sizes: str) -> pd.DataFrame:
    return pd.read_csv(
        chrom_sizes, sep='\t', header=None, names=['chrom', 'size'],
    )


def sum_tuple(tup: tuple[str, int]) -> int:
    return sum(i[1] for i in tup)


def pack_chromosomes(chrom_df: pd.DataFrame, partitions: int) -> list[list[tuple[str, int]]]:
    """
    Partition chromosomes into roughly equal groups by size
    using a greedy Largest-First strategy.
    """
    chrom_df = chrom_df.sort_values(by="size", ascending=False)

    # (total_size, partition_index)
    heap: List[Tuple[int, int]] = [(0, i) for i in range(partitions)]
    heapq.heapify(heap)

    folds: List[List[Tuple[str, int]]] = [[] for _ in range(partitions)]

    for row in chrom_df.itertuples(index=False):
        size = int(row.size)
        chrom = str(row.chrom)

        total, idx = heapq.heappop(heap)
        folds[idx].append((chrom, size))
        heapq.heappush(heap, (total + size, idx))

    return folds

def folds_to_df(folds: list[list[tuple[str, int]]]) -> pd.DataFrame:
    tups = []
    for idx, fold in enumerate(folds):
        for contig in fold:
            tups.append((contig[0], contig[1], idx))
    return pd.DataFrame(tups, columns=['chrom', 'size', 'fold'])


import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_chrom_partitions(df: pd.DataFrame, outfile: str | None = None) -> None:
    grouped = df.groupby(["fold", "chrom"], as_index=False)["size"].sum()

    pivoted = grouped.pivot(index="fold", columns="chrom", values="size").fillna(0)

    ax = pivoted.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        edgecolor="black",
        linewidth=0.5,
        colormap="tab20"
    )

    ax.set_ylabel("Total size (bp)")
    ax.set_xlabel("Fold")
    ax.set_title("Chromosome composition of each fold")
    plt.xticks(rotation=0)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()




def main(args) -> None:
    main_chroms = {
        "Fly": (
            "chr2L", "chr2R", "chr3L", "chr3R",
            "chr4", "chrX", "chrY"
        ),
        "Worm": (
            "chrI", "chrII", "chrIII", "chrIV",
            "chrV", "chrX"
        ),
        "Plant": (
            "chr1", "chr2", "chr3", "chr4", "chr5"
        ),
        "Human": (
            "chr1", "chr2", "chr3", "chr4", "chr5",
            "chr6", "chr7", "chr8", "chr9", "chr10",
            "chr11", "chr12", "chr13", "chr14", "chr15",
            "chr16", "chr17", "chr18", "chr19", "chr20",
            "chr21", "chr22", "chrX", "chrY"
        ),
        "Mouse": (
            "chr1", "chr2", "chr3", "chr4", "chr5",
            "chr6", "chr7", "chr8", "chr9", "chr10",
            "chr11", "chr12", "chr13", "chr14", "chr15",
            "chr16", "chr17", "chr18", "chr19", "chrX", "chrY"
        ),
    }

    sizes_df = chroms_sizes(args.chrom_sizes)
    main_only = sizes_df[sizes_df["chrom"].isin(main_chroms[args.organism])]

    bins = pack_chromosomes(main_only, args.k+1)
    partition_sizes = [sum_tuple(fold) for fold in bins]
    print(partition_sizes)

    plot_chrom_partitions(folds_to_df(bins), 'partitions.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--organism",
        required=True,
        type=str, 
        choices=['Fly', 'Worm', 'Plant', 'Human', 'Mouse'],
        help="Organism simple name to preprocess")
    parser.add_argument(
        "--chrom_sizes",
        required=True,
        type=str,
        help="The bed file of the assembly for the organism")
    parser.add_argument(
        "--k",
        type=int,
        help="Number of folds to partition the chromosomes into")
    args = parser.parse_args() 
    main(args)
