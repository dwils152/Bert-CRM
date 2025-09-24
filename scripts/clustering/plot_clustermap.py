import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: plot_clustermap.py <pairwise.tab> <out.png>")
    sys.exit(1)

pairwise_file = sys.argv[1]
out_file = sys.argv[2]

# Mash output columns:
# [query, ref, distance, p-value, shared-hashes]
df = pd.read_csv(pairwise_file, sep="\t", header=None,
                 names=["query", "ref", "dist", "pval", "shared"])

# Pivot into symmetric distance matrix
matrix = df.pivot(index="query", columns="ref", values="dist")

# Fill missing values with 0 (self-distances)
matrix = matrix.fillna(0)

# Draw clustered heatmap
sns.set(style="white")
g = sns.clustermap(matrix, cmap="viridis", figsize=(12, 12),
                   cbar_kws={"label": "Mash distance"},
                   xticklabels=False, yticklabels=False)

plt.savefig(out_file, dpi=300, bbox_inches="tight")
