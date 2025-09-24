import matplotlib.pyplot as plt
import sys

# Validate command line arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <input_file> <k-mer_size> <organism>")
    sys.exit(1)

input_file = sys.argv[1]
kmer_size = sys.argv[2]
organism = sys.argv[3]

# Read the k-mer data
kmers = []
counts = []
with open(input_file, 'r') as fin:
    for line in fin:
        kmer, count = line.strip().split()
        kmers.append(kmer)
        counts.append(int(count))

# Sort by k-mers (lexicographical order)
sorted_indices = sorted(range(len(kmers)), key=lambda i: kmers[i])
sorted_kmers = [kmers[i] for i in sorted_indices]
sorted_counts = [counts[i] for i in sorted_indices]

# Plotting
plt.figure(figsize=(10, 6))  # Increase figure size for better legibility
plt.bar(sorted_kmers, sorted_counts, color='red')  # You can customize the color
plt.xlabel('K-mers')
plt.ylabel('Counts')
plt.title(f'Counts of K-mers Sorted Lexicographically for {organism}, k={kmer_size}')
plt.xticks(rotation=45, ha='right')  # Rotate k-mer labels for better readability
plt.tight_layout()  # Adjust layout to make room for label rotation

# Save the plot as a PNG file
output_filename = f'output_{kmer_size}-{organism}.png'
plt.savefig(output_filename, dpi=300)  # Save with high resolution
plt.close()  # Close the plot to free up memory

print(f"Plot saved as {output_filename}")
