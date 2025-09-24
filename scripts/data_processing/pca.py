import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load Data
# Replace 'path_to_your_data.csv' with the path to your CSV file containing the k-mer counts
data_path = 'token_dataset.npy'
kmer_data = np.load(data_path)

# Step 2: Normalize Data
# Here, we'll use log normalization, you can switch to another method if preferred
kmer_data_normalized = np.log1p(kmer_data)
label_values = np.load('labels.npy')

# Step 3: Standardize the features (optional, but recommended for PCA)
scaler = StandardScaler()
kmer_data_standardized = scaler.fit_transform(kmer_data_normalized)

# Step 4: Perform PCA
pca = PCA(n_components=2)  # You can adjust the number of components
principal_components = pca.fit_transform(kmer_data_standardized)

# Step 5: Plot the PCA results
plt.figure(figsize=(10, 8))
unique_labels = np.unique(label_values)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    plt.scatter(principal_components[label_values == label, 0], 
                principal_components[label_values == label, 1], 
                color=colors[i], 
                label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of K-mer Spectra')
plt.legend(title='Label', loc='best')
plt.grid(True)
plt.savefig('token_pca.png', dpi=300)
plt.show()

# Step 6: Optionally, print out the explained variance ratio
print("Explained variance by component: ", pca.explained_variance_ratio_)
