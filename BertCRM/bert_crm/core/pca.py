import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the data
data = np.load('attention_weights.npy')

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Initialize PCA, specifying the number of components to reduce to
pca = PCA(n_components=2)

# Fit PCA on the standardized data and transform it
reduced_data = pca.fit_transform(data_standardized)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust these parameters based on your dataset
clusters = dbscan.fit_predict(reduced_data)

# Plot the first two principal components with cluster coloring
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, alpha=0.7, cmap='viridis')
plt.title('DBSCAN Clustering on PCA-Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.colorbar(label='Cluster label')
plt.savefig('pca_dbscan_clusters.png', dpi=300)
