###########################
##doing subsampling 
#total Cells: Calculate the total number of cells in your dataset (N).

#Subsample Size: Choose the number of cells (n) in each subsample. This is based on the capacity of your deep learning model and hardware.

#Subsampling Iterations (k): Estimate the number of subsamples needed:


#𝑘=⌈𝑁/𝑛⌉
import scanpy as sc
import pandas as pd
import os
import numpy as np

##read the data
adata=sc.read_h5ad('your_data.h5ad')


# Function to perform stratified subsampling
def stratified_subsample(adata, sample_size, cluster_key='clusters'):
    """
    Perform stratified subsampling on an AnnData object.

    Parameters:
    - adata: AnnData object
    - sample_size: Number of cells to subsample
    - cluster_key: Key in adata.obs to use for stratification

    Returns:
    - Subsampled AnnData object
    """
    # Calculate the proportion of cells in each cluster
    cluster_counts = adata.obs[cluster_key].value_counts()
    cluster_proportions = cluster_counts / cluster_counts.sum()
    
    # Determine the number of cells to sample per cluster
    cluster_sample_sizes = (cluster_proportions * sample_size).round().astype(int)
    
    # Perform stratified sampling
    sampled_indices = []
    for cluster, count in cluster_sample_sizes.items():
        cluster_indices = adata.obs.index[adata.obs[cluster_key] == cluster]
        sampled_indices.extend(np.random.choice(cluster_indices, size=min(len(cluster_indices), count), replace=False))#cells within each cluster are selected without replacement, meaning the same cell barcode won't be picked more than once within a single subsample
    
    # Return subsampled AnnData object
    return adata[sampled_indices].copy()


# Parameters
subsample_size = 30000
num_iterations = int(np.ceil(adata.n_obs / subsample_size))  # Estimate number of iterations

# Perform subsampling in iterations
subsampled_datasets = {}

for i in range(num_iterations):
    subsampled_adata = stratified_subsample(adata, subsample_size, cluster_key='clusters')
    subsampled_datasets[i]=subsampled_adata

    # Print progress
    print(f"Iteration {i + 1}/{num_iterations}: Subsampled {subsampled_adata.n_obs} cells")


subsampled_datasets

##save the subsamples 
import os

# Directory to save subsampled datasets
output_dir = "./subsamples"
os.makedirs(output_dir, exist_ok=True)

# Save each subsample with its iteration number
for iteration, adata in subsampled_datasets.items():
    file_path = os.path.join(output_dir, f"subsample_{iteration}.h5ad")
    adata.write(file_path)
    print(f"Saved: {file_path}")
    