import scanpy as sc
import numpy as np
import os
import argparse

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
    # Check if the cluster_key exists
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs. Ensure the dataset has clusters.")

    # Get cluster proportions
    cluster_counts = adata.obs[cluster_key].value_counts()
    cluster_proportions = cluster_counts / cluster_counts.sum()

    # Calculate per-cluster sample sizes
    cluster_sample_sizes = (cluster_proportions * sample_size).round().astype(int)

    # Perform stratified sampling
    sampled_indices = []
    for cluster, count in cluster_sample_sizes.items():
        cluster_indices = adata.obs.index[adata.obs[cluster_key] == cluster]
        sampled_indices.extend(np.random.choice(cluster_indices, size=min(len(cluster_indices), count), replace=False))
    
    # Return subsampled AnnData object
    return adata[sampled_indices].copy()

def main(input_file, subsample_size, output_dir):
    """
    Main function to perform subsampling and save results.

    Parameters:
    - input_file: Path to input .h5ad file
    - subsample_size: Number of cells per subsample
    - output_dir: Directory to save subsampled datasets
    """
    # Load the dataset
    print(f"Loading data from: {input_file}")
    adata = sc.read_h5ad(input_file)

    # Compute number of subsampling iterations
    num_iterations = int(np.ceil(adata.n_obs / subsample_size))
    print(f"Total cells: {adata.n_obs} | Subsample size: {subsample_size} | Estimated iterations: {num_iterations}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Perform subsampling and save outputs
    for i in range(num_iterations):
        subsampled_adata = stratified_subsample(adata, subsample_size, cluster_key='clusters')

        # Save the subsampled dataset
        output_file = os.path.join(output_dir, f"subsample_{i}.h5ad")
        subsampled_adata.write(output_file)
        print(f"Iteration {i+1}/{num_iterations}: Saved {subsampled_adata.n_obs} cells to {output_file}")

    print(f" Subsampling completed! All files saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample an h5ad file based on a given subsample size.")
    parser.add_argument("--input", required=True, help="Path to the input .h5ad file")
    parser.add_argument("--subsample_size", type=int, required=True, help="Number of cells per subsample")
    parser.add_argument("--output", required=True, help="Directory to save the output subsampled files")

    args = parser.parse_args()
    main(args.input, args.subsample_size, args.output)
