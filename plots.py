import seaborn as sns
import umap
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from model_Sencell import get_cluster_cell_dict, getPrototypeEmb
from model_Sencell import Sencell

def plot(d1_ls, d2_ls, d3_ls, d4_ls):
    """
    Plots histograms for four different types of distances between samples.
    
    Args:
    d1_ls, d2_ls, d3_ls, d4_ls (list of lists): Lists containing distance measurements.
    """
    f, ax = plt.subplots(len(d1_ls), 4, figsize=(20, 10), sharex=True)
    for i, (d1, d2, d3, d4) in enumerate(zip(d1_ls, d2_ls, d3_ls, d4_ls)):
        # Convert lists to tensors for plotting
        d1, d2, d3, d4 = map(lambda x: torch.tensor(x), [d1, d2, d3, d4])
        
        # Plot histograms for each distance type
        sns.histplot(d1, alpha=0.5, color='red', label='d1', ax=ax[i, 0])
        sns.histplot(d2, alpha=0.5, color='green', label='d2', ax=ax[i, 1])
        sns.histplot(d3, alpha=0.5, color='black', label='d3', ax=ax[i, 2])
        sns.histplot(d4, alpha=0.5, color='blue', label='d4', ax=ax[i, 3])

        # Customize axes labels
        ax[i, 0].set_ylabel('')
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_ylabel('')
        ax[i, 3].set_ylabel('')
        
    # Set the x-axis limits
    ax[0, 0].set_xlim([0, 2])
    ax[0, 1].set_xlim([0, 2])
    ax[0, 2].set_xlim([0, 2])
    ax[0, 3].set_xlim([0, 2])

    # Add legends to the first row plots
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    ax[0, 3].legend()
    
    plt.ylabel('Count')

def plot_distance(sencell_dict, nonsencell_dict, emb_pos=0):
    """
    Calculates and plots histograms of four types of distances based on embeddings.
    
    Args:
    sencell_dict, nonsencell_dict (dict): Dictionaries containing cell data.
    emb_pos (int): Index position in the dictionaries where embeddings are stored.
    """
    # Compute cell-cluster mappings and prototype embeddings
    cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(sencell_dict, nonsencell_dict)
    prototype_emb = getPrototypeEmb(sencell_dict, cluster_sencell, emb_pos)
    
    # Instantiate a model to calculate distances
    model = Sencell()
    d1, d2, d3, d4 = model.caculateDistance(
        sencell_dict, nonsencell_dict,
        cluster_sencell, cluster_nonsencell,
        prototype_emb, emb_pos
    )
    
    # Plot the calculated distances
    plot(d1, d2, d3, d4)

def plot_umap1(sencell_dict, nonsencell_dict, cluster_names, emb_pos=0):
    """
    Plots a UMAP visualization for all nodes in a subgraph, using labels provided.
    
    Args:
    sencell_dict, nonsencell_dict (dict): Dictionaries containing senescent and non-senescent cell embeddings.
    cluster_names (list): Names of the clusters to be displayed.
    emb_pos (int): Index position in the dictionaries where embeddings are stored.
    """
    # Compute cluster mappings
    cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(sencell_dict, nonsencell_dict)
    
    # Collect embeddings for senescent and non-senescent cells
    sencell_embs = [sencell_dict[cell_index][emb_pos].view(1, -1) for cluster in cluster_sencell.values() for cell_index in cluster]
    sencell_embs = torch.cat(sencell_embs)
    nonsencell_embs = []
    cluster_cell_indexs = []
    count = 0
    cluster_indexs = []

    for cluster, cell_indexs in cluster_nonsencell.items():
        cluster_indexs.append(cluster)
        cluster_cell_index = []
        for cell_index in cell_indexs:
            nonsencell_embs.append(nonsencell_dict[cell_index][emb_pos].view(1, -1))
            cluster_cell_index.append(count)
            count += 1
        cluster_cell_indexs.append(np.array(cluster_cell_index))

    nonsencell_embs = torch.cat(nonsencell_embs)
    embeddings = torch.cat([sencell_embs, nonsencell_embs], 0)

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=100)
    embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(6, 6), dpi=300)
    # Plot senescent cells
    plt.scatter(embeddings[:sencell_embs.shape[0], 0],
                embeddings[:sencell_embs.shape[0], 1], s=5, label='sencell', c='red')
    # Plot each cluster of non-senescent cells with a unique color
    for cell_indexs, cluster_index in zip(cluster_cell_indexs, cluster_indexs):
        cell_embeddings = embeddings[cell_indexs + sencell_embs.shape[0], :]
        plt.scatter(cell_embeddings[:, 0],
                    cell_embeddings[:, 1],
                    alpha=0.1, s=2, label=cluster_names[cluster_index])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

def plot_umap2(adata, sencell_dict, nonsencell_dict, gene_num,
               cell_cluster_indexs, emb_pos=0):
    """
    Plots UMAP visualizations based on precomputed UMAP coordinates from `adata`.
    
    Args:
    adata: Annotated data matrix, typically an AnnData object with UMAP coordinates.
    sencell_dict, nonsencell_dict (dict): Dictionaries containing senescent and non-senescent cell data.
    gene_num (int): Number of genes, used to adjust index calculations.
    cell_cluster_indexs (list of lists): Lists of indices for cells in each cluster.
    emb_pos (int): Position of embeddings in the cell data dictionaries.
    """
    celltype_names = list(adata.obs['ann_level_3_pred'].value_counts().index)
    plt.figure(figsize=(6, 6), dpi=300)
    cmap1 = matplotlib.cm.get_cmap('tab20')
    cmap2 = matplotlib.cm.get_cmap('Set3')
    color_ls = cmap1.colors + cmap2.colors

    # Plot each cell type in different colors
    for i, (cluster, label) in enumerate(zip(cell_cluster_indexs, celltype_names)):
        plt.scatter(adata.obsm['X_umap'][cluster, 0],
                    adata.obsm['X_umap'][cluster, 1],
                    color=color_ls[i], alpha=1, s=1, label=label)

    # Highlight senescent cells
    sencell_indexs = np.array([value[3] for key, value in sencell_dict.items()]) - gene_num
    nonsencell_indexs = np.array([value[3] for key, value in nonsencell_dict.items()]) - gene_num
    plt.scatter(adata.obsm['X_umap'][sencell_indexs, 0],
                adata.obsm['X_umap'][sencell_indexs, 1],
                s=10, label='sencell', c='black', marker='x')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# Additional functions umapPlot, umapPlot_v1, subUmapPlot, and subUmapPlot_v1
# These functions are presumably similar in functionality to plot_umap1 and plot_umap2,
# providing different visualization options for UMAP projections of embeddings.

