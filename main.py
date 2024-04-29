# Essential libraries for data manipulation, visualization, and deep learning
import numpy as np
import seaborn as sns
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import umap
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_networkx, from_networkx
import matplotlib
import utils  # Custom utility functions for specific data manipulations and handling
import plots  # Custom plotting utilities for visualizing data
from model_AE import reduction_AE  # Autoencoder model for reducing dimensionality of the data
from model_GAT import Encoder, SenGAE, train_GAT, train_GAT_new  # Graph Attention Network models for complex data structures
from model_Sencell import Sencell, cell_optim, update_cell_embeddings  # Models for senescent cell analysis
from sampling import sub_sampling_by_random, sub_sampling_by_random_v1
from sampling import identify_sengene_then_sencell, identify_sengene_then_sencell_v1
import logging
import os
import argparse  # For parsing command line arguments
import random
import datetime
import scanpy as sp  # Library for single-cell genomics
import wandb  # For tracking experiments, logging data, and visualizing results

is_jupyter = False  # Boolean to check if the script is being run in a Jupyter notebook

current_date = datetime.datetime.now()  # Fetch the current date and time
# Format the current timestamp to be used in file names and logging
datestamp = f"{str(current_date.year)[-2:]}-{current_date.month:02d}-{current_date.day:02d}-{current_date.hour:02d}-{current_date.minute:02d}-{current_date.second:02d}"

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Main program for analyzing senescent cells')
# Arguments for setting up the experiment directory, experiment name, and computational resources
parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to store output files')
parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment for easy identification')
parser.add_argument('--device_index', type=int, default=0, help='Index of the GPU to use for training models')
parser.add_argument('--retrain', action='store_true', default=False, help='Flag to determine whether to train the model from scratch')
parser.add_argument('--timestamp', type=str, default="", help='Timestamp to label output files for version control')

# Hyperparameters specific to the graph models and the senescent cell analysis
parser.add_argument('--gat_epoch', type=int, default=10, help='Number of epochs to train the Graph Attention Network')
parser.add_argument('--sencell_num', type=int, default=300, help='Target number of senescent cells to identify')
parser.add_argument('--sengene_num', type=int, default=200, help='Target number of senescent genes to identify')
parser.add_argument('--sencell_epoch', type=int, default=40, help='Number of epochs for optimizing the senescent cell detection')
parser.add_argument('--cell_optim_epoch', type=int, default=50, help='Number of epochs for optimizing cell embeddings')

parser.add_argument('--batch_id', type=int, default=0, help='Batch ID for processing if data is batched')

# Detect if the script is running in a Jupyter environment and adjust arguments accordingly
if is_jupyter:
    args = parser.parse_args(args=[])
    # Preset some values for testing in a notebook
    args.exp_name = 'OSU_disease_batch1'
    args.output_dir = f'./outputs/{datestamp}-{args.exp_name}'
    args.device_index = 1
    args.retrain = True
    args.gat_epoch = 30
    args.sencell_num = 100
else:
    args = parser.parse_args()

print(vars(args))  # Display all parsed command-line arguments

# Adjust output directory based on whether retraining is set
args.is_jupyter = is_jupyter
if args.retrain:
    args.output_dir = os.path.join(args.output_dir, f"{datestamp}-{args.exp_name}")
else:
    args.output_dir = f"./outputs/{args.timestamp}-{args.exp_name}/"
    print("outdir:", args.output_dir)

print("Outputs dir:", args.output_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Seed setting for reproducibility across random functions, PyTorch, and NumPy
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize Weights & Biases for tracking and logging the experiment
wandb.init(
    project="Sencell",
    name=f"{datestamp}-{args.exp_name}",
    config=vars(args),
    notes=""
)

# Configure basic logging for monitoring the execution process
logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='# %Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

# PART 1: Data Loading and Processing
print("\n====== Part 1: load and process data ======")
# Dynamic data loading based on experiment name, supporting multiple datasets
if 'OSU_healthy' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_OSU_healthy()
elif 'OSU_disease_batch' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_OSU_disease_batch(args.batch_id)
elif 'example' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_example()
elif 'fulldata' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_OSU()
elif 'fixRNA' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_fixRNA()
elif 'fixbatch' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_fixRNA_batch(args.batch_id)
elif 'mouse' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_mouse()
elif 'newfix' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_newfix()

# Additional processing to setup gene and cell arrays, identifying marker genes
new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
    adata, cluster_cell_ls, cell_cluster_arr)
gene_cell = new_data.X.toarray().T  # Create transposed matrix for model input
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]

print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

# If retraining, build a new graph from data, otherwise load existing graph
if args.retrain:
    graph_nx = utils.build_graph_nx(new_data, gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names)
else:
    graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))

if args.retrain:
    print("Save graph_nx")
    torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))

logger.info("Part 1, data loading and processing end!")

# PART 2: Generate Initial Embeddings
print("\n====== Part 2: generate init embedding ======")
device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
print('device:', device)
args.device = device

# Initialize embeddings via Autoencoder if retraining, else load from disk
if args.retrain:
    gene_embed, cell_embed = reduction_AE(gene_cell, device)
    print(gene_embed.shape, cell_embed.shape)
    torch.save(gene_embed, os.path.join(args.output_dir, f'{args.exp_name}_gene.emb'))
    torch.save(cell_embed, os.path.join(args.output_dir, f'{args.exp_name}_cell.emb'))
else:
    print('skip training!')
    gene_embed = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_gene.emb'))
    cell_embed = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_cell.emb'))

# Update graph with embeddings if retraining, prepare PyTorch geometric graph
if args.retrain:
    graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
    graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed)
    torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
    torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
else:
    graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
    graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed)

logger.info("Part 2, AE end!")

# PART 3: Train Graph Attention Network (GAT)
print("\n====== Part 3: train GAT ======")
# Training the GAT model for the specified epochs, allowing retraining and resampling
GAT_model = train_GAT(graph_nx, graph_pyg, args, retrain=args.retrain, resampling=args.retrain, wandb=wandb)
logger.info("Part 3, training GAT end!")

# PART 4: Senescent Cell Optimization
print("\n====== Part 4: sencell optim ======")
all_gene_ls = []  # List to store all gene names
cellmodel = Sencell().to(device)  # Initialize the Sencell model
all_marker_index = sen_gene_ls  # List of indices for senescent gene markers
iteration_results = []  # To store results of each iteration
ratio_cell_ls = []  # List to track cell convergence ratios
ratio_gene_ls = []  # List to track gene convergence ratios

# Sampling subgraphs and initializing dictionaries for senescent and non-senescent cells
sampled_graph, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict, embs = sub_sampling_by_random_v1(
    graph_nx, graph_pyg, sen_gene_ls, nonsen_gene_ls, GAT_model, args, all_marker_index, n_gene=len(all_marker_index),
    gene_rate=0.3, cell_rate=0.5, debug=False)
old_sengene_indexs = all_marker_index  # Track previous indices of senescent genes
lr = 0.01  # Learning rate for the optimizer
optimizer = torch.optim.Adam(cellmodel.parameters(), lr=lr, weight_decay=1e-3)  # Optimizer for model training
scheduler = ExponentialLR(optimizer, gamma=0.85)  # Learning rate scheduler

def check_heatmap(sencell_dict, adata, graph_nx, epoch):
    # Function to generate heatmaps of senescent and non-senescent markers
    nonsenmarkers = utils.load_nonsenmarkers(adata)  # Load non-senescent markers
    senmarkers = utils.load_markers()  # Load senescent markers
    senmarkers = [j for i in senmarkers for j in i]  # Flatten list of senescent markers

    sencell_names = []  # List of senescent cell names
    for key, value in sencell_dict.items():
        sencell_names.append(graph_nx.nodes[key]['name'])
    sen_adata = adata[adata.obs.index.isin(sencell_names)].copy()  # Subset of data for senescent cells

    sp.pl.heatmap(
        sen_adata,
        var_names=nonsenmarkers,
        groupby='cell_type',  # Group by cell type for visualization
        use_raw=False,  # Use processed data
        log=False,  # Do not use log scale
        dendrogram=False,  # Do not use dendrogram
        cmap="gray_r",  # Use grayscale for the heatmap
        figsize=(10, 20),  # Size of the figure
        save=f"{datestamp}-{args.exp_name}-{epoch}-nonsen.png"  # Save file name
    )

    sp.pl.heatmap(
        sen_adata,
        var_names=senmarkers,
        groupby='cell_type',  # Group by cell type
        use_raw=False,  # Use processed data
        log=False,  # Do not use log scale
        dendrogram=False,  # Do not use dendrogram
        cmap="gray_r",  # Use grayscale
        figsize=(10, 20),  # Size of the figure
        save=f"{datestamp}-{args.exp_name}-{epoch}-sen.png"  # Save file name
    )

for epoch in range(args.sencell_epoch):
    logger.info(f"epoch: {epoch}")  # Log the current epoch
    old_sencell_dict = sencell_dict  # Store previous sencell dictionary for comparison

    # Debugging prints for cell and gene details
    for key, value in sencell_dict.items():
        print(graph_nx.nodes[key], value[1], celltype_names[value[1]])
    for key, value in sencell_dict.items():
        print(graph_nx.nodes[key]['name'])

    for gene in sen_gene_indexs:
        print(graph_nx.nodes[gene]['name'])

    check_heatmap(sencell_dict, adata, graph_nx, epoch)  # Generate heatmaps for the epoch

    # Optimization step for sencell model
    cellmodel, sencell_dict, nonsencell_dict = cell_optim(cellmodel, optimizer, sencell_dict, nonsencell_dict, dgl_graph, args, train=True, wandb=wandb)
    scheduler.step()  # Step the scheduler
    current_lr = optimizer.param_groups[0]['lr']  # Log current learning rate
    wandb.log({"lr": current_lr})  # Log learning rate to Weights & Biases

    print("Skip Update CCC graph")  # Skip updating the CCC graph in this iteration
    # Identify senescent and non-senescent genes and cells from the sampled graph
    sencell_dict, nonsencell_dict, sen_gene_indexs, nonsen_gene_indexs = identify_sengene_then_sencell_v1(
        sampled_graph, GAT_model, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict, args)

    ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)  # Calculate cell coverage ratio
    ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_indexs)  # Calculate gene coverage ratio

    # Log coverage ratios and model levels to Weights & Biases
    wandb.log({"cell overlap": ratio_cell, "gene overlap": ratio_gene, "sencell_num": args.sencell_num,
               "level0": cellmodel.levels[0], "level1": cellmodel.levels[1], "level2": cellmodel.levels[2]
               })

    # Adjust number of senescent cells based on coverage ratio
    if len(ratio_cell_ls) > 0 and ratio_cell == ratio_cell_ls[-1]:
        args.sencell_num = int(args.sencell_num * ratio_cell)

    ratio_cell_ls.append(ratio_cell)  # Append current cell ratio to list
    ratio_gene_ls.append(ratio_gene)  # Append current gene ratio to list

    old_sengene_indexs = sen_gene_indexs  # Update old senescent gene indices
    if ratio_cell == 1 and ratio_gene == 1:
        print("Get convergence!")  # Print convergence message
        break  # Break the loop if converged

    break  # Break after one iteration for demonstration purposes

outputs_path = os.path.join(args.output_dir, f'{args.exp_name}_outputs.data')  # Path for saving experiment outputs
print("Experiments saved!", outputs_path)
torch.save([old_sencell_dict, nonsencell_dict, sen_gene_indexs], outputs_path)  # Save experiment data

logger.info("Part 4, sencell optim end!")  # Log end of sencell optimization

# Final results printout
for key, value in sencell_dict.items():
    print(graph_nx.nodes[key], value[1], celltype_names[value[1]])  # Print cell details
for key, value in sencell_dict.items():
    print(graph_nx.nodes[key]['name'])  # Print cell names

for gene in sen_gene_indexs:
    print(graph_nx.nodes[gene]['name'])  # Print gene names

wandb.finish()  # Finish the Weights & Biases session
