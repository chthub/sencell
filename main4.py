import numpy as np
import seaborn as sns
import torch
from torch.optim.lr_scheduler import StepLR,ExponentialLR
# from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GAE
import torch_scatter

# import umap
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_networkx, from_networkx
import matplotlib

import utils
from model_AE import reduction_AE
from model_GAT import GATEncoder,GAEModel
from model_Sencell import Sencell
from model_Sencell import cell_optim, update_cell_embeddings

import logging
import os
import argparse
import random
import datetime
import scanpy as sp

print(f"Start time: {datetime.datetime.now()}")
# import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:19456"

is_jupyter = False
use_wandb=False

datestamp='backbone'

# nohup python -u main.py --exp_name OSU_disease_batch0 --device_index 2 --batch_id 0 --retrain > OSU_disease_batch0.log 2>&1 &
# nohup python -u main.py --exp_name newfix --device_index 2 --retrain > ./log/newfix.log 2>&1 &

parser = argparse.ArgumentParser(description='Main program for sencells')

parser.add_argument('--output_dir', type=str, default='./outputs', help='')
parser.add_argument('--exp_name', type=str, default='data1', help='')
parser.add_argument('--device_index', type=int, default=6, help='')
parser.add_argument('--retrain', action='store_true', default=False, help='')
parser.add_argument('--timestamp', type=str,  default="", help='use default')

parser.add_argument('--seed', type=int, default=40, help='different seed for different experiments')
parser.add_argument('--n_genes', type=str, default='full', help='set 3000, 8000 or full')
parser.add_argument('--ccc', type=str, default='type2', help='type1: cell-cell edge with weight in 0 and 1. type2: cell-cell edge with weight in 0 to 1. type3: no cell-cell edge')
parser.add_argument('--gene_set', type=str, default='full', help='senmayo or fridman or cellage or goterm or goterm+fridman or senmayo+cellage or senmayo+fridman or senmayo+fridman+cellage or full')

parser.add_argument('--gat_epoch', type=int, default=30, help='use default')


# --------------------------------------------------------------------------------------------------- #
# Write these code to fit our data input. This is for our @Yi, @Ahmed, and @Hu.
parser.add_argument('--input_data_count', type=str, default="/bmbl_data/huchen/deepSAS_data/fixed_data_0525.h5ad", help='it is a path to a adata object (.h5ad)')
parser.add_argument('--input_data_CCC_file', type=str, default="", help='it is a path to a CCC file (.csv or .npy)')
# --------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------------------- #
# Subsampling argument for our following version. Please check this @Yi, @Ahmed, and @Hu.
parser.add_argument('--subsampling', action='store_true', default=False, help='subsampling')
# --------------------------------------------------------------------------------------------------- #


# For @Hao
# Hao: Just delete these 3 parameters.
# --------------------------------------------------------------------------------------------------- #
# is this sencell_num parameter not used? Please check this @Hao.
parser.add_argument('--sencell_num', type=int, default=600, help='use default')
# is this sengene_num parameter not used? Please check this @Hao.
parser.add_argument('--sengene_num', type=int, default=200, help='use default')
# is this sencell_epoch parameter not used? Please check this @Hao.
# NOTE: Not use
parser.add_argument('--sencell_epoch', type=int, default=40, help='use default')
# --------------------------------------------------------------------------------------------------- #
# For reproduce
parser.add_argument('--surfix', type=str, default='base_decimal.data')
parser.add_argument('--AE_surfix', type=str, default='base_decimal.pt')


# NOTE: Yi, epoch 1
parser.add_argument('--cell_optim_epoch', type=int, default=50, help='use default')


# For @Hao
# Hao: This is the emb size for GAT hidden embedding size
# --------------------------------------------------------------------------------------------------- #
# is this emb_size parameter for what, for GAT? is default 12? Please check this @Hao.
parser.add_argument('--emb_size', type=int, default=12, help='use default')
# --------------------------------------------------------------------------------------------------- #



parser.add_argument('--batch_id', type=int, default=0, help='use default')

if is_jupyter:
    # jupyter parameter injection
    args = parser.parse_args(args=[])
    args.exp_name = 'combined1'
    args.output_dir=f'/bmbl_data/chenghao/sencell/outputs/'
    args.device_index=4
    args.retrain = True
    args.gat_epoch=30
    args.sencell_num=600

    
    # For @Hao
    # Hao: For GAT
    # --------------------------------------------------------------------------------------------------- #
    # is this emb_size parameter for what, for GAT? is default 32 or 12? Please check this @Hao.
    args.emb_size=12
    args.timestamp=datestamp
    # --------------------------------------------------------------------------------------------------- #

    
    args.seed=40
    args.n_genes='full'
    args.ccc='type1'
    args.gene_set='full'
    
else:
    args = parser.parse_args()
    

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}-{current_date.month:02d}-{current_date.day:02d}-{current_date.hour:02d}-{current_date.minute:02d}-{current_date.second:02d}"
# args.timestamp=datestamp
    

print(vars(args))

args.is_jupyter = is_jupyter
if args.retrain:
    # args.output_dir=os.path.join(args.output_dir,f"{args.timestamp}-{args.exp_name}")
    args.output_dir=f"./outputs/{args.exp_name}/" 
else:
    # args.output_dir=f"/bmbl_data/chenghao/sencell/outputs/{args.timestamp}-{args.exp_name}/" 
    args.output_dir=f"./outputs/{args.exp_name}/" 
    print("outdir:",args.output_dir)
# else:
#     args.output_dir=os.path.join("./outputs/23-11-28-21-45-fixbatch")
print("Outputs dir:",args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# set random seed
seed=args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if use_wandb:
    import wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="Sencell",
        name=f"{datestamp}-{args.exp_name}",
        # track hyperparameters and run metadata
        config=vars(args),
        notes=""
    )
else:
    wandb=None


logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='# %Y-%m-%d %H:%M:%S')

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

# Part 1: load and process data
# cell_cluster_arr is used when get umap embedding
logger.info("====== Part 1: load and process data ======")



# For @Hao
# the new_fix will load data1, add new line in the last part for use specific data path
# --------------------------------------------------------------------------------------------------- #
# Which one is for the final data 1, just leave one, please check this!!!
# We only need one function that input our data, no need a lot of different functions!
if 'combined1' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_combined1()
elif 'combined2' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_combined2()
elif 'combined3' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_combined3()
elif 'newfix' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_newfix()
# --------------------------------------------------------------------------------------------------- #
elif 'data2' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data2()
else:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_newfix(args.input_data_count)

    
# plots.umapPlot(adata.obsm['X_umap'],clusters=cell_cluster_arr,labels=celltype_names)






# For @Hao
# Hao: senescence_marker_list.csv is required in this function
# new_data: processed adata
# markers_index: the list of gene index
# sen_gene_ls: list of senescent genes
# nonsen_gene_ls: list of non senescent genes
# gene_names: list of gene names
# cluster_cell_ls: list of cell indexs in different clusters
# cell_cluster_arr: list of cluster index of each cell
# args: arguments
# 
# --------------------------------------------------------------------------------------------------- #
# please confirm this code and provide interpretation for details, 
# for example, 
# (1) we find senescence_marker_list.csv this file appear in these code context, you need make sure every hidden parameter can be involved.
# (2) what is new_data, so, what is the differences previous data("combined1", "combined2", and "combined3") and it???
new_data, markers_index,\
    sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
        adata, cluster_cell_ls, cell_cluster_arr,args)
# --------------------------------------------------------------------------------------------------- #






    
gene_cell = new_data.X.toarray().T
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]

print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

# if args.retrain:
if not os.path.exists(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data')):
    graph_nx,edge_indexs,ccc_matrix = utils.build_graph_nx(
        new_data,gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names,args)


logger.info("Part 1, data loading and processing end!")


# Part 2: generate init embedding
logger.info("====== Part 2: generate init embedding ======")
use_autoencoder=False

device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print('device:', device)
args.device = device

def run_scanpy(adata):
    sp.pp.normalize_total(adata, target_sum=1e4)
    sp.pp.log1p(adata)
    sp.pp.scale(adata, max_value=10)
    sp.tl.pca(adata, svd_solver='arpack')
    sp.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sp.tl.umap(adata,n_components=args.emb_size)
    
    return adata.obsm['X_umap']

if use_autoencoder:
    if args.retrain:
        gene_embed, cell_embed = reduction_AE(gene_cell, device)
        print(gene_embed.shape, cell_embed.shape)
        torch.save(gene_embed, os.path.join(
            args.output_dir, f'{args.exp_name}_gene.emb'))
        torch.save(cell_embed, os.path.join(
            args.output_dir, f'{args.exp_name}_cell.emb'))
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed,edge_indexs)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
    else:
        print('skip training!')
        gene_embed = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_gene.emb'))
        cell_embed = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_cell.emb'))
        graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed, edge_indexs,ccc_matrix)       

 # NOTE: UMAP embedding as cell/gene embeeding       
else:
    # if args.retrain:
    if not os.path.exists(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data')):
    # For users
    # --------------------------------------------------------------------------------------------------- #
        cell_embed=run_scanpy(new_data.copy())
        print('cell embedding generated!')
        gene_embed=run_scanpy(new_data.copy().T)
        print('gene embedding generated!')
        cell_embed=torch.tensor(cell_embed)
        gene_embed=torch.tensor(gene_embed)
    # this is used to umap embeding
    # --------------------------------------------------------------------------------------------------- #
        
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed,edge_indexs,ccc_matrix)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
        print('graph nx and pyg saved!')
    else:
        print('Load graph nx and pyg ...')
        graph_nx=torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg=torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))

logger.info("Part 2, AE end!")
logger.info("====== Part 3: GAT training ======")

data = graph_pyg
data=data.to(device)
torch.cuda.empty_cache() 

np.random.seed(seed)
torch.manual_seed(seed)
GAT_path=os.path.join(args.output_dir, f'{args.exp_name}_GAT{args.AE_surfix}')
# if args.retrain:
if not os.path.exists(GAT_path):
    # Initialize model and optimizer
    model = GAEModel(args.emb_size, args.emb_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Training the model
    for epoch in range(args.gat_epoch):
        loss = train()
        print(f'Epoch {epoch:03d}, Loss: {loss}')

    torch.save(model, GAT_path)
    print(f'GAT model saved! {GAT_path}')
else:
    print(f'Load GAT from {GAT_path}')
    model=torch.load(GAT_path)
    model=model.to(device)
    
torch.cuda.empty_cache()

logger.info("Part 3, train GAT end!")








logger.info("====== Part 4: Contrastive learning ======")


# For @Hao
# print cell type names of the predicted_cell_indexs
# predicted_cell_indexs: list of cell indexs
# graph_nx: graph object
# celltype_names: list of cell type names
# --------------------------------------------------------------------------------------------------- #
# for each function, please write what the function do, describe parameters 
# For example, what check_celltyps() do??? And every parameter is for what, `predicted_cell_indexs`, `graph_nx`, and `celltype_names` are for what???
def check_celltypes(predicted_cell_indexs, graph_nx, celltype_names):
    cell_types=[]
    for i in predicted_cell_indexs:
        cluster=graph_nx.nodes[int(i)]['cluster']
        cell_types.append(celltype_names[cluster])
    from collections import Counter
    print("snc in different cell types: ",Counter(cell_types))
 # --------------------------------------------------------------------------------------------------- #



# For @Hao
# build a dict to include cell information
# gene_cell: gene cell matrix
# predicted_cell_indexs: list of cell indexs
# GAT_embeddings: embeddings from GAT
# graph_nx: graph object
# --------------------------------------------------------------------------------------------------- #
# what is this function? explain it, what do it produce? 
def build_cell_dict(gene_cell,predicted_cell_indexs,GAT_embeddings,graph_nx):
    sencell_dict={}
    nonsencell_dict={}

    for i in range(gene_cell.shape[0],gene_cell.shape[0]+gene_cell.shape[1]):
        if i in predicted_cell_indexs:
            sencell_dict[i]=[
                GAT_embeddings[i],
                graph_nx.nodes[int(i)]['cluster'],
                0,
                i]
        else:
            nonsencell_dict[i]=[
                GAT_embeddings[i],
                graph_nx.nodes[int(i)]['cluster'],
                0,
                i]
    return sencell_dict,nonsencell_dict
 # --------------------------------------------------------------------------------------------------- #


# For @Hao
# identify senescent genes
# sencell_dict: dict of sncs
# gene_cell: gene cell matrix
# edge_index_selfloop: the edge index with selfloop
# attention_scores: list of attention scores 
# sen_gene_ls: list of sngs
# --------------------------------------------------------------------------------------------------- #
# edge_index_selfloop is for what? 
# FIXME: select senescent genes (SnGs), Running time too long!
def identify_sengene_v1(sencell_dict, gene_cell, edge_index_selfloop, attention_scores, sen_gene_ls):
    print("identify_sengene_v1 ... (optimized) ")
    num_genes = gene_cell.shape[0]
    num_cells = gene_cell.shape[1]
    total_nodes = num_genes + num_cells

    # Create a mask for the selected cells
    cell_indices = torch.tensor(list(sencell_dict.keys()), dtype=torch.long)
    cell_mask = torch.zeros(total_nodes, dtype=torch.bool)
    cell_mask[cell_indices] = True

    # Create a mask for edges where the target node is a gene
    edge_mask_gene = edge_index_selfloop[1] < num_genes
    
    edge_mask_cell = cell_mask[edge_index_selfloop[0]]

    edge_mask_selected = edge_mask_gene & edge_mask_cell

    edges_selected_indices = edge_mask_selected.nonzero().squeeze()

    # Target cell indices and attention scores for selected edges
    selected_edges_targets = edge_index_selfloop[1][edges_selected_indices]
    selected_attention_scores = attention_scores[edges_selected_indices].squeeze()

    # Compute per-gene sums and counts using torch_scatter
    # Counts: Number of times each gene appears in masked_genes
    counts = torch_scatter.scatter(torch.ones_like(selected_attention_scores), selected_edges_targets,
                     dim=0, dim_size=num_genes, reduce='sum')
    # Sums: Sum of attention scores per gene
    sums = torch_scatter.scatter(selected_attention_scores, selected_edges_targets,
                   dim=0, dim_size=num_genes, reduce='sum')
      
    # Avoid division by zero
    res = torch.zeros(num_genes, dtype=torch.float32)
    nonzero_mask = counts > 0
    res[nonzero_mask] = sums[nonzero_mask] / counts[nonzero_mask]

    # Collect scores for sen_gene_ls
    sen_gene_ls = torch.tensor(sen_gene_ls, dtype=torch.long)
    score_sengene_ls = res[sen_gene_ls]

    # Number of genes to update
    # NOTE: select top 10 genes
    # BUGFIX: 1. top 10 genes
    # 2. final parameters
    # 3. abormal genes (28)
    num = 10

    # Get top 'num' new genes with highest scores
    new_genes = torch.topk(res, num).indices

    # Identify indices to keep from sen_gene_ls based on their scores
    sorted_indices = torch.argsort(score_sengene_ls)
    indices_to_keep = sorted_indices[num:]

    new_sen_gene_ls = sen_gene_ls[indices_to_keep]

    # Concatenate the new genes to the updated sen_gene_ls
    new_sen_gene_ls = torch.cat((new_sen_gene_ls, new_genes))
    
    # print("identify_sengene_v1 ... ")
    # cell_index = torch.tensor(list(sencell_dict.keys()))
    # cell_mask = torch.zeros(gene_cell.shape[0] + gene_cell.shape[1], dtype=torch.bool)
    # cell_mask[cell_index] = True

    # res = []
    # score_sengene_ls = []

    # for gene_index in range(gene_cell.shape[0]):
    #     connected_cells = edge_index_selfloop[0][edge_index_selfloop[1] == gene_index]
    #     masked_connected_cells = connected_cells[cell_mask[connected_cells]]

    #     if masked_connected_cells.numel() == 0:
    #         res.append(0)  # Store as integer, less memory
    #     else:
    #         tmp = attention_scores[edge_index_selfloop[1] == gene_index]
    #         attention_edge = torch.sum(tmp[cell_mask[connected_cells]], dim=1)
    #         attention_s = torch.mean(attention_edge)
    #         res.append(attention_s.item())  # Convert to Python scalar

    #     if gene_index in sen_gene_ls:
    #         score_sengene_ls.append(res[-1])
           
    # # NOTE: number of updated genes, top num for new genes, top 10 for sengene
    # num=10
    # res1=torch.tensor(res)
    # new_genes=torch.argsort(res1)[-num:]
    # score_sengene_ls=torch.tensor(score_sengene_ls)
    # if isinstance(sen_gene_ls, torch.Tensor):
    #     new_sen_gene_ls=sen_gene_ls[torch.argsort(score_sengene_ls)[num:].tolist()]
    # else:
    #     new_sen_gene_ls=torch.tensor(sen_gene_ls)[torch.argsort(score_sengene_ls)[num:].tolist()]
    # new_sen_gene_ls=torch.cat((new_sen_gene_ls,new_genes))
    # print(sen_gene_ls)
    # print(new_sen_gene_ls)
    # threshold = torch.mean(res1) + 2*torch.std(res1)  # Example threshold
    # print("the number of identified sen genes:", res1[res1>threshold].shape)
    return new_sen_gene_ls
# --------------------------------------------------------------------------------------------------- #


# For @Hao
# Hao: generate sorted sngs list
# sencell_dict: dict of sncs
# gene_cell: gene cell matrix
# edge_index_selfloop: the edge index with selfloop
# attention_scores: list of attention scores 
# sen_gene_ls: list of sngs
# --------------------------------------------------------------------------------------------------- #
# explain this model as above standard
def get_sorted_sengene(sencell_dict,gene_cell,edge_index_selfloop,attention_scores,sen_gene_ls):
    attention_scores=attention_scores.to('cpu')
    edge_index_selfloop=edge_index_selfloop.to('cpu')
    
    cell_index=torch.tensor(list(sencell_dict.keys()))
    cell_mask = torch.zeros(gene_cell.shape[0]+gene_cell.shape[1], dtype=torch.bool)
    cell_mask[cell_index] = True

    gene_index=0
    res=[]
    score_sengene_ls=[]
    while gene_index<gene_cell.shape[0]:
        connected_cells=edge_index_selfloop[0][edge_index_selfloop[1] == gene_index]
        if len(connected_cells[cell_mask[connected_cells]])==0:
            res.append(torch.tensor(0))
            # print('no sencell in this gene')
        else:
            attention_edge=torch.sum(attention_scores[edge_index_selfloop[1] == gene_index][cell_mask[connected_cells]],axis=1)
            attention_s=torch.mean(attention_edge)
            res.append(attention_s)
        if gene_index in sen_gene_ls:
            score_sengene_ls.append(res[-1])
        gene_index+=1
        
    res1=torch.tensor(res)
    score_sengene_ls=torch.tensor(score_sengene_ls)
    sorted_sengene_ls=torch.tensor(sen_gene_ls)[torch.argsort(score_sengene_ls).tolist()]
    return sorted_sengene_ls
# --------------------------------------------------------------------------------------------------- #


# For @Hao
# Hao: generate cell type specific snc scores
# graph_nx: graph object
# gene_cell: gene cell matrix··
# edge_index_selfloop: the edge index with selfloop
# attention_scores: list of attention scores 
# sen_gene_ls: list of sngs
# celltype_names: list of cell types
# --------------------------------------------------------------------------------------------------- #
# explain this model as above standard
# FIXME: Running time too long! Atention score(sum all sene_gene_ls) for each cell
def generate_ct_specific_scores(sen_gene_ls,gene_cell,edge_index_selfloop,
                                attention_scores,graph_nx,celltype_names):
    print('generate_ct_specific_scores (optimized) ...')
    
    # Convert the list of senescent genes to a tensor index
    gene_index = torch.tensor(sen_gene_ls, dtype=torch.long)
    
    # Create a boolean mask for senescent genes
    total_nodes = gene_cell.shape[0] + gene_cell.shape[1]
    gene_mask = torch.zeros(total_nodes, dtype=torch.bool)
    gene_mask[gene_index] = True

    # Identify edges where the target is a cell node
    cell_offset = gene_cell.shape[0]
    edge_mask_cell = edge_index_selfloop[1] >= cell_offset
    
    # Identify edges where the source is a senescent gene
    edge_mask_gene = gene_mask[edge_index_selfloop[0]]
    
    # Combined mask for edges from senescent genes to cells
    edge_mask_selected = edge_mask_cell & edge_mask_gene
    
    # Indices of selected edges
    edges_selected_indices = edge_mask_selected.nonzero().squeeze()

    # Target cell indices and attention scores for selected edges
    selected_edges_targets = edge_index_selfloop[1][edges_selected_indices]
    selected_attention_scores = attention_scores[edges_selected_indices].squeeze()
    
    # Adjust cell indices to start from 0
    cell_indices_in_range = selected_edges_targets - cell_offset
    
    # Number of cells
    num_cells = gene_cell.shape[1]
    
    # Sum and count attention scores per cell
    attention_sums = torch_scatter.scatter(selected_attention_scores, cell_indices_in_range,
                                           dim_size=num_cells, reduce='sum')
    attention_counts = torch_scatter.scatter(torch.ones_like(selected_attention_scores),
                                             cell_indices_in_range, dim_size=num_cells, reduce='sum')
    
    # Compute mean attention score per cell, handle division by zero
    attention_s_per_cell = torch.zeros(num_cells)
    valid_cells = attention_counts > 0
    attention_s_per_cell[valid_cells] = attention_sums[valid_cells] / attention_counts[valid_cells]
    # Get cluster labels for each cell
    cluster_labels = []
    for cell_idx in range(cell_offset, total_nodes):
        cluster = graph_nx.nodes[int(cell_idx)]['cluster']
        cluster_labels.append(cluster)
    
    # Aggregate scores per cluster
    ct_specific_scores = {}
    for i in range(num_cells):
        cluster = cluster_labels[i]
        attention_s = attention_s_per_cell[i]
        cell_index = cell_offset + i  # Original cell index
        
        if not valid_cells[i]:
            print('no sengene in this cell!')
            continue  # Skip cells with no sensitive genes
        
        score_entry = [float(attention_s), int(cell_index)]
        if cluster in ct_specific_scores:
            ct_specific_scores[cluster].append(score_entry)
        else:
            ct_specific_scores[cluster] = [score_entry]
    

    # print('generate_ct_specific_scores ...')
    # attention_scores=attention_scores.to('cpu')
    # edge_index_selfloop=edge_index_selfloop.to('cpu')
    
    # gene_index=torch.tensor(sen_gene_ls)

    # gene_mask = torch.zeros(gene_cell.shape[0]+gene_cell.shape[1], dtype=torch.bool)
    # gene_mask[gene_index] = True

    # res=[]
    
    # # key is cluster index, value is a 2d list, each row: [score, cell index]
    # ct_specific_scores={}

    # for cell_index in range(gene_cell.shape[0],gene_cell.shape[0]+gene_cell.shape[1]):
    #     connected_genes=edge_index_selfloop[0][edge_index_selfloop[1] == cell_index]
    #     # 这里要考虑到如果cell没有任何老化基因表达，score设为0，pytorch会将其计算为nan，需要额外处理
    #     if len(connected_genes[gene_mask[connected_genes]])==0:
    #         print('no sengene in this cell!')
    #         res.append(torch.tensor(0))
    #     else:
    #         attention_edge=torch.sum(attention_scores[edge_index_selfloop[1] == cell_index][gene_mask[connected_genes]],axis=1)
    #         attention_s=torch.mean(attention_edge)
    #         # res.append(attention_s)
            
    #         cluster=graph_nx.nodes[int(cell_index)]['cluster']
    #         if cluster in ct_specific_scores:
    #             ct_specific_scores[cluster].append([float(attention_s),int(cell_index)])
    #         else:
    #             ct_specific_scores[cluster]=[[float(attention_s),int(cell_index)]]
            
    
    return ct_specific_scores
# --------------------------------------------------------------------------------------------------- #


# For @Hao
# calculate outliers
# scores: SnC socres of cells
# --------------------------------------------------------------------------------------------------- #
# FIXME: select senescent cells (SnCs) 
def calculate_outliers_v1(scores_index):
    scores_index=np.array(scores_index)
    
    counts=0
    snc_index=[]
    
    outliers_ls=[]
    
    scores=scores_index[:,0]
    indexs=scores_index[:,1]
    
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    for i,score in enumerate(scores): 
        if score > upper_bound:
            counts+=1
            snc_index.append(indexs[i])
            outliers_ls.append([score,indexs[i]])
    
    
    return counts,snc_index,outliers_ls
# --------------------------------------------------------------------------------------------------- #


# For @Hao
# Hao: get the cell indexs
# ct_specific_scores: dict of cell type specific snc scores
# --------------------------------------------------------------------------------------------------- #
# explain this model as above standard
def extract_cell_indexs(ct_specific_scores):
    import numpy as np
    from scipy.special import softmax
    
    print("extract_cell_indexs ... ")
    
    data_for_plotting = []
    categories = []

    snc_indexs=[]

    ct_specific_outliers={}


    for key, values in ct_specific_scores.items():
        values_ls=np.array(values)
        data_for_plotting.extend(values_ls[:,0])
        categories.extend([celltype_names[key]] * len(values))

        counts,snc_index,outliers_ls=calculate_outliers_v1(values_ls)
        if counts>=10:
            ct_specific_outliers[key]=outliers_ls
            snc_indexs=snc_indexs+snc_index


    return snc_indexs
# --------------------------------------------------------------------------------------------------- #

cellmodel = Sencell(args.emb_size).to(device)
data=data.to(device)
lr=0.01
optimizer = torch.optim.Adam(cellmodel.parameters(), lr=lr,
                        weight_decay=1e-3)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.85)
sencell_dict=None


# Extracting attention scores
edge_index_selfloop_cell,attention_scores_cell = model.get_attention_scores(data)
attention_scores_cell=attention_scores_cell.cpu().detach()
attention_scores_cell = torch.trunc(attention_scores_cell*10000)/ 10000
edge_index_selfloop_cell=edge_index_selfloop_cell.cpu().detach()

model.eval()
for epoch in range(5):
    print(f'{datetime.datetime.now()}: Contrastive learning Epoch: {epoch:03d}')
    
    # cell part
    ct_specific_scores=generate_ct_specific_scores(sen_gene_ls,gene_cell,
                                                    edge_index_selfloop_cell,
                                                    attention_scores_cell,
                                                    graph_nx,celltype_names)
    

    predicted_cell_indexs=extract_cell_indexs(ct_specific_scores)
    check_celltypes(predicted_cell_indexs,graph_nx,celltype_names)
    
    if sencell_dict is not None:
        old_sencell_dict=sencell_dict
    else:
        old_sencell_dict=None
    
    GAT_embeddings=model.encode(data.x,data.edge_index).detach()
    GAT_embeddings = torch.trunc(GAT_embeddings*10000)/ 10000
    sencell_dict, nonsencell_dict=build_cell_dict(gene_cell,predicted_cell_indexs,GAT_embeddings,graph_nx)
    
    if old_sencell_dict is not None:
        ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)

    
    # gene part
    cellmodel, sencell_dict, nonsencell_dict = cell_optim(cellmodel, optimizer,
                                                            sencell_dict, nonsencell_dict,
                                                            None,
                                                            args,
                                                            train=True,
                                                            wandb=wandb)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print('current lr:',current_lr)
    
    # update gat embeddings

    new_GAT_embeddings=GAT_embeddings
    for key,value in sencell_dict.items():
        new_GAT_embeddings[key]=sencell_dict[key][2].detach()
    for key,value in nonsencell_dict.items():
        new_GAT_embeddings[key]=nonsencell_dict[key][2].detach()
        
    data.x=new_GAT_embeddings
    # new embeeding input to GAT
    edge_index_selfloop,attention_scores = model.get_attention_scores(data)
    
    attention_scores=attention_scores.to('cpu')
    attention_scores = torch.trunc(attention_scores*10000)/ 10000

    edge_index_selfloop=edge_index_selfloop.to('cpu')
    
    old_sengene_indexs=sen_gene_ls
    #     sen_gene_ls=identify_sengene_v2(new_data,
    #         sencell_dict,gene_cell,edge_index_selfloop,attention_scores,sen_gene_ls)
    #     ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_ls)
    sen_gene_ls=identify_sengene_v1(
        sencell_dict,gene_cell,edge_index_selfloop,attention_scores,sen_gene_ls)


    ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_ls)
    
    torch.save([sencell_dict,sen_gene_ls,attention_scores,edge_index_selfloop],
               os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-epoch{epoch}{args.surfix}'))
    
print(f"End time: {datetime.datetime.now()}")