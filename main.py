import numpy as np
import seaborn as sns
import torch

import umap
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_networkx, from_networkx
import matplotlib

import utils
import plots
from model_AE import reduction_AE
from model_GAT import Encoder, SenGAE, train_GAT
from model_Sencell import Sencell
from sampling import sub_sampling_by_random
from model_Sencell import cell_optim, update_cell_embeddings
from sampling import identify_sengene_then_sencell

import logging
import os
import argparse

is_jupyter = False

parser = argparse.ArgumentParser(description='Main program for sencells')

parser.add_argument('--output_dir', type=str, default='./outputs', help='')
parser.add_argument('--exp_name', type=str, default='', help='')
parser.add_argument('--sencell_num', type=int, default=100, help='')
parser.add_argument('--cell_optim_epoch', type=int, default=15, help='')
parser.add_argument('--device_index', type=int, default=0, help='')
parser.add_argument('--retrain', action='store_true', default=False, help='')


if is_jupyter:
    args = parser.parse_args(args=[])
    args.exp_name = 's5'
    args.retrain = False
else:
    args = parser.parse_args()
args.is_jupyter = is_jupyter


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='# %Y-%m-%d %H:%M:%S')

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

# Part 1: load and process data
# cell_cluster_arr在画umap的时候用
if 's5' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data()
elif 'healthy' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_healthy()
elif 'disease1' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_disease1()
elif 'disease' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_disease()

# plots.umapPlot(adata.obsm['X_umap'],clusters=cell_cluster_arr,labels=celltype_names)

new_data, markers_index,\
    sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
        adata, cluster_cell_ls, cell_cluster_arr)

gene_cell = new_data.X.toarray().T
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]

print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

graph_nx = utils.build_graph_nx(
    gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names)
logger.info("Part 1, data loading and processing end!")

# Part 2: generate init embedding
device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
print('device:', device)
args.device = device

if args.retrain:
    gene_embed, cell_embed = reduction_AE(gene_cell, device)
    print(gene_embed.shape, cell_embed.shape)
    torch.save(gene_embed, os.path.join(
        args.output_dir, f'{args.exp_name}_gene.emb'))
    torch.save(cell_embed, os.path.join(
        args.output_dir, f'{args.exp_name}_cell.emb'))
else:
    gene_embed = torch.load(os.path.join(
        args.output_dir, f'{args.exp_name}_gene.emb'))
    cell_embed = torch.load(os.path.join(
        args.output_dir, f'{args.exp_name}_cell.emb'))

graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed)
logger.info("Part 2, AE end!")

# Part 3: train GAT
# graph_pyg=graph_pyg.to('cpu')

GAT_model = train_GAT(graph_nx, graph_pyg, args,
                      retrain=args.retrain, resampling=args.retrain)
logger.info("Part 3, training GAT end!")


all_gene_ls = []

list_sencell_cover = []
list_sengene_cover = []

cellmodel = Sencell().to(device)
optimizer = torch.optim.Adam(cellmodel.parameters(), lr=0.001,
                             weight_decay=1e-3)

all_marker_index = sen_gene_ls

iteration_results = []
for iteration in range(5):
    logger.info(f"iteration: {iteration}")
    sampled_graph, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict = sub_sampling_by_random(graph_nx,
                                                                                                               sen_gene_ls,
                                                                                                               nonsen_gene_ls,
                                                                                                               GAT_model,
                                                                                                               args,
                                                                                                               all_marker_index,
                                                                                                               n_gene=len(
                                                                                                                   all_marker_index),
                                                                                                               gene_rate=0.3, cell_rate=0.5,
                                                                                                               debug=False)
    old_sengene_indexs = all_marker_index
    for epoch in range(args.cell_optim_epoch):
        logger.info(f"epoch: {epoch}")
        old_sencell_dict = sencell_dict
        cellmodel, sencell_dict, nonsencell_dict = cell_optim(cellmodel, optimizer,
                                                              sencell_dict, nonsencell_dict, args,
                                                              train=True)
        # sampled_graph=update_cell_embeddings(sampled_graph,sencell_dict,nonsencell_dict)
        sencell_dict, nonsencell_dict, \
            sen_gene_indexs, nonsen_gene_indexs = identify_sengene_then_sencell(sampled_graph, GAT_model,
                                                                                sencell_dict, nonsencell_dict,
                                                                                cell_clusters,
                                                                                big_graph_index_dict,
                                                                                len(all_marker_index), args)

        ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)
        ratio_gene = utils.get_sengene_cover(
            old_sengene_indexs, sen_gene_indexs)
        old_sengene_indexs = sen_gene_indexs
        if ratio_cell == 1 and ratio_gene == 1:
            print("Get convergence!")
            break
    iteration_results.append([sen_gene_indexs, sencell_dict])

outputs_path = os.path.join(args.output_dir, f'{args.exp_name}_outputs.data')
print("Experiments saved!", outputs_path)
torch.save([sencell_dict, sen_gene_indexs], outputs_path)
