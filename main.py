import numpy as np
import seaborn as sns
import torch
from torch.optim.lr_scheduler import StepLR,ExponentialLR

import umap
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_networkx, from_networkx
import matplotlib

import utils
import plots
from model_AE import reduction_AE
from model_GAT import Encoder, SenGAE, train_GAT, train_GAT_new
from model_Sencell import Sencell
from sampling import sub_sampling_by_random,sub_sampling_by_random_v1
from model_Sencell import cell_optim, update_cell_embeddings
from sampling import identify_sengene_then_sencell,identify_sengene_then_sencell_v1

import logging
import os
import argparse
import random
import datetime
import scanpy as sp



is_jupyter = False
use_wandb=False

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}-{current_date.month:02d}-{current_date.day:02d}-{current_date.hour:02d}-{current_date.minute:02d}-{current_date.second:02d}"

# nohup python -u main.py --exp_name OSU_disease_batch0 --device_index 2 --batch_id 0 --retrain > OSU_disease_batch0.log 2>&1 &
# nohup python -u main.py --exp_name newfix --device_index 2 --retrain > ./log/newfix.log 2>&1 &

parser = argparse.ArgumentParser(description='Main program for sencells')

parser.add_argument('--output_dir', type=str, default='./outputs', help='')
parser.add_argument('--exp_name', type=str, default='', help='')
parser.add_argument('--device_index', type=int, default=0, help='')
parser.add_argument('--retrain', action='store_true', default=False, help='')
parser.add_argument('--timestamp', type=str,  default="", help='')

parser.add_argument('--gat_epoch', type=int, default=10, help='')
parser.add_argument('--sencell_num', type=int, default=300, help='')
parser.add_argument('--sengene_num', type=int, default=200, help='')
parser.add_argument('--sencell_epoch', type=int, default=40, help='')
parser.add_argument('--cell_optim_epoch', type=int, default=50, help='')

parser.add_argument('--batch_id', type=int, default=0, help='')

if is_jupyter:
    # jupyter 参数注入
    args = parser.parse_args(args=[])
    args.exp_name = 'newfix'
    args.output_dir=f'./outputs/{datestamp}-{args.exp_name}'
    args.device_index=2
    args.retrain = True
    args.gat_epoch=30
    args.sencell_num=100
else:
    args = parser.parse_args()
    
print(vars(args))

args.is_jupyter = is_jupyter
if args.retrain:
    args.output_dir=os.path.join(args.output_dir,f"{datestamp}-{args.exp_name}")
else:
    args.output_dir=f"./outputs/{args.timestamp}-{args.exp_name}/"   
    print("outdir:",args.output_dir)
# else:
#     args.output_dir=os.path.join("./outputs/23-11-28-21-45-fixbatch")
print("Outputs dir:",args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# set random seed
seed=42
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
# cell_cluster_arr在画umap的时候用
print("\n====== Part 1: load and process data ======")
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
    
    
# plots.umapPlot(adata.obsm['X_umap'],clusters=cell_cluster_arr,labels=celltype_names)

new_data, markers_index,\
    sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
        adata, cluster_cell_ls, cell_cluster_arr)
    
gene_cell = new_data.X.toarray().T
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]

print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

if args.retrain:
    graph_nx = utils.build_graph_nx(
        new_data,gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names)
else:
    graph_nx=torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))

# dgl_graph,adj_matrix=utils.build_ccc_graph(gene_cell,list(new_data.var.index))

dgl_graph,adj_matri=None,None

if args.retrain:
    print("Save graph_nx")
    torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
    # torch.save([dgl_graph,adj_matrix], os.path.join(args.output_dir, f'{args.exp_name}_graphdgl.data'))

logger.info("Part 1, data loading and processing end!")


# Part 2: generate init embedding
print("\n====== Part 2: generate init embedding ======")


use_autoencoder=False

device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
print('device:', device)
args.device = device
if use_autoencoder:
    if args.retrain:
        gene_embed, cell_embed = reduction_AE(gene_cell, device)
        print(gene_embed.shape, cell_embed.shape)
        torch.save(gene_embed, os.path.join(
            args.output_dir, f'{args.exp_name}_gene.emb'))
        torch.save(cell_embed, os.path.join(
            args.output_dir, f'{args.exp_name}_cell.emb'))
    else:
        print('skip training!')
        gene_embed = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_gene.emb'))
        cell_embed = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_cell.emb'))

    if args.retrain:
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))

    else:
        graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed)
        
else:
    # use scanpy generate low dim embeddings
    pass
    
    # graph_pyg = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
    

logger.info("Part 2, AE end!")

# Part 3: train GAT
# graph_pyg=graph_pyg.to('cpu')
print("\n====== Part 3: train GAT ======")
GAT_model = train_GAT(graph_nx, graph_pyg, args,
                      retrain=args.retrain, resampling=args.retrain,wandb=wandb)
logger.info("Part 3, training GAT end!")


print("\n====== Part 4: sencell optim ======")
all_gene_ls = []

cellmodel = Sencell().to(device)
all_marker_index = sen_gene_ls

iteration_results = []

ratio_cell_ls=[]
ratio_gene_ls=[]

sampled_graph, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict,embs = sub_sampling_by_random_v1(graph_nx,
                                                                                                                graph_pyg,
                                                                                                            sen_gene_ls,
                                                                                                            nonsen_gene_ls,
                                                                                                            GAT_model,
                                                                                                            args,
                                                                                                            all_marker_index,
                                                                                                            n_gene=len(
                                                                                                                all_marker_index),
                                                                                                            gene_rate=0.3, 
                                                                                                            cell_rate=0.5,
                                                                                                            debug=False)
old_sengene_indexs = all_marker_index
lr=0.01
optimizer = torch.optim.Adam(cellmodel.parameters(), lr=lr,
                        weight_decay=1e-3)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.85)


def check_heatmap(sencell_dict,adata,graph_nx,epoch):
    nonsenmarkers=utils.load_nonsenmarkers(adata)   
    senmarkers=utils.load_markers() 
    senmarkers=[j for i in senmarkers for j in i]  

    sencell_names=[]
    for key,value in sencell_dict.items():
        sencell_names.append(graph_nx.nodes[key]['name'])
    sen_adata=adata[adata.obs.index.isin(sencell_names)].copy()
     
    sp.pl.heatmap(
        sen_adata,
        var_names=nonsenmarkers,
        groupby='cell_type',  # change to your clustering key if different
        use_raw=False,  # set to True if you want to use raw data
        log=False,  # set to True if you want log-scaled color map
        dendrogram=False,
        cmap="gray_r",
        figsize=(10,20),
        save=f"{datestamp}-{args.exp_name}-{epoch}-nonsen.png"
    )
    
    sp.pl.heatmap(
        sen_adata,
        var_names=nonsenmarkers,
        groupby='cell_type',  # change to your clustering key if different
        use_raw=False,  # set to True if you want to use raw data
        log=False,  # set to True if you want log-scaled color map
        dendrogram=False,
        cmap="gray_r",
        figsize=(10,20),
        save=f"{datestamp}-{args.exp_name}-{epoch}-sen.png"
    )



for epoch in range(args.sencell_epoch):
    logger.info(f"epoch: {epoch}")
    old_sencell_dict = sencell_dict
    
    
    # print results
    for key,value in sencell_dict.items():
        print(graph_nx.nodes[key],value[1],celltype_names[value[1]])
    for key,value in sencell_dict.items():
        print(graph_nx.nodes[key]['name'])
        
    for gene in sen_gene_indexs:
        print(graph_nx.nodes[gene]['name'])
    
    
    check_heatmap(sencell_dict,adata,graph_nx,epoch)


    cellmodel, sencell_dict, nonsencell_dict = cell_optim(cellmodel, optimizer,
                                                            sencell_dict, nonsencell_dict,
                                                            dgl_graph,
                                                            args,
                                                            train=True,
                                                            wandb=wandb)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"lr": current_lr})
    
    print("Skip Update CCC graph")
    # dgl_graph,_=utils.update_dglgraph(cellmodel,embs,dgl_graph,args)
    # sampled_graph=update_cell_embeddings(sampled_graph,sencell_dict,nonsencell_dict)
    sencell_dict, nonsencell_dict, \
        sen_gene_indexs, nonsen_gene_indexs = identify_sengene_then_sencell_v1(sampled_graph, GAT_model,
                                                                            sencell_dict, nonsencell_dict,
                                                                            cell_clusters,
                                                                            big_graph_index_dict, args)

    ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)
    ratio_gene = utils.get_sengene_cover(
        old_sengene_indexs, sen_gene_indexs)
    
    wandb.log({"cell overlap": ratio_cell, "gene overlap": ratio_gene,"sencell_num":args.sencell_num,
               "level0":cellmodel.levels[0],"level1":cellmodel.levels[1],"level2":cellmodel.levels[2]
               })
    
    if len(ratio_cell_ls)>0 and ratio_cell==ratio_cell_ls[-1]:
        args.sencell_num=int(args.sencell_num*ratio_cell)
        
    ratio_cell_ls.append(ratio_cell)
    ratio_gene_ls.append(ratio_gene)
    
    old_sengene_indexs = sen_gene_indexs
    if ratio_cell == 1 and ratio_gene == 1:
        print("Get convergence!")
        break
    
    break


outputs_path = os.path.join(args.output_dir, f'{args.exp_name}_outputs.data')
print("Experiments saved!", outputs_path)
torch.save([old_sencell_dict, nonsencell_dict, sen_gene_indexs], outputs_path)

logger.info("Part 4, sencell optim end!")

# print results
for key,value in sencell_dict.items():
    print(graph_nx.nodes[key],value[1],celltype_names[value[1]])
for key,value in sencell_dict.items():
    print(graph_nx.nodes[key]['name'])
    
for gene in sen_gene_indexs:
    print(graph_nx.nodes[gene]['name'])
    
wandb.finish()