import scanpy as sp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as Graphdata
from torch_geometric.utils import to_undirected
import networkx as nx
from tabulate import tabulate


def load_data(path="./data/SCB01S5.h5ad"):
    # path:"/users/PCON0022/haocheng/Basu_lab/rmarkdown/SCB01S5.h5ad"
    adata = sp.read_h5ad(path)

    celltype_names = list(adata.obs['ann_level_3_pred'].value_counts().index)
    print("cluster 数量：", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['ann_level_3_pred']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_healthy(path="./data/combined_g8.h5ad"):
    adata = sp.read_h5ad(path)
    labels=pd.read_csv("./data/g8_query_emb.csv",index_col=0)
    adata.obs['ann_level_3_pred']=labels['ann_level_3_pred']

    adata = adata[adata.obs['disease'] == 'Healthy']

    
    # celltype_names = list(adata.obs['cell_type_seurat'].value_counts().index)
    celltype_names=list(adata.obs['ann_level_3_pred'].value_counts().index)

    print("cluster 数量：", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['ann_level_3_pred']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_disease(path="./data/combined_g8.h5ad"):
    adata = sp.read_h5ad(path)
    labels=pd.read_csv("./data/g8_query_emb.csv",index_col=0)
    adata.obs['ann_level_3_pred']=labels['ann_level_3_pred']

    adata = adata[adata.obs['disease'] != 'Healthy']

    # celltype_names = list(adata.obs['cell_type_seurat'].value_counts().index)
    celltype_names=list(adata.obs['ann_level_3_pred'].value_counts().index)
    print("cluster 数量：", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['ann_level_3_pred']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_disease1(path="./data/combined_g8.h5ad"):
    print('load data disease1!')
    adata = sp.read_h5ad(path)
    labels=pd.read_csv("./data/g8_query_emb.csv",index_col=0)
    adata.obs['ann_level_3_pred']=labels['ann_level_3_pred']

    adata=adata[(adata.obs['orig.ident']=='SCB01S2') | (adata.obs['orig.ident']=='SCB01S4')]

    # celltype_names = list(adata.obs['cell_type_seurat'].value_counts().index)
    celltype_names=list(adata.obs['ann_level_3_pred'].value_counts().index)
    print("cluster 数量：", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['ann_level_3_pred']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_markers():
    markers = pd.read_csv("combined_senescence_list.csv", header=None)
    # Series
    markers1 = list(markers[0][markers[0].notnull()])
    markers2 = list(markers[1][markers[1].notnull()])
    markers3 = list(markers[2][markers[2].notnull()])
    markers4 = list(markers[3][markers[3].notnull()])

    print('各marker list所包含的gene数：')
    print(tabulate([["Markers1", "Markers2", "Markers3", "Markers4"],
                    [len(markers1), len(markers2), len(markers3), len(markers4)]],
                   headers="firstrow"))

    # markers_ls
    return [markers1, markers2, markers3, markers4]


def get_highly_genes_old(adata):
    return list(adata.var[adata.var['vst.variable'] == True].index)


def get_highly_genes(adata):
    sp.pp.highly_variable_genes(adata, n_top_genes=2000)
    highly_genes = list(adata.var[adata.var['highly_variable'] == True].index)
    return highly_genes


def combine_genes(adata, markers_ls):
    markers_set = set([gene for markers in markers_ls for gene in markers])
    print('total marker genes: ', len(markers_set))

    highly_genes = get_highly_genes(adata)
    print('highly_genes num: ', len(highly_genes))

    # 去除其中的老化基因，然后再把老化基因append到最后
    highly_genes = sorted(list(set(highly_genes)-markers_set))
    print('After highly genes dropped duplicate: ', len(highly_genes))

    # 这里有要注意的地方，有可能老化基因在所有cell里面都是0表达
    # 所以这里要加一步去除0表达的基因
    sen_gene_ls = []
    cell_gene = adata.X.toarray()
    gene_names = list(adata.var.index)

    for gene in markers_set:
        if gene in gene_names:
            gene_index = gene_names.index(gene)
            if max(cell_gene[:, gene_index]) != 0:
                sen_gene_ls.append(gene)

    # highly gene也有可能出现全0的情况
    filtered_highly_genes = []
    for gene in highly_genes:
        if gene in gene_names:
            gene_index = gene_names.index(gene)
            if max(cell_gene[:, gene_index]) != 0:
                filtered_highly_genes.append(gene)

    if len(filtered_highly_genes) != len(highly_genes):
        print("highly genes里面有全0！！")

    # combine them all
    sen_gene_ls = sorted(sen_gene_ls)
    gene_names = filtered_highly_genes+sen_gene_ls
    print('Total gene num:', len(gene_names))

    # 选择adata的一个子集
    adata_gene_names = list(adata.var.index)
    gene_indexs = [adata_gene_names.index(name) for name in gene_names]
    new_data = adata[:, gene_indexs]

    assert gene_names[100] == new_data.var.index[100], "Bug!!!"
    # 得到每个marker list里面marker的index
    # markers_index有重叠
    markers_index = []
    for markers in markers_ls:
        indexs = []
        for marker in markers:
            if marker in gene_names:
                indexs.append(gene_names.index(marker))
        markers_index.append(indexs)
    # 得到markers_index，包含4个list，是marker在genes里面的index
    # 所有marker的index
    sen_gene_ls = [gene_names.index(i) for i in sen_gene_ls]
    nonsen_gene_ls = [gene_names.index(i) for i in filtered_highly_genes]
    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def process_data(adata, cluster_cell_ls, cell_cluster_arr):
    # step 1: load marker
    markers_ls = load_markers()
    # step 2: append markers to highly genes
    new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = combine_genes(
        adata, markers_ls)

    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def build_graph_nx(gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names):
    # build nx graph and pyg graph
    # step 1: 计算edge index
    g_index, c_index = np.nonzero(gene_cell)
    print('The number of edges:', len(g_index))
    # 加上偏移量作为cell的节点标号
    gene_num = gene_cell.shape[0]
    c_index += gene_num
    edge_index = torch.tensor([g_index, c_index], dtype=torch.long)

    # step 2: build nx graph, add attributes
    graph_nx = nx.Graph(edge_index.T.tolist())

    # 再加一个属性，是每个节点在大图上的index
    for i in range(gene_num):
        graph_nx.nodes[i]['type'] = 'g'
        graph_nx.nodes[i]['index'] = i
        graph_nx.nodes[i]['name'] = gene_names[i]
        graph_nx.nodes[i]['is_sen'] = i in sen_gene_ls

    for i in range(gene_cell.shape[1]):
        graph_nx.nodes[i+gene_num]['type'] = 'c'
        graph_nx.nodes[i+gene_num]['cluster'] = cell_cluster_arr[i]
        graph_nx.nodes[i+gene_num]['index'] = i+gene_num

    return graph_nx


def add_nx_embedding(graph_nx, gene_embed, cell_embed):
    for i in range(gene_embed.shape[0]):
        graph_nx.nodes[i]['emb'] = gene_embed[i].detach().cpu()

    for i in range(cell_embed.shape[0]):
        graph_nx.nodes[i+gene_embed.shape[0]
                       ]['emb'] = cell_embed[i].detach().cpu()

    return graph_nx


def build_graph_pyg(gene_cell, gene_embed, cell_embed):
    # build nx graph and pyg graph
    # step 1: 计算edge index
    g_index, c_index = np.nonzero(gene_cell)
    print('the number of edges:', len(g_index))
    # 加上偏移量作为cell的节点标号
    c_index += gene_cell.shape[0]
    edge_index = torch.tensor([g_index, c_index], dtype=torch.long)

    # 代表节点的类别
    y = [True]*gene_cell.shape[0]+[False]*gene_cell.shape[1]
    y = torch.tensor(y)

    # step 3: build pyg graph
    print('edge index: ', edge_index.shape)
    x = torch.cat([gene_embed, cell_embed]).detach()
    print('node feature: ', x.shape)

    edge_index = to_undirected(edge_index)
    graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)

    print('Pyg graph:', graph_pyg)
    print('graph.is_directed():', graph_pyg.is_directed())

    return graph_pyg


def get_sencell_cover(old_sencell_dict, sencell_dict):
    set1 = set(list(old_sencell_dict.keys()))
    set2 = set(list(sencell_dict.keys()))
    set3 = set1.intersection(set2)
    print('sencell cover:', len(set3)/len(set2))

    return len(set3)/len(set2)


def get_sengene_cover(old_sengene_ls, sengene_ls):
    set1 = set(old_sengene_ls)
    set2 = set(sengene_ls)
    set3 = set1.intersection(set2)
    print('sengene cover:', len(set3)/len(set2))

    return len(set3)/len(set2)


def save_objs(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("obj saved", path)


def load_objs(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
