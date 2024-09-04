import scanpy as sp
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
from torch_geometric.data import Data as Graphdata
from torch_geometric.utils import to_undirected
import networkx as nx
from tabulate import tabulate
import dgl
import scipy
from scipy import sparse as scsp


def load_data_combined1(path="./data/sub_combined.h5ad"):
    print("load_data_combined1 ...")
    adata = sp.read_h5ad(path)
    
    print(f"genes: {adata.shape[1]}, cells:{adata.shape[0]}")
    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=3)

    celltype_names=list(adata.obs['Rationale_based_annotation_update'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['Rationale_based_annotation_update']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_combined2(path="./data/combined_part2.h5ad"):
    print("load_data_combined2 ...")
    adata = sp.read_h5ad(path)
    
    print(f"genes: {adata.shape[1]}, cells:{adata.shape[0]}")

    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=3)

    celltype_names=list(adata.obs['Rationale_based_annotation_update'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['Rationale_based_annotation_update']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names

def load_data_combined3(path="./data/combined_part3.h5ad"):
    print("load_data_combined3 ...")
    adata = sp.read_h5ad(path)
    
    print(f"genes: {adata.shape[1]}, cells:{adata.shape[0]}")

    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=3)

    celltype_names=list(adata.obs['Rationale_based_annotation_update'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['Rationale_based_annotation_update']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names



def load_data_fixRNA(path="/bmbl_data/chenghao/sencell/fixRNA_annotated.h5ad"):
    print("load_data_fixRNA ...")
    adata = sp.read_h5ad(path)
    
    scipy_matrix=scsp.csr_matrix(adata.layers['raw_counts'])
    adata.X=scipy_matrix
    
    # filter less than 100 cells
    # Step 1: Count the number of cells in each cell type
    cell_type_counts = adata.obs['hlca_update'].value_counts()

    # Step 2: Identify cell types with 100 or more cells
    cell_types_to_keep = cell_type_counts[cell_type_counts >= 100].index
    # Step 3: Filter the adata object to keep only the desired cell types
    adata = adata[adata.obs['hlca_update'].isin(cell_types_to_keep)]

    print("After filter ...")
    
    # adata = adata[adata.obs['disease_update'] == 'Healthy']
    celltype_names=list(adata.obs['hlca_update'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['hlca_update']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_mouse(path="/bmbl_data/chenghao/sencell/data/mouse/high/data.h5ad"):
    print("load_data_mouse ...")
    adata = sp.read_h5ad(path)
    
    celltype_names=list(adata.obs['cell_type'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['cell_type']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_newfix(path="/bmbl_data/chenghao/sencell/fixed_data_0525.h5ad"):
    # "/bmbl_data/chenghao/sencell/fixed_data_0520.h5ad", 7w cells
    print("load_data_newfix ...")
    adata = sp.read_h5ad(path)
    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)

    print(f'Number of cells: {adata.shape[0]}\nNumber of genes: {adata.shape[1]}')
    
    celltype_names=list(adata.obs['clusters'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['clusters']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names



def load_data2(path="/bmbl_data/chenghao/U54/deepSAS_data2.h5ad"):
    # "/bmbl_data/chenghao/sencell/fixed_data_0520.h5ad", 7w cells
    print("load_data2 ...")
    adata = sp.read_h5ad(path)
    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)

    print(f'Number of cells: {adata.shape[0]}\nNumber of genes: {adata.shape[1]}')
    
    celltype_names=list(adata.obs['ct'].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, 存储每个cluster里面包含的cell index
    cluster_cell_ls = []
    # 1d-array，存储每个cell的cluster index
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # 所有cell的index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs['ct']
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names




def get_ccc_markers():
    ligand_receptor_dict = {
        "IL6": ["IL6ST", "IL6R", "HRH1", "F3"],
        "CXCL10": ["DPP4", "CCR3", "GRM7", "ADRA2A", "SDC4", "CXCR3", "MTNR1A"],
        "IL1B": ["ADRB2", "SIGIRR", "IL1RAP", "IL1R2", "IL1R1"],
        "CCL2": ["ACKR1", "CCR2", "ACKR4", "CCR10", "CCR3", "CCR5", "CCR1", "CCR4", "ACKR2"],
        "CCL5": ["SDC1", "GPR75", "ADRA2A", "CCR3", "CCRL2", "CCR5", "GRM7", "SDC4", "ACKR1", "MTNR1A", "CCR1", "CCR4", "CXCR3", "ACKR4", "ACKR2"],
        "HMGB1": ["TLR4", "TLR9", "AGER", "CXCR4", "SDC1", "THBD", "TLR2", "CD163"],
        "TNF": ["PTPRS", "CELSR2", "RIPK1", "FLT4", "TRAF2", "FAS", "ICOS", "NOTCH1", "TNFRSF1B", "TRADD", "TRPM2", "FFAR2", "VSIR", "TNFRSF21", "TNFRSF1A"],
        "SERPINE1": ["LRP1", "ITGAV", "PLAUR", "ITGB5", "LRP2"],
    }
    
    new_gene_set=set()
    for key,value in ligand_receptor_dict.items():
        new_gene_set.add(key)
        new_gene_set.update(value)
    return ligand_receptor_dict,new_gene_set

def get_cellcyle_markers():
    return ["CDKN1A","CDKN2A","TP53","GADD45A","IGFBP7","SERPINE1","GLB1","IL6","IL8","MMP1","MMP3"]


def load_markers(args):
    markers = pd.read_csv("senescence_marker_list.csv")
    # Series
    markers_ls=[]
    for col_name, data in markers.items():
        markers_ls.append(list(data[data.notnull()]))
    
    markers5 = list(get_ccc_markers()[1])
    markers_ls.append(markers5)
    
    markers6 = get_cellcyle_markers()
    markers_ls.append(markers6)

    print('各marker list所包含的gene数：')
    # print(tabulate([["SenMayo", "FRIDMAN", "CellAge", "GO","L-R Markers","Cell Cycle Markers"],
    #                 [len(markers_ls[0]), len(markers_ls[1]), len(markers_ls[2]), len(markers_ls[3]),len(markers_ls[4]),len(markers_ls[5])]],
    #                headers="firstrow"))
    
    # remove go list and L-R markers
    print(tabulate([["SenMayo", "FRIDMAN", "CellAge", "Cell Cycle Markers"],
                    [len(markers_ls[0]), len(markers_ls[1]), len(markers_ls[2]),len(markers_ls[5])]],
                headers="firstrow"))
    
    markers_ls=[markers_ls[0],markers_ls[1],markers_ls[2],markers_ls[5]]
    
    # senmayo or fridman or cellage or senmayo+cellage 
    # or senmayo+fridman or senmayo+fridman+cellage or full
    if args.gene_set == 'full':
        return markers_ls
    elif args.gene_set == 'senmayo':
        return [markers_ls[0]]
    elif args.gene_set == 'fridman':
        return [markers_ls[1]]
    elif args.gene_set == 'cellage':
        return [markers_ls[2]]
    elif args.gene_set == 'senmayo+cellage':
        return [markers_ls[0],markers_ls[2]]
    elif args.gene_set == 'senmayo+fridman':
        return [markers_ls[0],markers_ls[1]]
    elif args.gene_set == 'senmayo+fridman+cellage':
        return [markers_ls[0],markers_ls[1],markers_ls[2]]
    # markers_ls
    return markers_ls


def load_nonsenmarkers(adata):
    nonsen_markers=["CCNB1","CDK1","CDC25C","WEE1","CHK1","CCNA2","PCNA","MCM","RPA",
                    "DHFR","CCNB1","AURKA","AURKB","PLK1","H3S10ph","BUB1"]
    nonsen_markers=[gene for gene in nonsen_markers if gene in adata.var_names]
    return nonsen_markers

def get_highly_genes_old(adata):
    return list(adata.var[adata.var['vst.variable'] == True].index)


def get_highly_genes(adata,n_genes):
    new_data=adata.copy()
    sp.pp.normalize_total(new_data, target_sum=1e4)
    sp.pp.log1p(new_data)
    
    sp.pp.highly_variable_genes(new_data, n_top_genes=n_genes)
    highly_genes = list(new_data.var[new_data.var['highly_variable'] == True].index)
    return highly_genes


def combine_genes(adata, markers_ls, args):
    markers_set = set([gene for markers in markers_ls for gene in markers])
    print('Total marker genes: ', len(markers_set))

    # 使用highly_genes
    # adata.X = scipy.sparse.csr_matrix(adata.X)
    if args.n_genes=='full':
        print("use all genes!")
        highly_genes= list(adata.var.index)
    else:
        print(f"use {args.n_genes} highly genes!")
        highly_genes = get_highly_genes(adata,int(args.n_genes))
        print('Highly genes num: ', len(highly_genes))
    
    # 使用全部基因
    # 全部基因会导致整个图太大，会大幅降低运行效率
    # highly_genes=list(adata.var.index)

    # 去除其中的老化基因，然后再把老化基因append到最后
    highly_genes = sorted(list(set(highly_genes)-markers_set))
    print('After genes dropped duplicated sengenes: ', len(highly_genes))

    # 这里有要注意的地方，有可能老化基因在所有cell里面都是0表达
    # 所以这里要加一步去除0表达的基因
    sen_gene_ls = []
    # cell_gene = adata.X.toarray()
    if scsp.issparse(adata.X):
        cell_gene = adata.X.toarray()
    else:
        cell_gene = adata.X
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
        print("Highly genes里面有全0！！")

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


def process_data(adata, cluster_cell_ls, cell_cluster_arr,args):
    # step 1: load marker
    markers_ls = load_markers(args)
    # step 2: append markers to highly genes
    new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = combine_genes(
        adata, markers_ls,args)

    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def build_graph_nx(adata,gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names,args):
    # build nx graph and pyg graph
    # step 1: 计算edge index
    g_index, c_index = np.nonzero(gene_cell)
    print('Cell-gene graph, the number of edges:', len(g_index))
    # 加上偏移量作为cell的节点标号
    gene_num = gene_cell.shape[0]
    c_index += gene_num
    if args.ccc=='type1':
        print("add cell-cell edges with weights in 0 and 1...")
        adj_matrix,_=build_ccc_graph(gene_cell,gene_names)
        index1, index2 = np.nonzero(adj_matrix)
        print('CCC graph, the number of edges:', len(index1))
        index1+=gene_num
        index2+=gene_num
        new_g_index=np.concatenate([g_index,index1])
        new_c_index=np.concatenate([c_index,index2])
        edge_index = torch.tensor(np.array([new_g_index, new_c_index]), dtype=torch.long)
        ccc_matrix=None
    elif args.ccc=='type2':
        print("add cell-cell edges with weights in 0 to 1...")
        adj_matrix,ccc_matrix=build_ccc_graph(gene_cell,gene_names)
        index1, index2 = np.nonzero(adj_matrix)
        print('CCC graph, the number of edges:', len(index1))
        index1+=gene_num
        index2+=gene_num
        new_g_index=np.concatenate([g_index,index1])
        new_c_index=np.concatenate([c_index,index2])
        edge_index = torch.tensor(np.array([new_g_index, new_c_index]), dtype=torch.long)
    else:
        print("no cell-cell edges ...")
        edge_index = torch.tensor(np.array([g_index, c_index]), dtype=torch.long)
        ccc_matrix=None
        
    # step 2: build nx graph, add attributes
    graph_nx = nx.Graph(edge_index.T.tolist())

    # 再加一个属性，是每个节点在大图上的index
    for i in range(gene_num):
        graph_nx.nodes[i]['type'] = 'g'
        graph_nx.nodes[i]['index'] = i
        graph_nx.nodes[i]['name'] = gene_names[i]
        graph_nx.nodes[i]['is_sen'] = i in sen_gene_ls

    cell_names=list(adata.obs.index)
    for i in range(gene_cell.shape[1]):
        graph_nx.nodes[i+gene_num]['type'] = 'c'
        graph_nx.nodes[i+gene_num]['cluster'] = cell_cluster_arr[i]
        graph_nx.nodes[i+gene_num]['index'] = i+gene_num
        graph_nx.nodes[i+gene_num]['name'] = cell_names[i]

    return graph_nx,edge_index,ccc_matrix

def add_nx_embedding(graph_nx, gene_embed, cell_embed):
    for i in range(gene_embed.shape[0]):
        graph_nx.nodes[i]['emb'] = gene_embed[i].detach().cpu()

    for i in range(cell_embed.shape[0]):
        graph_nx.nodes[i+gene_embed.shape[0]
                       ]['emb'] = cell_embed[i].detach().cpu()

    return graph_nx


def build_graph_pyg(gene_cell, gene_embed, cell_embed,edge_indexs,ccc_matrix):
    print("build graph pyg")
    # 代表节点的类别
    y = [True]*gene_cell.shape[0]+[False]*gene_cell.shape[1]
    y = torch.tensor(y)

    print('edge index: ', edge_indexs.shape)
    x = torch.cat([gene_embed, cell_embed]).detach()
    print('node feature: ', x.shape)

    if ccc_matrix is None:
        print('build graph pyg without edge weights ... ')
        edge_index = to_undirected(edge_indexs)
        graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)
    else:
        print('build graph pyg with edge weights ... ')
        flatten_edge_features=ccc_matrix[ccc_matrix!=0]
        min_val = np.min(flatten_edge_features)
        max_val = np.max(flatten_edge_features)
        normalized_array = (flatten_edge_features - min_val) / (max_val - min_val)
        
        edge_attr=np.concatenate([np.ones(edge_indexs.shape[1]-len(normalized_array)),normalized_array])
        undirected_edge_index, undirected_edge_attr = to_undirected(edge_indexs, 
                                                            edge_attr=torch.tensor(edge_attr),reduce='mean')
        
        graph_pyg = Graphdata(x=x, edge_index=undirected_edge_index,edge_attr=undirected_edge_attr, y=y)

    print('Pyg graph:', graph_pyg)
    print('graph.is_directed():', graph_pyg.is_directed())

    return graph_pyg


def build_ccc_matrix(expression_matrix,gene_names):
    # cell x gene
    ccc_matrix=None
    ligand_receptor_dict=get_ccc_markers()[0]
    
    for ligand in ligand_receptor_dict:
        if ligand not in gene_names:
            continue
        ligand_exp=expression_matrix[:,gene_names.index(ligand)].reshape(-1,1)
        receptor_indexs=[]
        for receptor in ligand_receptor_dict[ligand]:
            if receptor in gene_names:
                receptor_indexs.append(gene_names.index(receptor))
        receptor_exp=expression_matrix[:,receptor_indexs]
        receptor_exp=np.sum(receptor_exp, axis=1).reshape(1,-1)
        result=ligand_exp*receptor_exp
        if ccc_matrix is None:
            ccc_matrix=result
        else:
            ccc_matrix=ccc_matrix+result
            
    return ccc_matrix

def convert_to_adj(ccc_matrix):
    # p=exp(-1/x)
    masked_result = ma.masked_where(ccc_matrix == 0, ccc_matrix)
    result_transformed = np.exp(-1 / masked_result)
    result_transformed = result_transformed.filled(0)
    
    symmetric_result = 0.5 * (result_transformed + result_transformed.T)
    
    mask = symmetric_result >= 0.8
    result_transformed_masked = np.where(mask, symmetric_result, 0)

    return result_transformed_masked

def convert_to_adj_v1(ccc_matrix,t=0.8):
    print("convert_to_adj_v1")
    n=ccc_matrix.shape[0]
    # Parameters for your equation
    w = 1.0  # replace with the actual value
    b = 0.0  # replace with the actual value

    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((n, n))

    # Iterate over each pair of nodes
    for i in range(n):
        for j in range(n):
            # Calculate the norm of the difference between the embeddings
            diff_norm = np.linalg.norm(ccc_matrix[i] - ccc_matrix[j])

            # Use your equation to calculate the edge weight
            edge_weight = 1 / (1 + np.exp(w * diff_norm**2 + b))

            # Store the result in the adjacency matrix
            adj_matrix[i, j] = edge_weight
    
    mask = adj_matrix >= t
    result_transformed_masked = np.where(mask, adj_matrix, 0)
    return result_transformed_masked
    
    
import numpy as np
from numba import jit

@jit(nopython=True)
def compute_adj_matrix(ccc_matrix, w=1.0, b=0.0):
    n = ccc_matrix.shape[0]
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff_norm = np.linalg.norm(ccc_matrix[i] - ccc_matrix[j])
            adj_matrix[i, j] = 1 / (1 + np.exp(w * diff_norm**2 + b))
    
    return adj_matrix

def convert_to_adj_v2(ccc_matrix, t=0.8):
    print("convert_to_adj_v2")
    adj_matrix = compute_adj_matrix(ccc_matrix)
    result_transformed_masked = np.where(adj_matrix >= t, adj_matrix, 0)
    return result_transformed_masked
    


def positional_encoding(g, pos_enc_dim=32):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    print("Caculate pos encoding ...")
    adjacency_matrix = g.adjacency_matrix().to_dense()
    # convert to scipy sparse matrix
    A = scsp.csr_matrix(adjacency_matrix.numpy())
    N = scsp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = scsp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g


import scipy.sparse.linalg as lg

def positional_encoding_v1(g, pos_enc_dim=32):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    print("Calculate pos encoding ...")
    adjacency_matrix = g.adjacency_matrix().to_dense()
    # Convert to scipy sparse matrix
    A = scsp.csr_matrix(adjacency_matrix.numpy())
    N = scsp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = scsp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy for sparse matrices
    EigVal, EigVec = lg.eigsh(L, k=pos_enc_dim+1, which='SM')  # smallest eigenvalues
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:]).float() 
    
    return g


def build_ccc_graph(gene_cell,gene_names):
    ccc_matrix=build_ccc_matrix(gene_cell.T,gene_names)
    adj_matrix=convert_to_adj(ccc_matrix)

    ccc_matrix=ccc_matrix*adj_matrix
    return adj_matrix,ccc_matrix


def update_dglgraph(cellmodel,embs,dgl_graph,args):
    cell_embs=embs[args.gene_num:]
    pos_embs=[]
    for i in range(args.cell_num):
        pos_embs.append(dgl_graph.nodes[i].data['pos_enc'].reshape(1,-1))

    pos_embs=torch.cat(pos_embs)
    new_emb=torch.cat([cell_embs,pos_embs],axis=1)
    result_emb=cellmodel.get_embeddings(new_emb,args.device).detach().cpu().numpy()

    adj_matrix=convert_to_adj_v2(result_emb,t=0.4)
    
    index1, index2 = np.nonzero(adj_matrix)
    print('the number of edges:', len(index1))

    dgl_graph = dgl.graph((index1, index2),num_nodes=args.cell_num)
    
    dgl_graph=positional_encoding(dgl_graph)
    return dgl_graph,adj_matrix


def get_sencell_cover(old_sencell_dict, sencell_dict):
    set1 = set(list(old_sencell_dict.keys()))
    set2 = set(list(sencell_dict.keys()))
    set3 = set1.intersection(set2)
    print('sencell cover:', len(set3)/len(set2))

    return len(set3)/len(set2)

def get_sencell_intersection(old_sencell_dict, sencell_dict):
    set1 = set(list(old_sencell_dict.keys()))
    set2 = set(list(sencell_dict.keys()))
    set3 = set1.intersection(set2)
    new_sencell_dict={}
    for i in set3:
        new_sencell_dict[i]=sencell_dict[i]
    print('length of new sencell_dict: ', len(new_sencell_dict))
    return new_sencell_dict

def get_sengene_cover(old_sengene_ls, sengene_ls):
    if isinstance(old_sengene_ls, torch.Tensor):
        old_sengene_ls = old_sengene_ls.tolist()
    if isinstance(sengene_ls, torch.Tensor):
        sengene_ls= sengene_ls.tolist()
    
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
    
    
def caculate_GSEA(adata,args,use_onemarker=False,one_marker=None):
    # p16: CDKN2A, use_onemarker= True, one_marker="CDKN2A"
    # p21: CDKN1A
    import pandas as pd
    import gseapy as gp
    
    gene_expression_df = pd.DataFrame(adata.X.T, index=adata.var.index, columns=adata.obs.index)
    if use_onemarker:
        all_marker_genes=[one_marker]
    else:
        all_marker_genes=load_markers(args)
        all_marker_genes=list(set([j for i in all_marker_genes for j in i]))

    # Prepare gene sets as a dictionary
    gene_sets = {'GeneSet1': all_marker_genes}

    # Run ssGSEA
    ssgsea_results = gp.ssgsea(data=gene_expression_df, gene_sets=gene_sets, sample_norm_method='rank', outdir=None)

    # Extract enrichment scores
    enrichment_scores = ssgsea_results.res2d

    return enrichment_scores