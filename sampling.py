# Import necessary libraries and modules
from torch_geometric.utils import subgraph
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data as Graphdata
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
import random
from linetimer import CodeTimer

# Setup logging for debugging and tracking execution details
logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='# %Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()

# Function to convert NetworkX graph to PyTorch Geometric graph
def convertNxtoPyg(graph_nx, is_clusters=False, is_big_graph_index=False):
    """
    Converts a NetworkX graph to a PyTorch Geometric graph.
    Allows for tracking node clusters and original indices in a large graph.
    - graph_nx: The NetworkX graph to convert.
    - is_clusters: Boolean to determine if cluster information is included.
    - is_big_graph_index: Boolean to keep track of the original node indices in the large graph.
    """
    # Initialization of dictionaries for node reindexing
    relabel_dict = {} # Map original index to new index
    big_graph_index_dict = {} # Map new index back to original index
    for i in graph_nx.nodes:
        new_index = len(relabel_dict)
        relabel_dict[i] = new_index
        big_graph_index_dict[new_index] = i
    relabeled_nx = nx.relabel_nodes(graph_nx, relabel_dict)

    # Extract node features and types
    emb = [] # List to store node embeddings
    types = [] # List to store node types (gene or cell)
    cell_clusters_dict = {} # Dictionary to store cluster ids of cells
    for i in range(len(relabeled_nx.nodes)):
        emb.append(relabeled_nx.nodes[i]['emb'].view(1, -1))
        types.append(relabeled_nx.nodes[i]['type'] == 'g')
        if is_clusters and relabeled_nx.nodes[i]['type'] == 'c':
            cell_clusters_dict[i] = relabeled_nx.nodes[i]['cluster']
    emb = torch.cat(emb)

    # Convert edges to an undirected format suitable for PyTorch Geometric
    edge_index = torch.tensor(list(relabeled_nx.edges)).T
    edge_index = to_undirected(edge_index)

    # Create the PyTorch Geometric graph data object
    result = Graphdata(x=emb, edge_index=edge_index, y=torch.tensor(types))

    # Depending on flags, return additional mapping dictionaries
    if is_big_graph_index and is_clusters:
        return result, cell_clusters_dict, big_graph_index_dict
    if is_clusters:
        return result, cell_clusters_dict
    return result

# Function to perform subgraph sampling using GAT models and various sampling strategies
def sub_sampling_GAT(graph_nx, graph, gene_num=2245, cell_num=10558,
                     gene_rate=0.3, cell_rate=0.5,
                     debug=False):
    """
    Samples a subgraph based on specified rates for genes and cells.
    - graph_nx: The full NetworkX graph from which to sample.
    - graph: The graph object (could be for different uses, placeholder in this function).
    - gene_num, cell_num: Total numbers of genes and cells in the full graph.
    - gene_rate, cell_rate: Rates at which to sample genes and cells.
    - debug: Boolean to activate debugging outputs.
    """
    # Calculate the number of genes and cells to sample
    n_gene = int(gene_num * gene_rate)
    n_cell = int(cell_num * cell_rate)

    # Randomly select indices for genes and cells
    gene_indexs = np.random.choice(range(gene_num), n_gene, replace=False)
    cell_indexs = np.random.choice(range(gene_num, cell_num + gene_num), n_cell, replace=False)

    # Initialize sets to collect new indices for genes and cells based on their neighbors
    new_cell_indexs = set()
    new_gene_indexs = set()

    # Collect neighbors for each selected gene and cell
    for i, j in zip(gene_indexs, cell_indexs):
        new_cell_indexs.update(list(graph_nx[i]))
        new_gene_indexs.update(list(graph_nx[j]))

    # Subsample from the neighbors to get the final sets of indices
    new_gene_indexs = np.random.choice(list(new_gene_indexs), int(len(new_gene_indexs) * 0.4), replace=False)
    new_cell_indexs = np.random.choice(list(new_cell_indexs), int(len(new_cell_indexs) * 0.1), replace=False)

    # Combine gene and cell indices to form the final node set for the subgraph
    node_set = np.concatenate([new_gene_indexs, new_cell_indexs])

    # Filter function to select nodes for the subgraph
    def filter_node(node):
        return node in node_set

    # Create the subgraph and convert to PyTorch Geometric format
    view = nx.subgraph_view(graph_nx, filter_node=filter_node)
    result = convertNxtoPyg(view)
    if debug:
        return result, new_gene_indexs, new_cell_indexs
    else:
        return result

# Additional functions follow a similar pattern, adapting and extending the core functionality demonstrated above.


# Function to sample subgraph based on a set of marker genes
def sub_sampling_by_genes(graph_nx, marker_index):
    """
    Samples a subgraph based on a set of marker genes and their first-order neighbors.
    - graph_nx: The full NetworkX graph from which to sample.
    - marker_index: Indices of marker genes used to sample the subgraph.
    """
    # Collect all first-order neighbors for each marker gene
    cell_set = set()
    for marker in marker_index:
        cell_set.update(list(graph_nx[marker]))
    node_set = np.concatenate([marker_index, list(cell_set)])

    # Debug output
    print('subgraph total node num:', node_set.shape)

    # Filter function to select nodes for the subgraph
    def filter_node(node):
        return node in node_set

    # Create the subgraph view and convert to PyTorch Geometric format
    view = nx.subgraph_view(graph_nx, filter_node=filter_node)
    result, cell_clusters, big_graph_index_dict = convertNxtoPyg(view, is_clusters=True, is_big_graph_index=True)
    return result, cell_clusters, big_graph_index_dict

# Function to identify senescent cells using graph attention network outputs
def identify_sencell_marker_graph(sampled_graph, model, cell_clusters, big_graph_index_dict, device, ratio=0.1, plot=False):
    """
    Identifies senescent cells in a graph based on model attention outputs.
    - sampled_graph: The sampled PyTorch Geometric graph to analyze.
    - model: The GAT model used to analyze the graph.
    - cell_clusters: Dictionary mapping cell indices to their cluster ids.
    - big_graph_index_dict: Dictionary mapping sampled graph indices to original graph indices.
    - device: Computation device (e.g., 'cuda' or 'cpu').
    - ratio: The threshold ratio to consider a cell as senescent.
    - plot: Boolean to activate plotting of results for visual analysis.
    """
    # Prepare the model and graph for processing
    model.eval()
    model.to(device)
    sampled_graph.to(device)

    # Compute embeddings and attention weights using the model
    z, (edge, edge_att) = model.encoder.get_att(sampled_graph)
    att = edge_att.T[0].detach().cpu()
    z = z.detach().cpu()

    # Process attention weights to identify senescent cells
    cell_index = torch.arange(len(sampled_graph.y))[torch.bitwise_not(sampled_graph.y)]
    cell_att = []
    for cell in cell_index:
        cell_att.append(att[edge[0] == cell].sum())
    cell_att = torch.tensor(cell_att)

    # Sort cells based on their computed attention and identify the top senescent cells
    sorted_index = torch.argsort(cell_att, descending=True)
    sencell_num = int(len(cell_index) * ratio)  # Number of senescent cells based on ratio
    if plot:
        sns.displot(cell_att)
        plt.ylim(0, 20)
        sns.displot(cell_att[sorted_index][:sencell_num])
    print('sencell_num:', sencell_num)

    sencell_index = cell_index[sorted_index][:sencell_num]
    nonsencell_index = cell_index[sorted_index][sencell_num:]

    # Map identified cells back to their clusters
    sencell_cluster = [cell_clusters[int(i)] for i in sencell_index]
    nonsencell_cluster = [cell_clusters[int(i)] for i in nonsencell_index]

    # Output results
    sencell_dict, nonsencell_dict = get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)
    return sencell_dict, nonsencell_dict
# Function to select cells based on gene indices using attention scores
def select_cell_by_genes(sampled_graph, sen_gene_indexs, nonsen_gene_indexs, cell_clusters, big_graph_index_dict, model_GAT, device):
    """
    Identifies senescent and nonsenescent cells in a graph based on gene attention scores.
    - sampled_graph: The PyTorch Geometric graph containing the cells and genes.
    - sen_gene_indexs: Indices of genes associated with senescence.
    - nonsen_gene_indexs: Indices of genes not associated with senescence.
    - cell_clusters: Dictionary mapping cell indices to their cluster ids.
    - big_graph_index_dict: Dictionary mapping sampled graph indices to original graph indices.
    - model_GAT: Graph Attention Network model to analyze the graph.
    - device: Computation device (e.g., 'cuda' or 'cpu').
    """
    # Prepare model and data for analysis
    model_GAT.eval()
    model_GAT.to(device)
    sampled_graph.to(device)

    # Calculate attention scores using the GAT model
    z, (edge, edge_att) = model_GAT.encoder.get_att(sampled_graph)
    att = edge_att.T[0].detach().cpu()
    z = z.detach().cpu()

    # Initialize dictionary to hold cell scores
    cell_score_dict = {}
    cell_indexs = torch.arange(len(sampled_graph.y))[torch.bitwise_not(sampled_graph.y)]
    for cell_index in cell_indexs:
        # Identify linked genes and their attention scores
        linked_genes = edge[1][edge[0] == cell_index]
        linked_atts = att[edge[0] == cell_index]
        sen_score = 0
        nonsen_score = 0
        for gene_index, gene_att in zip(linked_genes, linked_atts):
            if big_graph_index_dict[int(gene_index)] in sen_gene_indexs:
                sen_score += gene_att
            elif big_graph_index_dict[int(gene_index)] in nonsen_gene_indexs:
                nonsen_score += gene_att
        cell_score_dict[cell_index] = [sen_score, nonsen_score]

    # Process results to distinguish senescent from nonsenescent cells
    sorted_sen = sorted(cell_score_dict.items(), key=lambda item: -item[1][0])
    sencell_index = [idx for idx, _ in sorted_sen[:100]]  # Top 100 senescent cells
    sorted_nonsen = sorted(cell_score_dict.items(), key=lambda item: -item[1][1])
    nonsencell_index = [idx for idx, _ in sorted_nonsen[:900]]  # Top 900 nonsenescent cells

    # Map cell indices to their clusters
    sencell_cluster = [cell_clusters[int(i)] for i in sencell_index]
    nonsencell_cluster = [cell_clusters[int(i)] for i in nonsencell_index]

    # Prepare final output
    sencell_dict, nonsencell_dict = get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)
    return sampled_graph, sencell_dict, nonsencell_dict

# Function to calculate attention weights
def calculateAtt(model_GAT, graph_pyg, device):
    """
    Computes attention weights for each edge in a graph using a GAT model.
    - model_GAT: The GAT model used for computing the attention.
    - graph_pyg: The graph data in PyTorch Geometric format.
    - device: Computation device (e.g., 'cuda' or 'cpu').
    """
    # Setup model on the appropriate device
    model_GAT.eval()
    model_GAT.to(device)
    graph_pyg.to(device)

    # Compute the attention weights and extract edge and attention tensors
    z, (edge, edge_att) = model_GAT.encoder.get_att(graph_pyg)
    att = edge_att.T[0].detach().cpu()
    z = z.detach().cpu()
    edge = edge.cpu()
    return z, edge, att
# Function to get cell scores for senescence based on gene attention
def get_cell_scores(cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs):
    """
    Computes cell scores based on attention weights to genes categorized as senescent or nonsenescent.
    - cell_indexs: List of cell indices in the sampled graph.
    - edge: Tensor representing connections (edges) in the graph.
    - att: Tensor of attention weights corresponding to edges.
    - big_graph_index_dict: Mapping of sampled graph indices to original indices.
    - sen_gene_indexs: Set of indices representing senescent genes.
    - nonsen_gene_indexs: Set of indices for nonsenescent genes.
    """
    cell_score_dict = {}
    sen_gene_indexs_set = set(sen_gene_indexs)
    nonsen_gene_indexs_set = set(nonsen_gene_indexs)

    for cell_index in cell_indexs:
        linked_genes = edge[1][edge[0] == cell_index]
        linked_atts = att[edge[0] == cell_index]
        sen_score = 0
        nonsen_score = 0
        for gene_index, gene_att in zip(linked_genes, linked_atts):
            if big_graph_index_dict[int(gene_index)] in sen_gene_indexs_set:
                sen_score += gene_att
            elif big_graph_index_dict[int(gene_index)] in nonsen_gene_indexs_set:
                nonsen_score += gene_att
        cell_score_dict[cell_index] = [float(sen_score), float(nonsen_score)]

    return cell_score_dict

# Parallel version of get_cell_scores using multiprocessing to handle large data sets
def get_cell_scores_par1(cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs):
    """
    Parallel version of the get_cell_scores function using multiprocessing to enhance performance.
    - cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs: Same as get_cell_scores function.
    """
    from multiprocessing import Pool
    pool = Pool()
    results = []
    cell_score_dict = {}
    sen_gene_indexs_set = set(sen_gene_indexs)
    nonsen_gene_indexs_set = set(nonsen_gene_indexs)

    for cell_index in cell_indexs:
        linked_genes = edge[1][edge[0] == cell_index]
        linked_atts = att[edge[0] == cell_index]
        parms = (cell_index, linked_genes, linked_atts, big_graph_index_dict, sen_gene_indexs_set, nonsen_gene_indexs_set,)
        results.append(pool.apply_async(sub_get_cell_scores_par, parms))
    pool.close()
    pool.join()

    for res in results:
        cell_score_dict[res.get()[0]] = [float(res.get()[1]), float(res.get()[2])]

    return cell_score_dict

# Threaded version of get_cell_scores for possibly better performance in some environments
def get_cell_scores_par2(cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs):
    """
    Parallel version using threads instead of processes to potentially reduce overhead and improve speed.
    - cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs: Same as get_cell_scores function.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
    cell_score_dict = {}
    sen_gene_indexs_set = set(sen_gene_indexs)
    nonsen_gene_indexs_set = set(nonsen_gene_indexs)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(sub_get_cell_scores_par1, cell_index, edge[1][edge[0] == cell_index],
                                   att[edge[0] == cell_index], big_graph_index_dict, sen_gene_indexs_set, nonsen_gene_indexs_set)
                   for cell_index in cell_indexs]
        for future in futures:
            result = future.result()
            cell_score_dict[result[0]] = [float(result[1]), float(result[2])]

    return cell_score_dict

# Helper function used in parallel processing to calculate scores for individual cells
def sub_get_cell_scores_par(cell_index, linked_genes, linked_atts, big_graph_index_dict, sen_gene_indexs_set, nonsen_gene_indexs_set):
    """
    Helper function for parallel processing to calculate cell scores based on gene attention.
    - cell_index: Index of the cell being processed.
    - linked_genes, linked_atts: Lists of linked genes and their attention weights for the cell.
    - big_graph_index_dict: Mapping of sampled graph indices to original indices.
    - sen_gene_indexs_set, nonsen_gene_indexs_set: Sets of indices for senescent and nonsenescent genes.
    """
    sen_score = 0
    nonsen_score = 0
    for gene_index, gene_att in zip(linked_genes, linked_atts):
        if big_graph_index_dict[int(gene_index)] in sen_gene_indexs_set:
            sen_score += gene_att
        elif big_graph_index_dict[int(gene_index)] in nonsen_gene_indexs_set:
            nonsen_score += gene_att
    return cell_index, sen_score, nonsen_score
# Function to compute outliers based on IQR method for senescent cell scores
def compute_outliners_v1(sencell_indexs, sencell_scores):
    """
    Identifies outliers among senescent cells based on the Interquartile Range (IQR) method.
    - sencell_indexs: Indices of the senescent cells.
    - sencell_scores: Corresponding scores of these senescent cells.
    """
    q1 = np.quantile(sencell_scores, 0.25)
    q3 = np.quantile(sencell_scores, 0.75)
    iqr = q3 - q1
    upper_bound = q3 + (3 * iqr)
    outliers = sencell_indexs[sencell_scores > upper_bound]
    print('The number of outliers: ', len(outliers))
    return outliers

# Function to compute outliers using the Elliptic Envelope method
def compute_outliners_v2(sencell_indexs, sencell_scores):
    """
    Applies the Elliptic Envelope method to identify outliers among senescent cells.
    - sencell_indexs: Indices of senescent cells.
    - sencell_scores: Scores of these cells.
    """
    from sklearn.covariance import EllipticEnvelope
    model = EllipticEnvelope(contamination=0.01)
    res = model.fit_predict(sencell_scores.reshape(-1, 1))
    outliers = sencell_indexs[res == -1]
    print('The number of outliers: ', len(outliers))
    return outliers

# Function to identify senescent and nonsenescent cells using attention scores and predetermined gene indices
def identify_sencell_nonsencell(edge, att, sen_gene_indexs, nonsen_gene_indexs, cell_clusters, big_graph_index_dict, args):
    """
    Identifies senescent and nonsenescent cells based on attention scores and gene indices.
    - edge: Edge tensor from the GAT model indicating connections between nodes.
    - att: Attention scores associated with edges.
    - sen_gene_indexs: Indices of genes considered senescent.
    - nonsen_gene_indexs: Indices of genes considered nonsenescent.
    - cell_clusters: Dictionary mapping cell indices to their clusters.
    - big_graph_index_dict: Mapping of graph indices to their original indices.
    - args: Additional arguments for processing, such as output directory and experiment name.
    """
    cell_indexs = list(cell_clusters.keys())
    with CodeTimer("get_cell_scores", unit="s"):
        cell_score_dict = get_cell_scores(
            cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs)
    from utils import save_objs
    save_objs([cell_score_dict, big_graph_index_dict], os.path.join(
        args.output_dir, f'{args.exp_name}_cell_score_dict'))

    # Process scores to separate senescent and nonsenescent cells
    sencell_indexs = np.array([k for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][0])])
    sencell_scores = np.array([v[0] for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][0])])

    # Select the top cells based on different strategies
    sencell_index = sencell_indexs[-args.sencell_num:]  # Top cells based on predefined number
    # Remove selected cells from consideration for nonsenescent classification
    for index in sencell_index:
        cell_score_dict.pop(index)

    nonsencell_num = len(sencell_index) * 10  # Nonsenescent cells are ten times the number of senescent cells
    nonsencell_indexs = [k for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][1])]
    nonsencell_index = nonsencell_indexs[-nonsencell_num:]

    # Map cell indices back to their clusters for output
    sencell_cluster = [cell_clusters[int(i)] for i in sencell_index]
    nonsencell_cluster = [cell_clusters[int(i)] for i in nonsencell_index]

    return sencell_index, nonsencell_index, sencell_cluster, nonsencell_cluster

# Function to prepare the final output dictionaries for senescent and nonsenescent cells
def get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster):
    """
    Prepares final output data for senescent and nonsenescent cells, including embeddings and cluster information.
    - sencell_index, sencell_cluster: Indices and clusters for senescent cells.
    - z: Node embeddings from the graph.
    - big_graph_index_dict: Mapping of sampled graph indices to original graph indices.
    - nonsencell_index, nonsencell_cluster: Indices and clusters for nonsenescent cells.
    """
    sencell_dict = {}
    nonsencell_dict = {}
    for index, cluster in zip(sencell_index, sencell_cluster):
        i = int(index)
        sencell_dict[i] = [z[i], int(cluster), 0, big_graph_index_dict[i]]  # Embedding, cluster, placeholder for new_emb, original index
    for index, cluster in zip(nonsencell_index, nonsencell_cluster):
        i = int(index)
        nonsencell_dict[i] = [z[i], int(cluster), 0, big_graph_index_dict[i]]
    return sencell_dict, nonsencell_dict

# Function for random sub-sampling from a networkx graph for GAT analysis
def sub_sampling_by_random(graph_nx, sen_gene_ls, nonsen_gene_ls, model_GAT, args, sengene_marker=None, n_gene=50, gene_rate=0.3, cell_rate=0.5, debug=False):
    """
    Random sub-sampling from a biological network graph, which could be applied with or without specific markers.
    - graph_nx: The full NetworkX graph.
    - sen_gene_ls: List of senescent gene indices.
    - nonsen_gene_ls: List of nonsenescent gene indices.
    - model_GAT: GAT model to use for graph processing.
    - args: Arguments for processing including number of cells and genes, device info, etc.
    - sengene_marker: Specific marker genes to use for senescence analysis, if provided.
    - n_gene, gene_rate, cell_rate: Parameters to define the rates and numbers for sub-sampling genes and cells.
    - debug: Enable debugging output for detailed process tracing.
    """
    print("Start sampling subgraph randomly ...")
    n_cell = int(args.cell_num * cell_rate)  # Compute the number of cells to sample based on rate

    # Select senescence and nonsenescence genes either randomly or based on provided markers
    if sengene_marker is None:
        sen_gene_indexs = np.random.choice(sen_gene_ls, n_gene, replace=False)
    else:
        sen_gene_indexs = sengene_marker
    nonsen_gene_indexs = np.array(nonsen_gene_ls)

    # Debug output for gene counts
    print(f"sengene num: {len(sen_gene_indexs)}, Nonsengene num: {len(nonsen_gene_indexs)}")

    # Sampling a subgraph based on the genes selected and converting to PyTorch Geometric format
    gene_ls = np.concatenate([sen_gene_indexs, nonsen_gene_indexs])
    sampled_graph, cell_clusters, big_graph_index_dict = sub_sampling_by_genes(graph_nx, gene_ls)
    print('after sampling, gene num: ', sum(sampled_graph.y))

    # Calculate attention scores using GAT
    z, edge, att = calculateAtt(model_GAT, sampled_graph, 'cpu')

    # Identify senescent and nonsenescent cells based on attention scores and gene indices
    print('identify sencell and nonsencell!')
    sencell_index, nonsencell_index, sencell_cluster, nonsencell_cluster = identify_sencell_nonsencell(edge, att, sen_gene_indexs, nonsen_gene_indexs, cell_clusters, big_graph_index_dict, args)

    # Prepare the final output, storing embeddings, clusters, and original indices for senescent and nonsenescent cells
    sencell_dict, nonsencell_dict = get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sampled_graph, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict

# A modified version of the random sub-sampling method which doesn't sub-sample but works with the full graph
def sub_sampling_by_random_v1(graph_nx, graph_pyg, sen_gene_ls, nonsen_gene_ls, model_GAT, args, sengene_marker=None, n_gene=50, gene_rate=0.3, cell_rate=0.5, debug=False):
    """
    A modified version of the random sub-sampling method that does not actually sub-sample but processes the full graph.
    - graph_nx, graph_pyg: NetworkX and PyTorch Geometric graph representations.
    - sen_gene_ls, nonsen_gene_ls, model_GAT, args: Similar parameters as the sub_sampling_by_random function.
    - sengene_marker, n_gene, gene_rate, cell_rate, debug: Similarly, control the processing logic and debugging output.
    """
    print("Start sampling subgraph randomly ...")
    if sengene_marker is None:
        sen_gene_indexs = np.random.choice(sen_gene_ls, n_gene, replace=False)
    else:
        sen_gene_indexs = sengene_marker
    nonsen_gene_indexs = np.array(nonsen_gene_ls)

    # Direct use of the entire graph without sub-sampling
    print(f"sengene num: {len(sen_gene_indexs)}, Nonsengene num: {len(nonsen_gene_indexs)}")
    sampled_graph = graph_pyg
    cell_clusters_dict = {i: graph_nx.nodes[i]['cluster'] for i in range(len(graph_nx.nodes)) if graph_nx.nodes[i]['type'] == 'c'}
    cell_clusters = cell_clusters_dict

    big_graph_index_dict = {i: i for i in range(args.gene_num + args.cell_num)}

    # Calculate attention scores for the entire graph
    with CodeTimer("calculateAtt", unit="s"):
        z, edge, att = calculateAtt(model_GAT, sampled_graph, args.device)

    # Identify and classify cells based on attention scores and gene indices
    print('identify sencell and nonsencell!')
    sencell_index, nonsencell_index, sencell_cluster, nonsencell_cluster = identify_sencell_nonsencell(edge, att, sen_gene_indexs, nonsen_gene_indexs, cell_clusters, big_graph_index_dict, args)

    # Prepare and return the final outputs
    sencell_dict, nonsencell_dict = get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sampled_graph, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict, z


def identify_sen_genes(sencell_dict, edge, att, sen_gene_num):
    """
    Identifies potential senescent genes based on attention weights from cells identified as senescent.
    - sencell_dict: Dictionary containing indices of cells identified as senescent.
    - edge: Edge tensor from the graph model where the first dimension is the source node and the second is the target node.
    - att: Attention weights corresponding to each edge.
    - sen_gene_num: Number of senescent genes to identify.
    """
    sencell_indexs = list(sencell_dict.keys())
    sen_gene_dict = {}

    # Accumulate attention scores for genes connected to senescent cells
    for sencell_index in sencell_indexs:
        candidate_genes = edge[0][edge[1] == sencell_index]
        candidate_genes_atts = att[edge[1] == sencell_index]
        for candidate_gene, candidate_genes_att in zip(candidate_genes, candidate_genes_atts):
            index = int(candidate_gene)
            if index not in sen_gene_dict:
                sen_gene_dict[index] = [candidate_genes_att, 1]
            else:
                sen_gene_dict[index][0] += candidate_genes_att
                sen_gene_dict[index][1] += 1

    # Sort genes based on the average attention score they received
    score_ls = [k for k, v in sorted(sen_gene_dict.items(), key=lambda item: item[1][0] / item[1][1])]
    sen_gene_ls = score_ls[-sen_gene_num:]
    nonsen_gene_ls = score_ls[:1000]  # Arbitrary cutoff for non-senescent genes for further analysis

    return sen_gene_ls, nonsen_gene_ls


def identify_nonsen_genes(sampled_graph, sen_gene_ls):
    """
    Identifies nonsenescent genes from the graph by excluding those identified as senescent.
    - sampled_graph: The PyTorch Geometric graph containing all nodes.
    - sen_gene_ls: List of indices for genes identified as senescent.
    """
    all_geneset = set(torch.arange(len(sampled_graph.y))[sampled_graph.y].tolist())
    nonsen_gene_indexs = list(all_geneset - set(sen_gene_ls))

    return nonsen_gene_indexs


def identify_nonsen_genes_v1(nonsencell_dict, edge, att, sen_gene_num):
    """
    Identifies nonsenescent genes by analyzing cells not classified as senescent.
    - nonsencell_dict: Dictionary containing indices of cells not identified as senescent.
    - edge: Edge tensor from the graph model.
    - att: Attention weights corresponding to each edge.
    - sen_gene_num: Number of nonsenescent genes to identify.
    """
    nonsencell_indexs = list(nonsencell_dict.keys())
    nonsen_gene_dict = {}

    for nonsencell_index in nonsencell_indexs:
        candidate_genes = edge[0][edge[1] == nonsencell_index]
        candidate_genes_atts = att[edge[1] == nonsencell_index]
        for candidate_gene, candidate_genes_att in zip(candidate_genes, candidate_genes_atts):
            index = int(candidate_gene)
            if index not in nonsen_gene_dict:
                nonsen_gene_dict[index] = [candidate_genes_att, 1]
            else:
                nonsen_gene_dict[index][0] += candidate_genes_att
                nonsen_gene_dict[index][1] += 1

    # Select genes with highest attention scores
    nonsen_gene_ls = [k for k, v in sorted(nonsen_gene_dict.items(), key=lambda item: item[1][0] / item[1][1])][-sen_gene_num:]

    return nonsen_gene_ls


def identify_sengene_then_sencell(sampled_graph, model_GAT, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict, args, ratio=0.1, plot=False):
    """
    First identifies senescent genes and then reclassifies senescent cells based on updated gene classifications.
    - sampled_graph: Graph containing all nodes.
    - model_GAT: Graph Attention Network model used for analysis.
    - sencell_dict, nonsencell_dict: Dictionaries containing initial classifications of senescent and nonsenescent cells.
    - cell_clusters: Mapping of cell indices to cluster information.
    - big_graph_index_dict: Mapping from graph indices to original data indices.
    - args: Additional arguments including the number of senescent genes to identify.
    - ratio: Ratio to use for selecting top results.
    - plot: Enable plotting for visualization (not implemented in this snippet).
    """
    print("identify_sengene_then_sencell")
    z, edge, att = calculateAtt(model_GAT, sampled_graph, args.device)

    sen_gene_indexs, nonsen_gene_indexs = identify_sen_genes(sencell_dict, edge, att, args.sengene_num)

    print('rechoice sengene num:', len(sen_gene_indexs), 'rechoice nonsengene num:', len(nonsen_gene_indexs))

    # Translate local indices to global indices
    sen_gene_indexs = [big_graph_index_dict[i] for i in sen_gene_indexs]
    nonsen_gene_indexs = [big_graph_index_dict[i] for i in nonsen_gene_indexs]

    sencell_index, nonsencell_index, sencell_cluster, nonsencell_cluster = identify_sencell_nonsencell(edge, att, sen_gene_indexs, nonsen_gene_indexs, cell_clusters, big_graph_index_dict, args)

    sencell_dict, nonsencell_dict = get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sencell_dict, nonsencell_dict, sen_gene_indexs, nonsen_gene_indexs