import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as Data
from linetimer import CodeTimer  # Utility to time code execution
from torch import nn, optim
from torch.nn import Dropout, Linear, ReLU
from torch.nn import functional as F
from tqdm import tqdm  # For displaying progress bars

# Function to map clusters to their respective cells
def get_cluster_cell_dict(sencell_dict, nonsencell_dict):
    """
    Creates dictionaries to map cluster IDs to lists of cell indices for senescent and non-senescent cells.
    Arguments:
    - sencell_dict: Dictionary of senescent cells.
    - nonsencell_dict: Dictionary of non-senescent cells.
    
    Returns:
    - tuple of two defaultdicts: (cluster_sencell, cluster_nonsencell)
    """
    cluster_sencell = defaultdict(list)
    cluster_nonsencell = defaultdict(list)

    for key, value in sencell_dict.items():
        cluster_sencell[value[1]].append(key)

    for key, value in nonsencell_dict.items():
        cluster_nonsencell[value[1]].append(key)

    return cluster_sencell, cluster_nonsencell

# Function to compute prototype embeddings for each cluster
def getPrototypeEmb(sencell_dict, cluster_sencell, emb_pos=2):
    """
    Calculates the prototype embeddings for senescent cells by averaging embeddings within each cluster.
    Arguments:
    - sencell_dict: Dictionary of senescent cells.
    - cluster_sencell: Dictionary mapping clusters to senescent cell indices.
    - emb_pos: Position in the dictionary where embeddings are stored.
    
    Returns:
    - prototype_emb: Dictionary of prototype embeddings for each cluster.
    """
    prototype_emb = {}
    for key, value in cluster_sencell.items():
        embs = [sencell_dict[i][emb_pos].view(1, -1) for i in value]
        prototype_emb[key] = torch.mean(torch.cat(embs), 0)

    return prototype_emb

# Senescent cell model class
class Sencell(nn.Module):
    def __init__(self, dim=160):
        super().__init__()
        # Neural network layers for transforming embeddings
        self.linear1 = Linear(dim, 256)
        self.linear2 = Linear(256, 256)
        self.linear21 = Linear(256, 256)
        self.linear22 = Linear(256, 256)
        self.linear3 = Linear(256, 256)
        self.linear4 = Linear(256, dim)

        self.act = nn.CELU()  # Activation function
        self.layer_norm = nn.LayerNorm(256)  # Layer normalization
        self.levels = nn.Parameter(torch.tensor([0, 0, 4., 4]), requires_grad=True)  # Levels for adjusting model learning

    # Method to concatenate embeddings from senescent and non-senescent cells
    def catEmbeddings(self, sencell_dict, nonsencell_dict):
        embeddings = [v[0].view(1, -1) for v in sencell_dict.values()] + [v[0].view(1, -1) for v in nonsencell_dict.values()]
        return torch.cat(embeddings)

    # Method to update dictionaries with new embeddings
    def updateDict(self, x, sencell_dict, nonsencell_dict):
        count = 0
        for key in sencell_dict:
            sencell_dict[key][2] = x[count]
            count += 1
        for key in nonsencell_dict:
            nonsencell_dict[key][2] = x[count]
            count += 1
        return sencell_dict, nonsencell_dict

    # Forward pass of the model
    def forward(self, sencell_dict, nonsencell_dict, device):
        x = self.catEmbeddings(sencell_dict, nonsencell_dict).to(device)
        self.device = device

        # Process embeddings through linear layers and activations
        x = self.act(self.linear1(x))
        x = x + self.act(self.linear2(x))
        x = self.layer_norm(x)
        x = x + self.act(self.linear22(self.act(self.linear21(x))))
        x = self.layer_norm(x)
        x = self.linear4(self.act(self.linear3(x)))

        result = self.updateDict(x, sencell_dict, nonsencell_dict)
        return result

    # Additional methods for distance calculations and loss functions omitted for brevity

# Function to process dictionary elements
def process_dict(cell_dict, dgl_graph, args):
    """
    Updates the cell dictionary with position encodings from a graph if available.
    Arguments:
    - cell_dict: Dictionary of cells, each key containing various cell properties.
    - dgl_graph: Graph structure that possibly contains additional node features.
    - args: Command line arguments or other configurations.
    
    Returns:
    - cell_dict: Updated cell dictionary with concatenated position encodings if dgl_graph is provided.
    """
    if dgl_graph is None:
        # If there is no graph data available, return the dictionary unchanged
        return cell_dict
    else:
        # Update each cell's data with positional encodings from the graph
        for key in cell_dict:
            cell_index = cell_dict[key][-1] - args.gene_num
            pos_enc = dgl_graph.nodes[cell_index].data['pos_enc'].reshape(-1,)
            cell_dict[key][0] = torch.cat([cell_dict[key][0], pos_enc])
        return cell_dict

def cell_optim(cellmodel, optimizer, sencell_dict, nonsencell_dict, dgl_graph, args, train=False, wandb=None):
    """
    Trains or loads a cell model, optimizing it based on the specified dictionaries of senescent and non-senescent cells.
    Arguments:
    - cellmodel: The cell model to be trained or evaluated.
    - optimizer: Optimizer used for training.
    - sencell_dict: Dictionary containing data of senescent cells.
    - nonsencell_dict: Dictionary containing data of non-senescent cells.
    - dgl_graph: Graph containing additional node features.
    - args: Additional arguments or configurations.
    - train: Flag to indicate if the model should be trained.
    - wandb: Optional Weights & Biases logging.

    Returns:
    - cellmodel: Trained or loaded cell model.
    - sencell_dict: Updated dictionary of senescent cells.
    - nonsencell_dict: Updated dictionary of non-senescent cells.
    """
    if train:
        cellmodel.train()
        # Process cell dictionaries with potentially new data from a graph
        sencell_dict = process_dict(sencell_dict, dgl_graph, args)
        nonsencell_dict = process_dict(nonsencell_dict, dgl_graph, args)

        for epoch in range(args.cell_optim_epoch):
            optimizer.zero_grad()
            # Update dictionaries with the latest model output
            sencell_dict, nonsencell_dict = cellmodel(sencell_dict, nonsencell_dict, args.device)
            loss = cellmodel.loss(sencell_dict, nonsencell_dict)  # Compute loss
            print(loss.item())
            if wandb is not None:
                wandb.log({"loss": loss})  # Log loss if Weights & Biases is used
            loss.backward()
            optimizer.step()

        # Save the trained model
        torch.save(cellmodel, os.path.join(args.output_dir, f'{args.exp_name}_cellmodel.pt'))
    else:
        # Load the model if not training
        cellmodel = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_cellmodel.pt'))
        sencell_dict, nonsencell_dict = cellmodel(sencell_dict, nonsencell_dict, args.device)

    return cellmodel, sencell_dict, nonsencell_dict

def update_cell_embeddings(sampled_graph, sencell_dict, nonsencell_dict):
    """
    Updates the embeddings in a graph based on the provided cell dictionaries.
    Arguments:
    - sampled_graph: Graph whose embeddings are to be updated.
    - sencell_dict: Dictionary of senescent cells with their embeddings.
    - nonsencell_dict: Dictionary of non-senescent cells with their embeddings.
    
    Returns:
    - sampled_graph: Graph with updated embeddings.
    """
    feature = sampled_graph.x
    for key, value in sencell_dict.items():
        feature[key] = value[2].detach()  # Update senescent cell embeddings
    for key, value in nonsencell_dict.items():
        feature[key] = value[2].detach()  # Update non-senescent cell embeddings

    sampled_graph.x = feature
    return sampled_graph

