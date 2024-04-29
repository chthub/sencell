import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np

from torch_geometric.nn import Sequential, GATConv, TransformerConv
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn.models import InnerProductDecoder, GAE, VGAE

from sampling import sub_sampling_GAT
from tqdm import tqdm

import os
import time

# Define an Encoder class as a subclass of torch.nn.Module
class Encoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # Linear layers to transform the input features into higher dimensional space
        self.linear1 = Linear(dim, dim)
        self.linear2 = Linear(dim, dim)

        # TransformerConv layers are used for graph convolution over nodes
        self.conv1 = TransformerConv(dim, dim, heads=1)
        self.conv2 = TransformerConv(dim, dim, heads=1)

        # Activation function for non-linearity
        self.act = torch.nn.CELU()

    # Custom method to concatenate gene and cell features based on a binary mask
    def cat(self, x_gene, x_cell, y):
        result = []
        count_gene = 0
        count_cell = 0

        for i in y:
            if i:
                result.append(x_gene[count_gene].view(1, -1))
                count_gene += 1
            else:
                result.append(x_cell[count_cell].view(1, -1))
                count_cell += 1

        return torch.cat(result)

    # Forward pass through the network
    def forward(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y

        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return x

    # Method to get attention weights from the second TransformerConv layer
    def get_att(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x, att = self.conv2(x, edge_index, return_attention_weights=True)
        return x, att

# Graph Autoencoder for learning graph embeddings
class SenGAE(GAE):
    def __init__(self):
        super(SenGAE, self).__init__(encoder=Encoder(), decoder=InnerProductDecoder())

    def forward(self, graph, split=10):
        z = self.encode(graph)
        return z

# Sequential sampling function for subgraphs
def sampling_jobs_seq(graph_nx, graph, args):
    t0 = time.time()
    num_subgraphs = 50
    jobs = []
    print("Start sampling ...")
    if args.is_jupyter:
        for _ in tqdm(range(num_subgraphs)):
            sampled_graph = sub_sampling_GAT(graph_nx, graph, gene_num=args.gene_num, cell_num=args.cell_num)
            jobs.append(sampled_graph)
    else:
        for _ in range(num_subgraphs):
            sampled_graph = sub_sampling_GAT(graph_nx, graph, gene_num=args.gene_num, cell_num=args.cell_num)
            jobs.append(sampled_graph)
    print('sampling end, time: ', time.time() - t0)
    return jobs

# Parallel sampling function using multiprocessing
def sampling_jobs_par(graph_nx, graph, args):
    t0 = time.time()
    num_subgraphs = 50
    jobs = []
    print("Start sampling ...")
    with Pool() as p:
        graph = graph.to('cpu')
        for _ in tqdm(range(num_subgraphs)):
            jobs.append(p.apply_async(sub_sampling_GAT, args=(graph_nx, graph, args.gene_num, args.cell_num,)))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
    jobs = [i.get() for i in jobs]
    print('sampling end, time: ', time.time() - t0)
    return jobs

# Training function for the Graph Attention Network
def train_GAT(graph_nx, graph, args, retrain=False, resampling=False, wandb=None):
    model = SenGAE().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    if not retrain:
        print('skip training!')
        return torch.load(os.path.join(args.output_dir, f'{args.exp_name}_gat.pt'))

    if resampling:
        jobs = sampling_jobs_seq(graph_nx, graph, args)
        torch.save(jobs, os.path.join(args.output_dir, f'{args.exp_name}_jobs'))
    else:
        jobs = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_jobs'))

    model.train()
    for EPOCH in range(args.gat_epoch):
        epoch_ls = []
        print('Epoch: ', EPOCH)
        for sampled_graph in jobs:
            sampled_graph = sampled_graph.to(args.device)
            loss_ls = []
            for _ in range(1):
                optimizer.zero_grad()
                z = model(sampled_graph)
                loss = model.recon_loss(z, sampled_graph.edge_index)
                loss.backward()
                optimizer.step()
                loss_ls.append(loss.item())
            subgraph_loss = np.mean(loss_ls)
            epoch_ls.append(subgraph_loss)
            print('subgraph loss: ', subgraph_loss)
            if wandb:
                wandb.log({"subgraph loss": subgraph_loss})
        print('EPOCH loss', np.mean(epoch_ls))

    torch.save(model, os.path.join(args.output_dir, f'{args.exp_name}_gat.pt'))
    return model

# Function to train the model on the entire graph directly, avoiding sub-sampling
def train_GAT_new(graph_nx, graph, args, retrain=False, resampling=False):
    model = SenGAE().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    if not retrain:
        return torch.load(os.path.join(args.output_dir, f'{args.exp_name}_gat.pt'))

    model.train()
    graph = graph.to(args.device)
    for EPOCH in range(30):
        print('Epoch: ', EPOCH)
        optimizer.zero_grad()
        z = model(graph)
        loss = model.recon_loss(z, graph.edge_index)
        loss.backward()
        optimizer.step()
        print('graph loss: ', loss.item())

    torch.save(model, os.path.join(args.output_dir, f'{args.exp_name}_gat.pt'))
    return model
