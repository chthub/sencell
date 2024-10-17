import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np

from torch_geometric.nn import Sequential, GATConv, TransformerConv
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn.models import InnerProductDecoder, GAE, VGAE
from torch_geometric.nn import GATConv, GAE


# from sampling import sub_sampling_GAT
from tqdm import tqdm

import os
import time


# Define GAT-based encoder for GAE
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.6)
        self.conv2 = GATConv(32 * 1, out_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize GAE model with GAT encoder and move it to the GPU
class GAEModel(GAE):
    def __init__(self, in_channels, out_channels):
        encoder = GATEncoder(in_channels, out_channels)
        super(GAEModel, self).__init__(encoder)

    def get_attention_scores(self, data):
        x, edge_index = data.x, data.edge_index
        # NOTE: Pass data through the first GAT layer to get attention scores
        _, (edge_index_selfloop, alpha) = self.encoder.conv1(x, edge_index, return_attention_weights=True)
        # matrix shape: number of edges x number of heads
        # Decimal point digit truncation, 10^6
        return edge_index_selfloop,alpha
    

class Encoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear1 = Linear(dim, dim)
        self.linear2 = Linear(dim, dim)

        # 默认有自环，所以attention里面也有自环的attention
        # 为了后面的处理，这里去掉自环
        self.conv1 = GATConv(dim, dim, add_self_loops=False)
        self.conv2 = GATConv(dim, dim, add_self_loops=False)
        
        # self.conv1 = TransformerConv(dim, dim, heads=1)
        # self.conv2 = TransformerConv(dim, dim, heads=1)

        self.act = torch.nn.CELU()

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

        result = torch.cat(result)
        return result

    def forward(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y

        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return x

    def get_att(self, graph):
        x, edge_index,  y = graph.x, graph.edge_index, graph.y
        print(x.shape,y.shape)
        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x, att = self.conv2(x, edge_index, return_attention_weights=True)

        return x, att


class SenGAE(GAE):
    def __init__(self):
        super(SenGAE, self).__init__(encoder=Encoder(),
                                     decoder=InnerProductDecoder())

    def forward(self, graph, split=10):
        z = self.encode(graph)
        # adj_pred = self.decoder(z)
        return z


# def sampling_jobs_seq(graph_nx, graph, args):
#     # 采样串行
#     t0 = time.time()
#     num_subgraphs = 50
#     jobs = []
#     print("Start sampling ...")
#     if args.is_jupyter:
#         for _ in tqdm(range(num_subgraphs)):
#             sampled_graph = sub_sampling_GAT(
#                 graph_nx, graph, gene_num=args.gene_num, cell_num=args.cell_num)
#             jobs.append(sampled_graph)
#     else:
#         for _ in range(num_subgraphs):
#             sampled_graph = sub_sampling_GAT(
#                 graph_nx, graph, gene_num=args.gene_num, cell_num=args.cell_num)
#             jobs.append(sampled_graph)
#     print('sampling end, time: ', time.time()-t0)
#     return jobs


# def sampling_jobs_par(graph_nx, graph, args):
#     from multiprocessing import Pool
#     # 采样并行，多进程版
#     # 并行版本报错，未解决
#     t0 = time.time()
#     num_subgraphs = 50
#     jobs = []
#     print("Start sampling ...")
#     p = Pool()
#     graph = graph.to('cpu')
#     for _ in tqdm(range(num_subgraphs)):
#         jobs.append(p.apply_async(sub_sampling_GAT, args=(
#             graph_nx, graph, args.gene_num, args.cell_num,)))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     jobs = [i.get() for i in jobs]
#     print('sampling end, time: ', time.time()-t0)

#     return jobs


def train_GAT(graph_nx, graph, args, retrain=False, resampling=False,wandb=None):
    # step 1: init model
    model = SenGAE().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                 weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', factor=0.5, patience=10, verbose=True)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99,
    #                                 weight_decay=1e-4)

    # step 2: load or training
    if not retrain:
        print('skip training!')
        model = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_gat.pt'))
        return model

    # step 3: 子图采样
    if resampling:
        jobs = sampling_jobs_seq(graph_nx, graph, args)
        torch.save(jobs, os.path.join(
            args.output_dir, f'{args.exp_name}_jobs'))
    else:
        jobs = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_jobs'))

    # step 4: training
    model.train()
    for EPOCH in range(args.gat_epoch):
        epoch_ls = []
        print('Epoch: ', EPOCH)
        for sampled_graph in jobs:
            sampled_graph = sampled_graph.to(args.device)
            loss_ls = []
            for epoch in range(1):
                optimizer.zero_grad()
                z = model(sampled_graph)
                loss = model.recon_loss(z, sampled_graph.edge_index)
                loss_ls.append(loss.item())
                loss.backward()
                optimizer.step()
            subgraph_loss = np.mean(loss_ls)
            epoch_ls.append(subgraph_loss)
            print('subgraph loss: ', subgraph_loss)
            if wandb is not None:
                wandb.log({"subgraph loss":loss})
    #         scheduler.step(subgraph_loss)
        print('EPOCH loss', np.mean(epoch_ls))

    torch.save(model, os.path.join(args.output_dir, f'{args.exp_name}_gat.pt'))
    return model


def train_GAT_new(graph_nx, graph, args, retrain=False, resampling=False):
    # 不采样了，直接训练整个图
    # CUDA out of memory
    # step 1: init model
    model = SenGAE().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                 weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', factor=0.5, patience=10, verbose=True)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99,
    #                                 weight_decay=1e-4)

    # step 2: load or training
    if not retrain:
        model = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_gat.pt'))
        return model

    # step 4: training
    model.train()
    graph=graph.to(args.device)
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