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

from linetimer import CodeTimer

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='# %Y-%m-%d %H:%M:%S')

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()


def convertNxtoPyg(graph_nx, is_clusters=False, is_big_graph_index=False):
    # 费时
    # is_clusters用来输出cell的cluster信息
    # is_big_graph_index用来输出节点在大图上的index
    # 下面这俩dict正好存储的key,value相反
    # relabel_dict存储大图上的index和新图上index的映射
    relabel_dict = {}
    # big_graph_index_dict存储新图上index和大图index的映射，{new_index: big_graph_index}
    big_graph_index_dict = {}
    for i in graph_nx.nodes:
        new_index = len(relabel_dict)
        relabel_dict[i] = new_index
        big_graph_index_dict[new_index] = i

    relabeled_nx = nx.relabel_nodes(graph_nx, relabel_dict)

    emb = []
    # gene是1，cell是0
    types = []
    # key是新图上节点的index, value是节点的cluster
    cell_clusters_dict = {}
    for i in range(len(relabeled_nx.nodes)):
        emb.append(relabeled_nx.nodes[i]['emb'].view(1, -1))
        types.append(relabeled_nx.nodes[i]['type'] == 'g')
        if is_clusters and relabeled_nx.nodes[i]['type'] == 'c':
            cell_clusters_dict[i] = relabeled_nx.nodes[i]['cluster']

    emb = torch.cat(emb)

    edge_index = torch.tensor(list(relabeled_nx.edges)).T
    edge_index = to_undirected(edge_index)

    result = Graphdata(x=emb, edge_index=edge_index, y=torch.tensor(types))

    if is_big_graph_index and is_clusters:
        return result, cell_clusters_dict, big_graph_index_dict
    if is_clusters:
        return result, cell_clusters_dict
    return result


def sub_sampling_GAT(graph_nx, graph, gene_num=2245, cell_num=10558,
                     gene_rate=0.3, cell_rate=0.5,
                     debug=False):
    # 每次采样的基因数量
    n_gene = int(gene_num*gene_rate)
    # 每次采样的细胞数量
    n_cell = int(cell_num*cell_rate)

    gene_indexs = np.random.choice(range(gene_num), n_gene, replace=False)
    # 这里也用n_gene
    cell_indexs = np.random.choice(
        range(gene_num, cell_num+gene_num), n_gene, replace=False)

    new_cell_indexs = set()
    new_gene_indexs = set()

    for i, j in zip(gene_indexs, cell_indexs):
        # 采gene的邻居
        new_cell_indexs.update(list(graph_nx[i]))
        # 采cell的邻居
        new_gene_indexs.update(list(graph_nx[j]))
    # 选择其中的10%
    new_gene_indexs = np.random.choice(list(new_gene_indexs),
                                       int(len(new_gene_indexs)*0.4),
                                       replace=False)
    new_cell_indexs = np.random.choice(list(new_cell_indexs),
                                       int(len(new_cell_indexs)*0.1),
                                       replace=False)

    # print('gene num:',len(new_gene_indexs),'cell num:',len(new_cell_indexs))
    # print(new_gene_indexs,new_cell_indexs)
    node_set = np.concatenate([new_gene_indexs, new_cell_indexs])

    def filter_node(node):
        return node in node_set

    view = nx.subgraph_view(graph_nx, filter_node=filter_node)

    result = convertNxtoPyg(view)
    if debug:
        return result, new_gene_indexs, new_cell_indexs
    else:
        return result


def sub_sampling_by_genes(graph_nx, marker_index):
    # 给定一组基因marker_index，采样一阶邻居子图
    # 这个函数还可以用来在给定一组gene，采样这组基因一阶邻居子图的情况
    cell_set = set()
    for marker in marker_index:
        cell_set.update(list(graph_nx[marker]))
    node_set = np.concatenate([marker_index, list(cell_set)])
    print('subgraph total node num:', node_set.shape)

    def filter_node(node):
        return node in node_set

    view = nx.subgraph_view(graph_nx, filter_node=filter_node)

    # cell_clusters是cell的index从小到大的cluster
    result, cell_clusters, big_graph_index_dict = convertNxtoPyg(view, is_clusters=True,
                                                                 is_big_graph_index=True)
    return result, cell_clusters, big_graph_index_dict


def identify_sencell_marker_graph(sampled_graph, model, cell_clusters,
                                  big_graph_index_dict,
                                  device,
                                  ratio=0.1, plot=False):
    # 这个函数专门用来识别gene全是marker的图上的sencell
    # cell_clusters: dict

    # step 1: get embedding and attention
    model.eval()
    model.to(device)
    sampled_graph.to(device)
    # edge_att: nx1
    # 注意这里的的edge是双向边
    # 对应的edge_att也是双向边的权重，而且二者不一定相同
    # 这里edge表示(source,target)
    # 所以一个target节点与所有source节点之间边attention之和为1
    # 所以这里找老化细胞考虑的cell score是以gene为target, cell score是其出现过的所有attention之和
    z, (edge, edge_att) = model.encoder.get_att(sampled_graph)
    att = edge_att.T[0].detach().cpu()
    z = z.detach().cpu()

    # step 2: identify sencell and nonsencell
    cell_index = torch.arange(len(sampled_graph.y))[
        torch.bitwise_not(sampled_graph.y)]
    cell_att = []
    for cell in cell_index:
        cell_att.append(att[edge[0] == cell].sum())
    cell_att = torch.tensor(cell_att)
    # 从大到小
    sorted_index = torch.argsort(cell_att, descending=True)
    sencell_num = 100
    if plot:
        sns.displot(cell_att)
        plt.ylim(0, 20)
        sns.displot(cell_att[sorted_index][:sencell_num])
    print('sencell_num:', sencell_num)

    sencell_index = cell_index[sorted_index][:sencell_num]
    nonsencell_index = cell_index[sorted_index][sencell_num:]

    sencell_cluster = [cell_clusters[int(i)] for i in sencell_index]
    nonsencell_cluster = [cell_clusters[int(i)] for i in nonsencell_index]

    # step 3: output
    # 最后的输出结果，存储[emb，cluster, new_emb=0, big_graph_index]信息
    # new_emb用来占位，big_graph_index是cell节点在大图上的index
    sencell_dict, nonsencell_dict = get_outputs(
        sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sencell_dict, nonsencell_dict


def select_cell_by_genes(sampled_graph,
                         sen_gene_indexs, nonsen_gene_indexs,
                         cell_clusters,
                         big_graph_index_dict,
                         model_GAT, device):
    # 在给定sengen和nonsengene的条件下，基于attention选取sampled_graph上的sencell和nonsencell

    # step 3: 计算attention
    model_GAT.eval()
    model_GAT.to(device)
    sampled_graph.to(device)
    # edge_att: nx1
    # 注意这里的的edge是双向边
    # 对应的edge_att也是双向边的权重，而且二者不一定相同
    # 这里edge表示(source,target)
    # 所以一个target节点与所有source节点之间边attention之和为1
    # 所以这里找老化细胞考虑的cell score是以gene为target, cell score是其出现过的所有attention之和
    z, (edge, edge_att) = model_GAT.encoder.get_att(sampled_graph)
    att = edge_att.T[0].detach().cpu()
    z = z.detach().cpu()

    # step 4: identify sencell and nonsencell
    cell_indexs = torch.arange(len(sampled_graph.y))[
        torch.bitwise_not(sampled_graph.y)]
    cell_att = []
    # 存储每个cell的score, {cell_index:[sen_score,nonsen_score]}
    cell_score_dict = {}
    for cell_index in cell_indexs:
        linked_genes = edge[1][edge[0] == cell_index]
        linked_atts = att[edge[0] == cell_index]
        sen_score = 0
        nonsen_score = 0
        for gene_index, gene_att in zip(linked_genes, linked_atts):
            if big_graph_index_dict[int(gene_index)] in sen_gene_indexs:
                sen_score += gene_att
            elif big_graph_index_dict[int(gene_index)] in nonsen_gene_indexs:
                nonsen_score += gene_att
            else:
                continue
                # 不一定是bug了，因为这两个基因列表不是全部
                print('Bug!!!')
        if cell_index in cell_score_dict:
            old_score = cell_score_dict[cell_index]
            cell_score_dict[cell_index] = [old_score[0]+sen_score,
                                           old_score[1]+nonsen_score]
        else:
            cell_score_dict[cell_index] = [sen_score, nonsen_score]
    # score从小到大
    sencell_indexs = [k for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][0])]
    sencell_num = 100
    sencell_index = sencell_indexs[-sencell_num:]
    # 去掉选择出的老化细胞
    for index in sencell_index:
        cell_score_dict.pop(index)

    nonsencell_num = 900
    nonsencell_indexs = [k for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][1])]
    nonsencell_index = nonsencell_indexs[-nonsencell_num:]

    sencell_cluster = [cell_clusters[int(i)] for i in sencell_index]
    nonsencell_cluster = [cell_clusters[int(i)] for i in nonsencell_index]

    # step 3: output
    # 最后的输出结果，存储[emb，cluster, new_emb=0, big_graph_index]信息
    # new_emb用来占位，big_graph_index是cell节点在大图上的index
    sencell_dict, nonsencell_dict = get_outputs(
        sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sampled_graph, sencell_dict, nonsencell_dict


def calculateAtt(model_GAT, graph_pyg, device):
    # logger.info("Calculate attentions ...")
    model_GAT.eval()
    model_GAT.to(device)
    graph_pyg.to(device)
    # edge_att: nx1
    # 注意这里的的edge是双向边
    # 对应的edge_att也是双向边的权重，而且二者不一定相同
    # 这里edge表示(source,target)
    # 所以一个target节点与所有source节点之间边attention之和为1
    # 所以这里找老化细胞考虑的cell score是以gene为target, cell score是其出现过的所有attention之和
    z, (edge, edge_att) = model_GAT.encoder.get_att(graph_pyg)
    att = edge_att.T[0].detach().cpu()
    z = z.detach().cpu()
    edge = edge.cpu()
    # logger.info("Calculate attentions end.")
    return z, edge, att


def get_cell_scores(cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs):
    # 费时
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
            else:
                print('Bug!!!')
        cell_score_dict[cell_index] = [float(sen_score), float(nonsen_score)]

    return cell_score_dict


def sub_get_cell_scores_par(cell_index, linked_genes, linked_atts, big_graph_index_dict, sen_gene_indexs_set, nonsen_gene_indexs_set):
    sen_score = 0
    nonsen_score = 0
    for gene_index, gene_att in zip(linked_genes, linked_atts):
        if big_graph_index_dict[int(gene_index)] in sen_gene_indexs_set:
            sen_score += gene_att
        elif big_graph_index_dict[int(gene_index)] in nonsen_gene_indexs_set:
            nonsen_score += gene_att
        else:
            print('Bug!!!')

    return cell_index, sen_score, nonsen_score


def get_cell_scores_par1(cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs):
    # 并行版本，多进程，耗时还变长了
    from multiprocessing import Pool
    pool = Pool()
    results = []

    cell_score_dict = {}
    sen_gene_indexs_set = set(sen_gene_indexs)
    nonsen_gene_indexs_set = set(nonsen_gene_indexs)

    for cell_index in cell_indexs:
        linked_genes = edge[1][edge[0] == cell_index]
        linked_atts = att[edge[0] == cell_index]
        parms = (cell_index, linked_genes, linked_atts, big_graph_index_dict,
                 sen_gene_indexs_set, nonsen_gene_indexs_set,)
        results.append(pool.apply_async(sub_get_cell_scores_par, parms))
    pool.close()
    pool.join()

    for res in results:
        cell_score_dict[res.get()[0]] = [float(
            res.get()[1]), float(res.get()[2])]

    return cell_score_dict


def get_cell_scores_par2(cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs):
    # 并行版本，多线程，效果提升显著
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
    import threading

    jobs = []
    pool = ThreadPoolExecutor(max_workers=1000)

    def sub_get_cell_scores_par1(cell_index):
        linked_genes = edge[1][edge[0] == cell_index]
        linked_atts = att[edge[0] == cell_index]
        sen_score = 0
        nonsen_score = 0
        for gene_index, gene_att in zip(linked_genes, linked_atts):
            if big_graph_index_dict[int(gene_index)] in sen_gene_indexs_set:
                sen_score += gene_att
            elif big_graph_index_dict[int(gene_index)] in nonsen_gene_indexs_set:
                nonsen_score += gene_att
            else:
                print('Bug!!!')

        cell_score_dict[cell_index] = [float(sen_score), float(nonsen_score)]

    cell_score_dict = {}
    sen_gene_indexs_set = set(sen_gene_indexs)
    nonsen_gene_indexs_set = set(nonsen_gene_indexs)

    for cell_index in cell_indexs:
        jobs.append(pool.submit(sub_get_cell_scores_par1, cell_index))

    wait(jobs, return_when=ALL_COMPLETED)
    pool.shutdown()

    return cell_score_dict


def identify_sencell_nonsencell(edge, att, sen_gene_indexs, nonsen_gene_indexs, cell_clusters, big_graph_index_dict, args):
    cell_indexs = list(cell_clusters.keys())
    # 存储每个cell的score, {cell_index:[sen_score,nonsen_score]}
    # with CodeTimer("get_cell_scores_par2",unit="s"):
    cell_score_dict = get_cell_scores_par2(
        cell_indexs, edge, att, big_graph_index_dict, sen_gene_indexs, nonsen_gene_indexs)
    from utils import save_objs
    save_objs([cell_score_dict, big_graph_index_dict], os.path.join(
        args.output_dir, f'{args.exp_name}_cell_score_dict_test'))

    # score从小到大
    sencell_indexs = np.array([k for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][0])])
    sencell_scores = np.array([v[0] for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][0])])
    # 选择score>0.1的
    # sencell_index=sencell_indexs[sencell_scores>0.1]
    #  或者直接选择socre最大的200个
    sencell_index = sencell_indexs[-args.sencell_num:]
    # 去掉选择出的老化细胞
    for index in sencell_index:
        cell_score_dict.pop(index)

    # nonsencell_num是sencell的10倍
    nonsencell_num = len(sencell_index)*10
    nonsencell_indexs = [k for k, v in sorted(
        cell_score_dict.items(), key=lambda item: item[1][1])]
    nonsencell_index = nonsencell_indexs[-nonsencell_num:]

    print(
        f"    Sencell num: {len(sencell_index)}, Nonsencell num: {len(nonsencell_index)}")
    sencell_cluster = [cell_clusters[int(i)] for i in sencell_index]
    nonsencell_cluster = [cell_clusters[int(i)] for i in nonsencell_index]

    return sencell_index, nonsencell_index, sencell_cluster, nonsencell_cluster


def get_outputs(sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster):
    sencell_dict = {}
    nonsencell_dict = {}

    for index, cluster in zip(sencell_index, sencell_cluster):
        # index是tensor
        i = int(index)
        sencell_dict[i] = [z[i], int(cluster), 0, big_graph_index_dict[i]]
    for index, cluster in zip(nonsencell_index, nonsencell_cluster):
        i = int(index)
        nonsencell_dict[i] = [z[i], int(cluster), 0, big_graph_index_dict[i]]
    return sencell_dict, nonsencell_dict


def sub_sampling_by_random(graph_nx,
                           sen_gene_ls,
                           nonsen_gene_ls,
                           model_GAT,
                           args,
                           sengene_marker=None,
                           n_gene=50,
                           gene_rate=0.3, cell_rate=0.5,
                           debug=False):
    # 这是采样的主函数
    # 可以支持两种方式的采样，区别是有无marker
    # 1. 给定一组老化的marker基因
    # 2. 随机采样
    # 这种采样基因这边既有marker基因，也有非marker基因
    print("Start sampling subgraph randomly ...")
    # 每次采样的细胞数量
    n_cell = int(args.cell_num*cell_rate)

    # np.random.seed(0)
    # step 1: 选择sen_gene and nonsen_gene
    if sengene_marker is None:
        sen_gene_indexs = np.random.choice(sen_gene_ls, n_gene, replace=False)
    else:
        sen_gene_indexs = sengene_marker
    # nonsen_gene_indexs=np.random.choice(nonsen_gene_ls,n_gene,replace=False)
    nonsen_gene_indexs = np.array(nonsen_gene_ls)

    print(
        f"    Sengene num: {len(sen_gene_indexs)}, Nonsengen num: {len(nonsen_gene_indexs)}")
    # step 2: 采样一阶邻居子图
    gene_ls = np.concatenate([sen_gene_indexs, nonsen_gene_indexs])
    assert len(set(gene_ls)) == len(sen_gene_indexs) + \
        len(nonsen_gene_indexs), '基因有重叠！'
    # sampled_graph里面节点的index是relabeled，需要通过big_graph_index_dict转化成大图上的index
    sampled_graph, cell_clusters, big_graph_index_dict = sub_sampling_by_genes(
        graph_nx, gene_ls)

    print('After sampling, gene num: ', sum(sampled_graph.y))

    # step 3: 计算attention
    z, edge, att = calculateAtt(model_GAT, sampled_graph, 'cpu')

    # step 4: identify sencell and nonsencell
    # with CodeTimer("identify sencell",unit="s"):
    sencell_index, nonsencell_index, \
        sencell_cluster, nonsencell_cluster = identify_sencell_nonsencell(edge, att,
                                                                          sen_gene_indexs, nonsen_gene_indexs,
                                                                          cell_clusters, big_graph_index_dict, args)

    # step 3: output
    # 最后的输出结果，存储[emb，cluster, new_emb=0, big_graph_index]信息
    # new_emb用来占位，big_graph_index是cell节点在大图上的index
    sencell_dict, nonsencell_dict = get_outputs(
        sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sampled_graph, sencell_dict, nonsencell_dict, cell_clusters, big_graph_index_dict


def identify_sen_genes(sencell_dict, nonsencell_dict, edge, att, sen_gene_num):
    sencell_indexs = list(sencell_dict.keys())
    nonsencell_indexs = list(nonsencell_dict.keys())

    # 为每一个sencell所连接的基因赋予attention权重
    # 这里面存储与sencell所相连的gene（key）和attention权重（value）
    sen_gene_dict = {}
    for sencell_index in sencell_indexs:
        candidate_genes = edge[0][edge[1] == sencell_index]
        candidate_genes_atts = att[edge[1] == sencell_index]
        for candidate_gene, candidate_genes_att in zip(candidate_genes, candidate_genes_atts):
            index = int(candidate_gene)
            if index not in sen_gene_dict:
                sen_gene_dict[index] = [candidate_genes_att, 1]
            else:
                sen_gene_dict[index] = [sen_gene_dict[index][0] +
                                        candidate_genes_att, sen_gene_dict[index][1]+1]
            # sen_gene_dict[index]=sen_gene_dict.get(index,0)+candidate_genes_att

    # 依据att从小到大
    # sen_gene_num=100
    # non_sen_gene_num=50
    # 怀疑这里的基因数量可能会影响结果
    sen_gene_ls = [k for k, v in sorted(sen_gene_dict.items(
    ), key=lambda item: item[1][0]/item[1][1])][-sen_gene_num:]
    return sen_gene_ls


def identify_nonsen_genes(sampled_graph, sen_gene_ls):
    all_geneset = set(torch.arange(len(sampled_graph.y))
                      [sampled_graph.y].tolist())
    nonsen_gene_indexs = list(all_geneset-set(sen_gene_ls))
    return nonsen_gene_indexs


def identify_sengene_then_sencell(sampled_graph, model_GAT,
                                  sencell_dict,
                                  nonsencell_dict,
                                  cell_clusters,
                                  big_graph_index_dict,
                                  sengene_num,
                                  args,
                                  ratio=0.1, plot=False):
    # 基于更新后的老化细胞embedding，重新选择老化基因
    # 这个函数里面首先会先识别老化基因
    # 然后再识别老化细胞
    # cell_clusters: tensor

    # step 1: get embedding and attention
    z, edge, att = calculateAtt(model_GAT, sampled_graph, 'cpu')

    # step 2: 识别老化基因和非老化基因
    # 这里的到的index都是relabled后的
    sen_gene_indexs = identify_sen_genes(
        sencell_dict, nonsencell_dict, edge, att, sengene_num)
    nonsen_gene_indexs = identify_nonsen_genes(sampled_graph, sen_gene_indexs)
    print('rechoice sengene num:', len(sen_gene_indexs),
          'rechoice nonsengene num:', len(nonsen_gene_indexs))

    # 把得到的基因转化成大图上的gene
    sen_gene_indexs = [big_graph_index_dict[i] for i in sen_gene_indexs]
    nonsen_gene_indexs = [big_graph_index_dict[i] for i in nonsen_gene_indexs]

    # step 3: 识别老化细胞
    sencell_index, nonsencell_index, \
        sencell_cluster, nonsencell_cluster = identify_sencell_nonsencell(edge, att,
                                                                          sen_gene_indexs, nonsen_gene_indexs,
                                                                          cell_clusters, big_graph_index_dict, args)

    # step 3: output
    # 最后的输出结果，存储[emb，cluster, new_emb=0, big_graph_index]信息
    # new_emb用来占位，big_graph_index是cell节点在大图上的index
    sencell_dict, nonsencell_dict = get_outputs(
        sencell_index, sencell_cluster, z, big_graph_index_dict, nonsencell_index, nonsencell_cluster)

    return sencell_dict, nonsencell_dict, sen_gene_indexs, nonsen_gene_indexs
