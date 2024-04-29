import seaborn as sns
import umap
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from model_Sencell import get_cluster_cell_dict, getPrototypeEmb
from model_Sencell import Sencell


def plot(d1_ls, d2_ls, d3_ls, d4_ls):
    f, ax = plt.subplots(len(d1_ls), 4, figsize=(20, 10), sharex=True)
    for i, (d1, d2, d3, d4) in enumerate(zip(d1_ls, d2_ls, d3_ls, d4_ls)):
        d1 = torch.tensor(d1)
        d2 = torch.tensor(d2)
        d3 = torch.tensor(d3)
        d4 = torch.tensor(d4)

        sns.histplot(d1, alpha=0.5, color='red', label='d1', ax=ax[i, 0])
        sns.histplot(d2, alpha=0.5, color='green', label='d2', ax=ax[i, 1])
        sns.histplot(d3, alpha=0.5, color='black', label='d3', ax=ax[i, 2])
        sns.histplot(d4, alpha=0.5, color='blue', label='d4', ax=ax[i, 3])
        ax[i, 0].set_ylabel('')
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_ylabel('')
        ax[i, 3].set_ylabel('')
    ax[0, 0].set_xlim([0, 2])
    ax[0, 1].set_xlim([0, 2])
    ax[0, 2].set_xlim([0, 2])
    ax[0, 3].set_xlim([0, 2])
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    ax[0, 3].legend()
    plt.ylabel('Count')


def plot_distance(sencell_dict, nonsencell_dict, emb_pos=0):
    # 绘制4种distance的直方图
    # emb_pos=0，计算优化之前的4种距离
    # emb_pos=2，计算优化之后的4种距离
    # step 1: 计算cluster和cell的映射表
    cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(
        sencell_dict,
        nonsencell_dict)
    # step 2: 计算老化细胞簇的prototype embedding
    prototype_emb = getPrototypeEmb(sencell_dict, cluster_sencell, emb_pos)
    # step 3: 计算不同的distance
    model = Sencell()
    d1, d2, d3, d4 = model.caculateDistance(sencell_dict, nonsencell_dict,
                                            cluster_sencell, cluster_nonsencell,
                                            prototype_emb, emb_pos)

    plot(d1, d2, d3, d4)


def plot_umap1(sencell_dict, nonsencell_dict, cluster_names, emb_pos=0):
    # 这个umap画的是包含采样的到的子图上所有节点的UMAP
    # 需要用到cluster_names
    cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(
        sencell_dict,
        nonsencell_dict)
    sencell_embs = []
    for cluster, cell_indexs in cluster_sencell.items():
        for cell_index in cell_indexs:
            sencell_embs.append(sencell_dict[cell_index][emb_pos].view(1, -1))
    sencell_embs = torch.cat(sencell_embs)

    nonsencell_embs = []
    cluster_cell_indexs = []
    count = 0
    cluster_indexs = []
    for cluster, cell_indexs in cluster_nonsencell.items():
        cluster_indexs.append(cluster)
        cluster_cell_index = []
        for cell_index in cell_indexs:
            nonsencell_embs.append(
                nonsencell_dict[cell_index][emb_pos].view(1, -1))
            cluster_cell_index.append(count)
            count += 1
        cluster_cell_indexs.append(np.array(cluster_cell_index))
    nonsencell_embs = torch.cat(nonsencell_embs)

    embeddings = torch.cat([sencell_embs, nonsencell_embs], 0)

    reducer = umap.UMAP(n_neighbors=100)
    embeddings = reducer.fit_transform(embeddings)
    print(embeddings.shape)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(embeddings[:sencell_embs.shape[0], 0],
                embeddings[:sencell_embs.shape[0], 1], s=5, label='sencell', c='red')
    for cell_indexs, cluster_index in zip(cluster_cell_indexs, cluster_indexs):
        cell_embeddings = embeddings[cell_indexs+sencell_embs.shape[0], :]
        plt.scatter(cell_embeddings[:, 0],
                    cell_embeddings[:, 1],
                    alpha=0.1, s=2, label=cluster_names[cluster_index])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def plot_umap2(adata, sencell_dict, nonsencell_dict, gene_num,
               cell_cluster_indexs,
               emb_pos=0):

    celltype_names = list(adata.obs['ann_level_3_pred'].value_counts().index)
    # cell_cluster_indexs 和 celltype_names
    clusters = cell_cluster_indexs
    labels = celltype_names

    plt.figure(figsize=(6, 6), dpi=300)
    cmap1 = matplotlib.cm.get_cmap('tab20')
    cmap2 = matplotlib.cm.get_cmap('Set3')
    color_ls = cmap1.colors+cmap2.colors

    # plot all cell type
    for i, (cluster, label) in enumerate(zip(clusters, labels)):
        # if i==18:
        plt.scatter(adata.obsm['X_umap'][cluster, 0],
                    adata.obsm['X_umap'][cluster, 1],
                    color=color_ls[i],
                    alpha=1, s=1,
                    label=label)
        # else:
        #     plt.scatter(adata.obsm['X_umap'][cluster,0],
        #     adata.obsm['X_umap'][cluster,1],
        #     color='grey',
        #     alpha=1,s=1,
        #     label=label)
    # plot sencell
    sencell_indexs = np.array([value[3]
                              for key, value in sencell_dict.items()])-gene_num
    nonsencell_indexs = np.array(
        [value[3] for key, value in nonsencell_dict.items()])-gene_num
    plt.scatter(adata.obsm['X_umap'][sencell_indexs, 0],
                adata.obsm['X_umap'][sencell_indexs, 1],
                s=10, label='sencell', c='black', marker='x')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def umapPlot(embedding, cell_index_ls, clusters=None, reduce=False, labels=None):
    # if tensor: embedding should be .cpu().detach()
    # clusters: Nxt
    # t里面存的是行的index
    if reduce:
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(embedding)

    plt.figure(figsize=(6, 6), dpi=300)
    cmap1 = matplotlib.cm.get_cmap('tab20')
    cmap2 = matplotlib.cm.get_cmap('Set3')
    color_ls = cmap1.colors+cmap2.colors
    if clusters is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=5)
    else:
        for i, (cluster, label) in enumerate(zip(clusters, labels)):
            plt.scatter(embedding[cluster, 0], embedding[cluster, 1],
                        alpha=0.4, s=5, color=color_ls[i], label=label)

        plt.scatter(embedding[cell_index_ls, 0], embedding[cell_index_ls,
                    1], s=5, color='black', marker='x', label='sencell')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def subUmapPlot(embedding, cluster_cell_dict, clusters=None, reduce=False, labels=None):
    # if tensor: embedding should be .cpu().detach()
    # clusters: Nxt
    # t里面存的是行的index
    if reduce:
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(embedding)

    fig, axes = plt.subplots(
        5, 5, sharex=False, sharey=False, figsize=(6, 6), dpi=300)

    cmap1 = matplotlib.cm.get_cmap('tab20')
    cmap2 = matplotlib.cm.get_cmap('Set3')
    color_ls = cmap1.colors+cmap2.colors
    if clusters is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=5)
    else:
        x_s = []
        y_s = []
        label_s = []
        for cluster, label in zip(clusters, labels):
            x_s.append(embedding[cluster, 0])
            y_s.append(embedding[cluster, 1])
            label_s.append(label)
        count = 0
        for i, row in enumerate(axes):
            for j, col in enumerate(row):
                if count < 21:
                    col.scatter(x_s[count], y_s[count],
                                alpha=0.5, color=color_ls[count], s=5)
                    sencell_num = 0
                    if count in cluster_cell_dict:
                        # 这一簇有老化细胞
                        col.scatter(embedding[cluster_cell_dict[count], 0], embedding[cluster_cell_dict[count], 1],
                                    s=3, color='black', marker='x', label='sencell')
                        sencell_num = len(cluster_cell_dict[count])

                    col.set_title(
                        f"{label_s[count]} ({sencell_num}/{len(clusters[count])})", fontsize=5)
                    count += 1
                else:
                    col.set_visible(False)
    plt.setp(fig.axes, yticks=[], xticks=[])

    plt.tight_layout()
