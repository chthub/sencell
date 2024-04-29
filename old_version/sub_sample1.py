import numpy as np
from collections import defaultdict


def sub_sample1(graph, sampling_size,gene_size, gene_shape, cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    
    intersect_genes=None
    for cell_id in cell_indexs:
        gene_ids=set(graph.edge_list['cell']['gene']['g_c'][cell_id].keys())
        if intersect_genes is None:
            intersect_genes=gene_ids
        else:
            intersect_genes=intersect_genes.intersection(gene_ids)

    # print('After intersection, gene nums: ', len(intersect_genes))

    gene_indexs=np.random.choice(np.array(list(intersect_genes),dtype=np.int64),gene_size,replace=False)    

    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }


    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    # gene和cell的index都是从0开始的
    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs

def sub_sample2(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    
    intersect_genes=None
    for cell_id in cell_indexs:
        gene_ids=set(graph.edge_list['cell']['gene']['g_c'][cell_id].keys())
        if intersect_genes is None:
            intersect_genes=gene_ids
        else:
            intersect_genes=intersect_genes.intersection(gene_ids)

    # print('After intersection, gene nums: ', len(intersect_genes))
    
    # 随机选取
    # gene_indexs=np.random.choice(np.array(list(intersect_genes),dtype=np.int64),gene_size,replace=False)    
    # 根据取值的大小选取
    # 这个是与所有cell都有连边的gene的ids
    gene_indexs=np.array(list(intersect_genes),dtype=np.int64)
    sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape]
    # 第一个元素是value之和最大的gene的index
    _indexs=np.argsort(np.sum(sub_matrix,axis=1))[::-1]
    # 排序后的gene_indexs
    gene_indexs=gene_indexs[_indexs]
    # 选择前gene_size个
    gene_indexs=gene_indexs[:gene_size]
    # 最多的选一半，最少的也选一半
    # part1=gene_indexs[:int(gene_size/2)]
    # part2=gene_indexs[-1*int(gene_size/2):]
    # gene_indexs=np.concatenate([part1,part2])
    
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }


    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs


def norm_rowcol(matrix):
    # 按行求和
    row_norm=np.sum(matrix,axis=1).reshape(-1,1)
    # 行归一化
    matrix=matrix/row_norm
    # 按列求和
    col_norm=np.sum(matrix,axis=0)
    return matrix/col_norm



def sub_sample3(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    gene_indexs=np.arange(gene_shape)
#     intersect_genes=None
#     for cell_id in cell_indexs:
#         gene_ids=set(graph.edge_list['cell']['gene']['g_c'][cell_id].keys())
#         if intersect_genes is None:
#             intersect_genes=gene_ids
#         else:
#             intersect_genes=intersect_genes.intersection(gene_ids)

    # print('After intersection, gene nums: ', len(intersect_genes))
    
    # 随机选取
    # gene_indexs=np.random.choice(np.array(list(intersect_genes),dtype=np.int64),gene_size,replace=False)    
    # 根据取值的大小选取
    # 这个是与所有cell都有连边的gene的ids
#     gene_indexs=np.array(list(intersect_genes),dtype=np.int64)
    # print('交集大小：',gene_indexs.shape[0])
    sub_matrix=GAS[:,cell_indexs-gene_shape]
    
    # 先行标准化，再列标准化
    sub_matrix=norm_rowcol(sub_matrix)
    # 第一个元素是value之和最大的gene的index，按行求和
    _indexs=np.argsort(np.sum(sub_matrix,axis=1))[::-1]
    # 排序后的gene_indexs
    gene_indexs=gene_indexs[_indexs]
    # 选择前gene_size个
    gene_indexs=gene_indexs[:gene_size]
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }

    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs

def samplingbyProb(index_arr,value_arr,size):
    # for 0 and negative, transform to [1,inf)
    min_value=np.min(value_arr)
    if min_value<=0:
        value_arr=value_arr-min_value+1
    prob=value_arr/np.sum(value_arr)
    return np.random.choice(index_arr,size=size,replace=False,p=prob)

def sub_sample4(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    sub_matrix=GAS[:,cell_indexs-gene_shape]
    # 子矩阵很可能某一行全为0
    gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0]

    sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape]

    # 先行标准化，再列标准化
    sub_matrix=norm_rowcol(sub_matrix)
    # 第一个元素是value之和最大的gene的index，按行求和
    sum_sub_matrix=np.sum(sub_matrix,axis=1)
    gene_indexs=samplingbyProb(gene_indexs,sum_sub_matrix,gene_size)
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }

    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs



def sub_sample5(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    sub_matrix=GAS[:,cell_indexs-gene_shape]
    # 子矩阵很可能某一行全为0
    gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0]

    sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape]

    # 先行标准化，再列标准化
    sub_matrix=norm_rowcol(sub_matrix)
    
    _indexs=np.argsort(np.sum(sub_matrix,axis=1))[::-1]
    # 排序后的gene_indexs
    gene_indexs=gene_indexs[_indexs]
    # 选择前gene_size个
    gene_indexs=gene_indexs[:gene_size]
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }

    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs


def sub_sample6(graph,gene_cell_arr, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    sub_matrix=gene_cell_arr[:,cell_indexs-gene_shape]
    # 子矩阵很可能某一行全为0
    gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0]
    gene_indexs=np.random.choice(gene_indexs,gene_size)
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }

    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs


def sub_sample_sen(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    sub_matrix=GAS[:,cell_indexs-gene_shape]
    # 子矩阵很可能某一行全为0
    gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0]

    sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape]

    # 先行标准化，再列标准化
    sub_matrix=norm_rowcol(sub_matrix)
    # 第一个元素是value之和最大的gene的index，按行求和
    sum_sub_matrix=np.sum(sub_matrix,axis=1)
    gene_indexs=samplingbyProb(gene_indexs,sum_sub_matrix,gene_size)
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
    }

    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }

    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])

    return feature, times, edge_list, indxs
