import os
import numpy as np
import torch
import dgl
from pathlib import Path

def load_dgraphfin_data(data_path='./datasets/dgraphfin'):
    data_name = 'dgraphfin.npz'
    npz_path = os.path.join(data_path, data_name)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"DGraphFin 数据文件不存在: {npz_path}")
    npz = np.load(npz_path)

    features   = torch.from_numpy(npz['x']).float()
    labels     = torch.from_numpy(npz['y']).long()
    edge_index = torch.from_numpy(npz['edge_index']).long()

    train_mask = torch.from_numpy(npz['train_mask']).long()
    valid_mask = torch.from_numpy(npz['valid_mask']).long()
    test_mask  = torch.from_numpy(npz['test_mask']).long()

    n = features.shape[0]
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=n)

    print(f" {data_name} 数据集加载成功:")
    print(f"  节点数: {n}, 特征维度: {features.shape[1]}")
    print(f"  训练集大小: {len(train_mask)}")
    print(f"  验证集大小: {len(valid_mask)}")
    print(f"  测试集大小: {len(test_mask)}")
    print(f"  边数: {edge_index.shape[1]}")

    return graph, features, labels, train_mask, valid_mask, test_mask


class dgraph:
    def __init__(self, data_path='../data'):
        graph, features, labels, train_mask, val_mask, test_mask = load_dgraphfin_data(data_path)

        graph.ndata['train_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata['val_mask']   = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata['test_mask']  = torch.zeros(graph.num_nodes(), dtype=torch.bool)

        graph.ndata['train_mask'][train_mask] = True
        graph.ndata['val_mask'][val_mask]     = True
        graph.ndata['test_mask'][test_mask]   = True

        graph.ndata['feature'] = features
        graph.ndata['label']   = labels

        self.g = dgl.add_self_loop(graph)
