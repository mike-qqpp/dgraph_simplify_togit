#!/usr/bin/env python
# encoding: utf-8
"""
把 elliptic.dat（pickle 版）转成 elliptic.npz
不做标准化，直接落盘
"""
import pickle
import numpy as np
import torch
import gc
import psutil, os
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.utils import shuffle
from tqdm import tqdm
def extract_subgraph(edge_index, mask_slc, node_feat=None, node_label=None, 
                     node_timestamp=None, edge_type=None, edge_timestamp=None):
    """
    提取子图：过滤边、重编码节点ID、同步提取边属性
    
    Parameters:
        edge_index: (num_edges, 2) 边索引
        mask_slc: 保留的节点ID列表（保持原始顺序作为新ID）
        node_feat: (num_nodes, feat_dim) 节点特征（可选）
        node_label: (num_nodes,) 节点标签（可选）
        node_timestamp: (num_nodes,) 节点时间戳（可选）
        edge_type: (num_edges,) 边类型（可选）
        edge_attr: (num_edges, attr_dim) 边特征（可选）
    
    Returns:
        new_edge_index: (num_filtered_edges, 2) 重编码后的边
        new_node_feat: 子图节点特征（可选）
        new_node_label: 子图节点标签（可选）
        new_node_timestamp: 子图节点时间戳（可选）
        new_edge_type: 子图边类型（可选）
        new_edge_attr: 子图边特征（可选）
        node_map: 旧ID到新ID的映射字典
    """
    # ========== 步骤1：过滤边 ==========
    mask_set = set(mask_slc)
    mask = np.array([u in mask_set and v in mask_set for u, v in edge_index])
    filtered_edges = edge_index[mask]
    
    # ========== 步骤2：同步过滤边属性 ==========
    new_edge_type = None
    if edge_type is not None:
        new_edge_type = edge_type[mask]
    
    new_edge_timestamp = None
    if edge_timestamp is not None:
        new_edge_timestamp = edge_timestamp[mask]
    
    # ========== 步骤3：重编码节点ID（保持mask_slc顺序） ==========
    node_map = {old: new for new, old in enumerate(mask_slc)}
    new_edge_index = np.vectorize(node_map.get)(filtered_edges)
    
    # ========== 步骤4：提取节点相关数据 ==========
    new_node_feat = None
    if node_feat is not None:
        new_node_feat = node_feat[mask_slc]

    new_node_label = None
    if node_label is not None:
        new_node_label = node_label[mask_slc]

    new_node_timestamp = None
    if node_timestamp is not None:
        new_node_timestamp = node_timestamp[mask_slc]
    
    return (new_edge_index, new_node_feat, new_node_label, new_edge_timestamp,
            new_edge_type, node_map)



data_np = np.load('../data/dgraphfin.npz')
print([f for f in data_np.keys()])

x = data_np['x']
y = data_np['y']
train_mask = data_np['train_mask']
valid_mask = data_np['valid_mask']
test_mask = data_np['test_mask']

edge_type = data_np['edge_type']
edge_index = data_np['edge_index']
edge_timestamp = data_np['edge_timestamp']


del data_np
[gc.collect() for _ in range(5)]
mem_gib = psutil.Process(os.getpid()).memory_info().rss / 1024**3
print(f"当前进程物理内存: {mem_gib:.2f} GiB")


# ---------- 5. 导出 ----------
mask_slc_lst = [train_mask, valid_mask, test_mask]
type_lst = ['train', 'valid', 'test']

for i in tqdm(range(len(type_lst)) ):
    # mask_slc = train_mask
    
    # 使用
    edge_index_slc, x_slc, y_slc, edge_timestamp_slc, edge_type_slc, mapping = extract_subgraph(
        edge_index, mask_slc_lst[i]
        ,node_feat=x
        ,node_label=y
        ,edge_type=edge_type
        ,edge_timestamp = edge_timestamp
    )
    
    
    # ---------- 5. 导出 ----------
    out_file = '../data_split/dgraphfin_{}.npz'.format(type_lst[i])
    np.savez(out_file,
             x=x_slc,
             y=y_slc,
             edge_index=edge_index_slc,
             edge_type=edge_type_slc,
             edge_timestamp=edge_timestamp_slc)
    
    print('✅ 已导出', out_file)
    print('shape 检查: x={}, y={}, edge_index={}, edge_type={}'.format(
        x_slc.shape, y_slc.shape, edge_index_slc.shape, edge_type_slc.shape))
