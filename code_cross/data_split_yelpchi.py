#!/usr/bin/env python
# encoding: utf-8
"""
把 elliptic.dat（pickle 版）转成 elliptic.npz
不做标准化，直接落盘
"""
import pickle
import numpy as np
import torch

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




import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.utils import shuffle
mat = loadmat('../data_init/YelpChi.mat')
print('mat 里所有变量名：', list(mat.keys()))



homos = mat['homo'].astype(bool).tocsr()
rur   = mat['net_rur'].astype(bool).tocsr()
rtr   = mat['net_rtr'].astype(bool).tocsr()
rsr   = mat['net_rsr'].astype(bool).tocsr()

# 节点特征 & 标签
x = mat['features'].toarray().astype(np.float32)      # (N, F)
y = mat['label'].ravel().astype(np.int64)   # (N,)

# ---------- 2. 构造边表 ----------
homo_coo = homos.tocoo()
src = homo_coo.row.astype(np.int64)
dst = homo_coo.col.astype(np.int64)

# 位图打标签（向量化）
hit_upu = np.asarray(rur[src, dst]).ravel()
hit_usu = np.asarray(rtr[src, dst]).ravel()
hit_uvu = np.asarray(rsr[src, dst]).ravel()
edge_type = (hit_upu.astype(np.int8) << 0) | \
            (hit_usu.astype(np.int8) << 1) | \
            (hit_uvu.astype(np.int8) << 2)          # 0-7

edge_index = np.column_stack([src, dst])            # (E, 2)

# ---------- 3. 6:2:2 随机划分 ----------
N = x.shape[0]
perm = np.random.default_rng(12345).permutation(N)
split1, split2 = int(0.6 * N), int(0.8 * N)
train_mask = perm[:split1].astype(np.int64)
valid_mask = perm[split1:split2].astype(np.int64)
test_mask  = perm[split2:].astype(np.int64)

# ---------- 4. 边属性（无时间戳填 0） ----------
E = edge_index.shape[0]
edge_timestamp = np.zeros(E, dtype=np.int64)

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
    out_file = '../data_split/yelpchi_{}.npz'.format(type_lst[i])
    np.savez(out_file,
             x=x_slc,
             y=y_slc,
             edge_index=edge_index_slc,
             edge_type=edge_type_slc,
             edge_timestamp=edge_timestamp_slc)
    
    print('✅ 已导出', out_file)
    print('shape 检查: x={}, y={}, edge_index={}, edge_type={}'.format(
        x_slc.shape, y_slc.shape, edge_index_slc.shape, edge_type_slc.shape))
