#!/usr/bin/env python
# encoding: utf-8
"""lliptic.dat to lliptic.npz
No standardization, no drive."""
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
    """Extract subcharts: filter edges, heavy-coding nodes ID, sync to extract edge properties

Parameters:
Edge index: (num edges, 2) border index
Mask slc: Retain the node ID list (maintain original order as new ID)
Node feat: (num nodes, feat dim) node features (optional)
Node label: (num nodes,) Node tab (optional)
Node timestamp: (num nodes,) Node Timestamp (optional)
Edge type: (num edges,) Edgetype (optional)
Edge attr: (num edges, attr dim) edge features (optional)

Returns:
new edge index: (num filed edges, 2) Recoded Edge
new node feat: subchart nodes feature (optional)
new node label: subchart node tag (optional)
new node timestamp: subchart nodes timet (optional)
new edge type: Subchart Sidetype (optional)
new edge attr: sub-chart side feature (optional)
Node map: old ID to new ID map dictionary"""
    # == sync, corrected by elderman == @elder man
    mask_set = set(mask_slc)
    mask = np.array([u in mask_set and v in mask_set for u, v in edge_index])
    filtered_edges = edge_index[mask]
    
    # == sync, corrected by elderman == @elder man
    new_edge_type = None
    if edge_type is not None:
        new_edge_type = edge_type[mask]
    
    new_edge_timestamp = None
    if edge_timestamp is not None:
        new_edge_timestamp = edge_timestamp[mask]
    
    # == sync, corrected by elderman == @elder man
    node_map = {old: new for new, old in enumerate(mask_slc)}
    new_edge_index = np.vectorize(node_map.get)(filtered_edges)
    
    # == sync, corrected by elderman == @elder man
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
import torch
from dgl.data.utils import load_graphs
import argparse

# 1. Reading maps
g, _ = load_graphs('../data_init/tfinance')
g = g[0]
print(g)

# 2. Extracting existing fields
x = g.ndata['feature'].numpy()          # (N, 10)
y = g.ndata['label'].numpy()
if y.ndim > 1:                          # If one-hot turns 0/1
    y = y.argmax(1)

# Side Table - (E, 2)
edge_index = torch.stack(g.edges(), dim=1).numpy().astype(np.int64)

# 4. Mask: Index List Format, 6:2:2 Randomly divided
N = g.num_nodes()
perm = np.random.permutation(N)
split1, split2 = int(0.6 * N), int(0.8 * N)
train_mask = perm[:split1].astype(np.int64)
valid_mask   = perm[split1:split2].astype(np.int64)
test_mask  = perm[split2:].astype(np.int64)

# 5. Edge properties (n.o. 0)
E = edge_index.shape[0]
edge_type      = np.zeros(E, dtype=np.int64)
edge_timestamp = np.zeros(E, dtype=np.int64)


# 5. Export -
mask_slc_lst = [train_mask, valid_mask, test_mask]
type_lst = ['train', 'valid', 'test']

for i in tqdm(range(len(type_lst)) ):
    # mask_slc = train_mask
    
    # Use
    edge_index_slc, x_slc, y_slc, edge_timestamp_slc, edge_type_slc, mapping = extract_subgraph(
        edge_index, mask_slc_lst[i]
        ,node_feat=x
        ,node_label=y
        ,edge_type=edge_type
        ,edge_timestamp = edge_timestamp
    )
    
    
    # 5. Export -
    out_file = '../data_split/tfinance_{}.npz'.format(type_lst[i])
    np.savez(out_file,
             x=x_slc,
             y=y_slc,
             edge_index=edge_index_slc,
             edge_type=edge_type_slc,
             edge_timestamp=edge_timestamp_slc)
    
    print('✅ Exported', out_file)
    print('Shape check: x={, y=}, edge index=}, edge type=}'.format(
        x_slc.shape, y_slc.shape, edge_index_slc.shape, edge_type_slc.shape))
