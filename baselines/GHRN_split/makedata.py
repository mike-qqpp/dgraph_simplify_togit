import numpy as np
import dgl, torch, os
from dgl.data.utils import save_graphs

# 1. 读 npz
npz = np.load('../data/dgraphfin.npz')
x        = torch.from_numpy(npz['x']).float()
y        = torch.from_numpy(npz['y']).long()
edge_idx = torch.from_numpy(npz['edge_index'].T).long()   # dgl 用 (src, dst)

# 2. 建图
g = dgl.graph((edge_idx[0], edge_idx[1]), num_nodes=x.shape[0])
g.ndata['feature'] = x
g.ndata['label'] = y
g.edata['type']  = torch.from_numpy(npz['edge_type']).long()
g.edata['timestamp'] = torch.from_numpy(npz['edge_timestamp']).long()

# 3. 保存 DGL bin
save_graphs('./datasets/dgraphfin/dgraphfin', [g])
print('dgraphfin.bin 已生成，节点数={}, 边数={}'.format(g.num_nodes(), g.num_edges()))

# 4. 追加：生成缺失的 dgraphfin_index.txt（全节点版）
index_path = './datasets/dgraphfin/dgraphfin_index.txt'
np.savetxt(index_path, np.arange(g.num_nodes()), fmt='%d')
print(f'{index_path} 已生成，共 {g.num_nodes()} 个节点索引')
