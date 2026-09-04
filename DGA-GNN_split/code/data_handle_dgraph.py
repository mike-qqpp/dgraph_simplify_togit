import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl.data.utils import save_graphs
from sklearn.preprocessing import StandardScaler
import gc
import psutil
import os
import argparse
import pandas as pd

# 1. 建立解析器
parser = argparse.ArgumentParser(description='data_handel_mk')

# 2. 定义参数
parser.add_argument('--file_input', type=str, default='dgraphfin.npz')
parser.add_argument('--file_output', type=str, default='dgraphfin.dgldata')
args = parser.parse_args()



# 设置路径
RAW_PATH = '../../data/'
DATA_PATH = '../data/processed/'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def index_to_mask(index, num_nodes):
    """将索引数组转换为布尔掩码"""
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index] = True
    return mask

def describe(graph):
    """打印图的统计信息"""
    print(f"图节点数: {graph.num_nodes()}")
    print(f"图边数: {graph.num_edges()}")
    print(f"特征维度: {graph.ndata['feat'].shape[1]}")
    if 'label' in graph.ndata:
        labels = graph.ndata['label']
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print(f"标签分布: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    if 'trn_msk' in graph.ndata:
        print(f"训练集大小: {graph.ndata['trn_msk'].sum().item()}")
    if 'val_msk' in graph.ndata:
        print(f"验证集大小: {graph.ndata['val_msk'].sum().item()}")
    if 'tst_msk' in graph.ndata:
        print(f"测试集大小: {graph.ndata['tst_msk'].sum().item()}")

print('====================================================================')
print(args.file_input)

# 加载原始数据
data_np = np.load(RAW_PATH + args.file_input)
print([f for f in data_np.keys()])

# 提取数据
x = pd.DataFrame(data_np['x']).fillna(-1).values[:, :]
y = data_np['y']
edge_index = data_np['edge_index']
train_mask = data_np['train_mask']  # 这是索引数组
valid_mask = data_np['valid_mask']  # 这是索引数组
test_mask = data_np['test_mask']    # 这是索引数组

num_nodes = x.shape[0]

print(f"节点数: {num_nodes}")
print(f"特征形状: {x.shape}")
print(f"标签形状: {y.shape}")
print(f"训练索引长度: {len(train_mask)}")
print(f"验证索引长度: {len(valid_mask)}")
print(f"测试索引长度: {len(test_mask)}")

# 特征标准化
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# 转换为torch tensor
x_tensor = torch.FloatTensor(x_std)
y_tensor = torch.LongTensor(y)

# 将索引数组转换为布尔掩码
train_mask_tensor = index_to_mask(train_mask, num_nodes)
valid_mask_tensor = index_to_mask(valid_mask, num_nodes)
test_mask_tensor = index_to_mask(test_mask, num_nodes)

print(f"布尔掩码 - 训练: {train_mask_tensor.sum().item()}, 验证: {valid_mask_tensor.sum().item()}, 测试: {test_mask_tensor.sum().item()}")

# 构建图
src_nodes = torch.LongTensor(edge_index[0])
dst_nodes = torch.LongTensor(edge_index[1])
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

print(f"图构建完成: 节点数={graph.num_nodes()}, 边数={graph.num_edges()}")

# 添加自环并转换为双向图
graph = dgl.add_self_loop(graph)
graph = dgl.to_bidirected(graph)
graph.create_formats_()

# 添加节点特征和标签
graph.ndata['feat'] = x_tensor
graph.ndata['label'] = y_tensor
graph.ndata['trn_msk'] = train_mask_tensor
graph.ndata['val_msk'] = valid_mask_tensor
graph.ndata['tst_msk'] = test_mask_tensor

# 创建分割字典
split_dict = dict()
split_dict['trn_msk'] = train_mask_tensor
split_dict['val_msk'] = valid_mask_tensor
split_dict['tst_msk'] = test_mask_tensor
split_dict['trn_idx'] = torch.LongTensor(train_mask)
split_dict['val_idx'] = torch.LongTensor(valid_mask)
split_dict['tst_idx'] = torch.LongTensor(test_mask)

# 清理内存
del data_np
[gc.collect() for _ in range(5)]
mem_gib = psutil.Process(os.getpid()).memory_info().rss / 1024**3
print(f"当前进程物理内存: {mem_gib:.2f} GiB")

# 保存图数据
output_path = DATA_PATH + args.file_output
save_graphs(output_path, graph, split_dict)
print(f"图数据已保存到: {output_path}")
describe(graph)

print('DGraphFin数据处理完成！')
