# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):
    """
    Handle S-FFSD dataset and do some feature engineering
    :param tmp_df: the feature of input dataset
    """
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows(), total=len(tmp_df), desc="Feature Engineering"):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """return indices of risk k-hop neighbors

    Args:
        graph (dgl.DGLGraph): dgl graph dataset
        center_idx (int): center node idx
        k (int): k-hop neighs
        where (str): {"predecessor", "successor"}
        risk_label (int, optional): value of fruad label. Defaults to 1.
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # delete center node itself
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_out.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata['label'][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:

    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map(graph, edge_feat):
    """
    生成邻居特征
    """
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes()), desc="Generating neighborhood features"):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree",
                  "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


def process_group_batch(group_data, edge_per_trans=3):
    """
    处理单个组的批处理版本 - 与原始代码逻辑完全相同
    """
    if len(group_data) <= 1:
        return [], []
    
    # 按时间排序（和原始代码一样）
    sorted_group = group_data.sort_values('Time')
    sorted_indices = sorted_group.index.values
    n = len(sorted_indices)
    
    # 确定窗口大小（和原始代码一样：edge_per_trans=3）
    window_size = min(edge_per_trans, n)
    
    # 预计算总边数
    total_edges = sum(n - j for j in range(1, window_size))
    
    if total_edges == 0:
        return [], []
    
    # 预分配数组（更高效）
    src_list = np.empty(total_edges, dtype=int)
    tgt_list = np.empty(total_edges, dtype=int)
    
    idx = 0
    for j in range(1, window_size):
        n_edges = n - j
        if n_edges <= 0:
            break
            
        # 向量化操作：一次处理所有i
        src_list[idx:idx+n_edges] = sorted_indices[:-j]
        tgt_list[idx:idx+n_edges] = sorted_indices[j:]
        idx += n_edges
    
    return src_list[:idx].tolist(), tgt_list[:idx].tolist()


def process_batch(batch_groups, edge_per_trans=3):
    """
    处理一个批次的组
    """
    batch_src = []
    batch_tgt = []
    
    for group_id, group in batch_groups:
        src, tgt = process_group_batch(group, edge_per_trans)
        if src:
            batch_src.extend(src)
            batch_tgt.extend(tgt)
    
    return batch_src, batch_tgt


def build_edges_threaded_batch(data, pair_columns, edge_per_trans=3, batch_size=50, max_workers=None):
    """
    使用多线程批处理构建边 - 功能与原始代码完全相同
    """
    if max_workers is None:
        max_workers = min(32, mp.cpu_count() * 2)  # I/O密集型，可以更多线程
    
    all_src = []
    all_tgt = []
    
    for column in pair_columns:
        print(f"Processing column: {column} (using {max_workers} threads, batch_size={batch_size})")
        
        # 获取所有分组（和原始代码一样）
        groups = list(data.groupby(column))
        total_groups = len(groups)
        print(f"Total groups to process: {total_groups}")
        
        # 准备批次
        batches = []
        for i in range(0, total_groups, batch_size):
            batch = groups[i:i + batch_size]
            batches.append(batch)
        
        total_batches = len(batches)
        print(f"Split into {total_batches} batches")
        
        # 使用ThreadPoolExecutor进行多线程处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {}
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(process_batch, batch, edge_per_trans)
                future_to_batch[future] = batch_idx
            
            # 收集结果并显示进度
            completed_batches = 0
            for future in tqdm(as_completed(future_to_batch), total=total_batches, desc=f"Processing {column} batches"):
                batch_idx = future_to_batch[future]
                try:
                    batch_src, batch_tgt = future.result()
                    if batch_src:
                        all_src.extend(batch_src)
                        all_tgt.extend(batch_tgt)
                    
                    completed_batches += 1
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
        
        print(f"Generated {len([x for x in all_src if x is not None])} edges for column {column}")
    
    return np.array(all_src), np.array(all_tgt)


def build_edges_threaded_vectorized(data, pair_columns, edge_per_trans=3, max_workers=None):
    """
    向量化多线程版本 - 最优化版本
    """
    if max_workers is None:
        max_workers = min(32, mp.cpu_count() * 2)
    
    all_src = []
    all_tgt = []
    
    for column in pair_columns:
        print(f"Processing column: {column} with vectorized threading (using {max_workers} threads)")
        
        # 获取分组
        groups = list(data.groupby(column))
        total_groups = len(groups)
        print(f"Total groups: {total_groups}")
        
        # 使用多线程处理所有组
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = []
            for _, group in groups:
                future = executor.submit(process_group_batch, group, edge_per_trans)
                futures.append(future)
            
            # 收集结果
            results = []
            for future in tqdm(as_completed(futures), total=total_groups, desc=f"Processing {column}"):
                try:
                    src, tgt = future.result()
                    if src:
                        results.append((src, tgt))
                except Exception as e:
                    print(f"Error processing group: {e}")
        
        # 合并结果
        for src, tgt in results:
            all_src.extend(src)
            all_tgt.extend(tgt)
        
        print(f"Generated {len(all_src)} edges for column {column}")
    
    return np.array(all_src), np.array(all_tgt)


def encode_columns_parallel(data, columns, max_workers=None):
    """并行编码多个列"""
    if max_workers is None:
        max_workers = len(columns)
    
    encoded_data = data.copy()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for col in columns:
            future = executor.submit(encode_single_column, data[col])
            futures[future] = col
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="Encoding columns"):
            col = futures[future]
            try:
                encoded_data[col] = future.result()
            except Exception as e:
                print(f"Error encoding column {col}: {e}")
    
    return encoded_data


def encode_single_column(column_data):
    """编码单个列"""
    le = LabelEncoder()
    return le.fit_transform(column_data.astype(str).values)


def process_sffsd_dataset_threaded():
    """处理S-FFSD数据集的线程优化版本 - 功能与原始代码完全相同"""
    print(f"Processing S-FFSD data (threaded optimized version)...")
    
    # 读取数据（和原始代码一样）
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    
    # 特征工程（和原始代码一样）
    print("Performing feature engineering...")
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    
    # 保存中间结果（和原始代码一样）
    intermediate_file = os.path.join(DATADIR, 'S-FFSDneofull.csv')
    data.to_csv(intermediate_file, index=None)
    print(f"Intermediate data saved to {intermediate_file}")
    
    # 重新加载数据（和原始代码一样）
    data = pd.read_csv(intermediate_file)
    data = data.reset_index(drop=True)
    
    print(f"Dataset loaded: {len(data)} rows, {len(data.columns)} columns")
    print(f"CPU cores available: {mp.cpu_count()}")
    
    # 构建边 - 使用多线程版本（功能与原始代码相同）
    pair = ["Source", "Target", "Location", "Type"]  # 和原始代码一样
    edge_per_trans = 3  # 和原始代码一样
    
    print(f"\nBuilding edges with edge_per_trans={edge_per_trans}...")
    start_time = time.time()
    
    # 方法1：批处理多线程（适合大量小分组）
    # print("Using threaded batch processing...")
    # alls, allt = build_edges_threaded_batch(data, pair, edge_per_trans, batch_size=100)
    
    # 方法2：向量化多线程（适合大分组）
    print("Using vectorized threading...")
    alls, allt = build_edges_threaded_vectorized(data, pair, edge_per_trans)
    
    elapsed_time = time.time() - start_time
    print(f"Edge building completed in {elapsed_time:.2f} seconds")
    print(f"Total edges generated: {len(alls)}")
    
    # 创建图（和原始代码一样）
    g = dgl.graph((alls, allt))
    print(f"Graph created with {g.num_nodes()} nodes and {g.num_edges()} edges")
    
    # 编码分类特征（和原始代码一样）- 使用并行编码
    print("Encoding categorical features in parallel...")
    cal_list = ["Source", "Target", "Location", "Type"]  # 和原始代码一样
    data = encode_columns_parallel(data, cal_list)
    
    # 准备特征和标签（和原始代码一样）
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    
    # 设置节点特征（和原始代码一样）
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    
    # 保存图（和原始代码一样）
    output_file = DATADIR + "graph-S-FFSD.bin"
    dgl.data.utils.save_graphs(output_file, [g])
    print(f"Graph saved to {output_file}")
    
    return g


def generate_neighbor_features():
    """生成邻居特征"""
    print("\n" + "="*50)
    print("Generating neighbor risk-aware features...")
    print("="*50)
    
    # 只处理S-FFSD，因为其他数据集注释了
    for file_name in ['S-FFSD']:  # 只保留S-FFSD
        print(f"\nProcessing {file_name} dataset...")
        
        # 加载图
        graph_file = DATADIR + "graph-" + file_name + ".bin"
        if not os.path.exists(graph_file):
            print(f"Graph file {graph_file} not found, skipping...")
            continue
            
        graph = dgl.load_graphs(graph_file)[0][0]
        print(f"Graph info: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        
        # 计算度特征和风险特征
        print("Calculating degree and risk features...")
        degree_feat = graph.in_degrees().unsqueeze_(1).float()
        risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()
        
        edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
        
        # 生成邻居特征
        print("Generating neighborhood features...")
        features_neigh, feat_names = feat_map(graph, edge_feat)
        
        # 合并特征
        features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
        origin_feat_name = ['degree', 'riskstat']
        feat_names = origin_feat_name + feat_names
        
        # 处理NaN值
        features_neigh[np.isnan(features_neigh)] = 0.
        
        # 标准化
        print("Standardizing features...")
        scaler = StandardScaler()
        features_neigh = scaler.fit_transform(features_neigh)
        
        # 保存特征
        output_path = DATADIR + file_name + "_neigh_feat.csv"
        features_df = pd.DataFrame(features_neigh, columns=feat_names)
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        print(f"Feature shape: {features_df.shape}")
    
    print("\n" + "="*50)
    print("All processing completed!")
    print("="*50)


if __name__ == "__main__":

    set_seed(42)
    
    '''
    # %%
    """
        For Yelpchi dataset
        Code partially from https://github.com/YingtongDou/CARE-GNN
    """
    print(f"Processing YELP data...")
    yelp = loadmat(os.path.join(DATADIR, 'YelpChi.mat'))
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    yelp_homo = yelp['homo']

    sparse_to_adjlist(net_rur, os.path.join(
        DATADIR, "yelp_rur_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_rtr_adjlists.pickle"))
    sparse_to_adjlist(net_rsr, os.path.join(
        DATADIR, "yelp_rsr_adjlists.pickle"))
    sparse_to_adjlist(yelp_homo, os.path.join(
        DATADIR, "yelp_homo_adjlists.pickle"))

    data_file = yelp
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    # load the preprocessed adj_lists
    with open(os.path.join(DATADIR, "yelp_homo_adjlists.pickle"), 'rb') as file:
        homo = pickle.load(file)
    file.close()
    
    # 构建图
    print("Building Yelp graph...")
    src = []
    tgt = []
    for i in tqdm(homo, desc="Processing adjacency lists"):
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    
    # 保存图
    dgl.data.utils.save_graphs(DATADIR + "graph-yelp.bin", [g])
    print(f"Yelp graph saved with {g.num_nodes()} nodes and {g.num_edges()} edges")

    # %%
    """
        For Amazon dataset
    """
    print(f"Processing AMAZON data...")
    amz = loadmat(os.path.join(DATADIR, 'Amazon.mat'))
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    amz_homo = amz['homo']

    sparse_to_adjlist(net_upu, os.path.join(
        DATADIR, "amz_upu_adjlists.pickle"))
    sparse_to_adjlist(net_usu, os.path.join(
        DATADIR, "amz_usu_adjlists.pickle"))
    sparse_to_adjlist(net_uvu, os.path.join(
        DATADIR, "amz_uvu_adjlists.pickle"))
    sparse_to_adjlist(amz_homo, os.path.join(
        DATADIR, "amz_homo_adjlists.pickle"))

    data_file = amz
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    
    # load the preprocessed adj_lists
    with open(DATADIR + 'amz_homo_adjlists.pickle', 'rb') as file:
        homo = pickle.load(file)
    file.close()
    
    print("Building Amazon graph...")
    src = []
    tgt = []
    for i in tqdm(homo, desc="Processing adjacency lists"):
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    
    # 保存图
    dgl.data.utils.save_graphs(DATADIR + "graph-amazon.bin", [g])
    print(f"Amazon graph saved with {g.num_nodes()} nodes and {g.num_edges()} edges")
    '''

    # %%
    """
        For S-FFSD dataset - 使用多线程批处理优化版本
    """
    sffsd_graph = process_sffsd_dataset_threaded()

    # %%
    """
        生成邻居风险感知特征
    """
    generate_neighbor_features()
