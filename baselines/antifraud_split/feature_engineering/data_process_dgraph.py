import pickle
import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from functools import partial
import time
import warnings
import random
import dgl
import math
import threading
import gc
warnings.filterwarnings('ignore')


class ProgressPrinter:
    """线程安全的进度打印器"""
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def print_progress(self, current, total, prefix="", bar_width=50):
        """打印进度条"""
        elapsed = time.time() - self.start_time
        percent = current / total
        filled = int(bar_width * percent)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # 估算剩余时间
        if current > 0:
            eta = elapsed / current * (total - current)
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "Unknown"
        
        with self.lock:
            print(f"\r{prefix} |{bar}| {percent*100:5.1f}% "
                  f"[{current}/{total}] ETA: {eta_str}", end="", flush=True)


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def k_neighs_sampled(graph_data, center_idx, k, where, n_samples=None):
    """采样版本的k_neighs函数，返回采样的邻居索引"""
    graph, labels = graph_data
    
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)
    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_out.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
    
    if n_samples is not None and len(neigh_idxs) > n_samples:
        neigh_list = neigh_idxs.tolist()
        sampled_neighs = random.sample(neigh_list, n_samples)
        return torch.tensor(sampled_neighs, dtype=torch.long)
    
    return neigh_idxs


def batch_count_risk_neighs_vectorized(graph_data, batch_indices, progress_queue=None, task_id=0):
    """
    优化的批量风险邻居计数 - 使用向量化操作
    
    Args:
        graph_data: (graph, labels) 元组
        batch_indices: 节点索引列表
        progress_queue: 进度队列（用于报告进度）
        task_id: 任务ID（用于标识进度）
    """
    graph, labels = graph_data
    n_nodes = len(batch_indices)
    
    # 预分配结果数组
    risk_counts = np.zeros(n_nodes, dtype=np.int32)
    
    # 使用向量化操作批量获取邻居
    for i, idx in enumerate(batch_indices):
        # 获取所有直接邻居
        neigh_idxs = graph.successors(idx)
        if len(neigh_idxs) > 0:
            # 向量化获取标签和计数
            neigh_labels = labels[neigh_idxs]
            risk_counts[i] = (neigh_labels == 1).sum().item()
        
        # 报告进度（每100个节点报告一次）
        if progress_queue is not None and i % 100 == 0:
            progress_queue.put((task_id, i + 1, n_nodes))
    
    return risk_counts


def batch_feat_map_worker_optimized(graph_data, edge_feat_np, batch_indices, 
                                    n_1hop_samples, n_2hop_samples,
                                    progress_queue=None, task_id=0):
    """
    优化的批量特征映射工作函数 - 使用向量化操作
    
    Args:
        graph_data: (graph, labels) 元组
        edge_feat_np: 边特征numpy数组
        batch_indices: 节点索引列表
        n_1hop_samples: 1-hop采样数
        n_2hop_samples: 2-hop采样数
        progress_queue: 进度队列
        task_id: 任务ID
    """
    graph, labels = graph_data
    n_nodes = len(batch_indices)
    
    # 转换为张量
    edge_feat_tensor = torch.from_numpy(edge_feat_np)
    
    # 预分配结果数组
    batch_features = np.zeros((n_nodes, 4), dtype=np.float32)
    
    for i, idx in enumerate(batch_indices):
        try:
            # 获取1-hop和2-hop邻居
            neighs_1 = k_neighs_sampled((graph, labels), idx, 1, "in", n_samples=n_1hop_samples)
            neighs_2 = k_neighs_sampled((graph, labels), idx, 2, "in", n_samples=n_2hop_samples)
            
            # 向量化计算特征
            if len(neighs_1) > 0:
                batch_features[i, 0] = edge_feat_tensor[neighs_1, 0].sum().item()
                batch_features[i, 2] = edge_feat_tensor[neighs_1, 1].sum().item()
            
            if len(neighs_2) > 0:
                batch_features[i, 1] = edge_feat_tensor[neighs_2, 0].sum().item()
                batch_features[i, 3] = edge_feat_tensor[neighs_2, 1].sum().item()
                
        except Exception as e:
            # 出错时保持零向量
            pass
        
        # 报告进度
        if progress_queue is not None and i % 50 == 0:
            progress_queue.put((task_id, i + 1, n_nodes))
    
    return batch_features


def batch_compute_neighbor_features_fast(graph, labels, edge_feat, batch_indices,
                                         n_1hop_samples=100, n_2hop_samples=50,
                                         neighbor_cache=None, progress_queue=None, task_id=0):
    """
    快速批量计算邻居特征 - 使用预构建邻接缓存
    
    Args:
        graph: DGL图
        labels: 标签张量
        edge_feat: 边特征numpy数组 (num_nodes, 2)
        batch_indices: 节点索引列表
        n_1hop_samples: 1-hop采样数
        n_2hop_samples: 2-hop采样数
        neighbor_cache: 预构建的邻居缓存字典
        progress_queue: 进度队列
        task_id: 任务ID
    """
    n_nodes = len(batch_indices)
    results = np.zeros((n_nodes, 4), dtype=np.float32)
    edge_feat_tensor = torch.from_numpy(edge_feat)
    
    for i, idx in enumerate(batch_indices):
        try:
            # 获取1-hop邻居（从缓存）
            neighs_1 = neighbor_cache.get(idx, [])
            if len(neighs_1) > n_1hop_samples:
                neighs_1 = random.sample(neighs_1, n_1hop_samples)
            
            # 计算1-hop特征
            if neighs_1:
                neighs_1_tensor = torch.tensor(neighs_1, dtype=torch.long)
                results[i, 0] = edge_feat_tensor[neighs_1_tensor, 0].sum().item()
                results[i, 2] = edge_feat_tensor[neighs_1_tensor, 1].sum().item()
            
            # 获取2-hop邻居（使用集合避免重复和自身）
            neighs_2_set = set()
            for n1 in neighs_1:
                neighs_2_set.update(neighbor_cache.get(n1, []))
            
            # 移除中心节点和1-hop邻居
            center_and_1hop = {idx} | set(neighs_1)
            neighs_2_set = neighs_2_set - center_and_1hop
            
            neighs_2 = list(neighs_2_set)
            if len(neighs_2) > n_2hop_samples:
                neighs_2 = random.sample(neighs_2, n_2hop_samples)
            
            if neighs_2:
                neighs_2_tensor = torch.tensor(neighs_2, dtype=torch.long)
                results[i, 1] = edge_feat_tensor[neighs_2_tensor, 0].sum().item()
                results[i, 3] = edge_feat_tensor[neighs_2_tensor, 1].sum().item()
                
        except Exception as e:
            pass
        
        # 报告进度
        if progress_queue is not None and i % 100 == 0:
            progress_queue.put((task_id, i + 1, n_nodes))
    
    return results


def build_neighbor_cache_from_adjlists(adj_lists, num_nodes):
    """
    从邻接列表构建邻居查找缓存
    
    Args:
        adj_lists: 邻接列表字典 (从稀疏矩阵构建的)
        num_nodes: 节点总数
    """
    print("  构建邻居查找缓存...")
    
    # adj_lists 已经是 {node: {neighbors}} 的格式
    # 转换为 {node: [neighbors]} 格式用于快速查找
    neighbor_dict = {}
    for node, neighbors in tqdm(adj_lists.items(), desc="构建缓存"):
        neighbor_dict[node] = list(neighbors)
    
    # 对于没有邻居的节点，添加空列表
    for i in range(num_nodes):
        if i not in neighbor_dict:
            neighbor_dict[i] = []
    
    print(f"  邻居缓存构建完成，包含 {len(neighbor_dict)} 个节点的邻居信息")
    return neighbor_dict


def build_neighbor_cache_from_edges(graph, num_nodes):
    """
    从图的边信息构建邻居查找缓存（不使用稀疏矩阵）
    
    Args:
        graph: DGL图
        num_nodes: 节点总数
    """
    print("  构建邻居查找缓存...")
    
    # 使用 edges() 方法获取源节点和目标节点
    src, dst = graph.edges()
    
    # 构建邻居字典
    neighbor_dict = {}
    for i in tqdm(range(len(src)), desc="构建缓存"):
        src_node = src[i].item()
        dst_node = dst[i].item()
        
        # dst_node 的入邻居是 src_node
        if dst_node not in neighbor_dict:
            neighbor_dict[dst_node] = []
        neighbor_dict[dst_node].append(src_node)
    
    # 对于没有邻居的节点，添加空列表
    all_nodes = set(range(num_nodes))
    nodes_with_neighbors = set(neighbor_dict.keys())
    nodes_without_neighbors = all_nodes - nodes_with_neighbors
    
    for node in nodes_without_neighbors:
        neighbor_dict[node] = []
    
    print(f"  邻居缓存构建完成，包含 {len(neighbor_dict)} 个节点的邻居信息")
    return neighbor_dict


def count_risk_neighs_gpu_optimized(graph, labels, batch_indices, device=None):
    """
    GPU优化的风险邻居计数 - 使用向量化矩阵操作
    
    Args:
        graph: DGL图
        labels: 标签张量（需要在CPU上）
        batch_indices: 节点索引列表
        device: 计算设备
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    n_nodes = len(batch_indices)
    
    # 确保labels在CPU上（因为graph.successors返回CPU张量）
    if isinstance(labels, torch.Tensor):
        labels_cpu = labels.cpu()
    else:
        labels_cpu = torch.tensor(labels, dtype=torch.long).cpu()
    
    # 获取批量邻居索引和标签
    all_neighbors = []
    neighbor_counts = []
    
    for idx in batch_indices:
        neigh = graph.successors(idx)
        all_neighbors.append(neigh)
        neighbor_counts.append(len(neigh))
    
    # 找到最大邻居数
    max_neighbors = max(neighbor_counts) if neighbor_counts else 0
    
    # 创建批量化张量（保持在CPU上以避免设备不匹配）
    if max_neighbors > 0:
        # 填充张量（在CPU上）
        batch_neigh_tensor = torch.zeros(n_nodes, max_neighbors, dtype=torch.long)
        mask_tensor = torch.zeros(n_nodes, max_neighbors, dtype=torch.bool)
        
        for i, neigh in enumerate(all_neighbors):
            n_len = len(neigh)
            if n_len > 0:
                batch_neigh_tensor[i, :n_len] = neigh
                mask_tensor[i, :n_len] = True
        
        # 批量获取标签（在CPU上完成索引）
        batch_labels = torch.zeros(n_nodes, max_neighbors, dtype=torch.long)
        batch_labels[mask_tensor] = labels_cpu[batch_neigh_tensor[mask_tensor]]
        
        # 批量计算风险邻居
        risk_counts = (batch_labels == 1).sum(dim=1)
        
        return risk_counts.numpy()
    else:
        return np.zeros(n_nodes, dtype=np.int32)


def create_dgraphfin_dataset(data_path='./datasets/dgraphfin', output_dir='../data', 
                           n_1hop_samples=100, n_2hop_samples=100,
                           use_gpu=False, batch_size=10000, num_workers=None):
    """
    将DGraphFin数据集转换为与Yelp/Amazon/S-FFSD相同格式的数据集
    使用采样和多线程优化大规模图处理
    
    Args:
        data_path: 数据集路径
        output_dir: 输出目录
        n_1hop_samples: 1-hop邻居采样数
        n_2hop_samples: 2-hop邻居采样数
        use_gpu: 是否使用GPU加速
        batch_size: 批次大小
        num_workers: 工作进程数（默认自动检测）
    """
    set_seed(42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检测设备和进程数
    use_cuda = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())
    
    print(f"计算设备: {device}")
    print(f"工作进程数: {num_workers}")
    
    # 加载原始DGraphFin数据
    npz_path = os.path.join(data_path, 'dgraphfin.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"DGraphFin数据文件不存在: {npz_path}")
    
    npz = np.load(npz_path)
    
    # 提取原始数据
    features = torch.from_numpy(npz['x']).float()
    labels = torch.from_numpy(npz['y']).long()
    edge_index = torch.from_numpy(npz['edge_index']).long()
    
    # 提取掩码索引
    train_mask = torch.from_numpy(npz['train_mask']).bool()
    valid_mask = torch.from_numpy(npz['valid_mask']).bool()
    test_mask = torch.from_numpy(npz['test_mask']).bool()
    
    # 计算变量
    num_nodes = features.shape[0]
    num_features = features.shape[1]
    num_edges = edge_index.shape[1]
    
    # 计算各数据集大小（使用sum而不是len，因为mask是布尔类型）
    train_size = train_mask.sum().item()
    valid_size = valid_mask.sum().item()
    test_size = test_mask.sum().item()
    
    print(f"\n原始DGraphFin数据集信息:")
    print(f"  节点数: {num_nodes}")
    print(f"  特征维度: {num_features}")
    print(f"  边数: {num_edges}")
    print(f"  训练集大小: {train_size}")
    print(f"  验证集大小: {valid_size}")
    print(f"  测试集大小: {test_size}")
    print(f"  训练集大小: {train_size}")
    print(f"  验证集大小: {valid_size}")
    print(f"  测试集大小: {test_size}")
    print(f"  1-hop采样数: {n_1hop_samples}")
    print(f"  2-hop采样数: {n_2hop_samples}")
    
    # 创建DGL图
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    
    # 添加节点特征和标签
    graph.ndata['feat'] = features
    graph.ndata['label'] = labels
    
    # 准备数据
    indices = list(range(num_nodes))
    labels_tensor = graph.ndata['label']
    
    # ========================================
    # 步骤1: 创建邻接列表
    # ========================================
    print(f"\n步骤1: 创建邻接列表...")
    
    # 使用稀疏矩阵格式更高效
    adj_matrix = sp.csr_matrix(
        (np.ones(num_edges), (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(num_nodes, num_nodes)
    )
    adj_matrix = adj_matrix + sp.eye(num_nodes, format='csr')
    
    print("  转换为邻接列表...")
    adj_lists = defaultdict(set)
    rows, cols = adj_matrix.nonzero()
    
    # 使用tqdm显示进度
    for i in tqdm(range(len(rows)), desc="构建邻接表"):
        adj_lists[rows[i]].add(cols[i])
    
    # 保存邻接列表
    adjlist_path = os.path.join(output_dir, "dgraphfin_homo_adjlists.pickle")
    with open(adjlist_path, 'wb') as file:
        pickle.dump(dict(adj_lists), file)
    print(f"  邻接列表已保存到: {adjlist_path}")
    
    # ========================================
    # 步骤2: 保存DGL图
    # ========================================
    print(f"\n步骤2: 保存DGL图...")
    graph_path = os.path.join(output_dir, "graph-dgraphfin.bin")
    dgl.save_graphs(graph_path, [graph])
    print(f"  DGL图已保存到: {graph_path}")
    
    # ========================================
    # 步骤3: 生成邻居风险感知特征
    # ========================================
    print(f"\n步骤3: 生成邻居风险感知特征...")
    
    # 生成度特征
    print("  生成度特征...")
    degree_feat = graph.in_degrees().unsqueeze_(1).float()
    
    # ========================================
    # 优化的风险邻居计数
    # ========================================
    print(f"\n  计算风险邻居数量（使用向量化优化）...")
    
    start_time = time.time()
    
    # 准备批次
    batches = [indices[i:i + batch_size] for i in range(0, num_nodes, batch_size)]
    num_batches = len(batches)
    
    # 选择计算策略
    if use_cuda:
        print(f"  使用GPU加速计算风险特征...")
        risk_feat_list = []
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="GPU计算风险特征")):
            risk_counts = count_risk_neighs_gpu_optimized(
                graph, labels_tensor, batch, device=device
            )
            risk_feat_list.extend(risk_counts)
            
            # 清理GPU内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        risk_feat = torch.tensor(risk_feat_list, dtype=torch.float32).unsqueeze_(1)
        
    else:
        print(f"  使用多进程计算风险特征（{num_workers}个工作进程）...")
        
        # 创建进度队列
        progress_queue = mp.Manager().Queue()
        
        # 启动进度显示线程
        progress_printer = ProgressPrinter()
        
        def progress_monitor():
            """进度监控线程"""
            total_batches = num_batches
            completed = 0
            
            while completed < total_batches:
                try:
                    while not progress_queue.empty():
                        task_id, current, total = progress_queue.get()
                        progress_printer.print_progress(
                            current, total, 
                            f"  Worker-{task_id}进度"
                        )
                    time.sleep(0.1)
                except:
                    break
        
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        
        # 准备graph_data用于多进程
        graph_data = (graph, labels_tensor)
        
        # 使用多进程计算
        with mp.Pool(processes=num_workers) as pool:
            partial_worker = partial(
                batch_count_risk_neighs_vectorized, 
                graph_data,
                progress_queue=progress_queue
            )
            
            results = []
            for i, batch in enumerate(tqdm(batches, desc="计算风险邻居")):
                # 传递任务ID用于进度跟踪
                result = pool.apply_async(
                    partial_worker, 
                    (batch,),
                    callback=lambda x, tid=i: progress_queue.put((tid, x.shape[0], x.shape[0]))
                )
                results.append(result)
            
            # 收集结果
            risk_feat_list = []
            for result in tqdm(results, desc="收集结果"):
                risk_feat_list.extend(result.get())
            
            risk_feat = torch.tensor(risk_feat_list, dtype=torch.float32).unsqueeze_(1)
        
        monitor_thread.join(timeout=1)
    
    elapsed_time = time.time() - start_time
    print(f"  风险邻居计算完成，用时: {elapsed_time:.2f}秒")
    print(f"  平均每个节点: {elapsed_time/num_nodes*1000:.3f}ms")
    
    # 生成度-风险特征对
    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
    edge_feat_np = edge_feat.numpy()
    
    # ========================================
    # 优化的特征映射生成 - 使用预构建缓存
    # ========================================
    print(f"\n  生成邻居特征映射（使用缓存优化）...")
    
    start_time = time.time()
    
    # 步骤1: 预构建邻居缓存（一次性构建，多次使用）
    neighbor_cache = build_neighbor_cache_from_edges(graph, num_nodes)
    
    # 准备特征映射的批次
    feat_batches = [indices[i:i + batch_size] for i in range(0, num_nodes, batch_size)]
    num_feat_batches = len(feat_batches)
    
    # 选择计算策略
    if use_cuda and num_nodes > 100000:
        print(f"  使用多进程+缓存计算邻居特征...")
        
        # 创建进度队列
        progress_queue = mp.Manager().Queue()
        
        # 启动进度显示线程
        progress_printer = ProgressPrinter()
        
        def progress_monitor():
            """进度监控线程"""
            total_batches = num_feat_batches
            completed = 0
            
            while completed < total_batches:
                try:
                    while not progress_queue.empty():
                        task_id, current, total = progress_queue.get()
                        progress_printer.print_progress(
                            current, total, 
                            f"  Worker-{task_id}进度"
                        )
                    time.sleep(0.1)
                except:
                    break
        
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        
        features_neigh_list = []
        
        with mp.Pool(processes=num_workers) as pool:
            partial_feat_worker = partial(
                batch_compute_neighbor_features_fast,
                graph,
                labels_tensor,
                edge_feat_np,
                n_1hop_samples=n_1hop_samples,
                n_2hop_samples=n_2hop_samples,
                neighbor_cache=neighbor_cache,
                progress_queue=progress_queue
            )
            
            results = []
            for i, batch in enumerate(tqdm(feat_batches, desc="计算邻居特征")):
                result = pool.apply_async(
                    partial_feat_worker,
                    (batch,),
                    callback=lambda x, tid=i: progress_queue.put((tid, x.shape[0], x.shape[0]))
                )
                results.append(result)
            
            # 收集结果
            for result in tqdm(results, desc="收集结果"):
                features_neigh_list.append(result.get())
            
            features_neigh = np.vstack(features_neigh_list)
        
        monitor_thread.join(timeout=1)
        
    else:
        # 直接使用缓存计算（对于中小规模数据更快）
        print(f"  使用缓存直接计算邻居特征...")
        
        features_neigh_list = []
        for batch_idx, batch in enumerate(tqdm(feat_batches, desc="计算邻居特征")):
            batch_features = batch_compute_neighbor_features_fast(
                graph, labels_tensor, edge_feat_np, batch,
                n_1hop_samples=n_1hop_samples,
                n_2hop_samples=n_2hop_samples,
                neighbor_cache=neighbor_cache
            )
            features_neigh_list.append(batch_features)
        
        features_neigh = np.vstack(features_neigh_list)
        
    elapsed_time = time.time() - start_time
    print(f"  邻居特征计算完成，用时: {elapsed_time:.2f}秒")
    print(f"  平均每个节点: {elapsed_time/num_nodes*1000:.3f}ms")
    
    # 合并所有特征
    features_neigh = torch.cat([edge_feat, torch.from_numpy(features_neigh)], dim=1).numpy()
    
    # 创建特征名称
    feat_names = ['degree', 'riskstat', '1hop_degree', '2hop_degree', 
                  '1hop_riskstat', '2hop_riskstat']
    
    # 处理NaN值
    features_neigh = np.nan_to_num(features_neigh, nan=0.0)
    
    # 标准化特征
    print(f"  标准化特征...")
    scaler = StandardScaler()
    features_neigh_df = pd.DataFrame(features_neigh, columns=feat_names)
    features_neigh_scaled = scaler.fit_transform(features_neigh_df)
    features_neigh_df = pd.DataFrame(features_neigh_scaled, columns=feat_names)
    
    # 保存邻居特征
    output_path = os.path.join(output_dir, "dgraphfin_neigh_feat.csv")
    features_neigh_df.to_csv(output_path, index=False)
    print(f"  邻居特征已保存到: {output_path}")
    
    # ========================================
    # 步骤4: 创建训练/验证/测试掩码
    # ========================================
    print(f"\n步骤4: 创建训练/验证/测试掩码...")
    
    # 创建全量掩码
    train_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 检查掩码类型 - 可能是索引数组或布尔数组
    # 如果掩码大小与总节点数相同，则是布尔掩码
    # 如果掩码大小等于数据集大小，则是索引数组
    if len(train_mask) == num_nodes:
        # 已经是完整掩码
        train_mask_full = train_mask.clone()
        valid_mask_full = valid_mask.clone()
        test_mask_full = test_mask.clone()
    else:
        # 是索引数组，需要转换为完整掩码
        print(f"  检测到索引格式掩码，转换为完整掩码...")
        
        # 将索引转换为张量并确保类型正确
        train_indices = torch.tensor(train_mask, dtype=torch.long)
        valid_indices = torch.tensor(valid_mask, dtype=torch.long)
        test_indices = torch.tensor(test_mask, dtype=torch.long)
        
        # 使用索引设置掩码
        train_mask_full[train_indices] = True
        valid_mask_full[valid_indices] = True
        test_mask_full[test_indices] = True
    
    mask_dict = {
        'train_mask': train_mask_full.numpy(),
        'valid_mask': valid_mask_full.numpy(),
        'test_mask': test_mask_full.numpy()
    }
    
    mask_path = os.path.join(output_dir, "dgraphfin_masks.npz")
    np.savez(mask_path, **mask_dict)
    print(f"  掩码已保存到: {mask_path}")
    
    # ========================================
    # 步骤5: 创建完整的.npz文件
    # ========================================
    print(f"\n步骤5: 创建包含所有信息的.npz文件...")
    full_npz_path = os.path.join(output_dir, "dgraphfin_full.npz")
    
    np.savez(full_npz_path,
             x=features.numpy(),
             y=labels.numpy(),
             edge_index=edge_index.numpy(),
             train_mask=train_mask_full.numpy(),
             valid_mask=valid_mask_full.numpy(),
             test_mask=test_mask_full.numpy(),
             neigh_feat=features_neigh_df.values)
    
    print(f"  完整数据集已保存到: {full_npz_path}")
    
    return {
        'graph': graph,
        'features': features,
        'labels': labels,
        'neigh_features': torch.from_numpy(features_neigh_df.values).float(),
        'train_mask': train_mask_full,
        'valid_mask': valid_mask_full,
        'test_mask': test_mask_full,
        'output_dir': output_dir
    }


def create_adjacency_matrix_fast(adj_list, n):
    """快速创建邻接矩阵 - 使用稀疏矩阵"""
    rows = []
    cols = []
    
    print(f"  创建邻接矩阵，节点数: {n}")
    for node, neighbors in tqdm(adj_list.items(), desc="处理节点"):
        for neighbor in neighbors:
            rows.append(node)
            cols.append(neighbor)
    
    data = np.ones(len(rows))
    adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return adj_matrix


def block_matrix_multiply_optimized(A, B, block_size, device):
    """优化的分块矩阵乘法"""
    n = A.shape[0]
    C = torch.zeros((n, n), device=device)
    
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        A_block = A[i:i_end, :]
        
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            B_block = B[:, j:j_end]
            
            C[i:i_end, j:j_end] = torch.matmul(A_block, B_block)
    
    return C


def matrix_powers_gpu_optimized_v2(adj_list, n, block_size, matrix_prefix):
    """
    矩阵幂计算 - 内存优化版本
    
    对于大规模图，使用稀疏矩阵方法避免内存溢出
    """
    print(f"\n计算1-10阶矩阵幂，节点数: {n}")
    print(f"方法: 稀疏矩阵迭代计算")
    
    # 检查是否有GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    has_cuda = torch.cuda.is_available()
    print(f"使用设备: {device}")
    
    if has_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU内存: {gpu_memory:.2f} GB")
        
        # 估算稠密矩阵内存
        dense_memory = (n * n * 4) / 1e9  # float32
        print(f"估计稠密矩阵内存需求: {dense_memory:.2f} GB")
        
        if dense_memory > gpu_memory * 0.3:
            print(f"\n⚠️  稠密矩阵({dense_memory:.1f}GB)超过可用GPU内存({gpu_memory:.1f}GB)")
            print("   使用稀疏矩阵+CPU计算模式...")
            use_sparse_cpu = True
        else:
            use_sparse_cpu = False
    else:
        print("使用CPU稀疏矩阵计算...")
        use_sparse_cpu = True
    
    # 1. 创建稀疏邻接矩阵
    print("\n创建稀疏邻接矩阵...")
    adj_matrix_sparse = create_adjacency_matrix_fast(adj_list, n)
    
    # 统计信息
    nnz = adj_matrix_sparse.nnz
    density = nnz / (n * n)
    print(f"邻接矩阵非零元素: {nnz}, 密度: {density:.8f}")
    
    if use_sparse_cpu:
        # 使用 scipy sparse 矩阵进行迭代计算
        _compute_sparse_matrix_powers(adj_matrix_sparse, n, matrix_prefix, max_k=10)
    else:
        # 使用GPU稠密矩阵计算
        _compute_dense_matrix_powers_gpu(adj_matrix_sparse, n, block_size, matrix_prefix, device)


def _compute_sparse_matrix_powers(adj_sparse, n, matrix_prefix, max_k=10):
    """
    使用 scipy 稀疏矩阵计算矩阵幂
    适合大规模图，内存效率高
    """
    import scipy.sparse as sp
    
    print(f"\n使用稀疏矩阵计算1-{max_k}阶矩阵幂...")
    
    # 保存 A^1
    file_name = f'{matrix_prefix}1.npz'
    sp.save_npz(file_name, adj_sparse)
    print(f"保存 A^1 到 {file_name}")
    
    # 迭代计算 A^k = A^(k-1) * A
    current = adj_sparse
    
    total_start_time = time.time()
    
    for k in tqdm(range(2, max_k + 1), desc="计算幂次"):
        k_start_time = time.time()
        
        # 稀疏矩阵乘法
        current = current.dot(adj_sparse)
        
        # 统计信息
        nnz = current.nnz
        density = nnz / (n * n)
        
        # 检查是否变得太稠密
        if density > 0.001:  # 0.1% 密度阈值
            print(f"\n  ⚠️  k={k}: 矩阵变得太稠密 (密度={density:.4f})")
            print(f"     停止计算，截断至 k={k-1}")
            break
        
        # 保存结果
        file_name = f'{matrix_prefix}{k}.npz'
        sp.save_npz(file_name, current)
        
        k_elapsed = time.time() - k_start_time
        print(f"  k={k}: 非零元素={nnz}, 密度={density:.6f}, 用时={k_elapsed:.2f}秒")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n稀疏矩阵幂计算完成！总用时: {total_elapsed:.2f}秒")
    print(f"\n注意: 矩阵幂以 .npz 格式保存（scipy sparse format）")


def _compute_dense_matrix_powers_gpu(adj_matrix_sparse, n, block_size, matrix_prefix, device):
    """
    GPU稠密矩阵计算（仅对小规模图有效）
    """
    print("\n转换为稠密矩阵...")
    adj_matrix_dense = adj_matrix_sparse.toarray()
    adj_matrix = torch.from_numpy(adj_matrix_dense).float().to(device)
    
    # 保存k=1
    file_name = f'{matrix_prefix}1.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(adj_matrix_dense, f)
    print(f"保存邻接矩阵: {file_name}")
    
    # 迭代计算
    print("\n迭代计算2-10阶矩阵幂...")
    
    prev_power = adj_matrix
    prev_power_binary = (prev_power != 0).float()
    
    total_start_time = time.time()
    
    for k in tqdm(range(2, 11), desc="计算幂次"):
        k_start_time = time.time()
        
        current_power = block_matrix_multiply_optimized(prev_power, adj_matrix, block_size, device)
        current_power_binary = (current_power != 0).float()
        
        result = current_power_binary - prev_power_binary
        result = torch.maximum(result, torch.tensor(0))
        
        result = result + torch.eye(n, device=device)
        
        file_name = f'{matrix_prefix}{k}.pkl'
        result_np = result.cpu().numpy()
        with open(file_name, 'wb') as file:
            pickle.dump(result_np, file)
        
        prev_power = current_power
        prev_power_binary = current_power_binary
        
        k_elapsed = time.time() - k_start_time
        print(f"  {k}阶矩阵幂计算完成，用时: {k_elapsed:.2f}秒")
        
        if k % 3 == 0:
            torch.cuda.empty_cache()
    
    total_elapsed = time.time() - total_start_time
    print(f"\n所有矩阵幂计算完成！总用时: {total_elapsed:.2f}秒")


def process_dgraphfin_matrix_powers(output_dir='../data'):
    """处理DGraphFin的矩阵幂计算"""
    print("\n" + "="*60)
    print("开始处理DGraphFin矩阵幂 (1-10阶)")
    print("="*60)
    
    adjlist_path = os.path.join(output_dir, "dgraphfin_homo_adjlists.pickle")
    if not os.path.exists(adjlist_path):
        print(f"错误: 邻接列表文件不存在: {adjlist_path}")
        return
    
    print(f"加载邻接列表: {adjlist_path}")
    with open(adjlist_path, 'rb') as file:
        adj_list = pickle.load(file)
    
    n = len(adj_list)
    print(f"节点数量: {n}")
    
    # 动态计算合适的block_size
    block_size_options = [1024, 512, 256, 128, 64]
    block_size = 64
    
    for bs in block_size_options:
        if n % bs == 0:
            block_size = bs
            print(f"找到合适的block_size: {block_size}")
            break
    
    if n % block_size != 0:
        g = math.gcd(n, 1024)
        block_size = g
        print(f"使用最大公约数作为block_size: {block_size}")
    
    print(f"最终使用block_size: {block_size}")
    
    matrix_prefix = os.path.join(output_dir, "dgraphfin_adj_power_matrix")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU内存: {gpu_memory:.2f} GB")
        
        matrix_memory = (n * n * 4) / 1e9
        print(f"估计单个矩阵内存需求: {matrix_memory:.2f} GB")
        
        if matrix_memory > gpu_memory * 0.7:
            print(f"警告: 矩阵大小({matrix_memory:.2f}GB)可能超过GPU内存({gpu_memory:.2f}GB)")
            print("建议使用更小的block_size或分块处理")
            
            if block_size > 64:
                block_size = 64
                print(f"调整block_size为: {block_size}")
    else:
        print("警告: 没有可用的GPU，将使用CPU计算")
        print("这可能会非常慢，建议在GPU上运行")
    
    start_time = time.time()
    
    try:
        matrix_powers_gpu_optimized_v2(adj_list, n, block_size, matrix_prefix)
    except torch.cuda.OutOfMemoryError:
        print("GPU内存不足，尝试使用更保守的策略...")
        
        block_size = 32
        print(f"使用更小的block_size: {block_size}")
        
        try:
            matrix_powers_gpu_optimized_v2(adj_list, n, block_size, matrix_prefix)
        except Exception as e:
            print(f"计算失败: {e}")
            print("建议使用CPU版本或减少矩阵大小")
    except Exception as e:
        print(f"计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    print(f"矩阵幂计算总用时: {end_time - start_time:.2f}秒")


def main():
    """主函数"""
    print("="*60)
    print("DGraphFin数据集处理流程 - 优化版")
    print("="*60)
    
    start_time = time.time()
    
    # 采样参数设置
    n_1hop_samples = 10
    n_2hop_samples = 20
    
    # 性能参数设置
    batch_size = 10000
    use_gpu = torch.cuda.is_available()
    
    print(f"\n阶段1: 创建数据集格式")
    print(f"采样参数: 1-hop邻居采样数={n_1hop_samples}, 2-hop邻居采样数={n_2hop_samples}")
    print(f"批次大小: {batch_size}")
    print(f"使用GPU加速: {'是' if use_gpu else '否'}")
    
    try:
        print("\n开始创建数据集...")
        processed_data = create_dgraphfin_dataset(
            data_path='../../data',
            output_dir='../data',
            n_1hop_samples=n_1hop_samples,
            n_2hop_samples=n_2hop_samples,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        
        dataset_time = time.time()
        print(f"\n数据集创建完成，用时: {dataset_time - start_time:.2f}秒")
        
        # 步骤2: 计算矩阵幂
        print("\n" + "="*60)
        print("阶段2: 计算矩阵幂 (1-10阶)")
        print("="*60)
        
        if not torch.cuda.is_available():
            print("警告: 没有检测到GPU，矩阵幂计算将非常缓慢")
            print("建议在有足够GPU内存的机器上运行此步骤")
            response = input("是否继续? (y/n): ")
            if response.lower() != 'y':
                print("跳过矩阵幂计算")
            else:
                process_dgraphfin_matrix_powers('../data')
        else:
            process_dgraphfin_matrix_powers('../data')
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        
        print("\n" + "="*60)
        print("DGraphFin数据集处理完成！")
        print("="*60)
        
        print(f"\n数据文件保存在: ../data/")
        print("包含以下文件:")
        print("  1. graph-dgraphfin.bin - DGL图文件")
        print("  2. dgraphfin_homo_adjlists.pickle - 邻接列表")
        print("  3. dgraphfin_neigh_feat.csv - 邻居特征")
        print("  4. dgraphfin_masks.npz - 训练/验证/测试掩码")
        print("  5. dgraphfin_full.npz - 完整数据集")
        
        # 检查矩阵幂文件
        matrix_files = []
        for k in range(1, 11):
            matrix_file = os.path.join('../data', f"dgraphfin_adj_power_matrix{k}.pkl")
            if os.path.exists(matrix_file):
                file_size = os.path.getsize(matrix_file) / (1024**3)
                matrix_files.append(f"  {k}. dgraphfin_adj_power_matrix{k}.pkl - {k}阶矩阵幂 ({file_size:.2f} GB)")
        
        if matrix_files:
            print("  6. 矩阵幂文件:")
            for file_info in matrix_files:
                print(file_info)
        else:
            print("  6. 矩阵幂文件: 未生成")
        
        print(f"\n处理统计:")
        print(f"  - 总节点数: {processed_data['features'].shape[0]}")
        print(f"  - 总边数: {processed_data['graph'].num_edges()}")
        print(f"  - 特征维度: {processed_data['neigh_features'].shape[1]}")
        print(f"  - 训练节点: {processed_data['train_mask'].sum().item()}")
        print(f"  - 验证节点: {processed_data['valid_mask'].sum().item()}")
        print(f"  - 测试节点: {processed_data['test_mask'].sum().item()}")
        
    except KeyboardInterrupt:
        print("\n\n处理被用户中断")
        print("部分文件可能已保存，请检查输出目录")
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

