import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse
import scipy.sparse as sp
from tqdm import tqdm

def compute_structural_features(edge_index: np.ndarray, 
                               n_nodes: int,
                               max_neighbors: int = 1000,
                               output_path: str = None) -> pd.DataFrame:
    """
    计算图结构特征：
    1. 度特征
    2. PageRank
    3. 三角形计数
    4. 中心性特征
    5. 社区特征
    """
    
    print("计算结构特征...")
    
    # 创建邻接矩阵
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    
    # 对称化邻接矩阵（无向图）
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    feature_dict = {}
    
    # 1. 度特征
    print("计算度特征...")
    # 出度
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    feature_dict['out_degree'] = out_degree.astype(np.float32)
    
    # 入度
    in_degree = np.array(adj_matrix.sum(axis=0)).flatten()
    feature_dict['in_degree'] = in_degree.astype(np.float32)
    
    # 总度（无向）
    total_degree = np.array(adj_matrix_sym.sum(axis=1)).flatten()
    feature_dict['total_degree'] = total_degree.astype(np.float32)
    
    # 度对数变换
    feature_dict['out_degree_log'] = np.log1p(out_degree).astype(np.float32)
    feature_dict['in_degree_log'] = np.log1p(in_degree).astype(np.float32)
    feature_dict['total_degree_log'] = np.log1p(total_degree).astype(np.float32)
    
    # 归一化度
    feature_dict['out_degree_norm'] = (out_degree / (out_degree.max() + 1e-8)).astype(np.float32)
    feature_dict['in_degree_norm'] = (in_degree / (in_degree.max() + 1e-8)).astype(np.float32)
    
    # 2. PageRank（近似计算）
    print("计算PageRank...")
    pagerank_scores = compute_pagerank(adj_matrix_sym, max_iter=20)
    feature_dict['pagerank'] = pagerank_scores.astype(np.float32)
    feature_dict['pagerank_log'] = np.log1p(pagerank_scores).astype(np.float32)
    
    # 3. 局部聚类系数（采样计算）
    print("计算聚类系数...")
    clustering_coeff = compute_clustering_coefficient_sampled(adj_matrix_sym, 
                                                             sample_size=min(1000, n_nodes))
    feature_dict['clustering_coeff'] = clustering_coeff.astype(np.float32)
    
    # 4. 度中心性
    feature_dict['degree_centrality'] = (total_degree / (n_nodes - 1)).astype(np.float32)
    
    # 5. 二阶邻居数
    print("计算二阶邻居...")
    adj_squared = adj_matrix_sym.dot(adj_matrix_sym)
    second_neighbors = np.array(adj_squared.sum(axis=1)).flatten()
    feature_dict['second_neighbors'] = second_neighbors.astype(np.float32)
    feature_dict['second_neighbors_log'] = np.log1p(second_neighbors).astype(np.float32)
    
    # 6. 邻居度统计
    print("计算邻居度统计...")
    neighbor_degree_stats = compute_neighbor_degree_stats(adj_matrix_sym, total_degree)
    for stat_name, stat_values in neighbor_degree_stats.items():
        feature_dict[f'neighbor_{stat_name}'] = stat_values.astype(np.float32)
    
    # 7. 结构洞指标（近似）
    print("计算结构洞指标...")
    constraint = compute_constraint_sampled(adj_matrix_sym, sample_size=min(500, n_nodes))
    feature_dict['constraint'] = constraint.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        with open(output_path, 'wb') as f:
            import pickle
            pickle.dump(features_df, f)
    
    return features_df

def compute_pagerank(adj_matrix: sp.csr_matrix, 
                    alpha: float = 0.85, 
                    max_iter: int = 20) -> np.ndarray:
    """计算PageRank（简化版）"""
    n_nodes = adj_matrix.shape[0]
    
    # 归一化邻接矩阵
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    norm_adj = adj_matrix.multiply(1.0 / degree[:, np.newaxis])
    
    # 初始化
    pr = np.ones(n_nodes) / n_nodes
    
    # 迭代计算
    for _ in range(max_iter):
        pr_new = alpha * norm_adj.T.dot(pr) + (1 - alpha) / n_nodes
        diff = np.abs(pr_new - pr).sum()
        pr = pr_new
        
        if diff < 1e-6:
            break
    
    return pr

def compute_clustering_coefficient_sampled(adj_matrix: sp.csr_matrix, 
                                         sample_size: int = 1000) -> np.ndarray:
    """采样计算聚类系数"""
    n_nodes = adj_matrix.shape[0]
    clustering = np.zeros(n_nodes, dtype=np.float32)
    
    # 随机采样节点
    if n_nodes > sample_size:
        sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    else:
        sampled_nodes = np.arange(n_nodes)
    
    adj_matrix = adj_matrix.tocsc()
    
    for node in tqdm(sampled_nodes, desc="计算聚类系数"):
        # 获取邻居
        neighbors = adj_matrix[:, node].indices
        k = len(neighbors)
        
        if k < 2:
            clustering[node] = 0
            continue
        
        # 计算邻居之间的连接数
        neighbor_pairs = 0
        for i in range(k):
            for j in range(i+1, k):
                if adj_matrix[neighbors[i], neighbors[j]]:
                    neighbor_pairs += 1
        
        clustering[node] = 2 * neighbor_pairs / (k * (k - 1))
    
    # 对于未采样的节点，使用均值填充
    if n_nodes > sample_size:
        mean_clustering = clustering[sampled_nodes].mean()
        all_indices = np.arange(n_nodes)
        unsampled = np.setdiff1d(all_indices, sampled_nodes)
        clustering[unsampled] = mean_clustering
    
    return clustering

def compute_neighbor_degree_stats(adj_matrix: sp.csr_matrix, 
                                node_degrees: np.ndarray) -> Dict[str, np.ndarray]:
    """计算邻居度的统计特征"""
    n_nodes = adj_matrix.shape[0]
    
    # 初始化统计量
    neighbor_mean = np.zeros(n_nodes, dtype=np.float32)
    neighbor_max = np.zeros(n_nodes, dtype=np.float32)
    neighbor_min = np.zeros(n_nodes, dtype=np.float32)
    neighbor_std = np.zeros(n_nodes, dtype=np.float32)
    
    adj_matrix = adj_matrix.tocsc()
    
    for node in tqdm(range(n_nodes), desc="计算邻居度统计"):
        neighbors = adj_matrix[:, node].indices
        
        if len(neighbors) == 0:
            neighbor_mean[node] = 0
            neighbor_max[node] = 0
            neighbor_min[node] = 0
            neighbor_std[node] = 0
            continue
        
        neighbor_degrees = node_degrees[neighbors]
        neighbor_mean[node] = neighbor_degrees.mean()
        neighbor_max[node] = neighbor_degrees.max()
        neighbor_min[node] = neighbor_degrees.min()
        neighbor_std[node] = neighbor_degrees.std() if len(neighbors) > 1 else 0
    
    return {
        'degree_mean': neighbor_mean,
        'degree_max': neighbor_max,
        'degree_min': neighbor_min,
        'degree_std': neighbor_std
    }

def compute_constraint_sampled(adj_matrix: sp.csr_matrix, 
                             sample_size: int = 500) -> np.ndarray:
    """采样计算结构洞约束系数"""
    n_nodes = adj_matrix.shape[0]
    constraint = np.zeros(n_nodes, dtype=np.float32)
    
    if n_nodes > sample_size:
        sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    else:
        sampled_nodes = np.arange(n_nodes)
    
    adj_matrix = adj_matrix.tocsc()
    
    for node in tqdm(sampled_nodes, desc="计算结构洞约束"):
        neighbors = adj_matrix[:, node].indices
        total_connection = len(neighbors)
        
        if total_connection == 0:
            constraint[node] = 1
            continue
        
        constraint_sum = 0
        for neighbor in neighbors:
            # 计算p_{iq}
            p_iq = 1.0 / total_connection
            
            # 计算公共邻居
            neighbor_neighbors = adj_matrix[:, neighbor].indices
            common_neighbors = np.intersect1d(neighbors, neighbor_neighbors)
            
            # 计算p_{iq} * p_{qj}
            for common in common_neighbors:
                if common != node and common != neighbor:
                    p_qj = 1.0 / len(neighbor_neighbors)
                    constraint_sum += p_iq * p_qj
        
        constraint[node] = constraint_sum
    
    # 填充未采样节点
    if n_nodes > sample_size:
        mean_constraint = constraint[sampled_nodes].mean()
        all_indices = np.arange(n_nodes)
        unsampled = np.setdiff1d(all_indices, sampled_nodes)
        constraint[unsampled] = mean_constraint
    
    return constraint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--max_neighbors', type=int, default=1000, help='最大邻居数限制')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    
    # 获取节点数
    if 'x' in data:
        n_nodes = data['x'].shape[0]
    else:
        n_nodes = edge_index.max() + 1
    
    # 计算结构特征
    features_df = compute_structural_features(edge_index, n_nodes, args.max_neighbors)
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"结构特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
