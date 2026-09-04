import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict

def compute_neighbor_features(edge_index: np.ndarray,
                            x: np.ndarray,
                            max_neighbors_1hop: int = 50,
                            max_neighbors_2hop: int = 500,
                            output_path: str = None) -> pd.DataFrame:
    """
    计算邻居聚合特征：
    1. 1-hop邻居特征聚合
    2. 2-hop邻居特征聚合
    3. 邻居特征统计量
    """
    
    print("计算邻居聚合特征...")
    
    n_nodes = x.shape[0]
    n_features = x.shape[1]
    
    # 创建邻接矩阵（有向）
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    
    # 对称化用于无向邻居计算
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    feature_dict = {}
    
    # 1. 1-hop邻居特征聚合
    print("计算1-hop邻居特征聚合...")
    neighbor_1hop_features = aggregate_neighbor_features(
        adj_matrix_sym, x, max_neighbors=max_neighbors_1hop
    )
    
    for i in range(n_features):
        for agg_type, values in neighbor_1hop_features[i].items():
            feature_dict[f'neighbor1_{agg_type}_feat{i}'] = values.astype(np.float32)
    
    # 2. 计算2-hop邻居特征（通过邻接矩阵平方）
    print("计算2-hop邻居...")
    # 限制计算规模
    if n_nodes > 100000:
        print("节点数过多，使用采样计算2-hop特征...")
        neighbor_2hop_features = compute_2hop_features_sampled(
            adj_matrix_sym, x, max_neighbors_2hop, sample_size=5000
        )
    else:
        adj_squared = adj_matrix_sym.dot(adj_matrix_sym)
        neighbor_2hop_features = aggregate_neighbor_features(
            adj_squared, x, max_neighbors=max_neighbors_2hop, prefix='neighbor2'
        )
    
    for i in range(min(5, n_features)):  # 只取前5个特征进行2-hop聚合
        for agg_type, values in neighbor_2hop_features[i].items():
            feature_dict[f'neighbor2_{agg_type}_feat{i}'] = values.astype(np.float32)
    
    # 3. 邻居度与特征的相关性
    print("计算邻居度相关性特征...")
    degree_corr_features = compute_degree_correlation_features(
        adj_matrix_sym, x
    )
    
    for feat_name, values in degree_corr_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 4. 邻居特征分布统计
    print("计算邻居特征分布...")
    neighbor_dist_stats = compute_neighbor_distribution_stats(
        adj_matrix_sym, x, max_neighbors_1hop
    )
    
    for stat_name, stat_values in neighbor_dist_stats.items():
        feature_dict[stat_name] = stat_values.astype(np.float32)
    
    # 5. 出入邻居特征差异
    print("计算出入邻居差异...")
    in_out_diff_features = compute_in_out_difference_features(
        adj_matrix, x, max_neighbors_1hop
    )
    
    for feat_name, values in in_out_diff_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        with open(output_path, 'wb') as f:
            import pickle
            pickle.dump(features_df, f)
    
    return features_df

def aggregate_neighbor_features(adj_matrix: sp.csr_matrix,
                              node_features: np.ndarray,
                              max_neighbors: int = 100,
                              prefix: str = 'neighbor') -> Dict[int, Dict[str, np.ndarray]]:
    """聚合邻居特征"""
    n_nodes, n_features = node_features.shape
    adj_matrix = adj_matrix.tocsc()
    
    # 初始化结果字典
    results = {i: {} for i in range(n_features)}
    
    # 为每个特征预先分配数组
    for feat_idx in range(n_features):
        results[feat_idx]['mean'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['max'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['min'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['std'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['sum'] = np.zeros(n_nodes, dtype=np.float32)
    
    # 逐节点处理
    for node in tqdm(range(n_nodes), desc=f"聚合{prefix}特征"):
        neighbors = adj_matrix[:, node].indices
        
        # 限制邻居数量
        if len(neighbors) > max_neighbors:
            neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
        
        if len(neighbors) == 0:
            continue
        
        neighbor_feats = node_features[neighbors]
        
        for feat_idx in range(n_features):
            feat_values = neighbor_feats[:, feat_idx]
            
            results[feat_idx]['mean'][node] = feat_values.mean()
            results[feat_idx]['max'][node] = feat_values.max()
            results[feat_idx]['min'][node] = feat_values.min()
            results[feat_idx]['std'][node] = feat_values.std() if len(neighbors) > 1 else 0
            results[feat_idx]['sum'][node] = feat_values.sum()
    
    return results

def compute_2hop_features_sampled(adj_matrix: sp.csr_matrix,
                                 node_features: np.ndarray,
                                 max_neighbors: int = 500,
                                 sample_size: int = 5000) -> Dict[int, Dict[str, np.ndarray]]:
    """采样计算2-hop邻居特征"""
    n_nodes = adj_matrix.shape[0]
    
    # 随机采样节点
    sampled_nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    
    # 初始化结果
    results = {i: {'mean': np.zeros(n_nodes, dtype=np.float32)} 
              for i in range(min(5, node_features.shape[1]))}
    
    adj_matrix = adj_matrix.tocsc()
    
    for node in tqdm(sampled_nodes, desc="采样计算2-hop特征"):
        # 获取1-hop邻居
        neighbors_1hop = adj_matrix[:, node].indices
        
        if len(neighbors_1hop) == 0:
            continue
        
        # 限制1-hop邻居数量
        if len(neighbors_1hop) > 50:
            neighbors_1hop = np.random.choice(neighbors_1hop, 50, replace=False)
        
        # 收集2-hop邻居
        neighbors_2hop = set()
        for neighbor in neighbors_1hop:
            second_neighbors = adj_matrix[:, neighbor].indices
            neighbors_2hop.update(second_neighbors)
        
        # 移除自己和1-hop邻居
        neighbors_2hop.discard(node)
        neighbors_2hop.difference_update(neighbors_1hop)
        
        neighbors_2hop = np.array(list(neighbors_2hop))
        
        # 限制2-hop邻居数量
        if len(neighbors_2hop) > max_neighbors:
            neighbors_2hop = np.random.choice(neighbors_2hop, max_neighbors, replace=False)
        
        if len(neighbors_2hop) == 0:
            continue
        
        neighbor_feats = node_features[neighbors_2hop]
        
        for feat_idx in results.keys():
            feat_values = neighbor_feats[:, feat_idx]
            results[feat_idx]['mean'][node] = feat_values.mean()
    
    # 填充未采样节点
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    for feat_idx in results.keys():
        sampled_mean = results[feat_idx]['mean'][sampled_nodes].mean()
        results[feat_idx]['mean'][unsampled] = sampled_mean
    
    return results

def compute_degree_correlation_features(adj_matrix: sp.csr_matrix,
                                      node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """计算邻居度与特征的相关性"""
    n_nodes = adj_matrix.shape[0]
    
    # 计算度
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 初始化结果
    results = {}
    n_features = node_features.shape[1]
    
    adj_matrix = adj_matrix.tocsc()
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="计算度相关性"):
        feature_values = node_features[:, feat_idx]
        
        # 计算邻居特征均值
        neighbor_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        neighbor_degree_mean = np.zeros(n_nodes, dtype=np.float32)
        
        for node in range(n_nodes):
            neighbors = adj_matrix[:, node].indices
            
            if len(neighbors) == 0:
                neighbor_feat_mean[node] = feature_values[node]
                neighbor_degree_mean[node] = degree[node]
                continue
            
            neighbor_feat_mean[node] = feature_values[neighbors].mean()
            neighbor_degree_mean[node] = degree[neighbors].mean()
        
        # 计算相关性特征
        results[f'feat{feat_idx}_degree_corr_coef'] = neighbor_feat_mean * neighbor_degree_mean
        results[f'feat{feat_idx}_degree_diff'] = feature_values - neighbor_degree_mean
        results[f'feat{feat_idx}_feat_degree_ratio'] = np.divide(
            feature_values, neighbor_degree_mean + 1e-8
        )
    
    return results

def compute_neighbor_distribution_stats(adj_matrix: sp.csr_matrix,
                                      node_features: np.ndarray,
                                      max_neighbors: int = 50) -> Dict[str, np.ndarray]:
    """计算邻居特征分布统计"""
    n_nodes, n_features = node_features.shape
    
    # 初始化结果
    results = {}
    adj_matrix = adj_matrix.tocsc()
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="计算邻居分布统计"):
        feature_values = node_features[:, feat_idx]
        
        # 初始化统计量
        skewness = np.zeros(n_nodes, dtype=np.float32)
        kurtosis = np.zeros(n_nodes, dtype=np.float32)
        iqr = np.zeros(n_nodes, dtype=np.float32)  # 四分位距
        
        for node in range(n_nodes):
            neighbors = adj_matrix[:, node].indices
            
            # 限制邻居数量
            if len(neighbors) > max_neighbors:
                neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
            
            if len(neighbors) < 3:
                skewness[node] = 0
                kurtosis[node] = 0
                iqr[node] = 0
                continue
            
            neighbor_feats = feature_values[neighbors]
            
            # 计算偏度
            mean = neighbor_feats.mean()
            std = neighbor_feats.std()
            if std > 0:
                skewness[node] = ((neighbor_feats - mean) ** 3).mean() / (std ** 3)
            
            # 计算峰度
            if std > 0:
                kurtosis[node] = ((neighbor_feats - mean) ** 4).mean() / (std ** 4) - 3
            
            # 计算四分位距
            q75, q25 = np.percentile(neighbor_feats, [75, 25])
            iqr[node] = q75 - q25
        
        results[f'feat{feat_idx}_neighbor_skew'] = skewness
        results[f'feat{feat_idx}_neighbor_kurt'] = kurtosis
        results[f'feat{feat_idx}_neighbor_iqr'] = iqr
    
    return results

def compute_in_out_difference_features(adj_matrix: sp.csr_matrix,
                                     node_features: np.ndarray,
                                     max_neighbors: int = 50) -> Dict[str, np.ndarray]:
    """计算出入邻居特征差异"""
    n_nodes, n_features = node_features.shape
    
    # 转置得到反向邻接
    adj_matrix_t = adj_matrix.T
    
    results = {}
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="计算出入邻居差异"):
        feature_values = node_features[:, feat_idx]
        
        in_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        out_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        
        # 计算入邻居特征均值
        adj_matrix_t_csc = adj_matrix_t.tocsc()
        for node in range(n_nodes):
            in_neighbors = adj_matrix_t_csc[:, node].indices
            
            if len(in_neighbors) > max_neighbors:
                in_neighbors = np.random.choice(in_neighbors, max_neighbors, replace=False)
            
            if len(in_neighbors) > 0:
                in_feat_mean[node] = feature_values[in_neighbors].mean()
            else:
                in_feat_mean[node] = feature_values[node]
        
        # 计算出邻居特征均值
        adj_matrix_csc = adj_matrix.tocsc()
        for node in range(n_nodes):
            out_neighbors = adj_matrix_csc[:, node].indices
            
            if len(out_neighbors) > max_neighbors:
                out_neighbors = np.random.choice(out_neighbors, max_neighbors, replace=False)
            
            if len(out_neighbors) > 0:
                out_feat_mean[node] = feature_values[out_neighbors].mean()
            else:
                out_feat_mean[node] = feature_values[node]
        
        # 计算差异特征
        results[f'feat{feat_idx}_in_out_diff'] = in_feat_mean - out_feat_mean
        results[f'feat{feat_idx}_in_out_ratio'] = np.divide(
            in_feat_mean, out_feat_mean + 1e-8
        )
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--max_neighbors_1hop', type=int, default=50, help='1-hop最大邻居数')
    parser.add_argument('--max_neighbors_2hop', type=int, default=500, help='2-hop最大邻居数')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    # 计算邻居特征
    features_df = compute_neighbor_features(
        edge_index, x,
        args.max_neighbors_1hop,
        args.max_neighbors_2hop
    )
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"邻居特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
