"""
谱特征模块：基于图频谱的特征
"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # 添加这行
def compute_spectral_features(edge_index: np.ndarray,
                             node_features: np.ndarray,
                             k_eigenvalues: int = 20,
                             use_chebyshev: bool = True,
                             output_path: str = None) -> pd.DataFrame:
    """
    计算谱特征：
    1. 拉普拉斯特征向量
    2. 图信号平滑度
    3. 谱聚类特征
    4. 切比雪夫多项式特征
    5. 谱间隙特征
    """
    
    print("计算谱特征...")
    n_nodes = node_features.shape[0]
    
    # 创建对称邻接矩阵
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    feature_dict = {}
    
    # 1. 度归一化拉普拉斯特征向量
    print("计算拉普拉斯特征向量...")
    laplacian_features = compute_laplacian_eigenvectors_sampled(
        adj_matrix_sym, node_features, k_eigenvalues=min(k_eigenvalues, 15)
    )
    
    for i in range(laplacian_features.shape[1]):
        feature_dict[f'laplacian_eig_{i}'] = laplacian_features[:, i].astype(np.float32)
    
    # 2. 图信号平滑度特征
    print("计算图信号平滑度...")
    smoothness_features = compute_smoothness_features(adj_matrix_sym, node_features)
    
    for feat_name, values in smoothness_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 3. 谱聚类系数
    print("计算谱聚类系数...")
    spectral_clustering_features = compute_spectral_clustering_features(
        adj_matrix_sym, n_clusters=10
    )
    
    for i in range(spectral_clustering_features.shape[1]):
        feature_dict[f'spectral_cluster_{i}'] = spectral_clustering_features[:, i].astype(np.float32)
    
    # 4. 切比雪夫多项式特征
    if use_chebyshev:
        print("计算切比雪夫多项式特征...")
        chebyshev_features = compute_chebyshev_features(
            adj_matrix_sym, node_features, order=3
        )
        
        for i in range(chebyshev_features.shape[1]):
            feature_dict[f'chebyshev_{i}'] = chebyshev_features[:, i].astype(np.float32)
    
    # 5. 谱间隙特征
    print("计算谱间隙特征...")
    spectral_gap_features = compute_spectral_gap_features(adj_matrix_sym)
    
    for feat_name, values in spectral_gap_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 6. 基于节点特征的谱响应
    print("计算节点特征谱响应...")
    spectral_response_features = compute_spectral_response_features(
        adj_matrix_sym, node_features
    )
    
    for feat_name, values in spectral_response_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    print(f"谱特征维度: {features_df.shape}")
    return features_df

def compute_laplacian_eigenvectors_sampled(adj_matrix: sp.csr_matrix,
                                         node_features: np.ndarray,
                                         k_eigenvalues: int = 10) -> np.ndarray:
    """计算拉普拉斯特征向量（使用采样和近似）"""
    n_nodes = adj_matrix.shape[0]
    
    # 对小图计算完整特征向量
    if n_nodes <= 10000:
        try:
            # 计算归一化拉普拉斯矩阵
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree + 1e-8))
            L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
            
            # 计算特征向量
            eigenvalues, eigenvectors = eigsh(L, k=min(k_eigenvalues+1, n_nodes-1), 
                                             which='SM', maxiter=1000)
            
            # 排除第一个零特征值
            if eigenvectors.shape[1] > 1:
                return eigenvectors[:, 1:k_eigenvalues+1].astype(np.float32)
            else:
                return compute_approximate_eigenvectors(adj_matrix, node_features, k_eigenvalues)
        
        except Exception as e:
            print(f"完整特征向量计算失败: {e}")
            return compute_approximate_eigenvectors(adj_matrix, node_features, k_eigenvalues)
    
    # 对大图使用近似
    return compute_approximate_eigenvectors(adj_matrix, node_features, k_eigenvalues)

def compute_approximate_eigenvectors(adj_matrix: sp.csr_matrix,
                                   node_features: np.ndarray,
                                   k_eigenvalues: int = 10) -> np.ndarray:
    """近似计算特征向量"""
    n_nodes, n_features = node_features.shape
    
    # 基于节点特征的简单近似
    eigenvectors = np.zeros((n_nodes, k_eigenvalues), dtype=np.float32)
    
    # 使用节点特征的线性组合作为近似特征向量
    for i in range(k_eigenvalues):
        # 创建随机权重
        if i < min(5, n_features):
            # 前几个使用原始特征
            eigenvectors[:, i] = node_features[:, i % n_features]
        else:
            # 后面的使用随机组合
            weights = np.random.randn(n_features).astype(np.float32)
            eigenvectors[:, i] = node_features @ weights
        
        # 归一化
        norm = np.linalg.norm(eigenvectors[:, i])
        if norm > 0:
            eigenvectors[:, i] = eigenvectors[:, i] / norm
    
    # 添加图结构信息
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    eigenvectors[:, 0] = degree / (np.max(degree) + 1e-8)
    
    if k_eigenvalues > 1:
        eigenvectors[:, 1] = np.log1p(degree) / (np.log1p(np.max(degree)) + 1)
    
    return eigenvectors.astype(np.float32)

def compute_smoothness_features(adj_matrix: sp.csr_matrix,
                              node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """计算图信号平滑度特征"""
    n_nodes = node_features.shape[0]
    
    # 计算归一化拉普拉斯矩阵
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree + 1e-8))
    L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
    
    features = {}
    
    # 计算每个特征的平滑度
    smoothness_scores = np.zeros((n_nodes, node_features.shape[1]), dtype=np.float32)
    
    for i in range(min(5, node_features.shape[1])):  # 只计算前5个特征
        feature_vec = node_features[:, i]
        # 平滑度：x^T L x
        smoothness = feature_vec * (L @ feature_vec)
        smoothness_scores[:, i] = smoothness
    
    # 统计量
    features['smoothness_mean'] = np.mean(smoothness_scores, axis=1).astype(np.float32)
    features['smoothness_max'] = np.max(smoothness_scores, axis=1).astype(np.float32)
    features['smoothness_std'] = np.std(smoothness_scores, axis=1).astype(np.float32)
    
    # 邻居平滑度一致性
    adj_matrix_csc = adj_matrix.tocsc()
    neighbor_smoothness = np.zeros(n_nodes, dtype=np.float32)
    
    for node in tqdm(range(min(10000, n_nodes)), desc="邻居平滑度", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) == 0:
            neighbor_smoothness[node] = 0
            continue
        
        # 计算节点与邻居平滑度的差异
        node_smoothness = features['smoothness_mean'][node]
        neighbor_smoothnesses = features['smoothness_mean'][neighbors]
        
        if len(neighbors) > 0:
            neighbor_smoothness[node] = np.mean(np.abs(node_smoothness - neighbor_smoothnesses))
    
    features['neighbor_smoothness_diff'] = neighbor_smoothness
    
    return features

def compute_spectral_clustering_features(adj_matrix: sp.csr_matrix,
                                       n_clusters: int = 10) -> np.ndarray:
    """计算谱聚类特征"""
    n_nodes = adj_matrix.shape[0]
    
    # 简化版谱聚类：使用度特征进行K-means
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 创建特征矩阵：度和度的变换
    features = np.column_stack([
        degree / (np.max(degree) + 1e-8),
        np.log1p(degree) / (np.log1p(np.max(degree)) + 1),
        np.sqrt(degree) / (np.sqrt(np.max(degree)) + 1)
    ])
    
    # 使用PCA或直接使用特征
    from sklearn.decomposition import PCA
    
    if n_nodes > n_clusters:
        pca = PCA(n_components=min(n_clusters, features.shape[1]))
        cluster_features = pca.fit_transform(features)
    else:
        cluster_features = features
    
    # 如果维度不够，填充零
    if cluster_features.shape[1] < n_clusters:
        padding = np.zeros((n_nodes, n_clusters - cluster_features.shape[1]), dtype=np.float32)
        cluster_features = np.hstack([cluster_features, padding])
    
    return cluster_features[:, :n_clusters].astype(np.float32)

def compute_chebyshev_features(adj_matrix: sp.csr_matrix,
                             node_features: np.ndarray,
                             order: int = 3) -> np.ndarray:
    """计算切比雪夫多项式特征"""
    n_nodes, n_features = node_features.shape
    
    # 归一化邻接矩阵
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    norm_adj = adj_matrix.multiply(1.0 / degree[:, np.newaxis])
    
    # 切比雪夫多项式展开
    chebyshev_features = []
    
    # T0(x) = x
    T0 = node_features
    chebyshev_features.append(T0)
    
    if order >= 1:
        # T1(x) = L x
        L = sp.eye(n_nodes) - norm_adj
        T1 = L @ node_features
        chebyshev_features.append(T1)
    
    if order >= 2:
        # T2(x) = 2L(T1(x)) - T0(x)
        T2 = 2 * (L @ T1) - T0
        chebyshev_features.append(T2)
    
    if order >= 3:
        # T3(x) = 2L(T2(x)) - T1(x)
        T3 = 2 * (L @ T2) - T1
        chebyshev_features.append(T3)
    
    # 合并所有阶数
    all_features = np.hstack(chebyshev_features)
    
    # 限制维度
    max_features = 20
    if all_features.shape[1] > max_features:
        # 选择方差最大的特征
        variances = np.var(all_features, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        all_features = all_features[:, top_indices]
    
    return all_features.astype(np.float32)

def compute_spectral_gap_features(adj_matrix: sp.csr_matrix) -> Dict[str, np.ndarray]:
    """计算谱间隙特征"""
    n_nodes = adj_matrix.shape[0]
    
    features = {}
    
    # 计算度
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 近似谱间隙：基于度的统计
    degree_sorted = np.sort(degree)
    
    # 谱间隙近似：最大度和次大度的差异
    if len(degree_sorted) >= 2:
        max_degree = degree_sorted[-1]
        second_max = degree_sorted[-2]
        spectral_gap_approx = (max_degree - second_max) / (max_degree + 1e-8)
    else:
        spectral_gap_approx = 0
    
    # 每个节点的度与最大度的比率
    if np.max(degree) > 0:
        degree_ratio = degree / np.max(degree)
        features['degree_spectral_gap'] = degree_ratio * spectral_gap_approx
    
    # 度分布的方差（谱展宽的度量）
    if len(degree) > 1:
        degree_variance = np.var(degree)
        features['degree_variance_norm'] = (degree_variance * np.ones(n_nodes) / (np.max(degree)**2 + 1e-8)).astype(np.float32)
    
    # 度的对数间隙
    log_degree = np.log1p(degree)
    if len(log_degree) >= 2:
        max_log = np.max(log_degree)
        min_log = np.min(log_degree)
        if max_log > min_log:
            log_gap = (log_degree - min_log) / (max_log - min_log)
            features['log_degree_gap'] = log_gap.astype(np.float32)
    
    return features

def compute_spectral_response_features(adj_matrix: sp.csr_matrix,
                                     node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """计算节点特征的谱响应"""
    n_nodes, n_features = node_features.shape
    
    features = {}
    
    # 计算归一化拉普拉斯矩阵
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree + 1e-8))
    L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
    
    # 低频响应（平滑分量）
    low_freq_response = np.zeros(n_nodes, dtype=np.float32)
    
    # 高频响应（细节分量）
    high_freq_response = np.zeros(n_nodes, dtype=np.float32)
    
    # 对每个特征计算谱响应
    for i in range(min(3, n_features)):  # 只计算前3个特征
        feature_vec = node_features[:, i]
        
        # 低频响应：L^{-1}的近似（平滑）
        try:
            # 使用雅可比迭代近似
            low_freq = feature_vec.copy()
            for _ in range(3):  # 3次迭代
                low_freq = 0.5 * (low_freq + (L @ low_freq))
            
            low_freq_response += np.abs(low_freq)
        except:
            pass
        
        # 高频响应：L的特征
        high_freq = L @ feature_vec
        high_freq_response += np.abs(high_freq)
    
    # 归一化
    if np.max(low_freq_response) > 0:
        low_freq_response = low_freq_response / np.max(low_freq_response)
    
    if np.max(high_freq_response) > 0:
        high_freq_response = high_freq_response / np.max(high_freq_response)
    
    features['low_freq_response'] = low_freq_response.astype(np.float32)
    features['high_freq_response'] = high_freq_response.astype(np.float32)
    features['spectral_ratio'] = np.divide(low_freq_response, high_freq_response + 1e-8).astype(np.float32)
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--k_eigenvalues', type=int, default=20, help='特征向量数量')
    parser.add_argument('--no_chebyshev', action='store_true', help='禁用切比雪夫特征')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    print(f"节点数: {x.shape[0]}, 特征维度: {x.shape[1]}")
    
    # 计算谱特征
    features_df = compute_spectral_features(
        edge_index, x,
        k_eigenvalues=args.k_eigenvalues,
        use_chebyshev=not args.no_chebyshev
    )
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"谱特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
