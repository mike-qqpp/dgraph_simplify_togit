"""
图嵌入特征模块：基于图结构的嵌入特征
"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # 添加这行
def compute_embedding_features(edge_index: np.ndarray,
                              node_features: np.ndarray,
                              embedding_dim: int = 64,
                              output_path: str = None) -> pd.DataFrame:
    """
    计算图嵌入特征：
    1. 邻接矩阵SVD
    2. 拉普拉斯特征向量
    3. 节点特征嵌入
    4. 度结构嵌入
    5. 随机游走统计
    6. 图神经网络风格嵌入
    """
    
    print("计算图嵌入特征...")
    n_nodes = node_features.shape[0]
    
    # 创建对称邻接矩阵
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    feature_dict = {}
    
    # 1. 邻接矩阵SVD（快速近似）
    print("计算邻接矩阵SVD...")
    svd_features = compute_adjacency_svd(
        adj_matrix_sym, n_components=min(32, embedding_dim, n_nodes-1)
    )
    
    for i in range(svd_features.shape[1]):
        feature_dict[f'svd_dim{i}'] = svd_features[:, i].astype(np.float32)
    
    # 2. 拉普拉斯矩阵特征向量
    print("计算拉普拉斯特征向量...")
    laplacian_features = compute_laplacian_eigenvectors(
        adj_matrix_sym, n_components=min(16, embedding_dim//2, n_nodes-1)
    )
    
    for i in range(laplacian_features.shape[1]):
        feature_dict[f'laplacian_dim{i}'] = laplacian_features[:, i].astype(np.float32)
    
    # 3. 基于节点特征的嵌入
    print("计算节点特征嵌入...")
    feature_embedding = compute_feature_based_embedding(
        node_features, embedding_dim=min(32, node_features.shape[1])
    )
    
    for i in range(feature_embedding.shape[1]):
        feature_dict[f'feature_embedding_dim{i}'] = feature_embedding[:, i].astype(np.float32)
    
    # 4. 度结构嵌入
    print("计算度结构嵌入...")
    degree_embedding = compute_degree_embedding(
        adj_matrix_sym, embedding_dim=min(16, embedding_dim//4)
    )
    
    for i in range(degree_embedding.shape[1]):
        feature_dict[f'degree_embedding_dim{i}'] = degree_embedding[:, i].astype(np.float32)
    
    # 5. 邻居聚合嵌入
    print("计算邻居聚合嵌入...")
    neighbor_embedding = compute_neighbor_embedding(
        adj_matrix_sym, node_features, embedding_dim=min(20, embedding_dim//3)
    )
    
    for i in range(neighbor_embedding.shape[1]):
        feature_dict[f'neighbor_embedding_dim{i}'] = neighbor_embedding[:, i].astype(np.float32)
    
    # 6. 图神经网络风格嵌入
    print("计算GNN风格嵌入...")
    gnn_embedding = compute_gnn_style_embedding(
        adj_matrix_sym, node_features, embedding_dim=min(24, embedding_dim//2)
    )
    
    for i in range(gnn_embedding.shape[1]):
        feature_dict[f'gnn_embedding_dim{i}'] = gnn_embedding[:, i].astype(np.float32)
    
    # 7. 随机游走统计特征
    print("计算随机游走统计...")
    rw_stats = compute_random_walk_stats_sampled(
        adj_matrix_sym, n_walks=20, walk_len=10, sample_size=min(3000, n_nodes)
    )
    
    for stat_name, stat_values in rw_stats.items():
        feature_dict[stat_name] = stat_values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    print(f"图嵌入特征维度: {features_df.shape}")
    return features_df

def compute_adjacency_svd(adj_matrix: sp.csr_matrix, n_components: int = 32) -> np.ndarray:
    """计算邻接矩阵的SVD降维"""
    try:
        # 使用TruncatedSVD进行快速降维
        svd = TruncatedSVD(
            n_components=min(n_components, adj_matrix.shape[0]-1),
            random_state=42,
            n_iter=5  # 减少迭代次数加快速度
        )
        svd_features = svd.fit_transform(adj_matrix)
        return svd_features.astype(np.float32)
    except Exception as e:
        print(f"SVD计算失败: {e}")
        # 返回零矩阵
        n_nodes = adj_matrix.shape[0]
        return np.zeros((n_nodes, n_components), dtype=np.float32)

def compute_laplacian_eigenvectors(adj_matrix: sp.csr_matrix, n_components: int = 16) -> np.ndarray:
    """计算拉普拉斯特征向量"""
    n_nodes = adj_matrix.shape[0]
    
    if n_nodes > 50000:
        # 对于超大图，使用快速近似
        print(f"图规模过大 ({n_nodes}节点)，使用快速近似...")
        return compute_fast_laplacian_approx(adj_matrix, n_components)
    
    try:
        from scipy.sparse.linalg import eigsh
        
        # 计算归一化拉普拉斯矩阵
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        degree_sqrt = np.sqrt(degree)
        degree_sqrt[degree_sqrt == 0] = 1
        
        D_sqrt_inv = sp.diags(1.0 / degree_sqrt)
        L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
        
        # 计算特征向量
        eigenvalues, eigenvectors = eigsh(
            L, 
            k=min(n_components+1, n_nodes-1), 
            which='SM', 
            maxiter=100
        )
        
        # 排除第一个零特征值对应的特征向量
        if eigenvectors.shape[1] > 1:
            return eigenvectors[:, 1:min(n_components+1, eigenvectors.shape[1])].astype(np.float32)
        else:
            return compute_fast_laplacian_approx(adj_matrix, n_components)
            
    except Exception as e:
        print(f"拉普拉斯特征向量计算失败: {e}")
        return compute_fast_laplacian_approx(adj_matrix, n_components)

def compute_fast_laplacian_approx(adj_matrix: sp.csr_matrix, n_components: int = 16) -> np.ndarray:
    """快速近似拉普拉斯特征向量"""
    n_nodes = adj_matrix.shape[0]
    
    # 计算度特征
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 生成特征矩阵
    features = np.zeros((n_nodes, n_components), dtype=np.float32)
    
    # 填充基于度的特征
    if degree.max() > 0:
        # 归一化的度
        features[:, 0] = degree / degree.max()
        
        # 度的对数
        if n_components > 1:
            features[:, 1] = np.log1p(degree) / np.log1p(degree.max())
        
        # 度的平方根
        if n_components > 2:
            features[:, 2] = np.sqrt(degree) / np.sqrt(degree.max())
        
        # PageRank近似
        if n_components > 3:
            features[:, 3] = degree / (degree.sum() + 1e-8)
        
        # 二阶邻居
        if n_components > 4:
            try:
                adj_squared = adj_matrix.dot(adj_matrix)
                second_neighbors = np.array(adj_squared.sum(axis=1)).flatten()
                if second_neighbors.max() > 0:
                    features[:, 4] = second_neighbors / second_neighbors.max()
            except:
                pass
    
    # 其他维度：使用三角函数投影
    for i in range(5, n_components):
        if i % 2 == 0:
            features[:, i] = np.sin(degree * (i / n_components)) * 0.5 + 0.5
        else:
            features[:, i] = np.cos(degree * (i / n_components)) * 0.5 + 0.5
    
    return features

def compute_feature_based_embedding(node_features: np.ndarray, embedding_dim: int = 32) -> np.ndarray:
    """基于节点特征的嵌入"""
    n_nodes, n_original_features = node_features.shape
    
    if n_original_features >= embedding_dim:
        # 如果原始特征维度大于等于目标维度，使用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(embedding_dim, n_nodes-1))
        embedding = pca.fit_transform(node_features)
    else:
        # 如果原始特征维度小于目标维度，直接使用原始特征并填充
        embedding = node_features.copy()
        
        # 如果需要填充
        if n_original_features < embedding_dim:
            # 使用随机投影扩展维度
            random_proj = np.random.randn(n_original_features, embedding_dim - n_original_features).astype(np.float32)
            random_features = node_features @ random_proj
            embedding = np.hstack([embedding, random_features])
    
    # 归一化
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
    
    return embedding.astype(np.float32)

def compute_degree_embedding(adj_matrix: sp.csr_matrix, embedding_dim: int = 16) -> np.ndarray:
    """基于度的嵌入"""
    n_nodes = adj_matrix.shape[0]
    
    # 计算度
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 收集度相关特征
    degree_features = []
    
    # 1. 原始度
    if degree.max() > 0:
        degree_features.append(degree / degree.max())
    
    # 2. 对数度
    degree_features.append(np.log1p(degree) / (np.log1p(degree.max()) + 1))
    
    # 3. 平方根度
    degree_features.append(np.sqrt(degree) / (np.sqrt(degree.max()) + 1))
    
    # 4. 归一化度（除以节点数）
    degree_features.append(degree / n_nodes)
    
    # 5. 度排名
    degree_features.append(np.argsort(np.argsort(degree)) / n_nodes)
    
    # 6. 度的累积分布
    sorted_degree = np.sort(degree)
    degree_cdf = np.searchsorted(sorted_degree, degree) / n_nodes
    degree_features.append(degree_cdf)
    
    # 转换为数组
    feature_matrix = np.column_stack(degree_features)
    
    # 如果维度不够，使用随机投影扩展
    if feature_matrix.shape[1] < embedding_dim:
        # 创建随机投影矩阵
        random_proj = np.random.randn(feature_matrix.shape[1], embedding_dim).astype(np.float32)
        embedding = feature_matrix @ random_proj
    else:
        embedding = feature_matrix[:, :embedding_dim]
    
    # 归一化
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
    
    return embedding.astype(np.float32)

def compute_neighbor_embedding(adj_matrix: sp.csr_matrix,
                             node_features: np.ndarray,
                             embedding_dim: int = 20) -> np.ndarray:
    """计算邻居聚合嵌入"""
    n_nodes, n_features = node_features.shape
    adj_matrix_csc = adj_matrix.tocsc()
    
    # 初始化嵌入矩阵
    embedding = np.zeros((n_nodes, embedding_dim), dtype=np.float32)
    
    # 采样计算（对大图）
    sample_size = min(5000, n_nodes)
    sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    
    for idx, node in enumerate(tqdm(sampled_nodes, desc="邻居嵌入", leave=False)):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) == 0:
            # 如果没有邻居，使用节点自身特征
            neighbor_features = node_features[node:node+1]
        else:
            # 限制邻居数量
            max_neighbors = min(50, len(neighbors))
            if len(neighbors) > max_neighbors:
                selected_neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
            else:
                selected_neighbors = neighbors
            
            neighbor_features = node_features[selected_neighbors]
        
        # 聚合邻居特征：均值池化
        if len(neighbor_features) > 0:
            neighbor_mean = np.mean(neighbor_features, axis=0)
            
            # 创建嵌入：节点特征和邻居特征的组合
            combined = np.hstack([
                node_features[node],
                neighbor_mean,
                node_features[node] - neighbor_mean  # 差异特征
            ])
            
            # 如果组合特征维度大于嵌入维度，使用随机投影
            if len(combined) > embedding_dim:
                # 使用固定随机投影保持一致性
                if not hasattr(compute_neighbor_embedding, 'proj_matrix'):
                    compute_neighbor_embedding.proj_matrix = np.random.randn(
                        len(combined), embedding_dim
                    ).astype(np.float32)
                
                embedding[node] = combined @ compute_neighbor_embedding.proj_matrix
            else:
                # 填充零
                embedding[node, :len(combined)] = combined
    
    # 填充未采样节点
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    if len(sampled_nodes) > 0:
        # 使用采样节点的均值填充
        mean_embedding = np.mean(embedding[sampled_nodes], axis=0)
        embedding[unsampled] = mean_embedding
    
    # 归一化
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
    
    return embedding.astype(np.float32)

def compute_gnn_style_embedding(adj_matrix: sp.csr_matrix,
                              node_features: np.ndarray,
                              embedding_dim: int = 24) -> np.ndarray:
    """计算GNN风格嵌入（模拟1层GCN）"""
    n_nodes, n_features = node_features.shape
    
    # 归一化邻接矩阵（GCN风格）
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_sqrt = np.sqrt(degree)
    degree_sqrt[degree_sqrt == 0] = 1
    
    D_sqrt_inv = sp.diags(1.0 / degree_sqrt)
    norm_adj = D_sqrt_inv @ adj_matrix @ D_sqrt_inv
    
    # 模拟GCN的传播：A'X
    propagated_features = norm_adj @ node_features
    
    # 组合原始特征和传播特征
    combined_features = np.hstack([
        node_features,
        propagated_features,
        node_features * propagated_features,  # 交互特征
        np.abs(node_features - propagated_features)  # 差异特征
    ])
    
    # 降维到目标维度
    if combined_features.shape[1] > embedding_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(embedding_dim, n_nodes-1))
        embedding = pca.fit_transform(combined_features)
    else:
        embedding = combined_features
        # 如果需要填充
        if embedding.shape[1] < embedding_dim:
            padding = np.random.randn(n_nodes, embedding_dim - embedding.shape[1]).astype(np.float32) * 0.1
            embedding = np.hstack([embedding, padding])
    
    return embedding.astype(np.float32)

def compute_random_walk_stats_sampled(adj_matrix: sp.csr_matrix,
                                     n_walks: int = 20,
                                     walk_len: int = 10,
                                     sample_size: int = 3000) -> Dict[str, np.ndarray]:
    """采样计算随机游走统计特征"""
    n_nodes = adj_matrix.shape[0]
    
    # 采样节点
    sampled_nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    
    # 初始化结果
    return_probs = np.zeros(n_nodes, dtype=np.float32)
    avg_return_steps = np.zeros(n_nodes, dtype=np.float32)
    visited_nodes_counts = np.zeros(n_nodes, dtype=np.float32)
    
    adj_matrix_csc = adj_matrix.tocsc()
    
    print(f"对 {len(sampled_nodes)} 个节点进行随机游走...")
    
    for idx, start_node in enumerate(tqdm(sampled_nodes, desc="随机游走", leave=False)):
        return_count = 0
        total_return_steps = 0
        visited_nodes = set()
        
        for walk in range(n_walks):
            current_node = start_node
            steps = 0
            
            for step in range(walk_len):
                # 获取邻居
                neighbors = adj_matrix_csc[:, current_node].indices
                
                if len(neighbors) == 0:
                    break
                
                # 随机选择下一个节点
                next_node = np.random.choice(neighbors)
                steps += 1
                
                # 记录访问的节点
                visited_nodes.add(next_node)
                
                # 检查是否返回起始节点
                if next_node == start_node:
                    return_count += 1
                    total_return_steps += steps
                    break
                
                current_node = next_node
        
        # 计算统计量
        if n_walks > 0:
            return_probs[start_node] = return_count / n_walks
            
            if return_count > 0:
                avg_return_steps[start_node] = total_return_steps / return_count
            
            visited_nodes_counts[start_node] = len(visited_nodes) / walk_len
    
    # 填充未采样节点
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    if len(sampled_nodes) > 0:
        mean_return_prob = return_probs[sampled_nodes].mean()
        mean_avg_steps = avg_return_steps[sampled_nodes].mean()
        mean_visited = visited_nodes_counts[sampled_nodes].mean()
        
        return_probs[unsampled] = mean_return_prob
        avg_return_steps[unsampled] = mean_avg_steps
        visited_nodes_counts[unsampled] = mean_visited
    
    return {
        'rw_return_prob': return_probs,
        'rw_avg_return_steps': avg_return_steps,
        'rw_visited_nodes': visited_nodes_counts
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    print(f"节点数: {x.shape[0]}, 特征维度: {x.shape[1]}")
    
    # 计算图嵌入特征
    features_df = compute_embedding_features(
        edge_index, x,
        embedding_dim=args.embedding_dim
    )
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"图嵌入特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
