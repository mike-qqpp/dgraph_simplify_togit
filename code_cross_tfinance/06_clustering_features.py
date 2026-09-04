"""
聚类特征模块：基于节点聚类的特征
"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # 添加这行
def compute_clustering_features(edge_index: np.ndarray,
                               node_features: np.ndarray,
                               n_clusters_list: list = [5, 10, 20],
                               use_sampling: bool = True,
                               sample_size: int = 10000,
                               output_path: str = None) -> pd.DataFrame:
    """
    计算聚类特征：
    1. K-means聚类标签
    2. 聚类中心距离
    3. 聚类内统计
    4. 聚类间关系
    5. 轮廓系数
    6. 邻居聚类分布
    """
    
    print("计算聚类特征...")
    n_nodes, n_features = node_features.shape
    
    feature_dict = {}
    
    # 标准化特征
    print("标准化节点特征...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(node_features)
    
    # 1. 对节点特征进行K-means聚类
    print("进行K-means聚类...")
    
    for n_clusters in n_clusters_list:
        if n_clusters >= n_nodes:
            print(f"聚类数 {n_clusters} 大于节点数，跳过...")
            continue
        
        print(f"  聚类数: {n_clusters}")
        
        # 使用MiniBatchKMeans加速
        if n_nodes > 50000 and use_sampling:
            print(f"  节点数过多 ({n_nodes})，使用采样聚类...")
            
            # 采样部分节点进行聚类训练
            if sample_size < n_nodes:
                sample_indices = np.random.choice(n_nodes, sample_size, replace=False)
                sample_features = features_scaled[sample_indices]
            else:
                sample_indices = np.arange(n_nodes)
                sample_features = features_scaled
            
            # 训练聚类模型
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                     random_state=42,
                                     batch_size=1000,
                                     n_init=3)
            kmeans.fit(sample_features)
            
            # 对所有节点预测聚类标签
            cluster_labels = kmeans.predict(features_scaled)
            cluster_centers = kmeans.cluster_centers_
            
        else:
            # 对中等规模图使用完整K-means
            if n_nodes > 10000:
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                         random_state=42,
                                         batch_size=1000,
                                         n_init=3)
            else:
                kmeans = KMeans(n_clusters=n_clusters, 
                               random_state=42,
                               n_init=10)
            
            cluster_labels = kmeans.fit_predict(features_scaled)
            cluster_centers = kmeans.cluster_centers_
        
        # 保存聚类标签
        feature_dict[f'kmeans_{n_clusters}_cluster'] = cluster_labels.astype(np.float32)
        
        # 计算到聚类中心的距离
        print(f"    计算聚类距离...")
        distances = np.zeros((n_nodes, n_clusters), dtype=np.float32)
        
        # 分批计算距离以节省内存
        batch_size = 10000
        for i in tqdm(range(0, n_nodes, batch_size), desc="计算距离", leave=False):
            end_idx = min(i + batch_size, n_nodes)
            batch_features = features_scaled[i:end_idx]
            
            # 计算到所有聚类中心的距离
            for j in range(n_clusters):
                dist = np.linalg.norm(batch_features - cluster_centers[j], axis=1)
                distances[i:end_idx, j] = dist
        
        # 最近聚类距离
        min_distances = np.min(distances, axis=1)
        feature_dict[f'kmeans_{n_clusters}_min_dist'] = min_distances.astype(np.float32)
        
        # 最近聚类索引
        nearest_clusters = np.argmin(distances, axis=1)
        feature_dict[f'kmeans_{n_clusters}_nearest_cluster'] = nearest_clusters.astype(np.float32)
        
        # 第二近聚类距离
        for i in range(n_nodes):
            distances[i, nearest_clusters[i]] = np.inf  # 将最近距离设为无穷大
        second_min_distances = np.min(distances, axis=1)
        feature_dict[f'kmeans_{n_clusters}_second_min_dist'] = second_min_distances.astype(np.float32)
        
        # 聚类内节点数统计
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
        feature_dict[f'kmeans_{n_clusters}_cluster_size'] = cluster_sizes[cluster_labels].astype(np.float32)
        feature_dict[f'kmeans_{n_clusters}_cluster_size_norm'] = (cluster_sizes[cluster_labels] / n_nodes).astype(np.float32)
        
        # 聚类内密度（平均距离）
        print(f"    计算聚类密度...")
        cluster_density = np.zeros(n_nodes, dtype=np.float32)
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 1:
                cluster_points = features_scaled[cluster_mask]
                # 计算平均距离
                if len(cluster_points) <= 1000:
                    # 对小聚类计算精确距离
                    from scipy.spatial.distance import pdist
                    pairwise_dist = pdist(cluster_points)
                    avg_dist = np.mean(pairwise_dist)
                else:
                    # 对大聚类采样计算
                    sample_idx = np.random.choice(np.sum(cluster_mask), min(1000, np.sum(cluster_mask)), replace=False)
                    sample_points = cluster_points[sample_idx]
                    from scipy.spatial.distance import cdist
                    pairwise_dist = cdist(sample_points, sample_points)
                    avg_dist = np.mean(pairwise_dist[np.triu_indices(len(sample_points), k=1)])
                
                cluster_density[cluster_mask] = avg_dist
        
        feature_dict[f'kmeans_{n_clusters}_cluster_density'] = cluster_density.astype(np.float32)
        
        # 计算轮廓系数（采样计算）
        if n_nodes <= 10000:  # 只对小图计算轮廓系数
            try:
                print(f"    计算轮廓系数...")
                silhouette_vals = silhouette_samples(features_scaled, cluster_labels)
                feature_dict[f'kmeans_{n_clusters}_silhouette'] = silhouette_vals.astype(np.float32)
            except:
                print(f"    轮廓系数计算失败，跳过...")
                feature_dict[f'kmeans_{n_clusters}_silhouette'] = np.zeros(n_nodes, dtype=np.float32)
        else:
            feature_dict[f'kmeans_{n_clusters}_silhouette'] = np.zeros(n_nodes, dtype=np.float32)
    
    # 2. 创建邻接矩阵
    print("创建邻接矩阵...")
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    # 3. 邻居聚类一致性（只对少量聚类计算）
    print("计算邻居聚类一致性...")
    for n_clusters in [5, 10]:  # 只计算少量聚类的邻居一致性
        cluster_key = f'kmeans_{n_clusters}_cluster'
        if cluster_key in feature_dict:
            print(f"  计算{n_clusters}聚类的邻居一致性...")
            cluster_labels = feature_dict[cluster_key].astype(int)
            neighbor_cluster_features = compute_neighbor_cluster_features(
                adj_matrix_sym, cluster_labels
            )
            
            for feat_name, values in neighbor_cluster_features.items():
                feature_dict[f'{cluster_key}_{feat_name}'] = values.astype(np.float32)
    
    # 4. 聚类间连接特征
    print("计算聚类间连接特征...")
    if 'kmeans_5_cluster' in feature_dict:
        cluster_labels = feature_dict['kmeans_5_cluster'].astype(int)
        inter_cluster_features = compute_inter_cluster_features(
            adj_matrix_sym, cluster_labels
        )
        
        for feat_name, values in inter_cluster_features.items():
            feature_dict[f'inter_cluster_{feat_name}'] = values.astype(np.float32)
    
    # 5. 基于度的聚类特征
    print("计算基于度的聚类特征...")
    degree_based_features = compute_degree_based_cluster_features(
        adj_matrix_sym, features_scaled
    )
    
    for feat_name, values in degree_based_features.items():
        feature_dict[f'degree_cluster_{feat_name}'] = values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    print(f"聚类特征维度: {features_df.shape}")
    return features_df

def compute_neighbor_cluster_features(adj_matrix: sp.csr_matrix,
                                     cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """计算邻居聚类特征"""
    n_nodes = adj_matrix.shape[0]
    n_clusters = len(np.unique(cluster_labels))
    adj_matrix_csc = adj_matrix.tocsc()
    
    # 初始化特征
    consistency = np.zeros(n_nodes, dtype=np.float32)  # 邻居聚类一致性
    entropy = np.zeros(n_nodes, dtype=np.float32)  # 邻居聚类分布熵
    dominant_cluster_ratio = np.zeros(n_nodes, dtype=np.float32)  # 主导聚类比例
    
    for node in tqdm(range(n_nodes), desc="邻居聚类特征", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) == 0:
            consistency[node] = 0
            entropy[node] = 0
            dominant_cluster_ratio[node] = 0
            continue
        
        # 统计邻居聚类分布
        neighbor_clusters = cluster_labels[neighbors]
        node_cluster = cluster_labels[node]
        
        # 1. 聚类一致性
        same_cluster_count = np.sum(neighbor_clusters == node_cluster)
        consistency[node] = same_cluster_count / len(neighbors)
        
        # 2. 聚类分布熵
        cluster_counts = np.bincount(neighbor_clusters, minlength=n_clusters)
        cluster_probs = cluster_counts / len(neighbors)
        
        # 计算熵（避免log(0)）
        non_zero_probs = cluster_probs[cluster_probs > 0]
        if len(non_zero_probs) > 0:
            entropy[node] = -np.sum(non_zero_probs * np.log(non_zero_probs)) / np.log(n_clusters)
        
        # 3. 主导聚类比例
        if len(cluster_counts) > 0:
            dominant_cluster_ratio[node] = np.max(cluster_counts) / len(neighbors)
    
    return {
        'neighbor_consistency': consistency,
        'neighbor_cluster_entropy': entropy,
        'neighbor_dominant_ratio': dominant_cluster_ratio
    }

def compute_inter_cluster_features(adj_matrix: sp.csr_matrix,
                                 cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """计算聚类间连接特征"""
    n_nodes = adj_matrix.shape[0]
    n_clusters = len(np.unique(cluster_labels))
    
    adj_matrix_csc = adj_matrix.tocsc()
    
    # 初始化特征
    intra_cluster_edges = np.zeros(n_nodes, dtype=np.float32)
    inter_cluster_edges = np.zeros(n_nodes, dtype=np.float32)
    cluster_degree = np.zeros(n_nodes, dtype=np.float32)
    cluster_bridging = np.zeros(n_nodes, dtype=np.float32)  # 桥接系数
    
    # 预计算聚类大小
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
    
    for node in tqdm(range(n_nodes), desc="聚类间连接", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) == 0:
            continue
        
        node_cluster = cluster_labels[node]
        
        # 统计连接类型
        intra_count = 0
        neighbor_clusters = set()
        
        for neighbor in neighbors:
            neighbor_cluster = cluster_labels[neighbor]
            if neighbor_cluster == node_cluster:
                intra_count += 1
            neighbor_clusters.add(neighbor_cluster)
        
        intra_cluster_edges[node] = intra_count
        inter_cluster_edges[node] = len(neighbors) - intra_count
        
        # 聚类内归一化度
        if cluster_sizes[node_cluster] > 1:
            cluster_degree[node] = intra_count / (cluster_sizes[node_cluster] - 1)
        
        # 桥接系数：连接的不同聚类数
        cluster_bridging[node] = len(neighbor_clusters)
    
    # 计算比率
    total_edges = intra_cluster_edges + inter_cluster_edges
    intra_ratio = np.divide(intra_cluster_edges, total_edges, where=total_edges>0)
    inter_ratio = np.divide(inter_cluster_edges, total_edges, where=total_edges>0)
    
    # 归一化桥接系数
    cluster_bridging_norm = cluster_bridging / n_clusters
    
    return {
        'intra_edges': intra_cluster_edges,
        'inter_edges': inter_cluster_edges,
        'intra_ratio': intra_ratio,
        'inter_ratio': inter_ratio,
        'cluster_degree': cluster_degree,
        'cluster_bridging': cluster_bridging,
        'cluster_bridging_norm': cluster_bridging_norm
    }

def compute_degree_based_cluster_features(adj_matrix: sp.csr_matrix,
                                         node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """基于度的聚类特征"""
    n_nodes = node_features.shape[0]
    
    # 计算度
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 基于度的简单聚类
    degree_percentiles = np.percentile(degree, [25, 50, 75, 90])
    degree_clusters = np.zeros(n_nodes, dtype=int)
    
    for i, node in enumerate(range(n_nodes)):
        if degree[node] <= degree_percentiles[0]:
            degree_clusters[i] = 0  # 低度节点
        elif degree[node] <= degree_percentiles[1]:
            degree_clusters[i] = 1  # 中低度节点
        elif degree[node] <= degree_percentiles[2]:
            degree_clusters[i] = 2  # 中高度节点
        elif degree[node] <= degree_percentiles[3]:
            degree_clusters[i] = 3  # 高度节点
        else:
            degree_clusters[i] = 4  # 超高度节点
    
    # 初始化特征
    features = {}
    features['degree_cluster'] = degree_clusters.astype(np.float32)
    
    # 各聚类内的平均特征
    for cluster_id in range(5):
        cluster_mask = degree_clusters == cluster_id
        if np.sum(cluster_mask) > 0:
            # 计算聚类内节点特征的平均值
            cluster_features = node_features[cluster_mask]
            avg_features = np.mean(cluster_features, axis=0)
            
            # 计算每个节点到聚类中心的距离
            for feat_idx in range(min(3, node_features.shape[1])):  # 只计算前3个特征
                feat_name = f'degree_cluster_{cluster_id}_feat{feat_idx}_dist'
                distances = np.abs(node_features[:, feat_idx] - avg_features[feat_idx])
                features[feat_name] = distances.astype(np.float32)
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--n_clusters', type=str, default='5,10,20', help='聚类数量列表，逗号分隔')
    parser.add_argument('--sample_size', type=int, default=10000, help='采样大小（大图时使用）')
    parser.add_argument('--no_sampling', action='store_true', help='禁用采样（小图时使用）')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    print(f"节点数: {x.shape[0]}, 特征维度: {x.shape[1]}")
    
    # 解析聚类数量
    n_clusters_list = [int(n) for n in args.n_clusters.split(',')]
    
    # 计算聚类特征
    features_df = compute_clustering_features(
        edge_index, x, 
        n_clusters_list,
        use_sampling=not args.no_sampling,
        sample_size=args.sample_size
    )
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"聚类特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
