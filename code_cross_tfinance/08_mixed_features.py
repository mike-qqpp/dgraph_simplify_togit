"""
混合高阶特征模块：组合多种特征的高阶特征
"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # 添加这行
def compute_mixed_features(edge_index: np.ndarray,
                          node_features: np.ndarray,
                          n_components: int = 50,
                          output_path: str = None) -> pd.DataFrame:
    """
    计算混合高阶特征：
    1. 多项式特征（交互特征）
    2. PCA主成分
    3. ICA独立成分
    4. 随机投影特征
    5. 特征选择得分
    6. 特征重要性排序
    7. 特征组合统计
    8. 图增强特征
    """
    
    print("计算混合高阶特征...")
    n_nodes, n_features = node_features.shape
    
    feature_dict = {}
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(node_features)
    
    # 1. 多项式特征（2阶）
    print("计算多项式特征...")
    poly_features = compute_polynomial_features(features_scaled, degree=2)
    
    for i in range(min(20, poly_features.shape[1])):  # 只取前20个多项式特征
        feature_dict[f'poly_feat_{i}'] = poly_features[:, i].astype(np.float32)
    
    # 2. PCA主成分
    print("计算PCA主成分...")
    pca_features = compute_pca_features(features_scaled, n_components=min(20, n_components, n_features))
    
    for i in range(pca_features.shape[1]):
        feature_dict[f'pca_component_{i}'] = pca_features[:, i].astype(np.float32)
    
    # 3. ICA独立成分
    print("计算ICA独立成分...")
    ica_features = compute_ica_features(features_scaled, n_components=min(10, n_components//2, n_features))
    
    for i in range(ica_features.shape[1]):
        feature_dict[f'ica_component_{i}'] = ica_features[:, i].astype(np.float32)
    
    # 4. 随机投影特征
    print("计算随机投影特征...")
    random_proj_features = compute_random_projection_features(
        features_scaled, n_components=min(15, n_components)
    )
    
    for i in range(random_proj_features.shape[1]):
        feature_dict[f'random_proj_{i}'] = random_proj_features[:, i].astype(np.float32)
    
    # 5. 特征选择得分（基于方差和相关性）
    print("计算特征选择得分...")
    feature_scores = compute_feature_selection_scores(features_scaled)
    
    for feat_name, values in feature_scores.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 6. 特征重要性特征
    print("计算特征重要性特征...")
    importance_features = compute_feature_importance_features(features_scaled)
    
    for feat_name, values in importance_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 7. 特征组合统计
    print("计算特征组合统计...")
    combination_features = compute_feature_combination_stats(features_scaled)
    
    for feat_name, values in combination_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 8. 图增强特征
    print("计算图增强特征...")
    graph_enhanced_features = compute_graph_enhanced_features(edge_index, features_scaled)
    
    for feat_name, values in graph_enhanced_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 9. 特征复杂度度量
    print("计算特征复杂度...")
    complexity_features = compute_feature_complexity(features_scaled)
    
    for feat_name, values in complexity_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    print(f"混合高阶特征维度: {features_df.shape}")
    return features_df

def compute_polynomial_features(features_scaled: np.ndarray, degree: int = 2) -> np.ndarray:
    """计算多项式特征"""
    n_samples, n_features = features_scaled.shape
    
    # 限制特征数量以避免维度爆炸
    max_features_for_poly = min(8, n_features)
    features_for_poly = features_scaled[:, :max_features_for_poly]
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=True,  # 只计算交互项，避免平方项
            include_bias=False
        )
        
        poly_features = poly.fit_transform(features_for_poly)
        
        # 如果维度太大，进行截断
        max_poly_features = 50
        if poly_features.shape[1] > max_poly_features:
            # 选择方差最大的特征
            variances = np.var(poly_features, axis=0)
            top_indices = np.argsort(variances)[-max_poly_features:]
            poly_features = poly_features[:, top_indices]
        
        # 如果原始特征维度较小，填充到目标维度
        if poly_features.shape[1] < 20:
            padding = np.zeros((n_samples, 20 - poly_features.shape[1]), dtype=np.float32)
            poly_features = np.hstack([poly_features, padding])
        
        return poly_features.astype(np.float32)
    
    except Exception as e:
        print(f"多项式特征计算失败: {e}")
        # 返回简单交互特征
        simple_poly = np.zeros((n_samples, 20), dtype=np.float32)
        
        # 添加一些简单的交互特征
        idx = 0
        for i in range(min(4, n_features)):
            for j in range(i+1, min(5, n_features)):
                if idx < 20:
                    simple_poly[:, idx] = features_scaled[:, i] * features_scaled[:, j]
                    idx += 1
        
        return simple_poly

def compute_pca_features(features_scaled: np.ndarray, n_components: int = 20) -> np.ndarray:
    """计算PCA主成分"""
    n_samples = features_scaled.shape[0]
    
    if n_components >= n_samples:
        n_components = min(10, n_samples - 1)
    
    try:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(n_components, features_scaled.shape[1], n_samples-1),
                 random_state=42)
        
        pca_features = pca.fit_transform(features_scaled)
        
        # 解释方差比例作为额外特征
        explained_variance_ratio = pca.explained_variance_ratio_
        
        return pca_features.astype(np.float32)
    
    except Exception as e:
        print(f"PCA计算失败: {e}")
        return np.zeros((n_samples, n_components), dtype=np.float32)

def compute_ica_features(features_scaled: np.ndarray, n_components: int = 10) -> np.ndarray:
    """计算ICA独立成分"""
    n_samples = features_scaled.shape[0]
    
    if n_components >= n_samples:
        n_components = min(5, n_samples - 1)
    
    try:
        from sklearn.decomposition import FastICA
        
        ica = FastICA(n_components=min(n_components, features_scaled.shape[1]),
                     random_state=42,
                     max_iter=100)
        
        ica_features = ica.fit_transform(features_scaled)
        
        return ica_features.astype(np.float32)
    
    except Exception as e:
        print(f"ICA计算失败: {e}")
        return np.zeros((n_samples, n_components), dtype=np.float32)

def compute_random_projection_features(features_scaled: np.ndarray, n_components: int = 15) -> np.ndarray:
    """计算随机投影特征"""
    n_samples = features_scaled.shape[0]
    
    try:
        from sklearn.random_projection import GaussianRandomProjection
        
        rp = GaussianRandomProjection(n_components=min(n_components, features_scaled.shape[1]),
                                     random_state=42)
        
        rp_features = rp.fit_transform(features_scaled)
        
        return rp_features.astype(np.float32)
    
    except Exception as e:
        print(f"随机投影计算失败: {e}")
        # 使用简单随机投影
        if features_scaled.shape[1] > 0:
            random_matrix = np.random.randn(features_scaled.shape[1], n_components).astype(np.float32)
            rp_features = features_scaled @ random_matrix
            return rp_features
        else:
            return np.zeros((n_samples, n_components), dtype=np.float32)

def compute_feature_selection_scores(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """计算特征选择得分"""
    n_samples, n_features = features_scaled.shape
    
    features = {}
    
    # 1. 方差得分
    variances = np.var(features_scaled, axis=0)
    features['feature_variance_score'] = np.mean(variances) * np.ones(n_samples, dtype=np.float32)
    
    # 2. 互相关性得分
    if n_features > 1 and n_samples > 100:
        try:
            # 计算特征间的平均相关性
            correlation_matrix = np.corrcoef(features_scaled.T)
            np.fill_diagonal(correlation_matrix, 0)  # 忽略自相关
            
            # 平均绝对相关性
            avg_correlation = np.mean(np.abs(correlation_matrix))
            features['feature_avg_correlation'] = avg_correlation * np.ones(n_samples, dtype=np.float32)
            
            # 最大相关性
            max_correlation = np.max(np.abs(correlation_matrix))
            features['feature_max_correlation'] = max_correlation * np.ones(n_samples, dtype=np.float32)
        
        except:
            features['feature_avg_correlation'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_max_correlation'] = np.zeros(n_samples, dtype=np.float32)
    
    # 3. 特征稳定性得分（通过bootstrap）
    if n_samples > 100 and n_features > 1:
        stability_scores = compute_feature_stability(features_scaled)
        features['feature_stability_score'] = stability_scores.astype(np.float32)
    
    return features

def compute_feature_stability(features_scaled: np.ndarray, n_bootstrap: int = 10) -> np.ndarray:
    """通过bootstrap计算特征稳定性"""
    n_samples, n_features = features_scaled.shape
    
    if n_samples < 50 or n_features < 2:
        return np.ones(n_samples, dtype=np.float32)
    
    # 存储每次bootstrap的特征重要性（使用方差作为重要性）
    importances_list = []
    
    for _ in range(min(n_bootstrap, 5)):
        # 有放回采样
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_features = features_scaled[bootstrap_indices]
        
        # 计算方差作为重要性
        variances = np.var(bootstrap_features, axis=0)
        importances_list.append(variances)
    
    # 计算重要性的一致性
    importances_array = np.array(importances_list)  # shape: (n_bootstrap, n_features)
    
    # 计算每个特征的稳定性（方差的倒数）
    stability_per_feature = 1.0 / (np.std(importances_array, axis=0) + 1e-8)
    
    # 平均稳定性作为每个样本的得分
    avg_stability = np.mean(stability_per_feature)
    
    return avg_stability * np.ones(n_samples, dtype=np.float32)

def compute_feature_importance_features(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """计算特征重要性相关特征"""
    n_samples, n_features = features_scaled.shape
    
    features = {}
    
    # 使用随机森林计算特征重要性（采样计算）
    if n_samples > 100 and n_features > 1:
        try:
            # 生成伪标签用于特征重要性计算
            # 使用K-means聚类生成伪标签
            from sklearn.cluster import KMeans
            
            n_clusters = min(5, n_samples // 20)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                pseudo_labels = kmeans.fit_predict(features_scaled)
                
                # 使用随机森林计算特征重要性
                from sklearn.ensemble import RandomForestClassifier
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(features_scaled, pseudo_labels)
                
                importances = rf.feature_importances_
                
                # 特征重要性统计
                features['feature_importance_mean'] = np.mean(importances) * np.ones(n_samples, dtype=np.float32)
                features['feature_importance_std'] = np.std(importances) * np.ones(n_samples, dtype=np.float32)
                features['feature_importance_max'] = np.max(importances) * np.ones(n_samples, dtype=np.float32)
                
                # 每个样本的特征加权和
                weighted_sum = features_scaled @ importances
                features['feature_weighted_sum'] = weighted_sum.astype(np.float32)
        
        except Exception as e:
            print(f"特征重要性计算失败: {e}")
            features['feature_importance_mean'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_importance_std'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_importance_max'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_weighted_sum'] = np.zeros(n_samples, dtype=np.float32)
    
    return features

def compute_feature_combination_stats(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """计算特征组合统计"""
    n_samples, n_features = features_scaled.shape
    
    features = {}
    
    # 1. 特征值和（L1范数）
    features['feature_l1_norm'] = np.sum(np.abs(features_scaled), axis=1).astype(np.float32)
    
    # 2. 特征值平方和（L2范数）
    features['feature_l2_norm'] = np.sqrt(np.sum(features_scaled**2, axis=1)).astype(np.float32)
    
    # 3. 特征最大值
    features['feature_max'] = np.max(features_scaled, axis=1).astype(np.float32)
    
    # 4. 特征最小值
    features['feature_min'] = np.min(features_scaled, axis=1).astype(np.float32)
    
    # 5. 特征范围
    features['feature_range'] = features['feature_max'] - features['feature_min']
    
    # 6. 特征均值
    features['feature_mean'] = np.mean(features_scaled, axis=1).astype(np.float32)
    
    # 7. 特征标准差
    if n_features > 1:
        features['feature_std'] = np.std(features_scaled, axis=1).astype(np.float32)
    
    # 8. 特征偏度
    if n_features > 2:
        from scipy import stats
        features['feature_skewness'] = stats.skew(features_scaled, axis=1).astype(np.float32)
    
    # 9. 特征峰度
    if n_features > 3:
        features['feature_kurtosis'] = stats.kurtosis(features_scaled, axis=1).astype(np.float32)
    
    # 10. 特征熵（信息量）
    # 将特征值转换为概率分布
    feature_probs = np.abs(features_scaled) / (np.sum(np.abs(features_scaled), axis=1, keepdims=True) + 1e-8)
    feature_entropy = -np.sum(feature_probs * np.log(feature_probs + 1e-8), axis=1)
    features['feature_entropy'] = feature_entropy.astype(np.float32)
    
    # 11. 特征基尼系数（不均匀性）
    sorted_features = np.sort(np.abs(features_scaled), axis=1)
    cumulative = np.cumsum(sorted_features, axis=1)
    
    if n_features > 1:
        # 归一化
        cumulative_norm = cumulative / (cumulative[:, -1:] + 1e-8)
        
        # 计算基尼系数
        gini = 1 - 2 * np.mean(cumulative_norm, axis=1)
        features['feature_gini'] = gini.astype(np.float32)
    
    return features

def compute_graph_enhanced_features(edge_index: np.ndarray,
                                  features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """计算图增强特征"""
    n_nodes = features_scaled.shape[0]
    
    # 创建邻接矩阵
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    features = {}
    
    # 1. 图拉普拉斯平滑特征
    print("    计算图拉普拉斯平滑...")
    try:
        # 计算归一化拉普拉斯矩阵
        degree = np.array(adj_matrix_sym.sum(axis=1)).flatten()
        degree_sqrt = np.sqrt(degree)
        degree_sqrt[degree_sqrt == 0] = 1
        
        D_sqrt_inv = sp.diags(1.0 / degree_sqrt)
        L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix_sym @ D_sqrt_inv
        
        # 计算平滑度：x^T L x
        smoothness = np.zeros(n_nodes, dtype=np.float32)
        for i in range(min(3, features_scaled.shape[1])):  # 只计算前3个特征
            feature_vec = features_scaled[:, i]
            smoothness += feature_vec * (L @ feature_vec)
        
        features['graph_smoothness'] = smoothness.astype(np.float32)
    
    except:
        features['graph_smoothness'] = np.zeros(n_nodes, dtype=np.float32)
    
    # 2. 图扩散特征
    print("    计算图扩散特征...")
    diffusion_features = compute_graph_diffusion_features(adj_matrix_sym, features_scaled)
    
    for feat_name, values in diffusion_features.items():
        features[f'graph_diffusion_{feat_name}'] = values.astype(np.float32)
    
    # 3. 图注意力特征
    print("    计算图注意力特征...")
    attention_features = compute_graph_attention_features(adj_matrix_sym, features_scaled)
    
    for feat_name, values in attention_features.items():
        features[f'graph_attention_{feat_name}'] = values.astype(np.float32)
    
    return features

def compute_graph_diffusion_features(adj_matrix: sp.csr_matrix,
                                   features_scaled: np.ndarray,
                                   diffusion_steps: int = 2) -> Dict[str, np.ndarray]:
    """计算图扩散特征"""
    n_nodes = features_scaled.shape[0]
    
    features = {}
    
    try:
        # 归一化邻接矩阵
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        norm_adj = adj_matrix.multiply(1.0 / degree[:, np.newaxis])
        
        # 初始特征
        diffused_features = features_scaled.copy()
        
        for step in range(1, diffusion_steps + 1):
            # 扩散：A * X
            diffused_features = norm_adj @ diffused_features
            
            # 保存每一步的统计量
            if step <= 2:  # 只保存前两步
                # 均值
                features[f'step{step}_mean'] = np.mean(diffused_features, axis=1).astype(np.float32)
                
                # 标准差
                if diffused_features.shape[1] > 1:
                    features[f'step{step}_std'] = np.std(diffused_features, axis=1).astype(np.float32)
                
                # 与原特征的差异
                if step == 1:
                    diff_norm = np.linalg.norm(diffused_features - features_scaled, axis=1)
                    features['diffusion_change_norm'] = diff_norm.astype(np.float32)
    
    except Exception as e:
        print(f"图扩散计算失败: {e}")
        features['diffusion_change_norm'] = np.zeros(n_nodes, dtype=np.float32)
    
    return features

def compute_graph_attention_features(adj_matrix: sp.csr_matrix,
                                   features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """计算简单的图注意力特征"""
    n_nodes = features_scaled.shape[0]
    adj_matrix_csc = adj_matrix.tocsc()
    
    features = {}
    
    # 简单的注意力机制：基于特征相似度
    attention_scores = np.zeros(n_nodes, dtype=np.float32)
    
    # 只计算部分节点的注意力
    sample_size = min(5000, n_nodes)
    sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    
    for node in tqdm(sampled_nodes, desc="图注意力", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) == 0:
            attention_scores[node] = 0
            continue
        
        # 计算节点特征与邻居特征的余弦相似度
        node_feat = features_scaled[node]
        neighbor_feats = features_scaled[neighbors]
        
        # 余弦相似度
        norms = np.linalg.norm(node_feat) * np.linalg.norm(neighbor_feats, axis=1)
        valid_mask = norms > 0
        
        if np.any(valid_mask):
            similarities = np.zeros(len(neighbors))
            similarities[valid_mask] = (node_feat @ neighbor_feats[valid_mask].T) / norms[valid_mask]
            
            # 注意力得分：平均相似度
            attention_scores[node] = np.mean(similarities[valid_mask])
    
    # 填充未采样节点
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    if len(sampled_nodes) > 0:
        mean_attention = attention_scores[sampled_nodes].mean()
        attention_scores[unsampled] = mean_attention
    
    features['attention_score'] = attention_scores
    
    return features

def compute_feature_complexity(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """计算特征复杂度度量"""
    n_samples = features_scaled.shape[0]
    
    features = {}
    
    # 1. 特征维度（常数）
    features['feature_dimensionality'] = np.ones(n_samples, dtype=np.float32) * features_scaled.shape[1]
    
    # 2. 有效维度（基于特征值）
    try:
        # 计算协方差矩阵的特征值
        if n_samples > 100:
            cov_matrix = np.cov(features_scaled.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # 计算有效维度（特征值熵）
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-8)
            effective_dim = np.exp(-np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-8)))
            
            features['feature_effective_dim'] = effective_dim * np.ones(n_samples, dtype=np.float32)
        else:
            features['feature_effective_dim'] = np.ones(n_samples, dtype=np.float32) * features_scaled.shape[1]
    
    except:
        features['feature_effective_dim'] = np.ones(n_samples, dtype=np.float32) * features_scaled.shape[1]
    
    # 3. 特征冗余度（基于相关性）
    if features_scaled.shape[1] > 1:
        try:
            correlation_matrix = np.corrcoef(features_scaled.T)
            np.fill_diagonal(correlation_matrix, 0)
            
            # 平均绝对相关性作为冗余度量
            redundancy = np.mean(np.abs(correlation_matrix))
            features['feature_redundancy'] = redundancy * np.ones(n_samples, dtype=np.float32)
        
        except:
            features['feature_redundancy'] = np.zeros(n_samples, dtype=np.float32)
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--n_components', type=int, default=50, help='降维组件数')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    print(f"节点数: {x.shape[0]}, 特征维度: {x.shape[1]}")
    
    # 计算混合高阶特征
    features_df = compute_mixed_features(
        edge_index, x, 
        n_components=args.n_components
    )
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"混合高阶特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
