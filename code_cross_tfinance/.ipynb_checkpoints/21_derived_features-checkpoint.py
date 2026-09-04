"""
特征衍生模块：基于现有特征创建高阶衍生特征
"""
import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple
import warnings
from scipy import stats
from itertools import combinations
from tqdm import tqdm
warnings.filterwarnings('ignore')

def compute_derived_features(feature_paths: List[str],
                            output_path: str = None) -> pd.DataFrame:
    """
    计算衍生特征：
    1. 特征交叉乘积（高阶交互）
    2. 特征比率和差异
    3. 特征分箱和统计
    4. 特征多项式展开
    5. 特征聚类衍生
    6. 异常特征组合
    """
    print("加载现有特征...")
    
    # 加载所有特征文件
    all_features = []
    for path in feature_paths:
        try:
            with open(path, 'rb') as f:
                features_df = pd.read_pickle(f)
                all_features.append(features_df)
                print(f"加载: {path}, 形状: {features_df.shape}")
        except Exception as e:
            print(f"加载失败 {path}: {e}")
    
    if not all_features:
        raise ValueError("没有成功加载任何特征文件")
    
    # 合并所有特征
    print("合并特征...")
    combined_df = pd.concat(all_features, axis=1)
    
    # 移除重复列
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    print(f"合并后特征维度: {combined_df.shape}")
    print(f"特征示例: {list(combined_df.columns[:20])}")
    
    feature_dict = {}
    
    # 1. 重要特征的衍生（基于您提供的头部特征）
    print("计算重要特征衍生...")
    important_features = [
        'feature_1', 'feature_2', 'feature_embedding_dim0', 
        'laplacian_dim0', 'gnn_embedding_dim0', 'isolation_forest_score',
        'lof_score', 'one_class_svm_score', 'feature_entropy'
    ]
    
    important_derived = compute_important_features_derived(combined_df, important_features)
    for feat_name, values in important_derived.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 2. 异常检测特征组合衍生
    print("计算异常特征组合...")
    anomaly_features = [col for col in combined_df.columns 
                       if 'score' in col.lower() or 'anomaly' in col.lower()]
    
    if anomaly_features:
        anomaly_derived = compute_anomaly_features_derived(combined_df, anomaly_features[:10])
        for feat_name, values in anomaly_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)
    
    # 3. 图嵌入特征交叉衍生
    print("计算图嵌入特征交叉...")
    embedding_features = [col for col in combined_df.columns 
                         if 'embedding' in col.lower() or 'dim' in col.lower()]
    
    if embedding_features:
        embedding_derived = compute_embedding_features_derived(combined_df, embedding_features[:15])
        for feat_name, values in embedding_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)
    
    # 4. 统计特征衍生
    print("计算统计特征衍生...")
    statistical_features = [col for col in combined_df.columns 
                          if any(stat in col.lower() for stat in ['mean', 'std', 'max', 'min', 'entropy', 'gini'])]
    
    if statistical_features:
        statistical_derived = compute_statistical_features_derived(combined_df, statistical_features[:10])
        for feat_name, values in statistical_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)
    
    # 5. 交互特征的高阶衍生
    print("计算高阶交互特征...")
    interaction_features = [col for col in combined_df.columns 
                           if 'interaction' in col.lower() or 'product' in col.lower()]
    
    if interaction_features:
        interaction_derived = compute_interaction_features_derived(combined_df, interaction_features[:8])
        for feat_name, values in interaction_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)
    
    # 6. 特征分箱衍生
    print("计算特征分箱衍生...")
    numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) > 10:
        binning_derived = compute_feature_binning_derived(combined_df, numeric_features[:10])
        for feat_name, values in binning_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)
    
    # 7. 时间序列风格衍生（伪时间序列）
    print("计算伪时间序列特征...")
    if len(numeric_features) > 5:
        timeseries_derived = compute_timeseries_style_derived(combined_df, numeric_features[:5])
        for feat_name, values in timeseries_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)
    
    # 8. 特征重要性模拟衍生
    print("计算特征重要性衍生...")
    importance_derived = compute_feature_importance_derived(combined_df)
    for feat_name, values in importance_derived.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    print(f"衍生特征维度: {features_df.shape}")
    return features_df

def compute_important_features_derived(df: pd.DataFrame, 
                                      important_features: List[str]) -> Dict[str, np.ndarray]:
    """基于重要特征计算衍生特征"""
    n_samples = len(df)
    features = {}
    
    # 只使用实际存在的特征
    existing_features = [f for f in important_features if f in df.columns]
    
    if len(existing_features) < 2:
        return features
    
    print(f"  基于 {len(existing_features)} 个重要特征衍生...")
    
    # 1. 特征两两乘积（二阶交互）
    for i in range(len(existing_features)):
        for j in range(i+1, len(existing_features)):
            feat1 = existing_features[i]
            feat2 = existing_features[j]
            
            # 乘积
            product = df[feat1] * df[feat2]
            features[f'important_product_{feat1}_{feat2}'] = product.values
            
            # 比率
            ratio = df[feat1] / (df[feat2] + 1e-8)
            features[f'important_ratio_{feat1}_over_{feat2}'] = ratio.values
            
            # 差异
            diff = df[feat1] - df[feat2]
            features[f'important_diff_{feat1}_minus_{feat2}'] = diff.values
            
            # 绝对差异
            abs_diff = np.abs(diff)
            features[f'important_abs_diff_{feat1}_{feat2}'] = abs_diff.values
    
    # 2. 特征与自身的变换
    for feat in existing_features[:5]:  # 只处理前5个重要特征
        if feat in df.columns:
            values = df[feat].values
            
            # 平方和立方
            features[f'{feat}_squared'] = values ** 2
            features[f'{feat}_cubed'] = values ** 3
            
            # 指数变换
            features[f'{feat}_exp'] = np.exp(values / (np.std(values) + 1e-8))
            
            # 分箱特征（十分位数）
            if len(values) > 10:
                quantiles = np.percentile(values, np.linspace(0, 100, 11))
                bin_indices = np.digitize(values, quantiles[1:-1])
                features[f'{feat}_decile'] = bin_indices.astype(np.float32)
            
            # Z-score异常得分
            zscore = np.abs((values - np.mean(values)) / (np.std(values) + 1e-8))
            features[f'{feat}_zscore_abs'] = zscore.astype(np.float32)
    
    # 3. 重要特征的聚合统计
    important_matrix = df[existing_features].values
    
    # 行统计
    features['important_features_mean'] = np.mean(important_matrix, axis=1)
    features['important_features_std'] = np.std(important_matrix, axis=1)
    features['important_features_max'] = np.max(important_matrix, axis=1)
    features['important_features_min'] = np.min(important_matrix, axis=1)
    features['important_features_range'] = features['important_features_max'] - features['important_features_min']
    
    # 变异系数
    features['important_features_cv'] = np.divide(
        features['important_features_std'], 
        np.abs(features['important_features_mean']) + 1e-8
    )
    
    # 偏度和峰度
    if important_matrix.shape[1] > 3:
        features['important_features_skew'] = stats.skew(important_matrix, axis=1)
        features['important_features_kurt'] = stats.kurtosis(important_matrix, axis=1)
    
    return features

def compute_anomaly_features_derived(df: pd.DataFrame, 
                                   anomaly_features: List[str]) -> Dict[str, np.ndarray]:
    """基于异常检测特征计算衍生特征"""
    features = {}
    
    if len(anomaly_features) < 2:
        return features
    
    print(f"  基于 {len(anomaly_features)} 个异常特征衍生...")
    
    # 创建异常特征矩阵
    anomaly_matrix = df[anomaly_features].values
    
    # 1. 异常得分统计
    features['anomaly_score_mean'] = np.mean(anomaly_matrix, axis=1)
    features['anomaly_score_std'] = np.std(anomaly_matrix, axis=1)
    features['anomaly_score_max'] = np.max(anomaly_matrix, axis=1)
    features['anomaly_score_min'] = np.min(anomaly_matrix, axis=1)
    
    # 2. 异常一致性特征
    # 异常投票（超过阈值的数量）
    threshold = np.percentile(anomaly_matrix.flatten(), 75)  # 上四分位数作为阈值
    anomaly_votes = (anomaly_matrix > threshold).sum(axis=1)
    features['anomaly_vote_count'] = anomaly_votes.astype(np.float32)
    features['anomaly_vote_ratio'] = anomaly_votes / len(anomaly_features)
    
    # 3. 异常排名特征
    # 每个样本在异常特征上的排名
    anomaly_ranks = np.argsort(np.argsort(anomaly_matrix, axis=0), axis=0)
    features['anomaly_avg_rank'] = np.mean(anomaly_ranks, axis=1) / len(anomaly_features)
    
    # 4. 异常特征相关性特征
    if len(anomaly_features) > 2:
        # 计算异常特征间的皮尔逊相关性（行方向）
        anomaly_correlations = np.zeros(len(df), dtype=np.float32)
        
        for i in range(len(df)):
            row_corrs = []
            for j in range(len(anomaly_features)):
                for k in range(j+1, len(anomaly_features)):
                    # 计算特征j和k在这个样本上的"相关性"
                    val_j = anomaly_matrix[i, j]
                    val_k = anomaly_matrix[i, k]
                    row_corrs.append(val_j * val_k)
            
            if row_corrs:
                anomaly_correlations[i] = np.mean(row_corrs)
        
        features['anomaly_feature_correlation'] = anomaly_correlations
    
    # 5. 异常特征熵（不确定性）
    # 将异常得分转换为概率分布
    anomaly_probs = anomaly_matrix / (np.sum(anomaly_matrix, axis=1, keepdims=True) + 1e-8)
    anomaly_entropy = -np.sum(anomaly_probs * np.log(anomaly_probs + 1e-8), axis=1)
    features['anomaly_entropy'] = anomaly_entropy.astype(np.float32)
    
    # 6. 异常特征组合
    # 创建新的异常特征组合
    for i in range(min(3, len(anomaly_features))):
        for j in range(i+1, min(4, len(anomaly_features))):
            feat1 = anomaly_features[i]
            feat2 = anomaly_features[j]
            
            # 乘积组合
            product = df[feat1] * df[feat2]
            features[f'anomaly_product_{feat1}_{feat2}'] = product.values
            
            # 差异组合
            diff = np.abs(df[feat1] - df[feat2])
            features[f'anomaly_diff_{feat1}_{feat2}'] = diff.values
            
            # 调和平均
            harmonic_mean = 2 * df[feat1] * df[feat2] / (df[feat1] + df[feat2] + 1e-8)
            features[f'anomaly_harmonic_{feat1}_{feat2}'] = harmonic_mean.values
    
    return features

def compute_embedding_features_derived(df: pd.DataFrame, 
                                     embedding_features: List[str]) -> Dict[str, np.ndarray]:
    """基于图嵌入特征计算衍生特征"""
    features = {}
    
    if len(embedding_features) < 3:
        return features
    
    print(f"  基于 {len(embedding_features)} 个嵌入特征衍生...")
    
    # 创建嵌入特征矩阵
    embedding_matrix = df[embedding_features].values
    
    # 1. 嵌入空间的几何特征
    # L2范数（嵌入向量长度）
    features['embedding_norm'] = np.linalg.norm(embedding_matrix, axis=1)
    
    # L1范数
    features['embedding_l1_norm'] = np.sum(np.abs(embedding_matrix), axis=1)
    
    # 余弦相似度特征（与平均向量的相似度）
    mean_embedding = np.mean(embedding_matrix, axis=0)
    mean_embedding_norm = np.linalg.norm(mean_embedding)
    
    cosine_similarities = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        vec_norm = np.linalg.norm(embedding_matrix[i])
        if vec_norm > 0 and mean_embedding_norm > 0:
            cosine_similarities[i] = embedding_matrix[i] @ mean_embedding / (vec_norm * mean_embedding_norm)
    
    features['embedding_cosine_to_mean'] = cosine_similarities
    
    # 2. 嵌入空间的主成分特征
    if embedding_matrix.shape[1] > 2 and len(df) > 100:
        try:
            from sklearn.decomposition import PCA
            
            # 使用PCA提取主要方向
            pca = PCA(n_components=min(3, embedding_matrix.shape[1]))
            pca_features = pca.fit_transform(embedding_matrix)
            
            for i in range(pca_features.shape[1]):
                features[f'embedding_pca_{i}'] = pca_features[:, i]
            
            # 解释方差特征
            features['embedding_pca_explained_ratio'] = pca.explained_variance_ratio_[0] * np.ones(len(df))
        
        except Exception as e:
            print(f"    PCA计算失败: {e}")
    
    # 3. 嵌入维度统计
    features['embedding_dim_mean'] = np.mean(embedding_matrix, axis=1)
    features['embedding_dim_std'] = np.std(embedding_matrix, axis=1)
    features['embedding_dim_max'] = np.max(embedding_matrix, axis=1)
    features['embedding_dim_min'] = np.min(embedding_matrix, axis=1)
    
    # 4. 嵌入维度相关性特征
    if embedding_matrix.shape[1] > 2:
        # 计算嵌入维度的"内相关性"
        embedding_corr_matrix = np.corrcoef(embedding_matrix.T)
        np.fill_diagonal(embedding_corr_matrix, 0)
        
        # 平均绝对相关性
        mean_abs_corr = np.mean(np.abs(embedding_corr_matrix))
        features['embedding_dim_correlation'] = mean_abs_corr * np.ones(len(df))
    
    # 5. 嵌入聚类特征
    if len(df) > 100 and embedding_matrix.shape[1] > 2:
        try:
            from sklearn.cluster import KMeans
            
            # 使用K-means进行简单聚类
            n_clusters = min(5, len(df) // 20)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                cluster_labels = kmeans.fit_predict(embedding_matrix)
                
                features['embedding_cluster'] = cluster_labels.astype(np.float32)
                
                # 到聚类中心的距离
                distances = kmeans.transform(embedding_matrix)
                features['embedding_cluster_min_distance'] = np.min(distances, axis=1)
                
                # 聚类大小特征
                cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
                features['embedding_cluster_size'] = cluster_sizes[cluster_labels].astype(np.float32) / len(df)
        
        except Exception as e:
            print(f"    嵌入聚类失败: {e}")
    
    # 6. 嵌入特征组合
    # 选择前几个嵌入特征进行组合
    for i in range(min(3, len(embedding_features))):
        for j in range(i+1, min(4, len(embedding_features))):
            feat1 = embedding_features[i]
            feat2 = embedding_features[j]
            
            # 点积特征
            dot_product = df[feat1] * df[feat2]
            features[f'embedding_dot_{feat1}_{feat2}'] = dot_product.values
            
            # 欧氏距离特征
            euclidean_dist = np.sqrt((df[feat1] - df[feat2]) ** 2)
            features[f'embedding_dist_{feat1}_{feat2}'] = euclidean_dist.values
    
    return features

def compute_statistical_features_derived(df: pd.DataFrame, 
                                       statistical_features: List[str]) -> Dict[str, np.ndarray]:
    """基于统计特征计算衍生特征"""
    features = {}
    
    if len(statistical_features) < 2:
        return features
    
    print(f"  基于 {len(statistical_features)} 个统计特征衍生...")
    
    # 创建统计特征矩阵
    stats_matrix = df[statistical_features].values
    
    # 1. 统计特征的统计量
    features['stats_of_stats_mean'] = np.mean(stats_matrix, axis=1)
    features['stats_of_stats_std'] = np.std(stats_matrix, axis=1)
    features['stats_of_stats_cv'] = np.divide(
        features['stats_of_stats_std'], 
        np.abs(features['stats_of_stats_mean']) + 1e-8
    )
    
    # 2. 统计特征的一致性度量
    # 计算统计特征间的相关性（行方向）
    stats_correlations = np.zeros(len(df), dtype=np.float32)
    
    for i in range(len(df)):
        row_values = stats_matrix[i]
        if len(row_values) > 1 and np.std(row_values) > 0:
            # 使用变异系数作为一致性度量
            stats_correlations[i] = 1.0 / (np.std(row_values) / np.mean(np.abs(row_values)) + 1e-8)
    
    features['stats_consistency'] = stats_correlations
    
    # 3. 统计特征归一化组合
    # 将统计特征归一化后组合
    stats_normalized = stats_matrix / (np.std(stats_matrix, axis=0, keepdims=True) + 1e-8)
    
    # 加权和（权重为特征重要性近似）
    weights = 1.0 / (1.0 + np.var(stats_normalized, axis=0))
    features['stats_weighted_sum'] = stats_normalized @ weights
    
    # 4. 统计特征分箱衍生
    for i in range(min(3, len(statistical_features))):
        feat = statistical_features[i]
        values = df[feat].values
        
        # 分位数分箱
        if len(values) > 10:
            # 五分位数
            quintiles = np.percentile(values, [20, 40, 60, 80])
            quintile_bins = np.digitize(values, quintiles)
            features[f'{feat}_quintile'] = quintile_bins.astype(np.float32)
            
            # 十分位数
            deciles = np.percentile(values, np.linspace(10, 90, 9))
            decile_bins = np.digitize(values, deciles)
            features[f'{feat}_decile'] = decile_bins.astype(np.float32)
        
        # 基于均值和标准差的分箱
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            # Z-score分箱
            zscore_bins = np.zeros_like(values, dtype=np.float32)
            zscore_bins[values < mean_val - std_val] = 0  # 低
            zscore_bins[(values >= mean_val - std_val) & (values <= mean_val + std_val)] = 1  # 中
            zscore_bins[values > mean_val + std_val] = 2  # 高
            
            features[f'{feat}_zscore_bin'] = zscore_bins
    
    # 5. 统计特征交互
    for i in range(min(3, len(statistical_features))):
        for j in range(i+1, min(4, len(statistical_features))):
            feat1 = statistical_features[i]
            feat2 = statistical_features[j]
            
            # 乘积交互
            product = df[feat1] * df[feat2]
            features[f'stats_product_{feat1}_{feat2}'] = product.values
            
            # 比率交互
            ratio = df[feat1] / (df[feat2] + 1e-8)
            features[f'stats_ratio_{feat1}_over_{feat2}'] = ratio.values
    
    return features

def compute_interaction_features_derived(df: pd.DataFrame, 
                                       interaction_features: List[str]) -> Dict[str, np.ndarray]:
    """基于交互特征计算高阶衍生特征"""
    features = {}
    
    if len(interaction_features) < 2:
        return features
    
    print(f"  基于 {len(interaction_features)} 个交互特征衍生...")
    
    # 1. 交互特征的三阶组合
    # 选择前几个交互特征进行三阶组合
    if len(interaction_features) >= 3:
        for i in range(min(3, len(interaction_features))):
            for j in range(i+1, min(4, len(interaction_features))):
                for k in range(j+1, min(5, len(interaction_features))):
                    feat1 = interaction_features[i]
                    feat2 = interaction_features[j]
                    feat3 = interaction_features[k]
                    
                    # 三阶乘积
                    product = df[feat1] * df[feat2] * df[feat3]
                    features[f'interaction_triple_{feat1}_{feat2}_{feat3}'] = product.values
                    
                    # 三阶调和平均
                    harmonic_triple = 3 / (1/(df[feat1]+1e-8) + 1/(df[feat2]+1e-8) + 1/(df[feat3]+1e-8))
                    features[f'interaction_harmonic_triple_{feat1}_{feat2}_{feat3}'] = harmonic_triple.values
    
    # 2. 交互特征的统计衍生
    interaction_matrix = df[interaction_features].values
    
    features['interaction_mean'] = np.mean(interaction_matrix, axis=1)
    features['interaction_std'] = np.std(interaction_matrix, axis=1)
    features['interaction_max'] = np.max(interaction_matrix, axis=1)
    features['interaction_min'] = np.min(interaction_matrix, axis=1)
    
    # 3. 交互特征与原始特征的再交互
    # 寻找可能的原始特征
    original_features = []
    for feat in interaction_features:
        # 尝试解析交互特征中的原始特征名
        if 'interaction_' in feat:
            parts = feat.replace('interaction_', '').split('_')
            if len(parts) >= 2:
                original_features.extend(parts[:2])
    
    original_features = list(set(original_features))
    
    # 交互特征与原始特征的再交互
    for inter_feat in interaction_features[:3]:
        for orig_feat in original_features[:3]:
            if orig_feat in df.columns:
                # 交互特征与原始特征的乘积
                product = df[inter_feat] * df[orig_feat]
                features[f'interaction_with_original_{inter_feat}_{orig_feat}'] = product.values
    
    # 4. 交互特征的分层衍生
    for feat in interaction_features[:5]:
        values = df[feat].values
        
        # 基于分位数的分层
        if len(values) > 5:
            percentiles = np.percentile(values, [25, 50, 75])
            
            # 创建分层特征
            level_feature = np.zeros_like(values, dtype=np.float32)
            level_feature[values <= percentiles[0]] = 0  # 低
            level_feature[(values > percentiles[0]) & (values <= percentiles[2])] = 1  # 中
            level_feature[values > percentiles[2]] = 2  # 高
            
            features[f'{feat}_level'] = level_feature
        
        # 交互特征的变换
        features[f'{feat}_log1p'] = np.log1p(np.abs(values))
        features[f'{feat}_sqrt'] = np.sqrt(np.abs(values))
    
    return features

def compute_feature_binning_derived(df: pd.DataFrame, 
                                  numeric_features: List[str]) -> Dict[str, np.ndarray]:
    """基于特征分箱计算衍生特征"""
    features = {}
    
    if len(numeric_features) == 0:
        return features
    
    print(f"  对 {len(numeric_features)} 个数值特征进行分箱衍生...")
    
    for feat in numeric_features[:10]:  # 只处理前10个特征
        values = df[feat].values
        
        if len(np.unique(values)) < 5:
            continue  # 类别太少，跳过
        
        # 1. 等宽分箱
        n_bins = min(10, len(np.unique(values)) // 2)
        if n_bins >= 2:
            # 等宽分箱
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                width = (max_val - min_val) / n_bins
                bins = [min_val + i * width for i in range(1, n_bins)]
                equal_width_bins = np.digitize(values, bins)
                features[f'{feat}_equal_width_bin'] = equal_width_bins.astype(np.float32)
        
        # 2. 等频分箱（分位数分箱）
        if len(values) > n_bins:
            percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            quantiles = np.percentile(values, percentiles)
            equal_freq_bins = np.digitize(values, quantiles)
            features[f'{feat}_equal_freq_bin'] = equal_freq_bins.astype(np.float32)
        
        # 3. 基于聚类的分箱
        if len(values) > 100:
            try:
                from sklearn.cluster import KMeans
                
                # 使用K-means进行聚类分箱
                n_clusters = min(5, len(np.unique(values)) // 3)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                    values_2d = values.reshape(-1, 1)
                    cluster_labels = kmeans.fit_predict(values_2d)
                    
                    # 按聚类中心排序标签
                    sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())
                    cluster_mapping = {old: new for new, old in enumerate(sorted_indices)}
                    ordered_labels = np.vectorize(cluster_mapping.get)(cluster_labels)
                    
                    features[f'{feat}_cluster_bin'] = ordered_labels.astype(np.float32)
            
            except Exception as e:
                print(f"    聚类分箱失败 {feat}: {e}")
        
        # 4. 分箱统计特征
        if f'{feat}_equal_freq_bin' in features:
            bin_labels = features[f'{feat}_equal_freq_bin']
            n_bins_actual = len(np.unique(bin_labels))
            
            # 计算每个分箱的统计量
            for bin_id in range(n_bins_actual):
                mask = (bin_labels == bin_id)
                if np.sum(mask) > 0:
                    # 分箱均值
                    bin_mean = np.mean(values[mask])
                    features[f'{feat}_bin{bin_id}_mean_ref'] = np.full_like(values, bin_mean, dtype=np.float32)
    
    return features

def compute_timeseries_style_derived(df: pd.DataFrame, 
                                   numeric_features: List[str]) -> Dict[str, np.ndarray]:
    """计算伪时间序列风格特征"""
    features = {}
    
    if len(numeric_features) < 2:
        return features
    
    print(f"  计算 {len(numeric_features)} 个特征的伪时间序列衍生...")
    
    n_samples = len(df)
    
    # 对每个特征创建伪时间序列
    for feat in numeric_features[:5]:
        values = df[feat].values
        
        # 1. 滑动窗口统计（伪时间序列）
        window_sizes = [5, 10, 20]
        
        for window_size in window_sizes:
            if window_size >= n_samples:
                continue
            
            # 初始化滑动窗口数组
            rolling_mean = np.zeros_like(values, dtype=np.float32)
            rolling_std = np.zeros_like(values, dtype=np.float32)
            rolling_max = np.zeros_like(values, dtype=np.float32)
            rolling_min = np.zeros_like(values, dtype=np.float32)
            
            for i in range(n_samples):
                start_idx = max(0, i - window_size + 1)
                window_data = values[start_idx:i+1]
                
                rolling_mean[i] = np.mean(window_data)
                if len(window_data) > 1:
                    rolling_std[i] = np.std(window_data)
                rolling_max[i] = np.max(window_data)
                rolling_min[i] = np.min(window_data)
            
            features[f'{feat}_rolling_mean_ws{window_size}'] = rolling_mean
            features[f'{feat}_rolling_std_ws{window_size}'] = rolling_std
            features[f'{feat}_rolling_max_ws{window_size}'] = rolling_max
            features[f'{feat}_rolling_min_ws{window_size}'] = rolling_min
            
            # 滚动变化特征
            if window_size > 1:
                rolling_change = values - rolling_mean
                features[f'{feat}_rolling_change_ws{window_size}'] = rolling_change
        
        # 2. 滞后特征（伪时间序列）
        for lag in [1, 2, 3]:
            if lag < n_samples:
                lagged_values = np.zeros_like(values, dtype=np.float32)
                lagged_values[lag:] = values[:-lag]
                lagged_values[:lag] = values[0]  # 用第一个值填充
                
                features[f'{feat}_lag{lag}'] = lagged_values
                
                # 滞后差异
                lag_diff = values - lagged_values
                features[f'{feat}_lag{lag}_diff'] = lag_diff
                
                # 滞后比率
                lag_ratio = values / (lagged_values + 1e-8)
                features[f'{feat}_lag{lag}_ratio'] = lag_ratio
        
        # 3. 时序趋势特征
        if n_samples > 10:
            # 简单线性趋势（伪时间）
            x = np.arange(n_samples).reshape(-1, 1)
            
            # 使用滚动窗口计算局部趋势
            trend_window = min(20, n_samples)
            local_trend = np.zeros_like(values, dtype=np.float32)
            
            for i in range(n_samples):
                start_idx = max(0, i - trend_window // 2)
                end_idx = min(n_samples, i + trend_window // 2)
                
                if end_idx - start_idx > 3:
                    x_window = x[start_idx:end_idx]
                    y_window = values[start_idx:end_idx]
                    
                    # 简单线性回归斜率
                    x_mean = np.mean(x_window)
                    y_mean = np.mean(y_window)
                    
                    numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
                    denominator = np.sum((x_window - x_mean) ** 2)
                    
                    if denominator > 0:
                        slope = numerator / denominator
                        local_trend[i] = slope
            
            features[f'{feat}_pseudo_trend'] = local_trend
    
    return features

def compute_feature_importance_derived(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """计算特征重要性衍生特征"""
    features = {}
    
    print("  计算特征重要性衍生...")
    
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 5:
        return features
    
    # 采样计算特征重要性（使用伪标签）
    n_samples = len(df)
    
    # 生成伪标签（使用K-means聚类）
    try:
        from sklearn.cluster import KMeans
        
        # 使用PCA降维后聚类
        from sklearn.decomposition import PCA
        
        numeric_data = df[numeric_cols[:min(50, len(numeric_cols))]].values
        
        # PCA降维
        pca = PCA(n_components=min(10, numeric_data.shape[1], n_samples-1))
        reduced_data = pca.fit_transform(numeric_data)
        
        # K-means聚类生成伪标签
        n_clusters = min(3, n_samples // 100)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            pseudo_labels = kmeans.fit_predict(reduced_data)
            
            # 使用随机森林计算特征重要性
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(numeric_data, pseudo_labels)
            importances = rf.feature_importances_
            
            # 1. 特征重要性加权和
            weighted_sum = numeric_data @ importances
            features['importance_weighted_sum'] = weighted_sum.astype(np.float32)
            
            # 2. 重要特征组合
            # 选择最重要的几个特征
            top_indices = np.argsort(importances)[-5:]  # 前5个重要特征
            
            for idx in top_indices:
                if idx < len(numeric_cols):
                    feat_name = numeric_cols[idx]
                    features[f'important_{feat_name}'] = df[feat_name].values
            
            # 3. 特征重要性排名特征
            importance_ranks = np.argsort(np.argsort(importances))
            # 创建每个样本的特征重要性排名向量
            for i, rank in enumerate(importance_ranks[-10:]):  # 只取最重要的10个
                if i < len(numeric_cols):
                    feat_name = numeric_cols[i]
                    features[f'{feat_name}_importance_rank'] = rank * np.ones(n_samples, dtype=np.float32)
            
            # 4. 特征重要性一致性
            # 计算特征重要性分布的熵
            importance_probs = importances / np.sum(importances)
            importance_entropy = -np.sum(importance_probs * np.log(importance_probs + 1e-8))
            features['feature_importance_entropy'] = importance_entropy * np.ones(n_samples, dtype=np.float32)
    
    except Exception as e:
        print(f"    特征重要性计算失败: {e}")
        # 使用简单替代
        numeric_data = df[numeric_cols[:10]].values
        features['simple_weighted_sum'] = np.mean(numeric_data, axis=1)
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_paths', type=str, nargs='+', required=True,
                       help='输入特征文件路径列表')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出特征文件路径')
    args = parser.parse_args()
    
    # 计算衍生特征
    features_df = compute_derived_features(args.feature_paths, args.output_path)
    
    print(f"衍生特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
