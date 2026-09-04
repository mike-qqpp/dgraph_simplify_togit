"""
异常检测专用特征模块：针对金融欺诈检测的特征
"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # 添加这行
def compute_anomaly_features(edge_index: np.ndarray,
                            node_features: np.ndarray,
                            contamination: float = 0.05,
                            output_path: str = None) -> pd.DataFrame:
    """
    计算异常检测专用特征：
    1. 孤立森林异常得分
    2. 局部离群因子（LOF）
    3. 马氏距离
    4. 一分类SVM得分
    5. 统计异常检测
    6. 图结构异常特征
    """
    
    print("计算异常检测特征...")
    n_nodes, n_features = node_features.shape
    
    feature_dict = {}
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(node_features)
    
    # 1. 孤立森林异常得分
    print("计算孤立森林异常得分...")
    iso_forest_scores = compute_isolation_forest_scores(
        features_scaled, contamination=contamination
    )
    feature_dict['isolation_forest_score'] = iso_forest_scores.astype(np.float32)
    
    # 2. 局部离群因子（LOF）
    print("计算局部离群因子...")
    lof_scores = compute_lof_scores(
        features_scaled, contamination=contamination
    )
    feature_dict['lof_score'] = lof_scores.astype(np.float32)
    
    # 3. 马氏距离
    print("计算马氏距离...")
    mahalanobis_distances = compute_mahalanobis_distances(features_scaled)
    feature_dict['mahalanobis_distance'] = mahalanobis_distances.astype(np.float32)
    
    # 4. 一分类SVM（采样计算）
    print("计算一分类SVM...")
    ocsvm_scores = compute_one_class_svm_scores(
        features_scaled, contamination=contamination
    )
    feature_dict['one_class_svm_score'] = ocsvm_scores.astype(np.float32)
    
    # 5. 统计异常检测
    print("计算统计异常特征...")
    statistical_features = compute_statistical_anomaly_features(node_features)
    
    for feat_name, values in statistical_features.items():
        feature_dict[f'stat_{feat_name}'] = values.astype(np.float32)
    
    # 6. 图结构异常特征
    print("计算图结构异常特征...")
    graph_anomaly_features = compute_graph_anomaly_features(
        edge_index, node_features
    )
    
    for feat_name, values in graph_anomaly_features.items():
        feature_dict[f'graph_{feat_name}'] = values.astype(np.float32)
    
    # 7. 特征组合异常得分
    print("计算特征组合异常得分...")
    combined_anomaly_features = compute_combined_anomaly_features(feature_dict)
    
    for feat_name, values in combined_anomaly_features.items():
        feature_dict[f'combined_{feat_name}'] = values.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    print(f"异常检测特征维度: {features_df.shape}")
    return features_df

def compute_isolation_forest_scores(features_scaled: np.ndarray,
                                  contamination: float = 0.05) -> np.ndarray:
    """计算孤立森林异常得分"""
    n_samples = features_scaled.shape[0]
    
    # 对大样本进行采样
    if n_samples > 10000:
        sample_size = min(10000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features_scaled[sample_indices]
    else:
        sample_features = features_scaled
    
    try:
        from sklearn.ensemble import IsolationForest
        
        # 训练孤立森林
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        if n_samples > 10000:
            # 在采样数据上训练
            iso_forest.fit(sample_features)
            # 预测所有样本
            scores = iso_forest.decision_function(features_scaled)
        else:
            # 在所有数据上训练
            scores = iso_forest.fit_predict(features_scaled)
            # 转换得分：异常为-1，正常为1
            scores = -scores  # 使得异常有更高的得分
        
        # 归一化到[0, 1]范围
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.astype(np.float32)
    
    except Exception as e:
        print(f"孤立森林计算失败: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_lof_scores(features_scaled: np.ndarray,
                      contamination: float = 0.05) -> np.ndarray:
    """计算局部离群因子"""
    n_samples = features_scaled.shape[0]
    
    # 对大样本进行采样
    if n_samples > 5000:
        sample_size = min(5000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features_scaled[sample_indices]
    else:
        sample_features = features_scaled
    
    try:
        from sklearn.neighbors import LocalOutlierFactor
        
        # 训练LOF
        lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,  # 允许预测新样本
            n_jobs=-1
        )
        
        if n_samples > 5000:
            # 在采样数据上训练
            lof.fit(sample_features)
            # 预测所有样本
            scores = lof.decision_function(features_scaled)
        else:
            # 在所有数据上训练
            lof.fit(features_scaled)
            scores = lof.negative_outlier_factor_
            scores = -scores  # 反转符号
        
        # 归一化
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.astype(np.float32)
    
    except Exception as e:
        print(f"LOF计算失败: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_mahalanobis_distances(features_scaled: np.ndarray) -> np.ndarray:
    """计算马氏距离"""
    n_samples = features_scaled.shape[0]
    
    try:
        # 计算协方差矩阵的伪逆
        cov_matrix = np.cov(features_scaled.T)
        
        # 添加小值确保可逆
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        # 计算均值
        mean_vector = np.mean(features_scaled, axis=0)
        
        # 计算马氏距离
        distances = np.zeros(n_samples, dtype=np.float32)
        
        # 分批计算
        batch_size = 10000
        for i in tqdm(range(0, n_samples, batch_size), desc="马氏距离", leave=False):
            end_idx = min(i + batch_size, n_samples)
            batch_features = features_scaled[i:end_idx]
            
            diff = batch_features - mean_vector
            batch_distances = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
            distances[i:end_idx] = batch_distances
        
        # 归一化
        if distances.max() > 0:
            distances = distances / distances.max()
        
        return distances.astype(np.float32)
    
    except Exception as e:
        print(f"马氏距离计算失败: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_one_class_svm_scores(features_scaled: np.ndarray,
                               contamination: float = 0.05) -> np.ndarray:
    """计算一分类SVM得分"""
    n_samples = features_scaled.shape[0]
    
    # 对大样本进行采样
    if n_samples > 5000:
        sample_size = min(5000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features_scaled[sample_indices]
    else:
        sample_features = features_scaled
    
    try:
        from sklearn.svm import OneClassSVM
        
        # 训练One-Class SVM
        nu_value = contamination  # OCSVM的参数nu对应异常比例
        
        ocsvm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=nu_value
        )
        
        if n_samples > 5000:
            # 在采样数据上训练
            ocsvm.fit(sample_features)
            # 预测所有样本
            scores = ocsvm.decision_function(features_scaled)
        else:
            # 在所有数据上训练
            ocsvm.fit(features_scaled)
            scores = ocsvm.decision_function(features_scaled)
        
        # 转换得分：异常有更低的得分
        scores = -scores  # 反转使得异常有更高的值
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.astype(np.float32)
    
    except Exception as e:
        print(f"一分类SVM计算失败: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_statistical_anomaly_features(node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """计算统计异常特征"""
    n_samples, n_features = node_features.shape
    
    features = {}
    
    # 1. Z-score异常
    print("    计算Z-score异常...")
    zscore_features = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for i in range(n_features):
        col = node_features[:, i]
        mean_val = np.mean(col)
        std_val = np.std(col)
        
        if std_val > 0:
            zscores = np.abs((col - mean_val) / std_val)
            zscore_features[:, i] = zscores
        else:
            zscore_features[:, i] = 0
    
    # 最大Z-score
    features['max_zscore'] = np.max(zscore_features, axis=1)
    
    # 平均Z-score
    features['mean_zscore'] = np.mean(zscore_features, axis=1)
    
    # Z-score大于3的比例
    features['zscore_gt3_ratio'] = np.mean(zscore_features > 3, axis=1)
    
    # 2. IQR异常（箱线图）
    print("    计算IQR异常...")
    iqr_features = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for i in range(min(5, n_features)):  # 只计算前5个特征
        col = node_features[:, i]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 计算异常得分
            outlier_score = np.zeros_like(col)
            outlier_score[col < lower_bound] = (lower_bound - col[col < lower_bound]) / iqr
            outlier_score[col > upper_bound] = (col[col > upper_bound] - upper_bound) / iqr
            
            iqr_features[:, i] = outlier_score
    
    features['max_iqr_outlier'] = np.max(iqr_features, axis=1)
    
    # 3. 偏度和峰度异常
    print("    计算偏度峰度异常...")
    if n_samples > 100:
        for i in range(min(3, n_features)):
            col = node_features[:, i]
            
            # 偏度
            skewness = stats.skew(col)
            features[f'feat{i}_skewness_abs'] = np.abs(skewness) * np.ones(n_samples, dtype=np.float32)
            
            # 峰度
            kurt = stats.kurtosis(col)
            features[f'feat{i}_kurtosis_abs'] = np.abs(kurt) * np.ones(n_samples, dtype=np.float32)
    
    # 4. 特征值异常组合
    print("    计算特征值异常组合...")
    # 特征乘积异常（交互异常）
    if n_features >= 2:
        for i in range(min(2, n_features)):
            for j in range(i+1, min(3, n_features)):
                product = node_features[:, i] * node_features[:, j]
                product_zscore = np.abs((product - np.mean(product)) / (np.std(product) + 1e-8))
                features[f'product_{i}_{j}_zscore'] = product_zscore.astype(np.float32)
    
    return features

def compute_graph_anomaly_features(edge_index: np.ndarray,
                                 node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """计算图结构异常特征"""
    n_nodes = node_features.shape[0]
    
    # 创建邻接矩阵
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    # 计算度
    degree = np.array(adj_matrix_sym.sum(axis=1)).flatten()
    
    features = {}
    
    # 1. 度异常特征
    print("    计算度异常特征...")
    if degree.max() > 0:
        # 度Z-score
        degree_mean = np.mean(degree)
        degree_std = np.std(degree)
        if degree_std > 0:
            features['degree_zscore'] = np.abs((degree - degree_mean) / degree_std).astype(np.float32)
        
        # 度对数异常
        degree_log = np.log1p(degree)
        degree_log_mean = np.mean(degree_log)
        degree_log_std = np.std(degree_log)
        if degree_log_std > 0:
            features['degree_log_zscore'] = np.abs((degree_log - degree_log_mean) / degree_log_std).astype(np.float32)
    
    # 2. 邻居特征异常
    print("    计算邻居特征异常...")
    adj_matrix_csc = adj_matrix_sym.tocsc()
    
    neighbor_feature_diff = np.zeros(n_nodes, dtype=np.float32)
    neighbor_feature_std = np.zeros(n_nodes, dtype=np.float32)
    
    for node in tqdm(range(min(10000, n_nodes)), desc="邻居异常", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) == 0:
            neighbor_feature_diff[node] = 0
            neighbor_feature_std[node] = 0
            continue
        
        # 计算节点特征与邻居平均特征的差异
        node_feat = node_features[node]
        neighbor_feats = node_features[neighbors]
        neighbor_mean = np.mean(neighbor_feats, axis=0)
        
        # 特征差异（欧氏距离）
        diff = np.linalg.norm(node_feat - neighbor_mean)
        neighbor_feature_diff[node] = diff
        
        # 邻居特征标准差（度量邻居同质性）
        if len(neighbors) > 1:
            neighbor_std = np.mean(np.std(neighbor_feats, axis=0))
            neighbor_feature_std[node] = neighbor_std
    
    # 归一化
    if neighbor_feature_diff.max() > 0:
        neighbor_feature_diff = neighbor_feature_diff / neighbor_feature_diff.max()
    
    if neighbor_feature_std.max() > 0:
        neighbor_feature_std = neighbor_feature_std / neighbor_feature_std.max()
    
    features['neighbor_feature_diff'] = neighbor_feature_diff
    features['neighbor_feature_std'] = neighbor_feature_std
    
    # 3. 结构洞异常
    print("    计算结构洞异常...")
    constraint_scores = compute_constraint_scores_sampled(adj_matrix_sym)
    features['constraint_score'] = constraint_scores.astype(np.float32)
    
    # 4. 桥接异常（连接不同社区的边）
    print("    计算桥接异常...")
    bridging_scores = compute_bridging_scores(adj_matrix_sym, degree)
    features['bridging_score'] = bridging_scores.astype(np.float32)
    
    return features

def compute_constraint_scores_sampled(adj_matrix: sp.csr_matrix) -> np.ndarray:
    """采样计算结构洞约束系数"""
    n_nodes = adj_matrix.shape[0]
    constraint = np.zeros(n_nodes, dtype=np.float32)
    
    # 采样计算
    sample_size = min(5000, n_nodes)
    sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    
    adj_matrix_csc = adj_matrix.tocsc()
    
    for node in tqdm(sampled_nodes, desc="结构洞", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        total_connection = len(neighbors)
        
        if total_connection == 0:
            constraint[node] = 1
            continue
        
        constraint_sum = 0
        for neighbor in neighbors:
            # 计算p_{iq}
            p_iq = 1.0 / total_connection
            
            # 计算公共邻居
            neighbor_neighbors = adj_matrix_csc[:, neighbor].indices
            common_neighbors = np.intersect1d(neighbors, neighbor_neighbors)
            
            # 计算p_{iq} * p_{qj}
            for common in common_neighbors:
                if common != node and common != neighbor:
                    p_qj = 1.0 / len(neighbor_neighbors)
                    constraint_sum += p_iq * p_qj
        
        constraint[node] = constraint_sum
    
    # 填充未采样节点
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    if len(sampled_nodes) > 0:
        mean_constraint = constraint[sampled_nodes].mean()
        constraint[unsampled] = mean_constraint
    
    return constraint

def compute_bridging_scores(adj_matrix: sp.csr_matrix, degree: np.ndarray) -> np.ndarray:
    """计算桥接得分"""
    n_nodes = adj_matrix.shape[0]
    
    # 简单桥接度量：低度节点的桥接潜力更高
    bridging = np.zeros(n_nodes, dtype=np.float32)
    
    # 归一化度
    if degree.max() > 0:
        degree_norm = degree / degree.max()
        # 低度节点有更高的桥接潜力
        bridging = 1.0 - degree_norm
    
    return bridging

def compute_combined_anomaly_features(feature_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """计算组合异常特征"""
    # 提取异常得分列
    anomaly_score_cols = [col for col in feature_dict.keys() 
                         if 'score' in col.lower() or 'distance' in col.lower() 
                         or 'zscore' in col.lower() or 'outlier' in col.lower()]
    
    if not anomaly_score_cols:
        return {}
    
    # 收集异常得分
    anomaly_scores = []
    for col in anomaly_score_cols[:10]:  # 只取前10个异常得分
        anomaly_scores.append(feature_dict[col])
    
    if not anomaly_scores:
        return {}
    
    anomaly_matrix = np.column_stack(anomaly_scores)
    
    features = {}
    
    # 1. 平均异常得分
    features['mean_anomaly_score'] = np.mean(anomaly_matrix, axis=1).astype(np.float32)
    
    # 2. 最大异常得分
    features['max_anomaly_score'] = np.max(anomaly_matrix, axis=1).astype(np.float32)
    
    # 3. 异常得分标准差（度量不一致性）
    if anomaly_matrix.shape[1] > 1:
        features['anomaly_score_std'] = np.std(anomaly_matrix, axis=1).astype(np.float32)
    
    # 4. 异常得分熵（度量不确定性）
    if anomaly_matrix.shape[1] > 1:
        # 将得分转换为概率
        score_probs = anomaly_matrix / (np.sum(anomaly_matrix, axis=1, keepdims=True) + 1e-8)
        
        # 计算熵
        entropy = -np.sum(score_probs * np.log(score_probs + 1e-8), axis=1)
        features['anomaly_score_entropy'] = entropy.astype(np.float32)
    
    # 5. 异常投票（多数投票）
    if anomaly_matrix.shape[1] > 2:
        # 阈值化
        anomaly_threshold = 0.5
        anomaly_votes = (anomaly_matrix > anomaly_threshold).astype(int)
        vote_counts = np.sum(anomaly_votes, axis=1)
        features['anomaly_vote_count'] = vote_counts.astype(np.float32)
        features['anomaly_vote_ratio'] = (vote_counts / anomaly_matrix.shape[1]).astype(np.float32)
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    parser.add_argument('--contamination', type=float, default=0.05, help='异常比例估计')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    print(f"节点数: {x.shape[0]}, 特征维度: {x.shape[1]}")
    
    # 计算异常检测特征
    features_df = compute_anomaly_features(
        edge_index, x, 
        contamination=args.contamination
    )
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"异常检测特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()