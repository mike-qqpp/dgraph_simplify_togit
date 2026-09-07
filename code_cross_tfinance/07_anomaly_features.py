"""Special feature module for unusual detection: features for financial fraud detection"""
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
from typing import Dict, List, Tuple  # Add this line
def compute_anomaly_features(edge_index: np.ndarray,
                            node_features: np.ndarray,
                            contamination: float = 0.05,
                            output_path: str = None) -> pd.DataFrame:
    """Special features for calculating abnormalities:
1. Isolated forests score abnormally
2. Local isolation factor (LOF)
3. Map distance
SVM scores for category I
5. Statistical anomaly detection
Figure structural anomalies"""

    print("Calculating abnormality detection...")
    n_nodes, n_features = node_features.shape

    feature_dict = {}

    # Standardized features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(node_features)

    # 1. Isolated forests score abnormally
    print("Calculating Isolated Forests Unusual Scores...")
    iso_forest_scores = compute_isolation_forest_scores(
        features_scaled, contamination=contamination
    )
    feature_dict['isolation_forest_score'] = iso_forest_scores.astype(np.float32)

    # 2. Local isolation factor (LOF)
    print("Calculating Local Isolation Factors...")
    lof_scores = compute_lof_scores(
        features_scaled, contamination=contamination
    )
    feature_dict['lof_score'] = lof_scores.astype(np.float32)

    # 3. Map distance
    print("Calculating the Mars distance...")
    mahalanobis_distances = compute_mahalanobis_distances(features_scaled)
    feature_dict['mahalanobis_distance'] = mahalanobis_distances.astype(np.float32)

    # 4. Classification I SVM (sampling calculations)
    print("Calculating a Class SVM...")
    ocsvm_scores = compute_one_class_svm_scores(
        features_scaled, contamination=contamination
    )
    feature_dict['one_class_svm_score'] = ocsvm_scores.astype(np.float32)

    # 5. Statistical anomaly detection
    print("Compute statistical anomalies...")
    statistical_features = compute_statistical_anomaly_features(node_features)

    for feat_name, values in statistical_features.items():
        feature_dict[f'stat_{feat_name}'] = values.astype(np.float32)

    # Figure structural anomalies
    print("Calculating Structural Anomalous Features...")
    graph_anomaly_features = compute_graph_anomaly_features(
        edge_index, node_features
    )

    for feat_name, values in graph_anomaly_features.items():
        feature_dict[f'graph_{feat_name}'] = values.astype(np.float32)

    # 7. Feature combination abnormal scores
    print("Calculates an abnormal score for feature combinations...")
    combined_anomaly_features = compute_combined_anomaly_features(feature_dict)

    for feat_name, values in combined_anomaly_features.items():
        feature_dict[f'combined_{feat_name}'] = values.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

    print(f"Feature dimension of abnormality detection: {features_df.shape}")
    return features_df

def compute_isolation_forest_scores(features_scaled: np.ndarray,
                                  contamination: float = 0.05) -> np.ndarray:
    """Calculating Isolated Forests Unusual Scores"""
    n_samples = features_scaled.shape[0]

    # Sample large samples
    if n_samples > 10000:
        sample_size = min(10000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features_scaled[sample_indices]
    else:
        sample_features = features_scaled

    try:
        from sklearn.ensemble import IsolationForest

        # Training for isolated forests
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )

        if n_samples > 10000:
            # Training in sampling data
            iso_forest.fit(sample_features)
            # Forecast all samples
            scores = iso_forest.decision_function(features_scaled)
        else:
            # Trained on all data
            scores = iso_forest.fit_predict(features_scaled)
            # Conversion score: abnormal 1 and normal 1
            scores = -scores  # It's got a higher score.

        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.astype(np.float32)

    except Exception as e:
        print(f"Isolation of forest calculations failed: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_lof_scores(features_scaled: np.ndarray,
                      contamination: float = 0.05) -> np.ndarray:
    """Calculate local ioning factors"""
    n_samples = features_scaled.shape[0]

    # Sample large samples
    if n_samples > 5000:
        sample_size = min(5000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features_scaled[sample_indices]
    else:
        sample_features = features_scaled

    try:
        from sklearn.neighbors import LocalOutlierFactor

        # Training LOF
        lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,  # Allow forecasting of new samples
            n_jobs=-1
        )

        if n_samples > 5000:
            # Training in sampling data
            lof.fit(sample_features)
            # Forecast all samples
            scores = lof.decision_function(features_scaled)
        else:
            # Trained on all data
            lof.fit(features_scaled)
            scores = lof.negative_outlier_factor_
            scores = -scores  # Invert Symbols

        # Normalization
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.astype(np.float32)

    except Exception as e:
        print(f"LOFCalculator Failed: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_mahalanobis_distances(features_scaled: np.ndarray) -> np.ndarray:
    """Calculating Mars Distance"""
    n_samples = features_scaled.shape[0]

    try:
        # Calculating the Hypothetical Matrix
        cov_matrix = np.cov(features_scaled.T)

        # Add small value to ensure reversibility
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        # Calculate Mean
        mean_vector = np.mean(features_scaled, axis=0)

        # Calculating Mars Distance
        distances = np.zeros(n_samples, dtype=np.float32)

        # Batch Count
        batch_size = 10000
        for i in tqdm(range(0, n_samples, batch_size), desc="Ma's distance.", leave=False):
            end_idx = min(i + batch_size, n_samples)
            batch_features = features_scaled[i:end_idx]

            diff = batch_features - mean_vector
            batch_distances = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
            distances[i:end_idx] = batch_distances

        # Normalization
        if distances.max() > 0:
            distances = distances / distances.max()

        return distances.astype(np.float32)

    except Exception as e:
        print(f"Ma's distance calculation failed.: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_one_class_svm_scores(features_scaled: np.ndarray,
                               contamination: float = 0.05) -> np.ndarray:
    """Calculate a class SVM score"""
    n_samples = features_scaled.shape[0]

    # Sample large samples
    if n_samples > 5000:
        sample_size = min(5000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features_scaled[sample_indices]
    else:
        sample_features = features_scaled

    try:
        from sklearn.svm import OneClassSVM

        # Training One-Class SVM
        nu_value = contamination  # OCSVM parameter Nu anomalous ratio

        ocsvm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=nu_value
        )

        if n_samples > 5000:
            # Training in sampling data
            ocsvm.fit(sample_features)
            # Forecast all samples
            scores = ocsvm.decision_function(features_scaled)
        else:
            # Trained on all data
            ocsvm.fit(features_scaled)
            scores = ocsvm.decision_function(features_scaled)

        # Convert score: abnormally low score
        scores = -scores  # Inverted makes abnormally high values.
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.astype(np.float32)

    except Exception as e:
        print(f"I ClassificationSVMCalculator Failed: {e}")
        return np.zeros(n_samples, dtype=np.float32)

def compute_statistical_anomaly_features(node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute statistical anomalies"""
    n_samples, n_features = node_features.shape

    features = {}

    # 1. Z-score anomaly
    print("Compute Z-score anomaly...")
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

    # Max. Z-score
    features['max_zscore'] = np.max(zscore_features, axis=1)

    # Average Z-score
    features['mean_zscore'] = np.mean(zscore_features, axis=1)

    # Z-score > 3
    features['zscore_gt3_ratio'] = np.mean(zscore_features > 3, axis=1)

    # 2. IQR anomalies (box charts)
    print("Compute IQR anomalies...")
    iqr_features = np.zeros((n_samples, n_features), dtype=np.float32)

    for i in range(min(5, n_features)):  # Only the first five features are calculated
        col = node_features[:, i]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1

        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Calculating Abnormal Scores
            outlier_score = np.zeros_like(col)
            outlier_score[col < lower_bound] = (lower_bound - col[col < lower_bound]) / iqr
            outlier_score[col > upper_bound] = (col[col > upper_bound] - upper_bound) / iqr

            iqr_features[:, i] = outlier_score

    features['max_iqr_outlier'] = np.max(iqr_features, axis=1)

    # 3. Distortion and peak anomalies
    print("Calculating the peak of the margin...")
    if n_samples > 100:
        for i in range(min(3, n_features)):
            col = node_features[:, i]

            # Argument
            skewness = stats.skew(col)
            features[f'feat{i}_skewness_abs'] = np.abs(skewness) * np.ones(n_samples, dtype=np.float32)

            # Peak
            kurt = stats.kurtosis(col)
            features[f'feat{i}_kurtosis_abs'] = np.abs(kurt) * np.ones(n_samples, dtype=np.float32)

    # 4. Unusual combination of characteristic values
    print("Calculate an abnormal combination of feature values...")
    # Feature product anomaly (interactive anomaly)
    if n_features >= 2:
        for i in range(min(2, n_features)):
            for j in range(i+1, min(3, n_features)):
                product = node_features[:, i] * node_features[:, j]
                product_zscore = np.abs((product - np.mean(product)) / (np.std(product) + 1e-8))
                features[f'product_{i}_{j}_zscore'] = product_zscore.astype(np.float32)

    return features

def compute_graph_anomaly_features(edge_index: np.ndarray,
                                 node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute structural anomalies"""
    n_nodes = node_features.shape[0]

    # Create an adjacent matrix
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)),
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)

    # Calculator
    degree = np.array(adj_matrix_sym.sum(axis=1)).flatten()

    features = {}

    # 1. Paranormal features
    print("Calculator anomaly...")
    if degree.max() > 0:
        # degreesZ-score
        degree_mean = np.mean(degree)
        degree_std = np.std(degree)
        if degree_std > 0:
            features['degree_zscore'] = np.abs((degree - degree_mean) / degree_std).astype(np.float32)

        # Logarithmic
        degree_log = np.log1p(degree)
        degree_log_mean = np.mean(degree_log)
        degree_log_std = np.std(degree_log)
        if degree_log_std > 0:
            features['degree_log_zscore'] = np.abs((degree_log - degree_log_mean) / degree_log_std).astype(np.float32)

    # 2. Anomalous neighbourhood features
    print("Calculating Neighbour Features...")
    adj_matrix_csc = adj_matrix_sym.tocsc()

    neighbor_feature_diff = np.zeros(n_nodes, dtype=np.float32)
    neighbor_feature_std = np.zeros(n_nodes, dtype=np.float32)

    for node in tqdm(range(min(10000, n_nodes)), desc="Neighbors are abnormal.", leave=False):
        neighbors = adj_matrix_csc[:, node].indices

        if len(neighbors) == 0:
            neighbor_feature_diff[node] = 0
            neighbor_feature_std[node] = 0
            continue

        # Calculation of differences between node features and average neighbour features
        node_feat = node_features[node]
        neighbor_feats = node_features[neighbors]
        neighbor_mean = np.mean(neighbor_feats, axis=0)

        # Featureal differences (Occidental distance)
        diff = np.linalg.norm(node_feat - neighbor_mean)
        neighbor_feature_diff[node] = diff

        # Neighbors ' features are poor (measure neighbour homogeneity)
        if len(neighbors) > 1:
            neighbor_std = np.mean(np.std(neighbor_feats, axis=0))
            neighbor_feature_std[node] = neighbor_std

    # Normalization
    if neighbor_feature_diff.max() > 0:
        neighbor_feature_diff = neighbor_feature_diff / neighbor_feature_diff.max()

    if neighbor_feature_std.max() > 0:
        neighbor_feature_std = neighbor_feature_std / neighbor_feature_std.max()

    features['neighbor_feature_diff'] = neighbor_feature_diff
    features['neighbor_feature_std'] = neighbor_feature_std

    # 3. Structural hole anomalies
    print("Calculating Structural Hole Anomalous...")
    constraint_scores = compute_constraint_scores_sampled(adj_matrix_sym)
    features['constraint_score'] = constraint_scores.astype(np.float32)

    # 4. Bridges are abnormal (sides connecting different communities)
    print("Calculating an abnormal bridge...")
    bridging_scores = compute_bridging_scores(adj_matrix_sym, degree)
    features['bridging_score'] = bridging_scores.astype(np.float32)

    return features

def compute_constraint_scores_sampled(adj_matrix: sp.csr_matrix) -> np.ndarray:
    """Sample calculation of structural hole binding factor"""
    n_nodes = adj_matrix.shape[0]
    constraint = np.zeros(n_nodes, dtype=np.float32)

    # Sample calculations
    sample_size = min(5000, n_nodes)
    sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)

    adj_matrix_csc = adj_matrix.tocsc()

    for node in tqdm(sampled_nodes, desc="Structure Hole", leave=False):
        neighbors = adj_matrix_csc[:, node].indices
        total_connection = len(neighbors)

        if total_connection == 0:
            constraint[node] = 1
            continue

        constraint_sum = 0
        for neighbor in neighbors:
            # Calculatep {iq}
            p_iq = 1.0 / total_connection

            # Counting Public Neighbors
            neighbor_neighbors = adj_matrix_csc[:, neighbor].indices
            common_neighbors = np.intersect1d(neighbors, neighbor_neighbors)

            # Calculate p {iq} *p {qj}
            for common in common_neighbors:
                if common != node and common != neighbor:
                    p_qj = 1.0 / len(neighbor_neighbors)
                    constraint_sum += p_iq * p_qj

        constraint[node] = constraint_sum

    # Fill unsampled nodes
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)

    if len(sampled_nodes) > 0:
        mean_constraint = constraint[sampled_nodes].mean()
        constraint[unsampled] = mean_constraint

    return constraint

def compute_bridging_scores(adj_matrix: sp.csr_matrix, degree: np.ndarray) -> np.ndarray:
    """Calculating Bridge Scores"""
    n_nodes = adj_matrix.shape[0]

    # Simple bridge scale: Low node with higher potential for bridge
    bridging = np.zeros(n_nodes, dtype=np.float32)

    # Normalization
    if degree.max() > 0:
        degree_norm = degree / degree.max()
        # The low node has a higher potential for bridge.
        bridging = 1.0 - degree_norm

    return bridging

def compute_combined_anomaly_features(feature_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute group abnormalities"""
    # The extraction must be disaggregated.
    anomaly_score_cols = [col for col in feature_dict.keys()
                         if 'score' in col.lower() or 'distance' in col.lower()
                         or 'zscore' in col.lower() or 'outlier' in col.lower()]

    if not anomaly_score_cols:
        return {}

    # Collect unusual scores
    anomaly_scores = []
    for col in anomaly_score_cols[:10]:  # Only the top 10 abnormal scores.
        anomaly_scores.append(feature_dict[col])

    if not anomaly_scores:
        return {}

    anomaly_matrix = np.column_stack(anomaly_scores)

    features = {}

    # 1. Average abnormal scores
    features['mean_anomaly_score'] = np.mean(anomaly_matrix, axis=1).astype(np.float32)

    # 2. Maximum abnormal score
    features['max_anomaly_score'] = np.max(anomaly_matrix, axis=1).astype(np.float32)

    # 3. Anomalous score standard differential (inconsistent measures)
    if anomaly_matrix.shape[1] > 1:
        features['anomaly_score_std'] = np.std(anomaly_matrix, axis=1).astype(np.float32)

    # Unusual scorers (measuring uncertainty)
    if anomaly_matrix.shape[1] > 1:
        # Convert score to probability
        score_probs = anomaly_matrix / (np.sum(anomaly_matrix, axis=1, keepdims=True) + 1e-8)

        # Calculating entropy
        entropy = -np.sum(score_probs * np.log(score_probs + 1e-8), axis=1)
        features['anomaly_score_entropy'] = entropy.astype(np.float32)

    # 5. Unusual voting (majority)
    if anomaly_matrix.shape[1] > 2:
        # threshold
        anomaly_threshold = 0.5
        anomaly_votes = (anomaly_matrix > anomaly_threshold).astype(int)
        vote_counts = np.sum(anomaly_votes, axis=1)
        features['anomaly_vote_count'] = vote_counts.astype(np.float32)
        features['anomaly_vote_ratio'] = (vote_counts / anomaly_matrix.shape[1]).astype(np.float32)

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--contamination', type=float, default=0.05, help='Estimates of abnormal proportions')
    args = parser.parse_args()

    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']

    print(f"Number of nodes: {x.shape[0]}, Feature shape: {x.shape[1]}")

    # Calculating abnormality detection features
    features_df = compute_anomaly_features(
        edge_index, x,
        contamination=args.contamination
    )

    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)

    print(f"Anomalous signature processing complete.，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
