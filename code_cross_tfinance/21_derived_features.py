"""Feature Derivative Module: Create high-level derivatives based on existing features"""
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
    """Calculate derivative features:
1. Feature cross-plication (high-level interaction)
2. Feature ratios and differences
3. Feature boxes and statistics
4. Feature multiplication
5. Feature cluster derivatives
6. Unusual combinations"""
    print("Loading existing features...")

    # Load all feature files
    all_features = []
    for path in feature_paths:
        try:
            with open(path, 'rb') as f:
                features_df = pd.read_pickle(f)
                all_features.append(features_df)
                print(f"Load: {path}, Shape: {features_df.shape}")
        except Exception as e:
            print(f"Loading failed {path}: {e}")

    if not all_features:
        raise ValueError("No feature file successfully loaded")

    # Merge all features
    print("Merge Features...")
    combined_df = pd.concat(all_features, axis=1)

    # Remove Repeat Column
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    print(f"Post-merger feature dimension: {combined_df.shape}")
    print(f"Examples of features: {list(combined_df.columns[:20])}")

    feature_dict = {}

    # 1. Derivation of important features (based on the head features you provide)
    print("Calculating Important Feature Derivatives...")
    important_features = [
        'feature_1', 'feature_2', 'feature_embedding_dim0',
        'laplacian_dim0', 'gnn_embedding_dim0', 'isolation_forest_score',
        'lof_score', 'one_class_svm_score', 'feature_entropy'
    ]

    important_derived = compute_important_features_derived(combined_df, important_features)
    for feat_name, values in important_derived.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 2. Abnormal detection characterization combination derivatives
    print("Compute an abnormal feature combination...")
    anomaly_features = [col for col in combined_df.columns
                       if 'score' in col.lower() or 'anomaly' in col.lower()]

    if anomaly_features:
        anomaly_derived = compute_anomaly_features_derived(combined_df, anomaly_features[:10])
        for feat_name, values in anomaly_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)

    # 3. Cross-fertilization of embedded features
    print("Calculating Embedded Feature Crossing...")
    embedding_features = [col for col in combined_df.columns
                         if 'embedding' in col.lower() or 'dim' in col.lower()]

    if embedding_features:
        embedding_derived = compute_embedding_features_derived(combined_df, embedding_features[:15])
        for feat_name, values in embedding_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)

    # 4. Statistical profiling
    print("Calculating Statistical Features Derivated...")
    statistical_features = [col for col in combined_df.columns
                          if any(stat in col.lower() for stat in ['mean', 'std', 'max', 'min', 'entropy', 'gini'])]

    if statistical_features:
        statistical_derived = compute_statistical_features_derived(combined_df, statistical_features[:10])
        for feat_name, values in statistical_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)

    # High-level derivative of interactive features
    print("Calculating high-level interactive features...")
    interaction_features = [col for col in combined_df.columns
                           if 'interaction' in col.lower() or 'product' in col.lower()]

    if interaction_features:
        interaction_derived = compute_interaction_features_derived(combined_df, interaction_features[:8])
        for feat_name, values in interaction_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)

    # 6. Feature compartment derivatives
    print("Calculating Feature Box Derivation...")
    numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) > 10:
        binning_derived = compute_feature_binning_derived(combined_df, numeric_features[:10])
        for feat_name, values in binning_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)

    # 7. Time-series style derivative (perfect time series)
    print("Calculating false time series features...")
    if len(numeric_features) > 5:
        timeseries_derived = compute_timeseries_style_derived(combined_df, numeric_features[:5])
        for feat_name, values in timeseries_derived.items():
            feature_dict[feat_name] = values.astype(np.float32)

    # 8. Feature importance simulations
    print("Calculating Feature Importance Derivated...")
    importance_derived = compute_feature_importance_derived(combined_df)
    for feat_name, values in importance_derived.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

    print(f"Derivative characterization dimension: {features_df.shape}")
    return features_df

def compute_important_features_derived(df: pd.DataFrame,
                                      important_features: List[str]) -> Dict[str, np.ndarray]:
    """Calculate derivative features based on important features"""
    n_samples = len(df)
    features = {}

    # Use only physical features
    existing_features = [f for f in important_features if f in df.columns]

    if len(existing_features) < 2:
        return features

    print(f"  Based on {len(existing_features)} A key feature derived...")

    # 1. Features two-to-two multipliers (second-stage interaction)
    for i in range(len(existing_features)):
        for j in range(i+1, len(existing_features)):
            feat1 = existing_features[i]
            feat2 = existing_features[j]

            # Product
            product = df[feat1] * df[feat2]
            features[f'important_product_{feat1}_{feat2}'] = product.values

            # Ratio
            ratio = df[feat1] / (df[feat2] + 1e-8)
            features[f'important_ratio_{feat1}_over_{feat2}'] = ratio.values

            # Variance
            diff = df[feat1] - df[feat2]
            features[f'important_diff_{feat1}_minus_{feat2}'] = diff.values

            # Absolute difference
            abs_diff = np.abs(diff)
            features[f'important_abs_diff_{feat1}_{feat2}'] = abs_diff.values

    # 2. Transformation of features and self
    for feat in existing_features[:5]:  # Only the first five important features are addressed.
        if feat in df.columns:
            values = df[feat].values

            # Square and Cubic
            features[f'{feat}_squared'] = values ** 2
            features[f'{feat}_cubed'] = values ** 3

            # Index transformation
            features[f'{feat}_exp'] = np.exp(values / (np.std(values) + 1e-8))

            # Boxing features (tenths)
            if len(values) > 10:
                quantiles = np.percentile(values, np.linspace(0, 100, 11))
                bin_indices = np.digitize(values, quantiles[1:-1])
                features[f'{feat}_decile'] = bin_indices.astype(np.float32)

            # Z-score's abnormal score
            zscore = np.abs((values - np.mean(values)) / (np.std(values) + 1e-8))
            features[f'{feat}_zscore_abs'] = zscore.astype(np.float32)

    # 3. Syndication statistics for key features
    important_matrix = df[existing_features].values

    # Line Statistics
    features['important_features_mean'] = np.mean(important_matrix, axis=1)
    features['important_features_std'] = np.std(important_matrix, axis=1)
    features['important_features_max'] = np.max(important_matrix, axis=1)
    features['important_features_min'] = np.min(important_matrix, axis=1)
    features['important_features_range'] = features['important_features_max'] - features['important_features_min']

    # Variable factor
    features['important_features_cv'] = np.divide(
        features['important_features_std'],
        np.abs(features['important_features_mean']) + 1e-8
    )

    # Distortion and Peak
    if important_matrix.shape[1] > 3:
        features['important_features_skew'] = stats.skew(important_matrix, axis=1)
        features['important_features_kurt'] = stats.kurtosis(important_matrix, axis=1)

    return features

def compute_anomaly_features_derived(df: pd.DataFrame,
                                   anomaly_features: List[str]) -> Dict[str, np.ndarray]:
    """Calculate derivatives based on abnormal detection features"""
    features = {}

    if len(anomaly_features) < 2:
        return features

    print(f"  Based on {len(anomaly_features)} An abnormal feature derived...")

    # Create an anomaly matrix
    anomaly_matrix = df[anomaly_features].values

    # 1. Unusual score statistics
    features['anomaly_score_mean'] = np.mean(anomaly_matrix, axis=1)
    features['anomaly_score_std'] = np.std(anomaly_matrix, axis=1)
    features['anomaly_score_max'] = np.max(anomaly_matrix, axis=1)
    features['anomaly_score_min'] = np.min(anomaly_matrix, axis=1)

    # 2. Unusual consistency features
    # Unusual voting (number above threshold)
    threshold = np.percentile(anomaly_matrix.flatten(), 75)  # Upper quartile as threshold
    anomaly_votes = (anomaly_matrix > threshold).sum(axis=1)
    features['anomaly_vote_count'] = anomaly_votes.astype(np.float32)
    features['anomaly_vote_ratio'] = anomaly_votes / len(anomaly_features)

    # 3. Features of unusual ranking
    # The ranking of each sample on abnormal features
    anomaly_ranks = np.argsort(np.argsort(anomaly_matrix, axis=0), axis=0)
    features['anomaly_avg_rank'] = np.mean(anomaly_ranks, axis=1) / len(anomaly_features)

    # 4. Features of abnormality
    if len(anomaly_features) > 2:
        # Calculating Pearson correlation between abnormal features (line direction)
        anomaly_correlations = np.zeros(len(df), dtype=np.float32)

        for i in range(len(df)):
            row_corrs = []
            for j in range(len(anomaly_features)):
                for k in range(j+1, len(anomaly_features)):
                    # Calculating the "relevance" of feature j and k on this sample.
                    val_j = anomaly_matrix[i, j]
                    val_k = anomaly_matrix[i, k]
                    row_corrs.append(val_j * val_k)

            if row_corrs:
                anomaly_correlations[i] = np.mean(row_corrs)

        features['anomaly_feature_correlation'] = anomaly_correlations

    # Unusual signature entropy (uncertainty)
    # Convert abnormal scores to probability distribution
    anomaly_probs = anomaly_matrix / (np.sum(anomaly_matrix, axis=1, keepdims=True) + 1e-8)
    anomaly_entropy = -np.sum(anomaly_probs * np.log(anomaly_probs + 1e-8), axis=1)
    features['anomaly_entropy'] = anomaly_entropy.astype(np.float32)

    # 6. Unusual combinations
    # Create a new abnormal feature combination
    for i in range(min(3, len(anomaly_features))):
        for j in range(i+1, min(4, len(anomaly_features))):
            feat1 = anomaly_features[i]
            feat2 = anomaly_features[j]

            # Multiplier Group
            product = df[feat1] * df[feat2]
            features[f'anomaly_product_{feat1}_{feat2}'] = product.values

            # Group of variances
            diff = np.abs(df[feat1] - df[feat2])
            features[f'anomaly_diff_{feat1}_{feat2}'] = diff.values

            # Reconciliation average
            harmonic_mean = 2 * df[feat1] * df[feat2] / (df[feat1] + df[feat2] + 1e-8)
            features[f'anomaly_harmonic_{feat1}_{feat2}'] = harmonic_mean.values

    return features

def compute_embedding_features_derived(df: pd.DataFrame,
                                     embedding_features: List[str]) -> Dict[str, np.ndarray]:
    """Calculate derivatives based on the embedded feature of the map"""
    features = {}

    if len(embedding_features) < 3:
        return features

    print(f"  Based on {len(embedding_features)} An embedded feature derivative...")

    # Create embedded feature matrix
    embedding_matrix = df[embedding_features].values

    # Geometric features of embedded space
    # L2 model (interpolated vector length)
    features['embedding_norm'] = np.linalg.norm(embedding_matrix, axis=1)

    # L1 model
    features['embedding_l1_norm'] = np.sum(np.abs(embedding_matrix), axis=1)

    # Cosine similarity feature (similarity to average vector)
    mean_embedding = np.mean(embedding_matrix, axis=0)
    mean_embedding_norm = np.linalg.norm(mean_embedding)

    cosine_similarities = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        vec_norm = np.linalg.norm(embedding_matrix[i])
        if vec_norm > 0 and mean_embedding_norm > 0:
            cosine_similarities[i] = embedding_matrix[i] @ mean_embedding / (vec_norm * mean_embedding_norm)

    features['embedding_cosine_to_mean'] = cosine_similarities

    # 2. Main composition features of embedded space
    if embedding_matrix.shape[1] > 2 and len(df) > 100:
        try:
            from sklearn.decomposition import PCA

            # Use PA to extract main directions
            pca = PCA(n_components=min(3, embedding_matrix.shape[1]))
            pca_features = pca.fit_transform(embedding_matrix)

            for i in range(pca_features.shape[1]):
                features[f'embedding_pca_{i}'] = pca_features[:, i]

            # Explanatory variance features
            features['embedding_pca_explained_ratio'] = pca.explained_variance_ratio_[0] * np.ones(len(df))

        except Exception as e:
            print(f"    PCACalculator Failed: {e}")

    # 3. Embedded dimensions statistics
    features['embedding_dim_mean'] = np.mean(embedding_matrix, axis=1)
    features['embedding_dim_std'] = np.std(embedding_matrix, axis=1)
    features['embedding_dim_max'] = np.max(embedding_matrix, axis=1)
    features['embedding_dim_min'] = np.min(embedding_matrix, axis=1)

    # 4. Features of embedded dimensions relevance
    if embedding_matrix.shape[1] > 2:
        # Compute the "inline correlation" of embedded dimensions
        embedding_corr_matrix = np.corrcoef(embedding_matrix.T)
        np.fill_diagonal(embedding_corr_matrix, 0)

        # Average absolute relevance
        mean_abs_corr = np.mean(np.abs(embedding_corr_matrix))
        features['embedding_dim_correlation'] = mean_abs_corr * np.ones(len(df))

    # 5. Embedded cluster features
    if len(df) > 100 and embedding_matrix.shape[1] > 2:
        try:
            from sklearn.cluster import KMeans

            # Use K-mes for simple grouping
            n_clusters = min(5, len(df) // 20)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                cluster_labels = kmeans.fit_predict(embedding_matrix)

                features['embedding_cluster'] = cluster_labels.astype(np.float32)

                # Distance to cluster centre
                distances = kmeans.transform(embedding_matrix)
                features['embedding_cluster_min_distance'] = np.min(distances, axis=1)

                # Cluster size features
                cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
                features['embedding_cluster_size'] = cluster_sizes[cluster_labels].astype(np.float32) / len(df)

        except Exception as e:
            print(f"    Embedded grouping failed: {e}")

    # 6. Embedded feature combinations
    # Select previous embedded features to group
    for i in range(min(3, len(embedding_features))):
        for j in range(i+1, min(4, len(embedding_features))):
            feat1 = embedding_features[i]
            feat2 = embedding_features[j]

            # Point-cum-Sex
            dot_product = df[feat1] * df[feat2]
            features[f'embedding_dot_{feat1}_{feat2}'] = dot_product.values

            # Oxygen distance signature.
            euclidean_dist = np.sqrt((df[feat1] - df[feat2]) ** 2)
            features[f'embedding_dist_{feat1}_{feat2}'] = euclidean_dist.values

    return features

def compute_statistical_features_derived(df: pd.DataFrame,
                                       statistical_features: List[str]) -> Dict[str, np.ndarray]:
    """Calculate derivative features based on statistical features"""
    features = {}

    if len(statistical_features) < 2:
        return features

    print(f"  Based on {len(statistical_features)} Statistical features derived...")

    # Create a statistical feature matrix
    stats_matrix = df[statistical_features].values

    # 1. Statistics of statistical features
    features['stats_of_stats_mean'] = np.mean(stats_matrix, axis=1)
    features['stats_of_stats_std'] = np.std(stats_matrix, axis=1)
    features['stats_of_stats_cv'] = np.divide(
        features['stats_of_stats_std'],
        np.abs(features['stats_of_stats_mean']) + 1e-8
    )

    # 2. Coherence of statistical features Volume
    # Calculate correlation between statistical features (line direction)
    stats_correlations = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        row_values = stats_matrix[i]
        if len(row_values) > 1 and np.std(row_values) > 0:
            # Use the variable factor as a measure of consistency
            stats_correlations[i] = 1.0 / (np.std(row_values) / np.mean(np.abs(row_values)) + 1e-8)

    features['stats_consistency'] = stats_correlations

    # 3. Integration of statistical features
    # Combining statistical features
    stats_normalized = stats_matrix / (np.std(stats_matrix, axis=0, keepdims=True) + 1e-8)

    # Weighting and weighting (weights are approximation of characteristic importance)
    weights = 1.0 / (1.0 + np.var(stats_normalized, axis=0))
    features['stats_weighted_sum'] = stats_normalized @ weights

    # 4. Derivation of the statistical features compartment
    for i in range(min(3, len(statistical_features))):
        feat = statistical_features[i]
        values = df[feat].values

        # Declination Box
        if len(values) > 10:
            # Centimeters
            quintiles = np.percentile(values, [20, 40, 60, 80])
            quintile_bins = np.digitize(values, quintiles)
            features[f'{feat}_quintile'] = quintile_bins.astype(np.float32)

            # Deciles
            deciles = np.percentile(values, np.linspace(10, 90, 9))
            decile_bins = np.digitize(values, deciles)
            features[f'{feat}_decile'] = decile_bins.astype(np.float32)

        # Subbox based on average and standard deviations
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val > 0:
            # Z-score
            zscore_bins = np.zeros_like(values, dtype=np.float32)
            zscore_bins[values < mean_val - std_val] = 0  # Low
            zscore_bins[(values >= mean_val - std_val) & (values <= mean_val + std_val)] = 1  # Medium
            zscore_bins[values > mean_val + std_val] = 2  # High

            features[f'{feat}_zscore_bin'] = zscore_bins

    # 5. Interaction of statistical features
    for i in range(min(3, len(statistical_features))):
        for j in range(i+1, min(4, len(statistical_features))):
            feat1 = statistical_features[i]
            feat2 = statistical_features[j]

            # Multiplication Interactive
            product = df[feat1] * df[feat2]
            features[f'stats_product_{feat1}_{feat2}'] = product.values

            # Ratio Interactive
            ratio = df[feat1] / (df[feat2] + 1e-8)
            features[f'stats_ratio_{feat1}_over_{feat2}'] = ratio.values

    return features

def compute_interaction_features_derived(df: pd.DataFrame,
                                       interaction_features: List[str]) -> Dict[str, np.ndarray]:
    """High-level derivatives based on interactive features"""
    features = {}

    if len(interaction_features) < 2:
        return features

    print(f"  Based on {len(interaction_features)} An interactive signature derivative...")

    # 1. Three-stage combination of interactive features
    # Select the previous interactive features for a third-stage combination
    if len(interaction_features) >= 3:
        for i in range(min(3, len(interaction_features))):
            for j in range(i+1, min(4, len(interaction_features))):
                for k in range(j+1, min(5, len(interaction_features))):
                    feat1 = interaction_features[i]
                    feat2 = interaction_features[j]
                    feat3 = interaction_features[k]

                    # Third factor product
                    product = df[feat1] * df[feat2] * df[feat3]
                    features[f'interaction_triple_{feat1}_{feat2}_{feat3}'] = product.values

                    # 3rd and average
                    harmonic_triple = 3 / (1/(df[feat1]+1e-8) + 1/(df[feat2]+1e-8) + 1/(df[feat3]+1e-8))
                    features[f'interaction_harmonic_triple_{feat1}_{feat2}_{feat3}'] = harmonic_triple.values

    # Statistical derivatives of interactive features
    interaction_matrix = df[interaction_features].values

    features['interaction_mean'] = np.mean(interaction_matrix, axis=1)
    features['interaction_std'] = np.std(interaction_matrix, axis=1)
    features['interaction_max'] = np.max(interaction_matrix, axis=1)
    features['interaction_min'] = np.min(interaction_matrix, axis=1)

    # 3. Interaction between interactive and original features
    # Looking for possible original features
    original_features = []
    for feat in interaction_features:
        # Try to parse original features in interactive features First Name
        if 'interaction_' in feat:
            parts = feat.replace('interaction_', '').split('_')
            if len(parts) >= 2:
                original_features.extend(parts[:2])

    original_features = list(set(original_features))

    # Interaction between interactive and original features
    for inter_feat in interaction_features[:3]:
        for orig_feat in original_features[:3]:
            if orig_feat in df.columns:
                # Multiplication of interactive and original features
                product = df[inter_feat] * df[orig_feat]
                features[f'interaction_with_original_{inter_feat}_{orig_feat}'] = product.values

    # Layer-based derivatives of interactive features
    for feat in interaction_features[:5]:
        values = df[feat].values

        # Layer based on fraction
        if len(values) > 5:
            percentiles = np.percentile(values, [25, 50, 75])

            # Create Layer Features
            level_feature = np.zeros_like(values, dtype=np.float32)
            level_feature[values <= percentiles[0]] = 0  # Low
            level_feature[(values > percentiles[0]) & (values <= percentiles[2])] = 1  # Medium
            level_feature[values > percentiles[2]] = 2  # High

            features[f'{feat}_level'] = level_feature

        # Interactive feature transformation
        features[f'{feat}_log1p'] = np.log1p(np.abs(values))
        features[f'{feat}_sqrt'] = np.sqrt(np.abs(values))

    return features

def compute_feature_binning_derived(df: pd.DataFrame,
                                  numeric_features: List[str]) -> Dict[str, np.ndarray]:
    """Calculate derivative features based on the feature compartment"""
    features = {}

    if len(numeric_features) == 0:
        return features

    print(f"  Yeah. {len(numeric_features)} Individual numeric features to be derived from the box...")

    for feat in numeric_features[:10]:  # Only the top 10 features are processed
        values = df[feat].values

        if len(np.unique(values)) < 5:
            continue  # Too few categories. Skip.

        # 1. Equivalent boxes
        n_bins = min(10, len(np.unique(values)) // 2)
        if n_bins >= 2:
            # Equivalent Halfbox
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                width = (max_val - min_val) / n_bins
                bins = [min_val + i * width for i in range(1, n_bins)]
                equal_width_bins = np.digitize(values, bins)
                features[f'{feat}_equal_width_bin'] = equal_width_bins.astype(np.float32)

        # 2. Equivalent fraction boxes
        if len(values) > n_bins:
            percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            quantiles = np.percentile(values, percentiles)
            equal_freq_bins = np.digitize(values, quantiles)
            features[f'{feat}_equal_freq_bin'] = equal_freq_bins.astype(np.float32)

        # 3. Cluster-based compartments
        if len(values) > 100:
            try:
                from sklearn.cluster import KMeans

                # Use K-mes for group boxes
                n_clusters = min(5, len(np.unique(values)) // 3)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                    values_2d = values.reshape(-1, 1)
                    cluster_labels = kmeans.fit_predict(values_2d)

                    # Sort labels by cluster centre
                    sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())
                    cluster_mapping = {old: new for new, old in enumerate(sorted_indices)}
                    ordered_labels = np.vectorize(cluster_mapping.get)(cluster_labels)

                    features[f'{feat}_cluster_bin'] = ordered_labels.astype(np.float32)

            except Exception as e:
                print(f"    Collar Box Failed {feat}: {e}")

        # 4. Statistical features of the boxes
        if f'{feat}_equal_freq_bin' in features:
            bin_labels = features[f'{feat}_equal_freq_bin']
            n_bins_actual = len(np.unique(bin_labels))

            # Calculate the amount of statistics per box
            for bin_id in range(n_bins_actual):
                mask = (bin_labels == bin_id)
                if np.sum(mask) > 0:
                    # Box average
                    bin_mean = np.mean(values[mask])
                    features[f'{feat}_bin{bin_id}_mean_ref'] = np.full_like(values, bin_mean, dtype=np.float32)

    return features

def compute_timeseries_style_derived(df: pd.DataFrame,
                                   numeric_features: List[str]) -> Dict[str, np.ndarray]:
    """Calculating false time series style features"""
    features = {}

    if len(numeric_features) < 2:
        return features

    print(f"  Calculate {len(numeric_features)} Hypothetical time series derived from each characteristic...")

    n_samples = len(df)

    # Create a false time series for each feature
    for feat in numeric_features[:5]:
        values = df[feat].values

        # 1. Slide window statistics (perfect time series)
        window_sizes = [5, 10, 20]

        for window_size in window_sizes:
            if window_size >= n_samples:
                continue

            # Initializing slide window arrays
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

            # Scroll change feature
            if window_size > 1:
                rolling_change = values - rolling_mean
                features[f'{feat}_rolling_change_ws{window_size}'] = rolling_change

        # 2. Ageing features (false time series)
        for lag in [1, 2, 3]:
            if lag < n_samples:
                lagged_values = np.zeros_like(values, dtype=np.float32)
                lagged_values[lag:] = values[:-lag]
                lagged_values[:lag] = values[0]  # Fill with first value

                features[f'{feat}_lag{lag}'] = lagged_values

                # Ageing variance
                lag_diff = values - lagged_values
                features[f'{feat}_lag{lag}_diff'] = lag_diff

                # Ageing rate
                lag_ratio = values / (lagged_values + 1e-8)
                features[f'{feat}_lag{lag}_ratio'] = lag_ratio

        # 3. Time-series trend features
        if n_samples > 10:
            # Simple linear trend (false time)
            x = np.arange(n_samples).reshape(-1, 1)

            # Calculate local trends using a rolling window
            trend_window = min(20, n_samples)
            local_trend = np.zeros_like(values, dtype=np.float32)

            for i in range(n_samples):
                start_idx = max(0, i - trend_window // 2)
                end_idx = min(n_samples, i + trend_window // 2)

                if end_idx - start_idx > 3:
                    x_window = x[start_idx:end_idx]
                    y_window = values[start_idx:end_idx]

                    # Simple linear slope
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
    """Calculating characteristic importance derivatives"""
    features = {}

    print("Calculating Feature Importance Derivated...")

    # Select Numerical Features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 5:
        return features

    # Feature importance of sampling (use of false labels)
    n_samples = len(df)

    # Generate false labels (using K-mes cluster)
    try:
        from sklearn.cluster import KMeans

        # Use the PCA downscaling grouping
        from sklearn.decomposition import PCA

        numeric_data = df[numeric_cols[:min(50, len(numeric_cols))]].values

        # PCA descends
        pca = PCA(n_components=min(10, numeric_data.shape[1], n_samples-1))
        reduced_data = pca.fit_transform(numeric_data)

        # K-means group generation of false labels
        n_clusters = min(3, n_samples // 100)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            pseudo_labels = kmeans.fit_predict(reduced_data)

            # Importance of using random forests to calculate features
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(numeric_data, pseudo_labels)
            importances = rf.feature_importances_

            # 1. Feature importance weighting
            weighted_sum = numeric_data @ importances
            features['importance_weighted_sum'] = weighted_sum.astype(np.float32)

            # 2. Key feature combinations
            # Select the most important features
            top_indices = np.argsort(importances)[-5:]  # First five important features

            for idx in top_indices:
                if idx < len(numeric_cols):
                    feat_name = numeric_cols[idx]
                    features[f'important_{feat_name}'] = df[feat_name].values

            # 3. Feature importance ranking features
            importance_ranks = np.argsort(np.argsort(importances))
            # Create a character importance vector for each sample
            for i, rank in enumerate(importance_ranks[-10:]):  # Only the top ten.
                if i < len(numeric_cols):
                    feat_name = numeric_cols[i]
                    features[f'{feat_name}_importance_rank'] = rank * np.ones(n_samples, dtype=np.float32)

            # 4. Consistency of features
            # Calculating the entropy of character importance distribution
            importance_probs = importances / np.sum(importances)
            importance_entropy = -np.sum(importance_probs * np.log(importance_probs + 1e-8))
            features['feature_importance_entropy'] = importance_entropy * np.ones(n_samples, dtype=np.float32)

    except Exception as e:
        print(f"    Character matter calculation failed: {e}")
        # Use simple replacements
        numeric_data = df[numeric_cols[:10]].values
        features['simple_weighted_sum'] = np.mean(numeric_data, axis=1)

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_paths', type=str, nargs='+', required=True,
                       help='Enter the list of paths for the feature file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output Profile Path')
    args = parser.parse_args()

    # Calculate derivative features
    features_df = compute_derived_features(args.feature_paths, args.output_path)

    print(f"Derivative processing complete.，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
