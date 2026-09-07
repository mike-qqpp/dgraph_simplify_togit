"""Mixed high-level feature module: high-level feature combining multiple features"""
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
from typing import Dict, List, Tuple  # Add this line
def compute_mixed_features(edge_index: np.ndarray,
                          node_features: np.ndarray,
                          n_components: int = 50,
                          output_path: str = None) -> pd.DataFrame:
    """Calculating high-level composite features:
1. Polynomial features (interactive features)
2. Main PCB constituents
3. Independent composition of ICA
4. Random projection features
5. Feature selection scores
6. Feature importance ranking
7. Feature combination statistics
8. Figure enhancement features"""

    print("Calculating Mixed High Level Features...")
    n_nodes, n_features = node_features.shape

    feature_dict = {}

    # Standardized features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(node_features)

    # 1. Polynomial features (2 steps)
    print("Calculating Multiple Features...")
    poly_features = compute_polynomial_features(features_scaled, degree=2)

    for i in range(min(20, poly_features.shape[1])):  # Only the top 20 multiple features are taken
        feature_dict[f'poly_feat_{i}'] = poly_features[:, i].astype(np.float32)

    # 2. Main PCB constituents
    print("Calculating PCA Main Component...")
    pca_features = compute_pca_features(features_scaled, n_components=min(20, n_components, n_features))

    for i in range(pca_features.shape[1]):
        feature_dict[f'pca_component_{i}'] = pca_features[:, i].astype(np.float32)

    # 3. Independent composition of ICA
    print("Calculating ICA Independence...")
    ica_features = compute_ica_features(features_scaled, n_components=min(10, n_components//2, n_features))

    for i in range(ica_features.shape[1]):
        feature_dict[f'ica_component_{i}'] = ica_features[:, i].astype(np.float32)

    # 4. Random projection features
    print("Calculates random projection features...")
    random_proj_features = compute_random_projection_features(
        features_scaled, n_components=min(15, n_components)
    )

    for i in range(random_proj_features.shape[1]):
        feature_dict[f'random_proj_{i}'] = random_proj_features[:, i].astype(np.float32)

    # Feature selection scores (based on variance and relevance)
    print("Calculate feature selection score...")
    feature_scores = compute_feature_selection_scores(features_scaled)

    for feat_name, values in feature_scores.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 6. Feature importance features
    print("Compute feature importance features...")
    importance_features = compute_feature_importance_features(features_scaled)

    for feat_name, values in importance_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 7. Feature combination statistics
    print("Compute feature combination statistics...")
    combination_features = compute_feature_combination_stats(features_scaled)

    for feat_name, values in combination_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 8. Figure enhancement features
    print("Calculator Chart Enhancement Features...")
    graph_enhanced_features = compute_graph_enhanced_features(edge_index, features_scaled)

    for feat_name, values in graph_enhanced_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 9. Feature complexity measures
    print("Calculating Feature Complexity...")
    complexity_features = compute_feature_complexity(features_scaled)

    for feat_name, values in complexity_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

    print(f"Mixed high-level characteristic dimensions: {features_df.shape}")
    return features_df

def compute_polynomial_features(features_scaled: np.ndarray, degree: int = 2) -> np.ndarray:
    """Calculating Multiple Features"""
    n_samples, n_features = features_scaled.shape

    # Limit the number of features to avoid a dimension explosion
    max_features_for_poly = min(8, n_features)
    features_for_poly = features_scaled[:, :max_features_for_poly]

    try:
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=True,  # Count interactive items only, avoid square items
            include_bias=False
        )

        poly_features = poly.fit_transform(features_for_poly)

        # If the dimensions are too big, cut.
        max_poly_features = 50
        if poly_features.shape[1] > max_poly_features:
            # Select the largest difference feature
            variances = np.var(poly_features, axis=0)
            top_indices = np.argsort(variances)[-max_poly_features:]
            poly_features = poly_features[:, top_indices]

        # Fill to target dimension if the original feature has a smaller dimension
        if poly_features.shape[1] < 20:
            padding = np.zeros((n_samples, 20 - poly_features.shape[1]), dtype=np.float32)
            poly_features = np.hstack([poly_features, padding])

        return poly_features.astype(np.float32)

    except Exception as e:
        print(f"Multiple feature calculation failed: {e}")
        # Returns simple interactive features
        simple_poly = np.zeros((n_samples, 20), dtype=np.float32)

        # Add some simple interactive features
        idx = 0
        for i in range(min(4, n_features)):
            for j in range(i+1, min(5, n_features)):
                if idx < 20:
                    simple_poly[:, idx] = features_scaled[:, i] * features_scaled[:, j]
                    idx += 1

        return simple_poly

def compute_pca_features(features_scaled: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Calculates the main PCB component"""
    n_samples = features_scaled.shape[0]

    if n_components >= n_samples:
        n_components = min(10, n_samples - 1)

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(n_components, features_scaled.shape[1], n_samples-1),
                 random_state=42)

        pca_features = pca.fit_transform(features_scaled)

        # Explanatory margin as an additional feature
        explained_variance_ratio = pca.explained_variance_ratio_

        return pca_features.astype(np.float32)

    except Exception as e:
        print(f"PCACalculator Failed: {e}")
        return np.zeros((n_samples, n_components), dtype=np.float32)

def compute_ica_features(features_scaled: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Calculating ICA Independence"""
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
        print(f"ICACalculator Failed: {e}")
        return np.zeros((n_samples, n_components), dtype=np.float32)

def compute_random_projection_features(features_scaled: np.ndarray, n_components: int = 15) -> np.ndarray:
    """Calculates random projection features"""
    n_samples = features_scaled.shape[0]

    try:
        from sklearn.random_projection import GaussianRandomProjection

        rp = GaussianRandomProjection(n_components=min(n_components, features_scaled.shape[1]),
                                     random_state=42)

        rp_features = rp.fit_transform(features_scaled)

        return rp_features.astype(np.float32)

    except Exception as e:
        print(f"Random projection failed: {e}")
        # Use simple random projection
        if features_scaled.shape[1] > 0:
            random_matrix = np.random.randn(features_scaled.shape[1], n_components).astype(np.float32)
            rp_features = features_scaled @ random_matrix
            return rp_features
        else:
            return np.zeros((n_samples, n_components), dtype=np.float32)

def compute_feature_selection_scores(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate feature selection score"""
    n_samples, n_features = features_scaled.shape

    features = {}

    # 1. Square score
    variances = np.var(features_scaled, axis=0)
    features['feature_variance_score'] = np.mean(variances) * np.ones(n_samples, dtype=np.float32)

    # 2. Interrelation scores
    if n_features > 1 and n_samples > 100:
        try:
            # Calculate the mean correlation between features
            correlation_matrix = np.corrcoef(features_scaled.T)
            np.fill_diagonal(correlation_matrix, 0)  # Ignore custom relevance

            # Average absolute relevance
            avg_correlation = np.mean(np.abs(correlation_matrix))
            features['feature_avg_correlation'] = avg_correlation * np.ones(n_samples, dtype=np.float32)

            # Maximum Relevance
            max_correlation = np.max(np.abs(correlation_matrix))
            features['feature_max_correlation'] = max_correlation * np.ones(n_samples, dtype=np.float32)

        except:
            features['feature_avg_correlation'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_max_correlation'] = np.zeros(n_samples, dtype=np.float32)

    # 3. Feature stability scores (through Bootstream)
    if n_samples > 100 and n_features > 1:
        stability_scores = compute_feature_stability(features_scaled)
        features['feature_stability_score'] = stability_scores.astype(np.float32)

    return features

def compute_feature_stability(features_scaled: np.ndarray, n_bootstrap: int = 10) -> np.ndarray:
    """Calculate signature stability through Bootstream"""
    n_samples, n_features = features_scaled.shape

    if n_samples < 50 or n_features < 2:
        return np.ones(n_samples, dtype=np.float32)

    # Storage of each Botstream feature importance (use variance as material)
    importances_list = []

    for _ in range(min(n_bootstrap, 5)):
        # It's put back for sampling.
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_features = features_scaled[bootstrap_indices]

        # Calculate variance as material
        variances = np.var(bootstrap_features, axis=0)
        importances_list.append(variances)

    # Consistency in calculating materiality
    importances_array = np.array(importances_list)  # shape: (n_bootstrap, n_features)

    # Calculate stability for each feature (minus of variance)
    stability_per_feature = 1.0 / (np.std(importances_array, axis=0) + 1e-8)

    # Average stability as a score for each sample
    avg_stability = np.mean(stability_per_feature)

    return avg_stability * np.ones(n_samples, dtype=np.float32)

def compute_feature_importance_features(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate features relevant to the importance of the feature"""
    n_samples, n_features = features_scaled.shape

    features = {}

    # Feature importance of using random forest (sampling)
    if n_samples > 100 and n_features > 1:
        try:
            # Generate false labels for character importance calculations
            # Generate false labels using K-means group
            from sklearn.cluster import KMeans

            n_clusters = min(5, n_samples // 20)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                pseudo_labels = kmeans.fit_predict(features_scaled)

                # Importance of using random forests to calculate features
                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(features_scaled, pseudo_labels)

                importances = rf.feature_importances_

                # Feature importance statistics
                features['feature_importance_mean'] = np.mean(importances) * np.ones(n_samples, dtype=np.float32)
                features['feature_importance_std'] = np.std(importances) * np.ones(n_samples, dtype=np.float32)
                features['feature_importance_max'] = np.max(importances) * np.ones(n_samples, dtype=np.float32)

                # Feature weighting of each sample
                weighted_sum = features_scaled @ importances
                features['feature_weighted_sum'] = weighted_sum.astype(np.float32)

        except Exception as e:
            print(f"Character matter calculation failed: {e}")
            features['feature_importance_mean'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_importance_std'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_importance_max'] = np.zeros(n_samples, dtype=np.float32)
            features['feature_weighted_sum'] = np.zeros(n_samples, dtype=np.float32)

    return features

def compute_feature_combination_stats(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute feature combination statistics"""
    n_samples, n_features = features_scaled.shape

    features = {}

    # 1. Feature values and (L1 model)
    features['feature_l1_norm'] = np.sum(np.abs(features_scaled), axis=1).astype(np.float32)

    # 2. Feature values squared (L2 model)
    features['feature_l2_norm'] = np.sqrt(np.sum(features_scaled**2, axis=1)).astype(np.float32)

    # 3. Maximum feature value
    features['feature_max'] = np.max(features_scaled, axis=1).astype(np.float32)

    # 4. Minimum feature values
    features['feature_min'] = np.min(features_scaled, axis=1).astype(np.float32)

    # 5. Feature scope
    features['feature_range'] = features['feature_max'] - features['feature_min']

    # 6. Feature mean
    features['feature_mean'] = np.mean(features_scaled, axis=1).astype(np.float32)

    # 7. Feature criteria Bad
    if n_features > 1:
        features['feature_std'] = np.std(features_scaled, axis=1).astype(np.float32)

    # 8. Feature bias
    if n_features > 2:
        from scipy import stats
        features['feature_skewness'] = stats.skew(features_scaled, axis=1).astype(np.float32)

    # Feature peaks
    if n_features > 3:
        features['feature_kurtosis'] = stats.kurtosis(features_scaled, axis=1).astype(np.float32)

    # 10. Feature entropy (volume of information)
    # Convert feature values to probability distribution
    feature_probs = np.abs(features_scaled) / (np.sum(np.abs(features_scaled), axis=1, keepdims=True) + 1e-8)
    feature_entropy = -np.sum(feature_probs * np.log(feature_probs + 1e-8), axis=1)
    features['feature_entropy'] = feature_entropy.astype(np.float32)

    # 11. Feature Gini coefficient (uneven)
    sorted_features = np.sort(np.abs(features_scaled), axis=1)
    cumulative = np.cumsum(sorted_features, axis=1)

    if n_features > 1:
        # Normalization
        cumulative_norm = cumulative / (cumulative[:, -1:] + 1e-8)

        # Calculation of the Gini coefficient
        gini = 1 - 2 * np.mean(cumulative_norm, axis=1)
        features['feature_gini'] = gini.astype(np.float32)

    return features

def compute_graph_enhanced_features(edge_index: np.ndarray,
                                  features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Computation chart enhancement feature"""
    n_nodes = features_scaled.shape[0]

    # Create an adjacent matrix
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)),
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)

    features = {}

    # Turapras smooth features
    print("Calculating Turapras Smooth...")
    try:
        # Calculating the normalized Lapras Matrix
        degree = np.array(adj_matrix_sym.sum(axis=1)).flatten()
        degree_sqrt = np.sqrt(degree)
        degree_sqrt[degree_sqrt == 0] = 1

        D_sqrt_inv = sp.diags(1.0 / degree_sqrt)
        L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix_sym @ D_sqrt_inv

        # Calculates smoothness: x^T L x
        smoothness = np.zeros(n_nodes, dtype=np.float32)
        for i in range(min(3, features_scaled.shape[1])):  # Only the first three features are calculated
            feature_vec = features_scaled[:, i]
            smoothness += feature_vec * (L @ feature_vec)

        features['graph_smoothness'] = smoothness.astype(np.float32)

    except:
        features['graph_smoothness'] = np.zeros(n_nodes, dtype=np.float32)

    # Figure proliferation features
    print("Calculating Fragmenting Features...")
    diffusion_features = compute_graph_diffusion_features(adj_matrix_sym, features_scaled)

    for feat_name, values in diffusion_features.items():
        features[f'graph_diffusion_{feat_name}'] = values.astype(np.float32)

    # 3. Figuring features
    print("Calculating Attention Features...")
    attention_features = compute_graph_attention_features(adj_matrix_sym, features_scaled)

    for feat_name, values in attention_features.items():
        features[f'graph_attention_{feat_name}'] = values.astype(np.float32)

    return features

def compute_graph_diffusion_features(adj_matrix: sp.csr_matrix,
                                   features_scaled: np.ndarray,
                                   diffusion_steps: int = 2) -> Dict[str, np.ndarray]:
    """Calculating Fragmentation Features"""
    n_nodes = features_scaled.shape[0]

    features = {}

    try:
        # Normalized Neighbourhood Matrix
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        norm_adj = adj_matrix.multiply(1.0 / degree[:, np.newaxis])

        # Initial features
        diffused_features = features_scaled.copy()

        for step in range(1, diffusion_steps + 1):
            # Proliferation: A*X
            diffused_features = norm_adj @ diffused_features

            # Save the amount of statistics per step
            if step <= 2:  # Save only the first two steps
                # Mean
                features[f'step{step}_mean'] = np.mean(diffused_features, axis=1).astype(np.float32)

                # Standard deviation
                if diffused_features.shape[1] > 1:
                    features[f'step{step}_std'] = np.std(diffused_features, axis=1).astype(np.float32)

                # Difference from original features
                if step == 1:
                    diff_norm = np.linalg.norm(diffused_features - features_scaled, axis=1)
                    features['diffusion_change_norm'] = diff_norm.astype(np.float32)

    except Exception as e:
        print(f"Figure proliferation calculation failed: {e}")
        features['diffusion_change_norm'] = np.zeros(n_nodes, dtype=np.float32)

    return features

def compute_graph_attention_features(adj_matrix: sp.csr_matrix,
                                   features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculating simple graphic attention features"""
    n_nodes = features_scaled.shape[0]
    adj_matrix_csc = adj_matrix.tocsc()

    features = {}

    # Simple attention mechanisms: based on features similarity
    attention_scores = np.zeros(n_nodes, dtype=np.float32)

    # Count only the attention of some nodes
    sample_size = min(5000, n_nodes)
    sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)

    for node in tqdm(sampled_nodes, desc="Focus.", leave=False):
        neighbors = adj_matrix_csc[:, node].indices

        if len(neighbors) == 0:
            attention_scores[node] = 0
            continue

        # Calculating cosine similarities between node features and neighbour features
        node_feat = features_scaled[node]
        neighbor_feats = features_scaled[neighbors]

        # Cosine Similarity
        norms = np.linalg.norm(node_feat) * np.linalg.norm(neighbor_feats, axis=1)
        valid_mask = norms > 0

        if np.any(valid_mask):
            similarities = np.zeros(len(neighbors))
            similarities[valid_mask] = (node_feat @ neighbor_feats[valid_mask].T) / norms[valid_mask]

            # Focus score: average similarity
            attention_scores[node] = np.mean(similarities[valid_mask])

    # Fill unsampled nodes
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)

    if len(sampled_nodes) > 0:
        mean_attention = attention_scores[sampled_nodes].mean()
        attention_scores[unsampled] = mean_attention

    features['attention_score'] = attention_scores

    return features

def compute_feature_complexity(features_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute feature complex measures"""
    n_samples = features_scaled.shape[0]

    features = {}

    # 1. Feature dimensions (constant)
    features['feature_dimensionality'] = np.ones(n_samples, dtype=np.float32) * features_scaled.shape[1]

    # 2. Validity dimensions (based on characteristic values)
    try:
        # Calculating the characteristic values of the AC matrix
        if n_samples > 100:
            cov_matrix = np.cov(features_scaled.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Calculates the effective dimension (characterium)
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-8)
            effective_dim = np.exp(-np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-8)))

            features['feature_effective_dim'] = effective_dim * np.ones(n_samples, dtype=np.float32)
        else:
            features['feature_effective_dim'] = np.ones(n_samples, dtype=np.float32) * features_scaled.shape[1]

    except:
        features['feature_effective_dim'] = np.ones(n_samples, dtype=np.float32) * features_scaled.shape[1]

    # 3. Feature redundancy (based on relevance)
    if features_scaled.shape[1] > 1:
        try:
            correlation_matrix = np.corrcoef(features_scaled.T)
            np.fill_diagonal(correlation_matrix, 0)

            # Average absolute relevance as redundancy measure
            redundancy = np.mean(np.abs(correlation_matrix))
            features['feature_redundancy'] = redundancy * np.ones(n_samples, dtype=np.float32)

        except:
            features['feature_redundancy'] = np.zeros(n_samples, dtype=np.float32)

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--n_components', type=int, default=50, help='Number of dimensions down')
    args = parser.parse_args()

    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']

    print(f"Number of nodes: {x.shape[0]}, Feature shape: {x.shape[1]}")

    # Calculating Mixed High Level Features
    features_df = compute_mixed_features(
        edge_index, x,
        n_components=args.n_components
    )

    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)

    print(f"Mixed high-level feature processing completed，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
