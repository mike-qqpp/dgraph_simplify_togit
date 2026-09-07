"""Spectrum feature module: features based on the map spectrum"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # Add this line
def compute_spectral_features(edge_index: np.ndarray,
                             node_features: np.ndarray,
                             k_eigenvalues: int = 20,
                             use_chebyshev: bool = True,
                             output_path: str = None) -> pd.DataFrame:
    """Calculator spectrum features:
La Plas characterization vector
Graphical signal smoothing degrees
3. Spectrum features
Chebishev Polygraph
5. Spacing features"""

    print("Calculating Spectrum Features...")
    n_nodes = node_features.shape[0]

    # Create symmetrical adjacent matrix
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)),
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)

    feature_dict = {}

    # 1. Normalization of the Lapras characterization vector
    print("Calculating the Lapras feature vector...")
    laplacian_features = compute_laplacian_eigenvectors_sampled(
        adj_matrix_sym, node_features, k_eigenvalues=min(k_eigenvalues, 15)
    )

    for i in range(laplacian_features.shape[1]):
        feature_dict[f'laplacian_eig_{i}'] = laplacian_features[:, i].astype(np.float32)

    # Figure signal smoothing features
    print("Calculating graph signal smoothness...")
    smoothness_features = compute_smoothness_features(adj_matrix_sym, node_features)

    for feat_name, values in smoothness_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 3. Spectrum concentration factors
    print("Calculating spectral fusion coefficients...")
    spectral_clustering_features = compute_spectral_clustering_features(
        adj_matrix_sym, n_clusters=10
    )

    for i in range(spectral_clustering_features.shape[1]):
        feature_dict[f'spectral_cluster_{i}'] = spectral_clustering_features[:, i].astype(np.float32)

    # Chebishev Polygraph
    if use_chebyshev:
        print("Calculating ChebbySchev Multiple Features...")
        chebyshev_features = compute_chebyshev_features(
            adj_matrix_sym, node_features, order=3
        )

        for i in range(chebyshev_features.shape[1]):
            feature_dict[f'chebyshev_{i}'] = chebyshev_features[:, i].astype(np.float32)

    # 5. Spacing features
    print("Calculating Spacing Features...")
    spectral_gap_features = compute_spectral_gap_features(adj_matrix_sym)

    for feat_name, values in spectral_gap_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # 6. Spectrum based on nodal features Response
    print("Calculating Node Feature Spectra Response...")
    spectral_response_features = compute_spectral_response_features(
        adj_matrix_sym, node_features
    )

    for feat_name, values in spectral_response_features.items():
        feature_dict[feat_name] = values.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

    print(f"Spectrum feature dimensions: {features_df.shape}")
    return features_df

def compute_laplacian_eigenvectors_sampled(adj_matrix: sp.csr_matrix,
                                         node_features: np.ndarray,
                                         k_eigenvalues: int = 10) -> np.ndarray:
    """Calculation of La Plas characterization vector (use of sampling and approximation)"""
    n_nodes = adj_matrix.shape[0]

    # Calculate a complete feature vector for small maps
    if n_nodes <= 10000:
        try:
            # Calculating the normalized Lapras Matrix
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree + 1e-8))
            L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv

            # Calculate feature vector
            eigenvalues, eigenvectors = eigsh(L, k=min(k_eigenvalues+1, n_nodes-1),
                                             which='SM', maxiter=1000)

            # Exclude first zero feature value
            if eigenvectors.shape[1] > 1:
                return eigenvectors[:, 1:k_eigenvalues+1].astype(np.float32)
            else:
                return compute_approximate_eigenvectors(adj_matrix, node_features, k_eigenvalues)

        except Exception as e:
            print(f"Full feature vector calculation failed: {e}")
            return compute_approximate_eigenvectors(adj_matrix, node_features, k_eigenvalues)

    # Use approximation for large maps
    return compute_approximate_eigenvectors(adj_matrix, node_features, k_eigenvalues)

def compute_approximate_eigenvectors(adj_matrix: sp.csr_matrix,
                                   node_features: np.ndarray,
                                   k_eigenvalues: int = 10) -> np.ndarray:
    """Approximate calculation of characteristic vectors"""
    n_nodes, n_features = node_features.shape

    # Simple approximation based on node features
    eigenvectors = np.zeros((n_nodes, k_eigenvalues), dtype=np.float32)

    # Use linear combinations of node features as approximate feature vectors
    for i in range(k_eigenvalues):
        # Create random weights
        if i < min(5, n_features):
            # Previous uses original features
            eigenvectors[:, i] = node_features[:, i % n_features]
        else:
            # Use random combinations behind
            weights = np.random.randn(n_features).astype(np.float32)
            eigenvectors[:, i] = node_features @ weights

        # Normalization
        norm = np.linalg.norm(eigenvectors[:, i])
        if norm > 0:
            eigenvectors[:, i] = eigenvectors[:, i] / norm

    # Add Chart Structure Information
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    eigenvectors[:, 0] = degree / (np.max(degree) + 1e-8)

    if k_eigenvalues > 1:
        eigenvectors[:, 1] = np.log1p(degree) / (np.log1p(np.max(degree)) + 1)

    return eigenvectors.astype(np.float32)

def compute_smoothness_features(adj_matrix: sp.csr_matrix,
                              node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculating signal smoothness features"""
    n_nodes = node_features.shape[0]

    # Calculating the normalized Lapras Matrix
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree + 1e-8))
    L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv

    features = {}

    # Calculates the smoothness of each feature
    smoothness_scores = np.zeros((n_nodes, node_features.shape[1]), dtype=np.float32)

    for i in range(min(5, node_features.shape[1])):  # Only the first five features are calculated
        feature_vec = node_features[:, i]
        # Smoothness: x^TL x
        smoothness = feature_vec * (L @ feature_vec)
        smoothness_scores[:, i] = smoothness

    # Statistics
    features['smoothness_mean'] = np.mean(smoothness_scores, axis=1).astype(np.float32)
    features['smoothness_max'] = np.max(smoothness_scores, axis=1).astype(np.float32)
    features['smoothness_std'] = np.std(smoothness_scores, axis=1).astype(np.float32)

    # Neighborship consistency
    adj_matrix_csc = adj_matrix.tocsc()
    neighbor_smoothness = np.zeros(n_nodes, dtype=np.float32)

    for node in tqdm(range(min(10000, n_nodes)), desc="Neighbors Smoothness", leave=False):
        neighbors = adj_matrix_csc[:, node].indices

        if len(neighbors) == 0:
            neighbor_smoothness[node] = 0
            continue

        # Calculate the difference between node and neighbour's smoothness
        node_smoothness = features['smoothness_mean'][node]
        neighbor_smoothnesses = features['smoothness_mean'][neighbors]

        if len(neighbors) > 0:
            neighbor_smoothness[node] = np.mean(np.abs(node_smoothness - neighbor_smoothnesses))

    features['neighbor_smoothness_diff'] = neighbor_smoothness

    return features

def compute_spectral_clustering_features(adj_matrix: sp.csr_matrix,
                                       n_clusters: int = 10) -> np.ndarray:
    """Calculating spectral group features"""
    n_nodes = adj_matrix.shape[0]

    # Simplified version spectral group: Usage feature for K-mes
    degree = np.array(adj_matrix.sum(axis=1)).flatten()

    # Create feature matrix: degrees and degrees change
    features = np.column_stack([
        degree / (np.max(degree) + 1e-8),
        np.log1p(degree) / (np.log1p(np.max(degree)) + 1),
        np.sqrt(degree) / (np.sqrt(np.max(degree)) + 1)
    ])

    # Use PCA or direct use feature
    from sklearn.decomposition import PCA

    if n_nodes > n_clusters:
        pca = PCA(n_components=min(n_clusters, features.shape[1]))
        cluster_features = pca.fit_transform(features)
    else:
        cluster_features = features

    # If not enough dimensions, fill zero.
    if cluster_features.shape[1] < n_clusters:
        padding = np.zeros((n_nodes, n_clusters - cluster_features.shape[1]), dtype=np.float32)
        cluster_features = np.hstack([cluster_features, padding])

    return cluster_features[:, :n_clusters].astype(np.float32)

def compute_chebyshev_features(adj_matrix: sp.csr_matrix,
                             node_features: np.ndarray,
                             order: int = 3) -> np.ndarray:
    """Calculating ChebbySchev Multiple Features"""
    n_nodes, n_features = node_features.shape

    # Normalized Neighbourhood Matrix
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    norm_adj = adj_matrix.multiply(1.0 / degree[:, np.newaxis])

    # Chebby Schiff's Multi-Range
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

    # Merge all steps
    all_features = np.hstack(chebyshev_features)

    # Limit dimensions
    max_features = 20
    if all_features.shape[1] > max_features:
        # Select the largest difference feature
        variances = np.var(all_features, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        all_features = all_features[:, top_indices]

    return all_features.astype(np.float32)

def compute_spectral_gap_features(adj_matrix: sp.csr_matrix) -> Dict[str, np.ndarray]:
    """Calculating spacing features"""
    n_nodes = adj_matrix.shape[0]

    features = {}

    # Calculator
    degree = np.array(adj_matrix.sum(axis=1)).flatten()

    # Approximate spectra: degree-based statistics
    degree_sorted = np.sort(degree)

    # spectral gap approximation: differences in maximum and lesser magnitude
    if len(degree_sorted) >= 2:
        max_degree = degree_sorted[-1]
        second_max = degree_sorted[-2]
        spectral_gap_approx = (max_degree - second_max) / (max_degree + 1e-8)
    else:
        spectral_gap_approx = 0

    # Ratio per node to maximum
    if np.max(degree) > 0:
        degree_ratio = degree / np.max(degree)
        features['degree_spectral_gap'] = degree_ratio * spectral_gap_approx

    # Square difference in degree distribution (measures of spectro width)
    if len(degree) > 1:
        degree_variance = np.var(degree)
        features['degree_variance_norm'] = (degree_variance * np.ones(n_nodes) / (np.max(degree)**2 + 1e-8)).astype(np.float32)

    # logarithmic gap of degrees
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
    """Calculate the spectra of node features Response"""
    n_nodes, n_features = node_features.shape

    features = {}

    # Calculating the normalized Lapras Matrix
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree + 1e-8))
    L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv

    # Low frequency response (soft mass)
    low_freq_response = np.zeros(n_nodes, dtype=np.float32)

    # HF response (details)
    high_freq_response = np.zeros(n_nodes, dtype=np.float32)

    # Computation spectrum response for each feature
    for i in range(min(3, n_features)):  # Only the first three features are calculated
        feature_vec = node_features[:, i]

        # Low frequency response: L^s approximation (soft)
        try:
            # Approximation of the Acacia
            low_freq = feature_vec.copy()
            for _ in range(3):  # 3 Times
                low_freq = 0.5 * (low_freq + (L @ low_freq))

            low_freq_response += np.abs(low_freq)
        except:
            pass

        # HF response: L features
        high_freq = L @ feature_vec
        high_freq_response += np.abs(high_freq)

    # Normalization
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
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--k_eigenvalues', type=int, default=20, help='Number of feature vectors')
    parser.add_argument('--no_chebyshev', action='store_true', help='Disable Chebishev feature')
    args = parser.parse_args()

    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']

    print(f"Number of nodes: {x.shape[0]}, Feature shape: {x.shape[1]}")

    # Calculating Spectrum Features
    features_df = compute_spectral_features(
        edge_index, x,
        k_eigenvalues=args.k_eigenvalues,
        use_chebyshev=not args.no_chebyshev
    )

    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)

    print(f"Spectrum feature processing complete.，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
