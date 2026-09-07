import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse
import scipy.sparse as sp
from tqdm import tqdm

def compute_structural_features(edge_index: np.ndarray,
                               n_nodes: int,
                               max_neighbors: int = 1000,
                               output_path: str = None) -> pd.DataFrame:
    """Calculate the chart structure features:
1. Degree features
PageRank
3. Triangle count
4. Central features
5. Community features"""

    print("Calculating Structural Features...")

    # Create an adjacent matrix
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)),
                              shape=(n_nodes, n_nodes))

    # Symmetrical Neighbourhood Matrix (no direction)
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)

    feature_dict = {}

    # 1. Degree features
    print("Calculator Character...")
    # Output
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    feature_dict['out_degree'] = out_degree.astype(np.float32)

    # Input
    in_degree = np.array(adj_matrix.sum(axis=0)).flatten()
    feature_dict['in_degree'] = in_degree.astype(np.float32)

    # Total (no direction)
    total_degree = np.array(adj_matrix_sym.sum(axis=1)).flatten()
    feature_dict['total_degree'] = total_degree.astype(np.float32)

    # Degree logarithmic Variable
    feature_dict['out_degree_log'] = np.log1p(out_degree).astype(np.float32)
    feature_dict['in_degree_log'] = np.log1p(in_degree).astype(np.float32)
    feature_dict['total_degree_log'] = np.log1p(total_degree).astype(np.float32)

    # Normalization
    feature_dict['out_degree_norm'] = (out_degree / (out_degree.max() + 1e-8)).astype(np.float32)
    feature_dict['in_degree_norm'] = (in_degree / (in_degree.max() + 1e-8)).astype(np.float32)

    # PageRank (approximate calculation)
    print("Calculate PageRank...")
    pagerank_scores = compute_pagerank(adj_matrix_sym, max_iter=20)
    feature_dict['pagerank'] = pagerank_scores.astype(np.float32)
    feature_dict['pagerank_log'] = np.log1p(pagerank_scores).astype(np.float32)

    # 3. Local concentration factor (sampling calculation)
    print("Calculating the polygraph coefficient...")
    clustering_coeff = compute_clustering_coefficient_sampled(adj_matrix_sym,
                                                             sample_size=min(1000, n_nodes))
    feature_dict['clustering_coeff'] = clustering_coeff.astype(np.float32)

    # 4. Centrality
    feature_dict['degree_centrality'] = (total_degree / (n_nodes - 1)).astype(np.float32)

    # 5. Number of secondary neighbours
    print("Counting the second-tier neighbor...")
    adj_squared = adj_matrix_sym.dot(adj_matrix_sym)
    second_neighbors = np.array(adj_squared.sum(axis=1)).flatten()
    feature_dict['second_neighbors'] = second_neighbors.astype(np.float32)
    feature_dict['second_neighbors_log'] = np.log1p(second_neighbors).astype(np.float32)

    # 6. Neighborhood statistics
    print("Calculating Neighborhood Statistics...")
    neighbor_degree_stats = compute_neighbor_degree_stats(adj_matrix_sym, total_degree)
    for stat_name, stat_values in neighbor_degree_stats.items():
        feature_dict[f'neighbor_{stat_name}'] = stat_values.astype(np.float32)

    # 7. Structural Hole Indicators (approximate)
    print("Calculating Structural Hole Indicators...")
    constraint = compute_constraint_sampled(adj_matrix_sym, sample_size=min(500, n_nodes))
    feature_dict['constraint'] = constraint.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        with open(output_path, 'wb') as f:
            import pickle
            pickle.dump(features_df, f)

    return features_df

def compute_pagerank(adj_matrix: sp.csr_matrix,
                    alpha: float = 0.85,
                    max_iter: int = 20) -> np.ndarray:
    """Calculate PageRank"""
    n_nodes = adj_matrix.shape[0]

    # Normalized Neighbourhood Matrix
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    norm_adj = adj_matrix.multiply(1.0 / degree[:, np.newaxis])

    # Initialize
    pr = np.ones(n_nodes) / n_nodes

    # iterative calculation
    for _ in range(max_iter):
        pr_new = alpha * norm_adj.T.dot(pr) + (1 - alpha) / n_nodes
        diff = np.abs(pr_new - pr).sum()
        pr = pr_new

        if diff < 1e-6:
            break

    return pr

def compute_clustering_coefficient_sampled(adj_matrix: sp.csr_matrix,
                                         sample_size: int = 1000) -> np.ndarray:
    """Sampling calculates a concentration factor"""
    n_nodes = adj_matrix.shape[0]
    clustering = np.zeros(n_nodes, dtype=np.float32)

    # Random sample nodes
    if n_nodes > sample_size:
        sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    else:
        sampled_nodes = np.arange(n_nodes)

    adj_matrix = adj_matrix.tocsc()

    for node in tqdm(sampled_nodes, desc="Calculating the concentration factor"):
        # Get the neighbors.
        neighbors = adj_matrix[:, node].indices
        k = len(neighbors)

        if k < 2:
            clustering[node] = 0
            continue

        # Calculate number of connections between neighbours
        neighbor_pairs = 0
        for i in range(k):
            for j in range(i+1, k):
                if adj_matrix[neighbors[i], neighbors[j]]:
                    neighbor_pairs += 1

        clustering[node] = 2 * neighbor_pairs / (k * (k - 1))

    # For unsampled nodes, fill with average
    if n_nodes > sample_size:
        mean_clustering = clustering[sampled_nodes].mean()
        all_indices = np.arange(n_nodes)
        unsampled = np.setdiff1d(all_indices, sampled_nodes)
        clustering[unsampled] = mean_clustering

    return clustering

def compute_neighbor_degree_stats(adj_matrix: sp.csr_matrix,
                                node_degrees: np.ndarray) -> Dict[str, np.ndarray]:
    """Statistical features of the calculation of a neighbour"""
    n_nodes = adj_matrix.shape[0]

    # Initialization of statistics
    neighbor_mean = np.zeros(n_nodes, dtype=np.float32)
    neighbor_max = np.zeros(n_nodes, dtype=np.float32)
    neighbor_min = np.zeros(n_nodes, dtype=np.float32)
    neighbor_std = np.zeros(n_nodes, dtype=np.float32)

    adj_matrix = adj_matrix.tocsc()

    for node in tqdm(range(n_nodes), desc="Compute neighbourhood statistics"):
        neighbors = adj_matrix[:, node].indices

        if len(neighbors) == 0:
            neighbor_mean[node] = 0
            neighbor_max[node] = 0
            neighbor_min[node] = 0
            neighbor_std[node] = 0
            continue

        neighbor_degrees = node_degrees[neighbors]
        neighbor_mean[node] = neighbor_degrees.mean()
        neighbor_max[node] = neighbor_degrees.max()
        neighbor_min[node] = neighbor_degrees.min()
        neighbor_std[node] = neighbor_degrees.std() if len(neighbors) > 1 else 0

    return {
        'degree_mean': neighbor_mean,
        'degree_max': neighbor_max,
        'degree_min': neighbor_min,
        'degree_std': neighbor_std
    }

def compute_constraint_sampled(adj_matrix: sp.csr_matrix,
                             sample_size: int = 500) -> np.ndarray:
    """Sample calculation of structural hole binding factor"""
    n_nodes = adj_matrix.shape[0]
    constraint = np.zeros(n_nodes, dtype=np.float32)

    if n_nodes > sample_size:
        sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
    else:
        sampled_nodes = np.arange(n_nodes)

    adj_matrix = adj_matrix.tocsc()

    for node in tqdm(sampled_nodes, desc="Compute structural hole constraints"):
        neighbors = adj_matrix[:, node].indices
        total_connection = len(neighbors)

        if total_connection == 0:
            constraint[node] = 1
            continue

        constraint_sum = 0
        for neighbor in neighbors:
            # Calculatep {iq}
            p_iq = 1.0 / total_connection

            # Counting Public Neighbors
            neighbor_neighbors = adj_matrix[:, neighbor].indices
            common_neighbors = np.intersect1d(neighbors, neighbor_neighbors)

            # Calculate p {iq} *p {qj}
            for common in common_neighbors:
                if common != node and common != neighbor:
                    p_qj = 1.0 / len(neighbor_neighbors)
                    constraint_sum += p_iq * p_qj

        constraint[node] = constraint_sum

    # Fill unsampled nodes
    if n_nodes > sample_size:
        mean_constraint = constraint[sampled_nodes].mean()
        all_indices = np.arange(n_nodes)
        unsampled = np.setdiff1d(all_indices, sampled_nodes)
        constraint[unsampled] = mean_constraint

    return constraint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--max_neighbors', type=int, default=1000, help='Maximum number of neighbours')
    args = parser.parse_args()

    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']

    # Get Nodes
    if 'x' in data:
        n_nodes = data['x'].shape[0]
    else:
        n_nodes = edge_index.max() + 1

    # Calculate structural features
    features_df = compute_structural_features(edge_index, n_nodes, args.max_neighbors)

    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)

    print(f"Structure feature processing complete, feature dimension:{features_df.shape}")
    print(f"Features saved to:{args.output_path}")

if __name__ == "__main__":
    main()
