"""Figure embedded feature module: embedded feature based on chart structure"""
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple  # Add this line
def compute_embedding_features(edge_index: np.ndarray,
                              node_features: np.ndarray,
                              embedding_dim: int = 64,
                              output_path: str = None) -> pd.DataFrame:
    """Calculating figure embedded features:
1. Neighbourhood Matrix SVD
2. Lapras characteristic vector
3. Nodal feature embedded
4. Degree structure embedded
5. Random travel statistics
6. Neural network style embedded"""

    print("Calculating the embedded feature of the map...")
    n_nodes = node_features.shape[0]

    # Create symmetrical adjacent matrix
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)),
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)

    feature_dict = {}

    # 1. Adjacent matrix SVD (fast approximation)
    print("Calculate an adjacent matrix SVD...")
    svd_features = compute_adjacency_svd(
        adj_matrix_sym, n_components=min(32, embedding_dim, n_nodes-1)
    )

    for i in range(svd_features.shape[1]):
        feature_dict[f'svd_dim{i}'] = svd_features[:, i].astype(np.float32)

    # 2. La Plas matrix characteristic vector
    print("Calculating the Lapras feature vector...")
    laplacian_features = compute_laplacian_eigenvectors(
        adj_matrix_sym, n_components=min(16, embedding_dim//2, n_nodes-1)
    )

    for i in range(laplacian_features.shape[1]):
        feature_dict[f'laplacian_dim{i}'] = laplacian_features[:, i].astype(np.float32)

    # 3. Embedding based on nodal features
    print("Calculate node feature embedded...")
    feature_embedding = compute_feature_based_embedding(
        node_features, embedding_dim=min(32, node_features.shape[1])
    )

    for i in range(feature_embedding.shape[1]):
        feature_dict[f'feature_embedding_dim{i}'] = feature_embedding[:, i].astype(np.float32)

    # 4. Degree structure embedded
    print("Calculator Structure Embedded...")
    degree_embedding = compute_degree_embedding(
        adj_matrix_sym, embedding_dim=min(16, embedding_dim//4)
    )

    for i in range(degree_embedding.shape[1]):
        feature_dict[f'degree_embedding_dim{i}'] = degree_embedding[:, i].astype(np.float32)

    # 5. Neighbor convergence embedded
    print("Calculating Neighbors Convergence Embedded...")
    neighbor_embedding = compute_neighbor_embedding(
        adj_matrix_sym, node_features, embedding_dim=min(20, embedding_dim//3)
    )

    for i in range(neighbor_embedding.shape[1]):
        feature_dict[f'neighbor_embedding_dim{i}'] = neighbor_embedding[:, i].astype(np.float32)

    # 6. Neural network style embedded
    print("Compute GNN style embedded...")
    gnn_embedding = compute_gnn_style_embedding(
        adj_matrix_sym, node_features, embedding_dim=min(24, embedding_dim//2)
    )

    for i in range(gnn_embedding.shape[1]):
        feature_dict[f'gnn_embedding_dim{i}'] = gnn_embedding[:, i].astype(np.float32)

    # 7. Random migration of statistical features
    print("Calculating Random Swim Statistics...")
    rw_stats = compute_random_walk_stats_sampled(
        adj_matrix_sym, n_walks=20, walk_len=10, sample_size=min(3000, n_nodes)
    )

    for stat_name, stat_values in rw_stats.items():
        feature_dict[stat_name] = stat_values.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

    print(f"Figure embedded feature dimensions: {features_df.shape}")
    return features_df

def compute_adjacency_svd(adj_matrix: sp.csr_matrix, n_components: int = 32) -> np.ndarray:
    """Calculating the SVD Declination for the adjacent matrix"""
    try:
        # Use RuncaedSVD for fast-downs
        svd = TruncatedSVD(
            n_components=min(n_components, adj_matrix.shape[0]-1),
            random_state=42,
            n_iter=5  # Reduction of the number of iterations accelerating
        )
        svd_features = svd.fit_transform(adj_matrix)
        return svd_features.astype(np.float32)
    except Exception as e:
        print(f"SVDCalculator Failed: {e}")
        # Return Zero Matrix
        n_nodes = adj_matrix.shape[0]
        return np.zeros((n_nodes, n_components), dtype=np.float32)

def compute_laplacian_eigenvectors(adj_matrix: sp.csr_matrix, n_components: int = 16) -> np.ndarray:
    """Calculating the Lapras feature vector"""
    n_nodes = adj_matrix.shape[0]

    if n_nodes > 50000:
        # For mega-graphs, use fast approximation
        print(f"It's too big. ({n_nodes}node)，Use fast approximation...")
        return compute_fast_laplacian_approx(adj_matrix, n_components)

    try:
        from scipy.sparse.linalg import eigsh

        # Calculating the normalized Lapras Matrix
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        degree_sqrt = np.sqrt(degree)
        degree_sqrt[degree_sqrt == 0] = 1

        D_sqrt_inv = sp.diags(1.0 / degree_sqrt)
        L = sp.eye(n_nodes) - D_sqrt_inv @ adj_matrix @ D_sqrt_inv

        # Calculate feature vector
        eigenvalues, eigenvectors = eigsh(
            L,
            k=min(n_components+1, n_nodes-1),
            which='SM',
            maxiter=100
        )

        # Exclude the characteristic vector for the first zero feature value
        if eigenvectors.shape[1] > 1:
            return eigenvectors[:, 1:min(n_components+1, eigenvectors.shape[1])].astype(np.float32)
        else:
            return compute_fast_laplacian_approx(adj_matrix, n_components)

    except Exception as e:
        print(f"Lapras character vector calculation failed: {e}")
        return compute_fast_laplacian_approx(adj_matrix, n_components)

def compute_fast_laplacian_approx(adj_matrix: sp.csr_matrix, n_components: int = 16) -> np.ndarray:
    """Quick approximation of La Plas characterization vector"""
    n_nodes = adj_matrix.shape[0]

    # Calculator Features
    degree = np.array(adj_matrix.sum(axis=1)).flatten()

    # Generate Feature Matrix
    features = np.zeros((n_nodes, n_components), dtype=np.float32)

    # Filling-based features
    if degree.max() > 0:
        # Normality
        features[:, 0] = degree / degree.max()

        # logarithm of degrees
        if n_components > 1:
            features[:, 1] = np.log1p(degree) / np.log1p(degree.max())

        # square root of degrees
        if n_components > 2:
            features[:, 2] = np.sqrt(degree) / np.sqrt(degree.max())

        # PageRank approximation
        if n_components > 3:
            features[:, 3] = degree / (degree.sum() + 1e-8)

        # Second-class neighbor.
        if n_components > 4:
            try:
                adj_squared = adj_matrix.dot(adj_matrix)
                second_neighbors = np.array(adj_squared.sum(axis=1)).flatten()
                if second_neighbors.max() > 0:
                    features[:, 4] = second_neighbors / second_neighbors.max()
            except:
                pass

    # Other dimensions: projecting using trigonometric functions
    for i in range(5, n_components):
        if i % 2 == 0:
            features[:, i] = np.sin(degree * (i / n_components)) * 0.5 + 0.5
        else:
            features[:, i] = np.cos(degree * (i / n_components)) * 0.5 + 0.5

    return features

def compute_feature_based_embedding(node_features: np.ndarray, embedding_dim: int = 32) -> np.ndarray:
    """Dock based on node features"""
    n_nodes, n_original_features = node_features.shape

    if n_original_features >= embedding_dim:
        # If original feature dimension is greater than or equal to target dimension, use PA dimension down
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(embedding_dim, n_nodes-1))
        embedding = pca.fit_transform(node_features)
    else:
        # Use and fill the original feature directly if it has a dimension smaller than the target dimension
        embedding = node_features.copy()

        # If you need to fill
        if n_original_features < embedding_dim:
            # Use random projection extension dimensions
            random_proj = np.random.randn(n_original_features, embedding_dim - n_original_features).astype(np.float32)
            random_features = node_features @ random_proj
            embedding = np.hstack([embedding, random_features])

    # Normalization
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

    return embedding.astype(np.float32)

def compute_degree_embedding(adj_matrix: sp.csr_matrix, embedding_dim: int = 16) -> np.ndarray:
    """Level based embedded"""
    n_nodes = adj_matrix.shape[0]

    # Calculator
    degree = np.array(adj_matrix.sum(axis=1)).flatten()

    # Collection-related features
    degree_features = []

    # 1. Original degrees
    if degree.max() > 0:
        degree_features.append(degree / degree.max())

    # 2. logarithmic degrees
    degree_features.append(np.log1p(degree) / (np.log1p(degree.max()) + 1))

    # 3. Square root degree
    degree_features.append(np.sqrt(degree) / (np.sqrt(degree.max()) + 1))

    # 4. Normalization (division by nodes)
    degree_features.append(degree / n_nodes)

    # 5. Degree ranking
    degree_features.append(np.argsort(np.argsort(degree)) / n_nodes)

    # 6. Cumulative distribution of degrees
    sorted_degree = np.sort(degree)
    degree_cdf = np.searchsorted(sorted_degree, degree) / n_nodes
    degree_features.append(degree_cdf)

    # Convert Group
    feature_matrix = np.column_stack(degree_features)

    # If not enough dimensions, use a random projection extension
    if feature_matrix.shape[1] < embedding_dim:
        # Create Random Projection Matrix
        random_proj = np.random.randn(feature_matrix.shape[1], embedding_dim).astype(np.float32)
        embedding = feature_matrix @ random_proj
    else:
        embedding = feature_matrix[:, :embedding_dim]

    # Normalization
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

    return embedding.astype(np.float32)

def compute_neighbor_embedding(adj_matrix: sp.csr_matrix,
                             node_features: np.ndarray,
                             embedding_dim: int = 20) -> np.ndarray:
    """Compute Neighbors Convergence Dock"""
    n_nodes, n_features = node_features.shape
    adj_matrix_csc = adj_matrix.tocsc()

    # Initialise Embedded Matrix
    embedding = np.zeros((n_nodes, embedding_dim), dtype=np.float32)

    # Sample calculations (for large maps)
    sample_size = min(5000, n_nodes)
    sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)

    for idx, node in enumerate(tqdm(sampled_nodes, desc="Neighbor embedding", leave=False)):
        neighbors = adj_matrix_csc[:, node].indices

        if len(neighbors) == 0:
            # If there's no neighbor, use the node's own features.
            neighbor_features = node_features[node:node+1]
        else:
            # Limiting the number of neighbours
            max_neighbors = min(50, len(neighbors))
            if len(neighbors) > max_neighbors:
                selected_neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
            else:
                selected_neighbors = neighbors

            neighbor_features = node_features[selected_neighbors]

        # Aggregation of neighbor features: average value pool
        if len(neighbor_features) > 0:
            neighbor_mean = np.mean(neighbor_features, axis=0)

            # Create embedded: combination of nodal and neighbourhood features
            combined = np.hstack([
                node_features[node],
                neighbor_mean,
                node_features[node] - neighbor_mean  # Variance features
            ])

            # Use a random projection if the character dimension is greater than the embedded dimension
            if len(combined) > embedding_dim:
                # Consistency with fixed random projection
                if not hasattr(compute_neighbor_embedding, 'proj_matrix'):
                    compute_neighbor_embedding.proj_matrix = np.random.randn(
                        len(combined), embedding_dim
                    ).astype(np.float32)

                embedding[node] = combined @ compute_neighbor_embedding.proj_matrix
            else:
                # Fill zero
                embedding[node, :len(combined)] = combined

    # Fill unsampled nodes
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)

    if len(sampled_nodes) > 0:
        # Fill with average of sample nodes
        mean_embedding = np.mean(embedding[sampled_nodes], axis=0)
        embedding[unsampled] = mean_embedding

    # Normalization
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

    return embedding.astype(np.float32)

def compute_gnn_style_embedding(adj_matrix: sp.csr_matrix,
                              node_features: np.ndarray,
                              embedding_dim: int = 24) -> np.ndarray:
    """Calculate GNN style embedded (simulation 1 floor GCN)"""
    n_nodes, n_features = node_features.shape

    # Normalized Neighbourhood Matrix (GCN style)
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_sqrt = np.sqrt(degree)
    degree_sqrt[degree_sqrt == 0] = 1

    D_sqrt_inv = sp.diags(1.0 / degree_sqrt)
    norm_adj = D_sqrt_inv @ adj_matrix @ D_sqrt_inv

    # Simulation of GCN dissemination: A'X
    propagated_features = norm_adj @ node_features

    # Combining original and dissemination features
    combined_features = np.hstack([
        node_features,
        propagated_features,
        node_features * propagated_features,  # Interactive feature
        np.abs(node_features - propagated_features)  # Variance features
    ])

    # Decrease to Target Dimension
    if combined_features.shape[1] > embedding_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(embedding_dim, n_nodes-1))
        embedding = pca.fit_transform(combined_features)
    else:
        embedding = combined_features
        # If you need to fill
        if embedding.shape[1] < embedding_dim:
            padding = np.random.randn(n_nodes, embedding_dim - embedding.shape[1]).astype(np.float32) * 0.1
            embedding = np.hstack([embedding, padding])

    return embedding.astype(np.float32)

def compute_random_walk_stats_sampled(adj_matrix: sp.csr_matrix,
                                     n_walks: int = 20,
                                     walk_len: int = 10,
                                     sample_size: int = 3000) -> Dict[str, np.ndarray]:
    """Sampling calculates randomly moving statistical features"""
    n_nodes = adj_matrix.shape[0]

    # Sample Nodes
    sampled_nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)

    # Initialization Results
    return_probs = np.zeros(n_nodes, dtype=np.float32)
    avg_return_steps = np.zeros(n_nodes, dtype=np.float32)
    visited_nodes_counts = np.zeros(n_nodes, dtype=np.float32)

    adj_matrix_csc = adj_matrix.tocsc()

    print(f"Yeah. {len(sampled_nodes)} A random swim at each node....")

    for idx, start_node in enumerate(tqdm(sampled_nodes, desc="Swim in randomly.", leave=False)):
        return_count = 0
        total_return_steps = 0
        visited_nodes = set()

        for walk in range(n_walks):
            current_node = start_node
            steps = 0

            for step in range(walk_len):
                # Get the neighbors.
                neighbors = adj_matrix_csc[:, current_node].indices

                if len(neighbors) == 0:
                    break

                # Randomly select the next node
                next_node = np.random.choice(neighbors)
                steps += 1

                # Record visited nodes
                visited_nodes.add(next_node)

                # Check to return to start point
                if next_node == start_node:
                    return_count += 1
                    total_return_steps += steps
                    break

                current_node = next_node

        # Calculate statistics
        if n_walks > 0:
            return_probs[start_node] = return_count / n_walks

            if return_count > 0:
                avg_return_steps[start_node] = total_return_steps / return_count

            visited_nodes_counts[start_node] = len(visited_nodes) / walk_len

    # Fill unsampled nodes
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
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedded dimensions')
    args = parser.parse_args()

    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']

    print(f"Number of nodes: {x.shape[0]}, Feature shape: {x.shape[1]}")

    # Calculating image embedded features
    features_df = compute_embedding_features(
        edge_index, x,
        embedding_dim=args.embedding_dim
    )

    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)

    print(f"Figure embedded feature processing completed，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
