"""Cluster feature module: features based on nodal grouping"""
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
from typing import Dict, List, Tuple  # Add this line
def compute_clustering_features(edge_index: np.ndarray,
                               node_features: np.ndarray,
                               n_clusters_list: list = [5, 10, 20],
                               use_sampling: bool = True,
                               sample_size: int = 10000,
                               output_path: str = None) -> pd.DataFrame:
    """Calculating cluster features:
1. K-means cluster labels
2. Cluster centre distance
3. Cluster statistics
4. Cluster relations
5. Outline factor
6. Neighbourhood cluster distribution"""

    print("Calculating Cluster Features...")
    n_nodes, n_features = node_features.shape

    feature_dict = {}

    # Standardized features
    print("Standardised Node Features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(node_features)

    # 1. K-means cluster for nodal features
    print("Organisation")

    for n_clusters in n_clusters_list:
        if n_clusters >= n_nodes:
            print(f"Clusters {n_clusters} Greater than Nodes，Skip...")
            continue

        print(f"  Clusters: {n_clusters}")

        # Use MiniBatchKMeans to accelerate
        if n_nodes > 50000 and use_sampling:
            print(f"  Too many nodes ({n_nodes})，Use sample grouping...")

            # Sampling of selected nodes for group training
            if sample_size < n_nodes:
                sample_indices = np.random.choice(n_nodes, sample_size, replace=False)
                sample_features = features_scaled[sample_indices]
            else:
                sample_indices = np.arange(n_nodes)
                sample_features = features_scaled

            # Training in cluster models
            kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                     random_state=42,
                                     batch_size=1000,
                                     n_init=3)
            kmeans.fit(sample_features)

            # Forecast cluster labels for all nodes
            cluster_labels = kmeans.predict(features_scaled)
            cluster_centers = kmeans.cluster_centers_

        else:
            # Use full K-mes for medium scale maps
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

        # Save Cluster Tabs
        feature_dict[f'kmeans_{n_clusters}_cluster'] = cluster_labels.astype(np.float32)

        # Calculates the distance to a cluster centre
        print(f"    Calculating Cluster Distance...")
        distances = np.zeros((n_nodes, n_clusters), dtype=np.float32)

        # Batch calculation distance to save memory
        batch_size = 10000
        for i in tqdm(range(0, n_nodes, batch_size), desc="Calculate Distance", leave=False):
            end_idx = min(i + batch_size, n_nodes)
            batch_features = features_scaled[i:end_idx]

            # Calculate the distance to all polycentres
            for j in range(n_clusters):
                dist = np.linalg.norm(batch_features - cluster_centers[j], axis=1)
                distances[i:end_idx, j] = dist

        # Recent cluster distance
        min_distances = np.min(distances, axis=1)
        feature_dict[f'kmeans_{n_clusters}_min_dist'] = min_distances.astype(np.float32)

        # Recent Cluster Index
        nearest_clusters = np.argmin(distances, axis=1)
        feature_dict[f'kmeans_{n_clusters}_nearest_cluster'] = nearest_clusters.astype(np.float32)

        # Second near concentration range
        for i in range(n_nodes):
            distances[i, nearest_clusters[i]] = np.inf  # Set the nearest distance as infinite.
        second_min_distances = np.min(distances, axis=1)
        feature_dict[f'kmeans_{n_clusters}_second_min_dist'] = second_min_distances.astype(np.float32)

        # Statistics of nodes within a cluster
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
        feature_dict[f'kmeans_{n_clusters}_cluster_size'] = cluster_sizes[cluster_labels].astype(np.float32)
        feature_dict[f'kmeans_{n_clusters}_cluster_size_norm'] = (cluster_sizes[cluster_labels] / n_nodes).astype(np.float32)

        # Cluster density (average distance)
        print(f"    Calculate a concentration density...")
        cluster_density = np.zeros(n_nodes, dtype=np.float32)
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 1:
                cluster_points = features_scaled[cluster_mask]
                # Calculate average distance
                if len(cluster_points) <= 1000:
                    # Accurate distance for small groups
                    from scipy.spatial.distance import pdist
                    pairwise_dist = pdist(cluster_points)
                    avg_dist = np.mean(pairwise_dist)
                else:
                    # Calculation of large group sampling
                    sample_idx = np.random.choice(np.sum(cluster_mask), min(1000, np.sum(cluster_mask)), replace=False)
                    sample_points = cluster_points[sample_idx]
                    from scipy.spatial.distance import cdist
                    pairwise_dist = cdist(sample_points, sample_points)
                    avg_dist = np.mean(pairwise_dist[np.triu_indices(len(sample_points), k=1)])

                cluster_density[cluster_mask] = avg_dist

        feature_dict[f'kmeans_{n_clusters}_cluster_density'] = cluster_density.astype(np.float32)

        # Calculating contours factor (sampling calculation)
        if n_nodes <= 10000:  # Calculating contour coefficients for small maps only
            try:
                print(f"    Calculate contours factor...")
                silhouette_vals = silhouette_samples(features_scaled, cluster_labels)
                feature_dict[f'kmeans_{n_clusters}_silhouette'] = silhouette_vals.astype(np.float32)
            except:
                print(f"    Failed to calculate contour coefficient，Skip...")
                feature_dict[f'kmeans_{n_clusters}_silhouette'] = np.zeros(n_nodes, dtype=np.float32)
        else:
            feature_dict[f'kmeans_{n_clusters}_silhouette'] = np.zeros(n_nodes, dtype=np.float32)

    # 2. Creation of an adjacent matrix
    print("Create an adjacent matrix...")
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)),
                              shape=(n_nodes, n_nodes))
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)

    # 3. Neighborhood cluster consistency (calculated only for a small number of clusters)
    print("Calculating Neighbor Group Consistency...")
    for n_clusters in [5, 10]:  # Only a small group of neighbors are counted.
        cluster_key = f'kmeans_{n_clusters}_cluster'
        if cluster_key in feature_dict:
            print(f"  Calculate{n_clusters}Consistency of clustered neighbours...")
            cluster_labels = feature_dict[cluster_key].astype(int)
            neighbor_cluster_features = compute_neighbor_cluster_features(
                adj_matrix_sym, cluster_labels
            )

            for feat_name, values in neighbor_cluster_features.items():
                feature_dict[f'{cluster_key}_{feat_name}'] = values.astype(np.float32)

    # 4. Cluster connectivity features
    print("Compute inter-group connectivity features...")
    if 'kmeans_5_cluster' in feature_dict:
        cluster_labels = feature_dict['kmeans_5_cluster'].astype(int)
        inter_cluster_features = compute_inter_cluster_features(
            adj_matrix_sym, cluster_labels
        )

        for feat_name, values in inter_cluster_features.items():
            feature_dict[f'inter_cluster_{feat_name}'] = values.astype(np.float32)

    # 5. Cluster features based on degrees
    print("Calculating concentration features based on degrees...")
    degree_based_features = compute_degree_based_cluster_features(
        adj_matrix_sym, features_scaled
    )

    for feat_name, values in degree_based_features.items():
        feature_dict[f'degree_cluster_{feat_name}'] = values.astype(np.float32)

    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)

    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

    print(f"Cluster characterization dimensions: {features_df.shape}")
    return features_df

def compute_neighbor_cluster_features(adj_matrix: sp.csr_matrix,
                                     cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculating Neighbour Cluster Features"""
    n_nodes = adj_matrix.shape[0]
    n_clusters = len(np.unique(cluster_labels))
    adj_matrix_csc = adj_matrix.tocsc()

    # Initialization feature
    consistency = np.zeros(n_nodes, dtype=np.float32)  # Neighbor Group Consistency
    entropy = np.zeros(n_nodes, dtype=np.float32)  # Neighbor cluster distribution entropy
    dominant_cluster_ratio = np.zeros(n_nodes, dtype=np.float32)  # Lead cluster ratio

    for node in tqdm(range(n_nodes), desc="Neighbor cluster features", leave=False):
        neighbors = adj_matrix_csc[:, node].indices

        if len(neighbors) == 0:
            consistency[node] = 0
            entropy[node] = 0
            dominant_cluster_ratio[node] = 0
            continue

        # Statistical neighbourhood cluster distribution
        neighbor_clusters = cluster_labels[neighbors]
        node_cluster = cluster_labels[node]

        # Cluster consistency
        same_cluster_count = np.sum(neighbor_clusters == node_cluster)
        consistency[node] = same_cluster_count / len(neighbors)

        # 2. Cluster distribution of entropy
        cluster_counts = np.bincount(neighbor_clusters, minlength=n_clusters)
        cluster_probs = cluster_counts / len(neighbors)

        # Calculating entropy (avoiding log(0))
        non_zero_probs = cluster_probs[cluster_probs > 0]
        if len(non_zero_probs) > 0:
            entropy[node] = -np.sum(non_zero_probs * np.log(non_zero_probs)) / np.log(n_clusters)

        # 3. Lead cluster ratio
        if len(cluster_counts) > 0:
            dominant_cluster_ratio[node] = np.max(cluster_counts) / len(neighbors)

    return {
        'neighbor_consistency': consistency,
        'neighbor_cluster_entropy': entropy,
        'neighbor_dominant_ratio': dominant_cluster_ratio
    }

def compute_inter_cluster_features(adj_matrix: sp.csr_matrix,
                                 cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute inter-group connectivity features"""
    n_nodes = adj_matrix.shape[0]
    n_clusters = len(np.unique(cluster_labels))

    adj_matrix_csc = adj_matrix.tocsc()

    # Initialization feature
    intra_cluster_edges = np.zeros(n_nodes, dtype=np.float32)
    inter_cluster_edges = np.zeros(n_nodes, dtype=np.float32)
    cluster_degree = np.zeros(n_nodes, dtype=np.float32)
    cluster_bridging = np.zeros(n_nodes, dtype=np.float32)  # Bridge factor

    # Expected cluster size
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)

    for node in tqdm(range(n_nodes), desc="Cluster connections", leave=False):
        neighbors = adj_matrix_csc[:, node].indices

        if len(neighbors) == 0:
            continue

        node_cluster = cluster_labels[node]

        # Statistical connection type
        intra_count = 0
        neighbor_clusters = set()

        for neighbor in neighbors:
            neighbor_cluster = cluster_labels[neighbor]
            if neighbor_cluster == node_cluster:
                intra_count += 1
            neighbor_clusters.add(neighbor_cluster)

        intra_cluster_edges[node] = intra_count
        inter_cluster_edges[node] = len(neighbors) - intra_count

        # Concentration Internal Integration
        if cluster_sizes[node_cluster] > 1:
            cluster_degree[node] = intra_count / (cluster_sizes[node_cluster] - 1)

        # Bridging factor: number of different clusters connected
        cluster_bridging[node] = len(neighbor_clusters)

    # Calculate rate
    total_edges = intra_cluster_edges + inter_cluster_edges
    intra_ratio = np.divide(intra_cluster_edges, total_edges, where=total_edges>0)
    inter_ratio = np.divide(inter_cluster_edges, total_edges, where=total_edges>0)

    # Subsistence bridging factor
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
    """Group features based on degrees"""
    n_nodes = node_features.shape[0]

    # Calculator
    degree = np.array(adj_matrix.sum(axis=1)).flatten()

    # Simple grouping based on degrees
    degree_percentiles = np.percentile(degree, [25, 50, 75, 90])
    degree_clusters = np.zeros(n_nodes, dtype=int)

    for i, node in enumerate(range(n_nodes)):
        if degree[node] <= degree_percentiles[0]:
            degree_clusters[i] = 0  # Low Node
        elif degree[node] <= degree_percentiles[1]:
            degree_clusters[i] = 1  # Medium Low Node
        elif degree[node] <= degree_percentiles[2]:
            degree_clusters[i] = 2  # Medium height node
        elif degree[node] <= degree_percentiles[3]:
            degree_clusters[i] = 3  # Height Node
        else:
            degree_clusters[i] = 4  # Superhigh node

    # Initialization feature
    features = {}
    features['degree_cluster'] = degree_clusters.astype(np.float32)

    # Average features within clusters
    for cluster_id in range(5):
        cluster_mask = degree_clusters == cluster_id
        if np.sum(cluster_mask) > 0:
            # Calculates the average of the features of the inner nodes of the group
            cluster_features = node_features[cluster_mask]
            avg_features = np.mean(cluster_features, axis=0)

            # Calculating the distance from each node to the cluster centre
            for feat_idx in range(min(3, node_features.shape[1])):  # Only the first three features are calculated
                feat_name = f'degree_cluster_{cluster_id}_feat{feat_idx}_dist'
                distances = np.abs(node_features[:, feat_idx] - avg_features[feat_idx])
                features[feat_name] = distances.astype(np.float32)

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--n_clusters', type=str, default='5,10,20', help='Group Number List, Comma-separated')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size (used in large maps)')
    parser.add_argument('--no_sampling', action='store_true', help='Disable sampling (used in small maps)')
    args = parser.parse_args()

    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']

    print(f"Number of nodes: {x.shape[0]}, Feature shape: {x.shape[1]}")

    # Number of Parsing Clusters
    n_clusters_list = [int(n) for n in args.n_clusters.split(',')]

    # Calculate group features
    features_df = compute_clustering_features(
        edge_index, x,
        n_clusters_list,
        use_sampling=not args.no_sampling,
        sample_size=args.sample_size
    )

    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)

    print(f"Group characterization processing completed，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
