import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict

def compute_neighbor_features(edge_index: np.ndarray,
                            x: np.ndarray,
                            max_neighbors_1hop: int = 50,
                            max_neighbors_2hop: int = 500,
                            output_path: str = None) -> pd.DataFrame:
    """Calculating Neighbour Convergence:
1. 1-hop Neighbour characterization aggregation
2. 2-hop neighbourhood characterization
3. Neighbourhood features statistics"""
    
    print("Calculating Neighbor Convergence...")
    
    n_nodes = x.shape[0]
    n_features = x.shape[1]
    
    # Create adjacent matrix (directed)
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    
    # Symmetrical for non-neighbor calculation
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    
    feature_dict = {}
    
    # 1. 1-hop Neighbour characterization aggregation
    print("Calculating 1-hop-neighbor profile...")
    neighbor_1hop_features = aggregate_neighbor_features(
        adj_matrix_sym, x, max_neighbors=max_neighbors_1hop
    )
    
    for i in range(n_features):
        for agg_type, values in neighbor_1hop_features[i].items():
            feature_dict[f'neighbor1_{agg_type}_feat{i}'] = values.astype(np.float32)
    
    # 2. Calculation of 2-hop neighbourhood features (through adjacent matrix squares)
    print("Calculating 2-hop neighbor...")
    # Limit size of calculation
    if n_nodes > 100000:
        print("Too many nodes, using sampling to calculate 2-hop features...")
        neighbor_2hop_features = compute_2hop_features_sampled(
            adj_matrix_sym, x, max_neighbors_2hop, sample_size=5000
        )
    else:
        adj_squared = adj_matrix_sym.dot(adj_matrix_sym)
        neighbor_2hop_features = aggregate_neighbor_features(
            adj_squared, x, max_neighbors=max_neighbors_2hop, prefix='neighbor2'
        )
    
    for i in range(min(5, n_features)):  # Only the top five features were taken for 2-hop aggregation
        for agg_type, values in neighbor_2hop_features[i].items():
            feature_dict[f'neighbor2_{agg_type}_feat{i}'] = values.astype(np.float32)
    
    # 3. Relevance of neighbourhood to identity
    print("Calculating Neighbourly Relevance Features...")
    degree_corr_features = compute_degree_correlation_features(
        adj_matrix_sym, x
    )
    
    for feat_name, values in degree_corr_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 4. Statistics on the distribution of neighbourhood features
    print("Calculating Neighbourly Feature Distribution...")
    neighbor_dist_stats = compute_neighbor_distribution_stats(
        adj_matrix_sym, x, max_neighbors_1hop
    )
    
    for stat_name, stat_values in neighbor_dist_stats.items():
        feature_dict[stat_name] = stat_values.astype(np.float32)
    
    # 5. Disparities in access to neighbours
    print("Calculating the difference in access...")
    in_out_diff_features = compute_in_out_difference_features(
        adj_matrix, x, max_neighbors_1hop
    )
    
    for feat_name, values in in_out_diff_features.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    if output_path:
        with open(output_path, 'wb') as f:
            import pickle
            pickle.dump(features_df, f)
    
    return features_df

def aggregate_neighbor_features(adj_matrix: sp.csr_matrix,
                              node_features: np.ndarray,
                              max_neighbors: int = 100,
                              prefix: str = 'neighbor') -> Dict[int, Dict[str, np.ndarray]]:
    """Combining Neighbours"""
    n_nodes, n_features = node_features.shape
    adj_matrix = adj_matrix.tocsc()
    
    # Initialise result dictionary
    results = {i: {} for i in range(n_features)}
    
    # Pre-allocate groups for each feature
    for feat_idx in range(n_features):
        results[feat_idx]['mean'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['max'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['min'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['std'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['sum'] = np.zeros(n_nodes, dtype=np.float32)
    
    # Node-by-Node
    for node in tqdm(range(n_nodes), desc=f"Aggregation{prefix}features"):
        neighbors = adj_matrix[:, node].indices
        
        # Limiting the number of neighbours
        if len(neighbors) > max_neighbors:
            neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
        
        if len(neighbors) == 0:
            continue
        
        neighbor_feats = node_features[neighbors]
        
        for feat_idx in range(n_features):
            feat_values = neighbor_feats[:, feat_idx]
            
            results[feat_idx]['mean'][node] = feat_values.mean()
            results[feat_idx]['max'][node] = feat_values.max()
            results[feat_idx]['min'][node] = feat_values.min()
            results[feat_idx]['std'][node] = feat_values.std() if len(neighbors) > 1 else 0
            results[feat_idx]['sum'][node] = feat_values.sum()
    
    return results

def compute_2hop_features_sampled(adj_matrix: sp.csr_matrix,
                                 node_features: np.ndarray,
                                 max_neighbors: int = 500,
                                 sample_size: int = 5000) -> Dict[int, Dict[str, np.ndarray]]:
    """Sample calculation of 2-hop neighbor features"""
    n_nodes = adj_matrix.shape[0]
    
    # Random sample nodes
    sampled_nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    
    # Initialization Results
    results = {i: {'mean': np.zeros(n_nodes, dtype=np.float32)} 
              for i in range(min(5, node_features.shape[1]))}
    
    adj_matrix = adj_matrix.tocsc()
    
    for node in tqdm(sampled_nodes, desc="Sampling to calculate 2-hop features"):
        # Get 1-hop neighbor
        neighbors_1hop = adj_matrix[:, node].indices
        
        if len(neighbors_1hop) == 0:
            continue
        
        # Limiting the number of neighbours
        if len(neighbors_1hop) > 50:
            neighbors_1hop = np.random.choice(neighbors_1hop, 50, replace=False)
        
        # Collection of 2-hop neighbors
        neighbors_2hop = set()
        for neighbor in neighbors_1hop:
            second_neighbors = adj_matrix[:, neighbor].indices
            neighbors_2hop.update(second_neighbors)
        
        # Remove yourself and one-hop neighbors.
        neighbors_2hop.discard(node)
        neighbors_2hop.difference_update(neighbors_1hop)
        
        neighbors_2hop = np.array(list(neighbors_2hop))
        
        # Limiting the number of 2-hop neighbours
        if len(neighbors_2hop) > max_neighbors:
            neighbors_2hop = np.random.choice(neighbors_2hop, max_neighbors, replace=False)
        
        if len(neighbors_2hop) == 0:
            continue
        
        neighbor_feats = node_features[neighbors_2hop]
        
        for feat_idx in results.keys():
            feat_values = neighbor_feats[:, feat_idx]
            results[feat_idx]['mean'][node] = feat_values.mean()
    
    # Fill unsampled nodes
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    for feat_idx in results.keys():
        sampled_mean = results[feat_idx]['mean'][sampled_nodes].mean()
        results[feat_idx]['mean'][unsampled] = sampled_mean
    
    return results

def compute_degree_correlation_features(adj_matrix: sp.csr_matrix,
                                      node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculating Neighborhood Relation to Characterism"""
    n_nodes = adj_matrix.shape[0]
    
    # Calculator
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Initialization Results
    results = {}
    n_features = node_features.shape[1]
    
    adj_matrix = adj_matrix.tocsc()
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="Calculate Relevance"):
        feature_values = node_features[:, feat_idx]
        
        # Calculate the neighbourhood feature average
        neighbor_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        neighbor_degree_mean = np.zeros(n_nodes, dtype=np.float32)
        
        for node in range(n_nodes):
            neighbors = adj_matrix[:, node].indices
            
            if len(neighbors) == 0:
                neighbor_feat_mean[node] = feature_values[node]
                neighbor_degree_mean[node] = degree[node]
                continue
            
            neighbor_feat_mean[node] = feature_values[neighbors].mean()
            neighbor_degree_mean[node] = degree[neighbors].mean()
        
        # Calculating Relevance Features
        results[f'feat{feat_idx}_degree_corr_coef'] = neighbor_feat_mean * neighbor_degree_mean
        results[f'feat{feat_idx}_degree_diff'] = feature_values - neighbor_degree_mean
        results[f'feat{feat_idx}_feat_degree_ratio'] = np.divide(
            feature_values, neighbor_degree_mean + 1e-8
        )
    
    return results

def compute_neighbor_distribution_stats(adj_matrix: sp.csr_matrix,
                                      node_features: np.ndarray,
                                      max_neighbors: int = 50) -> Dict[str, np.ndarray]:
    """Computation of neighbourhood feature distribution statistics"""
    n_nodes, n_features = node_features.shape
    
    # Initialization Results
    results = {}
    adj_matrix = adj_matrix.tocsc()
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="Computation of neighbourhood distribution statistics"):
        feature_values = node_features[:, feat_idx]
        
        # Initialization of statistics
        skewness = np.zeros(n_nodes, dtype=np.float32)
        kurtosis = np.zeros(n_nodes, dtype=np.float32)
        iqr = np.zeros(n_nodes, dtype=np.float32)  # Quadration
        
        for node in range(n_nodes):
            neighbors = adj_matrix[:, node].indices
            
            # Limiting the number of neighbours
            if len(neighbors) > max_neighbors:
                neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
            
            if len(neighbors) < 3:
                skewness[node] = 0
                kurtosis[node] = 0
                iqr[node] = 0
                continue
            
            neighbor_feats = feature_values[neighbors]
            
            # Calculating deviations
            mean = neighbor_feats.mean()
            std = neighbor_feats.std()
            if std > 0:
                skewness[node] = ((neighbor_feats - mean) ** 3).mean() / (std ** 3)
            
            # Calculating Peak
            if std > 0:
                kurtosis[node] = ((neighbor_feats - mean) ** 4).mean() / (std ** 4) - 3
            
            # Calculate a quartile distance
            q75, q25 = np.percentile(neighbor_feats, [75, 25])
            iqr[node] = q75 - q25
        
        results[f'feat{feat_idx}_neighbor_skew'] = skewness
        results[f'feat{feat_idx}_neighbor_kurt'] = kurtosis
        results[f'feat{feat_idx}_neighbor_iqr'] = iqr
    
    return results

def compute_in_out_difference_features(adj_matrix: sp.csr_matrix,
                                     node_features: np.ndarray,
                                     max_neighbors: int = 50) -> Dict[str, np.ndarray]:
    """Calculate differences in access to and from neighbours"""
    n_nodes, n_features = node_features.shape
    
    # The switch is reversed.
    adj_matrix_t = adj_matrix.T
    
    results = {}
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="Calculate access to and from neighbours"):
        feature_values = node_features[:, feat_idx]
        
        in_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        out_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        
        # Calculate an average of neighbourhood features
        adj_matrix_t_csc = adj_matrix_t.tocsc()
        for node in range(n_nodes):
            in_neighbors = adj_matrix_t_csc[:, node].indices
            
            if len(in_neighbors) > max_neighbors:
                in_neighbors = np.random.choice(in_neighbors, max_neighbors, replace=False)
            
            if len(in_neighbors) > 0:
                in_feat_mean[node] = feature_values[in_neighbors].mean()
            else:
                in_feat_mean[node] = feature_values[node]
        
        # Calculate an average of neighbourhood features
        adj_matrix_csc = adj_matrix.tocsc()
        for node in range(n_nodes):
            out_neighbors = adj_matrix_csc[:, node].indices
            
            if len(out_neighbors) > max_neighbors:
                out_neighbors = np.random.choice(out_neighbors, max_neighbors, replace=False)
            
            if len(out_neighbors) > 0:
                out_feat_mean[node] = feature_values[out_neighbors].mean()
            else:
                out_feat_mean[node] = feature_values[node]
        
        # Calculation of variance features
        results[f'feat{feat_idx}_in_out_diff'] = in_feat_mean - out_feat_mean
        results[f'feat{feat_idx}_in_out_ratio'] = np.divide(
            in_feat_mean, out_feat_mean + 1e-8
        )
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    parser.add_argument('--max_neighbors_1hop', type=int, default=50, help='Maximum number of neighbours 1-hop')
    parser.add_argument('--max_neighbors_2hop', type=int, default=500, help='2-hop maximum number of neighbours')
    args = parser.parse_args()
    
    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    edge_index = data['edge_index']
    x = data['x']
    
    # Calculating Neighbour Features
    features_df = compute_neighbor_features(
        edge_index, x,
        args.max_neighbors_1hop,
        args.max_neighbors_2hop
    )
    
    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"Neighborhood profile is complete.，Feature shape: {features_df.shape}")
    print(f"Organisation: {args.output_path}")

if __name__ == "__main__":
    main()
