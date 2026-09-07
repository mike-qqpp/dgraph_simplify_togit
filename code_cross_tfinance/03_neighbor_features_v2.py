"""Combining Neighbour's Feature Cohesion Script
Convergence of neighbours dealing with the original continuum and the discrete compartment."""

import numpy as np
import pandas as pd
import argparse
import scipy.sparse as sp
from tqdm import tqdm
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def compute_combined_neighbor_features(
    edge_index: np.ndarray,
    x: np.ndarray,
    max_neighbors_1hop: int = 50,
    max_neighbors_2hop: int = 500,
    n_bins: int = 5,
    output_path: str = None
) -> pd.DataFrame:
    """Calculating a combination of neighbourhood aggregation features:
1. Consistency of the original continuum
2. Consistency of the 5-box feature

Parameters:
edge index: border index array (2, n edges)
x: Original characterization matrix (n nodes, n features)
max neigbors 1hop:1-hop maximum number of neighbors
max neigbors 2hop: 2-hop maximum number of neighbours
n bins: Number of boxes
output path: Output file path

Return:
DataFrame with all polymeric features"""
    print("=" * 60)
    print("Start calculating group neighbor features...")
    print("=" * 60)
    
    n_nodes = x.shape[0]
    n_features = x.shape[1]
    print(f"  Number of nodes: {n_nodes}, Number of original features: {n_features}")
    
    # Create an adjacent matrix
    row, col = edge_index[0], edge_index[1]
    adj_matrix = sp.csr_matrix((np.ones_like(row), (row, col)), 
                              shape=(n_nodes, n_nodes))
    
    # Symmetrical for non-neighbor calculation
    adj_matrix_sym = adj_matrix + adj_matrix.T
    adj_matrix_sym.data = np.ones_like(adj_matrix_sym.data)
    adj_matrix_csc = adj_matrix_sym.tocsc()
    
    feature_dict = {}
    
    # ============================================================
    # Part I: Convergence of neighbours with original continuum features
    # ============================================================
    print("\n" + "=" * 60)
    print("Part one: Convergence of original continuum features")
    print("=" * 60)
    
    # 1. 1-hop Neighbour characterization aggregation
    print("\\n Calculating 1-hop PHP characterization fusion...")
    neighbor_1hop_features = aggregate_neighbor_features(
        adj_matrix_csc, x, max_neighbors=max_neighbors_1hop
    )
    
    for i in range(n_features):
        for agg_type, values in neighbor_1hop_features[i].items():
            feature_dict[f'orig_n1_{agg_type}_feat{i}'] = values.astype(np.float32)
    
    # 2. 2-hop neighbourhood characterization
    print("Calculating 2-hop Neighbourhood Features...")
    if n_nodes > 100000:
        print("Too many nodes, using sampling...")
        neighbor_2hop_features = compute_2hop_features_sampled(
            adj_matrix_sym, x, max_neighbors_2hop, sample_size=5000
        )
    else:
        adj_squared = adj_matrix_sym.dot(adj_matrix_sym)
        neighbor_2hop_features = aggregate_neighbor_features(
            adj_squared, x, max_neighbors=max_neighbors_2hop
        )
    
    for i in range(min(5, n_features)):
        for agg_type, values in neighbor_2hop_features[i].items():
            feature_dict[f'orig_n2_{agg_type}_feat{i}'] = values.astype(np.float32)
    
    # 3. Relevance of neighbourhood to identity
    print("Calculating Neighbourly Relevance Features...")
    degree_corr = compute_degree_correlation_features(adj_matrix_csc, x)
    for feat_name, values in degree_corr.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # 4. Statistics on the distribution of neighbourhood features
    print("Calculating Neighbourly Feature Distribution...")
    neighbor_dist = compute_neighbor_distribution_stats(adj_matrix_csc, x, max_neighbors_1hop)
    for stat_name, stat_values in neighbor_dist.items():
        feature_dict[stat_name] = stat_values.astype(np.float32)
    
    # 5. Disparities in access to neighbours
    print("Calculating the difference in access...")
    in_out_diff = compute_in_out_difference_features(adj_matrix, x, max_neighbors_1hop)
    for feat_name, values in in_out_diff.items():
        feature_dict[feat_name] = values.astype(np.float32)
    
    # ============================================================
    # Part II: Convergence of neighbours with semi-box features
    # ============================================================
    print("\n" + "=" * 60)
    print("Part Two: Sub-box Qualitative Neighbourhood Convergence")
    print("=" * 60)
    
    # Five minutes for original features.
    print("Five minutes for original features...")
    from sklearn.preprocessing import KBinsDiscretizer
    kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    bin_features = kbin.fit_transform(x).astype(int)
    print(f"  Box character dimensions: {bin_features.shape}")
    print(f"  Box range: [{bin_features.min()}, {bin_features.max()}]")
    
    # 1. Basic neighbourhood aggregation of the semi-box features
    print("\\n Calculating the Basic Aggregation of Subbox Features...")
    for feat_idx in tqdm(range(n_features), desc="Subbox Basic Aggregation"):
        feat_values = bin_features[:, feat_idx]
        
        neighbor_mean = np.zeros(n_nodes, dtype=np.float32)
        neighbor_max = np.zeros(n_nodes, dtype=np.float32)
        neighbor_min = np.zeros(n_nodes, dtype=np.float32)
        neighbor_std = np.zeros(n_nodes, dtype=np.float32)
        neighbor_sum = np.zeros(n_nodes, dtype=np.float32)
        neighbor_count = np.zeros(n_nodes, dtype=np.float32)
        
        for node in range(n_nodes):
            neighbors = adj_matrix_csc[:, node].indices
            
            if len(neighbors) > max_neighbors_1hop:
                neighbors = np.random.choice(neighbors, max_neighbors_1hop, replace=False)
            
            if len(neighbors) == 0:
                continue
            
            neighbor_vals = feat_values[neighbors]
            neighbor_mean[node] = neighbor_vals.mean()
            neighbor_max[node] = neighbor_vals.max()
            neighbor_min[node] = neighbor_vals.min()
            neighbor_std[node] = neighbor_vals.std() if len(neighbors) > 1 else 0
            neighbor_sum[node] = neighbor_vals.sum()
            neighbor_count[node] = len(neighbors)
        
        feature_dict[f'bin_n1_mean_feat{feat_idx}'] = neighbor_mean
        feature_dict[f'bin_n1_max_feat{feat_idx}'] = neighbor_max
        feature_dict[f'bin_n1_min_feat{feat_idx}'] = neighbor_min
        feature_dict[f'bin_n1_std_feat{feat_idx}'] = neighbor_std
        feature_dict[f'bin_n1_sum_feat{feat_idx}'] = neighbor_sum
        feature_dict[f'bin_n1_count_feat{feat_idx}'] = neighbor_count
    
    # 2. Box distribution statistics (number and proportion)
    print("\\n Calculating Box Distribution Statistics ( count and proportion)...")
    for feat_idx in tqdm(range(min(5, n_features)), desc="Box Distribution"):
        feat_values = bin_features[:, feat_idx]
        
        # Share of each box in the neighborhood
        for box_id in range(n_bins):
            box_ratio = np.zeros(n_nodes, dtype=np.float32)
            
            for node in range(n_nodes):
                neighbors = adj_matrix_csc[:, node].indices
                if len(neighbors) == 0:
                    continue
                neighbor_vals = feat_values[neighbors]
                box_ratio[node] = np.mean(neighbor_vals == box_id)
            
            feature_dict[f'bin_n1_box{box_id}_ratio_feat{feat_idx}'] = box_ratio
        
        # The number of neighbors in the box.
        neighbor_mode = np.zeros(n_nodes, dtype=np.float32)
        for node in range(n_nodes):
            neighbors = adj_matrix_csc[:, node].indices
            if len(neighbors) == 0:
                continue
            neighbor_vals = feat_values[neighbors]
            neighbor_mode[node] = np.bincount(neighbor_vals.astype(int)).argmax()
        
        feature_dict[f'bin_n1_mode_feat{feat_idx}'] = neighbor_mode
        
        # Neighbor's compartment entropy (uncertain distribution)
        neighbor_entropy = np.zeros(n_nodes, dtype=np.float32)
        for node in range(n_nodes):
            neighbors = adj_matrix_csc[:, node].indices
            if len(neighbors) < 2:
                continue
            neighbor_vals = feat_values[neighbors]
            hist = np.bincount(neighbor_vals.astype(int), minlength=n_bins)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            neighbor_entropy[node] = -np.sum(hist * np.log(hist + 1e-10))
        
        feature_dict[f'bin_n1_entropy_feat{feat_idx}'] = neighbor_entropy
    
    # 3. Consistency of cross-box features
    print("Calculating Cross-Performance Neighbor Convergence...")
    neighbor_bin_mean = np.zeros((n_nodes, n_features), dtype=np.float32)
    neighbor_bin_std = np.zeros((n_nodes, n_features), dtype=np.float32)
    
    for node in tqdm(range(n_nodes), desc="Cross-Purpose Aggregation"):
        neighbors = adj_matrix_csc[:, node].indices
        
        if len(neighbors) > max_neighbors_1hop:
            neighbors = np.random.choice(neighbors, max_neighbors_1hop, replace=False)
        
        if len(neighbors) == 0:
            continue
        
        neighbor_feats = bin_features[neighbors]
        neighbor_bin_mean[node] = neighbor_feats.mean(axis=0)
        neighbor_bin_std[node] = neighbor_feats.std(axis=0)
    
    feature_dict['bin_all_mean'] = np.mean(neighbor_bin_mean, axis=1).astype(np.float32)
    feature_dict['bin_all_std'] = np.mean(neighbor_bin_std, axis=1).astype(np.float32)
    feature_dict['bin_all_max'] = np.max(neighbor_bin_mean, axis=1).astype(np.float32)
    feature_dict['bin_all_min'] = np.min(neighbor_bin_mean, axis=1).astype(np.float32)
    feature_dict['bin_all_range'] = (np.max(neighbor_bin_mean, axis=1) - np.min(neighbor_bin_mean, axis=1)).astype(np.float32)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    print("\n" + "=" * 60)
    print("Feature work complete!")
    print("=" * 60)
    print(f"  General characteristic dimension: {features_df.shape}")
    print(f"  Original characterization aggregates: {sum(1 for c in features_df.columns if c.startswith('orig_'))}")
    print(f"  Case-by-box feature aggregates: {sum(1 for c in features_df.columns if c.startswith('bin_'))}")
    
    if output_path:
        with open(output_path, 'wb') as f:
            pd.to_pickle(features_df, f)
        print(f"  Save To: {output_path}")
    
    return features_df


def aggregate_neighbor_features(adj_matrix: sp.csc_matrix,
                                node_features: np.ndarray,
                                max_neighbors: int = 100,
                                prefix: str = 'neighbor') -> Dict[int, Dict[str, np.ndarray]]:
    """Combining Neighbours"""
    n_nodes, n_features = node_features.shape
    
    results = {i: {} for i in range(n_features)}
    
    for feat_idx in range(n_features):
        results[feat_idx]['mean'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['max'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['min'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['std'] = np.zeros(n_nodes, dtype=np.float32)
        results[feat_idx]['sum'] = np.zeros(n_nodes, dtype=np.float32)
    
    for node in tqdm(range(n_nodes), desc=f"Aggregation{prefix}features"):
        neighbors = adj_matrix[:, node].indices
        
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
    sampled_nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    
    results = {i: {'mean': np.zeros(n_nodes, dtype=np.float32)} 
              for i in range(min(5, node_features.shape[1]))}
    
    adj_matrix = adj_matrix.tocsc()
    
    for node in tqdm(sampled_nodes, desc="Sample calculation 2-hop"):
        neighbors_1hop = adj_matrix[:, node].indices
        
        if len(neighbors_1hop) == 0:
            continue
        
        if len(neighbors_1hop) > 50:
            neighbors_1hop = np.random.choice(neighbors_1hop, 50, replace=False)
        
        neighbors_2hop = set()
        for neighbor in neighbors_1hop:
            second_neighbors = adj_matrix[:, neighbor].indices
            neighbors_2hop.update(second_neighbors)
        
        neighbors_2hop.discard(node)
        neighbors_2hop.difference_update(neighbors_1hop)
        neighbors_2hop = np.array(list(neighbors_2hop))
        
        if len(neighbors_2hop) > max_neighbors:
            neighbors_2hop = np.random.choice(neighbors_2hop, max_neighbors, replace=False)
        
        if len(neighbors_2hop) == 0:
            continue
        
        neighbor_feats = node_features[neighbors_2hop]
        
        for feat_idx in results.keys():
            feat_values = neighbor_feats[:, feat_idx]
            results[feat_idx]['mean'][node] = feat_values.mean()
    
    all_indices = np.arange(n_nodes)
    unsampled = np.setdiff1d(all_indices, sampled_nodes)
    
    for feat_idx in results.keys():
        sampled_mean = results[feat_idx]['mean'][sampled_nodes].mean()
        results[feat_idx]['mean'][unsampled] = sampled_mean
    
    return results


def compute_degree_correlation_features(adj_matrix: sp.csc_matrix,
                                       node_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculating Neighborhood Relation to Characterism"""
    n_nodes = adj_matrix.shape[0]
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    results = {}
    n_features = node_features.shape[1]
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="Relevance"):
        feature_values = node_features[:, feat_idx]
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
        
        results[f'orig_feat{feat_idx}_degree_corr'] = neighbor_feat_mean * neighbor_degree_mean
        results[f'orig_feat{feat_idx}_degree_diff'] = feature_values - neighbor_degree_mean
        results[f'orig_feat{feat_idx}_feat_degree_ratio'] = np.divide(
            feature_values, neighbor_degree_mean + 1e-8
        )
    
    return results


def compute_neighbor_distribution_stats(adj_matrix: sp.csc_matrix,
                                       node_features: np.ndarray,
                                       max_neighbors: int = 50) -> Dict[str, np.ndarray]:
    """Computation of neighbourhood feature distribution statistics"""
    n_nodes, n_features = node_features.shape
    results = {}
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="Distribution statistics"):
        feature_values = node_features[:, feat_idx]
        skewness = np.zeros(n_nodes, dtype=np.float32)
        kurtosis = np.zeros(n_nodes, dtype=np.float32)
        iqr = np.zeros(n_nodes, dtype=np.float32)
        
        for node in range(n_nodes):
            neighbors = adj_matrix[:, node].indices
            
            if len(neighbors) > max_neighbors:
                neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
            
            if len(neighbors) < 3:
                continue
            
            neighbor_feats = feature_values[neighbors]
            mean = neighbor_feats.mean()
            std = neighbor_feats.std()
            
            if std > 0:
                skewness[node] = ((neighbor_feats - mean) ** 3).mean() / (std ** 3)
                kurtosis[node] = ((neighbor_feats - mean) ** 4).mean() / (std ** 4) - 3
            
            q75, q25 = np.percentile(neighbor_feats, [75, 25])
            iqr[node] = q75 - q25
        
        results[f'orig_feat{feat_idx}_neighbor_skew'] = skewness
        results[f'orig_feat{feat_idx}_neighbor_kurt'] = kurtosis
        results[f'orig_feat{feat_idx}_neighbor_iqr'] = iqr
    
    return results


def compute_in_out_difference_features(adj_matrix: sp.csr_matrix,
                                      node_features: np.ndarray,
                                      max_neighbors: int = 50) -> Dict[str, np.ndarray]:
    """Calculate differences in access to and from neighbours"""
    n_nodes, n_features = node_features.shape
    adj_matrix_t = adj_matrix.T
    results = {}
    
    for feat_idx in tqdm(range(min(3, n_features)), desc="Variance"):
        feature_values = node_features[:, feat_idx]
        in_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        out_feat_mean = np.zeros(n_nodes, dtype=np.float32)
        
        adj_matrix_t_csc = adj_matrix_t.tocsc()
        for node in range(n_nodes):
            in_neighbors = adj_matrix_t_csc[:, node].indices
            
            if len(in_neighbors) > max_neighbors:
                in_neighbors = np.random.choice(in_neighbors, max_neighbors, replace=False)
            
            if len(in_neighbors) > 0:
                in_feat_mean[node] = feature_values[in_neighbors].mean()
            else:
                in_feat_mean[node] = feature_values[node]
        
        adj_matrix_csc = adj_matrix.tocsc()
        for node in range(n_nodes):
            out_neighbors = adj_matrix_csc[:, node].indices
            
            if len(out_neighbors) > max_neighbors:
                out_neighbors = np.random.choice(out_neighbors, max_neighbors, replace=False)
            
            if len(out_neighbors) > 0:
                out_feat_mean[node] = feature_values[out_neighbors].mean()
            else:
                out_feat_mean[node] = feature_values[node]
        
        results[f'orig_feat{feat_idx}_in_out_diff'] = in_feat_mean - out_feat_mean
        results[f'orig_feat{feat_idx}_in_out_ratio'] = np.divide(
            in_feat_mean, out_feat_mean + 1e-8
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Combining Neighbors\' Feature Aggregation')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Enter a path to the npz file (including edge index and x)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output Pickle File Path')
    parser.add_argument('--max_neighbors_1hop', type=int, default=50,
                       help='Maximum number of neighbours 1-hop')
    parser.add_argument('--max_neighbors_2hop', type=int, default=500,
                       help='2-hop maximum number of neighbours')
    parser.add_argument('--n_bins', type=int, default=5,
                       help='Number of boxes')
    args = parser.parse_args()
    
    print(f"Loading data: {args.input_path}")
    data = np.load(args.input_path, allow_pickle=True)
    
    if 'edge_index' not in data:
        raise ValueError("npz file must contain edge index")
    if 'x' not in data:
        raise ValueError("npz files must contain x")
    
    edge_index = data['edge_index']
    x = data['x']
    
    print(f"  Number of nodes: {x.shape[0]}, Number of features: {x.shape[1]}")
    
    # Calculating Combination Neighbor Features
    features_df = compute_combined_neighbor_features(
        edge_index, x,
        args.max_neighbors_1hop,
        args.max_neighbors_2hop,
        args.n_bins,
        args.output_path
    )
    
    print(f"\nCombining neighbourhood feature work completed!")
    print(f"Output file: {args.output_path}")


if __name__ == "__main__":
    main()
