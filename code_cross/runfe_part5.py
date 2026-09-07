#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Node type distribution feature - pure DataFrame version quantified
Total elimination of for-cycle, extreme performance
Run: python forecast node type vectorized.py
Output: ../feature/face1/feature node type vectorized.pkl"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
import argparse

# 1. Creation of a solver
parser = argparse.ArgumentParser(description='feature_node_type_vectorized')

# 2. Definition parameters
parser.add_argument('--path_data', type=str, default='../data/phase1/gdata.npz')
parser.add_argument('--path_save_feature_type', type=str, default='../feature/phase1/feature_node_type_vectorized.pkl')
parser.add_argument('--sub_ratio', type=float, default=1.0)

# 3. Parsing orders Okay.
args = parser.parse_args()

def safe_div(a, b, fill=0.0):
    """Safety Division"""
    return np.divide(a, b, out=np.full_like(a, fill, dtype=float), where=b!=0)

def create_node_type_mapping(y):
    """Create node type map"""
    node_type = np.zeros_like(y)
    node_type[np.isin(y, [0, 1, -100])] = 0  # Type 0: 0, 1, -100
    node_type[y == 2] = 1                     # Type 1: 2
    node_type[y == 3] = 2                     # Type 2: 3
    return node_type

def create_vectorized_neighbor_type_features(edge_index, node_type):
    """Purely quantitative neighbour type features"""
    print("Create Quantified Neighbourhood Type Feature...")
    
    src = edge_index[:, 0].flatten()
    dst = edge_index[:, 1].flatten()
    N = len(node_type)
    
    # Create EdgeDataFrame
    edge_df = pd.DataFrame({'src': src, 'dst': dst})

    n_init = edge_df.shape[0]
    print('->'*10, 'df edge init sample quantity: {'.format(n_init) )
    if args.sub_ratio<1:
        
        edge_df = edge_df.sample(frac = args.sub_ratio, random_state=42)   # Return to New DataFrame
        n_subsample = edge_df.shape[0]
        print('->'*10, 'df edge sample ratio is: {, and sample size is: {'.format(n_init, n_subsample) )

    
    # Create Node TypeDataFrame
    node_type_series = pd.Series(node_type, name='node_type')
    
    # 1. Statistics on types of out-of-border neighbours
    print("Calculating Border Neighbor Type...")
    out_with_types = edge_df.merge(
        node_type_series.rename('dst_type'), left_on='dst', right_index=True
    )
    out_type_counts = pd.crosstab(out_with_types['src'], out_with_types['dst_type'])
    out_type_counts = out_type_counts.reindex(columns=[0,1,2], fill_value=0)
    out_type_counts.columns = [f'out_type_{i}_cnt' for i in out_type_counts.columns]
    
    # 2. Statistics on types of neighbours
    print("Calculating Border Neighbor Type...")
    in_with_types = edge_df.merge(
        node_type_series.rename('src_type'), left_on='src', right_index=True
    )
    in_type_counts = pd.crosstab(in_with_types['dst'], in_with_types['src_type'])
    in_type_counts = in_type_counts.reindex(columns=[0,1,2], fill_value=0)
    in_type_counts.columns = [f'in_type_{i}_cnt' for i in in_type_counts.columns]
    
    # 3. Create basic features DataFrame
    features = pd.DataFrame(index=range(N))
    features = features.join(out_type_counts, how='left')
    features = features.join(in_type_counts, how='left')
    features = features.fillna(0)
    
    # 4. Measurement features
    print("Calculator Character...")
    features['out_deg'] = edge_df.groupby('src').size().reindex(features.index, fill_value=0)
    features['in_deg'] = edge_df.groupby('dst').size().reindex(features.index, fill_value=0)
    features['total_deg'] = features['out_deg'] + features['in_deg']
    
    # Calculate the proportion of types - to quantitative operations
    print("Calculate Type Ratio...")
    for i in range(3):
        features[f'out_type_{i}_ratio'] = safe_div(features[f'out_type_{i}_cnt'], features['out_deg'])
        features[f'in_type_{i}_ratio'] = safe_div(features[f'in_type_{i}_cnt'], features['in_deg'])
    
    # 6. Major Neighbourhood Types - Quantified Operations
    print("Calculating Main Neighbor Type...")
    out_cnt_cols = [f'out_type_{i}_cnt' for i in range(3)]
    in_cnt_cols = [f'in_type_{i}_cnt' for i in range(3)]
    
    features['out_main_type'] = features[out_cnt_cols].idxmax(axis=1).str.replace('out_type_', '').str.replace('_cnt', '').astype(int)
    features['in_main_type'] = features[in_cnt_cols].idxmax(axis=1).str.replace('in_type_', '').str.replace('_cnt', '').astype(int)
    
    # Connect to type - Quantification
    print("Calculating Connection to Type...")
    node_type_df = pd.DataFrame({'node_type': node_type}, index=range(N))
    features = features.join(node_type_df, how='left')
    
    # Connecting quantitative calculations to type using the lookup method
    out_same_mask = pd.Series(range(N)).apply(lambda x: f'out_type_{features.loc[x, "node_type"]}_ratio')
    in_same_mask = pd.Series(range(N)).apply(lambda x: f'in_type_{features.loc[x, "node_type"]}_ratio')
    
    features['out_same_ratio'] = features.lookup(features.index, out_same_mask)
    features['in_same_ratio'] = features.lookup(features.index, in_same_mask)
    
    # High-risk connectivity features
    print("Calculating high-risk connections...")
    features['out_risk_ratio'] = features['out_type_2_ratio']
    features['in_risk_ratio'] = features['in_type_2_ratio']
    features['risk_propagation'] = features['out_deg'] * features['out_risk_ratio']
    
    # 9. Features of type imbalance - to quantify
    print("Calculating Type Uneven...")
    features['type_imbalance'] = (
        abs(features['out_type_0_ratio'] - features['in_type_0_ratio']) +
        abs(features['out_type_1_ratio'] - features['in_type_1_ratio']) + 
        abs(features['out_type_2_ratio'] - features['in_type_2_ratio'])
    )
    
    # 10. Clean-up of temporary columns
    features = features.drop(['out_deg', 'in_deg', 'total_deg', 'node_type'], axis=1)
    
    return features

def main():
    print("🚀 Start node type distribution feature creation - pure quantitative version...")
    
    # 1. Loading data
    print("Load raw data...")
    data = np.load(args.path_data, allow_pickle='True')
    
    edge_index = data['edge_index']
    
    y = data['y'].flatten()
    
    print(f"Number of edges: {len(edge_index)}, Number of nodes: {len(y)}")
    
    # 2. Create node type mapping
    print("Create node type map...")
    node_type = create_node_type_mapping(y)
    
    print("Type distribution:")
    unique, counts = np.unique(node_type, return_counts=True)
    for val, cnt in zip(unique, counts):
        print(f"  Type{val}: {cnt}")
    
    # 3. Creation features
    print("Create vectorized features...")
    features = create_vectorized_neighbor_type_features(edge_index, node_type)
    
    # Data cleansing
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    # Type optimization
    for col in features.columns:
        if features[col].dtype == np.float64:
            features[col] = features[col].astype(np.float32)
    
    # 5. Preservation of results
    output_path = args.path_save_feature_type
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"\n🎉 Vectorized feature creation complete!")
    print(f"📊 Feature statistics:")
    print(f"  - Total number of features: {features.shape[1]}")
    print(f"  - Number of rows: {features.shape[0]}")
    print(f"  - Output path: {output_path}")
    
    print(f"\n🔍 Feature list (20 items):")
    feature_types = {
        'Type counts': [c for c in features.columns if '_cnt' in c],
        'Type ratios': [c for c in features.columns if '_ratio' in c and 'same' not in c and 'risk' not in c],
        'Dominant types': [c for c in features.columns if 'main_type' in c],
        'Homophily risk': [c for c in features.columns if 'same' in c or 'risk' in c],
        'Imbalance': [c for c in features.columns if 'imbalance' in c or 'propagation' in c]
    }
    
    for category, cols in feature_types.items():
        print(f"  {category} ({len(cols)} items): {', '.join(cols)}")

if __name__ == '__main__':
    main()
