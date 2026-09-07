#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add empty information features to the 2025 CFC DGraph primary game - restore version
Run: python add null features.py-case 1
Output: Data/face1/face1 feature null.pkl"""
import os
import gc
import pickle
import click
import numpy as np
import pandas as pd
import argparse


# 1. Creation of a solver
parser = argparse.ArgumentParser(description='makefea_part3')

# bin_dict = pickle.load(open(opj(work_path,'feature','bin_dict.pkl'), 'rb'))
# bin_prob_dict = pickle.load(open(opj(work_path,'feature','bin_prob_dict.pkl'), 'rb'))
# data = np.load(opj(data_path,data_name,'raw','gdata.npz'))
# path = opj (data path, data name,'feasure.pkl')#to.pkl

# 2. Definition parameters
parser.add_argument('--path_bin_dict', type=str, default='../feature/phase1/bin_dict.pkl')
parser.add_argument('--path_bin_prob_dict', type=str, default='../feature/phase1/bin_prob_dict.pkl')
parser.add_argument('--path_data', type=str, default='../data/phase1/gdata.npz')
parser.add_argument('--path_save_feature', type=str, default='../feature/phase1/feature.pkl')
parser.add_argument('--path_save_feature_supplement', type=str, default='../feature/phase1/feature_supplement.pkl')
parser.add_argument('--path_save_feature_null', type=str, default='../feature/phase1/feature_null.pkl')
parser.add_argument('--path_save_feature_mk', type=str, default='../feature/phase1/df_node_enhanced_mk.pkl')
parser.add_argument('--sub_ratio', type=float, default=1.0)

# 3. Parsing orders Okay.
args = parser.parse_args()

def safe_div(a, b, fill=0):
    """Safety Division - Restore Version"""
    if hasattr(a, '__len__') and hasattr(b, '__len__'):
        # Processing arrays
        result = np.full_like(a, fill, dtype=float)
        mask = b != 0
        result[mask] = a[mask] / b[mask]
        return result
    else:
        # Processing of the subject matter
        if b == 0:
            return fill
        return a / b

def add_advanced_null_pattern_features(df, x):
    """Advanced Air Mode Features - Restore Version"""
    print("Add advanced empty mode features...")
    
    # Empty tag for original feature (-1 for empty)
    null_mask = (x == -1)
    n_nodes = x.shape[0]
    
    # 1.1 Basic space statistics
    null_count = null_mask.sum(axis=1)
    df['null_count'] = null_count
    df['null_ratio'] = safe_div(null_count, x.shape[1])
    
    # 1.2 Key feature empty value tags
    for i in range(5):  # Only the first five key features are marked.
        df[f'f{i}_is_null'] = (x[:, i] == -1).astype(np.float32)
    
    # 1.3 Space distribution features
    # Front feature empty value (assuming the first eight features are more important)
    front_features = list(range(8))
    front_null_ratio = null_mask[:, front_features].mean(axis=1)
    df['front_feature_null_ratio'] = front_null_ratio
    
    # Back feature empty value
    back_features = list(range(8, 17))
    back_null_ratio = null_mask[:, back_features].mean(axis=1)
    df['back_feature_null_ratio'] = back_null_ratio
    
    # 1.4 Simple statistics of empty value patterns
    df['null_pattern_mean'] = null_mask.mean(axis=1)
    df['null_pattern_std'] = null_mask.std(axis=1)
    
    return df

def add_simple_neighbor_null_features(df, edge_df, x):
    """2. Simplified non-neighborhood feature - Avoiding memory explosions"""
    print("Add simplified neighbourhood empty feature...")
    
    null_mask = (x == -1)
    node_null_ratio = null_mask.mean(axis=1)
    n_nodes = len(node_null_ratio)
    
    # 2.1 Outside neighbourhood empty value features
    print("Calculating an empty neighborhood feature...")
    out_stats = calculate_simple_neighbor_stats(edge_df, '0', '1', node_null_ratio, n_nodes)
    df['out_nei_null_ratio_mean'] = out_stats['mean']
    df['out_nei_null_ratio_max'] = out_stats['max']
    df['out_nei_high_null_ratio'] = out_stats['high_ratio']
    
    # 2.2 Quantities of entry neighbours
    print("Calculating the empty value of the border neighbor...")
    in_stats = calculate_simple_neighbor_stats(edge_df, '1', '0', node_null_ratio, n_nodes)
    df['in_nei_null_ratio_mean'] = in_stats['mean']
    df['in_nei_null_ratio_max'] = in_stats['max']
    df['in_nei_high_null_ratio'] = in_stats['high_ratio']
    
    return df

def calculate_simple_neighbor_stats(edge_df, src_col, dst_col, node_null_ratio, n_nodes):
    """Compute simplified neighbourhood statistics - use quantitative operations"""
    # Projected grouping
    src_to_dsts = {}
    for src, group in edge_df.groupby(src_col):
        src_to_dsts[src] = group[dst_col].values
    
    mean_stats = np.zeros(n_nodes)
    max_stats = np.zeros(n_nodes)
    high_ratio_stats = np.zeros(n_nodes)
    
    for src in range(n_nodes):
        if src in src_to_dsts:
            dst_nodes = src_to_dsts[src]
            if len(dst_nodes) > 0:
                nei_null_ratios = node_null_ratio[dst_nodes]
                mean_stats[src] = np.mean(nei_null_ratios)
                max_stats[src] = np.max(nei_null_ratios)
                high_ratio_stats[src] = np.mean(nei_null_ratios > 0.5)
    
    return {
        'mean': mean_stats,
        'max': max_stats,
        'high_ratio': high_ratio_stats
    }

def add_temporal_null_features(df, edge_df, x, edge_timestamp):
    """Time-related empty value features - simplified version"""
    print("Add empty values associated with time...")
    
    null_mask = (x == -1)
    node_null_ratio = null_mask.mean(axis=1)
    max_timestamp = np.max(edge_timestamp)
    n_nodes = len(node_null_ratio)
    
    # Only the last 30-day feature.
    window = 30
    print(f"Calculate{window}Sky Time Window Empty Value Character...")
    
    # Lately on the edge of Window.
    recent_mask = edge_df['edge_timestamp'] >= (max_timestamp - window + 1)
    recent_edges = edge_df[recent_mask]
    
    # Outside neighbor time window feature
    out_recent_stats = calculate_simple_neighbor_stats(recent_edges, '0', '1', node_null_ratio, n_nodes)
    df[f'out_recent_{window}d_null_mean'] = out_recent_stats['mean']
    
    # Quarterside Time Window Feature
    in_recent_stats = calculate_simple_neighbor_stats(recent_edges, '1', '0', node_null_ratio, n_nodes)
    df[f'in_recent_{window}d_null_mean'] = in_recent_stats['mean']
    
    return df

def add_null_risk_features(df, x, edge_df):
    """4. Empty value risk features"""
    print("Add empty value risk features...")
    
    null_mask = (x == -1)
    node_null_ratio = null_mask.mean(axis=1)
    n_nodes = len(node_null_ratio)
    
    # 4.1 Basic risk scores
    df['null_risk_basic'] = node_null_ratio
    
    # 4.2 Key characterization risks
    key_features_null = null_mask[:, :5].any(axis=1)
    df['key_feature_null_risk'] = key_features_null.astype(np.float32)
    
    # 4.3 Network risk dissemination
    # Compute simple statistics for nodes
    out_degree_dict = edge_df.groupby('0').size().to_dict()
    in_degree_dict = edge_df.groupby('1').size().to_dict()
    
    # Consolidation Statistics
    total_degree = np.zeros(n_nodes)
    for i in range(n_nodes):
        total_degree[i] = out_degree_dict.get(i, 0) + in_degree_dict.get(i, 0)
    
    # Weighted risk
    degree_weights = np.minimum(total_degree / 1000, 1.0)  # Limit the scope of weights
    df['degree_weighted_null_risk'] = node_null_ratio * degree_weights
    
    # 4.4 Anomalous points for empty values
    null_mean = np.mean(node_null_ratio)
    null_std = np.std(node_null_ratio)
    if null_std > 0:
        df['null_anomaly_score'] = (node_null_ratio - null_mean) / null_std
    else:
        df['null_anomaly_score'] = 0
    
    return df

def add_null_interaction_features(df, x):
    """5. Space interactive features"""
    print("Add empty interactive features...")
    
    null_mask = (x == -1)
    
    # 5.1 Empty Group Mode
    # Empty combination of the first three features
    f0_null = null_mask[:, 0]
    f1_null = null_mask[:, 1]
    f2_null = null_mask[:, 2]
    
    df['f0_f1_null_both'] = (f0_null & f1_null).astype(np.float32)
    df['f0_f2_null_both'] = (f0_null & f2_null).astype(np.float32)
    df['f1_f2_null_both'] = (f1_null & f2_null).astype(np.float32)
    df['f0_f1_f2_null_all'] = (f0_null & f1_null & f2_null).astype(np.float32)
    
    # 5.2 Diversity of empty value patterns
    # Calculate distribution of empty values in different feature groups
    group1_null = null_mask[:, :6].mean(axis=1)  # First half features
    group2_null = null_mask[:, 6:12].mean(axis=1)  # Intermediate segment features
    group3_null = null_mask[:, 12:].mean(axis=1)  # Second half features
    
    # The difference in calculation criteria as a diversity indicator
    null_groups = np.stack([group1_null, group2_null, group3_null])
    df['null_group_diversity'] = np.std(null_groups, axis=0)
    
    return df


def main():

    # print(f'== = Generate {case name} Empty value feature===')
    
    # 1. Reading raw data
    raw_path = args.path_data
    print(f"Load data:{raw_path}")
    data = np.load(raw_path, allow_pickle='True')
    
    x = data['x']
    edge_index = data['edge_index']
    edge_type = data['edge_type'] 
    edge_timestamp = data['edge_timestamp']
    
    print(f"Data shape: x={x.shape}, Edge ={edge_index.shape[0]}")
    
    # 2. Create border data frames
    edge_data = np.column_stack([
        edge_index[:, 0].astype(np.int32),
        edge_index[:, 1].astype(np.int32),
        edge_type.astype(np.int32),
        edge_timestamp.astype(np.int32)
    ])
    
    edge_df = pd.DataFrame(edge_data, columns=['0', '1', 'edge_type', 'edge_timestamp'])

    n_init = edge_df.shape[0]
    print('->'*10, 'df edge init sample quantity: {'.format(n_init) )
    if args.sub_ratio<1:
        
        edge_df = edge_df.sample(frac = args.sub_ratio, random_state=42)   # Return to New DataFrame
        n_subsample = edge_df.shape[0]
        print('->'*10, 'df edge sample ratio is: {, and sample size is: {'.format(n_init, n_subsample) )

    
    N = x.shape[0]
    df = pd.DataFrame(index=range(N))
    
    # Add empty-value feature - step-by-step, clean memory in time
    print("Start empty value feature calculation...")
    
    try:
        # Step 1: Empty value features of node itself
        print("Step 1: Node Empty Character...")
        df = add_advanced_null_pattern_features(df, x)
        gc.collect()
        
        # Step 2: Space interactive features
        print("Step 2: Empty interactive features...")
        df = add_null_interaction_features(df, x)
        gc.collect()
        
        # Step 3: Neighbours ' empty values feature (most time-consuming segment)
        print("Step 3: Neighbors' empty values...")
        df = add_simple_neighbor_null_features(df, edge_df, x)
        gc.collect()
        
        # Step 4: Time space feature
        print("Step 4: Time-space feature...")
        df = add_temporal_null_features(df, edge_df, x, edge_timestamp)
        gc.collect()
        
        # Step 5: Risk features
        print("Step 5: Risk features...")
        df = add_null_risk_features(df, x, edge_df)
        gc.collect()
        
    except Exception as e:
        print(f"Error during feature calculation:{e}")
        print("Try saving calculated features...")
    
    # Data cleansing and optimization
    print("\\\\n data cleansing and optimization...")
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    # Type Conversion
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    
    # 5. Preservation of results
    out_path = args.path_save_feature_null
    with open(out_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f'✅ Empty value feature saved:{out_path}')
    print(f'Final feature shape:{df.shape}')
    print(f'Number of features:{len(df.columns)}')
    
    # Show feature category
    node_features = len([c for c in df.columns if 'f' in c and 'null' in c])
    neighbor_features = len([c for c in df.columns if 'nei' in c])
    risk_features = len([c for c in df.columns if 'risk' in c or 'anomaly' in c])
    temporal_features = len([c for c in df.columns if 'recent' in c or 'd_' in c])
    interaction_features = len([c for c in df.columns if 'both' in c or 'all' in c or 'diversity' in c])
    
    print(f'Feature category statistics:')
    print(f'- Node empty values:{node_features}')
    print(f'- Neighbour transmission features:{neighbor_features}')
    print(f'- Time-space feature:{temporal_features}')
    print(f'- Risk indicator:{risk_features}')
    print(f'- Interaction features:{interaction_features}')

if __name__ == '__main__':
    main()
