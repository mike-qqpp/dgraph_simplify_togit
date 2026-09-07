#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add additional time series + structural features to 2025 CCS DGraph primarys
Run: python add supplement features.py-case1
Output: Data/phase1/phase1 feature supplement.pkl"""
import os
import gc
import pickle
import click
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import argparse
# from utils import data_path
# 1. Creation of a solver
parser = argparse.ArgumentParser(description='makefea_part2')

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
    return np.divide(a, b, out=np.full_like(a, fill, dtype=float), where=b!=0)

def add_days_derived(df, edge_df, max_day):
    """1. Day slice count"""
    print("Add day slice count feature...")
    
    # External Time Window Count
    for win in [7, 15, 30]:
        # Out.
        out_cnt = edge_df[edge_df['edge_timestamp'] >= max_day - win + 1].groupby('0').size().rename(f'td_{win}')
        df = df.join(out_cnt, how='left').fillna(0)
        
        # Step aside.
        in_cnt = edge_df[edge_df['edge_timestamp'] >= max_day - win + 1].groupby('1').size().rename(f'td_in_{win}')
        df = df.join(in_cnt, how='left').fillna(0)
    
    # Border type time window count
    for tp in range(11):
        sub = edge_df[edge_df['edge_type'] == tp]
        for win in [7, 15, 30]:
            cnt_series = sub[sub['edge_timestamp'] >= max_day - win + 1].groupby('0').size()
            df[f'td_type{tp}_{win}'] = df.index.map(cnt_series).fillna(0)
    
    return df

def add_degree_features(df, edge_df):
    """2. Degree features"""
    print("Add Degree Character...")
    
    # Out and in
    out_degree = edge_df.groupby('0').size().rename('out_degree')
    in_degree = edge_df.groupby('1').size().rename('in_degree')
    
    df = df.join(out_degree, how='left').fillna(0)
    df = df.join(in_degree, how='left').fillna(0)
    
    # Statistics
    df['total_degree'] = df['out_degree'] + df['in_degree']
    df['degree_ratio'] = safe_div(df['out_degree'], df['in_degree'])
    df['degree_imbalance'] = abs(df['out_degree'] - df['in_degree'])
    
    return df

def add_temporal_features(df, edge_df, max_day):
    """3. Time features"""
    print("Add Time Character...")
    
    # Active days
    active_days_out = edge_df.groupby('0')['edge_timestamp'].nunique().rename('active_days_out')
    active_days_in = edge_df.groupby('1')['edge_timestamp'].nunique().rename('active_days_in')
    df = df.join(active_days_out, how='left').fillna(0)
    df = df.join(active_days_in, how='left').fillna(0)
    
    # Recent Active Time
    last_active_out = edge_df.groupby('0')['edge_timestamp'].max().rename('last_active_out')
    last_active_in = edge_df.groupby('1')['edge_timestamp'].max().rename('last_active_in')
    df = df.join(last_active_out, how='left').fillna(0)
    df = df.join(last_active_in, how='left').fillna(0)
    
    # Inactive days
    df['inactive_days_out'] = max_day - df['last_active_out']
    df['inactive_days_in'] = max_day - df['last_active_in']
    
    return df

def add_type_ratio_features(df, edge_df, max_day):
    """4. Proportional features of marginal types"""
    print("Add border type ratio feature...")
    
    # Data for the last 7 days
    recent_edges = edge_df[edge_df['edge_timestamp'] >= max_day - 6]
    
    # Outside type %
    for tp in range(11):
        # Number of edges of this type
        type_edges = recent_edges[recent_edges['edge_type'] == tp]
        type_count = type_edges.groupby('0').size()
        
        # Total number of sides
        total_count = recent_edges.groupby('0').size()
        
        # Calculate share
        ratio_series = type_count / total_count
        df[f'td_type_ratio_7d_{tp}'] = df.index.map(ratio_series).fillna(0)
    
    # Percentage of type of input
    for tp in range(11):
        type_edges = recent_edges[recent_edges['edge_type'] == tp]
        type_count = type_edges.groupby('1').size()
        total_count = recent_edges.groupby('1').size()
        ratio_series = type_count / total_count
        df[f'td_type_ratio_in_7d_{tp}'] = df.index.map(ratio_series).fillna(0)
    
    return df

def add_risk_neighbor_features(df, edge_df, y, train_mask):
    """5. Neighbor risk features (training set only)"""
    print("Add Neighbour Risk Features...")
    
    # Create training set risk label
    train_risk_dict = {}
    for i, node_id in enumerate(train_mask):
        train_risk_dict[node_id] = y[node_id]
    
    # Outside neighbor risk.
    out_risk_sum = {}
    out_risk_max = {}
    out_risk_mean = {}
    
    for src, group in edge_df.groupby('0'):
        risks = [train_risk_dict.get(int(dst), 0) for dst in group['1']]
        if risks:
            out_risk_sum[src] = sum(risks)
            out_risk_max[src] = max(risks)
            out_risk_mean[src] = sum(risks) / len(risks)
        else:
            out_risk_sum[src] = 0
            out_risk_max[src] = 0
            out_risk_mean[src] = 0
    
    df['risk_nei_out_sum'] = df.index.map(out_risk_sum).fillna(0)
    df['risk_nei_out_max'] = df.index.map(out_risk_max).fillna(0)
    df['risk_nei_out_mean'] = df.index.map(out_risk_mean).fillna(0)
    
    # The risk to the neighbors.
    in_risk_sum = {}
    in_risk_max = {}
    in_risk_mean = {}
    
    for dst, group in edge_df.groupby('1'):
        risks = [train_risk_dict.get(int(src), 0) for src in group['0']]
        if risks:
            in_risk_sum[dst] = sum(risks)
            in_risk_max[dst] = max(risks)
            in_risk_mean[dst] = sum(risks) / len(risks)
        else:
            in_risk_sum[dst] = 0
            in_risk_max[dst] = 0
            in_risk_mean[dst] = 0
    
    df['risk_nei_in_sum'] = df.index.map(in_risk_sum).fillna(0)
    df['risk_nei_in_max'] = df.index.map(in_risk_max).fillna(0)
    df['risk_nei_in_mean'] = df.index.map(in_risk_mean).fillna(0)
    
    return df


def main():
    
    # 1. Reading raw data
    raw_path = args.path_data
    print(f"Load data:{raw_path}")
    data = np.load(raw_path, allow_pickle='True')
    
    x = data['x']
    y = data['y']
    edge_index = data['edge_index']
    edge_type = data['edge_type'] 
    edge_timestamp = data['edge_timestamp']
    # train_mask = data['train_mask']
    
    max_day = int(edge_timestamp.max())
    
    print(f"Data shape: x={x.shape}, Edge ={edge_index.shape[0]}, max ={max_day}")
    
    # Create border data frames - use pandas 1.x compatible writing
    # Create a 2-dimensional array and then set up a listing
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

    
    print(f"Border data frame shape:{edge_df.shape}")
    print(f"Edge type range:{edge_df['edge_type'].min()} ~ {edge_df['edge_type'].max()}")
    
    N = x.shape[0]
    df = pd.DataFrame(index=range(N))
    print(f"Initial feature data box:{df.shape}")
    
    # 3. Add feature in order
    print("Start feature calculation...")
    
    # Fast-calculated features first.
    df = add_degree_features(df, edge_df)
    df = add_temporal_features(df, edge_df, max_day)
    df = add_days_derived(df, edge_df, max_day)
    df = add_type_ratio_features(df, edge_df, max_day)
    
    # Features available only for Chase1
    # df = add_risk_neighbor_features(df, edge_df, y, train_mask)
    
    # Data cleansing and optimization
    print("\\\\n data cleansing and optimization...")
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    # Type Conversion
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    
    # 5. Preservation of results
    out_path = args.path_save_feature_supplement
    with open(out_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f'Additional features saved:{out_path}')
    print(f'Final feature shape:{df.shape}')
    print(f'Number of features:{len(df.columns)}')
    print(f'Top 10 features:{list(df.columns)[:10]}')

if __name__ == '__main__':
    main()
