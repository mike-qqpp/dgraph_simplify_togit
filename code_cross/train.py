#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script training - training CatBoost model using a 50-percent cross-validation
Function:
Load training data
2. Creation of cross-cutting features
3. Conduct of a 50-percent cross-validation training
4. Preservation of the 50-percent model

Usage method:
I don't know.
or
I don't know."""

import numpy as np
import pandas as pd
import pickle
import os
import gc
import psutil
import argparse
import sys
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import time

# == sync, corrected by elderman == @elder man
# These will be set according to the command line parameters.
DATA_DIR = None  # Set according to the data set name
LABEL_FILE = None  # Set according to the data set name
OUTPUT_DIR = None  # Set according to the data set name
DATASET_NAME = None  # Data set name

N_FOLDS = 5
RANDOM_SEED = 2022

# == sync, corrected by elderman == @elder man
def get_COLS_KEEP(dataset):
    if dataset=='amazon':
        COLS_FEA_SLC = [
            'fea_x_19', 'fea_x_12', 'fea_out_neighbor_fea_x_0_max', 'fea_out_neighbor_fea_x_7_max', 
            'fea_out_neighbor_fea_x_7_mean', 'fea_x_0', 'fea_in_neighbor_fea_x_0_max', 'fea_in_neighbor_fea_x_0_min', 
            'td_type_ratio_in_7d_1', 'fea_in_neighbor_fea_x_0_mean', 'fea_x_23', 'fea_x_18', 
            'fea_out_neighbor_fea_x_0_min', 'fea_x_7', 'td_type_ratio_7d_1', 'fea_x_13', 
            'fea_out_neighbor_fea_x_7_min', 'fea_x_15', 'td_type2_30', 'td_type_ratio_7d_0', 
            'fea_out_neighbor_fea_x_8_min', 'fea_in_edge_type_mean', 'directed_1hop_in_type_0_count', 
            'td_type1_15', 'financial_hub_type_0_count', 'fea_out_neighbor_fea_x_0_mean', 'fea_x_8', 
            'td_type1_30', 'fea_x_14', 'fea_x_17', 'fea_x_20', 'fea_out_edge_type_mean', 
            'directed_2hop_in_type_4_count', 'td_type2_7', 'td_type_ratio_in_7d_4', 'fea_out_neighbor_fea_x_6_min', 
            'td_type_ratio_7d_4', 'td_type3_15', 'directed_1hop_in_f26_std', 'td_type2_15', 
            'mixed_propagation_type_4_count', 'directed_1hop_out_type_4_count', 'td_type0_7', 
            'financial_hub_type_2_count', 'directed_2hop_in_f14_mean', 'fea_in_edge_type_std', 
            'td_in_15', 'directed_2hop_in_f23_max', 'financial_hub_f5_std', 'directed_2hop_in_f19_max', 
            'fea_x_2', 'undirected_1hop_type_0_count', 'td_type6_30', 'td_30', 
            'td_type_ratio_in_7d_7', 'transaction_chain_f0_mean', 'directed_1hop_out_type_0_count', 
            'undirected_1hop_f5_std', 'undirected_2hop_type_0_count', 'td_type1_7', 'td_type0_15', 
            'td_type0_30', 'transaction_chain_type_0_count', 'fea_out_neighbor_fea_x_2_mean', 
            'in_recent_30d_null_mean', 'mixed_propagation_f14_std', 'transaction_chain_f23_max', 
            'directed_2hop_out_type_0_count', 'fea_out_neighbor_fea_x_2_max', 'fea_out_neighbor_fea_x_6_max', 
            'directed_2hop_in_f16_min', 'td_type_ratio_in_7d_0', 'mixed_propagation_f3_mean', 
            'undirected_1hop_type_2_count', 'undirected_2hop_f0_mean', 'directed_2hop_in_f19_mean', 
            'td_type4_7', 'directed_2hop_out_f26_std', 'core_business_f48_std', 'td_type4_15', 
            'td_type3_30', 'core_business_f45_mean', 'financial_hub_f0_std', 'fea_in_edge_type_mode', 
            'transaction_chain_f23_std', 'td_type_ratio_7d_7', 'fea_out_edge_type_std', 
            'directed_1hop_in_f18_std', 'directed_2hop_in_f6_std', 'undirected_1hop_f3_mean', 
            'directed_2hop_out_f25_std', 'financial_hub_type_4_count', 'in_nei_null_ratio_mean', 
            'directed_2hop_in_f18_mean', 'directed_1hop_out_f39_mean', 'td_type_ratio_in_7d_3', 
            'directed_1hop_out_f26_mean', 'directed_2hop_in_f23_std', 'directed_1hop_in_f19_max', 
            'undirected_1hop_type_4_count', 'financial_hub_f5_max', 'transaction_chain_f19_mean', 
            'financial_hub_f15_std', 'financial_hub_f14_std', 'transaction_chain_f24_mean', 
            'core_business_f19_std', 'transaction_chain_f21_std', 'directed_1hop_in_f19_mean', 
            'financial_hub_f11_mean', 'mixed_propagation_f40_mean', 'financial_hub_f23_max', 
            'transaction_chain_f23_mean', 'td_type3_7', 'financial_hub_f5_mean', 
            'directed_1hop_in_f4_std', 'core_business_f15_mean', 'directed_2hop_out_f11_std', 
            'financial_hub_f13_mean', 'undirected_2hop_type_4_count', 'directed_2hop_in_f19_std', 
            'transaction_chain_f10_mean', 'financial_hub_f23_min', 'td_type_ratio_7d_6', 
            'directed_2hop_out_f14_mean', 'fea_x_6', 'core_business_f20_mean', 'core_business_f12_mean', 
            'financial_hub_f23_std', 'out_recent_30d_null_mean', 'transaction_chain_f14_max'
        ]
        
    elif dataset=='yelpchi':
        COLS_FEA_SLC = ['fea_x_18', 'fea_x_19', 'td_type_ratio_7d_1', 'fea_x_21', 'fea_x_13', 'td_type1_15', 'fea_x_20', 'fea_out_neighbor_fea_x_5_mean', 'fea_out_neighbor_fea_x_0_mean', 'fea_x_14', 'fea_x_0', 'td_type1_30', 'fea_x_16', 'fea_out_neighbor_fea_x_0_max', 'fea_x_1', 'fea_x_23', 'fea_out_neighbor_fea_x_5_max', 'fea_x_26', 'risk_nei_out_mean', 'td_type_ratio_in_7d_1', 'fea_out_neighbor_fea_x_1_max', 'fea_x_5', 'risk_nei_in_mean', 'td_type1_7', 'fea_out_neighbor_fea_x_0_min', 'fea_x_15', 'fea_x_22', 'fea_out_neighbor_fea_x_1_mean', 'fea_x_10', 'fea_x_25', 'fea_out_neighbor_fea_x_5_min', 'fea_x_30', 'fea_x_8', 'fea_out_neighbor_fea_x_1_min', 'fea_x_17', 'fea_x_28', 'fea_x_27', 'fea_out_neighbor_fea_x_8_max', 'fea_x_31', 'fea_in_edge_type_min', 'fea_x_29', 'directed_1hop_in_f24_std', 'undirected_1hop_f62_std', 'directed_2hop_out_f0_mean', 'directed_1hop_in_f6_min', 'fea_out_edge_type_min', 'directed_1hop_in_f13_mean', 'fea_out_neighbor_fea_x_2_mean', 'mixed_propagation_f0_mean', 'directed_1hop_in_f29_min', 'fea_x_2', 'fea_out_neighbor_fea_x_8_mean', 'fea_out_neighbor_fea_x_8_min', 'fea_x_24', 'mixed_propagation_f6_mean', 'fea_out_neighbor_fea_x_2_min', 'mixed_propagation_f16_std', 'directed_1hop_in_f63_std', 'financial_hub_f5_mean', 'directed_2hop_out_f8_mean', 'directed_1hop_out_f31_std', 'undirected_1hop_f63_std', 'undirected_1hop_f59_std', 'directed_1hop_in_f0_mean', 'directed_1hop_in_f62_std', 'transaction_chain_f50_std', 'directed_2hop_out_f30_std', 'directed_1hop_out_f7_std', 'fea_out_edge_type_nunique', 'mixed_propagation_f46_mean', 'directed_1hop_out_f62_min', 'directed_2hop_in_f0_std', 'directed_1hop_out_f30_std', 'undirected_1hop_f28_std', 'directed_2hop_in_f0_mean', 'mixed_propagation_f0_min', 'directed_1hop_out_f24_std', 'directed_2hop_in_f0_min', 'directed_1hop_in_f1_mean', 'directed_1hop_in_f58_std', 'directed_1hop_in_f31_std', 'mixed_propagation_f23_max', 'directed_1hop_in_f28_std', 'transaction_chain_f10_std', 'transaction_chain_f0_std', 'directed_1hop_in_f26_std', 'transaction_chain_f10_min', 'mixed_propagation_f25_std', 'directed_1hop_out_f62_std', 'undirected_1hop_f27_std', 'directed_2hop_out_f6_mean', 'mixed_propagation_f32_mean', 'directed_1hop_in_f7_mean', 'directed_2hop_in_f25_std', 'mixed_propagation_f28_std', 'mixed_propagation_f9_std', 'directed_1hop_in_f11_mean', 'directed_2hop_in_f8_std', 'transaction_chain_f62_std', 'directed_2hop_in_f42_std', 'transaction_chain_f0_mean', 'mixed_propagation_f10_min', 'directed_2hop_in_f25_mean', 'directed_2hop_in_f1_max', 'directed_2hop_in_f31_std', 'directed_1hop_out_f0_mean', 'directed_2hop_in_f14_mean', 'directed_2hop_in_f10_std', 'directed_2hop_in_f24_mean', 'directed_1hop_in_f10_mean', 'directed_1hop_in_f25_std', 'directed_1hop_in_f29_std', 'fea_out_neighbor_fea_x_2_max', 'directed_1hop_out_f28_std', 'directed_1hop_in_f14_std', 'directed_2hop_in_f62_std', 'directed_2hop_out_f6_min', 'directed_2hop_out_f23_max', 'mixed_propagation_f0_max', 'financial_hub_f0_mean', 'directed_1hop_in_f23_mean', 'fea_in_edge_type_nunique', 'directed_2hop_in_f24_std', 'directed_1hop_in_f8_std', 'directed_1hop_in_f32_mean', 'transaction_chain_f9_mean', 'directed_2hop_out_f32_mean', 'td_type_ratio_7d_2', 'directed_2hop_out_f6_std', 'mixed_propagation_f14_mean']
    
    elif dataset=='tfinance':
        COLS_FEA_SLC = ['risk_nei_out_mean', 'risk_nei_in_mean', 'risk_nei_in_sum', 'risk_nei_out_sum', 'directed_1hop_in_f16_mean', 'directed_1hop_in_f12_std', 'financial_hub_f10_mean', 'directed_1hop_in_f14_mean', 'financial_hub_f0_min', 'directed_2hop_in_f12_mean', 'directed_2hop_in_f9_std', 'directed_2hop_in_f10_mean', 'undirected_2hop_f6_mean', 'directed_1hop_out_f0_mean', 'directed_1hop_out_f1_min', 'undirected_1hop_f16_std', 'mixed_propagation_f8_min', 'directed_1hop_in_f1_min', 'financial_hub_f1_std', 'out_degree', 'mixed_propagation_f1_max', 'directed_2hop_in_f2_mean', 'undirected_2hop_f4_max', 'financial_hub_f18_std', 'undirected_2hop_f5_std', 'directed_1hop_out_f16_std', 'mixed_propagation_f11_std', 'transaction_chain_f15_std', 'directed_2hop_in_f14_mean', 'directed_2hop_in_f7_min', 'financial_hub_f2_std', 'td_type0_7', 'mixed_propagation_f9_min', 'directed_1hop_in_f9_min', 'undirected_1hop_f1_mean', 'directed_2hop_in_f8_min', 'directed_1hop_in_f5_min', 'financial_hub_f8_std', 'transaction_chain_f2_max', 'mixed_propagation_f3_max', 'directed_1hop_in_f15_std', 'directed_2hop_out_f15_std', 'transaction_chain_f11_std', 'financial_hub_f1_max', 'directed_1hop_out_f11_mean', 'transaction_chain_f7_mean', 'financial_hub_f9_min', 'directed_2hop_in_f3_mean', 'directed_1hop_in_f11_mean', 'directed_2hop_in_f17_mean', 'mixed_propagation_f14_mean', 'directed_2hop_in_f1_mean', 'undirected_2hop_f15_std', 'undirected_1hop_f1_min', 'undirected_1hop_f10_mean', 'financial_hub_f4_min', 'directed_2hop_out_f14_mean', 'directed_2hop_in_f0_mean', 'directed_1hop_out_f1_mean', 'directed_2hop_in_f2_std', 'directed_1hop_in_f1_std', 'financial_hub_f1_mean', 'directed_2hop_out_f4_min', 'td_in_15', 'transaction_chain_f10_mean', 'directed_2hop_in_f0_min', 'directed_1hop_in_f18_std', 'mixed_propagation_f16_std', 'directed_1hop_out_f14_mean', 'directed_1hop_out_f8_min', 'undirected_2hop_f9_min', 'directed_1hop_in_f10_mean', 'transaction_chain_f0_std', 'directed_1hop_out_f9_mean', 'financial_hub_f9_mean', 'directed_1hop_in_f9_mean', 'undirected_2hop_f7_std', 'td_15', 'directed_1hop_in_f8_min', 'financial_hub_f12_std', 'transaction_chain_f16_std', 'undirected_1hop_f14_mean', 'directed_2hop_in_f6_mean', 'directed_2hop_in_f11_mean', 'undirected_2hop_f12_mean', 'financial_hub_f12_mean', 'directed_2hop_out_f11_std', 'transaction_chain_f13_mean', 'financial_hub_f6_mean', 'directed_2hop_in_f18_std', 'financial_hub_f9_max', 'directed_1hop_in_f17_std', 'td_in_7', 'directed_2hop_in_f2_max', 'directed_1hop_in_f2_max', 'financial_hub_f2_mean', 'undirected_1hop_f11_mean', 'directed_1hop_out_f15_std', 'mixed_propagation_f3_mean', 'financial_hub_f19_mean', 'directed_2hop_in_f17_std', 'directed_1hop_in_f17_mean', 'transaction_chain_f5_min', 'financial_hub_f10_std', 'directed_1hop_out_f18_std', 'undirected_2hop_f2_mean', 'financial_hub_f9_std', 'directed_2hop_in_f1_std', 'directed_2hop_out_f0_mean', 'directed_2hop_in_f0_max', 'directed_2hop_in_f1_min', 'directed_1hop_in_f11_std', 'core_business_f18_std', 'undirected_2hop_f17_mean', 'directed_1hop_out_f16_mean', 'directed_1hop_in_f4_mean', 'directed_1hop_in_f1_mean', 'directed_1hop_in_f0_min', 'directed_2hop_in_f3_max', 'directed_2hop_out_f9_min', 'td_type0_15', 'directed_1hop_in_f8_std', 'directed_2hop_in_f6_std', 'financial_hub_f13_std', 'financial_hub_f8_min', 'transaction_chain_f2_mean', 'directed_1hop_in_f14_std', 'directed_1hop_out_f18_mean', 'directed_2hop_in_f15_mean', 'directed_1hop_in_f8_mean']
        
    elif dataset=='dgraphfin':
        COLS_FEA_SLC = ['fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11', 'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6', 'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean', 'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0', 'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8', 'directed_1hop_out_interval_mean', 'directed_1hop_in_f16_mean', 'directed_1hop_in_f7_max', 'undirected_1hop_f16_mean', 'undirected_2hop_f12_max', 'directed_1hop_out_timestamp_std', 'undirected_1hop_interval_mean', 'directed_1hop_out_timestamp_mean', 'fea_out_edge_timestamp_diff_1_target_mean', 'fea_x_4', 'fea_out_edge_type_min', 'undirected_1hop_f1_min', 'fea_in_edge_type_mean', 'directed_1hop_in_f7_mean', 'undirected_1hop_interval_min', 'fea_out_edge_timestamp_diff_1_target_min', 'undirected_2hop_interval_std', 'undirected_2hop_f16_max', 'fea_out_edge_timestamp_diff_1_target_max', 'undirected_1hop_timestamp_mean', 'fea_out_neighbor_fea_x_13_max', 'undirected_2hop_interval_min', 'undirected_1hop_f7_max', 'directed_1hop_out_f16_mean', 'fea_in_edge_timestamp_diff_1_source_max', 'directed_1hop_in_f15_mean', 'undirected_2hop_f1_min', 'undirected_1hop_timestamp_range', 'directed_1hop_out_interval_max', 'undirected_2hop_f15_max', 'directed_1hop_out_type_5_max', 'fea_out_edge_type_mode', 'mixed_propagation_f4_max', 'fea_out_edge_timestamp_diff_1_target_mode', 'undirected_1hop_type_4_min', 'directed_1hop_in_f16_max', 'undirected_2hop_f1_mean', 'undirected_1hop_f11_max', 'undirected_2hop_f1_std', 'degree_ratio', 'directed_1hop_out_interval_min', 'undirected_1hop_f7_mean', 'fea_x_3', 'fea_in_edge_type_max', 'directed_1hop_in_f13_mean', 'active_days_out', 'undirected_2hop_f9_max', 'undirected_2hop_f13_max', 'undirected_1hop_type_5_std', 'fea_out_edge_type_std', 'undirected_1hop_timestamp_std', 'fea_x_12', 'undirected_1hop_f12_max', 'undirected_2hop_timestamp_min', 'undirected_1hop_f1_std', 'undirected_2hop_interval_max', 'undirected_1hop_interval_max', 'undirected_1hop_f13_mean', 'fea_x_7', 'undirected_1hop_timestamp_max', 'directed_1hop_in_f13_max', 'undirected_2hop_timestamp_mean', 'undirected_1hop_interval_std', 'directed_1hop_in_f15_max', 'undirected_2hop_f2_std', 'directed_1hop_in_f7_min', 'transaction_chain_interval_mean', 'directed_1hop_out_type_6_max', 'directed_2hop_out_type_4_std', 'undirected_2hop_type_5_std', 'fea_out_neighbor_fea_x_1_max', 'undirected_2hop_f11_max', 'undirected_1hop_f14_max', 'undirected_1hop_f9_max', 'undirected_1hop_f1_mean', 'transaction_chain_f16_mean', 'undirected_2hop_interval_mean', 'financial_hub_timestamp_min', 'undirected_2hop_f11_std', 'fea_x_16', 'undirected_1hop_f1_max', 'fea_x_9', 'undirected_2hop_type_4_range', 'directed_1hop_in_f12_max', 'directed_1hop_in_f16_min', 'fea_out_neighbor_fea_x_1_min', 'mixed_propagation_f15_max', 'directed_1hop_out_type_5_min', 'directed_1hop_out_f15_mean', 'mixed_propagation_f16_max', 'undirected_2hop_timestamp_max', 'directed_2hop_out_type_4_min', 'mixed_propagation_f11_max', 'directed_1hop_out_f1_max', 'total_degree', 'undirected_1hop_f12_std', 'directed_2hop_out_type_6_range', 'undirected_1hop_f13_std', 'directed_1hop_in_timestamp_min', 'undirected_1hop_f10_mean', 'undirected_2hop_timestamp_std', 'directed_1hop_out_type_5_mean', 'undirected_1hop_f5_max', 'undirected_2hop_f7_max', 'transaction_chain_f7_std', 'mixed_propagation_timestamp_min', 'directed_2hop_out_timestamp_mean', 'fea_out_neighbor_fea_x_1_mean', 'directed_2hop_in_f9_mean', 'directed_2hop_in_f1_mean', 'transaction_chain_type_6_min', 'undirected_2hop_f4_std']
        

    COLS_KEEP = ['label'] + COLS_FEA_SLC
    return COLS_KEEP

# == sync, corrected by elderman == @elder man
def parse_args():
    """Parsing command line parameters"""
    parser = argparse.ArgumentParser(description='Train CatBoost model')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Data set name (e.g. amazon)')
    parser.add_argument('--data_dir', type=str, default='../feature_split',
                       help='Feature Data Directory (default: ./feature spit)')
    parser.add_argument('--data_split_dir', type=str, default='../data_split',
                       help='Data Split Directory (default: ./data spit)')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Model Output Directory (default: ./models)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Cross-validation discount (default: 5)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random Feed (default: 2022)')
    
    args = parser.parse_args()
    return args


# == sync, corrected by elderman == @elder man
def load_train_data(data_dir, label_file, dataset_name):
    """Load training data and labels"""
    print("=" * 60)
    print(f"Load training data - dataset: {dataset_name}")
    print("=" * 60)
    
    # Load all pkl files
    df_node = pd.read_pickle(os.path.join(data_dir, 'df_node_enhanced_mk.pkl'))
    feature_new = pd.read_pickle(os.path.join(data_dir, 'feature_new.pkl'))
    feature_null = pd.read_pickle(os.path.join(data_dir, 'feature_null.pkl'))
    
    print(f"  df_node_enhanced_mk: {df_node.shape}")
    print(f"  feature_new: {feature_new.shape}")
    print(f"  feature_null: {feature_null.shape}")
    
    # Before combining features, detect duplicate columns between files
    all_dfs = [df_node, feature_new, feature_null]
    df_names = ['df_node', 'feature_new', 'feature_null']
    
    print("\\n Test duplicate columns between data files...")
    
    # Check for duplicate columns for each file
    for name, df in zip(df_names, all_dfs):
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            print(f"    ⚠️ {name} There's a repeat column inside.: {duplicate_cols[:5]}...")
    
    # Check repeat columns between files
    all_columns = []
    for name, df in zip(df_names, all_dfs):
        all_columns.extend([(col, name) for col in df.columns])
    
    from collections import Counter
    col_counter = Counter([col for col, _ in all_columns])
    inter_file_duplicates = {col: count for col, count in col_counter.items() if count > 1}
    
    if inter_file_duplicates:
        print(f"    ⚠️ Repeat columns between files: {list(inter_file_duplicates.keys())[:10]}...")
        print(f"    Discovery {len(inter_file_duplicates)} Repeat columns between files")
    else:
        print("No repetition column between files")
    
    # Merge Features
    df_train = pd.concat(all_dfs, axis=1)
    
    print(f"\n  Merged Feature Data: {df_train.shape}")
    
    # Test the merged repeat column
    duplicate_cols_after = df_train.columns[df_train.columns.duplicated()].tolist()
    if duplicate_cols_after:
        print(f"  ⚠️ After MergeDataFrameYeah. {len(duplicate_cols_after)} Repeat Columns: {duplicate_cols_after[:10]}...")
        
        # Solutions: Add suffix for repeat columns
        print("Processing duplicate columns...")
        new_columns = []
        col_count = {}
        for col in df_train.columns:
            if col in col_count:
                col_count[col] += 1
                new_col = f"{col}_{col_count[col]}"
            else:
                col_count[col] = 0
                new_col = col
            new_columns.append(new_col)
        df_train.columns = new_columns
        print(f"  ✓ Repeated columns processed，Final Columns: {len(df_train.columns)}")
    else:
        print(f"  ✓ After MergeDataFrameNo Repeat Row")
    
    # Load tabs from npz file
    print(f"\nLoad Tab Data: {label_file}")
    label_np = np.load(label_file)
    labels = label_np['y']
    print(f"  Tab Data Shape: {labels.shape}")
    print(f"  Number of positive samples: {labels.sum()}")
    print(f"  Positive sample ratio: {labels.mean():.4f}")
    
    # Add Tab to DataFrame
    df_train['label'] = labels
    
    print(f"  Final Training Data Shape: {df_train.shape}")
    
    return df_train


# == sync, corrected by elderman == @elder man
def create_feature_safe(df, feature_name, expression, existing_set):
    """Secure creation features and avoid repetition of names"""
    if feature_name not in existing_set:
        df[feature_name] = expression
        existing_set.add(feature_name)
        return True
    else:
        print(f"  Skip repeat feature: {feature_name}")
        return False

# == sync, corrected by elderman == @elder man
def create_cross_features_amazon(df):
    """Create cross-group features (avoid duplication with existing feature names)
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
    # == sync, corrected by elderman == @elder man
    print("\n" + "=" * 60)
    print("Debug Information - Check Current Features")
    print("=" * 60)

    # Time-related cross-cutting features
    if 'fea_x_19' in existing_features and 'fea_x_12' in existing_features:
        if create_feature_safe(df, 'time_related_feature', 
                              df['fea_x_19'] * df['fea_x_12'], existing_features):
            new_features.append('time_related_feature')
    
    # Cross-cutting features associated with the structure of the figure
    if 'fea_out_neighbor_fea_x_0_max' in existing_features and 'fea_out_neighbor_fea_x_7_max' in existing_features:
        if create_feature_safe(df, 'graph_structure_feature',
                              df['fea_out_neighbor_fea_x_0_max'] + df['fea_out_neighbor_fea_x_7_max'], existing_features):
            new_features.append('graph_structure_feature')
    
    # Cross-cutting features associated with type distribution
    if 'td_type_ratio_7d_1' in existing_features and 'td_type2_30' in existing_features:
        if create_feature_safe(df, 'type_distribution_feature',
                              df['td_type_ratio_7d_1'] * df['td_type2_30'], existing_features):
            new_features.append('type_distribution_feature')
    
    # Cross feature of the border type associated with the time interval
    if 'fea_out_edge_type_mean' in existing_features and 'directed_1hop_in_type_0_count' in existing_features:
        if create_feature_safe(df, 'edge_type_time_interval_feature',
                              df['fea_out_edge_type_mean'] * df['directed_1hop_in_type_0_count'], existing_features):
            new_features.append('edge_type_time_interval_feature')
    
    # Cross-cutting features of risk communication and chart structure
    if 'mixed_propagation_type_4_count' in existing_features and 'directed_1hop_out_type_4_count' in existing_features:
        if create_feature_safe(df, 'risk_propagation_feature',
                              df['mixed_propagation_type_4_count'] + df['directed_1hop_out_type_4_count'], existing_features):
            new_features.append('risk_propagation_feature')
    
    # I'm not sure I'm going to do it.
    if 'undirected_1hop_type_0_count' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        if create_feature_safe(df, 'hop_behavior_contrast',
                              df['undirected_1hop_type_0_count'] - df['undirected_2hop_type_0_count'], existing_features):
            new_features.append('hop_behavior_contrast')
    
    # Operation-specific anomaly detection features
    if 'transaction_chain_f0_mean' in existing_features and 'directed_1hop_out_type_0_count' in existing_features:
        if create_feature_safe(df, 'business_specific_anomaly',
                              df['transaction_chain_f0_mean'] * df['directed_1hop_out_type_0_count'], existing_features):
            new_features.append('business_specific_anomaly')
    
    # Based on the cross-cutting features of the box
    if 'fea_x_2' in existing_features:
        if create_feature_safe(df, 'fea_x_2_bin',
                              pd.cut(df['fea_x_2'], bins=[-2, -1, 0, 1, 3, 10, 112], labels=False), existing_features):
            new_features.append('fea_x_2_bin')
    
    if 'fea_x_6' in existing_features:
        if create_feature_safe(df, 'fea_x_6_bin',
                              pd.cut(df['fea_x_6'], bins=[-1, 0, 100, 200, 400, 822], labels=False), existing_features):
            new_features.append('fea_x_6_bin')
    
    if 'fea_x_2_bin' in existing_features and 'fea_x_6_bin' in existing_features:
        if create_feature_safe(df, 'bin_interaction',
                              df['fea_x_2_bin'] * df['fea_x_6_bin'], existing_features):
            new_features.append('bin_interaction')
    
    # Statistical cluster features
    if 'undirected_1hop_f5_std' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        if create_feature_safe(df, 'statistical_combination',
                              df['undirected_1hop_f5_std'] + df['undirected_2hop_type_0_count'], existing_features):
            new_features.append('statistical_combination')
    
    # Original identity stability cross-cutting feature
    if 'fea_x_2' in existing_features and 'fea_x_6' in existing_features:
        if create_feature_safe(df, 'feature_stability',
                              df['fea_x_2'] * df['fea_x_6'], existing_features):
            new_features.append('feature_stability')
    
    print(f"  Created successfully {len(new_features)} Cross-cutting features")
    print(f"  New feature list: {new_features}")
    
    return df, new_features


import pandas as pd

def create_cross_features_yelpchi(df):
    """Create cross-assembly feature (strictly only Top130 feature)
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    
    # 1. Cross-cutting of time and type features
    if 'td_type_ratio_7d_1' in df.columns and 'td_type1_30' in df.columns:
        df['td_type_ratio_7d_1_td_type1_30'] = df['td_type_ratio_7d_1'] * df['td_type1_30']
        new_features.append('td_type_ratio_7d_1_td_type1_30')
    
    # 2. Maximum and average value of neighbourhood features Bad
    if 'fea_out_neighbor_fea_x_0_max' in df.columns and 'fea_out_neighbor_fea_x_0_mean' in df.columns:
        df['fea_out_neighbor_fea_x_0_diff'] = df['fea_out_neighbor_fea_x_0_max'] - df['fea_out_neighbor_fea_x_0_mean']
        new_features.append('fea_out_neighbor_fea_x_0_diff')
    
    # 3. Product of features
    if 'fea_x_18' in df.columns and 'fea_x_19' in df.columns:
        df['fea_x_18_x_19_product'] = df['fea_x_18'] * df['fea_x_19']
        new_features.append('fea_x_18_x_19_product')
    
    # 4. Portfolio of risk transmission features
    if 'risk_nei_out_mean' in df.columns and 'risk_nei_in_mean' in df.columns:
        df['risk_nei_diff_mean'] = df['risk_nei_out_mean'] - df['risk_nei_in_mean']
        new_features.append('risk_nei_diff_mean')
    
    # 5. Features and features
    if 'fea_x_13' in df.columns and 'fea_x_14' in df.columns:
        df['fea_x_13_x_14_sum'] = df['fea_x_13'] + df['fea_x_14']
        new_features.append('fea_x_13_x_14_sum')
    
    # 6. Neighbourly statistical profiles
    if 'fea_out_neighbor_fea_x_1_mean' in df.columns and 'fea_out_neighbor_fea_x_1_max' in df.columns:
        df['fea_out_neighbor_x_1_mean_max_ratio'] = df['fea_out_neighbor_fea_x_1_mean'] / (df['fea_out_neighbor_fea_x_1_max'] + 1e-6)
        new_features.append('fea_out_neighbor_x_1_mean_max_ratio')
    
    # 7. Time window characterization interaction
    if 'td_type1_15' in df.columns and 'td_type1_7' in df.columns:
        df['td_type1_15_7_diff'] = df['td_type1_15'] - df['td_type1_7']
        new_features.append('td_type1_15_7_diff')
    
    # 8. Convergence of image dissemination features
    if 'directed_1hop_in_f0_mean' in df.columns and 'directed_1hop_out_f0_mean' in df.columns:
        df['directed_1hop_in_out_ratio'] = df['directed_1hop_in_f0_mean'] / (df['directed_1hop_out_f0_mean'] + 1e-6)
        new_features.append('directed_1hop_in_out_ratio')
    
    # 9. Multiple-jump feature comparisons
    if 'directed_2hop_in_f0_mean' in df.columns and 'directed_1hop_in_f0_mean' in df.columns:
        df['directed_2hop_1hop_in_ratio'] = df['directed_2hop_in_f0_mean'] / (df['directed_1hop_in_f0_mean'] + 1e-6)
        new_features.append('directed_2hop_1hop_in_ratio')
    
    # 10. Total risk features
    if 'risk_nei_out_sum' in df.columns and 'risk_nei_in_sum' in df.columns:
        df['risk_nei_total_sum'] = df['risk_nei_out_sum'] + df['risk_nei_in_sum']
        new_features.append('risk_nei_total_sum')
    
    # Ensure that only 30 new features are returned
    if len(new_features) > 30:
        new_features = new_features[:30]
    
    print(f"Created successfully {len(new_features)} Cross-cutting features")
    print("New feature list:", new_features)
    
    return df, new_features

def create_cross_features_tfinance(df):
    """Create cross-cutting and grouping features
Input: DataFrame (includes 130 original features)
Output: DataFrame (with 30 additional combination features), new feature list"""
    result_df = df.copy()
    new_features = []
    
    # 1. Group of risk features
    result_df['risk_nei_in_out_ratio'] = np.where(
        result_df['risk_nei_out_sum'] > 0,
        result_df['risk_nei_in_sum'] / (result_df['risk_nei_out_sum'] + 1e-6),
        0
    )
    new_features.append('risk_nei_in_out_ratio')
    
    result_df['risk_nei_total'] = result_df['risk_nei_in_sum'] + result_df['risk_nei_out_sum']
    new_features.append('risk_nei_total')
    
    result_df['risk_nei_mean_ratio'] = np.where(
        result_df['risk_nei_out_mean'] > 0,
        result_df['risk_nei_in_mean'] / (result_df['risk_nei_out_mean'] + 1e-6),
        0
    )
    new_features.append('risk_nei_mean_ratio')
    
    # 2. Time window feature combination
    result_df['td_ratio_15_7'] = np.where(
        result_df['td_in_7'] > 0,
        result_df['td_in_15'] / (result_df['td_in_7'] + 1e-6),
        0
    )
    new_features.append('td_ratio_15_7')
    
    result_df['td_type0_ratio_15_7'] = np.where(
        result_df['td_type0_7'] > 0,
        result_df['td_type0_15'] / (result_df['td_type0_7'] + 1e-6),
        0
    )
    new_features.append('td_type0_ratio_15_7')
    
    # 3. Features of structure levels
    result_df['hop_ratio_2hop_1hop_in_min'] = np.where(
        result_df['directed_1hop_in_f1_min'] > 0,
        result_df['directed_2hop_in_f1_min'] / (result_df['directed_1hop_in_f1_min'] + 1e-6),
        0
    )
    new_features.append('hop_ratio_2hop_1hop_in_min')
    
    result_df['hop_ratio_2hop_1hop_out_min'] = np.where(
        result_df['directed_1hop_out_f1_min'] > 0,
        result_df['directed_2hop_out_f9_min'] / (result_df['directed_1hop_out_f1_min'] + 1e-6),
        0
    )
    new_features.append('hop_ratio_2hop_1hop_out_min')
    
    # 4. Retrospective versus no-directional features
    result_df['directed_undirected_min_ratio'] = np.where(
        result_df['undirected_1hop_f1_min'] > 0,
        result_df['directed_1hop_in_f1_min'] / (result_df['undirected_1hop_f1_min'] + 1e-6),
        0
    )
    new_features.append('directed_undirected_min_ratio')
    
    result_df['directed_undirected_1hop_mean_ratio'] = np.where(
        result_df['undirected_1hop_f1_mean'] > 0,
        result_df['directed_1hop_in_f1_mean'] / (result_df['undirected_1hop_f1_mean'] + 1e-6),
        0
    )
    new_features.append('directed_undirected_1hop_mean_ratio')

    return result_df, new_features


def create_cross_features_dgraphfin(df):
    """Create 16 refined cross-cutting features based on the importance of character ranking
The importance of identity: the importance of the features ahead
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    
    # Only the top 20 most important features are combined (based on the ranking provided)
    top_features = [
        'fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11',
        'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6',
        'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean',
        'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0',
        'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8'
    ]
    
    # Core time window analysis: interval between last active time and earliest transaction time
    if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
        df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
        new_features.append('active_window_span')
    
    # 2. High-risk transaction density: ratio of maximum transaction value to time span
    if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
        new_features.append('risk_transaction_density')
    
    # 3. Recent surprise trading: the last active time is close + the value of the transaction is huge
    if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
        new_features.append('recent_burst_risk')
    
    # 4. Transaction time anomalies: unusual time frames for the disbursement of accounts + unstable transaction intervals
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
        new_features.append('timing_anomaly_score')
    
    # 5. Core identity mix: interaction of the three most important features x
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
        new_features.append('core_feature_interaction')
    
    # 6. Account inactivity risk: abnormal combination of inactive days and transaction amounts
    if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
        df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
        new_features.append('dormant_high_value_risk')
    
    # 7. Transaction time concentration: the relationship between the minimum and maximum value at the time of payment
    if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
        df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
        new_features.append('out_timestamp_concentration')
    
    # 8. Edge type anomalies: difference between average and maximum
    if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
        df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
        new_features.append('edge_type_discrepancy')
    
    # 9. Minimum time stamp abnormal: unusual pattern of the earliest transaction time
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
        df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
        new_features.append('first_interaction_gap')
    
    # 10. Weighted combination of feature x: weights based on ranking of importance
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        # Distribution of weights: the more important the weight of features
        df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                    df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                    df['fea_x_8'] * 0.05)
        new_features.append('weighted_x_features')
    
    # 11. Transaction stability rating: a combination of time-frame and interval stability
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
        new_features.append('transaction_stability')
    
    # 12. High-frequency risk trading: small time horizon but high transaction value
    if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
        new_features.append('high_freq_high_value')
    
    # 13. Account activity anomalies: conflict between last active and inactive days
    if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
        df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
        new_features.append('activity_inconsistency')
    
    # 14. Anomalous time patterns: interaction between the earliest transaction time and the time frame for the disbursement of accounts
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
        new_features.append('early_wide_spread_risk')
    
    # 15. Multiple risk overlaps: abnormal patterns combining the most important features
    if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                     np.log1p(df['undirected_1hop_f16_max']))
        new_features.append('multi_risk_composite')
    
    # 16. Core time features interactive: a combination of the most important time features
    if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                     (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
        new_features.append('core_time_interaction')
    
    # Make sure it happens to be 16.
    if len(new_features) > 16:
        # Retain the first 16 features (in order of definition)
        features_to_keep = new_features[:16]
        features_to_remove = new_features[16:]
        
        for feat in features_to_remove:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)
        
        new_features = features_to_keep
    
    print(f"Create Based on Importance Sorting {len(new_features)} A precise cross-cutting feature")
    print("New feature list (based on pre-material 20 feature combinations):")
    for i, feat in enumerate(new_features, 1):
        print(f"{i:2d}. {feat}")
    
    return df, new_features


def create_cross_features(df_, dataname_):
    if dataname_ == 'amazon':
        return create_cross_features_amazon(df_)
    elif dataname_ == 'yelpchi':
        return create_cross_features_yelpchi(df_)
    elif dataname_ == 'tfinance':
        return create_cross_features_tfinance(df_)
    elif dataname_ == 'dgraphfin':
        return create_cross_features_dgraphfin(df_)
    


# == sync, corrected by elderman == @elder man
def train_with_cv(df_train, cols_fea, new_features, output_dir, dataset_name, n_folds=5, random_seed=2022):
    """Training CatBoost model using a 50-percent cross-validation

Return:
- oof pred: Out-of-Fold forecast results
- Moders: list of 50% off model
-cv Metrics: cross-validation of indicators
- features fea crosss:"""
    print("\n" + "=" * 60)
    print(f"Start{n_folds}Cross-check training - dataset: {dataset_name}")
    print("=" * 60)
    
    # Preparation features and labels (duplication of duplicate features)
    print("\n" + "=" * 60)
    print("Prepare features and labels...")
    print("=" * 60)
    
    # Detecting duplicates between cols fea and new features
    cols_fea_set = set(cols_fea)
    duplicate_features = [f for f in new_features if f in cols_fea_set]
    if duplicate_features:
        print(f"  ⚠️ Found {len(duplicate_features)} A signature in cols_fea Existing: {duplicate_features[:10]}...")
    
    # Add only new features without repetition
    new_features_unique = [f for f in new_features if f not in cols_fea_set]
    print(f"  new_features Original Number: {len(new_features)}")
    print(f"  new_features We're going to reload.: {len(new_features_unique)}")
    
    # Merge feature list
    cols_fea_cross = cols_fea + new_features_unique
    print(f"  Number of final features: {len(cols_fea_cross)}")
    
    # Repeat to detect repetition in final list
    from collections import Counter
    final_col_counts = Counter(cols_fea_cross)
    final_duplicates = {col: count for col, count in final_col_counts.items() if count > 1}
    if final_duplicates:
        print(f"  ⚠️ The final feature list is still duplicated: {final_duplicates}")
    else:
        print(f"  ✓ Final feature list does not repeat")
    
    x_train = df_train[cols_fea_cross].reset_index(drop=True)
    y_train = df_train['label'].reset_index(drop=True)
    
    print(f"\nNumber of training samples: {len(x_train)}")
    print(f"Positive sample ratio: {y_train.mean():.4f}")
    
    # Initialize
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    oof_pred = np.zeros((len(x_train), 2))
    models = []
    cv_auc = []
    cv_f1 = []
    
    mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"\nInitial Memory: {mem_start:.2f} GiB")
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train, y_train)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split Data
        trn_x, trn_y = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
        val_x, val_y = x_train.iloc[va_idx], y_train.iloc[va_idx]
        
        print(f"  Training Set: {len(trn_x)}, Authentication Set: {len(val_x)}")
        
        # Create Pool Object
        train_pool = Pool(trn_x, label=trn_y)
        valid_pool = Pool(val_x, label=val_y)
        
        # CatBoost model configuration
        model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.06,
            depth=5,
            l2_leaf_reg=3,
            border_count=254,
            bootstrap_type='Bernoulli',
            subsample=0.85,
            sampling_frequency='PerTree',
            grow_policy='SymmetricTree',
            random_strength=1.5,
            auto_class_weights='Balanced',
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='CPU',
            early_stopping_rounds=200,
            verbose=100,
            random_seed=random_seed
        )
        
        # Training model
        print(f"\n  Start training. Fold {fold + 1}...")
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        
        # Validation set projections
        val_pred = model.predict_proba(val_x)
        oof_pred[va_idx] = val_pred
        
        # Calculate validation set indicators
        val_pos_prob = val_pred[:, 1]
        val_prd_lbl = (val_pos_prob > 0.5).astype(int)
        true_binary = (val_y == 1).astype(int)
        
        fold_auc = roc_auc_score(true_binary, val_pos_prob)
        fold_f1 = f1_score(true_binary, val_prd_lbl)
        cv_auc.append(fold_auc)
        cv_f1.append(fold_f1)
        
        print(f"\n  Fold {fold + 1} - AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}")
        
        # Save Model
        model_path = os.path.join(output_dir, f'catboost_fold_{fold + 1}.cbm')
        model.save_model(model_path)
        models.append(model)
        print(f"  Model saved: {model_path}")
        
        # Memory Clearing
        del train_pool, valid_pool, model, val_pred, trn_x, trn_y, val_x, val_y
        gc.collect()
        
        mem_current = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f"  Memory Usage: {mem_current:.2f} GiB")
    
    # Calculate overall OOF indicators
    oof_pos_prob = oof_pred[:, 1]
    oof_prd_lbl = (oof_pos_prob > 0.5).astype(int)
    oof_auc = roc_auc_score(y_train, oof_pos_prob)
    oof_f1 = f1_score(y_train, oof_prd_lbl)
    
    print("\n" + "=" * 60)
    print("Summary of cross-validation results")
    print("=" * 60)
    print(f"DeclinesAUC: {[f'{auc:.4f}' for auc in cv_auc]}")
    print(f"AUC Mean ± Std: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print(f"DeclinesF1: {[f'{f1:.4f}' for f1 in cv_f1]}")
    print(f"F1 Mean ± Std: {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
    print(f"OOF AUC: {oof_auc:.4f}")
    print(f"OOF F1: {oof_f1:.4f}")
    
    cv_metrics = {
        'cv_auc': cv_auc,
        'cv_f1': cv_f1,
        'oof_auc': oof_auc,
        'oof_f1': oof_f1,
        'auc_mean': np.mean(cv_auc),
        'auc_std': np.std(cv_auc),
        'f1_mean': np.mean(cv_f1),
        'f1_std': np.std(cv_f1)
    }
    
    return oof_pred, cv_metrics, models, cols_fea_cross


# == sync, corrected by elderman == @elder man
def main():
    # Parsing command line parameters
    args = parse_args()
    
    # Set global variables
    global DATASET_NAME, DATA_DIR, LABEL_FILE, OUTPUT_DIR, N_FOLDS, RANDOM_SEED
    
    DATASET_NAME = args.dataset
    DATA_DIR = os.path.join(args.data_dir, DATASET_NAME, 'train')
    LABEL_FILE = os.path.join(args.data_split_dir, f'{DATASET_NAME}_train.npz')
    OUTPUT_DIR = os.path.join(args.output_dir, DATASET_NAME)
    N_FOLDS = args.n_folds
    RANDOM_SEED = args.seed
    
    print("\n" + "=" * 60)
    print(f"CatBoost {N_FOLDS}Cross-check training")
    print(f"dataset: {DATASET_NAME}")
    print("=" * 60)
    print(f"Identity Directory: {DATA_DIR}")
    print(f"Tag File: {LABEL_FILE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Cross-validation discounts: {N_FOLDS}")
    print(f"Random Feeds: {RANDOM_SEED}")
    print("=" * 60)
    
    # Ensure that the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load training data and labelling
    df_train = load_train_data(DATA_DIR, LABEL_FILE, DATASET_NAME)
    
    # 2. Screening features based on columns specified by the user
    print("\n" + "=" * 60)
    print("Filter feature by column specified by user...")
    print("=" * 60)

    COLS_KEEP = get_COLS_KEEP(DATASET_NAME)

    
    # Check which specified columns exist in the data
    available_cols = [col for col in COLS_KEEP if col in df_train.columns]
    missing_cols = [col for col in COLS_KEEP if col not in df_train.columns]
    
    print(f"  User-Specify Columns: {len(COLS_KEEP)}")
    print(f"  Number of columns actually available: {len(available_cols)}")
    if missing_cols:
        print(f"  ⚠️ Missing Columns: {missing_cols[:10]}...")
    
    # Filter Data Box (only keep specified columns)
    df_train = df_train[available_cols].copy()
    print(f"  Data shape after filtering: {df_train.shape}")
    
    # 3. Create cross-cutting features (based on filtered features)
    print("\\n Create Cross feature...")
    df_train, new_features = create_cross_features(df_train, DATASET_NAME)
    
    # 4. Use of filtered feature columns as basic features
    cols_fea = [f for f in df_train.columns if f != 'label']
    print(f"\nNumber of basic features: {len(cols_fea)}")
    
    # 5. 50-percent cross-validation training
    oof_pred, cv_metrics, models, cols_fea_cross = train_with_cv(
        df_train, cols_fea, new_features, 
        output_dir=OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        n_folds=N_FOLDS, 
        random_seed=RANDOM_SEED
    )
    
    # 6. Preservation of training metadata
    metadata = {
        'dataset': DATASET_NAME,
        'cols_fea_cross': cols_fea_cross,
        'cv_metrics': cv_metrics,
        'n_folds': N_FOLDS,
        'random_seed': RANDOM_SEED,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'training_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nTraining metadata saved: {metadata_path}")
    
    # 7. Preservation of OOF projections
    oof_path = os.path.join(OUTPUT_DIR, 'oof_predictions.npy')
    np.save(oof_path, oof_pred)
    print(f"OOFForecast results saved: {oof_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nFile Generation:")
    print(f"  - Model File: {OUTPUT_DIR}/catboost_fold_*.cbm")
    print(f"  - Training metadata: {OUTPUT_DIR}/training_metadata.pkl")
    print(f"  - OOFprediction: {OUTPUT_DIR}/oof_predictions.npy")
    print(f"\nTraining indicators:")
    print(f"  AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
    print(f"  F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")


if __name__ == "__main__":
    main()
