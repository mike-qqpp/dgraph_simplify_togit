#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - 使用5折交叉验证训练CatBoost模型
功能：
1. 加载训练数据
2. 创建交叉特征
3. 进行5折交叉验证训练
4. 保存5折模型

使用方法:
python train.py --dataset amazon
或
python train.py -d amazon
"""

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

# ============== 配置参数 ==============
# 这些将根据命令行参数设置
DATA_DIR = None  # 将根据数据集名称设置
LABEL_FILE = None  # 将根据数据集名称设置
OUTPUT_DIR = None  # 将根据数据集名称设置
DATASET_NAME = None  # 数据集名称

N_FOLDS = 5
RANDOM_SEED = 2022

# ============== 用户指定的特征列 ==============
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

# ============== 命令行参数解析 ==============
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练CatBoost模型')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='数据集名称 (如: amazon)')
    parser.add_argument('--data_dir', type=str, default='../feature_split',
                       help='特征数据目录 (默认: ../feature_split)')
    parser.add_argument('--data_split_dir', type=str, default='../data_split',
                       help='数据分割目录 (默认: ../data_split)')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='模型输出目录 (默认: ./models)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='交叉验证折数 (默认: 5)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='随机种子 (默认: 2022)')
    
    args = parser.parse_args()
    return args


# ============== 数据加载函数 ==============
def load_train_data(data_dir, label_file, dataset_name):
    """加载训练数据和标签"""
    print("=" * 60)
    print(f"加载训练数据 - 数据集: {dataset_name}")
    print("=" * 60)
    
    # 加载所有pkl文件
    df_node = pd.read_pickle(os.path.join(data_dir, 'df_node_enhanced_mk.pkl'))
    feature_new = pd.read_pickle(os.path.join(data_dir, 'feature_new.pkl'))
    feature_null = pd.read_pickle(os.path.join(data_dir, 'feature_null.pkl'))
    
    print(f"  df_node_enhanced_mk: {df_node.shape}")
    print(f"  feature_new: {feature_new.shape}")
    print(f"  feature_null: {feature_null.shape}")
    
    # 合并特征前，先检测各文件间的重复列
    all_dfs = [df_node, feature_new, feature_null]
    df_names = ['df_node', 'feature_new', 'feature_null']
    
    print("\n  检测各数据文件间的重复列...")
    
    # 检查每个文件是否有重复列
    for name, df in zip(df_names, all_dfs):
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            print(f"    ⚠️ {name} 内部有重复列: {duplicate_cols[:5]}...")
    
    # 检查文件间的重复列
    all_columns = []
    for name, df in zip(df_names, all_dfs):
        all_columns.extend([(col, name) for col in df.columns])
    
    from collections import Counter
    col_counter = Counter([col for col, _ in all_columns])
    inter_file_duplicates = {col: count for col, count in col_counter.items() if count > 1}
    
    if inter_file_duplicates:
        print(f"    ⚠️ 文件间重复列: {list(inter_file_duplicates.keys())[:10]}...")
        print(f"    共发现 {len(inter_file_duplicates)} 个文件间重复列")
    else:
        print("    ✓ 文件间无重复列")
    
    # 合并特征
    df_train = pd.concat(all_dfs, axis=1)
    
    print(f"\n  合并后特征数据: {df_train.shape}")
    
    # 检测合并后的重复列
    duplicate_cols_after = df_train.columns[df_train.columns.duplicated()].tolist()
    if duplicate_cols_after:
        print(f"  ⚠️ 合并后DataFrame有 {len(duplicate_cols_after)} 个重复列: {duplicate_cols_after[:10]}...")
        
        # 解决方案：为重复列添加后缀
        print("  正在处理重复列...")
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
        print(f"  ✓ 已处理重复列，最终列数: {len(df_train.columns)}")
    else:
        print(f"  ✓ 合并后DataFrame无重复列")
    
    # 从npz文件加载标签
    print(f"\n加载标签数据: {label_file}")
    label_np = np.load(label_file)
    labels = label_np['y']
    print(f"  标签数据形状: {labels.shape}")
    print(f"  正样本数量: {labels.sum()}")
    print(f"  正样本比例: {labels.mean():.4f}")
    
    # 将标签添加到DataFrame
    df_train['label'] = labels
    
    print(f"  最终训练数据形状: {df_train.shape}")
    
    return df_train


# ============== 调试信息结束 ==============
def create_feature_safe(df, feature_name, expression, existing_set):
    """安全创建特征，避免重复名称"""
    if feature_name not in existing_set:
        df[feature_name] = expression
        existing_set.add(feature_name)
        return True
    else:
        print(f"  跳过重复特征: {feature_name}")
        return False

# ============== 交叉特征创建函数 ==============
def create_cross_features_amazon(df):
    """
    创建交叉组合特征（避免与现有特征名称重复）
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
    # ============== 调试信息 ==============
    print("\n" + "=" * 60)
    print("调试信息 - 检查现有特征")
    print("=" * 60)

    # 时间相关的交叉特征
    if 'fea_x_19' in existing_features and 'fea_x_12' in existing_features:
        if create_feature_safe(df, 'time_related_feature', 
                              df['fea_x_19'] * df['fea_x_12'], existing_features):
            new_features.append('time_related_feature')
    
    # 图结构相关的交叉特征
    if 'fea_out_neighbor_fea_x_0_max' in existing_features and 'fea_out_neighbor_fea_x_7_max' in existing_features:
        if create_feature_safe(df, 'graph_structure_feature',
                              df['fea_out_neighbor_fea_x_0_max'] + df['fea_out_neighbor_fea_x_7_max'], existing_features):
            new_features.append('graph_structure_feature')
    
    # 类型分布相关的交叉特征
    if 'td_type_ratio_7d_1' in existing_features and 'td_type2_30' in existing_features:
        if create_feature_safe(df, 'type_distribution_feature',
                              df['td_type_ratio_7d_1'] * df['td_type2_30'], existing_features):
            new_features.append('type_distribution_feature')
    
    # 边类型与时间间隔相关的交叉特征
    if 'fea_out_edge_type_mean' in existing_features and 'directed_1hop_in_type_0_count' in existing_features:
        if create_feature_safe(df, 'edge_type_time_interval_feature',
                              df['fea_out_edge_type_mean'] * df['directed_1hop_in_type_0_count'], existing_features):
            new_features.append('edge_type_time_interval_feature')
    
    # 风险传播与图结构交叉特征
    if 'mixed_propagation_type_4_count' in existing_features and 'directed_1hop_out_type_4_count' in existing_features:
        if create_feature_safe(df, 'risk_propagation_feature',
                              df['mixed_propagation_type_4_count'] + df['directed_1hop_out_type_4_count'], existing_features):
            new_features.append('risk_propagation_feature')
    
    # 多跳邻居行为对比特征
    if 'undirected_1hop_type_0_count' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        if create_feature_safe(df, 'hop_behavior_contrast',
                              df['undirected_1hop_type_0_count'] - df['undirected_2hop_type_0_count'], existing_features):
            new_features.append('hop_behavior_contrast')
    
    # 业务特定的异常检测特征
    if 'transaction_chain_f0_mean' in existing_features and 'directed_1hop_out_type_0_count' in existing_features:
        if create_feature_safe(df, 'business_specific_anomaly',
                              df['transaction_chain_f0_mean'] * df['directed_1hop_out_type_0_count'], existing_features):
            new_features.append('business_specific_anomaly')
    
    # 基于分箱的交叉特征
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
    
    # 统计量组合特征
    if 'undirected_1hop_f5_std' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        if create_feature_safe(df, 'statistical_combination',
                              df['undirected_1hop_f5_std'] + df['undirected_2hop_type_0_count'], existing_features):
            new_features.append('statistical_combination')
    
    # 原始特征稳定性交叉特征
    if 'fea_x_2' in existing_features and 'fea_x_6' in existing_features:
        if create_feature_safe(df, 'feature_stability',
                              df['fea_x_2'] * df['fea_x_6'], existing_features):
            new_features.append('feature_stability')
    
    print(f"  成功创建 {len(new_features)} 个交叉特征")
    print(f"  新特征列表: {new_features}")
    
    return df, new_features


import pandas as pd

def create_cross_features_yelpchi(df):
    """
    创建交叉组合特征（严格仅使用top130特征）
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    
    # 1. 时间特征与类型特征的交叉
    if 'td_type_ratio_7d_1' in df.columns and 'td_type1_30' in df.columns:
        df['td_type_ratio_7d_1_td_type1_30'] = df['td_type_ratio_7d_1'] * df['td_type1_30']
        new_features.append('td_type_ratio_7d_1_td_type1_30')
    
    # 2. 邻居特征的最大值与平均值的差
    if 'fea_out_neighbor_fea_x_0_max' in df.columns and 'fea_out_neighbor_fea_x_0_mean' in df.columns:
        df['fea_out_neighbor_fea_x_0_diff'] = df['fea_out_neighbor_fea_x_0_max'] - df['fea_out_neighbor_fea_x_0_mean']
        new_features.append('fea_out_neighbor_fea_x_0_diff')
    
    # 3. 特征的乘积
    if 'fea_x_18' in df.columns and 'fea_x_19' in df.columns:
        df['fea_x_18_x_19_product'] = df['fea_x_18'] * df['fea_x_19']
        new_features.append('fea_x_18_x_19_product')
    
    # 4. 风险传播特征的组合
    if 'risk_nei_out_mean' in df.columns and 'risk_nei_in_mean' in df.columns:
        df['risk_nei_diff_mean'] = df['risk_nei_out_mean'] - df['risk_nei_in_mean']
        new_features.append('risk_nei_diff_mean')
    
    # 5. 特征的和
    if 'fea_x_13' in df.columns and 'fea_x_14' in df.columns:
        df['fea_x_13_x_14_sum'] = df['fea_x_13'] + df['fea_x_14']
        new_features.append('fea_x_13_x_14_sum')
    
    # 6. 邻居统计特征组合
    if 'fea_out_neighbor_fea_x_1_mean' in df.columns and 'fea_out_neighbor_fea_x_1_max' in df.columns:
        df['fea_out_neighbor_x_1_mean_max_ratio'] = df['fea_out_neighbor_fea_x_1_mean'] / (df['fea_out_neighbor_fea_x_1_max'] + 1e-6)
        new_features.append('fea_out_neighbor_x_1_mean_max_ratio')
    
    # 7. 时间窗口特征交互
    if 'td_type1_15' in df.columns and 'td_type1_7' in df.columns:
        df['td_type1_15_7_diff'] = df['td_type1_15'] - df['td_type1_7']
        new_features.append('td_type1_15_7_diff')
    
    # 8. 图传播特征的聚合
    if 'directed_1hop_in_f0_mean' in df.columns and 'directed_1hop_out_f0_mean' in df.columns:
        df['directed_1hop_in_out_ratio'] = df['directed_1hop_in_f0_mean'] / (df['directed_1hop_out_f0_mean'] + 1e-6)
        new_features.append('directed_1hop_in_out_ratio')
    
    # 9. 多跳特征对比
    if 'directed_2hop_in_f0_mean' in df.columns and 'directed_1hop_in_f0_mean' in df.columns:
        df['directed_2hop_1hop_in_ratio'] = df['directed_2hop_in_f0_mean'] / (df['directed_1hop_in_f0_mean'] + 1e-6)
        new_features.append('directed_2hop_1hop_in_ratio')
    
    # 10. 风险总和特征
    if 'risk_nei_out_sum' in df.columns and 'risk_nei_in_sum' in df.columns:
        df['risk_nei_total_sum'] = df['risk_nei_out_sum'] + df['risk_nei_in_sum']
        new_features.append('risk_nei_total_sum')
    
    # 确保只返回最多30个新特征
    if len(new_features) > 30:
        new_features = new_features[:30]
    
    print(f"成功创建 {len(new_features)} 个交叉特征")
    print("新特征列表:", new_features)
    
    return df, new_features

def create_cross_features_tfinance(df):
    """
    创建交叉特征和组合特征
    输入: DataFrame (包含原始130个特征)
    输出: DataFrame (新增30个组合特征), new_feature_list
    """
    result_df = df.copy()
    new_features = []
    
    # 1. 风险特征组合
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
    
    # 2. 时间窗口特征组合
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
    
    # 3. 图结构层次特征
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
    
    # 4. 有向与无向图特征对比
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
    """
    基于特征重要性排序创建16个精炼交叉特征
    特征重要性：越靠前的特征越重要
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    
    # 只取前20个最重要的特征进行组合（基于提供的排序）
    top_features = [
        'fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11',
        'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6',
        'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean',
        'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0',
        'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8'
    ]
    
    # 1. 核心时间窗口分析：最后活跃时间与最早交易时间的间隔
    if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
        df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
        new_features.append('active_window_span')
    
    # 2. 高风险交易密度：最大交易额与时间跨度的比值
    if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
        new_features.append('risk_transaction_density')
    
    # 3. 近期突击性交易：最后活跃时间很近 + 交易额巨大
    if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
        new_features.append('recent_burst_risk')
    
    # 4. 交易时间异常：出账时间范围异常 + 交易间隔不稳定
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
        new_features.append('timing_anomaly_score')
    
    # 5. 核心特征组合：最重要的3个特征x的交互
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
        new_features.append('core_feature_interaction')
    
    # 6. 账户不活跃风险：不活跃天数与交易额的异常组合
    if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
        df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
        new_features.append('dormant_high_value_risk')
    
    # 7. 交易时间集中度：出账时间最小值和最大值的相对关系
    if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
        df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
        new_features.append('out_timestamp_concentration')
    
    # 8. 边类型异常：均值与最大值的差异
    if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
        df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
        new_features.append('edge_type_discrepancy')
    
    # 9. 时间戳最小值异常：最早交易时间的异常模式
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
        df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
        new_features.append('first_interaction_gap')
    
    # 10. 特征x的加权组合：基于重要性排序的权重
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        # 权重分配：重要性越高的特征权重越大
        df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                    df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                    df['fea_x_8'] * 0.05)
        new_features.append('weighted_x_features')
    
    # 11. 交易稳定性评分：时间范围与间隔稳定性的综合
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
        new_features.append('transaction_stability')
    
    # 12. 高频风险交易：时间跨度小但交易额大
    if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
        new_features.append('high_freq_high_value')
    
    # 13. 账户活性异常：最后活跃与不活跃天数的矛盾
    if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
        df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
        new_features.append('activity_inconsistency')
    
    # 14. 时间模式异常：最早交易时间与出账时间范围的交互
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
        new_features.append('early_wide_spread_risk')
    
    # 15. 多重风险叠加：结合多个最重要特征的异常模式
    if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                     np.log1p(df['undirected_1hop_f16_max']))
        new_features.append('multi_risk_composite')
    
    # 16. 核心时间特征交互：最重要时间特征的组合
    if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                     (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
        new_features.append('core_time_interaction')
    
    # 确保正好16个特征
    if len(new_features) > 16:
        # 保留前16个特征（按定义的顺序）
        features_to_keep = new_features[:16]
        features_to_remove = new_features[16:]
        
        for feat in features_to_remove:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)
        
        new_features = features_to_keep
    
    print(f"基于重要性排序创建 {len(new_features)} 个精炼交叉特征")
    print("新特征列表（基于重要性前20的特征组合）:")
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
    


# ============== 5折交叉验证训练 ==============
def train_with_cv(df_train, cols_fea, new_features, output_dir, dataset_name, n_folds=5, random_seed=2022):
    """
    使用5折交叉验证训练CatBoost模型
    
    返回:
        - oof_pred: Out-of-Fold预测结果
        - models: 5折模型的列表
        - cv_metrics: 交叉验证指标
        - cols_fea_cross: 使用的特征列
    """
    print("\n" + "=" * 60)
    print(f"开始{n_folds}折交叉验证训练 - 数据集: {dataset_name}")
    print("=" * 60)
    
    # 准备特征和标签（避免重复特征）
    print("\n" + "=" * 60)
    print("准备特征和标签...")
    print("=" * 60)
    
    # 检测 cols_fea 和 new_features 之间的重复
    cols_fea_set = set(cols_fea)
    duplicate_features = [f for f in new_features if f in cols_fea_set]
    if duplicate_features:
        print(f"  ⚠️ 发现 {len(duplicate_features)} 个特征在 cols_fea 中已存在: {duplicate_features[:10]}...")
    
    # 只添加不重复的新特征
    new_features_unique = [f for f in new_features if f not in cols_fea_set]
    print(f"  new_features 原始数量: {len(new_features)}")
    print(f"  new_features 去重后数量: {len(new_features_unique)}")
    
    # 合并特征列表
    cols_fea_cross = cols_fea + new_features_unique
    print(f"  最终特征数量: {len(cols_fea_cross)}")
    
    # 再次检测最终列表中的重复
    from collections import Counter
    final_col_counts = Counter(cols_fea_cross)
    final_duplicates = {col: count for col, count in final_col_counts.items() if count > 1}
    if final_duplicates:
        print(f"  ⚠️ 最终特征列表仍有重复: {final_duplicates}")
    else:
        print(f"  ✓ 最终特征列表无重复")
    
    x_train = df_train[cols_fea_cross].reset_index(drop=True)
    y_train = df_train['label'].reset_index(drop=True)
    
    print(f"\n训练样本数: {len(x_train)}")
    print(f"正样本比例: {y_train.mean():.4f}")
    
    # 初始化
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    oof_pred = np.zeros((len(x_train), 2))
    models = []
    cv_auc = []
    cv_f1 = []
    
    mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"\n初始内存: {mem_start:.2f} GiB")
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train, y_train)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # 分割数据
        trn_x, trn_y = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
        val_x, val_y = x_train.iloc[va_idx], y_train.iloc[va_idx]
        
        print(f"  训练集: {len(trn_x)}, 验证集: {len(val_x)}")
        
        # 创建Pool对象
        train_pool = Pool(trn_x, label=trn_y)
        valid_pool = Pool(val_x, label=val_y)
        
        # CatBoost模型配置
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
        
        # 训练模型
        print(f"\n  开始训练 Fold {fold + 1}...")
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        
        # 验证集预测
        val_pred = model.predict_proba(val_x)
        oof_pred[va_idx] = val_pred
        
        # 计算验证集指标
        val_pos_prob = val_pred[:, 1]
        val_prd_lbl = (val_pos_prob > 0.5).astype(int)
        true_binary = (val_y == 1).astype(int)
        
        fold_auc = roc_auc_score(true_binary, val_pos_prob)
        fold_f1 = f1_score(true_binary, val_prd_lbl)
        cv_auc.append(fold_auc)
        cv_f1.append(fold_f1)
        
        print(f"\n  Fold {fold + 1} - AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}")
        
        # 保存模型
        model_path = os.path.join(output_dir, f'catboost_fold_{fold + 1}.cbm')
        model.save_model(model_path)
        models.append(model)
        print(f"  模型已保存: {model_path}")
        
        # 内存清理
        del train_pool, valid_pool, model, val_pred, trn_x, trn_y, val_x, val_y
        gc.collect()
        
        mem_current = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f"  内存使用: {mem_current:.2f} GiB")
    
    # 计算总体OOF指标
    oof_pos_prob = oof_pred[:, 1]
    oof_prd_lbl = (oof_pos_prob > 0.5).astype(int)
    oof_auc = roc_auc_score(y_train, oof_pos_prob)
    oof_f1 = f1_score(y_train, oof_prd_lbl)
    
    print("\n" + "=" * 60)
    print("交叉验证结果汇总")
    print("=" * 60)
    print(f"各折AUC: {[f'{auc:.4f}' for auc in cv_auc]}")
    print(f"AUC Mean ± Std: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print(f"各折F1: {[f'{f1:.4f}' for f1 in cv_f1]}")
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


# ============== 主函数 ==============
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置全局变量
    global DATASET_NAME, DATA_DIR, LABEL_FILE, OUTPUT_DIR, N_FOLDS, RANDOM_SEED
    
    DATASET_NAME = args.dataset
    DATA_DIR = os.path.join(args.data_dir, DATASET_NAME, 'train')
    LABEL_FILE = os.path.join(args.data_split_dir, f'{DATASET_NAME}_train.npz')
    OUTPUT_DIR = os.path.join(args.output_dir, DATASET_NAME)
    N_FOLDS = args.n_folds
    RANDOM_SEED = args.seed
    
    print("\n" + "=" * 60)
    print(f"CatBoost {N_FOLDS}折交叉验证训练")
    print(f"数据集: {DATASET_NAME}")
    print("=" * 60)
    print(f"特征目录: {DATA_DIR}")
    print(f"标签文件: {LABEL_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"交叉验证折数: {N_FOLDS}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载训练数据和标签
    df_train = load_train_data(DATA_DIR, LABEL_FILE, DATASET_NAME)
    
    # 2. 先根据用户指定的列筛选特征
    print("\n" + "=" * 60)
    print("根据用户指定的列筛选特征...")
    print("=" * 60)

    COLS_KEEP = get_COLS_KEEP(DATASET_NAME)

    
    # 检查哪些指定的列存在于数据中
    available_cols = [col for col in COLS_KEEP if col in df_train.columns]
    missing_cols = [col for col in COLS_KEEP if col not in df_train.columns]
    
    print(f"  用户指定列数: {len(COLS_KEEP)}")
    print(f"  实际可用列数: {len(available_cols)}")
    if missing_cols:
        print(f"  ⚠️ 缺失的列: {missing_cols[:10]}...")
    
    # 筛选数据框（只保留指定的列）
    df_train = df_train[available_cols].copy()
    print(f"  筛选后数据形状: {df_train.shape}")
    
    # 3. 创建交叉特征（在筛选后的特征基础上）
    print("\n创建交叉特征...")
    df_train, new_features = create_cross_features(df_train, DATASET_NAME)
    
    # 4. 使用筛选后的特征列作为基础特征
    cols_fea = [f for f in df_train.columns if f != 'label']
    print(f"\n基础特征数量: {len(cols_fea)}")
    
    # 5. 5折交叉验证训练
    oof_pred, cv_metrics, models, cols_fea_cross = train_with_cv(
        df_train, cols_fea, new_features, 
        output_dir=OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        n_folds=N_FOLDS, 
        random_seed=RANDOM_SEED
    )
    
    # 6. 保存训练元数据
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
    print(f"\n训练元数据已保存: {metadata_path}")
    
    # 7. 保存OOF预测
    oof_path = os.path.join(OUTPUT_DIR, 'oof_predictions.npy')
    np.save(oof_path, oof_pred)
    print(f"OOF预测结果已保存: {oof_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - 模型文件: {OUTPUT_DIR}/catboost_fold_*.cbm")
    print(f"  - 训练元数据: {OUTPUT_DIR}/training_metadata.pkl")
    print(f"  - OOF预测: {OUTPUT_DIR}/oof_predictions.npy")
    print(f"\n训练指标:")
    print(f"  AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
    print(f"  F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")


if __name__ == "__main__":
    main()