#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 加载5折模型进行推理和评估
功能：
1. 加载测试数据
2. 应用与训练时相同的特征工程
3. 加载5折CatBoost模型进行集成预测
4. 计算并输出评估指标

使用方法:
python test.py --dataset amazon
或
python test.py -d amazon
"""

import numpy as np
import pandas as pd
import pickle
import os
import gc
import psutil
import argparse
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix

# ============== 全局配置变量 ==============
# 这些将根据命令行参数设置
DATASET_NAME = None
DATA_DIR = None
TEST_LABEL_FILE = None
MODEL_DIR = None
N_FOLDS = 5
RANDOM_SEED = 2022


# ============== 用户指定的特征列 ==============
def get_COLS_KEEP(dataset):
    """获取指定数据集需要保留的特征列"""
    if dataset == 'amazon':
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
        
    elif dataset == 'yelpchi':
        COLS_FEA_SLC = ['fea_x_18', 'fea_x_19', 'td_type_ratio_7d_1', 'fea_x_21', 'fea_x_13', 'td_type1_15', 
                       'fea_x_20', 'fea_out_neighbor_fea_x_5_mean', 'fea_out_neighbor_fea_x_0_mean', 'fea_x_14', 
                       'fea_x_0', 'td_type1_30', 'fea_x_16', 'fea_out_neighbor_fea_x_0_max', 'fea_x_1', 
                       'fea_x_23', 'fea_out_neighbor_fea_x_5_max', 'fea_x_26', 'risk_nei_out_mean', 
                       'td_type_ratio_in_7d_1', 'fea_out_neighbor_fea_x_1_max', 'fea_x_5', 'risk_nei_in_mean', 
                       'td_type1_7', 'fea_out_neighbor_fea_x_0_min', 'fea_x_15', 'fea_x_22', 
                       'fea_out_neighbor_fea_x_1_mean', 'fea_x_10', 'fea_x_25', 'fea_out_neighbor_fea_x_5_min', 
                       'fea_x_30', 'fea_x_8', 'fea_out_neighbor_fea_x_1_min', 'fea_x_17', 'fea_x_28', 
                       'fea_x_27', 'fea_out_neighbor_fea_x_8_max', 'fea_x_31', 'fea_in_edge_type_min', 
                       'fea_x_29', 'directed_1hop_in_f24_std', 'undirected_1hop_f62_std', 
                       'directed_2hop_out_f0_mean', 'directed_1hop_in_f6_min', 'fea_out_edge_type_min', 
                       'directed_1hop_in_f13_mean', 'fea_out_neighbor_fea_x_2_mean', 'mixed_propagation_f0_mean', 
                       'directed_1hop_in_f29_min', 'fea_x_2', 'fea_out_neighbor_fea_x_8_mean', 
                       'fea_out_neighbor_fea_x_8_min', 'fea_x_24', 'mixed_propagation_f6_mean', 
                       'fea_out_neighbor_fea_x_2_min', 'mixed_propagation_f16_std', 'directed_1hop_in_f63_std', 
                       'financial_hub_f5_mean', 'directed_2hop_out_f8_mean', 'directed_1hop_out_f31_std', 
                       'undirected_1hop_f63_std', 'undirected_1hop_f59_std', 'directed_1hop_in_f0_mean', 
                       'directed_1hop_in_f62_std', 'transaction_chain_f50_std', 'directed_2hop_out_f30_std', 
                       'directed_1hop_out_f7_std', 'fea_out_edge_type_nunique', 'mixed_propagation_f46_mean', 
                       'directed_1hop_out_f62_min', 'directed_2hop_in_f0_std', 'directed_1hop_out_f30_std', 
                       'undirected_1hop_f28_std', 'directed_2hop_in_f0_mean', 'mixed_propagation_f0_min', 
                       'directed_1hop_out_f24_std', 'directed_2hop_in_f0_min', 'directed_1hop_in_f1_mean', 
                       'directed_1hop_in_f58_std', 'directed_1hop_in_f31_std', 'mixed_propagation_f23_max', 
                       'directed_1hop_in_f28_std', 'transaction_chain_f10_std', 'transaction_chain_f0_std', 
                       'directed_1hop_in_f26_std', 'transaction_chain_f10_min', 'mixed_propagation_f25_std', 
                       'directed_1hop_out_f62_std', 'undirected_1hop_f27_std', 'directed_2hop_out_f6_mean', 
                       'mixed_propagation_f32_mean', 'directed_1hop_in_f7_mean', 'directed_2hop_in_f25_std', 
                       'mixed_propagation_f28_std', 'mixed_propagation_f9_std', 'directed_1hop_in_f11_mean', 
                       'directed_2hop_in_f8_std', 'transaction_chain_f62_std', 'directed_2hop_in_f42_std', 
                       'transaction_chain_f0_mean', 'mixed_propagation_f10_min', 'directed_2hop_in_f25_mean', 
                       'directed_2hop_in_f1_max', 'directed_2hop_in_f31_std', 'directed_1hop_out_f0_mean', 
                       'directed_2hop_in_f14_mean', 'directed_2hop_in_f10_std', 'directed_2hop_in_f24_mean', 
                       'directed_1hop_in_f10_mean', 'directed_1hop_in_f25_std', 'directed_1hop_in_f29_std', 
                       'fea_out_neighbor_fea_x_2_max', 'directed_1hop_out_f28_std', 'directed_1hop_in_f14_std', 
                       'directed_2hop_in_f62_std', 'directed_2hop_out_f6_min', 'directed_2hop_out_f23_max', 
                       'mixed_propagation_f0_max', 'financial_hub_f0_mean', 'directed_1hop_in_f23_mean', 
                       'fea_in_edge_type_nunique', 'directed_2hop_in_f24_std', 'directed_1hop_in_f8_std', 
                       'directed_1hop_in_f32_mean', 'transaction_chain_f9_mean', 'directed_2hop_out_f32_mean', 
                       'td_type_ratio_7d_2', 'directed_2hop_out_f6_std', 'mixed_propagation_f14_mean']
    
    elif dataset == 'tfinance':
        COLS_FEA_SLC = ['risk_nei_out_mean', 'risk_nei_in_mean', 'risk_nei_in_sum', 'risk_nei_out_sum', 
                       'directed_1hop_in_f16_mean', 'directed_1hop_in_f12_std', 'financial_hub_f10_mean', 
                       'directed_1hop_in_f14_mean', 'financial_hub_f0_min', 'directed_2hop_in_f12_mean', 
                       'directed_2hop_in_f9_std', 'directed_2hop_in_f10_mean', 'undirected_2hop_f6_mean', 
                       'directed_1hop_out_f0_mean', 'directed_1hop_out_f1_min', 'undirected_1hop_f16_std', 
                       'mixed_propagation_f8_min', 'directed_1hop_in_f1_min', 'financial_hub_f1_std', 
                       'out_degree', 'mixed_propagation_f1_max', 'directed_2hop_in_f2_mean', 
                       'undirected_2hop_f4_max', 'financial_hub_f18_std', 'undirected_2hop_f5_std', 
                       'directed_1hop_out_f16_std', 'mixed_propagation_f11_std', 'transaction_chain_f15_std', 
                       'directed_2hop_in_f14_mean', 'directed_2hop_in_f7_min', 'financial_hub_f2_std', 
                       'td_type0_7', 'mixed_propagation_f9_min', 'directed_1hop_in_f9_min', 
                       'undirected_1hop_f1_mean', 'directed_2hop_in_f8_min', 'directed_1hop_in_f5_min', 
                       'financial_hub_f8_std', 'transaction_chain_f2_max', 'mixed_propagation_f3_max', 
                       'directed_1hop_in_f15_std', 'directed_2hop_out_f15_std', 'transaction_chain_f11_std', 
                       'financial_hub_f1_max', 'directed_1hop_out_f11_mean', 'transaction_chain_f7_mean', 
                       'financial_hub_f9_min', 'directed_2hop_in_f3_mean', 'directed_1hop_in_f11_mean', 
                       'directed_2hop_in_f17_mean', 'mixed_propagation_f14_mean', 'directed_2hop_in_f1_mean', 
                       'undirected_2hop_f15_std', 'undirected_1hop_f1_min', 'undirected_1hop_f10_mean', 
                       'financial_hub_f4_min', 'directed_2hop_out_f14_mean', 'directed_2hop_in_f0_mean', 
                       'directed_1hop_out_f1_mean', 'directed_2hop_in_f2_std', 'directed_1hop_in_f1_std', 
                       'financial_hub_f1_mean', 'directed_2hop_out_f4_min', 'td_in_15', 
                       'transaction_chain_f10_mean', 'directed_2hop_in_f0_min', 'directed_1hop_in_f18_std', 
                       'mixed_propagation_f16_std', 'directed_1hop_out_f14_mean', 'directed_1hop_out_f8_min', 
                       'undirected_2hop_f9_min', 'directed_1hop_in_f10_mean', 'transaction_chain_f0_std', 
                       'directed_1hop_out_f9_mean', 'financial_hub_f9_mean', 'directed_1hop_in_f9_mean', 
                       'undirected_2hop_f7_std', 'td_15', 'directed_1hop_in_f8_min', 'financial_hub_f12_std', 
                       'transaction_chain_f16_std', 'undirected_1hop_f14_mean', 'directed_2hop_in_f6_mean', 
                       'directed_2hop_in_f11_mean', 'undirected_2hop_f12_mean', 'financial_hub_f12_mean', 
                       'directed_2hop_out_f11_std', 'transaction_chain_f13_mean', 'financial_hub_f6_mean', 
                       'directed_2hop_in_f18_std', 'financial_hub_f9_max', 'directed_1hop_in_f17_std', 
                       'td_in_7', 'directed_2hop_in_f2_max', 'directed_1hop_in_f2_max', 'financial_hub_f2_mean', 
                       'undirected_1hop_f11_mean', 'directed_1hop_out_f15_std', 'mixed_propagation_f3_mean', 
                       'financial_hub_f19_mean', 'directed_2hop_in_f17_std', 'directed_1hop_in_f17_mean', 
                       'transaction_chain_f5_min', 'financial_hub_f10_std', 'directed_1hop_out_f18_std', 
                       'undirected_2hop_f2_mean', 'financial_hub_f9_std', 'directed_2hop_in_f1_std', 
                       'directed_2hop_out_f0_mean', 'directed_2hop_in_f0_max', 'directed_2hop_in_f1_min', 
                       'directed_1hop_in_f11_std', 'core_business_f18_std', 'undirected_2hop_f17_mean', 
                       'directed_1hop_out_f16_mean', 'directed_1hop_in_f4_mean', 'directed_1hop_in_f1_mean', 
                       'directed_1hop_in_f0_min', 'directed_2hop_in_f3_max', 'directed_2hop_out_f9_min', 
                       'td_type0_15', 'directed_1hop_in_f8_std', 'directed_2hop_in_f6_std', 
                       'financial_hub_f13_std', 'financial_hub_f8_min', 'transaction_chain_f2_mean', 
                       'directed_1hop_in_f14_std', 'directed_1hop_out_f18_mean', 'directed_2hop_in_f15_mean', 
                       'directed_1hop_in_f8_mean']
        
    elif dataset == 'dgraphfin':
        COLS_FEA_SLC = ['fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11', 
                       'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6', 
                       'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean', 
                       'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0', 
                       'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8', 
                       'directed_1hop_out_interval_mean', 'directed_1hop_in_f16_mean', 'directed_1hop_in_f7_max', 
                       'undirected_1hop_f16_mean', 'undirected_2hop_f12_max', 'directed_1hop_out_timestamp_std', 
                       'undirected_1hop_interval_mean', 'directed_1hop_out_timestamp_mean', 
                       'fea_out_edge_timestamp_diff_1_target_mean', 'fea_x_4', 'fea_out_edge_type_min', 
                       'undirected_1hop_f1_min', 'fea_in_edge_type_mean', 'directed_1hop_in_f7_mean', 
                       'undirected_1hop_interval_min', 'fea_out_edge_timestamp_diff_1_target_min', 
                       'undirected_2hop_interval_std', 'undirected_2hop_f16_max', 
                       'fea_out_edge_timestamp_diff_1_target_max', 'undirected_1hop_timestamp_mean', 
                       'fea_out_neighbor_fea_x_13_max', 'undirected_2hop_interval_min', 'undirected_1hop_f7_max', 
                       'directed_1hop_out_f16_mean', 'fea_in_edge_timestamp_diff_1_source_max', 
                       'directed_1hop_in_f15_mean', 'undirected_2hop_f1_min', 'undirected_1hop_timestamp_range', 
                       'directed_1hop_out_interval_max', 'undirected_2hop_f15_max', 
                       'directed_1hop_out_type_5_max', 'fea_out_edge_type_mode', 'mixed_propagation_f4_max', 
                       'fea_out_edge_timestamp_diff_1_target_mode', 'undirected_1hop_type_4_min', 
                       'directed_1hop_in_f16_max', 'undirected_2hop_f1_mean', 'undirected_1hop_f11_max', 
                       'undirected_2hop_f1_std', 'degree_ratio', 'directed_1hop_out_interval_min', 
                       'undirected_1hop_f7_mean', 'fea_x_3', 'fea_in_edge_type_max', 
                       'directed_1hop_in_f13_mean', 'active_days_out', 'undirected_2hop_f9_max', 
                       'undirected_2hop_f13_max', 'undirected_1hop_type_5_std', 'fea_out_edge_type_std', 
                       'undirected_1hop_timestamp_std', 'fea_x_12', 'undirected_1hop_f12_max', 
                       'undirected_2hop_timestamp_min', 'undirected_1hop_f1_std', 'undirected_2hop_interval_max', 
                       'undirected_1hop_interval_max', 'undirected_1hop_f13_mean', 'fea_x_7', 
                       'undirected_1hop_timestamp_max', 'directed_1hop_in_f13_max', 'undirected_2hop_timestamp_mean', 
                       'undirected_1hop_interval_std', 'directed_1hop_in_f15_max', 'undirected_2hop_f2_std', 
                       'directed_1hop_in_f7_min', 'transaction_chain_interval_mean', 
                       'directed_1hop_out_type_6_max', 'directed_2hop_out_type_4_std', 
                       'undirected_2hop_type_5_std', 'fea_out_neighbor_fea_x_1_max', 'undirected_2hop_f11_max', 
                       'undirected_1hop_f14_max', 'undirected_1hop_f9_max', 'undirected_1hop_f1_mean', 
                       'transaction_chain_f16_mean', 'undirected_2hop_interval_mean', 'financial_hub_timestamp_min', 
                       'undirected_2hop_f11_std', 'fea_x_16', 'undirected_1hop_f1_max', 'fea_x_9', 
                       'undirected_2hop_type_4_range', 'directed_1hop_in_f12_max', 'directed_1hop_in_f16_min', 
                       'fea_out_neighbor_fea_x_1_min', 'mixed_propagation_f15_max', 
                       'directed_1hop_out_type_5_min', 'directed_1hop_out_f15_mean', 'mixed_propagation_f16_max', 
                       'undirected_2hop_timestamp_max', 'directed_2hop_out_type_4_min', 'mixed_propagation_f11_max', 
                       'directed_1hop_out_f1_max', 'total_degree', 'undirected_1hop_f12_std', 
                       'directed_2hop_out_type_6_range', 'undirected_1hop_f13_std', 
                       'directed_1hop_in_timestamp_min', 'undirected_1hop_f10_mean', 
                       'undirected_2hop_timestamp_std', 'directed_1hop_out_type_5_mean', 'undirected_1hop_f5_max', 
                       'undirected_2hop_f7_max', 'transaction_chain_f7_std', 'mixed_propagation_timestamp_min', 
                       'directed_2hop_out_timestamp_mean', 'fea_out_neighbor_fea_x_1_mean', 
                       'directed_2hop_in_f9_mean', 'directed_2hop_in_f1_mean', 'transaction_chain_type_6_min', 
                       'undirected_2hop_f4_std']
    else:
        raise ValueError(f"未知的数据集: {dataset}")
    
    COLS_KEEP = ['label'] + COLS_FEA_SLC
    return COLS_KEEP


# ============== 命令行参数解析 ==============
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试CatBoost模型')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='数据集名称 (如: amazon)')
    parser.add_argument('--data_dir', type=str, default='../feature_split',
                       help='特征数据目录 (默认: ../feature_split)')
    parser.add_argument('--data_split_dir', type=str, default='../data_split',
                       help='数据分割目录 (默认: ../data_split)')
    parser.add_argument('--model_dir', type=str, default='../models_ours',
                       help='模型目录 (默认: ./models)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='交叉验证折数 (默认: 5)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='随机种子 (默认: 2022)')
    
    args = parser.parse_args()
    return args


# ============== 安全创建特征函数 ==============
def create_feature_safe(df, feature_name, expression, existing_set):
    """安全创建特征，避免重复名称"""
    if feature_name not in existing_set:
        df[feature_name] = expression
        existing_set.add(feature_name)
        return True
    else:
        print(f"  跳过重复特征: {feature_name}")
        return False


# ============== Amazon 数据集交叉特征 ==============
def create_cross_features_amazon(df):
    """
    创建Amazon数据集的交叉组合特征
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
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


# ============== YelpChi 数据集交叉特征 ==============
def create_cross_features_yelpchi(df):
    """
    创建YelpChi数据集的交叉组合特征
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    
    # 时间特征与类型特征的交叉
    if 'td_type_ratio_7d_1' in df.columns and 'td_type1_30' in df.columns:
        df['td_type_ratio_7d_1_td_type1_30'] = df['td_type_ratio_7d_1'] * df['td_type1_30']
        new_features.append('td_type_ratio_7d_1_td_type1_30')
    
    # 邻居特征的最大值与平均值的差
    if 'fea_out_neighbor_fea_x_0_max' in df.columns and 'fea_out_neighbor_fea_x_0_mean' in df.columns:
        df['fea_out_neighbor_fea_x_0_diff'] = df['fea_out_neighbor_fea_x_0_max'] - df['fea_out_neighbor_fea_x_0_mean']
        new_features.append('fea_out_neighbor_fea_x_0_diff')
    
    # 特征的乘积
    if 'fea_x_18' in df.columns and 'fea_x_19' in df.columns:
        df['fea_x_18_x_19_product'] = df['fea_x_18'] * df['fea_x_19']
        new_features.append('fea_x_18_x_19_product')
    
    # 风险传播特征的组合
    if 'risk_nei_out_mean' in df.columns and 'risk_nei_in_mean' in df.columns:
        df['risk_nei_diff_mean'] = df['risk_nei_out_mean'] - df['risk_nei_in_mean']
        new_features.append('risk_nei_diff_mean')
    
    # 特征的和
    if 'fea_x_13' in df.columns and 'fea_x_14' in df.columns:
        df['fea_x_13_x_14_sum'] = df['fea_x_13'] + df['fea_x_14']
        new_features.append('fea_x_13_x_14_sum')
    
    # 邻居统计特征组合
    if 'fea_out_neighbor_fea_x_1_mean' in df.columns and 'fea_out_neighbor_fea_x_1_max' in df.columns:
        df['fea_out_neighbor_x_1_mean_max_ratio'] = df['fea_out_neighbor_fea_x_1_mean'] / (df['fea_out_neighbor_fea_x_1_max'] + 1e-6)
        new_features.append('fea_out_neighbor_x_1_mean_max_ratio')
    
    # 时间窗口特征交互
    if 'td_type1_15' in df.columns and 'td_type1_7' in df.columns:
        df['td_type1_15_7_diff'] = df['td_type1_15'] - df['td_type1_7']
        new_features.append('td_type1_15_7_diff')
    
    # 图传播特征的聚合
    if 'directed_1hop_in_f0_mean' in df.columns and 'directed_1hop_out_f0_mean' in df.columns:
        df['directed_1hop_in_out_ratio'] = df['directed_1hop_in_f0_mean'] / (df['directed_1hop_out_f0_mean'] + 1e-6)
        new_features.append('directed_1hop_in_out_ratio')
    
    # 多跳特征对比
    if 'directed_2hop_in_f0_mean' in df.columns and 'directed_1hop_in_f0_mean' in df.columns:
        df['directed_2hop_1hop_in_ratio'] = df['directed_2hop_in_f0_mean'] / (df['directed_1hop_in_f0_mean'] + 1e-6)
        new_features.append('directed_2hop_1hop_in_ratio')
    
    # 风险总和特征
    if 'risk_nei_out_sum' in df.columns and 'risk_nei_in_sum' in df.columns:
        df['risk_nei_total_sum'] = df['risk_nei_out_sum'] + df['risk_nei_in_sum']
        new_features.append('risk_nei_total_sum')
    
    print(f"  成功创建 {len(new_features)} 个交叉特征")
    print(f"  新特征列表: {new_features}")
    
    return df, new_features


# ============== TFinance 数据集交叉特征 ==============
def create_cross_features_tfinance(df):
    """
    创建TFinance数据集的交叉特征
    返回: (包含新特征的数据框, 新特征名称列表)
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
    
    print(f"  成功创建 {len(new_features)} 个交叉特征")
    print(f"  新特征列表: {new_features}")
    
    return result_df, new_features


# ============== DGraphFin 数据集交叉特征 ==============
def create_cross_features_dgraphfin(df):
    """
    基于特征重要性排序创建16个精炼交叉特征
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
    
    # 1. 核心时间窗口分析
    if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
        df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
        new_features.append('active_window_span')
    
    # 2. 高风险交易密度
    if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
        new_features.append('risk_transaction_density')
    
    # 3. 近期突击性交易
    if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
        new_features.append('recent_burst_risk')
    
    # 4. 交易时间异常
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
        new_features.append('timing_anomaly_score')
    
    # 5. 核心特征组合
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
        new_features.append('core_feature_interaction')
    
    # 6. 账户不活跃风险
    if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
        df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
        new_features.append('dormant_high_value_risk')
    
    # 7. 交易时间集中度
    if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
        df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
        new_features.append('out_timestamp_concentration')
    
    # 8. 边类型异常
    if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
        df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
        new_features.append('edge_type_discrepancy')
    
    # 9. 时间戳最小值异常
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
        df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
        new_features.append('first_interaction_gap')
    
    # 10. 特征x的加权组合
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                    df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                    df['fea_x_8'] * 0.05)
        new_features.append('weighted_x_features')
    
    # 11. 交易稳定性评分
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
        new_features.append('transaction_stability')
    
    # 12. 高频风险交易
    if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
        new_features.append('high_freq_high_value')
    
    # 13. 账户活性异常
    if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
        df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
        new_features.append('activity_inconsistency')
    
    # 14. 时间模式异常
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
        new_features.append('early_wide_spread_risk')
    
    # 15. 多重风险叠加
    if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                     np.log1p(df['undirected_1hop_f16_max']))
        new_features.append('multi_risk_composite')
    
    # 16. 核心时间特征交互
    if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                     (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
        new_features.append('core_time_interaction')
    
    print(f"  成功创建 {len(new_features)} 个交叉特征")
    print(f"  新特征列表: {new_features}")
    
    return df, new_features


# ============== 统一的交叉特征创建函数 ==============
def create_cross_features(df, dataset_name):
    """
    根据数据集名称选择对应的交叉特征创建函数
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    if dataset_name == 'amazon':
        return create_cross_features_amazon(df)
    elif dataset_name == 'yelpchi':
        return create_cross_features_yelpchi(df)
    elif dataset_name == 'tfinance':
        return create_cross_features_tfinance(df)
    elif dataset_name == 'dgraphfin':
        return create_cross_features_dgraphfin(df)
    else:
        print(f"  ⚠️ 未知数据集 {dataset_name}，跳过交叉特征创建")
        return df, []


# ============== 数据加载函数 ==============
def load_test_data(data_dir, label_file):
    """加载测试数据和标签"""
    print("=" * 60)
    print(f"加载测试数据...")
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
    df_test = pd.concat(all_dfs, axis=1)
    
    print(f"\n  合并后特征数据: {df_test.shape}")
    
    # 检测合并后的重复列
    duplicate_cols_after = df_test.columns[df_test.columns.duplicated()].tolist()
    if duplicate_cols_after:
        print(f"  ⚠️ 合并后DataFrame有 {len(duplicate_cols_after)} 个重复列: {duplicate_cols_after[:10]}...")
        
        # 解决方案：为重复列添加后缀
        print("  正在处理重复列...")
        new_columns = []
        col_count = {}
        for col in df_test.columns:
            if col in col_count:
                col_count[col] += 1
                new_col = f"{col}_{col_count[col]}"
            else:
                col_count[col] = 0
                new_col = col
            new_columns.append(new_col)
        df_test.columns = new_columns
        print(f"  ✓ 已处理重复列，最终列数: {len(df_test.columns)}")
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
    df_test['label'] = labels
    
    print(f"  最终测试数据形状: {df_test.shape}")
    
    return df_test


# ============== 模型加载和预测函数 ==============
def load_models_and_predict(df_test, cols_fea_cross, model_dir, n_folds=5):
    """
    加载5折模型并进行集成预测
    
    返回:
        - test_pred: 集成预测结果
        - fold_predictions: 各折预测结果
        - models: 加载的模型列表
        - y_test: 真实标签
    """
    print("\n" + "=" * 60)
    print("加载模型并进行预测...")
    print("=" * 60)
    
    # 准备测试数据
    x_test = df_test[cols_fea_cross].reset_index(drop=True)
    y_test = df_test['label'].reset_index(drop=True)
    
    print(f"\n测试集样本数: {len(x_test)}")
    print(f"特征数量: {len(cols_fea_cross)}")
    print(f"正样本比例: {y_test.mean():.4f}")
    
    # 初始化预测结果
    test_pred = np.zeros((len(x_test), 2))
    fold_predictions = []
    models = []
    
    mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"\n初始内存: {mem_start:.2f} GiB")
    
    for fold in range(1, n_folds + 1):
        print(f"\n加载模型 Fold {fold}/{n_folds}...")
        
        # 加载模型
        model_path = os.path.join(model_dir, f'catboost_fold_{fold}.cbm')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = CatBoostClassifier()
        model.load_model(model_path)
        models.append(model)
        
        print(f"  模型已加载: {model_path}")
        
        # 单折预测
        fold_pred = model.predict_proba(x_test)
        fold_predictions.append(fold_pred)
        
        # 累加预测结果（用于集成）
        test_pred += fold_pred / n_folds
        
        # 清理内存
        del model, fold_pred
        gc.collect()
    
    mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"\n预测完成，内存使用: {mem_end:.2f} GiB")
    
    return test_pred, fold_predictions, models, y_test


# ============== 评估指标计算函数 ==============
def evaluate_predictions(y_true, y_pred_proba, y_pred_proba_list=None, n_folds=5):
    """
    计算并输出评估指标
    
    参数:
        y_true: 真实标签
        y_pred_proba: 集成预测概率
        y_pred_proba_list: 各折预测概率列表
        n_folds: 折数
    """
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    
    # 集成预测结果
    y_pos_prob = y_pred_proba[:, 1]
    y_pred_label = (y_pos_prob > 0.5).astype(int)
    
    # 计算基础指标
    auc_score = roc_auc_score(y_true, y_pos_prob)
    f1 = f1_score(y_true, y_pred_label)
    ap_score = average_precision_score(y_true, y_pos_prob)
    
    print(f"\n【集成模型（5折平均）评估结果】")
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Average Precision (AP): {ap_score:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_label)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  混淆矩阵:")
    print(f"    TN: {tn}, FP: {fp}")
    print(f"    FN: {fn}, TP: {tp}")
    
    # 计算更多指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  特异性 (Specificity): {specificity:.4f}")
    
    # 各折独立评估
    if y_pred_proba_list is not None:
        print(f"\n【各折独立评估结果】")
        fold_aucs = []
        fold_f1s = []
        
        for fold, fold_pred in enumerate(y_pred_proba_list, 1):
            fold_pos_prob = fold_pred[:, 1]
            fold_pred_label = (fold_pos_prob > 0.5).astype(int)
            
            fold_auc = roc_auc_score(y_true, fold_pos_prob)
            fold_f1 = f1_score(y_true, fold_pred_label)
            
            fold_aucs.append(fold_auc)
            fold_f1s.append(fold_f1)
            
            print(f"  Fold {fold}: AUC = {fold_auc:.4f}, F1 = {fold_f1:.4f}")
        
        print(f"\n  各折AUC Mean ± Std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"  各折F1 Mean ± Std: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    
    # 汇总所有指标
    metrics = {
        'auc': auc_score,
        'f1': f1,
        'ap': ap_score,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    return metrics


# ============== 保存预测结果函数 ==============
def save_predictions(y_test, test_pred, fold_predictions, output_dir):
    """保存预测结果"""
    print("\n" + "=" * 60)
    print("保存预测结果...")
    print("=" * 60)
    
    # 保存集成预测结果
    ensemble_path = os.path.join(output_dir, 'test_ensemble_predictions.npy')
    np.save(ensemble_path, test_pred)
    print(f"  集成预测结果已保存: {ensemble_path}")
    
    # 保存各折预测结果
    for fold, fold_pred in enumerate(fold_predictions, 1):
        fold_path = os.path.join(output_dir, f'test_fold_{fold}_predictions.npy')
        np.save(fold_path, fold_pred)
        print(f"  Fold {fold} 预测结果已保存: {fold_path}")
    
    # 保存真实标签和预测概率
    results_df = pd.DataFrame({
        'true_label': y_test.values,
        'pred_probability': test_pred[:, 1],
        'pred_label': (test_pred[:, 1] > 0.5).astype(int)
    })
    results_path = os.path.join(output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  预测结果表格已保存: {results_path}")
    
    return ensemble_path, results_path


# ============== 主函数 ==============
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置全局变量
    global DATASET_NAME, DATA_DIR, TEST_LABEL_FILE, MODEL_DIR, N_FOLDS, RANDOM_SEED
    
    DATASET_NAME = args.dataset
    DATA_DIR = os.path.join(args.data_dir, DATASET_NAME, 'test')
    TEST_LABEL_FILE = os.path.join(args.data_split_dir, f'{DATASET_NAME}_test.npz')
    MODEL_DIR = os.path.join(args.model_dir, DATASET_NAME)
    N_FOLDS = args.n_folds
    RANDOM_SEED = args.seed
    
    print("\n" + "=" * 60)
    print("CatBoost 5折模型测试与评估")
    print("=" * 60)
    print(f"数据集: {DATASET_NAME}")
    print(f"特征目录: {DATA_DIR}")
    print(f"标签文件: {TEST_LABEL_FILE}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"交叉验证折数: {N_FOLDS}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)
    
    # 1. 检查模型文件是否存在
    print("\n检查模型文件...")
    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f'catboost_fold_{fold}.cbm')
        if os.path.exists(model_path):
            print(f"  ✓ Fold {fold} 模型存在")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 2. 加载训练元数据（获取特征列）
    print("\n加载训练元数据...")
    metadata_path = os.path.join(MODEL_DIR, 'training_metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        cols_fea_cross = metadata['cols_fea_cross']
        print(f"  已加载特征配置，共 {len(cols_fea_cross)} 个特征")
    else:
        raise FileNotFoundError(f"训练元数据不存在: {metadata_path}")
    
    # 3. 加载测试数据
    df_test = load_test_data(DATA_DIR, TEST_LABEL_FILE)
    
    # 4. 先根据用户指定的列筛选特征（与训练时一致）
    print("\n" + "=" * 60)
    print("根据用户指定的列筛选特征...")
    print("=" * 60)
    
    COLS_KEEP = get_COLS_KEEP(DATASET_NAME)
    
    # 检查哪些指定的列存在于数据中
    available_cols = [col for col in COLS_KEEP if col in df_test.columns]
    missing_cols = [col for col in COLS_KEEP if col not in df_test.columns]
    
    print(f"  用户指定列数: {len(COLS_KEEP)}")
    print(f"  实际可用列数: {len(available_cols)}")
    if missing_cols:
        print(f"  ⚠️ 缺失的列: {missing_cols[:10]}...")
    
    # 筛选数据框（只保留指定的列）
    df_test = df_test[available_cols].copy()
    print(f"  筛选后数据形状: {df_test.shape}")
    
    # 5. 创建交叉特征（必须与训练时一致）
    print("\n创建交叉特征...")
    df_test, new_features = create_cross_features(df_test, DATASET_NAME)
    
    # 6. 加载模型并进行预测
    test_pred, fold_predictions, models, y_test = load_models_and_predict(
        df_test, cols_fea_cross, model_dir=MODEL_DIR, n_folds=N_FOLDS
    )
    
    # 7. 计算评估指标
    metrics = evaluate_predictions(y_test, test_pred, fold_predictions, n_folds=N_FOLDS)
    
    # 8. 保存预测结果
    ensemble_path, results_path = save_predictions(
        y_test, test_pred, fold_predictions, MODEL_DIR
    )
    
    # 9. 输出最终汇总
    print("\n" + "=" * 60)
    print("测试评估完成 - 最终汇总")
    print("=" * 60)
    print(f"\n测试集样本数: {len(y_test)}")
    print(f"正样本数: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"负样本数: {len(y_test) - y_test.sum()} ({(1-y_test.mean())*100:.2f}%)")
    print(f"\n【核心指标】")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  Average Precision: {metrics['ap']:.4f}")
    print(f"\n【输出文件】")
    print(f"  集成预测: {ensemble_path}")
    print(f"  结果表格: {results_path}")
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
