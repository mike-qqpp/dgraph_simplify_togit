#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Script - Load a 50-percent model for reasoning and evaluation
Function:
1. Load test data
2. Application of the same features as during training
3. Loading of the CatBoost model for integrated prediction
Calculation and output of assessment indicators

Usage method:
I don't know, python test.
or
I'm sorry, Python test."""

import numpy as np
import pandas as pd
import pickle
import os
import gc
import psutil
import argparse
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix

# == sync, corrected by elderman == @elder man
# These will be set according to the command line parameters.
DATASET_NAME = None
DATA_DIR = None
TEST_LABEL_FILE = None
MODEL_DIR = None
N_FOLDS = 5
RANDOM_SEED = 2022


# == sync, corrected by elderman == @elder man
def get_COLS_KEEP(dataset):
    """Feature column to retain to acquire the specified data set"""
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
        raise ValueError(f"Unknown data set: {dataset}")
    
    COLS_KEEP = ['label'] + COLS_FEA_SLC
    return COLS_KEEP


# == sync, corrected by elderman == @elder man
def parse_args():
    """Parsing command line parameters"""
    parser = argparse.ArgumentParser(description='Test CatBoost model')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Data set name (e.g. amazon)')
    parser.add_argument('--data_dir', type=str, default='../feature_split',
                       help='Feature Data Directory (default: ./feature spit)')
    parser.add_argument('--data_split_dir', type=str, default='../data_split',
                       help='Data Split Directory (default: ./data spit)')
    parser.add_argument('--model_dir', type=str, default='../models_ours',
                       help='Model directory (default: ./models)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Cross-validation discount (default: 5)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random Feed (default: 2022)')
    
    args = parser.parse_args()
    return args


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
    """Create cross-cluster features of Amazon data sets
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
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


# == sync, corrected by elderman == @elder man
def create_cross_features_yelpchi(df):
    """Create cross-group features of Yelpchi data sets
Return: (data box with new features, list of new feature names)"""
    df = df.copy()
    new_features = []
    
    # Cross-cutting of time and type features
    if 'td_type_ratio_7d_1' in df.columns and 'td_type1_30' in df.columns:
        df['td_type_ratio_7d_1_td_type1_30'] = df['td_type_ratio_7d_1'] * df['td_type1_30']
        new_features.append('td_type_ratio_7d_1_td_type1_30')
    
    # Maximum value and average value of neighbor's features Bad
    if 'fea_out_neighbor_fea_x_0_max' in df.columns and 'fea_out_neighbor_fea_x_0_mean' in df.columns:
        df['fea_out_neighbor_fea_x_0_diff'] = df['fea_out_neighbor_fea_x_0_max'] - df['fea_out_neighbor_fea_x_0_mean']
        new_features.append('fea_out_neighbor_fea_x_0_diff')
    
    # Product of Features
    if 'fea_x_18' in df.columns and 'fea_x_19' in df.columns:
        df['fea_x_18_x_19_product'] = df['fea_x_18'] * df['fea_x_19']
        new_features.append('fea_x_18_x_19_product')
    
    # Combination of risk transmission features
    if 'risk_nei_out_mean' in df.columns and 'risk_nei_in_mean' in df.columns:
        df['risk_nei_diff_mean'] = df['risk_nei_out_mean'] - df['risk_nei_in_mean']
        new_features.append('risk_nei_diff_mean')
    
    # The sum of the features
    if 'fea_x_13' in df.columns and 'fea_x_14' in df.columns:
        df['fea_x_13_x_14_sum'] = df['fea_x_13'] + df['fea_x_14']
        new_features.append('fea_x_13_x_14_sum')
    
    # Neighbors Statistical Profile
    if 'fea_out_neighbor_fea_x_1_mean' in df.columns and 'fea_out_neighbor_fea_x_1_max' in df.columns:
        df['fea_out_neighbor_x_1_mean_max_ratio'] = df['fea_out_neighbor_fea_x_1_mean'] / (df['fea_out_neighbor_fea_x_1_max'] + 1e-6)
        new_features.append('fea_out_neighbor_x_1_mean_max_ratio')
    
    # Time window feature interaction
    if 'td_type1_15' in df.columns and 'td_type1_7' in df.columns:
        df['td_type1_15_7_diff'] = df['td_type1_15'] - df['td_type1_7']
        new_features.append('td_type1_15_7_diff')
    
    # Convergence of image dissemination features
    if 'directed_1hop_in_f0_mean' in df.columns and 'directed_1hop_out_f0_mean' in df.columns:
        df['directed_1hop_in_out_ratio'] = df['directed_1hop_in_f0_mean'] / (df['directed_1hop_out_f0_mean'] + 1e-6)
        new_features.append('directed_1hop_in_out_ratio')
    
    # Multiple-jump feature comparisons
    if 'directed_2hop_in_f0_mean' in df.columns and 'directed_1hop_in_f0_mean' in df.columns:
        df['directed_2hop_1hop_in_ratio'] = df['directed_2hop_in_f0_mean'] / (df['directed_1hop_in_f0_mean'] + 1e-6)
        new_features.append('directed_2hop_1hop_in_ratio')
    
    # Total risk features
    if 'risk_nei_out_sum' in df.columns and 'risk_nei_in_sum' in df.columns:
        df['risk_nei_total_sum'] = df['risk_nei_out_sum'] + df['risk_nei_in_sum']
        new_features.append('risk_nei_total_sum')
    
    print(f"  Created successfully {len(new_features)} Cross-cutting features")
    print(f"  New feature list: {new_features}")
    
    return df, new_features


# == sync, corrected by elderman == @elder man
def create_cross_features_tfinance(df):
    """Create cross-cutting features of TFinance data sets
Return: (data box with new features, list of new feature names)"""
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
    
    print(f"  Created successfully {len(new_features)} Cross-cutting features")
    print(f"  New feature list: {new_features}")
    
    return result_df, new_features


# == sync, corrected by elderman == @elder man
def create_cross_features_dgraphfin(df):
    """Create 16 refined cross-cutting features based on the importance of character ranking
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
    
    # 1. Core time window analysis
    if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
        df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
        new_features.append('active_window_span')
    
    # 2. High-risk transaction density
    if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
        new_features.append('risk_transaction_density')
    
    # 3. Recent surprise trading
    if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
        new_features.append('recent_burst_risk')
    
    # 4. Transaction time anomalies
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
        new_features.append('timing_anomaly_score')
    
    # 5. Core identity mix
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
        new_features.append('core_feature_interaction')
    
    # 6. Risk of inactive accounts
    if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
        df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
        new_features.append('dormant_high_value_risk')
    
    # 7. Transaction time concentration degrees
    if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
        df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
        new_features.append('out_timestamp_concentration')
    
    # 8. Border type anomalies
    if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
        df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
        new_features.append('edge_type_discrepancy')
    
    # Minimum time stamp value abnormal
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
        df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
        new_features.append('first_interaction_gap')
    
    # 10. Weighted combination of feature x
    if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                    df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                    df['fea_x_8'] * 0.05)
        new_features.append('weighted_x_features')
    
    # 11. Transaction stability rating
    if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
        df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
        new_features.append('transaction_stability')
    
    # 12. HF risk trading
    if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
        df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
        new_features.append('high_freq_high_value')
    
    # 13. Account activity abnormal
    if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
        df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
        new_features.append('activity_inconsistency')
    
    # 14. Anomalous time patterns
    if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
        df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
        new_features.append('early_wide_spread_risk')
    
    # 15. Multiple risk superimposed
    if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                     np.log1p(df['undirected_1hop_f16_max']))
        new_features.append('multi_risk_composite')
    
    # 16. Interaction of core time features
    if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                     (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
        new_features.append('core_time_interaction')
    
    print(f"  Created successfully {len(new_features)} Cross-cutting features")
    print(f"  New feature list: {new_features}")
    
    return df, new_features


# == sync, corrected by elderman == @elder man
def create_cross_features(df, dataset_name):
    """Create function by selecting the corresponding cross-cutting feature of the data set name
Return: (data box with new features, list of new feature names)"""
    if dataset_name == 'amazon':
        return create_cross_features_amazon(df)
    elif dataset_name == 'yelpchi':
        return create_cross_features_yelpchi(df)
    elif dataset_name == 'tfinance':
        return create_cross_features_tfinance(df)
    elif dataset_name == 'dgraphfin':
        return create_cross_features_dgraphfin(df)
    else:
        print(f"  ⚠️ Unknown data set {dataset_name}，Skip cross feature creation")
        return df, []


# == sync, corrected by elderman == @elder man
def load_test_data(data_dir, label_file):
    """Load test data and labels"""
    print("=" * 60)
    print(f"Load test data...")
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
    df_test = pd.concat(all_dfs, axis=1)
    
    print(f"\n  Merged Feature Data: {df_test.shape}")
    
    # Test the merged repeat column
    duplicate_cols_after = df_test.columns[df_test.columns.duplicated()].tolist()
    if duplicate_cols_after:
        print(f"  ⚠️ After MergeDataFrameYeah. {len(duplicate_cols_after)} Repeat Columns: {duplicate_cols_after[:10]}...")
        
        # Solutions: Add suffix for repeat columns
        print("Processing duplicate columns...")
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
        print(f"  ✓ Repeated columns processed，Final Columns: {len(df_test.columns)}")
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
    df_test['label'] = labels
    
    print(f"  Final test data shape: {df_test.shape}")
    
    return df_test


# == sync, corrected by elderman == @elder man
def load_models_and_predict(df_test, cols_fea_cross, model_dir, n_folds=5):
    """Loading 5-percent model and making integrated predictions

Return:
-test pred: Integrated prediction results
-Fold preventions: each discount forecast
- Models: Loaded Model List
- y test: Real label"""
    print("\n" + "=" * 60)
    print("Load and predict models...")
    print("=" * 60)
    
    # Prepare test data
    x_test = df_test[cols_fea_cross].reset_index(drop=True)
    y_test = df_test['label'].reset_index(drop=True)
    
    print(f"\nNumber of test set samples: {len(x_test)}")
    print(f"Number of features: {len(cols_fea_cross)}")
    print(f"Positive sample ratio: {y_test.mean():.4f}")
    
    # Initialized projection results
    test_pred = np.zeros((len(x_test), 2))
    fold_predictions = []
    models = []
    
    mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"\nInitial Memory: {mem_start:.2f} GiB")
    
    for fold in range(1, n_folds + 1):
        print(f"\nLoad Model Fold {fold}/{n_folds}...")
        
        # Load Model
        model_path = os.path.join(model_dir, f'catboost_fold_{fold}.cbm')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        model = CatBoostClassifier()
        model.load_model(model_path)
        models.append(model)
        
        print(f"  Model loaded: {model_path}")
        
        # Single-rate forecast
        fold_pred = model.predict_proba(x_test)
        fold_predictions.append(fold_pred)
        
        # Cumulative projection results (for integration)
        test_pred += fold_pred / n_folds
        
        # Clear Memory
        del model, fold_pred
        gc.collect()
    
    mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"\nProjection complete.，Memory Usage: {mem_end:.2f} GiB")
    
    return test_pred, fold_predictions, models, y_test


# == sync, corrected by elderman == @elder man
def evaluate_predictions(y_true, y_pred_proba, y_pred_proba_list=None, n_folds=5):
    """Calculate and output assessment indicators

Parameters:
y true: Real tab
y pred proba: integration prediction probability
y pred proba list: probability list for each discount
n folds: discount"""
    print("\n" + "=" * 60)
    print("Assessment indicators")
    print("=" * 60)
    
    # Integrated projections
    y_pos_prob = y_pred_proba[:, 1]
    y_pred_label = (y_pos_prob > 0.5).astype(int)
    
    # Calculation of basic indicators
    auc_score = roc_auc_score(y_true, y_pos_prob)
    f1 = f1_score(y_true, y_pred_label)
    ap_score = average_precision_score(y_true, y_pos_prob)
    
    print(f"\n[Ensemble Model (5-fold average) Results]")
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Average Precision (AP): {ap_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_label)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn}, FP: {fp}")
    print(f"    FN: {fn}, TP: {tp}")
    
    # Calculating More Indicators
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n  Accuracy (Precision): {precision:.4f}")
    print(f"  Recall rate (Recall): {recall:.4f}")
    print(f"  Specificity (Specificity): {specificity:.4f}")
    
    # Independent assessments
    if y_pred_proba_list is not None:
        print(f"\n[Independent Evaluation Results]")
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
        
        print(f"\n  DeclinesAUC Mean ± Std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"  DeclinesF1 Mean ± Std: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    
    # Summary of all indicators
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


# == sync, corrected by elderman == @elder man
def save_predictions(y_test, test_pred, fold_predictions, output_dir):
    """Save forecast results"""
    print("\n" + "=" * 60)
    print("Save predictive results...")
    print("=" * 60)
    
    # Save integrated predictions
    ensemble_path = os.path.join(output_dir, 'test_ensemble_predictions.npy')
    np.save(ensemble_path, test_pred)
    print(f"  Integrated prediction results saved: {ensemble_path}")
    
    # Save discount forecast results
    for fold, fold_pred in enumerate(fold_predictions, 1):
        fold_path = os.path.join(output_dir, f'test_fold_{fold}_predictions.npy')
        np.save(fold_path, fold_pred)
        print(f"  Fold {fold} Forecast results saved: {fold_path}")
    
    # Save real labels and predict probabilities
    results_df = pd.DataFrame({
        'true_label': y_test.values,
        'pred_probability': test_pred[:, 1],
        'pred_label': (test_pred[:, 1] > 0.5).astype(int)
    })
    results_path = os.path.join(output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  Forecast results table saved: {results_path}")
    
    return ensemble_path, results_path


# == sync, corrected by elderman == @elder man
def main():
    # Parsing command line parameters
    args = parse_args()
    
    # Set global variables
    global DATASET_NAME, DATA_DIR, TEST_LABEL_FILE, MODEL_DIR, N_FOLDS, RANDOM_SEED
    
    DATASET_NAME = args.dataset
    DATA_DIR = os.path.join(args.data_dir, DATASET_NAME, 'test')
    TEST_LABEL_FILE = os.path.join(args.data_split_dir, f'{DATASET_NAME}_test.npz')
    MODEL_DIR = os.path.join(args.model_dir, DATASET_NAME)
    N_FOLDS = args.n_folds
    RANDOM_SEED = args.seed
    
    print("\n" + "=" * 60)
    print("CatBoost 50-percent model testing and evaluation")
    print("=" * 60)
    print(f"dataset: {DATASET_NAME}")
    print(f"Identity Directory: {DATA_DIR}")
    print(f"Tag File: {TEST_LABEL_FILE}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Cross-validation discounts: {N_FOLDS}")
    print(f"Random Feeds: {RANDOM_SEED}")
    print("=" * 60)
    
    # 1. Check the existence of model files
    print("Check model files...")
    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f'catboost_fold_{fold}.cbm')
        if os.path.exists(model_path):
            print(f"  ✓ Fold {fold} Model exists")
        else:
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    # Load training metadata (access feature column)
    print("\\n Load training metadata...")
    metadata_path = os.path.join(MODEL_DIR, 'training_metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        cols_fea_cross = metadata['cols_fea_cross']
        print(f"  Loaded feature configuration，Total {len(cols_fea_cross)} Features")
    else:
        raise FileNotFoundError(f"Training metadata does not exist: {metadata_path}")
    
    # Load test data
    df_test = load_test_data(DATA_DIR, TEST_LABEL_FILE)
    
    # 4. Screening features based on column specified by the user (in line with training)
    print("\n" + "=" * 60)
    print("Filter feature by column specified by user...")
    print("=" * 60)
    
    COLS_KEEP = get_COLS_KEEP(DATASET_NAME)
    
    # Check which specified columns exist in the data
    available_cols = [col for col in COLS_KEEP if col in df_test.columns]
    missing_cols = [col for col in COLS_KEEP if col not in df_test.columns]
    
    print(f"  User-Specify Columns: {len(COLS_KEEP)}")
    print(f"  Number of columns actually available: {len(available_cols)}")
    if missing_cols:
        print(f"  ⚠️ Missing Columns: {missing_cols[:10]}...")
    
    # Filter Data Box (only keep specified columns)
    df_test = df_test[available_cols].copy()
    print(f"  Data shape after filtering: {df_test.shape}")
    
    # 5. Creation of cross-cutting features (must be consistent with training)
    print("\\n Create Cross feature...")
    df_test, new_features = create_cross_features(df_test, DATASET_NAME)
    
    # 6. Load and forecast models
    test_pred, fold_predictions, models, y_test = load_models_and_predict(
        df_test, cols_fea_cross, model_dir=MODEL_DIR, n_folds=N_FOLDS
    )
    
    # 7. Calculation of assessment indicators
    metrics = evaluate_predictions(y_test, test_pred, fold_predictions, n_folds=N_FOLDS)
    
    # 8. Preservation of projections
    ensemble_path, results_path = save_predictions(
        y_test, test_pred, fold_predictions, MODEL_DIR
    )
    
    # Final summary of outputs
    print("\n" + "=" * 60)
    print("Test evaluation completed - final summary")
    print("=" * 60)
    print(f"\nNumber of test set samples: {len(y_test)}")
    print(f"Positive sample: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"Number of negative samples: {len(y_test) - y_test.sum()} ({(1-y_test.mean())*100:.2f}%)")
    print(f"\n[Core Metrics]")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  Average Precision: {metrics['ap']:.4f}")
    print(f"\n[Output File]")
    print(f"  Integrated projections: {ensemble_path}")
    print(f"  Results table: {results_path}")
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
