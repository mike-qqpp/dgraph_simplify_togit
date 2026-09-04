#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 加载训练好的模型进行推理和评估
支持多种数据集和模型类型

使用方法:
python test.py --dataset tfinance --model cab
或
python test.py -d tfinance -m cab
"""

import os
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test models with cross-validation')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Dataset name (e.g., amazon, yelpchi, tfinance, dgraphfin)')
    parser.add_argument('--model', type=str, default='cab', help='Model type: lgb, xgb, or cab')
    parser.add_argument('--data_dir', type=str, default='../feature_split',
                       help='Feature data directory (default: ../feature_split)')
    parser.add_argument('--data_split_dir', type=str, default='../data_split',
                       help='Data split directory (default: ../data_split)')
    parser.add_argument('--model_dir', type=str, default='../models',
                       help='Model directory (default: ./models)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed (default: 2022)')
    return parser.parse_args()


def get_COLS_KEEP(dataset):
    """Get selected feature columns based on dataset."""
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
        COLS_FEA_SLC = ['fea_x_18', 'fea_x_19', 'td_type_ratio_7d_1', 'fea_x_21', 'fea_x_13', 'td_type1_15', 'fea_x_20', 'fea_out_neighbor_fea_x_5_mean', 'fea_out_neighbor_fea_x_0_mean', 'fea_x_14', 'fea_x_0', 'td_type1_30', 'fea_x_16', 'fea_out_neighbor_fea_x_0_max', 'fea_x_1', 'fea_x_23', 'fea_out_neighbor_fea_x_5_max', 'fea_x_26', 'risk_nei_out_mean', 'td_type_ratio_in_7d_1', 'fea_out_neighbor_fea_x_1_max', 'fea_x_5', 'risk_nei_in_mean', 'td_type1_7', 'fea_out_neighbor_fea_x_0_min', 'fea_x_15', 'fea_x_22', 'fea_out_neighbor_fea_x_1_mean', 'fea_x_10', 'fea_x_25', 'fea_out_neighbor_fea_x_5_min', 'fea_x_30', 'fea_x_8', 'fea_out_neighbor_fea_x_1_min', 'fea_x_17', 'fea_x_28', 'fea_x_27', 'fea_out_neighbor_fea_x_8_max', 'fea_x_31', 'fea_in_edge_type_min', 'fea_x_29', 'directed_1hop_in_f24_std', 'undirected_1hop_f62_std', 'directed_2hop_out_f0_mean', 'directed_1hop_in_f6_min', 'fea_out_edge_type_min', 'directed_1hop_in_f13_mean', 'fea_out_neighbor_fea_x_2_mean', 'mixed_propagation_f0_mean', 'directed_1hop_in_f29_min', 'fea_x_2', 'fea_out_neighbor_fea_x_8_mean', 'fea_out_neighbor_fea_x_8_min', 'fea_x_24', 'mixed_propagation_f6_mean', 'fea_out_neighbor_fea_x_2_min', 'mixed_propagation_f16_std', 'directed_1hop_in_f63_std', 'financial_hub_f5_mean', 'directed_2hop_out_f8_mean', 'directed_1hop_out_f31_std', 'undirected_1hop_f63_std', 'undirected_1hop_f59_std', 'directed_1hop_in_f0_mean', 'directed_1hop_in_f62_std', 'transaction_chain_f50_std', 'directed_2hop_out_f30_std', 'directed_1hop_out_f7_std', 'fea_out_edge_type_nunique', 'mixed_propagation_f46_mean', 'directed_1hop_out_f62_min', 'directed_2hop_in_f0_std', 'directed_1hop_out_f30_std', 'undirected_1hop_f28_std', 'directed_2hop_in_f0_mean', 'mixed_propagation_f0_min', 'directed_1hop_out_f24_std', 'directed_2hop_in_f0_min', 'directed_1hop_in_f1_mean', 'directed_1hop_in_f58_std', 'directed_1hop_in_f31_std', 'mixed_propagation_f23_max', 'directed_1hop_in_f28_std', 'transaction_chain_f10_std', 'transaction_chain_f0_std', 'directed_1hop_in_f26_std', 'transaction_chain_f10_min', 'mixed_propagation_f25_std', 'directed_1hop_out_f62_std', 'undirected_1hop_f27_std', 'directed_2hop_out_f6_mean', 'mixed_propagation_f32_mean', 'directed_1hop_in_f7_mean', 'directed_2hop_in_f25_std', 'mixed_propagation_f28_std', 'mixed_propagation_f9_std', 'directed_1hop_in_f11_mean', 'directed_2hop_in_f8_std', 'transaction_chain_f62_std', 'directed_2hop_in_f42_std', 'transaction_chain_f0_mean', 'mixed_propagation_f10_min', 'directed_2hop_in_f25_mean', 'directed_2hop_in_f1_max', 'directed_2hop_in_f31_std', 'directed_1hop_out_f0_mean', 'directed_2hop_in_f14_mean', 'directed_2hop_in_f10_std', 'directed_2hop_in_f24_mean', 'directed_1hop_in_f10_mean', 'directed_1hop_in_f25_std', 'directed_1hop_in_f29_std', 'fea_out_neighbor_fea_x_2_max', 'directed_1hop_out_f28_std', 'directed_1hop_in_f14_std', 'directed_2hop_in_f62_std', 'directed_2hop_out_f6_min', 'directed_2hop_out_f23_max', 'mixed_propagation_f0_max', 'financial_hub_f0_mean', 'directed_1hop_in_f23_mean', 'fea_in_edge_type_nunique', 'directed_2hop_in_f24_std', 'directed_1hop_in_f8_std', 'directed_1hop_in_f32_mean', 'transaction_chain_f9_mean', 'directed_2hop_out_f32_mean', 'td_type_ratio_7d_2', 'directed_2hop_out_f6_std', 'mixed_propagation_f14_mean']
    elif dataset == 'tfinance':
        COLS_FEA_SLC = ['feature_1_rank', 'feature_1_normalized', 'chebyshev_19', 'feature_1_squared', 'interaction_7_8', 'important_ratio_feature_1_over_isolation_forest_score', 'important_abs_diff_feature_1_laplacian_dim0', 'important_product_feature_1_lof_score', 'important_ratio_feature_2_over_lof_score', 'important_abs_diff_feature_1_feature_embedding_dim0', 'pca_component_6', 'important_abs_diff_feature_1_isolation_forest_score', 'feature_1_poly2', 'chebyshev_18', 'laplacian_eig_9', 'interaction_7_9', 'feature_2_lag1_ratio', 'orig_feat1_degree_diff', 'feature_8_cluster_bin', 'interaction_triple_interaction_0_1_interaction_0_2_interaction_1_2', 'interaction_1_2_sqrt', 'important_diff_feature_1_minus_isolation_forest_score', 'feature_8_normalized', 'important_diff_feature_1_minus_laplacian_dim0', 'ica_component_8', 'feature_max', 'important_diff_feature_1_minus_feature_embedding_dim0', 'important_abs_diff_feature_2_feature_entropy', 'feature_embedding_dim6', 'pca_component_8', 'random_proj_7', 'feature_skewness', 'important_diff_gnn_embedding_dim0_minus_isolation_forest_score', 'feature_embedding_dim9', 'poly_feat_7', 'feature_kurtosis', 'important_ratio_feature_2_over_one_class_svm_score', 'laplacian_dim7', 'feature_embedding_dim1', 'gnn_embedding_dim0_exp', 'kmeans_5_cluster_size_norm', 'ica_component_1', 'stat_product_0_2_zscore', 'ica_component_9', 'pca_component_7', 'feature_8', 'important_product_isolation_forest_score_one_class_svm_score', 'interaction_triple_interaction_0_1_interaction_0_2_interaction_1_3', 'random_proj_4', 'feature_1_lag3_ratio', 'random_proj_5', 'ica_component_3', 'feature_embedding_dim7', 'important_product_feature_embedding_dim0_one_class_svm_score', 'pca_component_3', 'gnn_embedding_dim2', 'important_ratio_feature_1_over_lof_score', 'important_features_cv', 'laplacian_dim2', 'feature_8_standardized', 'important_diff_laplacian_dim0_minus_one_class_svm_score', 'interaction_harmonic_triple_interaction_0_1_interaction_1_3_interaction_2_3', 'combined_max_anomaly_score', 'interaction_3_4', 'feature_8_sqrt', 'important_ratio_isolation_forest_score_over_lof_score', 'feature_8_rank', 'interaction_5_7', 'poly_feat_17', 'chebyshev_11', 'feature_9_rank', 'important_abs_diff_isolation_forest_score_lof_score', 'poly_feat_19', 'feature_8_bucket', 'feature_8_log', 'kmeans_10_min_dist', 'degree_cluster_degree_cluster_4_feat1_dist', 'feature_7_bucket', 'feature_entropy', 'feature_4_lag3_diff', 'feature_8_equal_width_bin', 'random_proj_0', 'feature_0_rolling_max_ws10', 'feature_2_rolling_max_ws5', 'interaction_6_8', 'feature_8_equal_freq_bin', 'feature_0_rolling_min_ws10', 'important_ratio_isolation_forest_score_over_feature_entropy', 'feature_1_lag2_diff', 'orig_feat0_feat_degree_ratio', 'feature_3_rolling_change_ws20', 'feature_4_lag2_ratio', 'gnn_embedding_dim0_squared', 'feature_9', 'poly_feat_6', 'pca_component_9', 'important_abs_diff_isolation_forest_score_feature_entropy', 'feature_6_sqrt', 'pca_component_1', 'feature_range', 'degree_cluster_degree_cluster_0_feat2_dist', 'kmeans_20_cluster_size', 'important_abs_diff_gnn_embedding_dim0_isolation_forest_score', 'chebyshev_8', 'feature_5_standardized', 'feature_6', 'anomaly_score_max', 'interaction_1_2_level', 'feature_6_log', 'pca_component_0', 'feature_1_standardized', 'bin_9', 'mahalanobis_distance', 'important_ratio_isolation_forest_score_over_one_class_svm_score', 'important_product_feature_embedding_dim0_feature_entropy', 'bin_8', 'kmeans_5_nearest_cluster', 'anomaly_harmonic_lof_score_one_class_svm_score', 'important_abs_diff_gnn_embedding_dim0_lof_score', 'important_diff_isolation_forest_score_minus_lof_score', 'lof_score', 'gnn_embedding_dim0_decile', 'poly_feat_11', 'important_ratio_one_class_svm_score_over_feature_entropy', 'gnn_embedding_dim0_zscore_abs', 'feature_6_bucket', 'feature_9_equal_width_bin', 'feature_3_bucket', 'kmeans_10_cluster_size', 'kmeans_10_nearest_cluster']
    elif dataset == 'dgraphfin':
        COLS_FEA_SLC = ['fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11', 'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6', 'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean', 'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0', 'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8', 'directed_1hop_out_interval_mean', 'directed_1hop_in_f16_mean', 'directed_1hop_in_f7_max', 'undirected_1hop_f16_mean', 'undirected_2hop_f12_max', 'directed_1hop_out_timestamp_std', 'undirected_1hop_interval_mean', 'directed_1hop_out_timestamp_mean', 'fea_out_edge_timestamp_diff_1_target_mean', 'fea_x_4', 'fea_out_edge_type_min', 'undirected_1hop_f1_min', 'fea_in_edge_type_mean', 'directed_1hop_in_f7_mean', 'undirected_1hop_interval_min', 'fea_out_edge_timestamp_diff_1_target_min', 'undirected_2hop_interval_std', 'undirected_2hop_f16_max', 'fea_out_edge_timestamp_diff_1_target_max', 'undirected_1hop_timestamp_mean', 'fea_out_neighbor_fea_x_13_max', 'undirected_2hop_interval_min', 'undirected_1hop_f7_max', 'directed_1hop_out_f16_mean', 'fea_in_edge_timestamp_diff_1_source_max', 'directed_1hop_in_f15_mean', 'undirected_2hop_f1_min', 'undirected_1hop_timestamp_range', 'directed_1hop_out_interval_max', 'undirected_2hop_f15_max', 'directed_1hop_out_type_5_max', 'fea_out_edge_type_mode', 'mixed_propagation_f4_max', 'fea_out_edge_timestamp_diff_1_target_mode', 'undirected_1hop_type_4_min', 'directed_1hop_in_f16_max', 'undirected_2hop_f1_mean', 'undirected_1hop_f11_max', 'undirected_2hop_f1_std', 'degree_ratio', 'directed_1hop_out_interval_min', 'undirected_1hop_f7_mean', 'fea_x_3', 'fea_in_edge_type_max', 'directed_1hop_in_f13_mean', 'active_days_out', 'undirected_2hop_f9_max', 'undirected_2hop_f13_max', 'undirected_1hop_type_5_std', 'fea_out_edge_type_std', 'undirected_1hop_timestamp_std', 'fea_x_12', 'undirected_1hop_f12_max', 'undirected_2hop_timestamp_min', 'undirected_1hop_f1_std', 'undirected_2hop_interval_max', 'undirected_1hop_interval_max', 'undirected_1hop_f13_mean', 'fea_x_7', 'undirected_1hop_timestamp_max', 'directed_1hop_in_f13_max', 'undirected_2hop_timestamp_mean', 'undirected_1hop_interval_std', 'directed_1hop_in_f15_max', 'undirected_2hop_f2_std', 'directed_1hop_in_f7_min', 'transaction_chain_interval_mean', 'directed_1hop_out_type_6_max', 'directed_2hop_out_type_4_std', 'undirected_2hop_type_5_std', 'fea_out_neighbor_fea_x_1_max', 'undirected_2hop_f11_max', 'undirected_1hop_f14_max', 'undirected_1hop_f9_max', 'undirected_1hop_f1_mean', 'transaction_chain_f16_mean', 'undirected_2hop_interval_mean', 'financial_hub_timestamp_min', 'undirected_2hop_f11_std', 'fea_x_16', 'undirected_1hop_f1_max', 'fea_x_9', 'undirected_2hop_type_4_range', 'directed_1hop_in_f12_max', 'directed_1hop_in_f16_min', 'fea_out_neighbor_fea_x_1_min', 'mixed_propagation_f15_max', 'directed_1hop_out_type_5_min', 'directed_1hop_out_f15_mean', 'mixed_propagation_f16_max', 'undirected_2hop_timestamp_max', 'directed_2hop_out_type_4_min', 'mixed_propagation_f11_max', 'directed_1hop_out_f1_max', 'total_degree', 'undirected_1hop_f12_std', 'directed_2hop_out_type_6_range', 'undirected_1hop_f13_std', 'directed_1hop_in_timestamp_min', 'undirected_1hop_f10_mean', 'undirected_2hop_timestamp_std', 'directed_1hop_out_type_5_mean', 'undirected_1hop_f5_max', 'undirected_2hop_f7_max', 'transaction_chain_f7_std', 'mixed_propagation_timestamp_min', 'directed_2hop_out_timestamp_mean', 'fea_out_neighbor_fea_x_1_mean', 'directed_2hop_in_f9_mean', 'directed_2hop_in_f1_mean', 'transaction_chain_type_6_min', 'undirected_2hop_f4_std']
    else:
        COLS_FEA_SLC = []
    
    COLS_KEEP = ['label'] + COLS_FEA_SLC
    return COLS_KEEP


def create_feature_safe(df, feature_name, expression, existing_set):
    """Safely create feature, avoid duplicate names."""
    if feature_name not in existing_set:
        df[feature_name] = expression
        existing_set.add(feature_name)
        return True
    else:
        print(f"  Skipping duplicate feature: {feature_name}")
        return False


def create_cross_features_amazon(df):
    """Create cross features for amazon dataset."""
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
    # Time-related cross features
    if 'fea_x_19' in existing_features and 'fea_x_12' in existing_features:
        if create_feature_safe(df, 'time_related_feature', 
                              df['fea_x_19'] * df['fea_x_12'], existing_features):
            new_features.append('time_related_feature')
    
    # Graph structure related cross features
    if 'fea_out_neighbor_fea_x_0_max' in existing_features and 'fea_out_neighbor_fea_x_7_max' in existing_features:
        if create_feature_safe(df, 'graph_structure_feature',
                              df['fea_out_neighbor_fea_x_0_max'] + df['fea_out_neighbor_fea_x_7_max'], existing_features):
            new_features.append('graph_structure_feature')
    
    # Type distribution related cross features
    if 'td_type_ratio_7d_1' in existing_features and 'td_type2_30' in existing_features:
        if create_feature_safe(df, 'type_distribution_feature',
                              df['td_type_ratio_7d_1'] * df['td_type2_30'], existing_features):
            new_features.append('type_distribution_feature')
    
    # Edge type and time interval related cross features
    if 'fea_out_edge_type_mean' in existing_features and 'directed_1hop_in_type_0_count' in existing_features:
        if create_feature_safe(df, 'edge_type_time_interval_feature',
                              df['fea_out_edge_type_mean'] * df['directed_1hop_in_type_0_count'], existing_features):
            new_features.append('edge_type_time_interval_feature')
    
    # Risk propagation and graph structure cross features
    if 'mixed_propagation_type_4_count' in existing_features and 'directed_1hop_out_type_4_count' in existing_features:
        if create_feature_safe(df, 'risk_propagation_feature',
                              df['mixed_propagation_type_4_count'] + df['directed_1hop_out_type_4_count'], existing_features):
            new_features.append('risk_propagation_feature')
    
    # Multi-hop neighbor behavior contrast features
    if 'undirected_1hop_type_0_count' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        if create_feature_safe(df, 'hop_behavior_contrast',
                              df['undirected_1hop_type_0_count'] - df['undirected_2hop_type_0_count'], existing_features):
            new_features.append('hop_behavior_contrast')
    
    # Business-specific anomaly detection features
    if 'transaction_chain_f0_mean' in existing_features and 'directed_1hop_out_type_0_count' in existing_features:
        if create_feature_safe(df, 'business_specific_anomaly',
                              df['transaction_chain_f0_mean'] * df['directed_1hop_out_type_0_count'], existing_features):
            new_features.append('business_specific_anomaly')
    
    # Statistical combination features
    if 'undirected_1hop_f5_std' in existing_features and 'undirected_2hop_type_0_count' in existing_features:
        if create_feature_safe(df, 'statistical_combination',
                              df['undirected_1hop_f5_std'] + df['undirected_2hop_type_0_count'], existing_features):
            new_features.append('statistical_combination')
    
    # Original feature stability cross features
    if 'fea_x_2' in existing_features and 'fea_x_6' in existing_features:
        if create_feature_safe(df, 'feature_stability',
                              df['fea_x_2'] * df['fea_x_6'], existing_features):
            new_features.append('feature_stability')
    
    print(f"  Successfully created {len(new_features)} cross features")
    return df, new_features


def create_cross_features_yelpchi(df):
    """Create cross features for yelpchi dataset."""
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
    # Time feature and type feature cross
    if 'td_type_ratio_7d_1' in existing_features and 'td_type1_30' in existing_features:
        if create_feature_safe(df, 'td_type_ratio_7d_1_td_type1_30',
                              df['td_type_ratio_7d_1'] * df['td_type1_30'], existing_features):
            new_features.append('td_type_ratio_7d_1_td_type1_30')
    
    # Difference between max and mean of neighbor features
    if 'fea_out_neighbor_fea_x_0_max' in existing_features and 'fea_out_neighbor_fea_x_0_mean' in existing_features:
        if create_feature_safe(df, 'fea_out_neighbor_fea_x_0_diff',
                              df['fea_out_neighbor_fea_x_0_max'] - df['fea_out_neighbor_fea_x_0_mean'], existing_features):
            new_features.append('fea_out_neighbor_fea_x_0_diff')
    
    # Product of features
    if 'fea_x_18' in existing_features and 'fea_x_19' in existing_features:
        if create_feature_safe(df, 'fea_x_18_x_19_product',
                              df['fea_x_18'] * df['fea_x_19'], existing_features):
            new_features.append('fea_x_18_x_19_product')
    
    # Combination of risk propagation features
    if 'risk_nei_out_mean' in existing_features and 'risk_nei_in_mean' in existing_features:
        if create_feature_safe(df, 'risk_nei_diff_mean',
                              df['risk_nei_out_mean'] - df['risk_nei_in_mean'], existing_features):
            new_features.append('risk_nei_diff_mean')
    
    # Sum of features
    if 'fea_x_13' in existing_features and 'fea_x_14' in existing_features:
        if create_feature_safe(df, 'fea_x_13_x_14_sum',
                              df['fea_x_13'] + df['fea_x_14'], existing_features):
            new_features.append('fea_x_13_x_14_sum')
    
    # Neighbor statistical feature combination
    if 'fea_out_neighbor_fea_x_1_mean' in existing_features and 'fea_out_neighbor_fea_x_1_max' in existing_features:
        if create_feature_safe(df, 'fea_out_neighbor_x_1_mean_max_ratio',
                              df['fea_out_neighbor_fea_x_1_mean'] / (df['fea_out_neighbor_fea_x_1_max'] + 1e-6), existing_features):
            new_features.append('fea_out_neighbor_x_1_mean_max_ratio')
    
    # Time window feature interaction
    if 'td_type1_15' in existing_features and 'td_type1_7' in existing_features:
        if create_feature_safe(df, 'td_type1_15_7_diff',
                              df['td_type1_15'] - df['td_type1_7'], existing_features):
            new_features.append('td_type1_15_7_diff')
    
    # Graph propagation feature aggregation
    if 'directed_1hop_in_f0_mean' in existing_features and 'directed_1hop_out_f0_mean' in existing_features:
        if create_feature_safe(df, 'directed_1hop_in_out_ratio',
                              df['directed_1hop_in_f0_mean'] / (df['directed_1hop_out_f0_mean'] + 1e-6), existing_features):
            new_features.append('directed_1hop_in_out_ratio')
    
    # Multi-hop feature contrast
    if 'directed_2hop_in_f0_mean' in existing_features and 'directed_1hop_in_f0_mean' in existing_features:
        if create_feature_safe(df, 'directed_2hop_1hop_in_ratio',
                              df['directed_2hop_in_f0_mean'] / (df['directed_1hop_in_f0_mean'] + 1e-6), existing_features):
            new_features.append('directed_2hop_1hop_in_ratio')
    
    # Risk sum feature
    if 'risk_nei_out_sum' in existing_features and 'risk_nei_in_sum' in existing_features:
        if create_feature_safe(df, 'risk_nei_total_sum',
                              df['risk_nei_out_sum'] + df['risk_nei_in_sum'], existing_features):
            new_features.append('risk_nei_total_sum')
    
    print(f"  Successfully created {len(new_features)} cross features")
    return df, new_features


def create_cross_features_tfinance(df):
    """
    创建有意义的交叉组合特征（基于特征的实际含义和业务逻辑）
    返回: (包含新特征的数据框, 新特征名称列表)
    """
    df = df.copy()
    new_features = []
    
    # 1. 异常检测强度交叉特征（异常分数之间的相互作用）
    if 'lof_score' in df.columns and 'anomaly_score_max' in df.columns:
        # LOF分数与最大异常分数的相互作用
        df['lof_anomaly_max_interaction'] = df['lof_score'] * df['anomaly_score_max']
        new_features.append('lof_anomaly_max_interaction')
    
    if 'combined_max_anomaly_score' in df.columns and 'anomaly_harmonic_lof_score_one_class_svm_score' in df.columns:
        # 最大异常分数与调和异常分数的交叉
        df['max_harmonic_anomaly_product'] = df['combined_max_anomaly_score'] * df['anomaly_harmonic_lof_score_one_class_svm_score']
        new_features.append('max_harmonic_anomaly_product')
    
    # 2. 图结构与异常检测的交叉（图嵌入与异常分数的相互作用）
    if 'gnn_embedding_dim0_exp' in df.columns and 'lof_score' in df.columns:
        # 图嵌入与LOF分数的相互作用，可能捕获结构异常
        df['graph_embedding_lof_interaction'] = df['gnn_embedding_dim0_exp'] * df['lof_score']
        new_features.append('graph_embedding_lof_interaction')
    
    if 'feature_embedding_dim1' in df.columns and 'important_ratio_feature_1_over_isolation_forest_score' in df.columns:
        # 特征嵌入与异常检测比值的交叉
        df['embedding_anomaly_ratio_product'] = df['feature_embedding_dim1'] * df['important_ratio_feature_1_over_isolation_forest_score']
        new_features.append('embedding_anomaly_ratio_product')
    
    # 3. 时间序列特征与统计特征的交叉
    if 'feature_1_lag2_diff' in df.columns and 'feature_skewness' in df.columns:
        # 滞后差分与偏度的交叉，可能捕获时间序列的非对称性
        df['lag_diff_skewness_interaction'] = df['feature_1_lag2_diff'] * df['feature_skewness']
        new_features.append('lag_diff_skewness_interaction')
    
    if 'feature_2_lag1_ratio' in df.columns and 'feature_kurtosis' in df.columns:
        # 滞后比率与峰度的交叉，可能捕获尖峰厚尾效应
        df['lag_ratio_kurtosis_interaction'] = df['feature_2_lag1_ratio'] * df['feature_kurtosis']
        new_features.append('lag_ratio_kurtosis_interaction')
    
    # 4. 交互特征与异常评分的交叉
    if 'interaction_7_8' in df.columns and 'important_ratio_feature_1_over_lof_score' in df.columns:
        # 节点7-8交互与特征1异常比率的交叉
        df['interaction_anomaly_ratio_product'] = df['interaction_7_8'] * df['important_ratio_feature_1_over_lof_score']
        new_features.append('interaction_anomaly_ratio_product')
    
    if 'interaction_7_9' in df.columns and 'important_ratio_isolation_forest_score_over_lof_score' in df.columns:
        # 节点7-9交互与异常检测方法对比的交叉
        df['interaction_anomaly_comparison_product'] = df['interaction_7_9'] * df['important_ratio_isolation_forest_score_over_lof_score']
        new_features.append('interaction_anomaly_comparison_product')
    
    # 5. 聚类特征与异常特征的交叉
    if 'kmeans_10_cluster_size' in df.columns and 'lof_score' in df.columns:
        # 聚类规模与局部异常因子的交叉
        df['cluster_size_lof_interaction'] = df['kmeans_10_cluster_size'] * df['lof_score']
        new_features.append('cluster_size_lof_interaction')
    
    if 'kmeans_5_nearest_cluster' in df.columns and 'mahalanobis_distance' in df.columns:
        # 最近邻聚类与马氏距离的交叉
        df['nearest_cluster_mahalanobis_product'] = df['kmeans_5_nearest_cluster'] * df['mahalanobis_distance']
        new_features.append('nearest_cluster_mahalanobis_product')
    
    # 6. 特征变换与异常检测的交叉
    if 'feature_1_normalized' in df.columns and 'important_abs_diff_feature_1_isolation_forest_score' in df.columns:
        # 标准化特征与其异常检测差异的交叉
        df['normalized_feature_anomaly_diff_product'] = df['feature_1_normalized'] * df['important_abs_diff_feature_1_isolation_forest_score']
        new_features.append('normalized_feature_anomaly_diff_product')
    
    if 'feature_8_sqrt' in df.columns and 'important_diff_feature_1_minus_isolation_forest_score' in df.columns:
        # 平方根变换特征与异常差异的交叉
        df['sqrt_feature_anomaly_diff_product'] = df['feature_8_sqrt'] * df['important_diff_feature_1_minus_isolation_forest_score']
        new_features.append('sqrt_feature_anomaly_diff_product')
    
    # 7. 降维特征与统计特征的交叉
    if 'pca_component_6' in df.columns and 'feature_entropy' in df.columns:
        # PCA主成分与信息熵的交叉
        df['pca_entropy_interaction'] = df['pca_component_6'] * df['feature_entropy']
        new_features.append('pca_entropy_interaction')
    
    if 'ica_component_8' in df.columns and 'feature_skewness' in df.columns:
        # ICA独立成分与偏度的交叉
        df['ica_skewness_interaction'] = df['ica_component_8'] * df['feature_skewness']
        new_features.append('ica_skewness_interaction')
    
    # 8. 特征重要性相关的交叉
    if 'important_ratio_feature_1_over_lof_score' in df.columns and 'important_ratio_feature_2_over_lof_score' in df.columns:
        # 不同特征相对于LOF的重要性比率的相互作用
        df['feature_importance_ratio_product'] = df['important_ratio_feature_1_over_lof_score'] * df['important_ratio_feature_2_over_lof_score']
        new_features.append('feature_importance_ratio_product')
    
    if 'important_product_feature_1_lof_score' in df.columns and 'important_product_isolation_forest_score_one_class_svm_score' in df.columns:
        # 不同异常检测方法的重要性乘积的交叉
        df['anomaly_importance_product_interaction'] = df['important_product_feature_1_lof_score'] * df['important_product_isolation_forest_score_one_class_svm_score']
        new_features.append('anomaly_importance_product_interaction')
    
    # 9. 滚动统计与异常检测的交叉
    if 'feature_0_rolling_max_ws10' in df.columns and 'important_ratio_isolation_forest_score_over_feature_entropy' in df.columns:
        # 滚动最大值与异常/信息熵比率的交叉
        df['rolling_max_anomaly_entropy_ratio_product'] = df['feature_0_rolling_max_ws10'] * df['important_ratio_isolation_forest_score_over_feature_entropy']
        new_features.append('rolling_max_anomaly_entropy_ratio_product')
    
    if 'feature_2_rolling_max_ws5' in df.columns and 'lof_score' in df.columns:
        # 短期滚动最大值与LOF分数的交叉
        df['short_rolling_max_lof_interaction'] = df['feature_2_rolling_max_ws5'] * df['lof_score']
        new_features.append('short_rolling_max_lof_interaction')
    
    # 10. 特征差异与图嵌入的交叉
    if 'orig_feat1_degree_diff' in df.columns and 'gnn_embedding_dim2' in df.columns:
        # 原始特征度差与图嵌入的交叉
        df['degree_diff_graph_embedding_product'] = df['orig_feat1_degree_diff'] * df['gnn_embedding_dim2']
        new_features.append('degree_diff_graph_embedding_product')
    
    if 'important_abs_diff_feature_1_laplacian_dim0' in df.columns and 'laplacian_dim7' in df.columns:
        # 拉普拉斯特征差异与拉普拉斯维度的交叉
        df['laplacian_diff_dimension_product'] = df['important_abs_diff_feature_1_laplacian_dim0'] * df['laplacian_dim7']
        new_features.append('laplacian_diff_dimension_product')
    
    # 11. 多项式特征与异常检测的交叉
    if 'feature_1_poly2' in df.columns and 'important_diff_feature_1_minus_isolation_forest_score' in df.columns:
        # 二次多项式特征与异常差异的交叉
        df['poly2_feature_anomaly_diff_product'] = df['feature_1_poly2'] * df['important_diff_feature_1_minus_isolation_forest_score']
        new_features.append('poly2_feature_anomaly_diff_product')
    
    if 'poly_feat_7' in df.columns and 'anomaly_score_max' in df.columns:
        # 多项式特征与最大异常分数的交叉
        df['poly_feature_anomaly_max_product'] = df['poly_feat_7'] * df['anomaly_score_max']
        new_features.append('poly_feature_anomaly_max_product')
    
    # 12. 特征分桶与统计量的交叉
    if 'feature_8_bucket' in df.columns and 'feature_range' in df.columns:
        # 特征分桶与值域的交叉
        df['bucket_range_interaction'] = df['feature_8_bucket'] * df['feature_range']
        new_features.append('bucket_range_interaction')
    
    if 'feature_6_bucket' in df.columns and 'feature_skewness' in df.columns:
        # 特征分桶与偏度的交叉
        df['bucket_skewness_interaction'] = df['feature_6_bucket'] * df['feature_skewness']
        new_features.append('bucket_skewness_interaction')
    
    # 13. 切比雪夫特征与异常检测的交叉
    if 'chebyshev_19' in df.columns and 'important_ratio_feature_1_over_isolation_forest_score' in df.columns:
        # 切比雪夫多项式与异常比率的交叉
        df['chebyshev_anomaly_ratio_product'] = df['chebyshev_19'] * df['important_ratio_feature_1_over_isolation_forest_score']
        new_features.append('chebyshev_anomaly_ratio_product')
    
    if 'chebyshev_11' in df.columns and 'lof_score' in df.columns:
        # 切比雪夫多项式与LOF分数的交叉
        df['chebyshev_lof_interaction'] = df['chebyshev_11'] * df['lof_score']
        new_features.append('chebyshev_lof_interaction')
    
    # 14. 特征熵与异常比率的交叉
    if 'feature_entropy' in df.columns and 'important_ratio_isolation_forest_score_over_feature_entropy' in df.columns:
        # 信息熵与异常/熵比率的交叉（二次效应）
        df['entropy_anomaly_entropy_ratio_product'] = df['feature_entropy'] * df['important_ratio_isolation_forest_score_over_feature_entropy']
        new_features.append('entropy_anomaly_entropy_ratio_product')
    
    # 15. 特征排名与异常检测的交叉
    if 'feature_1_rank' in df.columns and 'important_ratio_feature_1_over_lof_score' in df.columns:
        # 特征排名与其异常比率的交叉
        df['rank_anomaly_ratio_product'] = df['feature_1_rank'] * df['important_ratio_feature_1_over_lof_score']
        new_features.append('rank_anomaly_ratio_product')
    
    # 16. 创建一些有意义的比值特征（不是简单的乘积）
    if 'feature_max' in df.columns and 'feature_range' in df.columns and 'feature_max' not in [0, np.nan]:
        # 最大值与值域的比值，反映分布形态
        df['max_to_range_ratio'] = df['feature_max'] / (df['feature_range'] + 1e-8)
        new_features.append('max_to_range_ratio')
    
    if 'lof_score' in df.columns and 'important_ratio_isolation_forest_score_over_lof_score' in df.columns:
        # LOF分数与异常方法对比的加权组合
        df['lof_weighted_comparison'] = df['lof_score'] * np.log1p(df['important_ratio_isolation_forest_score_over_lof_score'])
        new_features.append('lof_weighted_comparison')
    
    if 'feature_skewness' in df.columns and 'feature_kurtosis' in df.columns:
        # 偏度与峰度的相互作用，反映分布的非正态性
        df['skewness_kurtosis_interaction'] = df['feature_skewness'] * np.abs(df['feature_kurtosis'])
        new_features.append('skewness_kurtosis_interaction')
    
    # 确保只返回最多30个新特征
    if len(new_features) > 30:
        new_features = new_features[:30]
        # 删除多余的特征列
        for feat in new_features[30:]:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)
    
    print(f"成功创建 {len(new_features)} 个有意义的交叉特征")
    print("新特征列表:", new_features)
    
    return df, new_features


def create_cross_features_dgraphfin(df):
    """Create cross features for dgraphfin dataset."""
    df = df.copy()
    new_features = []
    existing_features = set(df.columns.tolist())
    
    # Core time window analysis
    if 'last_active_out' in existing_features and 'undirected_1hop_timestamp_min' in existing_features:
        if create_feature_safe(df, 'active_window_span',
                              df['last_active_out'] - df['undirected_1hop_timestamp_min'], existing_features):
            new_features.append('active_window_span')
    
    # High-risk transaction density
    if 'undirected_1hop_f16_max' in existing_features and 'directed_1hop_out_timestamp_range' in existing_features:
        if create_feature_safe(df, 'risk_transaction_density',
                              df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6), existing_features):
            new_features.append('risk_transaction_density')
    
    # Recent burst transactions
    if 'last_active_out' in existing_features and 'undirected_1hop_f16_max' in existing_features:
        if create_feature_safe(df, 'recent_burst_risk',
                              (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max'], existing_features):
            new_features.append('recent_burst_risk')
    
    # Transaction time anomaly
    if 'directed_1hop_out_timestamp_range' in existing_features and 'directed_1hop_out_interval_std' in existing_features:
        if create_feature_safe(df, 'timing_anomaly_score',
                              df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'], existing_features):
            new_features.append('timing_anomaly_score')
    
    # Core feature interaction
    if all(f in existing_features for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
        if create_feature_safe(df, 'core_feature_interaction',
                              df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6), existing_features):
            new_features.append('core_feature_interaction')
    
    # Dormant account risk
    if 'inactive_days_out' in existing_features and 'undirected_1hop_f15_max' in existing_features:
        if create_feature_safe(df, 'dormant_high_value_risk',
                              df['inactive_days_out'] * df['undirected_1hop_f15_max'], existing_features):
            new_features.append('dormant_high_value_risk')
    
    # Transaction time concentration
    if 'directed_1hop_out_timestamp_min' in existing_features and 'directed_1hop_out_timestamp_max' in existing_features:
        if create_feature_safe(df, 'out_timestamp_concentration',
                              1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1), existing_features):
            new_features.append('out_timestamp_concentration')
    
    # Edge type anomaly
    if 'fea_out_edge_type_mean' in existing_features and 'fea_out_edge_type_max' in existing_features:
        if create_feature_safe(df, 'edge_type_discrepancy',
                              df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean'], existing_features):
            new_features.append('edge_type_discrepancy')
    
    # Timestamp min anomaly
    if 'undirected_1hop_timestamp_min' in existing_features and 'directed_1hop_out_timestamp_min' in existing_features:
        if create_feature_safe(df, 'first_interaction_gap',
                              df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min'], existing_features):
            new_features.append('first_interaction_gap')
    
    # Weighted combination of feature x based on importance
    if all(f in existing_features for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
        if create_feature_safe(df, 'weighted_x_features',
                              (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                               df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                               df['fea_x_8'] * 0.05), existing_features):
            new_features.append('weighted_x_features')
    
    # Transaction stability score
    if 'directed_1hop_out_timestamp_range' in existing_features and 'directed_1hop_out_interval_std' in existing_features:
        if create_feature_safe(df, 'transaction_stability',
                              1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1), existing_features):
            new_features.append('transaction_stability')
    
    # High-frequency high-risk transactions
    if 'directed_1hop_out_timestamp_range' in existing_features and 'undirected_1hop_f16_max' in existing_features:
        if create_feature_safe(df, 'high_freq_high_value',
                              df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1), existing_features):
            new_features.append('high_freq_high_value')
    
    # Account activity anomaly
    if 'last_active_out' in existing_features and 'inactive_days_out' in existing_features:
        if create_feature_safe(df, 'activity_inconsistency',
                              df['last_active_out'] * df['inactive_days_out'], existing_features):
            new_features.append('activity_inconsistency')
    
    # Time pattern anomaly
    if 'undirected_1hop_timestamp_min' in existing_features and 'directed_1hop_out_timestamp_range' in existing_features:
        if create_feature_safe(df, 'early_wide_spread_risk',
                              df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range']), existing_features):
            new_features.append('early_wide_spread_risk')
    
    # Multiple risk overlay
    if all(f in existing_features for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
        if create_feature_safe(df, 'multi_risk_composite',
                              (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                               np.log1p(df['undirected_1hop_f16_max'])), existing_features):
            new_features.append('multi_risk_composite')
    
    # Core time feature interaction
    if all(f in existing_features for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
        if create_feature_safe(df, 'core_time_interaction',
                              (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * 
                              (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1)), existing_features):
            new_features.append('core_time_interaction')
    
    print(f"  Successfully created {len(new_features)} cross features")
    return df, new_features


def create_cross_features(df_, dataset_name):
    """Create cross features based on dataset."""
    if dataset_name == 'amazon':
        return create_cross_features_amazon(df_)
    elif dataset_name == 'yelpchi':
        return create_cross_features_yelpchi(df_)
    elif dataset_name == 'tfinance':
        return create_cross_features_tfinance(df_)
    elif dataset_name == 'dgraphfin':
        return create_cross_features_dgraphfin(df_)
    else:
        return df_, []


def load_test_data(data_dir, label_file, dataset_name):
    """加载测试数据、标签并进行特征选择"""
    print("=" * 60)
    print(f"Loading test data - Dataset: {dataset_name}")
    print("=" * 60)
    
    if dataset_name == 'tfinance':
        feature_base = data_dir
        
        # 加载各个特征文件
        print("Loading feature files...")
        df_node_enhanced = pd.read_pickle(os.path.join(feature_base, '01_base_features.pkl'))
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '02_structural_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '03_neighbor_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '04_spectral_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '05_embedding_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '06_clustering_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '07_anomaly_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '08_mixed_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '21_derived_features.pkl'))], axis=1)
        df_node_enhanced = pd.concat([df_node_enhanced, pd.read_pickle(os.path.join(feature_base, '09_advanced_financial_features.pkl'))], axis=1)
        
        print(f"  Feature data shape: {df_node_enhanced.shape}")
    else:
        # 对于其他数据集，加载常规的特征文件
        print("Loading feature files...")
        df_node = pd.read_pickle(os.path.join(data_dir, 'df_node_enhanced_mk.pkl'))
        feature_new = pd.read_pickle(os.path.join(data_dir, 'feature_new.pkl'))
        feature_null = pd.read_pickle(os.path.join(data_dir, 'feature_null.pkl'))
        df_node_enhanced = pd.concat([df_node, feature_new, feature_null], axis=1)
        
        print(f"  df_node_enhanced_mk: {df_node.shape}")
        print(f"  feature_new: {feature_new.shape}")
        print(f"  feature_null: {feature_null.shape}")
        print(f"  Merged feature data: {df_node_enhanced.shape}")
    
    # 加载标签
    print(f"\nLoading label data: {label_file}")
    data_np = np.load(label_file)
    y = data_np['y']
    print(f"  Label shape: {y.shape}")
    print(f"  Positive samples: {y.sum()}")
    print(f"  Positive ratio: {y.mean():.4f}")
    
    # 添加label列
    try:
        df_node_enhanced['label'] = [f[0] for f in y]
    except:
        df_node_enhanced['label'] = y
    
    print(f"  Final test data shape: {df_node_enhanced.shape}")
    
    # 返回元组 (df, y)，main函数会解包
    return df_node_enhanced, y


def select_features(df, cols_keep):
    """Select features based on predefined column list."""
    available_cols = [col for col in cols_keep if col in df.columns]
    missing_cols = [col for col in cols_keep if col not in df.columns]
    
    print(f"  Specified columns: {len(cols_keep)}")
    print(f"  Available columns: {len(available_cols)}")
    if missing_cols:
        print(f"  ⚠️ Missing columns: {missing_cols[:10]}...")
    
    df_selected = df[available_cols].copy()
    print(f"  Selected data shape: {df_selected.shape}")
    
    return df_selected


def load_models_and_predict(X_test, model_dir, model_type, n_folds=5):
    """加载各折模型并进行集成预测"""
    print(f"\n加载 {n_folds} 折 {model_type.upper()} 模型...")
    
    test_predictions = np.zeros(len(X_test))
    fold_predictions = []
    
    for fold in range(1, n_folds + 1):
        # 构建模型文件路径 - 与train.py保存格式一致
        if model_type == 'lgb':
            model_path = os.path.join(model_dir, f'lightgbm_fold_{fold}.txt')
        elif model_type == 'xgb':
            model_path = os.path.join(model_dir, f'xgboost_fold_{fold}.json')
        elif model_type == 'cab':
            model_path = os.path.join(model_dir, f'catboost_fold_{fold}.cbm')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"  ⚠️ Model file not found: {model_path}")
            continue
        
        print(f"  Loading fold {fold} model: {model_path}")
        
        # 加载模型
        if model_type == 'lgb':
            model = lgb.Booster(model_file=model_path)
            fold_pred = model.predict(X_test)
        elif model_type == 'xgb':
            model = xgb.Booster()
            model.load_model(model_path)
            dtest = xgb.DMatrix(X_test)
            fold_pred = model.predict(dtest)
        elif model_type == 'cab':
            model = CatBoostClassifier()
            model.load_model(model_path)
            fold_pred = model.predict_proba(X_test)[:, 1]
        
        test_predictions += fold_pred / n_folds
        fold_predictions.append(fold_pred)
    
    return test_predictions, fold_predictions


def evaluate_predictions(y_true, y_pred_proba, fold_predictions=None):
    """计算并输出评估指标"""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # 基础指标
    auc_score = roc_auc_score(y_true, y_pred_proba)
    y_pred_label = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    print(f"\n【Ensemble Model Results】")
    print(f"  AUC-ROC: {auc_score:.6f}")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  F1-Score: {f1:.6f}")
    print(f"  Average Precision: {ap_score:.6f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_label)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn}, FP: {fp}")
    print(f"    FN: {fn}, TP: {tp}")
    
    # 更多指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    # 各折指标
    if fold_predictions is not None:
        print(f"\n【Per-Fold Results】")
        fold_aucs = []
        fold_f1s = []
        
        for fold, fold_pred in enumerate(fold_predictions, 1):
            fold_auc = roc_auc_score(y_true, fold_pred)
            fold_pred_label = (fold_pred > 0.5).astype(int)
            fold_f1 = f1_score(y_true, fold_pred_label)
            
            fold_aucs.append(fold_auc)
            fold_f1s.append(fold_f1)
            
            print(f"  Fold {fold}: AUC = {fold_auc:.6f}, F1 = {fold_f1:.6f}")
        
        print(f"\n  Mean AUC: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
        print(f"  Mean F1: {np.mean(fold_f1s):.6f} ± {np.std(fold_f1s):.6f}")
    
    # 汇总所有指标
    metrics = {
        'auc': auc_score,
        'accuracy': accuracy,
        'f1': f1,
        'ap': ap_score,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'confusion_matrix': cm
    }
    
    return metrics


def save_results(y_true, y_pred_proba, fold_predictions, output_dir, dataset_name, model_type, metrics):
    """保存预测结果和评估指标"""
    print(f"\n保存结果到 {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存集成预测结果
    ensemble_path = os.path.join(output_dir, f'{model_type}_test_predictions.npz')
    np.savez(ensemble_path, 
             y_true=y_true, 
             y_pred=y_pred_proba,
             y_pred_label=(y_pred_proba > 0.5).astype(int))
    print(f"  Ensemble predictions: {ensemble_path}")
    
    # 保存各折预测结果
    for fold, fold_pred in enumerate(fold_predictions, 1):
        fold_path = os.path.join(output_dir, f'{model_type}_fold_{fold}_predictions.npy')
        np.save(fold_path, fold_pred)
    print(f"  Fold predictions saved to {output_dir}")
    
    # 保存评估指标摘要
    metrics_summary = {
        'dataset': dataset_name,
        'model': model_type,
        'metrics': metrics
    }
    metrics_path = os.path.join(output_dir, f'{model_type}_test_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_summary, f)
    print(f"  Metrics summary: {metrics_path}")
    
    # 保存详细结果CSV
    results_df = pd.DataFrame({
        'true_label': y_true,
        'pred_probability': y_pred_proba,
        'pred_label': (y_pred_proba > 0.5).astype(int)
    })
    csv_path = os.path.join(output_dir, f'{model_type}_test_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Results CSV: {csv_path}")


def main():
    """Main testing function"""
    args = parse_args()
    
    dataset_name = args.dataset
    model_type = args.model
    seed = args.seed
    n_folds = args.n_folds
    
    # 设置路径
    data_dir = os.path.join(args.data_dir, dataset_name, 'test')
    label_file = os.path.join(args.data_split_dir, f'{dataset_name}_test.npz')
    model_dir = os.path.join(args.model_dir, dataset_name)
    output_dir = model_dir
    
    print("=" * 60)
    print(f"{model_type.upper()} Model Testing")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    print(f"Feature directory: {data_dir}")
    print(f"Label file: {label_file}")
    print(f"Model directory: {model_dir}")
    print("=" * 60)
    
    # 1. 加载测试数据和标签
    print("\n" + "=" * 60)
    print("Loading test data...")
    print("=" * 60)
    df_test, y_test = load_test_data(data_dir, label_file, dataset_name)
    
    # 2. 特征选择
    print("\n" + "=" * 60)
    print("Selecting features...")
    print("=" * 60)
    
    COLS_KEEP = get_COLS_KEEP(dataset_name)
    df_selected = select_features(df_test, COLS_KEEP)
    
    # 3. 创建交叉特征
    print("\nCreating cross features...")
    df_cross, new_features = create_cross_features(df_selected, dataset_name)
    
    # 4. 构建最终特征列表
    cols_fea = [f for f in df_cross.columns if f != 'label']
    
    # 避免重复特征
    cols_fea_set = set(cols_fea)
    new_features_unique = [f for f in new_features if f not in cols_fea_set]
    cols_fea_cross = cols_fea + new_features_unique
    
    print(f"\nBase features: {len(cols_fea)}")
    print(f"Cross features: {len(new_features_unique)}")
    print(f"Final features: {len(cols_fea_cross)}")
    
    # 5. 准备测试数据
    X_test = df_cross[cols_fea_cross].reset_index(drop=True)
    print(f"\nTest samples: {len(X_test)}")
    print(f"Positive ratio: {y_test.mean():.4f}")
    
    # 6. 加载模型并进行预测
    print("\n" + "=" * 60)
    print("Loading models and making predictions...")
    print("=" * 60)
    
    test_predictions, fold_predictions = load_models_and_predict(
        X_test, model_dir, model_type, n_folds
    )
    
    # 7. 计算评估指标
    metrics = evaluate_predictions(y_test, test_predictions, fold_predictions)
    
    # 8. 保存结果
    save_results(y_test, test_predictions, fold_predictions, output_dir, 
                dataset_name, model_type, metrics)
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
