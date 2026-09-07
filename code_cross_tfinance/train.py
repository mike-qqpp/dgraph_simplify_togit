#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script with 5-fold cross-validation for stock prediction.

This script performs stratified k-fold cross-validation training on stock data
using LightGBM, XGBoost, or CatBoost models. It supports multiple datasets
including amazon, yelpchi, tfinance, and dgraphfin.
"""

import os
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train models with cross-validation')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Dataset name (e.g., amazon, yelpchi, tfinance, dgraphfin)')
    parser.add_argument('--model', type=str, default='cab', help='Model type: lgb, xgb, or cab')
    parser.add_argument('--data_dir', type=str, default='../feature_split',
                       help='Feature data directory (default: ../feature_split)')
    parser.add_argument('--data_split_dir', type=str, default='../data_split',
                       help='Data split directory (default: ../data_split)')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Model output directory (default: ./models)')
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
        COLS_FEA_SLC = ['fea_x_2', 'last_active_out', 'undirected_1hop_timestamp_min', 'fea_x_11', 'undirected_1hop_f16_max', 'directed_1hop_out_timestamp_max', 'fea_x_6', 'directed_1hop_out_timestamp_min', 'undirected_1hop_f15_max', 'fea_out_edge_type_mean', 'inactive_days_out', 'directed_1hop_out_timestamp_range', 'fea_x_1', 'fea_x_0', 'directed_1hop_out_interval_std', 'fea_out_edge_type_max', 'fea_x_15', 'fea_x_8', 'directed_1hop_out_interval_mean', 'directed_1hop_in_f16_mean', 'directed_1hop_in_f7_max', 'undirected_1hop_f16_mean', 'undirected_2hop_f12_max', 'directed_1hop_out_timestamp_std', 'undirected_1hop_interval_mean', 'directed_1hop_out_timestamp_mean', 'fea_out_edge_timestamp_diff_1_target_mean', 'fea_x_4', 'fea_out_edge_type_min', 'undirected_1hop_f1_min', 'fea_in_edge_type_mean', 'directed_1hop_in_f7_mean', 'undirected_1hop_interval_min', 'fea_out_edge_timestamp_diff_1_target_min', 'undirected_2hop_interval_std', 'undirected_2hop_f16_max', 'fea_out_edge_timestamp_diff_1_target_max', 'undirected_1hop_timestamp_mean', 'fea_out_neighbor_fea_x_13_max', 'undirected_2hop_interval_min', 'undirected_1hop_f7_max', 'directed_1hop_out_f16_mean', 'fea_in_edge_timestamp_diff_1_source_max', 'directed_1hop_in_f15_mean', 'undirected_2hop_f1_min', 'undirected_1hop_timestamp_range', 'directed_1hop_out_interval_max', 'undirected_2hop_f15_max', 'directed_1hop_out_type_5_max', 'fea_out_edge_type_mode', 'mixed_propagation_f4_max', 'fea_out_edge_timestamp_diff_1_target_mode', 'undirected_1hop_type_4_min', 'directed_1hop_in_f16_max', 'undirected_2hop_f1_mean', 'undirected_1hop_f11_max', 'undirected_2hop_f1_std', 'degree_ratio', 'directed_1hop_out_interval_min', 'undirected_1hop_f7_mean', 'fea_x_3', 'fea_in_edge_type_max', 'directed_1hop_in_f13_mean', 'active_days_out', 'undirected_2hop_f9_max', 'undirected_2hop_f13_max', 'undirected_1hop_type_5_std', 'fea_out_edge_type_std', 'undirected_1hop_timestamp_std', 'fea_x_12', 'undirected_1hop_f12_max', 'undirected_2hop_timestamp_min', 'undirected_1hop_f1_std', 'undirected_2hop_interval_max', 'undirected_1hop_interval_max', 'undirected_1hop_f13_mean', 'fea_x_7', 'undirected_1hop_timestamp_max', 'directed_1hop_in_f13_max', 'undirected_2hop_timestamp_mean', 'undirected_1hop_interval_std', 'directed_1hop_in_f15_max', 'undirected_2hop_f2_std', 'directed_1hop_in_f7_min', 'transaction_chain_interval_mean', 'directed_1hop_out_type_6_max', 'directed_2hop_out_type_4_std', 'undirected_2hop_type_5_std', 'fea_out_neighbor_fea_x_1_max', 'undirected_2hop_f11_max', 'undirected_1hop_f14_max', 'undirected_2hop_f9_max', 'undirected_1hop_f1_mean', 'transaction_chain_f16_mean', 'undirected_2hop_interval_mean', 'financial_hub_timestamp_min', 'undirected_2hop_f11_std', 'fea_x_16', 'undirected_1hop_f1_max', 'fea_x_9', 'undirected_2hop_type_4_range', 'directed_1hop_in_f12_max', 'directed_1hop_in_f16_min', 'fea_out_neighbor_fea_x_1_min', 'mixed_propagation_f15_max', 'directed_1hop_out_type_5_min', 'directed_1hop_out_f15_mean', 'mixed_propagation_f16_max', 'undirected_2hop_timestamp_max', 'directed_2hop_out_type_4_min', 'mixed_propagation_f11_max', 'directed_1hop_out_f1_max', 'total_degree', 'undirected_1hop_f12_std', 'directed_2hop_out_type_6_range', 'undirected_1hop_f13_std', 'directed_1hop_in_timestamp_min', 'undirected_1hop_f10_mean', 'undirected_2hop_timestamp_std', 'directed_1hop_out_type_5_mean', 'undirected_1hop_f5_max', 'undirected_2hop_f7_max', 'transaction_chain_f7_std', 'mixed_propagation_timestamp_min', 'directed_2hop_out_timestamp_mean', 'fea_out_neighbor_fea_x_1_mean', 'directed_2hop_in_f9_mean', 'directed_2hop_in_f1_mean', 'transaction_chain_type_6_min', 'undirected_2hop_f4_std']
    else:
        COLS_FEA_SLC = []
    
    COLS_KEEP = ['label'] + COLS_FEA_SLC
    return COLS_KEEP


def create_cross_features(df_, dataset_name):
    """Create cross features based on dataset."""
    df = df_.copy()
    new_features = []
    
    if dataset_name == 'amazon':
        # Time-related cross-cutting features
        if 'fea_x_19' in df.columns and 'fea_x_12' in df.columns:
            df['time_related_feature'] = df['fea_x_19'] * df['fea_x_12']
            new_features.append('time_related_feature')
        
        # Cross-cutting features associated with the structure of the figure
        if 'fea_out_neighbor_fea_x_0_max' in df.columns and 'fea_out_neighbor_fea_x_7_max' in df.columns:
            df['graph_structure_feature'] = df['fea_out_neighbor_fea_x_0_max'] + df['fea_out_neighbor_fea_x_7_max']
            new_features.append('graph_structure_feature')
        
        # Cross-cutting features associated with type distribution
        if 'td_type_ratio_7d_1' in df.columns and 'td_type2_30' in df.columns:
            df['type_distribution_feature'] = df['td_type_ratio_7d_1'] * df['td_type2_30']
            new_features.append('type_distribution_feature')
        
        # Cross feature of the border type associated with the time interval
        if 'fea_out_edge_type_mean' in df.columns and 'directed_1hop_in_type_0_count' in df.columns:
            df['edge_type_time_interval_feature'] = df['fea_out_edge_type_mean'] * df['directed_1hop_in_type_0_count']
            new_features.append('edge_type_time_interval_feature')
        
        # Cross-cutting features of risk communication and chart structure
        if 'mixed_propagation_type_4_count' in df.columns and 'directed_1hop_out_type_4_count' in df.columns:
            df['risk_propagation_feature'] = df['mixed_propagation_type_4_count'] + df['directed_1hop_out_type_4_count']
            new_features.append('risk_propagation_feature')
        
        # I'm not sure I'm going to do it.
        if 'undirected_1hop_type_0_count' in df.columns and 'undirected_2hop_type_0_count' in df.columns:
            df['hop_behavior_contrast'] = df['undirected_1hop_type_0_count'] - df['undirected_2hop_type_0_count']
            new_features.append('hop_behavior_contrast')
        
        # Operation-specific anomaly detection features
        if 'transaction_chain_f0_mean' in df.columns and 'directed_1hop_out_type_0_count' in df.columns:
            df['business_specific_anomaly'] = df['transaction_chain_f0_mean'] * df['directed_1hop_out_type_0_count']
            new_features.append('business_specific_anomaly')
        
        # Statistical cluster features
        if 'undirected_1hop_f5_std' in df.columns and 'undirected_2hop_type_0_count' in df.columns:
            df['statistical_combination'] = df['undirected_1hop_f5_std'] + df['undirected_2hop_type_0_count']
            new_features.append('statistical_combination')
        
        # Original identity stability cross-cutting feature
        if 'fea_x_2' in df.columns and 'fea_x_6' in df.columns:
            df['feature_stability'] = df['fea_x_2'] * df['fea_x_6']
            new_features.append('feature_stability')
    
    elif dataset_name == 'yelpchi':
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
    
    elif dataset_name == 'tfinance':        
        # 1. Intersectional features of abnormality detection strength (interactivity between abnormal fractions)
        if 'lof_score' in df.columns and 'anomaly_score_max' in df.columns:
            # Interaction of LOF fractions with maximum abnormal fractions
            df['lof_anomaly_max_interaction'] = df['lof_score'] * df['anomaly_score_max']
            new_features.append('lof_anomaly_max_interaction')
        
        if 'combined_max_anomaly_score' in df.columns and 'anomaly_harmonic_lof_score_one_class_svm_score' in df.columns:
            # Maximum abnormality fraction crosses the adjustment and abnormality fraction
            df['max_harmonic_anomaly_product'] = df['combined_max_anomaly_score'] * df['anomaly_harmonic_lof_score_one_class_svm_score']
            new_features.append('max_harmonic_anomaly_product')
        
        # 2. Cross-section of chart structure and anomaly detection (intersection of graph embedding and anomaly fractions)
        if 'gnn_embedding_dim0_exp' in df.columns and 'lof_score' in df.columns:
            # Figure embedded interaction with LOF fractions, possible capture structure anomalies
            df['graph_embedding_lof_interaction'] = df['gnn_embedding_dim0_exp'] * df['lof_score']
            new_features.append('graph_embedding_lof_interaction')
        
        if 'feature_embedding_dim1' in df.columns and 'important_ratio_feature_1_over_isolation_forest_score' in df.columns:
            # The intersection of feature embedding with abnormality detection ratio
            df['embedding_anomaly_ratio_product'] = df['feature_embedding_dim1'] * df['important_ratio_feature_1_over_isolation_forest_score']
            new_features.append('embedding_anomaly_ratio_product')
        
        # 3. Cross-cutting of time series features and statistical features
        if 'feature_1_lag2_diff' in df.columns and 'feature_skewness' in df.columns:
            # Crossing of lag differentials and deviations, possible capture of asymmetrics of time series
            df['lag_diff_skewness_interaction'] = df['feature_1_lag2_diff'] * df['feature_skewness']
            new_features.append('lag_diff_skewness_interaction')
        
        if 'feature_2_lag1_ratio' in df.columns and 'feature_kurtosis' in df.columns:
            # The lag ratio intersects with the peak and may capture the peak thick tail effect Response
            df['lag_ratio_kurtosis_interaction'] = df['feature_2_lag1_ratio'] * df['feature_kurtosis']
            new_features.append('lag_ratio_kurtosis_interaction')
        
        # 4. Cross-cutting of interactive features and abnormal ratings
        if 'interaction_7_8' in df.columns and 'important_ratio_feature_1_over_lof_score' in df.columns:
            # Cross-cutting of node 7-8 with abnormal ratio for feature 1
            df['interaction_anomaly_ratio_product'] = df['interaction_7_8'] * df['important_ratio_feature_1_over_lof_score']
            new_features.append('interaction_anomaly_ratio_product')
        
        if 'interaction_7_9' in df.columns and 'important_ratio_isolation_forest_score_over_lof_score' in df.columns:
            # Cross-referenced comparison of node 7-9 with abnormality detection methods
            df['interaction_anomaly_comparison_product'] = df['interaction_7_9'] * df['important_ratio_isolation_forest_score_over_lof_score']
            new_features.append('interaction_anomaly_comparison_product')
        
        # 5. Cross-cutting of cluster features and anomalies
        if 'kmeans_10_cluster_size' in df.columns and 'lof_score' in df.columns:
            # Cross-cutting of cluster size with local abnormalities
            df['cluster_size_lof_interaction'] = df['kmeans_10_cluster_size'] * df['lof_score']
            new_features.append('cluster_size_lof_interaction')
        
        if 'kmeans_5_nearest_cluster' in df.columns and 'mahalanobis_distance' in df.columns:
            # The recent intersection between the Allied Cluster and the Mars.
            df['nearest_cluster_mahalanobis_product'] = df['kmeans_5_nearest_cluster'] * df['mahalanobis_distance']
            new_features.append('nearest_cluster_mahalanobis_product')
        
        # 6. Cross-cutting of characterization and anomaly detection
        if 'feature_1_normalized' in df.columns and 'important_abs_diff_feature_1_isolation_forest_score' in df.columns:
            # Cross-cutting of standardized features and their abnormality detection
            df['normalized_feature_anomaly_diff_product'] = df['feature_1_normalized'] * df['important_abs_diff_feature_1_isolation_forest_score']
            new_features.append('normalized_feature_anomaly_diff_product')
        
        if 'feature_8_sqrt' in df.columns and 'important_diff_feature_1_minus_isolation_forest_score' in df.columns:
            # Crossing of square root transformation features and anomalies
            df['sqrt_feature_anomaly_diff_product'] = df['feature_8_sqrt'] * df['important_diff_feature_1_minus_isolation_forest_score']
            new_features.append('sqrt_feature_anomaly_diff_product')
        
        # 7. Cross-cutting of downscaling and statistical features
        if 'pca_component_6' in df.columns and 'feature_entropy' in df.columns:
            # Intersection of the main PCB component with information entropy
            df['pca_entropy_interaction'] = df['pca_component_6'] * df['feature_entropy']
            new_features.append('pca_entropy_interaction')
        
        if 'ica_component_8' in df.columns and 'feature_skewness' in df.columns:
            # Intersection of ICA independent constituents with bias
            df['ica_skewness_interaction'] = df['ica_component_8'] * df['feature_skewness']
            new_features.append('ica_skewness_interaction')
        
        # 8. Cross-cutting in relation to the importance of features
        if 'important_ratio_feature_1_over_lof_score' in df.columns and 'important_ratio_feature_2_over_lof_score' in df.columns:
            # Interaction of different features relative to LOF importance ratio
            df['feature_importance_ratio_product'] = df['important_ratio_feature_1_over_lof_score'] * df['important_ratio_feature_2_over_lof_score']
            new_features.append('feature_importance_ratio_product')
        
        if 'important_product_feature_1_lof_score' in df.columns and 'important_product_isolation_forest_score_one_class_svm_score' in df.columns:
            # Cross-cutting of the weight of different abnormality detection methods
            df['anomaly_importance_product_interaction'] = df['important_product_feature_1_lof_score'] * df['important_product_isolation_forest_score_one_class_svm_score']
            new_features.append('anomaly_importance_product_interaction')
        
        # 9. Cross-cutting of rolling statistics and anomaly detection
        if 'feature_0_rolling_max_ws10' in df.columns and 'important_ratio_isolation_forest_score_over_feature_entropy' in df.columns:
            # Cross-scroll maximum value with anomaly/information entropy ratio
            df['rolling_max_anomaly_entropy_ratio_product'] = df['feature_0_rolling_max_ws10'] * df['important_ratio_isolation_forest_score_over_feature_entropy']
            new_features.append('rolling_max_anomaly_entropy_ratio_product')
        
        if 'feature_2_rolling_max_ws5' in df.columns and 'lof_score' in df.columns:
            # Maximum short-term scrolling value crosses the LOF fraction
            df['short_rolling_max_lof_interaction'] = df['feature_2_rolling_max_ws5'] * df['lof_score']
            new_features.append('short_rolling_max_lof_interaction')
        
        # 10. Cross-cutting feature differences and graph embedding
        if 'orig_feat1_degree_diff' in df.columns and 'gnn_embedding_dim2' in df.columns:
            # Crossing of original feature differentials and embedded maps
            df['degree_diff_graph_embedding_product'] = df['orig_feat1_degree_diff'] * df['gnn_embedding_dim2']
            new_features.append('degree_diff_graph_embedding_product')
        
        if 'important_abs_diff_feature_1_laplacian_dim0' in df.columns and 'laplacian_dim7' in df.columns:
            # The intersection of La Plasz identity differences with La Plasvido
            df['laplacian_diff_dimension_product'] = df['important_abs_diff_feature_1_laplacian_dim0'] * df['laplacian_dim7']
            new_features.append('laplacian_diff_dimension_product')
        
        # 11. Cross-cutting of multiple features and abnormality detection
        if 'feature_1_poly2' in df.columns and 'important_diff_feature_1_minus_isolation_forest_score' in df.columns:
            # Cross-cutting of secondary polygraphs and anomalies
            df['poly2_feature_anomaly_diff_product'] = df['feature_1_poly2'] * df['important_diff_feature_1_minus_isolation_forest_score']
            new_features.append('poly2_feature_anomaly_diff_product')
        
        if 'poly_feat_7' in df.columns and 'anomaly_score_max' in df.columns:
            # Crossing of multiple features with maximum anomaly fraction
            df['poly_feature_anomaly_max_product'] = df['poly_feat_7'] * df['anomaly_score_max']
            new_features.append('poly_feature_anomaly_max_product')
        
        # 12. Cross-cutting of the feature drums and statistics
        if 'feature_8_bucket' in df.columns and 'feature_range' in df.columns:
            # Crossing of the feature sub-barrel and the value field
            df['bucket_range_interaction'] = df['feature_8_bucket'] * df['feature_range']
            new_features.append('bucket_range_interaction')
        
        if 'feature_6_bucket' in df.columns and 'feature_skewness' in df.columns:
            # Crossing of the feature sub-barrel and dichotomy
            df['bucket_skewness_interaction'] = df['feature_6_bucket'] * df['feature_skewness']
            new_features.append('bucket_skewness_interaction')
        
        # 13. Cross-checking of Chebyschev features with abnormalities
        if 'chebyshev_19' in df.columns and 'important_ratio_feature_1_over_isolation_forest_score' in df.columns:
            # Chebby Scheffer's Multiform and Unusual Ratios
            df['chebyshev_anomaly_ratio_product'] = df['chebyshev_19'] * df['important_ratio_feature_1_over_isolation_forest_score']
            new_features.append('chebyshev_anomaly_ratio_product')
        
        if 'chebyshev_11' in df.columns and 'lof_score' in df.columns:
            # Chebby Schiff crosses the LOF score
            df['chebyshev_lof_interaction'] = df['chebyshev_11'] * df['lof_score']
            new_features.append('chebyshev_lof_interaction')
        
        # Intersection of characteristic entropy with abnormality ratio
        if 'feature_entropy' in df.columns and 'important_ratio_isolation_forest_score_over_feature_entropy' in df.columns:
            # Intersection of information entropy with abnormality/permium ratio (secondary effect)
            df['entropy_anomaly_entropy_ratio_product'] = df['feature_entropy'] * df['important_ratio_isolation_forest_score_over_feature_entropy']
            new_features.append('entropy_anomaly_entropy_ratio_product')
        
        # 15. Cross-checking of feature ranking with abnormality detection
        if 'feature_1_rank' in df.columns and 'important_ratio_feature_1_over_lof_score' in df.columns:
            # Crossing of the feature ranking with its abnormal ratio
            df['rank_anomaly_ratio_product'] = df['feature_1_rank'] * df['important_ratio_feature_1_over_lof_score']
            new_features.append('rank_anomaly_ratio_product')
        
        # 16. Create some meaningful margin features (not simple product)
        if 'feature_max' in df.columns and 'feature_range' in df.columns and 'feature_max' not in [0, np.nan]:
            # Maximum value to range, reflecting distribution pattern
            df['max_to_range_ratio'] = df['feature_max'] / (df['feature_range'] + 1e-8)
            new_features.append('max_to_range_ratio')
        
        if 'lof_score' in df.columns and 'important_ratio_isolation_forest_score_over_lof_score' in df.columns:
            # Weighted combination of LOF fractions and abnormal methods
            df['lof_weighted_comparison'] = df['lof_score'] * np.log1p(df['important_ratio_isolation_forest_score_over_lof_score'])
            new_features.append('lof_weighted_comparison')
        
        if 'feature_skewness' in df.columns and 'feature_kurtosis' in df.columns:
            # The interaction of bias and peaks, reflecting non-normality of distribution
            df['skewness_kurtosis_interaction'] = df['feature_skewness'] * np.abs(df['feature_kurtosis'])
            new_features.append('skewness_kurtosis_interaction')
        
        # Ensure that only 30 new features are returned
        if len(new_features) > 30:
            new_features = new_features[:30]
            # Remove superfluous feature column
            for feat in new_features[30:]:
                if feat in df.columns:
                    df.drop(columns=[feat], inplace=True)
        
        print(f"Created successfully {len(new_features)} A meaningful cross-cutting feature.")
        print("New feature list:", new_features)
        
    
    elif dataset_name == 'dgraphfin':
        # Core Time Window Analysis
        if 'last_active_out' in df.columns and 'undirected_1hop_timestamp_min' in df.columns:
            df['active_window_span'] = df['last_active_out'] - df['undirected_1hop_timestamp_min']
            new_features.append('active_window_span')
        
        # High-risk transaction density
        if 'undirected_1hop_f16_max' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
            df['risk_transaction_density'] = df['undirected_1hop_f16_max'] / (df['directed_1hop_out_timestamp_range'] + 1e-6)
            new_features.append('risk_transaction_density')
        
        # Recent surprise sex deals
        if 'last_active_out' in df.columns and 'undirected_1hop_f16_max' in df.columns:
            df['recent_burst_risk'] = (1 / (df['last_active_out'] + 1)) * df['undirected_1hop_f16_max']
            new_features.append('recent_burst_risk')
        
        # Transaction time is abnormal.
        if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
            df['timing_anomaly_score'] = df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std']
            new_features.append('timing_anomaly_score')
        
        # Core identity combination
        if all(f in df.columns for f in ['fea_x_2', 'fea_x_11', 'fea_x_6']):
            df['core_feature_interaction'] = df['fea_x_2'] * df['fea_x_11'] / (np.abs(df['fea_x_6']) + 1e-6)
            new_features.append('core_feature_interaction')
        
        # Risk of inactive accounts
        if 'inactive_days_out' in df.columns and 'undirected_1hop_f15_max' in df.columns:
            df['dormant_high_value_risk'] = df['inactive_days_out'] * df['undirected_1hop_f15_max']
            new_features.append('dormant_high_value_risk')
        
        # Transaction time concentration
        if 'directed_1hop_out_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_max' in df.columns:
            df['out_timestamp_concentration'] = 1 / (df['directed_1hop_out_timestamp_max'] - df['directed_1hop_out_timestamp_min'] + 1)
            new_features.append('out_timestamp_concentration')
        
        # Border type abnormal.
        if 'fea_out_edge_type_mean' in df.columns and 'fea_out_edge_type_max' in df.columns:
            df['edge_type_discrepancy'] = df['fea_out_edge_type_max'] - df['fea_out_edge_type_mean']
            new_features.append('edge_type_discrepancy')
        
        # Time stamp minimum value abnormal
        if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_min' in df.columns:
            df['first_interaction_gap'] = df['directed_1hop_out_timestamp_min'] - df['undirected_1hop_timestamp_min']
            new_features.append('first_interaction_gap')
        
        # Weighted combination of feature x
        if all(f in df.columns for f in ['fea_x_2', 'fea_x_1', 'fea_x_0', 'fea_x_15', 'fea_x_8']):
            df['weighted_x_features'] = (df['fea_x_2'] * 0.35 + df['fea_x_1'] * 0.25 + 
                                        df['fea_x_0'] * 0.20 + df['fea_x_15'] * 0.15 + 
                                        df['fea_x_8'] * 0.05)
            new_features.append('weighted_x_features')
        
        # Transaction stability rating
        if 'directed_1hop_out_timestamp_range' in df.columns and 'directed_1hop_out_interval_std' in df.columns:
            df['transaction_stability'] = 1 / (df['directed_1hop_out_timestamp_range'] * df['directed_1hop_out_interval_std'] + 1)
            new_features.append('transaction_stability')
        
        # High frequency risk trading
        if 'directed_1hop_out_timestamp_range' in df.columns and 'undirected_1hop_f16_max' in df.columns:
            df['high_freq_high_value'] = df['undirected_1hop_f16_max'] / (np.log1p(df['directed_1hop_out_timestamp_range']) + 1)
            new_features.append('high_freq_high_value')
        
        # Account activity abnormal.
        if 'last_active_out' in df.columns and 'inactive_days_out' in df.columns:
            df['activity_inconsistency'] = df['last_active_out'] * df['inactive_days_out']
            new_features.append('activity_inconsistency')
        
        # Time pattern abnormal.
        if 'undirected_1hop_timestamp_min' in df.columns and 'directed_1hop_out_timestamp_range' in df.columns:
            df['early_wide_spread_risk'] = df['undirected_1hop_timestamp_min'] * np.log1p(df['directed_1hop_out_timestamp_range'])
            new_features.append('early_wide_spread_risk')
        
        # Multiple risk superimposed
        if all(f in df.columns for f in ['fea_x_2', 'last_active_out', 'undirected_1hop_f16_max']):
            df['multi_risk_composite'] = (df['fea_x_2'] * (1 / (df['last_active_out'] + 1)) * 
                                         np.log1p(df['undirected_1hop_f16_max']))
            new_features.append('multi_risk_composite')
        
        # Core time features interactive
        if all(f in df.columns for f in ['last_active_out', 'undirected_1hop_timestamp_min', 'directed_1hop_out_timestamp_max']):
            df['core_time_interaction'] = (df['last_active_out'] - df['undirected_1hop_timestamp_min']) * \
                                         (1 / (df['directed_1hop_out_timestamp_max'] - df['undirected_1hop_timestamp_min'] + 1))
            new_features.append('core_time_interaction')
    
    print(f"Created successfully {len(new_features)} Cross-cutting features")
    return df, new_features


def load_train_data(data_dir, label_file, dataset_name):
    """Load training data, labels and select features"""
    print("=" * 60)
    print(f"Loading training data - Dataset: {dataset_name}")
    print("=" * 60)
    
    if dataset_name == 'tfinance':
        feature_base = data_dir
        
        # Loading all feature files
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
        
        # Load Tabs
        print(f"\nLoading label data: {label_file}")
        data_np = np.load(label_file)
        y = data_np['y']
        print(f"  Label shape: {y.shape}")
        print(f"  Positive samples: {y.sum()}")
        print(f"  Positive ratio: {y.mean():.4f}")
        
        # Add Label Column to DataFrame
        try:
            df_node_enhanced['label'] = [f[0] for f in y]
        except:
            df_node_enhanced['label'] = y
        
        print(f"  Final training data shape: {df_node_enhanced.shape}")
        
        # Only DataFrame returns, not members
        return df_node_enhanced
    
    else:
        # Processing logic for other data sets
        df_train = pd.read_pickle(os.path.join(data_dir, 'df_node_enhanced_mk.pkl'))
        feature_new = pd.read_pickle(os.path.join(data_dir, 'feature_new.pkl'))
        feature_null = pd.read_pickle(os.path.join(data_dir, 'feature_null.pkl'))
        
        print(f"  df_node_enhanced_mk: {df_train.shape}")
        print(f"  feature_new: {feature_new.shape}")
        print(f"  feature_null: {feature_null.shape}")
        
        # Merge Features
        df_train = pd.concat([df_train, feature_new, feature_null], axis=1)
        print(f"\n  Merged feature data: {df_train.shape}")
        
        # Test the merged repeat column
        duplicate_cols_after = df_train.columns[df_train.columns.duplicated()].tolist()
        if duplicate_cols_after:
            print(f"  ⚠️ Found {len(duplicate_cols_after)} duplicate columns, processing...")
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
            print(f"  ✓ Processed duplicates, final columns: {len(df_train.columns)}")
        
        # Load Tabs
        print(f"\nLoading label data: {label_file}")
        label_np = np.load(label_file)
        labels = label_np['y']
        print(f"  Label shape: {labels.shape}")
        print(f"  Positive samples: {labels.sum()}")
        print(f"  Positive ratio: {labels.mean():.4f}")
        
        # Add Tab
        df_train['label'] = labels
        print(f"  Final training data shape: {df_train.shape}")
        
        return df_train


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


def train_lgb(X_train, y_train, X_test, feature_names, seed, n_folds, output_dir):
    """Train LightGBM model with cross-validation."""
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': seed
    }
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Storage of indicators for each discount
    fold_aucs = []
    fold_f1s = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Saves every discount model
        model_path = os.path.join(output_dir, f'lightgbm_fold_{fold + 1}.txt')
        model.save_model(model_path)
        
        oof_preds[valid_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds
        
        # Indicators for calculating the current discount
        fold_auc = roc_auc_score(y_val, oof_preds[valid_idx])
        fold_f1 = f1_score(y_val, (oof_preds[valid_idx] > 0.5).astype(int))
        fold_aucs.append(fold_auc)
        fold_f1s.append(fold_f1)
        
        print(f'Fold {fold + 1} AUC: {fold_auc:.6f}, F1: {fold_f1:.6f}')
    
    # Compute overall indicators
    cv_metrics = {
        'auc_mean': np.mean(fold_aucs),
        'auc_std': np.std(fold_aucs),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s)
    }
    
    print(f"\nCV Results: AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}, "
          f"F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    
    return oof_preds, test_preds, cv_metrics


def train_xgb(X_train, y_train, X_test, feature_names, seed, n_folds, output_dir):
    """Train XGBoost model with cross-validation."""
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': seed,
        'verbosity': 0
    }
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Storage of indicators for each discount
    fold_aucs = []
    fold_f1s = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Saves every discount model
        model_path = os.path.join(output_dir, f'xgboost_fold_{fold + 1}.json')
        model.save_model(model_path)
        
        oof_preds[valid_idx] = model.predict(dvalid)
        test_preds += model.predict(dtest) / n_folds
        
        # Indicators for calculating the current discount
        fold_auc = roc_auc_score(y_val, oof_preds[valid_idx])
        fold_f1 = f1_score(y_val, (oof_preds[valid_idx] > 0.5).astype(int))
        fold_aucs.append(fold_auc)
        fold_f1s.append(fold_f1)
        
        print(f'Fold {fold + 1} AUC: {fold_auc:.6f}, F1: {fold_f1:.6f}')
    
    # Compute overall indicators
    cv_metrics = {
        'auc_mean': np.mean(fold_aucs),
        'auc_std': np.std(fold_aucs),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s)
    }
    
    print(f"\nCV Results: AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}, "
          f"F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    
    return oof_preds, test_preds, cv_metrics


def train_cab(X_train, y_train, X_test, feature_names, seed, n_folds, output_dir):
    """Train CatBoost model with cross-validation."""
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Storage of indicators for each discount
    fold_aucs = []
    fold_f1s = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=seed,
            verbose=100,
            early_stopping_rounds=50
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        
        # Saves every discount model
        model_path = os.path.join(output_dir, f'catboost_fold_{fold + 1}.cbm')
        model.save_model(model_path)
        
        oof_preds[valid_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        
        # Indicators for calculating the current discount
        fold_auc = roc_auc_score(y_val, oof_preds[valid_idx])
        fold_f1 = f1_score(y_val, (oof_preds[valid_idx] > 0.5).astype(int))
        fold_aucs.append(fold_auc)
        fold_f1s.append(fold_f1)
        
        print(f'Fold {fold + 1} AUC: {fold_auc:.6f}, F1: {fold_f1:.6f}')
    
    # Compute overall indicators
    cv_metrics = {
        'auc_mean': np.mean(fold_aucs),
        'auc_std': np.std(fold_aucs),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s)
    }
    
    print(f"\nCV Results: AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}, "
          f"F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    
    return oof_preds, test_preds, cv_metrics


def main():
    """Main training function with cross-validation."""
    args = parse_args()
    
    dataset_name = args.dataset
    model_name = args.model
    seed = args.seed
    n_folds = args.n_folds
    
    # Set Path
    data_dir = os.path.join(args.data_dir, dataset_name, 'train')
    label_file = os.path.join(args.data_split_dir, f'{dataset_name}_train.npz')
    output_dir = os.path.join(args.output_dir, dataset_name)
    
    print("=" * 60)
    print(f"{model_name.upper()} {n_folds}-Fold Cross-Validation Training")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    print(f"Feature directory: {data_dir}")
    print(f"Label file: {label_file}")
    print(f"Output directory: {output_dir}")
    print(f"Folds: {n_folds}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Ensure that the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load training data and labelling
    df_train = load_train_data(data_dir, label_file, dataset_name)
    
    # 2. Identity selection
    print("\n" + "=" * 60)
    print("Selecting features...")
    print("=" * 60)
    
    COLS_KEEP = get_COLS_KEEP(dataset_name)
    df_selected = select_features(df_train, COLS_KEEP)
    
    # 3. Create cross-cutting features
    print("\nCreating cross features...")
    df_cross, new_features = create_cross_features(df_selected, dataset_name)
    
    # 4. Build final feature list
    cols_fea = [f for f in df_cross.columns if f != 'label']
    
    # Avoid duplication of features
    cols_fea_set = set(cols_fea)
    new_features_unique = [f for f in new_features if f not in cols_fea_set]
    cols_fea_cross = cols_fea + new_features_unique
    
    print(f"\nBase features: {len(cols_fea)}")
    print(f"Cross features: {len(new_features_unique)}")
    print(f"Final features: {len(cols_fea_cross)}")
    
    # Preparation of training data
    X = df_cross[cols_fea_cross]
    y = df_cross['label'].values
    X_test = X.copy()
    
    print(f"\nTraining samples: {len(X)}")
    print(f"Positive ratio: {y.mean():.4f}")
    
    # 6. Training models
    print("\n" + "=" * 60)
    print(f"Training {model_name.upper()} model...")
    print("=" * 60)
    
    if model_name == 'lgb':
        oof_preds, test_preds, cv_metrics = train_lgb(X, y, X_test, cols_fea_cross, seed, n_folds, output_dir)
    elif model_name == 'xgb':
        oof_preds, test_preds, cv_metrics = train_xgb(X, y, X_test, cols_fea_cross, seed, n_folds, output_dir)
    elif model_name == 'cab':
        oof_preds, test_preds, cv_metrics = train_cab(X, y, X_test, cols_fea_cross, seed, n_folds, output_dir)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    # 7. Calculation of overall indicators
    total_auc = roc_auc_score(y, oof_preds)
    total_f1 = f1_score(y, (oof_preds > 0.5).astype(int))
    total_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print("=" * 60)
    print(f"\nFile Generation:")
    print(f"  - Model File: {output_dir}/{model_name}_fold_*.cbm")
    print(f"  - Training metadata: {output_dir}/training_metadata.pkl")
    print(f"  - OOFprediction: {output_dir}/oof_predictions.npy")
    print(f"\nTraining indicators:")
    print(f"  AUC: {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
    print(f"  F1: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    
    # 8. Preservation of results
    print(f"\nSaving results to {output_dir}...")
    
    # Save OOF projections
    oof_path = os.path.join(output_dir, 'oof_predictions.npy')
    np.save(oof_path, oof_preds)
    print(f"  OOF predictions: {oof_path}")
    
    # Save test predictions
    test_path = os.path.join(output_dir, f'{model_name}_test.npz')
    np.savez(test_path, test_preds=test_preds)
    print(f"  Test predictions: {test_path}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'model': model_name,
        'cols_fea_cross': cols_fea_cross,
        'n_folds': n_folds,
        'seed': seed,
        'auc_mean': cv_metrics['auc_mean'],
        'auc_std': cv_metrics['auc_std'],
        'f1_mean': cv_metrics['f1_mean'],
        'f1_std': cv_metrics['f1_std'],
        'total_auc': total_auc,
        'total_f1': total_f1,
        'total_accuracy': total_acc
    }
    metadata_path = os.path.join(output_dir, 'training_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata: {metadata_path}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
