#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 2025 CCF DGraph 初赛追加空值信息特征 - 修复版本
运行: python add_null_features.py --phase 1
产出: data/phase1/phase1_feature_null.pkl
"""
import os
import gc
import pickle
import click
import numpy as np
import pandas as pd
import argparse


# 1. 建立解析器
parser = argparse.ArgumentParser(description='makefea_part3')

# bin_dict = pickle.load(open(opj(work_path,'feature','bin_dict.pkl'), 'rb'))
# bin_prob_dict = pickle.load(open(opj(work_path,'feature','bin_prob_dict.pkl'), 'rb'))
# data = np.load(opj(data_path,data_name,'raw','gdata.npz'))
# path = opj(data_path,data_name,'feature.pkl')  # 改为.pkl

# 2. 定义参数
parser.add_argument('--path_bin_dict', type=str, default='../feature/phase1/bin_dict.pkl')
parser.add_argument('--path_bin_prob_dict', type=str, default='../feature/phase1/bin_prob_dict.pkl')
parser.add_argument('--path_data', type=str, default='../data/phase1/gdata.npz')
parser.add_argument('--path_save_feature', type=str, default='../feature/phase1/feature.pkl')
parser.add_argument('--path_save_feature_supplement', type=str, default='../feature/phase1/feature_supplement.pkl')
parser.add_argument('--path_save_feature_null', type=str, default='../feature/phase1/feature_null.pkl')
parser.add_argument('--path_save_feature_mk', type=str, default='../feature/phase1/df_node_enhanced_mk.pkl')
parser.add_argument('--sub_ratio', type=float, default=1.0)

# 3. 解析命令行
args = parser.parse_args()

def safe_div(a, b, fill=0):
    """安全除法 - 修复版本"""
    if hasattr(a, '__len__') and hasattr(b, '__len__'):
        # 处理数组情况
        result = np.full_like(a, fill, dtype=float)
        mask = b != 0
        result[mask] = a[mask] / b[mask]
        return result
    else:
        # 处理标量情况
        if b == 0:
            return fill
        return a / b

def add_advanced_null_pattern_features(df, x):
    """1. 高级空值模式特征 - 修复版本"""
    print("添加高级空值模式特征...")
    
    # 原始特征的空值标记 (-1表示空值)
    null_mask = (x == -1)
    n_nodes = x.shape[0]
    
    # 1.1 基础空值统计
    null_count = null_mask.sum(axis=1)
    df['null_count'] = null_count
    df['null_ratio'] = safe_div(null_count, x.shape[1])
    
    # 1.2 关键特征空值标记
    for i in range(5):  # 只标记前5个关键特征
        df[f'f{i}_is_null'] = (x[:, i] == -1).astype(np.float32)
    
    # 1.3 空值分布特征
    # 前部特征空值（假设前8个特征更重要）
    front_features = list(range(8))
    front_null_ratio = null_mask[:, front_features].mean(axis=1)
    df['front_feature_null_ratio'] = front_null_ratio
    
    # 后部特征空值
    back_features = list(range(8, 17))
    back_null_ratio = null_mask[:, back_features].mean(axis=1)
    df['back_feature_null_ratio'] = back_null_ratio
    
    # 1.4 空值模式简单统计
    df['null_pattern_mean'] = null_mask.mean(axis=1)
    df['null_pattern_std'] = null_mask.std(axis=1)
    
    return df

def add_simple_neighbor_null_features(df, edge_df, x):
    """2. 简化的邻居空值特征 - 避免内存爆炸"""
    print("添加简化的邻居空值特征...")
    
    null_mask = (x == -1)
    node_null_ratio = null_mask.mean(axis=1)
    n_nodes = len(node_null_ratio)
    
    # 2.1 出边邻居空值特征
    print("计算出边邻居空值特征...")
    out_stats = calculate_simple_neighbor_stats(edge_df, '0', '1', node_null_ratio, n_nodes)
    df['out_nei_null_ratio_mean'] = out_stats['mean']
    df['out_nei_null_ratio_max'] = out_stats['max']
    df['out_nei_high_null_ratio'] = out_stats['high_ratio']
    
    # 2.2 入边邻居空值特征
    print("计算入边邻居空值特征...")
    in_stats = calculate_simple_neighbor_stats(edge_df, '1', '0', node_null_ratio, n_nodes)
    df['in_nei_null_ratio_mean'] = in_stats['mean']
    df['in_nei_null_ratio_max'] = in_stats['max']
    df['in_nei_high_null_ratio'] = in_stats['high_ratio']
    
    return df

def calculate_simple_neighbor_stats(edge_df, src_col, dst_col, node_null_ratio, n_nodes):
    """计算简化的邻居统计 - 使用向量化操作"""
    # 预计算分组
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
    """3. 时间相关的空值特征 - 简化版本"""
    print("添加时间相关的空值特征...")
    
    null_mask = (x == -1)
    node_null_ratio = null_mask.mean(axis=1)
    max_timestamp = np.max(edge_timestamp)
    n_nodes = len(node_null_ratio)
    
    # 只计算最近30天的特征
    window = 30
    print(f"计算{window}天时间窗口空值特征...")
    
    # 最近window天的边
    recent_mask = edge_df['edge_timestamp'] >= (max_timestamp - window + 1)
    recent_edges = edge_df[recent_mask]
    
    # 出边邻居时间窗口特征
    out_recent_stats = calculate_simple_neighbor_stats(recent_edges, '0', '1', node_null_ratio, n_nodes)
    df[f'out_recent_{window}d_null_mean'] = out_recent_stats['mean']
    
    # 入边邻居时间窗口特征
    in_recent_stats = calculate_simple_neighbor_stats(recent_edges, '1', '0', node_null_ratio, n_nodes)
    df[f'in_recent_{window}d_null_mean'] = in_recent_stats['mean']
    
    return df

def add_null_risk_features(df, x, edge_df):
    """4. 空值风险特征"""
    print("添加空值风险特征...")
    
    null_mask = (x == -1)
    node_null_ratio = null_mask.mean(axis=1)
    n_nodes = len(node_null_ratio)
    
    # 4.1 基础风险分数
    df['null_risk_basic'] = node_null_ratio
    
    # 4.2 关键特征风险
    key_features_null = null_mask[:, :5].any(axis=1)
    df['key_feature_null_risk'] = key_features_null.astype(np.float32)
    
    # 4.3 网络风险传播
    # 计算节点的简单度统计
    out_degree_dict = edge_df.groupby('0').size().to_dict()
    in_degree_dict = edge_df.groupby('1').size().to_dict()
    
    # 合并度统计
    total_degree = np.zeros(n_nodes)
    for i in range(n_nodes):
        total_degree[i] = out_degree_dict.get(i, 0) + in_degree_dict.get(i, 0)
    
    # 度加权风险
    degree_weights = np.minimum(total_degree / 1000, 1.0)  # 限制权重范围
    df['degree_weighted_null_risk'] = node_null_ratio * degree_weights
    
    # 4.4 空值异常分数
    null_mean = np.mean(node_null_ratio)
    null_std = np.std(node_null_ratio)
    if null_std > 0:
        df['null_anomaly_score'] = (node_null_ratio - null_mean) / null_std
    else:
        df['null_anomaly_score'] = 0
    
    return df

def add_null_interaction_features(df, x):
    """5. 空值交互特征"""
    print("添加空值交互特征...")
    
    null_mask = (x == -1)
    
    # 5.1 空值组合模式
    # 前3个特征的空值组合
    f0_null = null_mask[:, 0]
    f1_null = null_mask[:, 1]
    f2_null = null_mask[:, 2]
    
    df['f0_f1_null_both'] = (f0_null & f1_null).astype(np.float32)
    df['f0_f2_null_both'] = (f0_null & f2_null).astype(np.float32)
    df['f1_f2_null_both'] = (f1_null & f2_null).astype(np.float32)
    df['f0_f1_f2_null_all'] = (f0_null & f1_null & f2_null).astype(np.float32)
    
    # 5.2 空值模式多样性
    # 计算空值在不同特征组中的分布
    group1_null = null_mask[:, :6].mean(axis=1)  # 前半部分特征
    group2_null = null_mask[:, 6:12].mean(axis=1)  # 中间部分特征
    group3_null = null_mask[:, 12:].mean(axis=1)  # 后半部分特征
    
    # 计算标准差作为多样性指标
    null_groups = np.stack([group1_null, group2_null, group3_null])
    df['null_group_diversity'] = np.std(null_groups, axis=0)
    
    return df


def main():

    # print(f'=== 生成 {phase_name} 空值特征 ===')
    
    # 1. 读原始数据
    raw_path = args.path_data
    print(f"加载数据: {raw_path}")
    data = np.load(raw_path, allow_pickle='True')
    
    x = data['x']
    edge_index = data['edge_index']
    edge_type = data['edge_type'] 
    edge_timestamp = data['edge_timestamp']
    
    print(f"数据形状: x={x.shape}, 边数={edge_index.shape[0]}")
    
    # 2. 创建边数据框
    edge_data = np.column_stack([
        edge_index[:, 0].astype(np.int32),
        edge_index[:, 1].astype(np.int32),
        edge_type.astype(np.int32),
        edge_timestamp.astype(np.int32)
    ])
    
    edge_df = pd.DataFrame(edge_data, columns=['0', '1', 'edge_type', 'edge_timestamp'])

    n_init = edge_df.shape[0]
    print('->'*10, 'df_edge init 样本量为: {}'.format(n_init) )
    if args.sub_ratio<1:
        
        edge_df = edge_df.sample(frac = args.sub_ratio, random_state=42)   # 返回新 DataFrame
        n_subsample = edge_df.shape[0]
        print('->'*10, 'df_edge 下采样比例为: {}, 下采样后样本量为: {}'.format(n_init, n_subsample) )

    
    N = x.shape[0]
    df = pd.DataFrame(index=range(N))
    
    # 3. 添加空值特征 - 分步进行，及时清理内存
    print("\n开始空值特征计算...")
    
    try:
        # 第一步：节点本身的空值特征
        print("步骤1: 节点空值特征...")
        df = add_advanced_null_pattern_features(df, x)
        gc.collect()
        
        # 第二步：空值交互特征
        print("步骤2: 空值交互特征...")
        df = add_null_interaction_features(df, x)
        gc.collect()
        
        # 第三步：邻居空值特征（最耗时的部分）
        print("步骤3: 邻居空值特征...")
        df = add_simple_neighbor_null_features(df, edge_df, x)
        gc.collect()
        
        # 第四步：时间空值特征
        print("步骤4: 时间空值特征...")
        df = add_temporal_null_features(df, edge_df, x, edge_timestamp)
        gc.collect()
        
        # 第五步：风险特征
        print("步骤5: 风险特征...")
        df = add_null_risk_features(df, x, edge_df)
        gc.collect()
        
    except Exception as e:
        print(f"特征计算过程中出现错误: {e}")
        print("尝试保存已计算的特征...")
    
    # 4. 数据清理和优化
    print("\n数据清理和优化...")
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    # 类型转换
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    
    # 5. 保存结果
    out_path = args.path_save_feature_null
    with open(out_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f'✅ 空值特征已保存: {out_path}')
    print(f'最终特征形状: {df.shape}')
    print(f'特征数量: {len(df.columns)}')
    
    # 显示特征类别
    node_features = len([c for c in df.columns if 'f' in c and 'null' in c])
    neighbor_features = len([c for c in df.columns if 'nei' in c])
    risk_features = len([c for c in df.columns if 'risk' in c or 'anomaly' in c])
    temporal_features = len([c for c in df.columns if 'recent' in c or 'd_' in c])
    interaction_features = len([c for c in df.columns if 'both' in c or 'all' in c or 'diversity' in c])
    
    print(f'特征类别统计:')
    print(f'  - 节点空值特征: {node_features}')
    print(f'  - 邻居传播特征: {neighbor_features}')
    print(f'  - 时间空值特征: {temporal_features}')
    print(f'  - 风险指示器: {risk_features}')
    print(f'  - 交互特征: {interaction_features}')

if __name__ == '__main__':
    main()