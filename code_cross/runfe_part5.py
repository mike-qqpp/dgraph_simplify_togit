#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点类型分布特征 - 纯DataFrame向量化版本
完全消除for循环，极致性能
运行: python feature_node_type_vectorized.py
产出: ../feature/phase1/feature_node_type_vectorized.pkl
"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
import argparse

# 1. 建立解析器
parser = argparse.ArgumentParser(description='feature_node_type_vectorized')

# 2. 定义参数
parser.add_argument('--path_data', type=str, default='../data/phase1/gdata.npz')
parser.add_argument('--path_save_feature_type', type=str, default='../feature/phase1/feature_node_type_vectorized.pkl')
parser.add_argument('--sub_ratio', type=float, default=1.0)

# 3. 解析命令行
args = parser.parse_args()

def safe_div(a, b, fill=0.0):
    """安全除法"""
    return np.divide(a, b, out=np.full_like(a, fill, dtype=float), where=b!=0)

def create_node_type_mapping(y):
    """创建节点类型映射"""
    node_type = np.zeros_like(y)
    node_type[np.isin(y, [0, 1, -100])] = 0  # 类型0: 0, 1, -100
    node_type[y == 2] = 1                     # 类型1: 2
    node_type[y == 3] = 2                     # 类型2: 3
    return node_type

def create_vectorized_neighbor_type_features(edge_index, node_type):
    """纯向量化邻居类型特征"""
    print("创建向量化邻居类型特征...")
    
    src = edge_index[:, 0].flatten()
    dst = edge_index[:, 1].flatten()
    N = len(node_type)
    
    # 创建边DataFrame
    edge_df = pd.DataFrame({'src': src, 'dst': dst})

    n_init = edge_df.shape[0]
    print('->'*10, 'df_edge init 样本量为: {}'.format(n_init) )
    if args.sub_ratio<1:
        
        edge_df = edge_df.sample(frac = args.sub_ratio, random_state=42)   # 返回新 DataFrame
        n_subsample = edge_df.shape[0]
        print('->'*10, 'df_edge 下采样比例为: {}, 下采样后样本量为: {}'.format(n_init, n_subsample) )

    
    # 创建节点类型DataFrame
    node_type_series = pd.Series(node_type, name='node_type')
    
    # 1. 出边邻居类型统计
    print("  计算出边邻居类型...")
    out_with_types = edge_df.merge(
        node_type_series.rename('dst_type'), left_on='dst', right_index=True
    )
    out_type_counts = pd.crosstab(out_with_types['src'], out_with_types['dst_type'])
    out_type_counts = out_type_counts.reindex(columns=[0,1,2], fill_value=0)
    out_type_counts.columns = [f'out_type_{i}_cnt' for i in out_type_counts.columns]
    
    # 2. 入边邻居类型统计
    print("  计算入边邻居类型...")
    in_with_types = edge_df.merge(
        node_type_series.rename('src_type'), left_on='src', right_index=True
    )
    in_type_counts = pd.crosstab(in_with_types['dst'], in_with_types['src_type'])
    in_type_counts = in_type_counts.reindex(columns=[0,1,2], fill_value=0)
    in_type_counts.columns = [f'in_type_{i}_cnt' for i in in_type_counts.columns]
    
    # 3. 创建基础特征DataFrame
    features = pd.DataFrame(index=range(N))
    features = features.join(out_type_counts, how='left')
    features = features.join(in_type_counts, how='left')
    features = features.fillna(0)
    
    # 4. 计算度特征
    print("  计算度特征...")
    features['out_deg'] = edge_df.groupby('src').size().reindex(features.index, fill_value=0)
    features['in_deg'] = edge_df.groupby('dst').size().reindex(features.index, fill_value=0)
    features['total_deg'] = features['out_deg'] + features['in_deg']
    
    # 5. 计算类型占比 - 向量化操作
    print("  计算类型占比...")
    for i in range(3):
        features[f'out_type_{i}_ratio'] = safe_div(features[f'out_type_{i}_cnt'], features['out_deg'])
        features[f'in_type_{i}_ratio'] = safe_div(features[f'in_type_{i}_cnt'], features['in_deg'])
    
    # 6. 主要邻居类型 - 向量化操作
    print("  计算主要邻居类型...")
    out_cnt_cols = [f'out_type_{i}_cnt' for i in range(3)]
    in_cnt_cols = [f'in_type_{i}_cnt' for i in range(3)]
    
    features['out_main_type'] = features[out_cnt_cols].idxmax(axis=1).str.replace('out_type_', '').str.replace('_cnt', '').astype(int)
    features['in_main_type'] = features[in_cnt_cols].idxmax(axis=1).str.replace('in_type_', '').str.replace('_cnt', '').astype(int)
    
    # 7. 同类型连接 - 向量化操作
    print("  计算同类型连接...")
    node_type_df = pd.DataFrame({'node_type': node_type}, index=range(N))
    features = features.join(node_type_df, how='left')
    
    # 使用lookup方法向量化计算同类型连接
    out_same_mask = pd.Series(range(N)).apply(lambda x: f'out_type_{features.loc[x, "node_type"]}_ratio')
    in_same_mask = pd.Series(range(N)).apply(lambda x: f'in_type_{features.loc[x, "node_type"]}_ratio')
    
    features['out_same_ratio'] = features.lookup(features.index, out_same_mask)
    features['in_same_ratio'] = features.lookup(features.index, in_same_mask)
    
    # 8. 高风险连接特征
    print("  计算高风险连接...")
    features['out_risk_ratio'] = features['out_type_2_ratio']
    features['in_risk_ratio'] = features['in_type_2_ratio']
    features['risk_propagation'] = features['out_deg'] * features['out_risk_ratio']
    
    # 9. 类型不平衡特征 - 向量化操作
    print("  计算类型不平衡...")
    features['type_imbalance'] = (
        abs(features['out_type_0_ratio'] - features['in_type_0_ratio']) +
        abs(features['out_type_1_ratio'] - features['in_type_1_ratio']) + 
        abs(features['out_type_2_ratio'] - features['in_type_2_ratio'])
    )
    
    # 10. 清理临时列
    features = features.drop(['out_deg', 'in_deg', 'total_deg', 'node_type'], axis=1)
    
    return features

def main():
    print("🚀 开始节点类型分布特征创造 - 纯向量化版本...")
    
    # 1. 加载数据
    print("加载原始数据...")
    data = np.load(args.path_data, allow_pickle='True')
    
    edge_index = data['edge_index']
    
    y = data['y'].flatten()
    
    print(f"边数: {len(edge_index)}, 节点数: {len(y)}")
    
    # 2. 创建节点类型映射
    print("创建节点类型映射...")
    node_type = create_node_type_mapping(y)
    
    print("类型分布:")
    unique, counts = np.unique(node_type, return_counts=True)
    for val, cnt in zip(unique, counts):
        print(f"  类型{val}: {cnt}")
    
    # 3. 创建特征
    print("创建向量化特征...")
    features = create_vectorized_neighbor_type_features(edge_index, node_type)
    
    # 4. 数据清理
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    # 类型优化
    for col in features.columns:
        if features[col].dtype == np.float64:
            features[col] = features[col].astype(np.float32)
    
    # 5. 保存结果
    output_path = args.path_save_feature_type
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"\n🎉 向量化特征创造完成!")
    print(f"📊 特征统计:")
    print(f"  - 总特征数量: {features.shape[1]}")
    print(f"  - 数据行数: {features.shape[0]}")
    print(f"  - 保存路径: {output_path}")
    
    print(f"\n🔍 特征列表 (20个):")
    feature_types = {
        '类型计数': [c for c in features.columns if '_cnt' in c],
        '类型占比': [c for c in features.columns if '_ratio' in c and 'same' not in c and 'risk' not in c],
        '主要类型': [c for c in features.columns if 'main_type' in c],
        '同质风险': [c for c in features.columns if 'same' in c or 'risk' in c],
        '不平衡性': [c for c in features.columns if 'imbalance' in c or 'propagation' in c]
    }
    
    for category, cols in feature_types.items():
        print(f"  {category} ({len(cols)}个): {', '.join(cols)}")

if __name__ == '__main__':
    main()