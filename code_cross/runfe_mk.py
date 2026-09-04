#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cab
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import warnings
import time
import argparse
import gc


# 1. 建立解析器
parser = argparse.ArgumentParser(description='makefea_mk')

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

args = parser.parse_args()

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 计时装饰器
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"\n⏰ 开始执行: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"✅ 完成: {func.__name__}, 耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
        return result
    return wrapper

# 加载数据
@timer_decorator
def load_data():
    data_np = np.load(args.path_data, allow_pickle='True')

    print([f for f in data_np.keys()])

    x = data_np['x']
    y = data_np['y']
    # train_mask = data_np['train_mask']
    # test_mask = data_np['test_mask']
    edge_index = data_np['edge_index']
    edge_type = data_np['edge_type']
    edge_timestamp = data_np['edge_timestamp']

    # print(f"节点数: {len(x)}, 边数: {len(edge_index)}, 训练样本: {len(train_mask)}")

    df_node = pd.DataFrame(x, columns=[f'fea_x_{f}' for f in range(x.shape[1])])
    df_node['label'] = y
    df_node['node_id'] = range(len(df_node))

    df_edge = pd.DataFrame(edge_index, columns=['source', 'target'])
    df_edge['edge_type'] = edge_type
    df_edge['edge_timestamp'] = edge_timestamp

    
    n_init = df_edge.shape[0]
    print('->'*10, 'df_edge init 样本量为: {}'.format(n_init) )
    if args.sub_ratio<1:
        
        df_edge = df_edge.sample(frac = args.sub_ratio, random_state=42)   # 返回新 DataFrame
        n_subsample = df_edge.shape[0]
        print('->'*10, 'df_edge 下采样比例为: {}, 下采样后样本量为: {}'.format(n_init, n_subsample) )


    print("原始标签分布:")
    print(df_node['label'].value_counts())
    
    return df_node, df_edge #, train_mask, test_mask

print("🚀 开始数据加载...")
# df_node, df_edge, train_mask, test_mask = load_data()
df_node, df_edge = load_data()

# ========== 修复版的原始特征工程 ==========
@timer_decorator
def create_original_features(df_node, df_edge):
    """修复版的原始特征工程 - 确保node_id唯一"""
    print("🔧 开始创建原始特征...")
    
    # 基础度特征 - 使用更安全的方法
    print("  计算基础度特征...")
    
    # 出度
    out_degree = df_edge['source'].value_counts().reset_index()
    out_degree.columns = ['node_id', 'fea_outdgree']
    
    # 入度
    in_degree = df_edge['target'].value_counts().reset_index()
    in_degree.columns = ['node_id', 'fea_indgree']
    
    # 合并度特征 - 使用外连接确保所有节点都被包含
    edge_fea = pd.DataFrame({'node_id': range(len(df_node))})
    edge_fea = edge_fea.merge(out_degree, on='node_id', how='left')
    edge_fea = edge_fea.merge(in_degree, on='node_id', how='left')
    
    # 填充缺失值（没有边的节点）
    edge_fea['fea_outdgree'] = edge_fea['fea_outdgree'].fillna(0)
    edge_fea['fea_indgree'] = edge_fea['fea_indgree'].fillna(0)
    
    sts_lst = ['mean', 'max', 'min', 'std', 'nunique']
    
    # 出边特征统计 - 修复重复问题
    print("  计算出边特征统计...")
    df_edge_sorted = df_edge.sort_values(by=['target','edge_timestamp'], ascending=True)
    df_edge_sorted['edge_timestamp_diff_1_target'] = df_edge_sorted['edge_timestamp'] - df_edge_sorted.groupby(['target']).edge_timestamp.shift(1)
    
    for col in ['edge_type', 'edge_timestamp_diff_1_target']:
        print(f"    处理出边 {col}...")
        
        # 为每个统计量单独合并，避免重复
        for stat in tqdm(sts_lst, desc=f"出边 {col}"):  # 修复这里：移除了{stat}
            edge_sts = (
                df_edge_sorted
                .groupby('source')[col]
                .agg(stat)
                .rename(f'fea_out_{col}_{stat}')
                .reset_index()
                .rename(columns={'source': 'node_id'})
            )
            # 确保每个node_id只有一行
            edge_sts = edge_sts.drop_duplicates(subset=['node_id'], keep='first')
            edge_fea = edge_fea.merge(edge_sts, on='node_id', how='left')
        
        # 计算range - 修复重复问题
        range_stats = (
            df_edge_sorted
            .groupby('source')[col]
            .agg(lambda x: x.max() - x.min() if len(x) > 0 else 0)
            .rename(f'fea_out_{col}_range')
            .reset_index()
            .rename(columns={'source': 'node_id'})
        )
        range_stats = range_stats.drop_duplicates(subset=['node_id'], keep='first')
        edge_fea = edge_fea.merge(range_stats, on='node_id', how='left')
        
        # 计算mode - 修复重复问题
        print(f"    计算出边 {col} mode...")
        mode_sts_out = (
            df_edge_sorted
            .groupby(['source', col])
            .size()
            .reset_index(name='_count')
            .sort_values(['source', '_count'], ascending=[True, False])
            .drop_duplicates('source', keep='first')
            .rename(columns={
                'source': 'node_id',
                col: f'fea_out_{col}_mode'
            })[['node_id', f'fea_out_{col}_mode']]
        )
        mode_sts_out = mode_sts_out.drop_duplicates(subset=['node_id'], keep='first')
        edge_fea = edge_fea.merge(mode_sts_out, on='node_id', how='left')
        
    # 入边特征统计 - 修复重复问题
    print("  计算入边特征统计...")
    df_edge_sorted_in = df_edge.sort_values(by=['source','edge_timestamp'], ascending=True)
    df_edge_sorted_in['edge_timestamp_diff_1_source'] = df_edge_sorted_in['edge_timestamp'] - df_edge_sorted_in.groupby(['source']).edge_timestamp.shift(1)
    
    for col in ['edge_type', 'edge_timestamp_diff_1_source']:
        print(f"    处理入边 {col}...")
        
        for stat in tqdm(sts_lst, desc=f"入边 {col}"):  # 修复这里：移除了{stat}
            edge_sts = (
                df_edge_sorted_in
                .groupby('target')[col]
                .agg(stat)
                .rename(f'fea_in_{col}_{stat}')
                .reset_index()
                .rename(columns={'target': 'node_id'})
            )
            # 确保每个node_id只有一行
            edge_sts = edge_sts.drop_duplicates(subset=['node_id'], keep='first')
            edge_fea = edge_fea.merge(edge_sts, on='node_id', how='left')
        
        # 计算range - 修复重复问题
        range_stats = (
            df_edge_sorted_in
            .groupby('target')[col]
            .agg(lambda x: x.max() - x.min() if len(x) > 0 else 0)
            .rename(f'fea_in_{col}_range')
            .reset_index()
            .rename(columns={'target': 'node_id'})
        )
        range_stats = range_stats.drop_duplicates(subset=['node_id'], keep='first')
        edge_fea = edge_fea.merge(range_stats, on='node_id', how='left')
        
        # 计算mode - 修复重复问题
        print(f"    计算入边 {col} mode...")
        mode_sts_in = (
            df_edge_sorted_in
            .groupby(['target', col])
            .size()
            .reset_index(name='_count')
            .sort_values(['target', '_count'], ascending=[True, False])
            .drop_duplicates('target', keep='first')
            .rename(columns={
                'target': 'node_id',
                col: f'fea_in_{col}_mode'
            })[['node_id', f'fea_in_{col}_mode']]
        )
        mode_sts_in = mode_sts_in.drop_duplicates(subset=['node_id'], keep='first')
        edge_fea = edge_fea.merge(mode_sts_in, on='node_id', how='left')
    
    # 邻居特征统计 - 修复重复问题
    print("  计算邻居特征统计...")
    if 'dgraphfin' in args.path_data:
        fea_cols = ['fea_x_0', 'fea_x_1', 'fea_x_2', 'fea_x_3', 'fea_x_4', 'fea_x_5',
               'fea_x_6', 'fea_x_7', 'fea_x_8', 'fea_x_9', 'fea_x_10', 'fea_x_11',
               'fea_x_12', 'fea_x_13', 'fea_x_14', 'fea_x_15', 'fea_x_16']
    else:
        fea_cols = ['fea_x_0', 'fea_x_1', 'fea_x_2', 'fea_x_3', 'fea_x_4', 'fea_x_5',
               'fea_x_6', 'fea_x_7', 'fea_x_8', 'fea_x_9']
    # 出边邻居特征统计 - 修复重复问题
    print("    计算出边邻居特征...")
    for col in tqdm(fea_cols, desc="出边邻居特征"):
        df_edge_with_node_fea = df_edge.merge(
            df_node[['node_id', col]], 
            left_on='source', 
            right_on='node_id', 
            how='left'
        )
        
        out_stats = (
            df_edge_with_node_fea
            .groupby('source')[col]
            .agg(['max', 'min', 'mean', 'std'])
            .rename(columns={
                'max': f'fea_out_neighbor_{col}_max',
                'min': f'fea_out_neighbor_{col}_min', 
                'mean': f'fea_out_neighbor_{col}_mean',
                'std': f'fea_out_neighbor_{col}_std'
            })
            .reset_index()
            .rename(columns={'source': 'node_id'})
        )
        # 确保每个node_id只有一行
        out_stats = out_stats.drop_duplicates(subset=['node_id'], keep='first')
        edge_fea = edge_fea.merge(out_stats, on='node_id', how='left')
    
    # 入边邻居特征统计 - 修复重复问题
    print("    计算入边邻居特征...")
    for col in tqdm(fea_cols, desc="入边邻居特征"):
        df_edge_with_node_fea = df_edge.merge(
            df_node[['node_id', col]], 
            left_on='target', 
            right_on='node_id', 
            how='left'
        )
        
        in_stats = (
            df_edge_with_node_fea
            .groupby('target')[col]
            .agg(['max', 'min', 'mean', 'std'])
            .rename(columns={
                'max': f'fea_in_neighbor_{col}_max',
                'min': f'fea_in_neighbor_{col}_min',
                'mean': f'fea_in_neighbor_{col}_mean', 
                'std': f'fea_in_neighbor_{col}_std'
            })
            .reset_index()
            .rename(columns={'target': 'node_id'})
        )
        # 确保每个node_id只有一行
        in_stats = in_stats.drop_duplicates(subset=['node_id'], keep='first')
        edge_fea = edge_fea.merge(in_stats, on='node_id', how='left')
    
    # 最终检查：确保node_id唯一
    print("  最终唯一性检查...")
    if edge_fea['node_id'].nunique() != len(edge_fea):
        print(f"    ⚠️ 发现重复node_id，进行最终去重...")
        edge_fea = edge_fea.drop_duplicates(subset=['node_id'], keep='first')
    
    original_feature_count = edge_fea.shape[1] - 1  # 减去node_id列
    print(f"  ✅ 原始特征创建完成，特征数: {original_feature_count}")
    print(f"  🔍 最终检查: node_id唯一性 {edge_fea['node_id'].nunique()}/{len(edge_fea)}")
    
    return edge_fea, original_feature_count

# ========== 增强特征工程 ==========
@timer_decorator
def create_enhanced_features(df_edge, df_node):
    """新增的特征工程"""
    print("✨ 开始创建增强特征...")
    
    # 初始化增强特征DataFrame
    enhanced_features = pd.DataFrame({'node_id': range(len(df_node))})
    
    # 1. 时间周期特征（基于天的周期）
    print("  创建时间周期特征...")
    df_edge_copy = df_edge.copy()
    
    # 周周期特征（假设7天为一周）
    df_edge_copy['time_week'] = (df_edge_copy['edge_timestamp'] - 1) % 7
    df_edge_copy['time_sin_week'] = np.sin(2 * np.pi * df_edge_copy['time_week'] / 7)
    df_edge_copy['time_cos_week'] = np.cos(2 * np.pi * df_edge_copy['time_week'] / 7)
    
    # 月周期特征（假设30天为一月）
    df_edge_copy['time_month'] = ((df_edge_copy['edge_timestamp'] - 1) % 30)
    df_edge_copy['time_sin_month'] = np.sin(2 * np.pi * df_edge_copy['time_month'] / 30)
    df_edge_copy['time_cos_month'] = np.cos(2 * np.pi * df_edge_copy['time_month'] / 30)
    
    # 时间密度特征
    time_density = df_edge_copy.groupby('source').agg({
        'edge_timestamp': ['count', 'nunique'],
        'time_sin_week': ['mean', 'std'],
        'time_cos_week': ['mean', 'std'],
        'time_sin_month': ['mean', 'std'], 
        'time_cos_month': ['mean', 'std']
    }).reset_index()
    
    time_density.columns = ['node_id', 'enh_time_total_count', 'enh_time_unique_days', 
                           'enh_time_sin_week_mean', 'enh_time_sin_week_std',
                           'enh_time_cos_week_mean', 'enh_time_cos_week_std',
                           'enh_time_sin_month_mean', 'enh_time_sin_month_std',
                           'enh_time_cos_month_mean', 'enh_time_cos_month_std']
    
    time_density['enh_time_density'] = time_density['enh_time_total_count'] / (time_density['enh_time_unique_days'] + 1)
    
    enhanced_features = enhanced_features.merge(time_density, on='node_id', how='left')
    
    # 2. 高级图结构特征
    print("  创建高级图结构特征...")
    
    # 度特征组合
    out_degree = df_edge['source'].value_counts().reset_index()
    out_degree.columns = ['node_id', 'enh_out_degree']
    in_degree = df_edge['target'].value_counts().reset_index()
    in_degree.columns = ['node_id', 'enh_in_degree']
    
    enhanced_features = enhanced_features.merge(out_degree, on='node_id', how='left')
    enhanced_features = enhanced_features.merge(in_degree, on='node_id', how='left')
    
    enhanced_features['enh_total_degree'] = enhanced_features['enh_out_degree'] + enhanced_features['enh_in_degree']
    enhanced_features['enh_degree_ratio'] = enhanced_features['enh_in_degree'] / (enhanced_features['enh_out_degree'] + 1)
    enhanced_features['enh_degree_imbalance'] = abs(enhanced_features['enh_out_degree'] - enhanced_features['enh_in_degree'])
    enhanced_features['enh_degree_balance_ratio'] = enhanced_features['enh_degree_imbalance'] / (enhanced_features['enh_total_degree'] + 1)
    
    # # 3. 二跳邻居特征
    # print("  计算二跳邻居特征...")
    # second_hop_out = df_edge.merge(
    #     df_edge, left_on='target', right_on='source', suffixes=('', '_second')
    # ).groupby('source')['target_second'].nunique().reset_index()
    # second_hop_out.columns = ['node_id', 'enh_second_hop_out_count']
    
    # second_hop_in = df_edge.merge(
    #     df_edge, left_on='source', right_on='target', suffixes=('', '_second')
    # ).groupby('target')['source_second'].nunique().reset_index()
    # second_hop_in.columns = ['node_id', 'enh_second_hop_in_count']
    
    # enhanced_features = enhanced_features.merge(second_hop_out, on='node_id', how='left')
    # enhanced_features = enhanced_features.merge(second_hop_in, on='node_id', how='left')


    # 0. 预定义超级节点阈值
    HUB_THRES = 5000          # 可按场景调（1 万、2 万都行）
    
    # 1. 先算度，找出超级节点
    out_deg = df_edge['source'].value_counts()
    in_deg  = df_edge['target'].value_counts()
    hub_out = out_deg[out_deg > HUB_THRES].index
    hub_in  = in_deg[in_deg > HUB_THRES].index
    hub_nodes = set(hub_out) | set(hub_in)          # 超级节点集合
    normal_nodes = df_edge['source'].drop_duplicates().index.difference(hub_nodes)
    print(f'超级节点 {len(hub_nodes)} 个，剩余普通节点 {len(normal_nodes)} 个')
    
    def batch_second_hop_hub(df, side='out', batch_neighbor=5000):
        """
        对 hub 节点批量计算二跳邻居数
        side='out'/'in'
        返回 DataFrame: node_id | enh_second_hop_{side}_count
        """
        col_self = 'source' if side == 'out' else 'target'
        col_nei  = 'target' if side == 'out' else 'source'
    
        # 1. hub 列表
        deg = df[col_self].value_counts()
        hub_nodes = deg[deg > 5000].index.to_numpy()
        print(f'批量处理 {len(hub_nodes)} 个 hub-{side}')
    
        # 2. 轻量表
        df_light = df[[col_self, col_nei]].copy()
        gc.collect()
    
        # 3. 分批计算
        res = []
        # 每批 500 个 hub 左右
        for hub_batch in tqdm(np.array_split(hub_nodes, max(1, len(hub_nodes)//500)),
                              desc=f'hub-{side}-batch'):
            # 当前批次 1-hop 边
            chunk_1hop = df_light[df_light[col_self].isin(hub_batch)]
            # 采样上限
            if len(chunk_1hop) > batch_neighbor:
                chunk_1hop = chunk_1hop.sample(n=batch_neighbor, random_state=42)
    
            # merge 得到二跳表
            tmp = chunk_1hop.merge(df_light, left_on=col_nei, right_on=col_self,
                                   suffixes=('', '_2hop'))
            # 列名此时是 target_2hop 或 source_2hop，写死即可
            second_col = 'target_2hop' if side == 'out' else 'source_2hop'
    
            # 先整体 value_counts，再映射回每个 hub
            second_counts = tmp[second_col].value_counts()
            hop_cnt = (tmp.groupby(col_self)[second_col]
                         .apply(lambda x: second_counts.reindex(x).sum())
                         .reset_index(name=f'enh_second_hop_{side}_count'))
            res.append(hop_cnt)
            del chunk_1hop, tmp, second_counts
            gc.collect()
    
        return pd.concat(res, ignore_index=True) if res else \
               pd.DataFrame(columns=['node_id', f'enh_second_hop_{side}_count'])
    
    
    # 替换原「二跳近似」两行
    hub_out_df = batch_second_hop_hub(df_edge, 'out', batch_neighbor=5000)
    hub_in_df  = batch_second_hop_hub(df_edge, 'in',  batch_neighbor=5000)
    
    # 3. 普通节点走「分批+裁剪」原逻辑（内存安全）
    chunk_size = 40000
    sources_norm = normal_nodes.to_numpy()
    n_chunks = (len(sources_norm) + chunk_size - 1) // chunk_size
    
    # 出边二跳（普通节点）
    second_hop_out_buff = []
    df_edge_right = df_edge[['source', 'target']].copy()
    for i in tqdm(range(n_chunks), desc='二跳出边-普通'):
        chunk_src = sources_norm[i*chunk_size : (i+1)*chunk_size]
        chunk_1hop = df_edge[df_edge['source'].isin(chunk_src)]
        cand_tgt = chunk_1hop['target'].unique()
        chunk_right = df_edge_right[df_edge_right['source'].isin(cand_tgt)]
        tmp = chunk_1hop.merge(chunk_right, left_on='target', right_on='source', suffixes=('','_2hop'))
        second_hop_out_buff.append(
            tmp.groupby('source')['target_2hop'].nunique().rename('enh_second_hop_out_count')
        )
        del chunk_1hop, chunk_right, tmp
        gc.collect()
    
    norm_out_df = pd.concat(second_hop_out_buff).reset_index()
    norm_out_df.columns = ['node_id', 'enh_second_hop_out_count']
    
    # 入边同理（略）
    second_hop_in_buff = []
    for i in tqdm(range(n_chunks), desc='二跳入边-普通'):
        chunk_tgt = sources_norm[i*chunk_size : (i+1)*chunk_size]
        chunk_1hop = df_edge[df_edge['target'].isin(chunk_tgt)]
        cand_src = chunk_1hop['source'].unique()
        chunk_right = df_edge_right[df_edge_right['target'].isin(cand_src)]
        tmp = chunk_1hop.merge(chunk_right, left_on='source', right_on='target', suffixes=('','_2hop'))
        second_hop_in_buff.append(
            tmp.groupby('target')['source_2hop'].nunique().rename('enh_second_hop_in_count')
        )
        del chunk_1hop, chunk_right, tmp
        gc.collect()
    
    norm_in_df = pd.concat(second_hop_in_buff).reset_index()
    norm_in_df.columns = ['node_id', 'enh_second_hop_in_count']
    
    # 4. 合并结果
    second_hop_out = pd.concat([hub_out_df, norm_out_df], ignore_index=True)
    second_hop_in  = pd.concat([hub_in_df,  norm_in_df],  ignore_index=True)
    
    enhanced_features = enhanced_features.merge(second_hop_out, on='node_id', how='left')
    enhanced_features = enhanced_features.merge(second_hop_in,  on='node_id', how='left')
    


    
    # 4. 节点交互模式特征
    print("  计算节点交互模式特征...")
    interaction_patterns = df_edge.groupby('source').agg({
        'target': ['nunique', 'count'],
        'edge_type': ['nunique']
    }).reset_index()
    interaction_patterns.columns = ['node_id', 'enh_unique_targets', 'enh_total_interactions', 
                                   'enh_unique_edge_types']
    
    enhanced_features = enhanced_features.merge(interaction_patterns, on='node_id', how='left')
    
    # 5. 时间活跃度特征
    print("  计算时间活跃度特征...")
    temporal_activity = df_edge.groupby('source').agg({
        'edge_timestamp': ['min', 'max', 'nunique']
    }).reset_index()
    temporal_activity.columns = ['node_id', 'enh_first_activity', 'enh_last_activity', 'enh_active_days']
    
    enhanced_features = enhanced_features.merge(temporal_activity, on='node_id', how='left')
    
    # 6. 欺诈邻居特征
    print("  计算欺诈邻居特征...")
    fraud_neighbor_ratio_out = df_edge.merge(
        df_node[df_node['label'] == 1][['node_id']],
        left_on='target', right_on='node_id', how='inner'
    ).groupby('source').size().reset_index()
    fraud_neighbor_ratio_out.columns = ['node_id', 'enh_fraud_neighbors_out']
    
    fraud_neighbor_ratio_in = df_edge.merge(
        df_node[df_node['label'] == 1][['node_id']],
        left_on='source', right_on='node_id', how='inner'  
    ).groupby('target').size().reset_index()
    fraud_neighbor_ratio_in.columns = ['node_id', 'enh_fraud_neighbors_in']
    
    enhanced_features = enhanced_features.merge(fraud_neighbor_ratio_out, on='node_id', how='left')
    enhanced_features = enhanced_features.merge(fraud_neighbor_ratio_in, on='node_id', how='left')
    
    # 计算衍生特征
    enhanced_features['enh_activity_span'] = enhanced_features['enh_last_activity'] - enhanced_features['enh_first_activity']
    enhanced_features['enh_daily_activity_rate'] = enhanced_features['enh_total_interactions'] / (enhanced_features['enh_active_days'] + 1)
    enhanced_features['enh_fraud_neighbor_ratio_out'] = enhanced_features['enh_fraud_neighbors_out'] / (enhanced_features['enh_total_interactions'] + 1)
    enhanced_features['enh_fraud_neighbor_ratio_in'] = enhanced_features['enh_fraud_neighbors_in'] / (enhanced_features['enh_total_interactions'] + 1)
    
    enhanced_feature_count = enhanced_features.shape[1] - 1  # 减去node_id列
    print(f"  ✅ 增强特征创建完成，特征数: {enhanced_feature_count}")
    print(f"  🔍 最终检查: node_id唯一性 {enhanced_features['node_id'].nunique()}/{len(enhanced_features)}")
    
    return enhanced_features, enhanced_feature_count

# ========== 金融反欺诈业务特征 ==========
# ========== 金融反欺诈业务特征 ==========
@timer_decorator
def create_business_features(df_edge, df_node):
    """金融反欺诈业务特征（超级节点友好版）"""
    print("💼 开始创建金融反欺诈业务特征...")
    
    # 0. 使用原始数据的所有列，不进行限制
    df_edge_copy = df_edge.copy()
    gc.collect()
    
    business_features = pd.DataFrame({'node_id': range(len(df_node))})
    fraud_nodes = set(df_node.loc[df_node['label'] == 1, 'node_id'].tolist())

    # 1. 欺诈传播网络特征
    print("  计算欺诈传播网络特征...")
    fraud_conn_out = (df_edge_copy[df_edge_copy['target'].isin(fraud_nodes)]
                      .groupby('source').size()
                      .reset_index(name='biz_fraud_conn_out')
                      .rename(columns={'source': 'node_id'}))
    fraud_conn_in  = (df_edge_copy[df_edge_copy['source'].isin(fraud_nodes)]
                      .groupby('target').size()
                      .reset_index(name='biz_fraud_conn_in')
                      .rename(columns={'target': 'node_id'}))
    business_features = business_features.merge(fraud_conn_out, on='node_id', how='left') \
                                         .merge(fraud_conn_in,  on='node_id', how='left')

    # 2. 网络结构异常特征
    print("  计算网络结构异常特征...")
    deg_out = df_edge_copy['source'].value_counts().rename('out_deg')
    deg_in  = df_edge_copy['target'].value_counts().rename('in_deg')
    degree_stats = pd.DataFrame({'node_id': deg_out.index, 'out_deg': deg_out.values}) \
                   .merge(pd.DataFrame({'node_id': deg_in.index, 'in_deg': deg_in.values}),
                          on='node_id', how='outer').fillna(0)
    degree_stats['biz_degree_ratio']      = degree_stats['in_deg'] / (degree_stats['out_deg'] + 1)
    degree_stats['biz_degree_imbalance']  = abs(degree_stats['out_deg'] - degree_stats['in_deg'])
    degree_stats['biz_isolated_node']     = (degree_stats['out_deg'] == 0) & (degree_stats['in_deg'] == 0)
    degree_stats['biz_one_way_node']      = (degree_stats['out_deg'] == 0) | (degree_stats['in_deg'] == 0)
    business_features = business_features.merge(
        degree_stats[['node_id', 'biz_degree_ratio', 'biz_degree_imbalance',
                     'biz_isolated_node', 'biz_one_way_node']], on='node_id', how='left')

    # 3. 时间模式异常特征（基于天）
    print("  计算时间模式异常特征...")
    time_conc = df_edge_copy.groupby('source').agg({
        'edge_timestamp': ['count', 'nunique', lambda x: (x.max() - x.min()) if len(x) > 1 else 0]
    }).reset_index()
    time_conc.columns = ['node_id', 'biz_total_trans', 'biz_active_days', 'biz_time_span']
    time_conc['biz_trans_concentration'] = time_conc['biz_total_trans'] / (time_conc['biz_time_span'] + 1)
    time_conc['biz_daily_intensity']     = time_conc['biz_total_trans'] / (time_conc['biz_active_days'] + 1)
    business_features = business_features.merge(time_conc, on='node_id', how='left')

    # 4. 交易行为多样性特征
    print("  计算交易行为多样性特征...")
    edge_div = (df_edge_copy.groupby('source')
                       .agg({'edge_type': 'nunique', 'target': 'nunique'})
                       .reset_index()
                       .rename(columns={'source': 'node_id',
                                        'edge_type': 'biz_edge_type_diversity',
                                        'target':    'biz_unique_partners'}))
    business_features = business_features.merge(edge_div, on='node_id', how='left')

    # 5. 多跳欺诈风险传播（超级节点友好版）
    print("  计算多跳欺诈风险传播（分批+采样）...")
    HUB_THRES = 5_000
    col_self, col_nei = 'source', 'target'
    deg = df_edge_copy[col_self].value_counts()
    hub_nodes = set(deg[deg > HUB_THRES].index)
    fraud_set = fraud_nodes

    def batch_second_hop_fraud(df, side='out', sample_n=3_000):
        """对 hub 分批采样二跳欺诈连接数"""
        col_s = 'source' if side == 'out' else 'target'
        col_t = 'target' if side == 'out' else 'source'
        deg = df[col_s].value_counts()
        hubs = deg[deg > HUB_THRES].index.to_numpy()
        normal = deg[deg <= HUB_THRES].index.to_numpy()

        # 普通节点：直接 groupby
        df_norm = df[df[col_s].isin(normal)]
        normal_res = (df_norm[df_norm[col_t].isin(fraud_set)]
                      .groupby(col_s).size()
                      .reset_index(name=f'biz_second_hop_fraud_{side}')
                      .rename(columns={col_s: 'node_id'}))

        # 超级节点：采样二跳
        hub_res = []
        df_light = df[[col_s, col_t]].copy()
        for hub_batch in tqdm(np.array_split(hubs, max(1, len(hubs)//500)),
                              desc=f'fraud-2hop-{side}'):
            chunk_1hop = df_light[df_light[col_s].isin(hub_batch)]
            if len(chunk_1hop) > sample_n:
                chunk_1hop = chunk_1hop.sample(n=sample_n, random_state=42)
            tmp = chunk_1hop.merge(df_light, left_on=col_t, right_on=col_s,
                                   suffixes=('','_2hop'))
            fraud_2nd = tmp[tmp['target_2hop'].isin(fraud_set)]
            cnt = fraud_2nd.groupby(col_s).size().reset_index(name=f'biz_second_hop_fraud_{side}')
            hub_res.append(cnt)
            del chunk_1hop, tmp, fraud_2nd
            gc.collect()
        hub_df = pd.concat(hub_res, ignore_index=True) if hub_res else \
                 pd.DataFrame(columns=['node_id', f'biz_second_hop_fraud_{side}'])
        normal_res = normal_res.rename(columns={col_s: 'node_id'})
        hub_df     = hub_df.rename(columns={col_s: 'node_id'})
        return pd.concat([normal_res, hub_df], ignore_index=True)

    fraud_out = batch_second_hop_fraud(df_edge_copy, 'out', sample_n=3_000)
    fraud_in  = batch_second_hop_fraud(df_edge_copy, 'in',  sample_n=3_000)
    business_features = business_features.merge(fraud_out, on='node_id', how='left') \
                                         .merge(fraud_in,  on='node_id', how='left')

    # 6. 时间序列异常特征（修复版）- 这是关键修复
    print("  计算时间序列异常特征...")
    
    # 方法1：更安全的计算方式
    # 首先对每个source的时间戳进行排序
    df_sorted = df_edge_copy.sort_values(['source', 'edge_timestamp']).copy()
    
    # 计算时间差
    time_diffs = df_sorted.groupby('source')['edge_timestamp'].diff()
    
    # 获取原始索引以便合并
    time_diffs = time_diffs.reset_index()
    
    # 合并时间差回原数据
    df_sorted = df_sorted.reset_index(drop=True)
    time_diffs = time_diffs.reset_index(drop=True)
    
    # 确保列名正确
    if len(time_diffs) > 0:
        # 重命名列
        time_diffs.columns = ['source', 'time_diff']
        
        # 计算统计量
        time_gap_stats = time_diffs.groupby('source')['time_diff'].agg(['mean', 'std', 'max', 'min']).reset_index()
        time_gap_stats.columns = ['node_id', 'biz_time_gap_mean', 'biz_time_gap_std', 'biz_time_gap_max', 'biz_time_gap_min']
        
        # 填充缺失值
        time_gap_stats = time_gap_stats.fillna(0)
        
        business_features = business_features.merge(time_gap_stats, on='node_id', how='left')
    else:
        # 如果没有时间差数据，创建空列
        business_features['biz_time_gap_mean'] = 0
        business_features['biz_time_gap_std'] = 0
        business_features['biz_time_gap_max'] = 0
        business_features['biz_time_gap_min'] = 0

    # 7. 团伙欺诈特征——三角形计数（采样版）
    print("  计算三角形计数（采样版）...")
    TRI_SAMPLE = 5_000
    
    # 重新采样
    df_edge_sampled = df_edge_copy.sample(n=min(TRI_SAMPLE, len(df_edge_copy)), random_state=42)
    
    # 重命名列以避免冲突
    df_edge1 = df_edge_sampled[['source', 'target']].rename(columns={'source': 's1', 'target': 't1'})
    df_edge2 = df_edge_sampled[['source', 'target']].rename(columns={'source': 's2', 'target': 't2'})
    df_edge3 = df_edge_sampled[['source', 'target']].rename(columns={'source': 's3', 'target': 't3'})
    
    # 合并查找三角形
    merged = df_edge1.merge(df_edge2, left_on='t1', right_on='s2')
    merged = merged.merge(df_edge3, left_on=['s1', 't2'], right_on=['s3', 't3'])
    
    # 计数三角形
    tri_counts = merged.groupby('s1').size().reset_index(name='biz_triangle_count')
    tri_counts = tri_counts.rename(columns={'s1': 'node_id'})
    
    business_features = business_features.merge(tri_counts, on='node_id', how='left')
    business_features['biz_triangle_count'] = business_features['biz_triangle_count'].fillna(0)

    # 8. 背景节点利用特征
    print("  计算背景节点利用特征...")
    bg_nodes = set(df_node.loc[df_node['label'].isin([2, 3]), 'node_id'].tolist())
    
    bg_out = df_edge_copy[df_edge_copy['target'].isin(bg_nodes)].groupby('source').size().reset_index(name='biz_bg_conn_out')
    bg_in  = df_edge_copy[df_edge_copy['source'].isin(bg_nodes)].groupby('target').size().reset_index(name='biz_bg_conn_in')
    bg_out = bg_out.rename(columns={'source': 'node_id'})
    bg_in  = bg_in.rename(columns={'target': 'node_id'})
    business_features = business_features.merge(bg_out, on='node_id', how='left') \
                                         .merge(bg_in,  on='node_id', how='left')

    # 9. 风险累积特征（直接 agg，无大 merge）
    print("  计算风险累积特征...")
    risk = (df_edge_copy.sort_values(['source', 'edge_timestamp'])
                       .groupby('source')
                       .agg({
                           'edge_timestamp': ['count', 'max'],
                           'target': lambda x: len(set(x) & fraud_set)
                       })
                       .reset_index())
    
    # 重命名列
    risk.columns = ['node_id', 'biz_total_transactions', 'biz_last_activity', 'biz_cumulative_fraud_conn']
    risk['biz_fraud_conn_ratio'] = risk['biz_cumulative_fraud_conn'] / (risk['biz_total_transactions'] + 1)
    
    business_features = business_features.merge(risk, on='node_id', how='left')

    business_feature_count = business_features.shape[1] - 1
    print(f"  ✅ 业务特征创建完成，特征数: {business_feature_count}")
    print(f"  🔍 最终检查: node_id唯一性 {business_features['node_id'].nunique()}/{len(business_features)}")
    return business_features, business_feature_count

# ========== 主特征工程流程 ==========
@timer_decorator
def main_feature_engineering(df_node, df_edge):
    """主特征工程流程"""
    print("🔧 开始特征工程...")
    start_time = time.time()
    
    # 创建原始特征
    original_features, original_count = create_original_features(df_node, df_edge)
    
    # 创建增强特征
    enhanced_features, enhanced_count = create_enhanced_features(df_edge, df_node)
    
    # 创建业务特征
    business_features, business_count = create_business_features(df_edge, df_node)
    
    # 合并所有特征 - 安全版本
    print("🔄 合并所有特征...")
    
    # 1. 首先检查每个特征表的node_id唯一性
    print("  检查特征表唯一性...")
    print(f"    df_node node_id 唯一性: {df_node['node_id'].nunique()} / {len(df_node)}")
    print(f"    original_features node_id 唯一性: {original_features['node_id'].nunique()} / {len(original_features)}")
    print(f"    enhanced_features node_id 唯一性: {enhanced_features['node_id'].nunique()} / {len(enhanced_features)}")
    print(f"    business_features node_id 唯一性: {business_features['node_id'].nunique()} / {len(business_features)}")
    
    # 2. 使用左连接确保行数不变
    df_node_enhanced = df_node.copy()
    
    # 逐步合并并检查行数
    initial_count = len(df_node_enhanced)
    print(f"    初始行数: {initial_count}")
    
    df_node_enhanced = df_node_enhanced.merge(original_features, on='node_id', how='left')
    print(f"    合并原始特征后: {len(df_node_enhanced)}")
    
    df_node_enhanced = df_node_enhanced.merge(enhanced_features, on='node_id', how='left')
    print(f"    合并增强特征后: {len(df_node_enhanced)}")
    
    df_node_enhanced = df_node_enhanced.merge(business_features, on='node_id', how='left')
    print(f"    合并业务特征后: {len(df_node_enhanced)}")
    
    # 3. 验证行数一致性
    if len(df_node_enhanced) != initial_count:
        print(f"    ❗ 警告: 合并后行数变化! 初始: {initial_count}, 现在: {len(df_node_enhanced)}")
        # 如果行数变化，进行去重
        df_node_enhanced = df_node_enhanced.drop_duplicates(subset=['node_id'], keep='first')
        print(f"    去重后行数: {len(df_node_enhanced)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 特征工程完成!")
    print(f"📊 特征统计:")
    print(f"  - 原始特征数量: {original_count}")
    print(f"  - 增强特征数量: {enhanced_count}") 
    print(f"  - 业务特征数量: {business_count}")
    print(f"  - 总特征维度: {df_node_enhanced.shape}")
    print(f"  - 数据行数验证: {len(df_node_enhanced)} (应与原始 {len(df_node)} 一致)")
    print(f"⏱️ 总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    return df_node_enhanced

# 执行特征工程
print("\n" + "="*60)
print("🚀 开始完整特征工程流程")
print("="*60)

df_node_enhanced = main_feature_engineering(df_node, df_edge)


def makefe_exp(df_node_enhanced_p4):
    # df_node_enhanced_p4 = pd.read_pickle('../feature/DGraphFin/df_node_enhanced_mk.pkl')
    df_node_enhanced_p4 = df_node_enhanced_p4.reset_index(drop=True)

    if 'dgraphfin' in args.path_data: 
        cols_init = ['fea_x_0', 'fea_x_1', 'fea_x_2', 'fea_x_3', 'fea_x_4', 'fea_x_5', 'fea_x_6', 'fea_x_7'
         , 'fea_x_8', 'fea_x_9', 'fea_x_10', 'fea_x_11', 'fea_x_12', 'fea_x_13', 'fea_x_14'
         , 'fea_x_15', 'fea_x_16']
    else:
        cols_init = ['fea_x_0', 'fea_x_1', 'fea_x_2', 'fea_x_3', 'fea_x_4', 'fea_x_5', 'fea_x_6', 'fea_x_7'
         , 'fea_x_8', 'fea_x_9']
    
    cols_null = []
    for col in tqdm(cols_init):
        df_node_enhanced_p4['isnull_{}'.format(col)] = df_node_enhanced_p4[col].apply(lambda x: 1 if x==-1 else 0)
        cols_null.append('isnull_{}'.format(col))
    
    # 基础缺失统计
    df_node_enhanced_p4['isnull_fea_sum'] = df_node_enhanced_p4[cols_null].sum(axis=1)
    df_node_enhanced_p4['isnull_fea_ratio'] = df_node_enhanced_p4['isnull_fea_sum'] / len(cols_init)
    
    # ========= 超快速计算序列特征 =========
    print("🔄 超快速计算序列特征...")
    
    # 获取缺失矩阵
    missing_matrix = df_node_enhanced_p4[cols_null].values.astype(np.int8)
    n_rows, n_cols = missing_matrix.shape
    
    # 1. 序列变化次数和波动率 (最快方式)
    changes_matrix = np.diff(missing_matrix, axis=1)
    seq_changes = np.sum(np.abs(changes_matrix), axis=1)
    seq_volatility = seq_changes / (n_cols - 1)
    
    # 2. 连续缺失段统计 (优化版本)
    print("📊 计算连续缺失段统计...")
    seq_streak_count = np.zeros(n_rows, dtype=np.int8)
    seq_max_streak = np.zeros(n_rows, dtype=np.int8)
    seq_avg_streak = np.zeros(n_rows, dtype=np.float32)
    
    for i in range(n_rows):
        row = missing_matrix[i]
        current_streak = 0
        max_streak = 0
        streak_count = 0
        total_streak_length = 0
        
        for j in range(n_cols):
            if row[j] == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streak_count += 1
                    total_streak_length += current_streak
                    max_streak = max(max_streak, current_streak)
                    current_streak = 0
        
        # 处理行尾的连续段
        if current_streak > 0:
            streak_count += 1
            total_streak_length += current_streak
            max_streak = max(max_streak, current_streak)
        
        seq_streak_count[i] = streak_count
        seq_max_streak[i] = max_streak
        seq_avg_streak[i] = total_streak_length / streak_count if streak_count > 0 else 0
    
    # 3. 缺失位置统计 (极简版本 - 只计算最重要的)
    print("📊 计算缺失位置统计...")
    seq_first_missing_pos = np.full(n_rows, -1.0, dtype=np.float32)
    seq_last_missing_pos = np.full(n_rows, -1.0, dtype=np.float32)
    
    # 只计算第一个和最后一个缺失位置，这是最有信息量的
    for i in range(n_rows):
        row = missing_matrix[i]
        
        # 找到第一个缺失位置
        first_pos = -1
        for j in range(n_cols):
            if row[j] == 1:
                first_pos = j / n_cols
                break
        
        # 找到最后一个缺失位置  
        last_pos = -1
        for j in range(n_cols-1, -1, -1):
            if row[j] == 1:
                last_pos = j / n_cols
                break
        
        seq_first_missing_pos[i] = first_pos
        seq_last_missing_pos[i] = last_pos
    
    # 4. 快速计算缺失密度分布 (替代平均位置)
    print("📊 计算缺失密度分布...")
    # 将序列分成3段，计算每段的缺失密度
    segment_size = n_cols // 3
    seq_density_front = np.mean(missing_matrix[:, :segment_size], axis=1)
    seq_density_middle = np.mean(missing_matrix[:, segment_size:2*segment_size], axis=1)
    seq_density_back = np.mean(missing_matrix[:, 2*segment_size:], axis=1)
    
    # 将所有特征添加到数据框
    df_node_enhanced_p4['seq_streak_count'] = seq_streak_count
    df_node_enhanced_p4['seq_max_streak'] = seq_max_streak
    df_node_enhanced_p4['seq_avg_streak'] = seq_avg_streak
    df_node_enhanced_p4['seq_changes'] = seq_changes
    df_node_enhanced_p4['seq_volatility'] = seq_volatility
    df_node_enhanced_p4['seq_first_missing_pos'] = seq_first_missing_pos
    df_node_enhanced_p4['seq_last_missing_pos'] = seq_last_missing_pos
    df_node_enhanced_p4['seq_density_front'] = seq_density_front
    df_node_enhanced_p4['seq_density_middle'] = seq_density_middle
    df_node_enhanced_p4['seq_density_back'] = seq_density_back
    
    # ========= 新衍生的序列特征列表 =========
    sequence_feature_cols = [
        'seq_streak_count', 'seq_max_streak', 'seq_avg_streak',
        'seq_changes', 'seq_volatility', 
        'seq_first_missing_pos', 'seq_last_missing_pos',
        'seq_density_front', 'seq_density_middle', 'seq_density_back'
    ]
    
    print(f"🎯 共衍生 {len(sequence_feature_cols)} 个序列特征")
    
    # 显示特征统计信息
    print("\n📈 序列特征统计信息:")
    print(df_node_enhanced_p4[sequence_feature_cols].describe())
    
    df_node_enhanced_p4 = df_node_enhanced_p4.drop(['label', 'node_id'], axis=1)
    
    # 更新特征列列表
    cols_p4 = df_node_enhanced_p4.columns.tolist()[:100] + cols_null + sequence_feature_cols
    print(f"📊 总特征数量: {len(cols_p4)}")
    
    df_node_enhanced_p4 = df_node_enhanced_p4[cols_p4]
    print("✅ 超快速序列特征衍生完成！")
    
    del sequence_feature_cols
    [gc.collect() for _ in range(5)]

    return df_node_enhanced_p4


df_node_enhanced = makefe_exp(df_node_enhanced)

print(f"\n📋 最终数据形状检查:")
print(f"  原始数据: {df_node.shape}")
print(f"  特征工程后: {df_node_enhanced.shape}")



print(df_node.shape, df_node_enhanced.shape)
df_node_enhanced.to_pickle(args.path_save_feature_mk)