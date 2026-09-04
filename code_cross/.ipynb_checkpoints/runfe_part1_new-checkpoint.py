import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
import time
from multiprocessing import Pool, cpu_count
import joblib
from joblib import Parallel, delayed
import gc
import os

# 1. 建立解析器
parser = argparse.ArgumentParser(description='makefea_part1_new')

# 2. 定义参数
parser.add_argument('--path_data', type=str, default='../data/phase1/gdata.npz')
parser.add_argument('--path_save_feature', type=str, default='../feature/phase1/feature_sampled_graph.pkl')
parser.add_argument('--sub_ratio', type=float, default=1.0)
parser.add_argument('--n_jobs', type=int, default=-1, help='并行进程数，-1表示使用所有CPU核心')
parser.add_argument('--batch_size', type=int, default=3, help='每批处理的任务数量')
parser.add_argument('--max_neighbors', type=int, default=15, help='最大邻居采样数')

# 3. 解析命令行
args = parser.parse_args()

# 自动设置进程数
if args.n_jobs == -1:
    args.n_jobs = min(cpu_count(), 12)
print(f"🚀 使用 {args.n_jobs} 个CPU核心进行并行计算")

def timer_decorator(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"\n⏰ 开始执行: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"✅ 完成: {func.__name__}, 耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
        return result
    return wrapper

def safe_div(a, b, fill=0):
    return np.divide(a, b, out=np.full_like(a, fill, dtype=float), where=b!=0)

def memory_cleanup():
    """强制垃圾回收"""
    gc.collect()

def save_features_batch(features, batch_id, save_dir):
    """保存特征批次"""
    batch_path = os.path.join(save_dir, f'features_batch_{batch_id:03d}.pkl')
    with open(batch_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"💾 保存批次 {batch_id} -> {batch_path} (特征数: {features.shape[1]})")
    return batch_path

def load_and_merge_batches(batch_files, final_path):
    """加载并合并所有批次"""
    print(f"🔄 开始合并 {len(batch_files)} 个批次...")
    
    merged_features = None
    for i, batch_file in enumerate(batch_files):
        print(f"📂 加载批次 {i+1}/{len(batch_files)}: {os.path.basename(batch_file)}")
        with open(batch_file, 'rb') as f:
            batch_features = pickle.load(f)
        
        if merged_features is None:
            merged_features = batch_features
        else:
            merged_features = pd.concat([merged_features, batch_features], axis=1)
        
        # 删除临时文件
        os.remove(batch_file)
        print(f"🗑️ 删除临时文件: {batch_file}")
        memory_cleanup()
    
    print(f"🎉 合并完成! 总特征数: {merged_features.shape[1]}")
    
    # 保存最终特征
    with open(final_path, 'wb') as f:
        pickle.dump(merged_features, f)
    print(f"💾 最终特征保存到: {final_path}")
    
    return merged_features

@timer_decorator
def build_sampled_graph_features_fastest(edge_index, node_features, edge_timestamp, edge_type, max_neighbors=15):
    """
    分批构建采样邻居的特征聚合
    """
    N = len(node_features)
    
    print("📊 创建节点特征DataFrame...")
    if node_features.dtype == np.float64:
        print("  🔧 将原始特征从float64转换为float32...")
        node_features = node_features.astype(np.float32)
    
    feat_df = pd.DataFrame(node_features, columns=[f'f{i}' for i in range(node_features.shape[1])])
    feat_df['node_id'] = range(N)
    
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    feat_df[numeric_cols] = feat_df[numeric_cols].astype(np.float32)
    
    print("🔗 构建边DataFrame...")
    edge_df = pd.DataFrame({
        'source': edge_index[:, 0],
        'target': edge_index[:, 1],
        'timestamp': edge_timestamp.astype(np.float32),
        'edge_type': edge_type.astype(np.int32)
    })

    n_init = edge_df.shape[0]
    print('->'*10, 'df_edge init 样本量为: {}'.format(n_init))
    if args.sub_ratio < 1:
        print(f"📉 进行下采样，比例: {args.sub_ratio}")
        edge_df = edge_df.sample(frac=args.sub_ratio, random_state=42)
        n_subsample = edge_df.shape[0]
        print('->'*10, 'df_edge 下采样后样本量为: {}'.format(n_subsample))

    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(args.path_save_feature), 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)
    print(f"📁 临时目录: {temp_dir}")
    
    # 定义所有图任务
    all_graph_tasks = [
        # 基础拓扑图
        ('directed_1hop_out', build_directed_out_features, (feat_df, edge_df, max_neighbors)),
        ('directed_1hop_in', build_directed_in_features, (feat_df, edge_df, max_neighbors)),
        ('undirected_1hop', build_undirected_features, (feat_df, edge_df, max_neighbors)),
        ('directed_2hop_out', build_directed_two_hop_out_features, (feat_df, edge_df, max_neighbors)),
        ('directed_2hop_in', build_directed_two_hop_in_features, (feat_df, edge_df, max_neighbors)),
        ('undirected_2hop', build_undirected_two_hop_features, (feat_df, edge_df, max_neighbors)),
        ('mixed_propagation', build_mixed_propagation_features, (feat_df, edge_df, max_neighbors)),
        
        # 业务拓扑图
        ('recent_activity', build_recent_activity_subgraph, (feat_df, edge_df, max_neighbors)),
        ('high_risk_propagation', build_high_risk_propagation_subgraph, (feat_df, edge_df, max_neighbors)),
        ('core_business', build_core_business_subgraph, (feat_df, edge_df, max_neighbors)),
        ('financial_hub', build_financial_hub_subgraph, (feat_df, edge_df, max_neighbors)),
        ('transaction_chain', build_transaction_chain_subgraph, (feat_df, edge_df, max_neighbors)),
    ]
    
    # 分批处理
    batch_files = []
    total_batches = (len(all_graph_tasks) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(all_graph_tasks))
        batch_tasks = all_graph_tasks[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"🔄 处理批次 {batch_idx + 1}/{total_batches} (任务 {start_idx + 1}-{end_idx})")
        print(f"📋 本批任务: {[task[0] for task in batch_tasks]}")
        
        # 处理当前批次
        batch_features = process_batch_tasks(batch_tasks, N)
        
        # 保存当前批次
        batch_file = save_features_batch(batch_features, batch_idx, temp_dir)
        batch_files.append(batch_file)
        
        # 彻底清理内存
        del batch_features, batch_tasks
        memory_cleanup()
    
    # 清理原始数据
    del feat_df, edge_df
    memory_cleanup()
    
    # 合并所有批次
    final_features = load_and_merge_batches(batch_files, args.path_save_feature)
    
    # 清理临时目录
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)
        print(f"🗑️ 删除临时目录: {temp_dir}")
    
    return final_features

def process_batch_tasks(batch_tasks, N):
    """处理单个批次的任务"""
    batch_features = pd.DataFrame(index=range(N))
    
    def process_single_task(task):
        name, func, func_args = task
        print(f"  🎯 开始: {name}")
        task_features = pd.DataFrame(index=range(N))
        task_features = func(task_features, *func_args)
        print(f"  ✅ 完成: {name} -> {task_features.shape[1]} 个特征")
        return task_features
    
    # 并行处理当前批次的任务
    results = Parallel(n_jobs=min(args.n_jobs, len(batch_tasks)), backend='loky')(
        delayed(process_single_task)(task) for task in batch_tasks
    )
    
    # 合并当前批次结果（使用concat避免碎片化）
    for result in results:
        batch_features = pd.concat([batch_features, result], axis=1)
    
    return batch_features

@timer_decorator
def add_neighbor_features_with_timestamp(features, feat_df, neighbor_edges, node_col, neighbor_col, prefix):
    """优化版本：并行计算邻居特征统计"""
    if len(neighbor_edges) == 0:
        print(f"  ⚠️ {prefix}: 没有邻居边数据，跳过")
        return features
    
    print(f"  📈 {prefix}: 合并邻居特征...")
    
    # 优化合并：只选择需要的列
    neighbor_feat_df = neighbor_edges[[node_col, neighbor_col, 'timestamp', 'edge_type']].merge(
        feat_df, 
        left_on=neighbor_col, 
        right_on='node_id'
    )
    
    if len(neighbor_feat_df) == 0:
        print(f"  ⚠️ {prefix}: 合并后无数据，跳过")
        return features
    
    print(f"  📊 {prefix}: 计算特征统计量...")
    
    # 1. 节点特征统计
    feature_cols = [fe for fe in feat_df.columns if 'f' in fe]
    stats_dfs = []
    
    for col in feature_cols:
        if col in neighbor_feat_df.columns:
            stats = neighbor_feat_df.groupby(node_col)[col].agg(['mean', 'std', 'max', 'min'])
            stats.columns = [f'{prefix}_{col}_mean', f'{prefix}_{col}_std', f'{prefix}_{col}_max', f'{prefix}_{col}_min']
            stats_dfs.append(stats)
    
    # 2. 时间戳特征统计
    timestamp_stats = neighbor_edges.groupby(node_col)['timestamp'].agg([
        'mean', 'std', 'max', 'min'
    ])
    timestamp_stats.columns = [f'{prefix}_timestamp_mean', f'{prefix}_timestamp_std', 
                             f'{prefix}_timestamp_max', f'{prefix}_timestamp_min']
    timestamp_stats[f'{prefix}_timestamp_range'] = timestamp_stats[f'{prefix}_timestamp_max'] - timestamp_stats[f'{prefix}_timestamp_min']
    stats_dfs.append(timestamp_stats)
    
    # 3. 时间间隔统计
    def compute_intervals(group):
        if len(group) < 2:
            return pd.Series({'mean': 0, 'std': 0, 'max': 0, 'min': 0})
        sorted_times = np.sort(group.values)
        intervals = np.diff(sorted_times)
        return pd.Series({
            'mean': np.mean(intervals), 'std': np.std(intervals),
            'max': np.max(intervals), 'min': np.min(intervals)
        })
    
    interval_stats = neighbor_edges.groupby(node_col)['timestamp'].apply(compute_intervals).unstack()
    if not interval_stats.empty:
        interval_stats.columns = [f'{prefix}_interval_mean', f'{prefix}_interval_std', 
                                f'{prefix}_interval_max', f'{prefix}_interval_min']
        stats_dfs.append(interval_stats)
    
    # 4. 边类型统计（只计算前3种主要类型）
    main_types = neighbor_edges['edge_type'].value_counts().head(3).index
    for etype in main_types:
        type_data = neighbor_edges[neighbor_edges['edge_type'] == etype]
        if len(type_data) > 0:
            type_stats = type_data.groupby(node_col)['timestamp'].agg([
                'mean', 'std', 'max', 'min', 'count'
            ])
            type_stats.columns = [f'{prefix}_type_{etype}_mean', f'{prefix}_type_{etype}_std',
                                f'{prefix}_type_{etype}_max', f'{prefix}_type_{etype}_min',
                                f'{prefix}_type_{etype}_count']
            type_stats[f'{prefix}_type_{etype}_range'] = type_stats[f'{prefix}_type_{etype}_max'] - type_stats[f'{prefix}_type_{etype}_min']
            stats_dfs.append(type_stats)
    
    # 一次性合并所有统计结果
    if stats_dfs:
        combined_stats = pd.concat(stats_dfs, axis=1)
        features = pd.concat([features, combined_stats], axis=1)
    
    # 统计信息
    node_feat_count = len(feature_cols) * 4
    time_feat_count = len(stats_dfs) * 4 - node_feat_count  # 近似计算
    print(f"  ✅ {prefix}: 完成 {node_feat_count} 节点特征 + {time_feat_count} 时间特征")
    
    # 清理内存
    del neighbor_feat_df, stats_dfs
    if 'combined_stats' in locals():
        del combined_stats
    memory_cleanup()
    
    return features

# 图构建函数保持不变，但会调用优化后的add_neighbor_features_with_timestamp
@timer_decorator
def build_directed_out_features(features, feat_df, edge_df, max_neighbors):
    """构建有向图出边特征"""
    sampled_out_edges = (edge_df.sample(frac=1, random_state=42)
                        .groupby('source')
                        .head(max_neighbors)
                        .reset_index(drop=True))
    
    print(f"  采样后出边数量: {len(sampled_out_edges):,}")
    features = add_neighbor_features_with_timestamp(features, feat_df, sampled_out_edges, 
                                                  'source', 'target', 'directed_1hop_out')
    del sampled_out_edges
    memory_cleanup()
    return features

@timer_decorator
def build_directed_in_features(features, feat_df, edge_df, max_neighbors):
    """构建有向图入边特征"""
    sampled_in_edges = (edge_df.sample(frac=1, random_state=42)
                       .groupby('target')
                       .head(max_neighbors)
                       .reset_index(drop=True))
    
    print(f"  采样后入边数量: {len(sampled_in_edges):,}")
    features = add_neighbor_features_with_timestamp(features, feat_df, sampled_in_edges,
                                                  'target', 'source', 'directed_1hop_in')
    del sampled_in_edges
    memory_cleanup()
    return features

@timer_decorator
def build_undirected_features(features, feat_df, edge_df, max_neighbors):
    """构建无向图特征"""
    undirected_edges = pd.concat([
        edge_df.rename(columns={'source': 'node', 'target': 'neighbor'}),
        edge_df.rename(columns={'source': 'neighbor', 'target': 'node'})
    ]).drop_duplicates(subset=['node', 'neighbor']).reset_index(drop=True)
    
    print(f"  去重后无向边总数: {len(undirected_edges):,}")
    
    sampled_undirected = (undirected_edges.sample(frac=1, random_state=42)
                         .groupby('node')
                         .head(max_neighbors)
                         .reset_index(drop=True))
    
    print(f"  采样后无向边数量: {len(sampled_undirected):,}")
    features = add_neighbor_features_with_timestamp(features, feat_df, sampled_undirected,
                                                  'node', 'neighbor', 'undirected_1hop')
    del undirected_edges, sampled_undirected
    memory_cleanup()
    return features

@timer_decorator
def build_directed_two_hop_out_features(features, feat_df, edge_df, max_neighbors):
    """构建有向图二跳特征 - 出边传播"""
    sampled_out_edges = (edge_df.sample(frac=1, random_state=42)
                        .groupby('source')
                        .head(max_neighbors)
                        .reset_index(drop=True))
    
    if len(sampled_out_edges) > 0:
        two_hop_edges = (sampled_out_edges.merge(
            sampled_out_edges[['source', 'target', 'timestamp', 'edge_type']].rename(
                columns={'source': 'mid_node', 'target': 'target2', 'timestamp': 'timestamp2', 'edge_type': 'type2'}
            ),
            left_on='target', right_on='mid_node'
        )[['source', 'target2', 'timestamp', 'edge_type', 'timestamp2', 'type2']]
         .rename(columns={'target2': 'target'})
         .drop_duplicates(subset=['source', 'target'])
         .query('source != target')
        )
        
        print(f"  去重后二跳出边数量: {len(two_hop_edges):,}")
        
        sampled_two_hop = (two_hop_edges.sample(frac=1, random_state=42)
                          .groupby('source')
                          .head(max_neighbors)
                          .reset_index(drop=True))
        
        print(f"  采样后二跳出边数量: {len(sampled_two_hop):,}")
        
        sampled_two_hop['timestamp'] = sampled_two_hop[['timestamp', 'timestamp2']].mean(axis=1)
        sampled_two_hop['edge_type'] = sampled_two_hop['edge_type']
        
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_two_hop,
                                                      'source', 'target', 'directed_2hop_out')
        del two_hop_edges, sampled_two_hop
    else:
        print("  ⚠️ 没有出边数据，跳过有向二跳出边特征")
    
    del sampled_out_edges
    memory_cleanup()
    return features

@timer_decorator
def build_directed_two_hop_in_features(features, feat_df, edge_df, max_neighbors):
    """构建有向图二跳特征 - 入边传播"""
    sampled_in_edges = (edge_df.sample(frac=1, random_state=42)
                       .groupby('target')
                       .head(max_neighbors)
                       .reset_index(drop=True))
    
    if len(sampled_in_edges) > 0:
        two_hop_edges = (sampled_in_edges.merge(
            sampled_in_edges[['source', 'target', 'timestamp', 'edge_type']].rename(
                columns={'source': 'source2', 'target': 'mid_node', 'timestamp': 'timestamp2', 'edge_type': 'type2'}
            ),
            left_on='source', right_on='mid_node'
        )[['target', 'source2', 'timestamp', 'edge_type', 'timestamp2', 'type2']]
         .rename(columns={'source2': 'source', 'target': 'node'})
         .drop_duplicates(subset=['node', 'source'])
         .query('node != source')
         .rename(columns={'source': 'neighbor'})
        )
        
        print(f"  去重后二跳入边数量: {len(two_hop_edges):,}")
        
        sampled_two_hop = (two_hop_edges.sample(frac=1, random_state=42)
                          .groupby('node')
                          .head(max_neighbors)
                          .reset_index(drop=True))
        
        print(f"  采样后二跳入边数量: {len(sampled_two_hop):,}")
        
        sampled_two_hop['timestamp'] = sampled_two_hop[['timestamp', 'timestamp2']].mean(axis=1)
        sampled_two_hop['edge_type'] = sampled_two_hop['edge_type']
        
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_two_hop,
                                                      'node', 'neighbor', 'directed_2hop_in')
        del two_hop_edges, sampled_two_hop
    else:
        print("  ⚠️ 没有入边数据，跳过有向二跳入边特征")
    
    del sampled_in_edges
    memory_cleanup()
    return features

@timer_decorator
def build_undirected_two_hop_features(features, feat_df, edge_df, max_neighbors):
    """构建无向图二跳特征"""
    undirected_edges = pd.concat([
        edge_df.rename(columns={'source': 'node', 'target': 'neighbor'}),
        edge_df.rename(columns={'source': 'neighbor', 'target': 'node'})
    ]).drop_duplicates(subset=['node', 'neighbor']).reset_index(drop=True)
    
    sampled_undirected = (undirected_edges.sample(frac=1, random_state=42)
                         .groupby('node')
                         .head(max_neighbors)
                         .reset_index(drop=True))
    
    if len(sampled_undirected) > 0:
        undirected_two_hop = (sampled_undirected.merge(
            sampled_undirected[['node', 'neighbor', 'timestamp', 'edge_type']].rename(
                columns={'node': 'mid_node', 'neighbor': 'neighbor2', 'timestamp': 'timestamp2', 'edge_type': 'type2'}
            ),
            left_on='neighbor', right_on='mid_node'
        )[['node', 'neighbor2', 'timestamp', 'edge_type', 'timestamp2', 'type2']]
         .rename(columns={'neighbor2': 'neighbor'})
         .drop_duplicates(subset=['node', 'neighbor'])
         .query('node != neighbor')
        )
        
        print(f"  去重后无向二跳边数量: {len(undirected_two_hop):,}")
        
        sampled_undirected_two_hop = (undirected_two_hop.sample(frac=1, random_state=42)
                                     .groupby('node')
                                     .head(max_neighbors)
                                     .reset_index(drop=True))
        
        print(f"  采样后无向二跳边数量: {len(sampled_undirected_two_hop):,}")
        
        sampled_undirected_two_hop['timestamp'] = sampled_undirected_two_hop[['timestamp', 'timestamp2']].mean(axis=1)
        sampled_undirected_two_hop['edge_type'] = sampled_undirected_two_hop['edge_type']
        
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_undirected_two_hop,
                                                      'node', 'neighbor', 'undirected_2hop')
        del undirected_two_hop, sampled_undirected_two_hop
    else:
        print("  ⚠️ 没有无向边数据，跳过无向二跳特征")
    
    del undirected_edges, sampled_undirected
    memory_cleanup()
    return features

@timer_decorator
def build_mixed_propagation_features(features, feat_df, edge_df, max_neighbors):
    """构建混合传播图特征"""
    sampled_out_edges = (edge_df.sample(frac=1, random_state=42)
                        .groupby('source')
                        .head(max_neighbors)
                        .reset_index(drop=True))
    
    sampled_in_edges = (edge_df.sample(frac=1, random_state=42)
                       .groupby('target')
                       .head(max_neighbors)
                       .reset_index(drop=True))
    
    if len(sampled_out_edges) > 0 and len(sampled_in_edges) > 0:
        mixed_edges = (sampled_out_edges.merge(
            sampled_in_edges[['source', 'target', 'timestamp', 'edge_type']].rename(
                columns={'source': 'neighbor', 'target': 'mid_node', 'timestamp': 'timestamp2', 'edge_type': 'type2'}
            ),
            left_on='target', right_on='mid_node'
        )[['source', 'neighbor', 'timestamp', 'edge_type', 'timestamp2', 'type2']]
         .rename(columns={'source': 'node'})
         .drop_duplicates(subset=['node', 'neighbor'])
         .query('node != neighbor')
        )
        
        print(f"  混合传播边数量: {len(mixed_edges):,}")
        
        sampled_mixed = (mixed_edges.sample(frac=1, random_state=42)
                         .groupby('node')
                         .head(max_neighbors)
                         .reset_index(drop=True))
        
        print(f"  采样后混合传播边数量: {len(sampled_mixed):,}")
        
        sampled_mixed['timestamp'] = sampled_mixed[['timestamp', 'timestamp2']].mean(axis=1)
        sampled_mixed['edge_type'] = sampled_mixed['edge_type']
        
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_mixed,
                                                      'node', 'neighbor', 'mixed_propagation')
        del mixed_edges, sampled_mixed
    
    del sampled_out_edges, sampled_in_edges
    memory_cleanup()
    return features

@timer_decorator
def build_recent_activity_subgraph(features, feat_df, edge_df, max_neighbors=15):
    """构建最近活动子图"""
    print("🕒 构建最近活动子图...")
    
    time_threshold = edge_df['timestamp'].quantile(0.7)
    recent_edges = edge_df[edge_df['timestamp'] >= time_threshold].copy()
    
    print(f"  最近活动边数量: {len(recent_edges):,}")
    
    if len(recent_edges) == 0:
        print("  ⚠️ 没有最近活动边，跳过")
        return features
    
    # 构建无向特征
    recent_undirected = pd.concat([
        recent_edges.rename(columns={'source': 'node', 'target': 'neighbor'}),
        recent_edges.rename(columns={'source': 'neighbor', 'target': 'node'})
    ]).drop_duplicates(subset=['node', 'neighbor']).reset_index(drop=True)
    
    sampled_recent_undirected = (recent_undirected.sample(frac=1, random_state=42)
                                .groupby('node')
                                .head(max_neighbors)
                                .reset_index(drop=True))
    
    if len(sampled_recent_undirected) > 0:
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_recent_undirected,
                                                      'node', 'neighbor', 'recent_activity')
    
    del recent_edges, recent_undirected, sampled_recent_undirected
    memory_cleanup()
    return features

@timer_decorator
def build_high_risk_propagation_subgraph(features, feat_df, edge_df, max_neighbors=15):
    """构建高风险传播子图"""
    print("⚠️ 构建高风险传播子图...")

    if 'dgraphfin' in args.path_data:  
        risk_threshold = feat_df['f16'].quantile(0.8)
        high_risk_nodes = feat_df[feat_df['f16'] > risk_threshold]['node_id'].values
    else:
        risk_threshold = feat_df['f9'].quantile(0.8)
        high_risk_nodes = feat_df[feat_df['f9'] > risk_threshold]['node_id'].values
    
    print(f"  识别到高风险节点: {len(high_risk_nodes):,}")
    
    if len(high_risk_nodes) == 0:
        print("  ⚠️ 没有高风险节点，跳过")
        return features
    
    # 高风险节点之间的边
    risk_cluster_edges = edge_df[
        edge_df['source'].isin(high_risk_nodes) & 
        edge_df['target'].isin(high_risk_nodes)
    ].copy()
    
    print(f"  风险集群边数量: {len(risk_cluster_edges):,}")
    
    if len(risk_cluster_edges) > 0:
        risk_cluster_undirected = pd.concat([
            risk_cluster_edges.rename(columns={'source': 'node', 'target': 'neighbor'}),
            risk_cluster_edges.rename(columns={'source': 'neighbor', 'target': 'node'})
        ]).drop_duplicates(subset=['node', 'neighbor']).reset_index(drop=True)
        
        sampled_risk_cluster = (risk_cluster_undirected.sample(frac=1, random_state=42)
                              .groupby('node')
                              .head(max_neighbors)
                              .reset_index(drop=True))
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_risk_cluster,
                                                      'node', 'neighbor', 'risk_cluster')
    
    del risk_cluster_edges
    memory_cleanup()
    return features

@timer_decorator
def build_core_business_subgraph(features, feat_df, edge_df, max_neighbors=15):
    """构建核心业务子图"""
    print("🏢 构建核心业务子图...")
    
    out_degree = edge_df.groupby('source').size()
    degree_threshold = out_degree.quantile(0.8)
    core_business_nodes = out_degree[out_degree > degree_threshold].index.values
    
    print(f"  识别到核心业务节点: {len(core_business_nodes):,}")
    
    if len(core_business_nodes) == 0:
        print("  ⚠️ 没有核心业务节点，跳过")
        return features
    
    # 核心业务节点之间的边
    core_edges = edge_df[
        edge_df['source'].isin(core_business_nodes) & 
        edge_df['target'].isin(core_business_nodes)
    ].copy()
    
    print(f"  核心内部边数量: {len(core_edges):,}")
    
    if len(core_edges) > 0:
        core_undirected = pd.concat([
            core_edges.rename(columns={'source': 'node', 'target': 'neighbor'}),
            core_edges.rename(columns={'source': 'neighbor', 'target': 'node'})
        ]).drop_duplicates(subset=['node', 'neighbor']).reset_index(drop=True)
        
        sampled_core = (core_undirected.sample(frac=1, random_state=42)
                       .groupby('node')
                       .head(max_neighbors)
                       .reset_index(drop=True))
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_core,
                                                      'node', 'neighbor', 'core_business')
    
    del core_edges
    memory_cleanup()
    return features

@timer_decorator
def build_financial_hub_subgraph(features, feat_df, edge_df, max_neighbors=15):
    """构建金融枢纽子图"""
    print("🎯 构建金融枢纽子图...")
    
    out_degree = edge_df.groupby('source').size()
    in_degree = edge_df.groupby('target').size()
    total_degree = out_degree.add(in_degree, fill_value=0)
    
    hub_threshold = total_degree.quantile(0.85)
    hub_nodes = total_degree[total_degree > hub_threshold].index.values
    
    print(f"  识别到金融枢纽节点: {len(hub_nodes):,}")
    
    if len(hub_nodes) == 0:
        print("  ⚠️ 没有金融枢纽节点，跳过")
        return features
    
    # 枢纽节点之间的连接
    hub_connections = edge_df[
        (edge_df['source'].isin(hub_nodes)) | (edge_df['target'].isin(hub_nodes))
    ].copy()
    
    print(f"  枢纽连接边数量: {len(hub_connections):,}")
    
    if len(hub_connections) > 0:
        hub_undirected = pd.concat([
            hub_connections.rename(columns={'source': 'node', 'target': 'neighbor'}),
            hub_connections.rename(columns={'source': 'neighbor', 'target': 'node'})
        ]).drop_duplicates(subset=['node', 'neighbor']).reset_index(drop=True)
        
        hub_undirected = hub_undirected[
            hub_undirected['node'].isin(hub_nodes) | 
            hub_undirected['neighbor'].isin(hub_nodes)
        ]
        
        sampled_hub = (hub_undirected.sample(frac=1, random_state=42)
                      .groupby('node')
                      .head(max_neighbors)
                      .reset_index(drop=True))
        
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_hub,
                                                      'node', 'neighbor', 'financial_hub')
    
    del hub_connections
    memory_cleanup()
    return features

@timer_decorator
def build_transaction_chain_subgraph(features, feat_df, edge_df, max_neighbors=15):
    """构建交易链子图"""
    print("🔗 构建交易链子图...")
    
    sampled_out = (edge_df.sample(frac=1, random_state=42)
                  .groupby('source')
                  .head(max_neighbors)
                  .reset_index(drop=True))
    
    if len(sampled_out) == 0:
        print("  ⚠️ 没有出边数据，跳过交易链子图")
        return features
    
    # 构建三跳交易链
    three_hop_chain = (sampled_out.merge(
        sampled_out[['source', 'target', 'timestamp', 'edge_type']].rename(
            columns={'source': 'hop2_source', 'target': 'hop2_target', 'timestamp': 'timestamp2', 'edge_type': 'type2'}
        ),
        left_on='target', right_on='hop2_source'
    ).merge(
        sampled_out[['source', 'target', 'timestamp', 'edge_type']].rename(
            columns={'source': 'hop3_source', 'target': 'hop3_target', 'timestamp': 'timestamp3', 'edge_type': 'type3'}
        ),
        left_on='hop2_target', right_on='hop3_source'
    )[['source', 'hop3_target', 'timestamp', 'edge_type', 'timestamp2', 'type2', 'timestamp3', 'type3']]
     .rename(columns={'hop3_target': 'target'})
     .drop_duplicates(subset=['source', 'target'])
     .query('source != target')
    )
    
    print(f"  三跳交易链数量: {len(three_hop_chain):,}")
    
    if len(three_hop_chain) > 0:
        three_hop_chain['timestamp'] = three_hop_chain[['timestamp', 'timestamp2', 'timestamp3']].mean(axis=1)
        three_hop_chain['edge_type'] = three_hop_chain['edge_type']
        
        sampled_chain = (three_hop_chain.sample(frac=1, random_state=42)
                        .groupby('source')
                        .head(max_neighbors)
                        .reset_index(drop=True))
        
        features = add_neighbor_features_with_timestamp(features, feat_df, sampled_chain,
                                                      'source', 'target', 'transaction_chain')
    
    del sampled_out, three_hop_chain
    memory_cleanup()
    return features

def parallel_data_type_optimization(df, n_jobs=-1):
    """并行优化数据类型为float32"""
    if n_jobs == -1:
        n_jobs = min(cpu_count(), 8)
    
    print(f"🔧 使用 {n_jobs} 个进程并行优化数据类型...")
    
    float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
    int64_cols = df.select_dtypes(include=['int64']).columns.tolist()
    
    print(f"  需要转换的float64列数: {len(float64_cols)}")
    print(f"  需要转换的int64列数: {len(int64_cols)}")
    
    if not float64_cols and not int64_cols:
        print("  ✅ 所有列已经是float32/int32格式")
        return df
    
    # 分批处理列转换
    batch_size = 20
    for i in range(0, len(float64_cols), batch_size):
        batch_cols = float64_cols[i:i + batch_size]
        for col in batch_cols:
            df[col] = df[col].astype(np.float32)
        memory_cleanup()
    
    for i in range(0, len(int64_cols), batch_size):
        batch_cols = int64_cols[i:i + batch_size]
        for col in batch_cols:
            df[col] = df[col].astype(np.int32)
        memory_cleanup()
    
    print(f"  ✅ 完成数据类型优化")
    return df

def main():
    start_time = time.time()
    print("🚀 开始特征工程...")
    
    print("📂 加载数据...")
    data = np.load(args.path_data, allow_pickle='True')
    
    print(f"📊 数据统计:")
    print(f"  - 节点数: {data['x'].shape[0]}")
    print(f"  - 特征数: {data['x'].shape[1]}")
    print(f"  - 边索引形状: {data['edge_index'].shape}")
    print(f"  - 时间戳形状: {data['edge_timestamp'].shape}")
    print(f"  - 边类型形状: {data['edge_type'].shape}")
    
    print("🎯 开始构建图结构特征...")
    features = build_sampled_graph_features_fastest(
        data['edge_index'], 
        data['x'], 
        data['edge_timestamp'],
        data['edge_type'],
        max_neighbors=args.max_neighbors
    )
    
    print("💾 数据清理和优化...")
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    print("🔧 优化数据类型...")
    features = parallel_data_type_optimization(features, args.n_jobs)
    
    memory_cleanup()
    
    # 最终保存
    with open(args.path_save_feature, 'wb') as f:
        pickle.dump(features, f)
    
    del features, data
    memory_cleanup()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 特征工程完成!")
    print(f"⏱️ 总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"💾 保存路径: {args.path_save_feature}")

if __name__ == '__main__':
    main()