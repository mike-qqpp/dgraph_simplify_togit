"""
5分箱特征工程模块 - 基于sklearn的KBinsDiscretizer
简洁高效，从npz文件加载数据，为节点特征创建分箱标签
"""
import numpy as np
import pandas as pd
import argparse
from typing import List
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')


def add_box_features(
    input_path: str,
    output_path: str,
    n_bins: int = 5,
    strategy: str = "quantile"
) -> pd.DataFrame:
    """
    对npz文件中的节点特征进行5分箱
    
    参数:
        input_path: 输入npz文件路径
        output_path: 输出pickle文件路径
        n_bins: 分箱数量，默认5
        strategy: 分箱策略，"quantile"(分位数) 或 "uniform"(等宽)
    
    返回:
        包含分箱特征的DataFrame
    """
    print(f"加载数据: {input_path}")
    
    # 从npz文件加载
    data = np.load(input_path, allow_pickle=True)
    x = data['x']
    
    n_samples, n_features = x.shape
    print(f"  节点数: {n_samples}, 特征维度: {n_features}")
    
    # 原始特征列名
    cols = [f"feature_{i}" for i in range(n_features)]
    
    # 构建DataFrame
    df = pd.DataFrame(x.astype(np.float32), columns=cols)
    
    # 初始化分箱器
    kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    
    # fit
    print(f"分箱fit (strategy={strategy})...")
    kbin.fit(df[cols])
    
    # 打印分箱边界示例
    bin_edges = kbin.bin_edges_
    print(f"\n分箱边界示例 (前3个特征):")
    for i in range(min(3, n_features)):
        edges = bin_edges[i]
        print(f"  feature_{i}: [{', '.join([f'{e:.4f}' for e in edges[:n_bins+1]])}]")
    
    # transform
    print(f"\n分箱transform...")
    box_vals = kbin.transform(df[cols]).astype(int)
    
    # 分箱列名（简洁命名）
    box_cols = [f"bin_{i}" for i in range(n_features)]
    
    # 只保留分箱列，丢弃原始feature列
    df = pd.DataFrame(box_vals, columns=box_cols, index=df.index)
    
    # 打印分箱分布
    print(f"\n分箱分布示例 (前5个特征):")
    for i in range(min(5, n_features)):
        col = f"bin_{i}"
        dist = df[col].value_counts().sort_index()
        dist_str = ", ".join([f"Box{int(k)}:{int(v)}" for k, v in dist.items()])
        print(f"  {col}: {dist_str}")
    
    # 保存
    with open(output_path, 'wb') as f:
        pd.to_pickle(df, f)
    print(f"\n保存: {output_path}")
    print(f"  形状: {df.shape}")
    print(f"  新增列: {len(box_cols)} 个 (bin_i)")
    
    return df


def add_box_features_multi(
    input_paths: List[str],
    output_paths: List[str],
    n_bins: int = 5,
    strategy: str = "quantile"
) -> List[pd.DataFrame]:
    """
    对多个npz文件进行统一的5分箱（第一张表fit，其他表transform）
    
    参数:
        input_paths: 输入npz文件路径列表
        output_paths: 输出pickle文件路径列表（与输入一一对应）
        n_bins: 分箱数量，默认5
        strategy: 分箱策略，"quantile"(分位数) 或 "uniform"(等宽)
    
    返回:
        DataFrame列表
    """
    if len(input_paths) != len(output_paths):
        raise ValueError("input_paths 和 output_paths 长度必须一致")
    
    print(f"加载第一个文件进行fit: {input_paths[0]}")
    
    # 加载第一个文件
    data = np.load(input_paths[0], allow_pickle=True)
    x = data['x']
    
    n_samples, n_features = x.shape
    cols = [f"feature_{i}" for i in range(n_features)]
    
    print(f"  节点数: {n_samples}, 特征维度: {n_features}")
    
    # 构建DataFrame
    df_first = pd.DataFrame(x.astype(np.float32), columns=cols)
    
    # 初始化分箱器并fit
    kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    print(f"分箱fit (strategy={strategy})...")
    kbin.fit(df_first[cols])
    
    # 打印分箱边界
    bin_edges = kbin.bin_edges_
    print(f"\n分箱边界示例 (前3个特征):")
    for i in range(min(3, n_features)):
        edges = bin_edges[i]
        print(f"  feature_{i}: [{', '.join([f'{e:.4f}' for e in edges[:n_bins+1]])}]")
    
    box_cols = [f"bin_{i}" for i in range(n_features)]
    results = []
    
    for idx, input_path in enumerate(input_paths):
        print(f"\n处理 [{idx+1}/{len(input_paths)}]: {input_path}")
        
        # 加载数据
        data = np.load(input_path, allow_pickle=True)
        x = data['x']
        n_samples = x.shape[0]
        
        # transform
        box_vals = kbin.transform(x).astype(int)
        
        # 只保留分箱列
        df = pd.DataFrame(box_vals, columns=box_cols, index=range(n_samples))
        
        # 打印分布
        print(f"  节点数: {n_samples}")
        print(f"  分箱分布 (前3个):")
        for i in range(min(3, n_features)):
            col = f"bin_{i}"
            dist = df[col].value_counts().sort_index()
            dist_str = ", ".join([f"Box{int(k)}:{int(v)}" for k, v in dist.items()])
            print(f"    {col}: {dist_str}")
        
        # 保存
        with open(output_paths[idx], 'wb') as f:
            pd.to_pickle(df, f)
        print(f"  保存: {output_paths[idx]}, 形状: {df.shape}")
        
        results.append(df)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='5分箱特征工程')
    parser.add_argument('--input_path', type=str, default=None,
                       help='输入npz文件路径（单文件模式）')
    parser.add_argument('--output_path', type=str, default=None,
                       help='输出pickle文件路径（单文件模式）')
    parser.add_argument('--input_paths', type=str, nargs='+', default=None,
                       help='输入npz文件路径列表（多文件模式，统一分箱）')
    parser.add_argument('--output_paths', type=str, nargs='+', default=None,
                       help='输出pickle文件路径列表（与输入一一对应）')
    parser.add_argument('--n_bins', type=int, default=5,
                       help='分箱数量，默认5')
    parser.add_argument('--strategy', type=str, default='quantile',
                       choices=['quantile', 'uniform'],
                       help='分箱策略，quantile(分位数)或uniform(等宽)，默认quantile')
    args = parser.parse_args()
    
    # 模式判断
    if args.input_paths and args.output_paths:
        # 多文件模式
        add_box_features_multi(
            input_paths=args.input_paths,
            output_paths=args.output_paths,
            n_bins=args.n_bins,
            strategy=args.strategy
        )
    elif args.input_path and args.output_path:
        # 单文件模式
        add_box_features(
            input_path=args.input_path,
            output_path=args.output_path,
            n_bins=args.n_bins,
            strategy=args.strategy
        )
    else:
        print("请指定 --input_path 和 --output_path (单文件模式)")
        print("或 --input_paths 和 --output_paths (多文件模式，统一分箱)")
    
    print("\n5分箱特征工程完成!")


if __name__ == "__main__":
    main()
