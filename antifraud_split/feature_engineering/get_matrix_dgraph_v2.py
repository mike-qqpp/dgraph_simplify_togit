#!/usr/bin/env python
# encoding: utf-8

########get_matrix.py
"""
数据集的矩阵幂计算脚本
用于生成1-10阶邻接矩阵幂，格式与Yelp/Amazon/S-FFSD数据集兼容

功能说明:
    - 计算邻接矩阵的1-10阶幂
    - 支持GPU加速计算
    - 内存优化：大规模图自动切换到稀疏矩阵模式
    - 支持灵活的数据集名称参数

使用方法:
    # 处理 dgraphfin 数据集（默认）
    python get_matrix.py
    
    # 处理其他数据集
    python get_matrix.py --dataset mydataset --data_dir ../data
"""

import pickle
import os
import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
import time
import math
import random
import warnings
import argparse

warnings.filterwarnings('ignore')


def create_adjacency_matrix(adj_list, n):
    """
    从邻接列表创建邻接矩阵
    
    Args:
        adj_list: 邻接列表字典 {node: {neighbors}}
        n: 节点总数
    Returns:
        numpy邻接矩阵
    """
    adj_matrix = np.zeros((n, n))
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1
    return adj_matrix


def create_adjacency_matrix_sparse(adj_list, n):
    """
    从邻接列表创建稀疏邻接矩阵（内存优化版本）
    
    Args:
        adj_list: 邻接列表字典 {node: {neighbors}}
        n: 节点总数
    Returns:
        scipy稀疏矩阵
    """
    rows = []
    cols = []
    
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            rows.append(node)
            cols.append(neighbor)
    
    data = np.ones(len(rows))
    adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return adj_matrix


def block_matrix_multiply(A, B, block_size, device):
    """
    分块矩阵乘法
    
    Args:
        A, B: 输入矩阵
        block_size: 分块大小
        device: 计算设备
    Returns:
        乘积矩阵C
    """
    n = A.shape[0]
    C = torch.zeros((n, n), device=device)
    
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        A_block = A[i:i_end, :]
        
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            B_block = B[:, j:j_end]
            
            C[i:i_end, j:j_end] = torch.matmul(A_block, B_block)
    
    return C


def compute_sparse_matrix_powers(adj_sparse, n, matrix_prefix, max_k=10):
    """
    使用scipy稀疏矩阵计算矩阵幂（适合大规模图）
    
    Args:
        adj_sparse: 稀疏邻接矩阵
        n: 节点数
        matrix_prefix: 输出文件前缀
        max_k: 最大幂次
    """
    print(f"\n使用稀疏矩阵计算1-{max_k}阶矩阵幂...")
    print(f"节点数: {n}")
    
    # 保存A^1
    file_name = f'{matrix_prefix}1.npz'
    sp.save_npz(file_name, adj_sparse)
    print(f"保存A^1到 {file_name}")
    
    # 迭代计算A^k = A^(k-1) * A
    current = adj_sparse
    
    total_start_time = time.time()
    
    for k in tqdm(range(2, max_k + 1), desc="计算幂次"):
        k_start_time = time.time()
        
        # 稀疏矩阵乘法
        current = current.dot(adj_sparse)
        
        # 统计信息
        nnz = current.nnz
        density = nnz / (n * n)
        
        # 检查是否变得太稠密
        if density > 0.001:  # 0.1%密度阈值
            print(f"\n  警告: k={k}时矩阵变得太稠密 (密度={density:.4f})")
            print(f"     停止计算，截断至k={k-1}")
            break
        
        # 保存结果
        file_name = f'{matrix_prefix}{k}.npz'
        sp.save_npz(file_name, current)
        
        k_elapsed = time.time() - k_start_time
        print(f"  k={k}: 非零元素={nnz}, 密度={density:.6f}, 用时={k_elapsed:.2f}秒")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n稀疏矩阵幂计算完成！总用时: {total_elapsed:.2f}秒")
    print(f"注意: 矩阵幂以.npz格式保存（scipy sparse format）")


def compute_dense_matrix_powers_gpu(adj_matrix, n, block_size, matrix_prefix, device):
    """
    使用GPU稠密矩阵计算矩阵幂（仅对小规模图有效）
    
    Args:
        adj_matrix: numpy邻接矩阵
        n: 节点数
        block_size: 分块大小
        matrix_prefix: 输出文件前缀
        device: 计算设备
    """
    print("\n转换为GPU稠密矩阵...")
    print(f"矩阵大小: {n}x{n}")
    
    adj_tensor = torch.from_numpy(adj_matrix).float().to(device)
    
    # 保存k=1
    file_name = f'{matrix_prefix}1.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(adj_matrix, f)
    print(f"保存邻接矩阵: {file_name}")
    
    # 迭代计算
    print("\n迭代计算2-10阶矩阵幂...")
    
    prev_power = adj_tensor.clone()
    prev_power_binary = (prev_power != 0).float()
    
    total_start_time = time.time()
    
    for k in tqdm(range(2, 11), desc="计算幂次"):
        k_start_time = time.time()
        
        current_power = block_matrix_multiply(prev_power, adj_tensor, block_size, device)
        current_power_binary = (current_power != 0).float()
        
        # 计算A^k - A^(k-1)用于路径计数
        result = current_power_binary - prev_power_binary
        result = torch.maximum(result, torch.tensor(0, device=device))
        
        # 添加自环
        result = result + torch.eye(n, device=device)
        
        file_name = f'{matrix_prefix}{k}.pkl'
        result_np = result.cpu().numpy()
        with open(file_name, 'wb') as file:
            pickle.dump(result_np, file)
        
        prev_power = current_power
        prev_power_binary = current_power_binary
        
        k_elapsed = time.time() - k_start_time
        print(f"  {k}阶矩阵幂计算完成，用时={k_elapsed:.2f}秒")
        
        if k % 3 == 0:
            torch.cuda.empty_cache()
    
    total_elapsed = time.time() - total_start_time
    print(f"\n所有矩阵幂计算完成！总用时: {total_elapsed:.2f}秒")


def matrix_powers_compute(adj_list, n, matrix_prefix, dataset_name='dataset'):
    """
    为数据集计算1-10阶矩阵幂
    
    自动选择最优计算方法（GPU稠密或CPU稀疏）
    
    Args:
        adj_list: 邻接列表字典
        n: 节点数
        matrix_prefix: 输出文件前缀
        dataset_name: 数据集名称（用于显示）
    """
    print(f"\n{'='*50}")
    print(f"开始计算{dataset_name}矩阵幂")
    print(f"{'='*50}")
    print(f"节点数: {n}")
    print(f"输出前缀: {matrix_prefix}")
    print(f"数据集名称: {dataset_name}")
    
    # 检查是否有GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    has_cuda = torch.cuda.is_available()
    print(f"使用设备: {device}")
    
    # 估算内存需求
    dense_memory = (n * n * 4) / 1e9  # float32, GB
    
    if has_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU内存: {gpu_memory:.2f} GB")
        print(f"估计稠密矩阵内存需求: {dense_memory:.2f} GB")
        
        if dense_memory > gpu_memory * 0.01:
            print(f"\n警告: 稠密矩阵({dense_memory:.1f}GB)超过可用GPU内存({gpu_memory:.1f}GB)的30%")
            print("   使用稀疏矩阵+CPU计算模式...")
            use_sparse_cpu = True
        else:
            use_sparse_cpu = False
    else:
        print("未检测到GPU，使用CPU稀疏矩阵计算...")
        use_sparse_cpu = True
    
    # 动态计算合适的block_size
    if n % 1024 == 0:
        block_size = 1024
    elif n % 512 == 0:
        block_size = 512
    elif n % 256 == 0:
        block_size = 256
    elif n % 128 == 0:
        block_size = 128
    else:
        block_size = math.gcd(n, 1024)
    
    print(f"使用block_size: {block_size}")
    
    if use_sparse_cpu:
        # 使用稀疏矩阵方法
        print("\n使用稀疏矩阵方法计算...")
        adj_sparse = create_adjacency_matrix_sparse(adj_list, n)
        compute_sparse_matrix_powers(adj_sparse, n, matrix_prefix, max_k=10)
    else:
        # 使用GPU稠密矩阵方法
        print("\n使用GPU稠密矩阵方法计算...")
        adj_matrix = create_adjacency_matrix(adj_list, n)
        compute_dense_matrix_powers_gpu(adj_matrix, n, block_size, device, matrix_prefix)


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_output_files(DATADIR, dataset_name='dgraphfin'):
    """
    检查并显示已生成的矩阵幂文件
    
    Args:
        DATADIR: 数据目录路径
        dataset_name: 数据集名称
    """
    print(f"\n{'='*50}")
    print("检查生成的矩阵幂文件")
    print(f"{'='*50}")
    
    matrix_files = []
    for k in range(1, 11):
        # 检查.pkl文件
        pkl_file = os.path.join(DATADIR, f"{dataset_name}_adj_power_matrix{k}.pkl")
        npz_file = os.path.join(DATADIR, f"{dataset_name}_adj_power_matrix{k}.npz")
        
        if os.path.exists(pkl_file):
            file_size = os.path.getsize(pkl_file) / (1024**3)
            matrix_files.append(f"  {dataset_name}_adj_power_matrix{k}.pkl - {file_size:.3f} GB")
        elif os.path.exists(npz_file):
            file_size = os.path.getsize(npz_file) / (1024**2)
            matrix_files.append(f"  {dataset_name}_adj_power_matrix{k}.npz - {file_size:.2f} MB")
    
    if matrix_files:
        print("已生成的文件:")
        for file_info in matrix_files:
            print(file_info)
    else:
        print("未找到生成的矩阵幂文件")


def main(dataset_name='dgraphfin', data_dir='../data'):
    """主函数
    
    Args:
        dataset_name: 数据集名称（用于生成输出文件名）
        data_dir: 数据目录路径
    """
    
    print("=" * 60)
    print(f"{dataset_name}数据集矩阵幂计算脚本")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 路径设置
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATADIR = os.path.join(SCRIPT_DIR, data_dir)
    
    # 确保输出目录存在
    os.makedirs(DATADIR, exist_ok=True)
    
    # 邻接列表路径（使用动态数据集名称）
    adjlist_filename = f"{dataset_name}_homo_adjlists.pickle"
    adjlist_path = os.path.join(DATADIR, adjlist_filename)
    
    # 检查邻接列表是否存在
    print(f"\n检查邻接列表文件: {adjlist_path}")
    if not os.path.exists(adjlist_path):
        print(f"错误: 邻接列表文件不存在")
        print(f"请先运行dataset_processor.py生成邻接列表")
        print(f"预期文件路径: {adjlist_path}")
        return
    
    # 加载邻接列表
    print(f"加载邻接列表...")
    with open(adjlist_path, 'rb') as file:
        adj_list = pickle.load(file)
    
    n = len(adj_list)
    print(f"节点数量: {n}")
    
    # 统计边数
    total_edges = sum(len(neighbors) for neighbors in adj_list.values())
    print(f"总边数: {total_edges}")
    
    # 设置输出文件前缀（使用动态数据集名称）
    matrix_prefix = os.path.join(DATADIR, f"{dataset_name}_adj_power_matrix")
    
    # 开始计算矩阵幂
    print("\n" + "-" * 40)
    print("开始计算矩阵幂")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        matrix_powers_compute(adj_list, n, matrix_prefix, dataset_name)
    except torch.cuda.OutOfMemoryError:
        print("\nGPU内存不足，切换到CPU稀疏矩阵模式...")
        adj_sparse = create_adjacency_matrix_sparse(adj_list, n)
        compute_sparse_matrix_powers(adj_sparse, n, matrix_prefix, max_k=10)
    except Exception as e:
        print(f"\n计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 检查输出文件
    check_output_files(DATADIR, dataset_name)
    
    print("\n" + "=" * 60)
    print(f"{dataset_name}矩阵幂计算完成!")
    print("=" * 60)
    
    print(f"\n总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"输出文件保存在: {DATADIR}")
    print(f"\n生成的文件格式:")
    print(f"  - {dataset_name}_adj_power_matrix1.pkl/npz: 1阶邻接矩阵")
    print(f"  - {dataset_name}_adj_power_matrix2.pkl/npz: 2阶邻接矩阵幂")
    print(f"  - ...")
    print(f"  - {dataset_name}_adj_power_matrix10.pkl/npz: 10阶邻接矩阵幂")


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description='数据集矩阵幂计算脚本 - 支持灵活的数据集名称参数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 处理 dgraphfin 数据集（默认）
    python get_matrix.py
    
    # 处理其他数据集
    python get_matrix.py --dataset mydataset --data_dir ../data
    
    # 完整参数示例
    python get_matrix.py --dataset amazon --data_dir ../../processed_data
        """
    )
    
    parser.add_argument('--dataset', type=str, default='dgraphfin', 
                        help='数据集名称 (默认: dgraphfin)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录路径 (默认: ../data)')
    
    args = parser.parse_args()
    
    # 调用主函数，传入所有参数
    main(
        dataset_name=args.dataset,
        data_dir=args.data_dir
    )

