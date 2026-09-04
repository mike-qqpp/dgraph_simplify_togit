import argparse

def get_args():
    parser = argparse.ArgumentParser(description='T-Finance特征工程')
    
    # 文件路径参数
    parser.add_argument('--input_path', type=str, default='./data/tfinance.npz',
                       help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, default='./output/features.pkl',
                       help='输出特征文件路径')
    
    # 邻居采样参数
    parser.add_argument('--max_neighbors_1hop', type=int, default=50,
                       help='1-hop邻居最大采样数')
    parser.add_argument('--max_neighbors_2hop', type=int, default=500,
                       help='2-hop邻居最大采样数')
    
    # 特征选择参数
    parser.add_argument('--use_base_features', type=bool, default=True,
                       help='是否使用基础特征')
    parser.add_argument('--use_structural_features', type=bool, default=True,
                       help='是否使用结构特征')
    parser.add_argument('--use_neighbor_features', type=bool, default=True,
                       help='是否使用邻居特征')
    parser.add_argument('--use_spectral_features', type=bool, default=False,
                       help='是否使用谱特征（计算量大）')
    parser.add_argument('--use_embedding_features', type=bool, default=True,
                       help='是否使用图嵌入特征')
    
    # 图嵌入参数
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='图嵌入维度')
    parser.add_argument('--walk_length', type=int, default=40,
                       help='随机游走长度')
    parser.add_argument('--num_walks', type=int, default=10,
                       help='每个节点随机游走次数')
    
    # 计算参数
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行工作数')
    
    return parser.parse_args()
