import numpy as np
import dgl, torch, os
import argparse
from dgl.data.utils import save_graphs

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='将npz数据转换为DGL格式')
    parser.add_argument('--dataset', type=str, default='tfinance', 
                       help='数据集名称（如：tfinance）')
    parser.add_argument('--npz_path', type=str, default='../data/', 
                       help='npz文件所在目录路径，默认为../data/')
    parser.add_argument('--output_dir', type=str, default='./datasets/', 
                       help='输出目录路径，默认为./datasets/')
    
    args = parser.parse_args()
    
    # 构建完整的文件路径
    npz_file = os.path.join(args.npz_path, f'{args.dataset}.npz')
    output_dir = os.path.join(args.output_dir, args.dataset)
    bin_file = os.path.join(output_dir, args.dataset)
    index_file = os.path.join(output_dir, f'{args.dataset}_index.txt')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在处理数据集: {args.dataset}")
    print(f"输入文件: {npz_file}")
    print(f"输出目录: {output_dir}")
    
    # 1. 读 npz
    if not os.path.exists(npz_file):
        print(f"错误: 文件不存在 - {npz_file}")
        return
    
    try:
        npz = np.load(npz_file)
        print(f"成功加载 {args.dataset}.npz 文件")
    except Exception as e:
        print(f"加载npz文件时出错: {e}")
        return
    
    # 检查必要的字段
    required_fields = ['x', 'y', 'edge_index']
    for field in required_fields:
        if field not in npz:
            print(f"错误: npz文件中缺少必要字段 '{field}'")
            return
    
    x = torch.from_numpy(npz['x']).float()
    y = torch.from_numpy(npz['y']).long()
    edge_idx = torch.from_numpy(npz['edge_index'].T).long()   # dgl 用 (src, dst)
    
    print(f"数据统计:")
    print(f"  节点数: {x.shape[0]}")
    print(f"  特征维度: {x.shape[1]}")
    print(f"  边数: {edge_idx.shape[1]}")
    print(f"  类别数: {len(torch.unique(y))}")
    
    # 2. 建图
    g = dgl.graph((edge_idx[0], edge_idx[1]), num_nodes=x.shape[0])
    g.ndata['feature'] = x
    g.ndata['label'] = y
    
    # 可选字段
    if 'edge_type' in npz:
        g.edata['type'] = torch.from_numpy(npz['edge_type']).long()
        print(f"  边类型字段已添加")
    
    if 'edge_timestamp' in npz:
        g.edata['timestamp'] = torch.from_numpy(npz['edge_timestamp']).long()
        print(f"  边时间戳字段已添加")
    
    # 检查是否有掩码字段
    mask_fields = ['train_mask', 'valid_mask', 'test_mask']
    for mask_field in mask_fields:
        if mask_field in npz:
            g.ndata[mask_field] = torch.from_numpy(npz[mask_field]).bool()
            print(f"  {mask_field}字段已添加")
    
    # 3. 保存 DGL bin
    save_graphs(bin_file, [g])
    print(f"\nDGL图文件已保存: {bin_file}.bin")
    print(f'节点数={g.num_nodes()}, 边数={g.num_edges()}')
    
    # 4. 生成索引文件
    np.savetxt(index_file, np.arange(g.num_nodes()), fmt='%d')
    print(f'索引文件已生成: {index_file}')
    print(f'共 {g.num_nodes()} 个节点索引')
    
    # 保存节点特征维度信息（可选）
    dim_file = os.path.join(output_dir, f'{args.dataset}_dim.txt')
    with open(dim_file, 'w') as f:
        f.write(f"num_nodes: {g.num_nodes()}\n")
        f.write(f"num_edges: {g.num_edges()}\n")
        f.write(f"feature_dim: {x.shape[1]}\n")
        f.write(f"num_classes: {len(torch.unique(y))}\n")
    print(f'维度信息文件已生成: {dim_file}')
    
    print(f"\n{args.dataset} 数据集转换完成！")

if __name__ == '__main__':
    main()
