# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import dgl
import time
import json
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InferenceData:
    """推理数据加载 - 加载推理所需的数据"""
    def __init__(self, data_path, device='cpu'):
        print("从 {} 加载推理数据...".format(data_path))
        
        if not os.path.exists(data_path):
            raise FileNotFoundError("数据文件不存在: {}".format(data_path))
        
        # 加载数据
        data = np.load(data_path, allow_pickle=True)
        
        self.x = torch.FloatTensor(data['x'])
        self.y = torch.LongTensor(data['y'])
        self.edge_index = torch.LongTensor(data['edge_index'])
        
        # 加载test_mask（索引数组）
        test_indices = torch.LongTensor(data['test_mask'])
        
        self.num_nodes = self.x.shape[0]
        
        # 将索引转换为布尔掩码
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask[test_indices] = True
        
        # 构建图
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()
        self.g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        
        self.g.ndata['feature'] = self.x
        self.g.ndata['label'] = self.y
        
        self.device = torch.device(device)
        
        print("推理数据加载成功:")
        print("  节点数: {:,}, 特征维度: {}".format(self.num_nodes, self.x.shape[1]))
        print("  边数: {:,}".format(self.g.num_edges()))
        print("  测试集大小: {:,}".format(self.test_mask.sum().item()))
        
        # 统计类别分布
        unique, counts = torch.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            percentage = cnt/self.num_nodes*100
            print("  类别 {}: {:,} 个节点 ({:.2f}%)".format(cls, cnt, percentage))
        
    def to(self, device):
        self.device = torch.device(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.test_mask = self.test_mask.to(device)
        self.g = self.g.to(device)
        return self

def filter_labels_01(data):
    """只保留类别0和1的节点 - 推理版本"""
    print("\n过滤数据，只保留类别0和1的节点...")
    
    print("原始总节点数: {:,}".format(data.num_nodes))
    
    device = data.device
    data_cpu = data.to('cpu')
    
    mask_01 = (data_cpu.y == 0) | (data_cpu.y == 1)
    print("类别0和1的节点数: {:,}".format(mask_01.sum().item()))
    
    if mask_01.sum().item() == 0:
        print("错误: 没有找到类别0或1的节点！")
        return data
    
    original_num_nodes = data_cpu.num_nodes
    
    data_cpu.x = data_cpu.x[mask_01]
    data_cpu.y = data_cpu.y[mask_01]
    data_cpu.test_mask = data_cpu.test_mask[mask_01]
    
    old_to_new = torch.full((original_num_nodes,), -1, dtype=torch.long)
    new_indices = torch.arange(mask_01.sum().item())
    old_to_new[mask_01] = new_indices
    
    edge_mask = mask_01[data_cpu.edge_index[0]] & mask_01[data_cpu.edge_index[1]]
    filtered_edges = data_cpu.edge_index[:, edge_mask]
    
    if filtered_edges.shape[1] > 0:
        data_cpu.edge_index = torch.stack([
            old_to_new[filtered_edges[0]],
            old_to_new[filtered_edges[1]]
        ])
    else:
        data_cpu.edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    data_cpu.num_nodes = mask_01.sum().item()
    
    if data_cpu.edge_index.shape[1] > 0:
        src = data_cpu.edge_index[0].numpy()
        dst = data_cpu.edge_index[1].numpy()
        data_cpu.g = dgl.graph((src, dst), num_nodes=data_cpu.num_nodes)
    else:
        data_cpu.g = dgl.graph(([], []), num_nodes=data_cpu.num_nodes)
    
    data_cpu.g.ndata['feature'] = data_cpu.x
    data_cpu.g.ndata['label'] = data_cpu.y
    
    print("\n过滤后数据统计:")
    print("  节点数: {:,}".format(data_cpu.num_nodes))
    print("  边数: {:,}".format(data_cpu.g.num_edges()))
    print("  测试集大小: {:,}".format(data_cpu.test_mask.sum().item()))
    
    if data_cpu.num_nodes > 0:
        unique, counts = torch.unique(data_cpu.y, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            percentage = cnt/data_cpu.num_nodes*100
            print("  类别 {}: {:,} 个节点 ({:.2f}%)".format(cls, cnt, percentage))
    else:
        print("警告: 过滤后没有节点剩余！")
    
    return data_cpu.to(device)

def batch_inference(model, data, mask, batch_size=512, use_amp=False):
    """
    批处理推理并测量时间和显存峰值（只推理mask指定的节点）
    返回: (输出, 推理时间, 峰值显存, 推理的节点索引)
    """
    model.eval()
    
    # 检查mask是否有节点
    if mask is None or mask.sum().item() == 0:
        print("警告: 推理mask为空")
        return None, 0.0, 0.0, None
    
    # 清空GPU缓存并记录初始显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    masked_indices = torch.where(mask)[0]
    num_masked_nodes = masked_indices.shape[0]
    all_outputs = []
    
    # 记录开始时间（从这里开始计算推理时间）
    start_time = time.time()
    
    with torch.no_grad():
        for start_idx in range(0, num_masked_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_masked_nodes)
            batch_indices = masked_indices[start_idx:end_idx]
            
            # 前向传播（整个图，然后取需要的节点）
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    batch_output = model(data.x, data.edge_index)
            else:
                batch_output = model(data.x, data.edge_index)
            
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            
            all_outputs.append(batch_output[batch_indices])
    
    # 记录结束时间
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 获取峰值显存
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0.0
    
    # 合并所有输出
    if all_outputs:
        final_output = torch.cat(all_outputs, dim=0)
    else:
        final_output = None
    
    return final_output, inference_time, peak_memory, masked_indices

def load_model(model_path, device, in_channels=None):
    """
    加载模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    args = checkpoint.get('args', {})
    
    print(f"检查点中的参数: {args}")
    
    # 从检查点获取参数
    hid_channels = args.get('hidden_channels', 32)
    num_class = 2  # 固定为2分类
    K = args.get('M', 8)
    filter_num = args.get('K', 2)
    
    # 使用提供的输入特征维度
    if in_channels is None:
        in_channels = args.get('in_channels', hid_channels)
    
    print(f"创建模型参数: in_channels={in_channels}, hid_channels={hid_channels}, K={K}, filter_num={filter_num}")
    
    # 创建模型
    from model import AMNet
    model = AMNet(
        in_channels=in_channels,
        hid_channels=hid_channels,
        num_class=num_class,
        K=K,
        filter_num=filter_num
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"从 {model_path} 加载模型")
    print(f"模型信息: epoch={checkpoint.get('epoch', 'unknown')}, "
          f"AUC-PR={checkpoint.get('auc_pr_val', 0.0):.4f}, "
          f"AUC-ROC={checkpoint.get('auc_roc_val', 0.0):.4f}")
    
    return model, checkpoint

def main(args):
    print("="*60)
    print("AMNet 图异常检测推理")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    print(f"数据文件: {args.data_path}")
    print(f"模型文件: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"分类阈值: {args.threshold}")
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")
    
    # 1. 加载数据（不计时）
    data_load_start = time.time()
    print("\n[阶段1] 从磁盘加载数据...")
    try:
        data = InferenceData(args.data_path, device='cpu')
        data = filter_labels_01(data)
        
        if data.num_nodes == 0:
            print("错误: 过滤后数据为空！")
            return
        
        print(f"数据加载完成:")
        print(f"  总节点数: {data.num_nodes:,}")
        print(f"  测试集节点数: {data.test_mask.sum().item():,}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    data_load_end = time.time()
    print(f"数据加载时间: {data_load_end - data_load_start:.4f} 秒")
    
    # 2. 加载模型（不计时）
    print("\n[阶段2] 加载模型...")
    # 3. 开始推理计时（从这里开始计算推理时间）
    print("\n[阶段3] 开始推理（包含数据传输、预处理和推理）...")
    # 开始计时（包含数据传送到GPU的时间）
    inference_start = time.time()    
    
    
    try:
        model, checkpoint = load_model(args.model_path, device, in_channels=data.x.shape[1])
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    

    
    # 清空GPU缓存，准备测量
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    

    
    # 3.1 数据传送到GPU（包含在推理时间内）
    data = data.to(device)
    
    # 3.2 执行推理（只推理test_mask指定的节点）
    outputs, inference_time, peak_memory, indices = batch_inference(
        model, data, data.test_mask, batch_size=args.batch_size, use_amp=args.use_amp
    )
    
    inference_end = time.time()
    total_inference_time = inference_end - inference_start
    
    if outputs is None:
        print("错误: 推理失败，没有输出结果")
        return
    
    # 4. 推理后处理（不计时）
    print("\n[阶段4] 推理后处理（不计时）...")
    
    # 获取预测概率
    probs = F.softmax(outputs, dim=1)
    y_pred_prob = probs[:, 1].cpu().numpy()  # 异常类别的概率
    y_true = data.y[indices].cpu().numpy()
    
    # 5. 输出结果
    print(f"\n=== 推理结果 ===")
    print(f"测试集节点数: {len(indices):,}")
    print(f"纯推理时间（模型前向传播）: {inference_time:.4f} 秒")
    print(f"总推理时间（包含数据传输）: {total_inference_time:.4f} 秒")
    print(f"峰值显存: {peak_memory:.2f} MB")
    print(f"平均每个节点推理时间: {inference_time/len(indices)*1000:.4f} 毫秒")
    
    # 计算评估指标
    print(f"\n评估指标 (阈值={args.threshold}):")
    
    if len(np.unique(y_true)) < 2:
        print("警告: 数据中只有一个类别，无法计算AUC")
        auc_roc = 0.0
        auc_pr = 0.0
    else:
        auc_roc = roc_auc_score(y_true, y_pred_prob)
        auc_pr = average_precision_score(y_true, y_pred_prob)
        print(f"  AUC-ROC: {auc_roc:.5f}")
        print(f"  AUC-PR:  {auc_pr:.5f}")
        
        # 根据阈值计算分类结果
        y_pred = (y_pred_prob >= args.threshold).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        if cm is not None and cm.size > 0 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n混淆矩阵:")
            print(f"          预测正常    预测异常")
            print(f"真实正常    {tn:>8}    {fp:>8}")
            print(f"真实异常    {fn:>8}    {tp:>8}")
            
            # 计算具体指标
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n详细指标:")
            print(f"  准确率: {accuracy:.5f}")
            print(f"  精确率: {precision:.5f}")
            print(f"  召回率: {recall:.5f}")
            print(f"  F1分数: {f1:.5f}")
    
    # 保存预测结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    predictions_file = os.path.join(args.output_dir, f"predictions_{timestamp}.npz")
    
    predictions = {
        'node_indices': indices.cpu().numpy(),
        'predictions': outputs.cpu().numpy(),
        'probabilities': probs.cpu().numpy(),
        'labels': data.y[indices].cpu().numpy(),
        'threshold': args.threshold
    }
    
    np.savez(predictions_file, **predictions)
    print(f"\n预测结果已保存到: {predictions_file}")
    
    # 保存评估结果
    results = {
        'data_path': args.data_path,
        'model_path': args.model_path,
        'inference_time': float(inference_time),
        'total_inference_time': float(total_inference_time),
        'peak_memory': float(peak_memory),
        'num_test_nodes': len(indices),
        'threshold': args.threshold,
        'metrics': {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
        }
    }
    
    if 'accuracy' in locals():
        results['metrics']['accuracy'] = float(accuracy)
        results['metrics']['precision'] = float(precision)
        results['metrics']['recall'] = float(recall)
        results['metrics']['f1'] = float(f1)
    
    results_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存到: {results_file}")
    
    # 打印预测样本
    print(f"\n预测样本 (前10个测试节点):")
    print(f"{'节点ID':<10} {'真实标签':<10} {'异常概率':<12} {'预测标签':<10}")
    print("-" * 45)
    for i in range(min(10, len(indices))):
        node_id = indices[i].item()
        true_label = y_true[i]
        anomaly_prob = y_pred_prob[i]
        pred_label = 1 if anomaly_prob >= args.threshold else 0
        print(f"{node_id:<10} {true_label:<10} {anomaly_prob:<12.6f} {pred_label:<10}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMNet 图异常检测推理')
    
    parser.add_argument('--data_path', type=str, required=True, help='.npz数据文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=512, help='推理批大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度推理')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    main(args)
