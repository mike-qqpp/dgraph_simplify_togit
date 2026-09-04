# The file for test inference

import torch
import torch.nn.functional as F
import dgl
import sklearn.metrics as skmetrics
import os
import sys
import time
import argparse
import warnings
import numpy as np
from models.base import get_thres
from models.DSGAD import DSGAD

# 添加 datasets 目录到 Python 路径
datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
if datasets_path not in sys.path:
    sys.path.insert(0, datasets_path)

import datasets


def setup_device(args):
    """设置设备"""
    if torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device('cuda')
        print(f'使用GPU设备: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('使用CPU设备')
    return device


def load_model(model_path, in_nodes, in_feats, model_config, device):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    h_feats = model_config.get('h_feats', 32)
    d = model_config.get('d', 2)
    mix_beta = model_config.get('mix_beta', 2)
    use_reduced_memory = model_config.get('use_reduced_memory', True)
    
    model = DSGAD(
        in_nodes=in_nodes,
        in_feats=in_feats,
        h_feats=h_feats,
        d=d,
        mix_beta=mix_beta,
        use_reduced_memory=use_reduced_memory
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def batch_inference(model, g, features, labels, test_mask, threshold, batch_size, device):
    """批处理推理并测量时间和显存（优化版本）"""
    model.eval()
    
    # 清空GPU缓存并记录初始显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # 获取测试索引
    test_idx = torch.where(test_mask)[0]
    
    # 直接分批处理，避免使用DataLoader
    all_test_probs = []
    all_test_labels = []
    
    inference_start = time.time()
    
    with torch.no_grad():
        for start in range(0, len(test_idx), batch_size):
            end = min(start + batch_size, len(test_idx))
            batch_idx = test_idx[start:end].to(device)
            
            # 批次前向传播
            out = model(g, features, node_indices=batch_idx)
            
            # 收集概率和标签
            batch_probs = out.softmax(1)[:, 1].cpu()
            batch_labels = labels[batch_idx].cpu()
            
            all_test_probs.append(batch_probs)
            all_test_labels.append(batch_labels)
    
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # 获取峰值显存
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peak_memory_usage = peak_memory - initial_memory
    else:
        peak_memory = 0.0
        peak_memory_usage = 0.0
    
    # 合并所有批次的结果
    test_probs = torch.cat(all_test_probs) if all_test_probs else torch.tensor([])
    test_labels = torch.cat(all_test_labels) if all_test_labels else torch.tensor([])
    
    # 使用阈值进行预测
    if len(test_probs) > 0:
        test_y_pred = torch.zeros_like(test_probs)
        test_y_pred[test_probs >= threshold] = 1
    else:
        test_y_pred = torch.tensor([])
    
    return test_probs, test_labels, test_y_pred, inference_time, peak_memory_usage


def evaluate_test_set(model, g, features, labels, test_mask, threshold, batch_size, device):
    """在测试集上进行评估（优化版本）"""
    print(f"\n测试集推理 - 设备: {device}")
    
    if test_mask.sum().item() == 0:
        print("警告: 测试集掩码为空")
        return {
            'test_auc': 0.0,
            'test_ap': 0.0,
            'test_recall': 0.0,
            'test_precision': 0.0,
            'test_f1_macro': 0.0,
            'test_f1_micro': 0.0,
            'test_accuracy': 0.0,
            'inference_time': 0.0,
            'test_probs': torch.tensor([]),
            'test_labels': torch.tensor([]),
            'test_y_pred': torch.tensor([]),
            'peak_memory': 0.0
        }
    
    # 批处理推理
    test_probs, test_labels, test_y_pred, inference_time, peak_memory = batch_inference(
        model, g, features, labels, test_mask, threshold, batch_size, device
    )
    
    # 检查是否有结果
    if len(test_probs) == 0 or len(test_labels) == 0:
        print("警告: 没有推理结果")
        return {
            'test_auc': 0.0,
            'test_ap': 0.0,
            'test_recall': 0.0,
            'test_precision': 0.0,
            'test_f1_macro': 0.0,
            'test_f1_micro': 0.0,
            'test_accuracy': 0.0,
            'inference_time': inference_time,
            'test_probs': test_probs,
            'test_labels': test_labels,
            'test_y_pred': test_y_pred,
            'peak_memory': peak_memory
        }
    
    # 计算测试集指标
    try:
        test_auc = skmetrics.roc_auc_score(test_labels, test_probs)
        test_ap = skmetrics.average_precision_score(test_labels, test_probs)
        test_recall = skmetrics.recall_score(test_labels, test_y_pred)
        test_precision = skmetrics.precision_score(test_labels, test_y_pred)
        test_f1_macro = skmetrics.f1_score(test_labels, test_y_pred, average='macro')
        test_f1_micro = skmetrics.f1_score(test_labels, test_y_pred, average='micro')
        test_accuracy = skmetrics.accuracy_score(test_labels, test_y_pred)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        test_auc = test_ap = test_recall = test_precision = test_f1_macro = test_f1_micro = test_accuracy = 0.0
    
    return {
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_recall': test_recall,
        'test_precision': test_precision,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_accuracy': test_accuracy,
        'inference_time': inference_time,
        'test_probs': test_probs,
        'test_labels': test_labels,
        'test_y_pred': test_y_pred,
        'peak_memory': peak_memory
    }


def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='DSGAD Test Inference (Optimized)')
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device: cuda or cpu')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for inference')
    parser.add_argument('--save_predictions', action='store_true', help='save predictions to file')
    parser.add_argument('--predictions_path', type=str, default='./predictions.npz', help='path to save predictions')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件不存在：{args.model_path}")
        exit(1)
    
    print('=' * 60)
    print('DSGAD 测试推理（优化版本）')
    print('=' * 60)
    print(f'模型路径：{args.model_path}')
    print(f'数据集：{args.dataset}')
    print(f'设备：{args.device}')
    print(f'批次大小：{args.batch_size}')
    print()
    
    # 开始计时：加载数据到推理完成的总时间
    total_time_start = time.time()
    
    # 设置设备
    device = setup_device(args)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告：指定了GPU但CUDA不可用，将使用CPU")
        args.device = 'cpu'
        device = torch.device('cpu')
    
    # 开始数据加载计时
    data_load_start = time.time()
    
    # Load the graph dataset
    print("\n加载数据集...")
    dataset = eval(f'datasets.{args.dataset}')()
    g: dgl.DGLGraph = dataset.g
    
    # 打印图的详细信息
    print(f"数据集：{args.dataset}")
    print(f"图结构：{g}")
    print(f"节点数：{g.num_nodes():,}")
    print(f"边数：{g.num_edges():,}")
    print(f"节点特征维度：{g.ndata['feature'].shape[1]}")
    
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    print(f'\n训练集/验证集/测试集样本数：{train_mask.sum().item():,} / {val_mask.sum().item():,} / {test_mask.sum().item():,}')
    
    features = g.ndata['feature']
    labels = g.ndata['label']
    
    # 移动数据到设备
    print(f"\n移动数据到 {device}...")
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)
    
    # 结束数据加载计时
    data_load_end = time.time()
    data_load_time = data_load_end - data_load_start
    print(f'数据加载时间：{data_load_time:.4f}s')
    
    # 加载模型
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint.get('config', {})
    
    model, loaded_checkpoint = load_model(
        args.model_path,
        in_nodes=features.shape[0],
        in_feats=features.shape[1],
        model_config=model_config,
        device=device
    )
    
    best_epoch = loaded_checkpoint.get('best_epoch', 0)
    best_val_auc = loaded_checkpoint.get('val_auc', 0)
    best_val_ap = loaded_checkpoint.get('val_ap', 0)
    threshold = loaded_checkpoint.get('threshold', 0.5)
    
    print(f"模型加载成功！")
    print(f"  最佳验证Epoch：{best_epoch}")
    print(f"  最佳验证AUC：{best_val_auc:.5f}")
    print(f"  最佳验证AP：{best_val_ap:.5f}")
    print(f"  阈值：{threshold:.5f}")
    
    # 显示GPU内存使用情况（如果使用GPU）
    if device.type == 'cuda':
        print(f"\nGPU内存使用情况:")
        print(f"  已分配: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        print(f"  缓存: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")
    
    # 在测试集上评估
    print("\n" + "="*60)
    print("测试集推理")
    print("="*60)
    
    results = evaluate_test_set(
        model, g, features, labels, test_mask, 
        threshold=threshold,
        batch_size=args.batch_size,
        device=device
    )
    
    # 结束计时
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    
    # 计算数据加载+推理总时长
    data_inference_total_time = data_load_time + results['inference_time']
    
    # 输出结果
    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)
    
    # 打印指标表格
    print('指标            | 值')
    print('-' * 30)
    print(f'AUC-ROC        | {results["test_auc"]:.5f}')
    print(f'AUC-PR         | {results["test_ap"]:.5f}')
    print(f'Recall         | {results["test_recall"]:.5f}')
    print(f'Precision      | {results["test_precision"]:.5f}')
    print(f'F1-Macro       | {results["test_f1_macro"]:.5f}')
    print(f'F1-Micro       | {results["test_f1_micro"]:.5f}')
    print(f'Accuracy       | {results["test_accuracy"]:.5f}')
    
    print(f'\n推理时间        | {results["inference_time"]:.4f}s')
    print(f'数据加载时间    | {data_load_time:.4f}s')
    print(f'数据加载+推理   | {data_inference_total_time:.4f}s')
    print(f'总时间          | {total_time:.4f}s')
    print(f'阈值           | {threshold:.5f}（来自验证集）')
    
    if device.type == 'cuda':
        print(f'峰值内存        | {results["peak_memory"]:.2f} MB')
    
    print("="*60)
    
    # 结果格式
    result = (f'REC {results["test_recall"]*100:.2f} PRE {results["test_precision"]*100:.2f} '
              f'MF1 {results["test_f1_macro"]*100:.2f} AUC {results["test_auc"]*100:.2f}')
    print(f'\n结果格式：{result}')
    
    # 保存结果到文件
    with open('test_result.txt', 'a+') as f:
        f.write(f'{result}\n')
    
    # 保存预测结果
    if args.save_predictions:
        np.savez(args.predictions_path,
                 predictions=results['test_y_pred'].numpy() if len(results['test_y_pred']) > 0 else np.array([]),
                 probabilities=results['test_probs'].numpy() if len(results['test_probs']) > 0 else np.array([]),
                 labels=results['test_labels'].numpy() if len(results['test_labels']) > 0 else np.array([]))
        print(f"\n预测结果已保存至：{args.predictions_path}")
    
    return results


if __name__ == '__main__':
    main()
