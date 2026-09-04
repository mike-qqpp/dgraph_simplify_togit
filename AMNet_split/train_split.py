# -*- coding: utf-8 -*-
import argparse
import torch
import torch.optim as optim
from model import AMNet
from copy import deepcopy
import pickle
import numpy as np
import os
import scipy.sparse as sp
import dgl
import time
import gc
from torch.cuda.amp import autocast
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NpzGraphData:
    """直接从npz文件加载图数据"""
    def __init__(self, data_path, device='cpu'):
        print("从 {} 加载数据...".format(data_path))
        
        if not os.path.exists(data_path):
            raise FileNotFoundError("数据文件不存在: {}".format(data_path))
        
        data = np.load(data_path, allow_pickle=True)
        
        self.x = torch.FloatTensor(data['x'])
        self.y = torch.LongTensor(data['y'])
        self.edge_index = torch.LongTensor(data['edge_index'])
        
        self.train_mask = torch.BoolTensor(data['train_mask'])
        self.val_mask = torch.BoolTensor(data['valid_mask'])
        self.test_mask = torch.BoolTensor(data['test_mask'])
        
        self.num_nodes = self.x.shape[0]
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()
        self.g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        
        self.g.ndata['feature'] = self.x
        self.g.ndata['label'] = self.y
        
        self.device = torch.device(device)
        
        print("数据加载成功:")
        print("  节点数: {:,}, 特征维度: {}".format(self.num_nodes, self.x.shape[1]))
        print("  边数: {:,}".format(self.g.num_edges()))
        print("  训练集大小: {:,}".format(self.train_mask.sum().item()))
        print("  验证集大小: {:,}".format(self.val_mask.sum().item()))
        print("  测试集大小: {:,}".format(self.test_mask.sum().item()))
        
        unique, counts = torch.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            percentage = cnt/self.num_nodes*100
            print("  类别 {}: {:,} 个节点 ({:.2f}%)".format(cls, cnt, percentage))
        
    def to(self, device):
        self.device = torch.device(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.g = self.g.to(device)
        return self

def filter_labels_01(data):
    """只保留类别0和1的节点"""
    print("\n过滤数据，只保留类别0和1的节点...")
    
    print("原始总节点数: {:,}".format(data.num_nodes))
    
    device = data.device
    data_cpu = data.to('cpu')
    
    print("检查掩码尺寸:")
    print("  data.y 尺寸: {}".format(data_cpu.y.shape))
    print("  data.train_mask 尺寸: {}".format(data_cpu.train_mask.shape))
    print("  data.val_mask 尺寸: {}".format(data_cpu.val_mask.shape))
    print("  data.test_mask 尺寸: {}".format(data_cpu.test_mask.shape))
    
    if data_cpu.train_mask.shape[0] != data_cpu.num_nodes:
        print("警告: 掩码尺寸与节点数不匹配！将重新创建随机掩码...")
        
        num_nodes = data_cpu.num_nodes
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        data_cpu.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data_cpu.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data_cpu.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data_cpu.train_mask[train_idx] = True
        data_cpu.val_mask[val_idx] = True
        data_cpu.test_mask[test_idx] = True
        
        print("重新创建的掩码尺寸:")
        print("  data.train_mask: {:,}".format(data_cpu.train_mask.sum().item()))
        print("  data.val_mask: {:,}".format(data_cpu.val_mask.sum().item()))
        print("  data.test_mask: {:,}".format(data_cpu.test_mask.sum().item()))
    
    mask_01 = (data_cpu.y == 0) | (data_cpu.y == 1)
    print("类别0和1的节点数: {:,}".format(mask_01.sum().item()))
    
    if mask_01.sum().item() == 0:
        print("错误: 没有找到类别0或1的节点！")
        return data
    
    original_num_nodes = data_cpu.num_nodes
    
    data_cpu.x = data_cpu.x[mask_01]
    data_cpu.y = data_cpu.y[mask_01]
    
    if data_cpu.train_mask.shape[0] == original_num_nodes:
        data_cpu.train_mask = data_cpu.train_mask[mask_01]
        data_cpu.val_mask = data_cpu.val_mask[mask_01]
        data_cpu.test_mask = data_cpu.test_mask[mask_01]
    else:
        print("注意: 掩码已经被过滤，保持原样")
    
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
    print("  训练集大小: {:,}".format(data_cpu.train_mask.sum().item()))
    print("  验证集大小: {:,}".format(data_cpu.val_mask.sum().item()))
    print("  测试集大小: {:,}".format(data_cpu.test_mask.sum().item()))
    
    if data_cpu.num_nodes > 0:
        unique, counts = torch.unique(data_cpu.y, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            percentage = cnt/data_cpu.num_nodes*100
            print("  类别 {}: {:,} 个节点 ({:.2f}%)".format(cls, cnt, percentage))
    else:
        print("警告: 过滤后没有节点剩余！")
    
    return data_cpu.to(device)

def train(data, model, criterion, optimizer, label, beta=.5):
    anomaly, normal = label
    idx_train = data.train_mask
    
    model.train()
    optimizer.zero_grad()
    
    result = model(
        data.x, 
        data.edge_index, 
        label=(data.train_mask & anomaly, data.train_mask & normal)
    )
    
    if isinstance(result, tuple):
        output = result[0]
        bias_loss = result[1] if len(result) > 1 else 0.0
    else:
        output = result
        bias_loss = 0.0
    
    loss_train = criterion(output[idx_train], data.y[idx_train]) + bias_loss * beta
    
    loss_train.backward()
    optimizer.step()
    
    return loss_train.item()

def evaluate_model(model, data, mask, batch_size=512):
    """评估模型 - 使用批处理"""
    model.eval()
    
    # 检查掩码是否有节点
    if mask is None or mask.sum().item() == 0:
        print("警告: 评估掩码为空")
        return 0.0, 0.0
    
    # 获取掩码中的节点索引
    masked_indices = torch.where(mask)[0]
    num_masked_nodes = masked_indices.shape[0]
    
    all_outputs = []
    
    # 批处理推理
    with torch.no_grad():
        for start_idx in range(0, num_masked_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_masked_nodes)
            batch_indices = masked_indices[start_idx:end_idx]
            
            # 获取当前批次的子图
            batch_output = model(data.x, data.edge_index)
            
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            
            all_outputs.append(batch_output[batch_indices])
    
    # 合并所有批次的输出
    if all_outputs:
        output = torch.cat(all_outputs, dim=0)
        y_true = data.y[masked_indices].cpu().numpy()
        
        # 计算预测概率
        y_pred_prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        if len(np.unique(y_true)) < 2:
            print("警告: 掩码中只有一个类别，无法计算AUC")
            return 0.0, 0.0
        
        try:
            auc_roc = roc_auc_score(y_true, y_pred_prob)
            auc_pr = average_precision_score(y_true, y_pred_prob)
        except Exception as e:
            print(f"计算指标时出错: {e}")
            return 0.0, 0.0
    else:
        print("警告: 没有输出结果")
        return 0.0, 0.0
    
    return auc_roc, auc_pr

def batch_inference(model, data, mask, batch_size=512, use_amp=False):
    """
    批处理推理并测量时间和显存峰值（只推理掩码指定的节点）
    Args:
        model: 训练好的模型
        data: 图数据
        mask: 需要推理的节点掩码
        batch_size: 批大小
        use_amp: 是否使用混合精度
    Returns:
        tuple: (总输出, 推理时间, 峰值显存)
    """
    model.eval()
    
    # 检查掩码是否有节点
    if mask is None or mask.sum().item() == 0:
        print("警告: 推理掩码为空")
        return None, 0.0, 0.0
    
    # 清空GPU缓存并记录初始显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    masked_indices = torch.where(mask)[0]
    num_masked_nodes = masked_indices.shape[0]
    all_outputs = []
    
    # 记录开始时间
    start_time = time.time()
    
    with torch.no_grad():
        for start_idx in range(0, num_masked_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_masked_nodes)
            batch_indices = masked_indices[start_idx:end_idx]
            
            # 前向传播（整个图，然后取需要的节点）
            if use_amp:
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
    
    return final_output, inference_time, peak_memory

def evaluate_with_batch_inference(model, data, mask, batch_size=512):
    """
    使用批处理推理进行评估
    """
    model.eval()
    
    if mask is None or mask.sum().item() == 0:
        print("警告: 评估掩码为空")
        return 0.0, 0.0, 0.0, 0.0
    
    # 进行批处理推理（只推理测试集节点）
    output, inference_time, peak_memory = batch_inference(
        model, data, mask, batch_size=batch_size
    )
    
    if output is None:
        return 0.0, 0.0, inference_time, peak_memory
    
    # 计算指标
    y_masked = data.y[mask].cpu().numpy()
    y_pred_prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    if len(np.unique(y_masked)) < 2:
        print("警告: 掩码中只有一个类别，无法计算AUC")
        return 0.0, 0.0, inference_time, peak_memory
    
    try:
        auc_roc = roc_auc_score(y_masked, y_pred_prob)
        auc_pr = average_precision_score(y_masked, y_pred_prob)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return 0.0, 0.0, inference_time, peak_memory
    
    return auc_roc, auc_pr, inference_time, peak_memory

def save_model(model, optimizer, args, epoch, auc_pr_val, auc_roc_val, save_path):
    """保存模型和训练状态"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auc_pr_val': auc_pr_val,
        'auc_roc_val': auc_roc_val,
        'args': vars(args)  # 保存所有参数
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")

def main(args, exp_num=0):
    print("\n实验 {}/{}".format(exp_num+1, args.exp_num))
    print("-" * 50)
    
    seed = args.seed + exp_num
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    data = NpzGraphData(args.data_path, device='cpu')
    data = filter_labels_01(data)
    
    if data.num_nodes == 0:
        print("错误: 过滤后数据为空！")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    data = data.to(device)
    
    in_channels = data.x.shape[1]
    hid_channels = args.hidden_channels
    num_class = 2
    K = args.M
    filter_num = args.K

    net = AMNet(
        in_channels=in_channels, 
        hid_channels=hid_channels, 
        num_class=num_class,
        K=K, 
        filter_num=filter_num
    )
    net.to(device)
    
    print("\n模型配置:")
    print("  输入特征: {}".format(in_channels))
    print("  隐藏维度: {}".format(hid_channels))
    print("  滤波器数量: {}".format(K))
    print("  滤波器阶数: {}".format(filter_num))
    print("  总参数量: {:,}".format(sum(p.numel() for p in net.parameters())))

    optimizer = optim.Adam([
        dict(params=net.filters.parameters(), lr=args.lr_f),
        dict(params=net.lin, lr=args.lr, weight_decay=args.weight_decay),
        dict(params=net.attn, lr=args.lr, weight_decay=args.weight_decay)]
    )
    
    print("\n优化器配置:")
    print("  滤波器学习率: {}".format(args.lr_f))
    print("  其他参数学习率: {}".format(args.lr))
    print("  权重衰减: {}".format(args.weight_decay))

    num_normal = (data.y == 0).sum().item()
    num_anomaly = (data.y == 1).sum().item()
    
    if args.use_class_weight:
        weight_normal = 1.0
        weight_anomaly = max(num_normal / max(num_anomaly, 1), 1.0)
        weights = torch.Tensor([weight_normal, weight_anomaly])
        print("类别权重: 正常={:.2f}, 异常={:.2f} (正常:异常 = {:,}:{:,})".format(
            weight_normal, weight_anomaly, num_normal, num_anomaly))
    else:
        weights = torch.Tensor([1., 1.])
        print("不使用类别权重 (正常:异常 = {:,}:{:,})".format(num_normal, num_anomaly))
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    
    anomaly = (data.y == 1)
    normal = (data.y == 0)
    label = (anomaly, normal)

    c = 0
    auc_pr_best = 0
    auc_roc_best = 0
    auc_roc_test_epoch = 0
    auc_pr_test_epoch = 0
    best_net = None
    best_epoch = 0
    
    # 保存模型路径
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model_save_path = os.path.join(args.save_dir, f"model_exp{exp_num+1}_best.pth")

    print("\n开始训练 (epochs={}, patience={})...".format(args.epochs, args.patience))
    for epoch in range(args.epochs):
        loss = train(data, net, criterion, optimizer, label, beta=args.beta)
        
        auc_roc_val, auc_pr_val = evaluate_model(net, data, data.val_mask, batch_size=args.batch_size)
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == 0:
            print('Epoch:{:04d}\tloss:{:.4f}\tVal AUC-ROC:{:.4f}\tVal AUC-PR:{:.4f}'
                  '\tBest AUC-ROC:{:.4f}\tBest AUC-PR:{:.4f}'
                        .format(epoch + 1, loss, auc_roc_val, auc_pr_val, auc_roc_test_epoch, auc_pr_test_epoch))

        if auc_pr_val >= auc_pr_best:
            auc_pr_best = auc_pr_val
            auc_roc_best = auc_roc_val
            auc_roc_test_epoch, auc_pr_test_epoch, _, _ = evaluate_with_batch_inference(
                net, data, data.test_mask, batch_size=args.batch_size
            )
            best_net = deepcopy(net)
            best_epoch = epoch + 1
            c = 0
            
            # 保存最佳模型
            save_model(best_net, optimizer, args, best_epoch, auc_pr_best, auc_roc_best, model_save_path)
        else:
            c += 1
            
        if c == args.patience:
            print('早停触发! 在 epoch {} 停止训练'.format(epoch+1))
            print('最佳模型在 epoch {}'.format(best_epoch))
            break

    print("\n最终评估最佳模型 (来自 epoch {})...".format(best_epoch))
    
    # 只计算测试集（推理数据集）的推理时间
    test_inference_time = 0
    test_peak_memory = 0
    
    if best_net is not None:
        # 只对测试集进行推理评估
        auc_roc_test_exp, auc_pr_test_exp, test_inference_time, test_peak_memory = evaluate_with_batch_inference(
            best_net, data, data.test_mask, batch_size=args.batch_size
        )
        
        # 验证集评估（不计算推理时间）
        auc_roc_val_exp, auc_pr_val_exp = evaluate_model(
            best_net, data, data.val_mask, batch_size=args.batch_size
        )
    else:
        # 只对测试集进行推理评估
        auc_roc_test_exp, auc_pr_test_exp, test_inference_time, test_peak_memory = evaluate_with_batch_inference(
            net, data, data.test_mask, batch_size=args.batch_size
        )
        
        # 验证集评估（不计算推理时间）
        auc_roc_val_exp, auc_pr_val_exp = evaluate_model(
            net, data, data.val_mask, batch_size=args.batch_size
        )
    
    print("\n实验 {} 结果:".format(exp_num+1))
    print("  验证集 AUC-ROC: {:.5f}".format(auc_roc_val_exp))
    print("  验证集 AUC-PR: {:.5f}".format(auc_pr_val_exp))
    print("  测试集 AUC-ROC: {:.5f}".format(auc_roc_test_exp))
    print("  测试集 AUC-PR: {:.5f}".format(auc_pr_test_exp))
    print("  测试集推理时间: {:.4f} 秒".format(test_inference_time))
    print("  峰值显存: {:.2f} MB".format(test_peak_memory))
    print("  最佳模型保存到: {}".format(model_save_path))
    
    # 保存实验结果
    results = {
        'exp_num': exp_num + 1,
        'val_auc_roc': float(auc_roc_val_exp),
        'val_auc_pr': float(auc_pr_val_exp),
        'test_auc_roc': float(auc_roc_test_exp),
        'test_auc_pr': float(auc_pr_test_exp),
        'test_inference_time': float(test_inference_time),
        'peak_memory': float(test_peak_memory),
        'best_epoch': best_epoch,
        'model_path': model_save_path
    }
    
    results_file = os.path.join(args.save_dir, f"results_exp{exp_num+1}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return auc_roc_val_exp, auc_pr_val_exp, auc_roc_test_exp, auc_pr_test_exp, test_inference_time, test_peak_memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMNet 图异常检测训练和验证')

    parser.add_argument('--data_path', type=str, required=True, help='.npz数据文件路径')
    parser.add_argument('--save_dir', type=str, required=True, help='模型保存目录')
    parser.add_argument('--hidden_channels', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--M', type=int, default=8, help='滤波器数量')
    parser.add_argument('--K', type=int, default=2, help='滤波器阶数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--lr_f', type=float, default=1e-1, help='滤波器学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--beta', type=float, default=0.5, help='偏差损失权重')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--exp_num', type=int, default=10, help='实验重复次数')
    parser.add_argument('--eval_interval', type=int, default=10, help='评估间隔')
    parser.add_argument('--use_class_weight', action='store_true', help='使用类别权重')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=512, help='推理批大小')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("AMNet 训练验证配置")
    print("="*60)
    print("数据文件: {}".format(args.data_path))
    print("模型保存目录: {}".format(args.save_dir))
    print("模型参数:")
    print("  隐藏维度: {}".format(args.hidden_channels))
    print("  滤波器数量 (M): {}".format(args.M))
    print("  滤波器阶数 (K): {}".format(args.K))
    print("训练参数:")
    print("  学习率: {}".format(args.lr))
    print("  滤波器学习率: {}".format(args.lr_f))
    print("  权重衰减: {}".format(args.weight_decay))
    print("  epochs: {}".format(args.epochs))
    print("  早停耐心值: {}".format(args.patience))
    print("  偏差权重 (beta): {}".format(args.beta))
    print("实验参数:")
    print("  重复次数: {}".format(args.exp_num))
    print("  评估间隔: {}".format(args.eval_interval))
    print("  使用类别权重: {}".format(args.use_class_weight))
    print("  随机种子: {}".format(args.seed))
    print("推理参数:")
    print("  批大小: {}".format(args.batch_size))
    print("="*60)

    if not os.path.exists(args.data_path):
        print("错误: 文件不存在 - {}".format(args.data_path))
        exit(1)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("错误: 需要安装scikit-learn库")
        print("请运行: pip install scikit-learn")
        exit(1)

    # 分别记录验证集和测试集的指标
    val_auc_roc_list = []
    val_auc_pr_list = []
    test_auc_roc_list = []
    test_auc_pr_list = []
    inference_time_list = []
    peak_memory_list = []

    for i in range(args.exp_num):
        print("\n" + "="*60)
        print("实验 {}/{}".format(i+1, args.exp_num))
        print("="*60)
        
        auc_roc_val, auc_pr_val, auc_roc_test, auc_pr_test, inf_time, peak_mem = main(args, exp_num=i)
        
        val_auc_roc_list.append(auc_roc_val)
        val_auc_pr_list.append(auc_pr_val)
        test_auc_roc_list.append(auc_roc_test)
        test_auc_pr_list.append(auc_pr_test)
        inference_time_list.append(inf_time)
        peak_memory_list.append(peak_mem)
        
        print("\n实验 {} 完成:".format(i+1))
        print("  Valid: AUC-ROC={:.5f}, AUC-PR={:.5f}".format(auc_roc_val, auc_pr_val))
        print("  Test:  AUC-ROC={:.5f}, AUC-PR={:.5f}".format(auc_roc_test, auc_pr_test))
        print("  测试集推理时间: {:.4f} 秒".format(inf_time))
        print("  峰值显存: {:.2f} MB".format(peak_mem))

    # 输出统计结果
    print("\n" + "="*60)
    print("最终统计结果 ({}折交叉验证)".format(args.exp_num))
    print("="*60)
    print("数据集: {}".format(args.data_path))
    print("实验次数: {}".format(args.exp_num))
    print("模型保存目录: {}".format(args.save_dir))
    
    print("\n验证集 (Valid) 结果:")
    print("  AUC-ROC: {:.5f} ± {:.5f}".format(np.mean(val_auc_roc_list), np.std(val_auc_roc_list)))
    print("  AUC-PR:  {:.5f} ± {:.5f}".format(np.mean(val_auc_pr_list), np.std(val_auc_pr_list)))
    
    print("\n测试集 (Test) 结果:")
    print("  AUC-ROC: {:.5f} ± {:.5f}".format(np.mean(test_auc_roc_list), np.std(test_auc_roc_list)))
    print("  AUC-PR:  {:.5f} ± {:.5f}".format(np.mean(test_auc_pr_list), np.std(test_auc_pr_list)))
    
    print("\n推理性能统计:")
    print("  每折测试集推理时间: {:.4f} ± {:.4f} 秒".format(np.mean(inference_time_list), np.std(inference_time_list)))
    print("  {}折测试集推理总时间: {:.4f} 秒".format(args.exp_num, np.sum(inference_time_list)))
    print("  峰值显存: {:.2f} ± {:.2f} MB".format(np.mean(peak_memory_list), np.std(peak_memory_list)))
    
    # 打印详细结果
    print("\n详细结果:")
    print("{:^6} | {:^12} {:^12} | {:^12} {:^12} | {:^12} {:^10}".format(
        "Fold", "Val AUC-ROC", "Val AUC-PR", "Test AUC-ROC", "Test AUC-PR", "Test Time(s)", "Mem(MB)"))
    print("-" * 84)
    for i in range(args.exp_num):
        print("{:^6} | {:^12.5f} {:^12.5f} | {:^12.5f} {:^12.5f} | {:^12.4f} {:^10.2f}".format(
            i+1, val_auc_roc_list[i], val_auc_pr_list[i], 
            test_auc_roc_list[i], test_auc_pr_list[i],
            inference_time_list[i], peak_memory_list[i]))
    
    # 保存结果到文件
    if args.exp_num > 1:
        results_file = os.path.join(args.save_dir, "final_results.txt")
        with open(results_file, 'w') as f:
            f.write("数据集: {}\n".format(args.data_path))
            f.write("实验次数: {}\n".format(args.exp_num))
            f.write("批大小: {}\n".format(args.batch_size))
            f.write("模型保存目录: {}\n\n".format(args.save_dir))
            
            f.write("验证集 (Valid) 结果:\n")
            f.write("  AUC-ROC: {:.5f} ± {:.5f}\n".format(np.mean(val_auc_roc_list), np.std(val_auc_roc_list)))
            f.write("  AUC-PR:  {:.5f} ± {:.5f}\n\n".format(np.mean(val_auc_pr_list), np.std(val_auc_pr_list)))
            
            f.write("测试集 (Test) 结果:\n")
            f.write("  AUC-ROC: {:.5f} ± {:.5f}\n".format(np.mean(test_auc_roc_list), np.std(test_auc_roc_list)))
            f.write("  AUC-PR:  {:.5f} ± {:.5f}\n\n".format(np.mean(test_auc_pr_list), np.std(test_auc_pr_list)))
            
            f.write("推理性能统计:\n")
            f.write("  每折测试集推理时间: {:.4f} ± {:.4f} 秒\n".format(np.mean(inference_time_list), np.std(inference_time_list)))
            f.write("  {}折测试集推理总时间: {:.4f} 秒\n".format(args.exp_num, np.sum(inference_time_list)))
            f.write("  峰值显存: {:.2f} ± {:.2f} MB\n\n".format(np.mean(peak_memory_list), np.std(peak_memory_list)))
            
            f.write("详细结果:\n")
            f.write("{:^6} | {:^12} {:^12} | {:^12} {:^12} | {:^12} {:^10}\n".format(
                "Fold", "Val AUC-ROC", "Val AUC-PR", "Test AUC-ROC", "Test AUC-PR", "Test Time(s)", "Mem(MB)"))
            f.write("-" * 84 + "\n")
            for i in range(args.exp_num):
                f.write("{:^6} | {:^12.5f} {:^12.5f} | {:^12.5f} {:^12.5f} | {:^12.4f} {:^10.2f}\n".format(
                    i+1, val_auc_roc_list[i], val_auc_pr_list[i], 
                    test_auc_roc_list[i], test_auc_pr_list[i],
                    inference_time_list[i], peak_memory_list[i]))
        
        print("\n最终结果已保存到: {}".format(results_file))
