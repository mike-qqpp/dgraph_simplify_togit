# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score
from BWGNN import *
from sklearn.model_selection import train_test_split
import pickle as pkl
import dgl
import dgl.dataloading as dgl_dataloading
import gc
import psutil
import subprocess
from tqdm import tqdm, trange

# GPU显存监控函数 - 使用PyTorch的峰值内存统计
class GPUMemoryMonitor:
    """GPU显存监控器"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.initial_allocated = 0
        self.initial_cached = 0
        
    def start(self):
        """开始监控"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_id)
            # 重置峰值内存统计
            torch.cuda.reset_peak_memory_stats(self.gpu_id)
            # 记录初始内存
            self.initial_allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024  # MB
            self.initial_cached = torch.cuda.memory_reserved(self.gpu_id) / 1024 / 1024  # MB
    
    def get_current_memory(self):
        """获取当前内存使用"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_id)
            allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
            cached = torch.cuda.memory_reserved(self.gpu_id) / 1024 / 1024
            return allocated, cached
        return 0.0, 0.0
    
    def get_peak_memory(self):
        """获取峰值内存使用"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_id)
            # 获取峰值分配的内存
            peak_allocated = torch.cuda.max_memory_allocated(self.gpu_id) / 1024 / 1024
            peak_cached = torch.cuda.max_memory_reserved(self.gpu_id) / 1024 / 1024
            
            # 计算净增加（减去初始内存）
            net_peak_allocated = max(peak_allocated - self.initial_allocated, 0)
            net_peak_cached = max(peak_cached - self.initial_cached, 0)
            
            return net_peak_allocated, net_peak_cached
        return 0.0, 0.0


class GraphDataLoader:
    """图数据加载器，支持minibatch训练和推理"""
    def __init__(self, graph, features, labels, masks, batch_size=1024, shuffle=True, drop_last=False):
        self.graph = graph
        self.features = features
        self.labels = labels
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def get_loader(self, mask_type='train'):
        """获取指定掩码类型的DataLoader"""
        mask = self.masks[mask_type]
        indices = torch.nonzero(mask, as_tuple=True)[0]
        
        if self.shuffle and mask_type == 'train':
            indices = indices[torch.randperm(len(indices))]
        
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            batches.append({
                'node_ids': batch_indices,
                'features': self.features[batch_indices],
                'labels': self.labels[batch_indices],
                'batch_size': len(batch_indices),
                'global_features': self.features,
                'global_labels': self.labels
            })
        
        return batches
    
    def get_full_batch(self, mask_type='train'):
        """获取完整批次（用于验证和测试）"""
        mask = self.masks[mask_type]
        indices = torch.nonzero(mask, as_tuple=True)[0]
        
        return [{
            'node_ids': indices,
            'features': self.features[indices],
            'labels': self.labels[indices],
            'batch_size': len(indices),
            'global_features': self.features,
            'global_labels': self.labels
        }]


def truncate_mask(mask, labels, truncate_size, mask_name="mask", mode="random"):
    """截断掩码，只保留N个样本"""
    if truncate_size <= 0:
        return mask
    
    true_indices = torch.nonzero(mask, as_tuple=True)[0]
    
    if len(true_indices) <= truncate_size:
        return mask
    
    if mode == "first":
        selected_indices = true_indices[:truncate_size]
    elif mode == "random":
        selected_indices = true_indices[torch.randperm(len(true_indices))[:truncate_size]]
    elif mode == "balanced":
        true_labels = labels[true_indices]
        pos_indices = true_indices[true_labels == 1]
        neg_indices = true_indices[true_labels == 0]
        
        pos_target = max(1, int(truncate_size * len(pos_indices) / len(true_indices)))
        neg_target = truncate_size - pos_target
        
        if len(pos_indices) > 0:
            pos_selected = pos_indices[torch.randperm(len(pos_indices))[:min(pos_target, len(pos_indices))]]
        else:
            pos_selected = torch.tensor([], dtype=torch.long)
            
        if len(neg_indices) > 0:
            neg_selected = neg_indices[torch.randperm(len(neg_indices))[:min(neg_target, len(neg_indices))]]
        else:
            neg_selected = torch.tensor([], dtype=torch.long)
        
        selected_indices = torch.cat([pos_selected, neg_selected])
    
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[selected_indices] = True
    
    return new_mask


def load_dgraphfin_data(data_path='./datasets/dgraphfin', data_name='dgraphfin', 
                       train_truncate=0, val_truncate=0, test_truncate=0,
                       truncate_mode="random"):
    """加载DGraphFin数据集，返回布尔掩码，支持截断"""
    npz_path = os.path.join(data_path, '{}.npz'.format(data_name))
    if not os.path.exists(npz_path):
        raise FileNotFoundError("DGraphFin数据文件不存在: {}".format(npz_path))

    npz = np.load(npz_path)

    features = torch.from_numpy(npz['x']).float()
    labels = torch.from_numpy(npz['y']).long()
    edge_index = torch.from_numpy(npz['edge_index']).long()

    train_idx = npz['train_mask']
    valid_idx = npz['valid_mask']
    test_idx = npz['test_mask']

    num_nodes = features.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    if isinstance(train_idx, np.ndarray):
        train_mask[train_idx] = True
    if isinstance(valid_idx, np.ndarray):
        val_mask[valid_idx] = True
    if isinstance(test_idx, np.ndarray):
        test_mask[test_idx] = True

    print("DGraphFin数据集加载成功:")
    print("  节点数: {:,}, 特征维度: {}".format(features.shape[0], features.shape[1]))
    print("  训练集大小: {:,}".format(train_mask.sum().item()))
    print("  验证集大小: {:,}".format(val_mask.sum().item()))
    print("  测试集大小: {:,}".format(test_mask.sum().item()))
    print("  边数: {:,}".format(edge_index.shape[1]))
    
    if train_truncate > 0 or val_truncate > 0 or test_truncate > 0:
        print(f"\n应用截断 (模式: {truncate_mode}):")
        
        if train_truncate > 0:
            train_mask = truncate_mask(train_mask, labels, train_truncate, "train_mask", truncate_mode)
            print(f"  训练集截断为: {train_mask.sum().item():,}")
        
        if val_truncate > 0:
            val_mask = truncate_mask(val_mask, labels, val_truncate, "val_mask", truncate_mode)
            print(f"  验证集截断为: {val_mask.sum().item():,}")
        
        if test_truncate > 0:
            test_mask = truncate_mask(test_mask, labels, test_truncate, "test_mask", truncate_mode)
            print(f"  测试集截断为: {test_mask.sum().item():,}")
    
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    graph.ndata['feature'] = features
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    return graph, features, labels, train_mask, val_mask, test_mask


def train_minibatch(model, graph, features, labels, train_mask, val_mask, args, batch_size=1024, gpu_id=0):
    """Minibatch训练函数"""
    masks = {
        'train': train_mask,
        'val': val_mask,
        'test': val_mask
    }
    data_loader = GraphDataLoader(graph, features, labels, masks, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_labels = labels[train_mask]
    if (train_labels == 1).sum().item() == 0:
        weight = 1.0
    else:
        weight = (1-train_labels).sum().item() / train_labels.sum().item()
    
    weight_tensor = torch.tensor([1., weight], device=features.device)
    
    best_val_f1 = 0.0
    best_model_state = None
    best_epoch = 0
    best_threshold = 0.5
    
    # 创建局部内存监控器
    memory_monitor = GPUMemoryMonitor(gpu_id)
    memory_monitor.start()
    
    time_start = time.time()
    
    for epoch in trange(1, args.epoch + 1, desc="训练轮次", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        
        train_batches = data_loader.get_loader('train')
        batch_progress = tqdm(train_batches, desc=f'轮次 {epoch}/{args.epoch}', leave=False, unit='批次')
        
        for batch_idx, batch in enumerate(batch_progress):
            logits = model(features)
            batch_loss = F.cross_entropy(
                logits[batch['node_ids']], 
                labels[batch['node_ids']], 
                weight=weight_tensor
            )
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            batch_progress.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        batch_progress.close()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            probs = logits.softmax(1)
            
            val_f1, thres = get_best_f1(labels[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy())
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                best_threshold = thres
        
        print(f'轮次 {epoch:3d}, 损失: {epoch_loss/len(train_batches):.4f}, '
              f'验证F1: {val_f1:.4f} (最佳: {best_val_f1:.4f})')
        
        torch.cuda.empty_cache()
    
    time_end = time.time()
    training_time = time_end - time_start
    
    # 获取训练峰值显存
    train_peak_allocated, train_peak_cached = memory_monitor.get_peak_memory()
    
    print(f'\n训练完成! 时间: {training_time:.2f}s, 最佳轮次: {best_epoch}, 验证F1: {best_val_f1:.4f}')
    print(f'训练峰值显存使用: {train_peak_allocated:,.2f} MB')
    
    return best_model_state, best_val_f1, best_threshold, training_time, train_peak_allocated


def evaluate_minibatch(model, graph, features, labels, test_mask, best_threshold, batch_size=1024, 
                      mask_name='测试', measure_time=True, gpu_id=0):
    """Minibatch推理评估函数"""
    model.eval()
    
    masks = {'test': test_mask}
    data_loader = GraphDataLoader(graph, features, labels, masks, batch_size=batch_size, shuffle=False)
    test_batches = data_loader.get_full_batch('test')
    
    all_probs = []
    all_labels = []
    
    # 创建局部内存监控器
    memory_monitor = GPUMemoryMonitor(gpu_id)
    memory_monitor.start()
    
    inference_start = time.time()
    
    inference_progress = tqdm(test_batches, desc=f'{mask_name}推理', unit='批次', leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(inference_progress):
            logits = model(features)
            probs = logits.softmax(1)
            
            all_probs.append(probs[batch['node_ids']].cpu().numpy())
            all_labels.append(labels[batch['node_ids']].cpu().numpy())
            
            inference_progress.set_postfix({'节点': f'{batch["batch_size"]}'})
    
    inference_progress.close()
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # 获取推理峰值显存
    inference_peak_allocated, inference_peak_cached = memory_monitor.get_peak_memory()
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    y_pred = np.zeros_like(all_labels)
    y_pred[all_probs[:, 1] > best_threshold] = 1
    
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    ap = average_precision_score(all_labels, all_probs[:, 1])
    f1 = f1_score(all_labels, y_pred, average='macro')
    rec = recall_score(all_labels, y_pred)
    pre = precision_score(all_labels, y_pred)
    
    if measure_time:
        print(f'\n{mask_name}推理完成! 时间: {inference_time:.4f}s')
        print(f'峰值显存: {inference_peak_allocated:,.2f} MB')
        print(f'AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}, Precision: {pre:.4f}')
    
    return auc, ap, f1, rec, pre, inference_time, inference_peak_allocated


def train_and_evaluate_minibatch(model, g, args, train_batch_size=1024, eval_batch_size=2048):
    """使用minibatch进行训练和评估"""
    features = g.ndata['feature']
    labels = g.ndata['label']
    
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    print('训练/验证/测试样本数: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    
    print("\n" + "="*60)
    print("开始Minibatch训练")
    print("="*60)
    
    best_model_state, best_val_f1, best_threshold, training_time, train_peak_memory = train_minibatch(
        model, g, features, labels, train_mask, val_mask, args, 
        batch_size=train_batch_size, gpu_id=args.gpu
    )
    
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*60)
    print("验证集评估")
    print("="*60)
    
    val_auc, val_ap, val_f1, val_rec, val_pre, val_inf_time, val_peak_memory = evaluate_minibatch(
        model, g, features, labels, val_mask, best_threshold, 
        batch_size=eval_batch_size, mask_name='验证集', gpu_id=args.gpu
    )
    
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)
    
    test_auc, test_ap, test_f1, test_rec, test_pre, test_inf_time, test_peak_memory = evaluate_minibatch(
        model, g, features, labels, test_mask, best_threshold, 
        batch_size=eval_batch_size, mask_name='测试集', gpu_id=args.gpu
    )
    
    # 计算总体峰值
    overall_peak_memory = max(train_peak_memory, val_peak_memory, test_peak_memory)
    
    # 输出总结
    print("\n" + "="*60)
    print("性能总结")
    print("="*60)
    print(f'最佳验证F1: {best_val_f1:.4f} (阈值: {best_threshold:.4f})')
    print(f'验证集 AUC: {val_auc:.4f}, AP: {val_ap:.4f}')
    print(f'测试集 AUC: {test_auc:.4f}, AP: {test_ap:.4f}')
    print('\n时间总结:')
    print(f'  总训练时间: {training_time:.2f}s')
    print(f'  总推理时间: {val_inf_time + test_inf_time:.4f}s')
    print('\n峰值显存使用:')
    print(f'  训练阶段: {train_peak_memory:,.2f} MB')
    print(f'  验证阶段: {val_peak_memory:,.2f} MB')
    print(f'  测试阶段: {test_peak_memory:,.2f} MB')
    print(f'  总体峰值: {overall_peak_memory:,.2f} MB')
    print("="*60)
    
    result = f'REC {test_rec*100:.2f} PRE {test_pre*100:.2f} MF1 {test_f1*100:.2f} AUC {test_auc*100:.2f}'
    print(f'\n结果格式: {result}')
    
    with open('result.txt', 'a+') as f:
        f.write(f'{result}\n')
    
    return {
        'best_val_f1': best_val_f1,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'training_time': training_time,
        'total_inference_time': val_inf_time + test_inf_time,
        'train_peak_memory': train_peak_memory,
        'val_peak_memory': val_peak_memory,
        'test_peak_memory': test_peak_memory,
        'overall_peak_memory': overall_peak_memory
    }


# 最佳F1阈值调整函数
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BWGNN with Minibatch and Mask Truncation')
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset for this model")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--del_ratio", type=float, default=0., help="delete ratios")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--load_epoch", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")
    parser.add_argument("--train_batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    # 新增的截断参数
    parser.add_argument("--train_truncate", type=int, default=0, 
                       help="Truncate training mask to top N samples (0 for no truncation)")
    parser.add_argument("--val_truncate", type=int, default=0, 
                       help="Truncate validation mask to top N samples (0 for no truncation)")
    parser.add_argument("--test_truncate", type=int, default=0, 
                       help="Truncate test mask to top N samples (0 for no truncation)")
    parser.add_argument("--truncate_mode", type=str, default="random", 
                       choices=["random", "first", "balanced"],
                       help="How to truncate masks: random, first (keep first N), balanced (keep balanced classes)")

    args = parser.parse_args()
    print(args)
    
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        print(f'使用GPU设备: {args.gpu}')
    else:
        device = torch.device('cpu')
        print('使用CPU设备')
    
    dataset_name = args.dataset
    data_path = args.data_path
    
    # 判断是否是dgraphfin数据集
    if 1:
        graph, features, labels, train_mask, val_mask, test_mask = load_dgraphfin_data(
            data_path=data_path, 
            data_name=dataset_name,
            train_truncate=args.train_truncate,
            val_truncate=args.val_truncate,
            test_truncate=args.test_truncate,
            truncate_mode=args.truncate_mode
        )
        
        graph = graph.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        
        graph.ndata['feature'] = features
        graph.ndata['label'] = labels
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        
        in_feats = features.shape[1]
        num_classes = 2
    else:
        try:
            from dataset import Dataset
            graph = Dataset(load_epoch, dataset_name, del_ratio, homo, data_path, adj_type=adj_type).graph
            in_feats = graph.ndata['feature'].shape[1]
            num_classes = 2
            
            graph = graph.to(device)
            features = graph.ndata['feature'].to(device)
            labels = graph.ndata['label'].to(device)
            train_mask = graph.ndata['train_mask'].to(device)
            val_mask = graph.ndata['val_mask'].to(device)
            test_mask = graph.ndata['test_mask'].to(device)
        except ImportError:
            print("错误: 无法导入Dataset模块")
            exit(1)

    set_random_seed(717)
    
    # 获取GPU总显存
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(args.gpu)
        gpu_total_memory = props.total_memory / 1024 / 1024  # MB
        print(f'GPU总显存: {gpu_total_memory:,.2f} MB')
    
    if args.run == 1:
        if args.homo:
            model = BWGNN(in_feats, args.hid_dim, num_classes, graph, d=args.order)
        else:
            model = BWGNN_Hetero(in_feats, args.hid_dim, num_classes, graph, d=args.order)
        
        model = model.to(device)
        
        results = train_and_evaluate_minibatch(
            model, graph, args, 
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size
        )
        
        print('\n' + '='*60)
        print('最终总结')
        print('='*60)
        print(f'最佳验证F1: {results["best_val_f1"]:.4f}')
        print(f'验证集AUC: {results["val_auc"]:.4f}')
        print(f'验证集AP:  {results["val_ap"]:.4f}')
        print(f'测试集AUC: {results["test_auc"]:.4f}')
        print(f'测试集AP:  {results["test_ap"]:.4f}')
        print(f'训练时间:  {results["training_time"]:.2f}s')
        print(f'总推理时间: {results["total_inference_time"]:.4f}s')
        print(f'峰值显存:  {results["overall_peak_memory"]:,.2f} MB')
        print('='*60)
    else:
        final_results = []
        for tt in trange(args.run, desc="多次运行", unit="次"):
            print('\n' + '='*60)
            print(f'运行 {tt+1}/{args.run}')
            print('='*60)
            
            if args.homo:
                model = BWGNN(in_feats, args.hid_dim, num_classes, graph, d=args.order)
            else:
                model = BWGNN_Hetero(in_feats, args.hid_dim, num_classes, graph, d=args.order)
            
            model = model.to(device)
            
            results = train_and_evaluate_minibatch(
                model, graph, args,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size
            )
            final_results.append(results)
            
            del model
            torch.cuda.empty_cache()
        
        # 多次运行统计
        summary_keys = ['best_val_f1', 'val_auc', 'val_ap', 'test_auc', 'test_ap', 
                       'training_time', 'total_inference_time', 'overall_peak_memory']
        
        print('\n' + '='*60)
        print(f'多次运行总结 ({args.run}次)')
        print('='*60)
        
        for key in summary_keys:
            values = [r[key] for r in final_results]
            print(f'\n{key}:')
            print(f'  均值: {np.mean(values):.4f}')
            print(f'  标准差: {np.std(values):.4f}')
            print(f'  最小值: {np.min(values):.4f}')
            print(f'  最大值: {np.max(values):.4f}')
            if 'time' in key:
                print(f'  单位: 秒')
            elif 'memory' in key:
                print(f'  单位: MB')
        
        print('='*60)

