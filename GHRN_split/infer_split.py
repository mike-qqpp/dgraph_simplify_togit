# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score
from BWGNN import *
import dgl
from tqdm import tqdm, trange


# GPU显存监控函数
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
            torch.cuda.reset_peak_memory_stats(self.gpu_id)
            self.initial_allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
            self.initial_cached = torch.cuda.memory_reserved(self.gpu_id) / 1024 / 1024
    
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
            peak_allocated = torch.cuda.max_memory_allocated(self.gpu_id) / 1024 / 1024
            peak_cached = torch.cuda.max_memory_reserved(self.gpu_id) / 1024 / 1024
            
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


def load_dgraphfin_data(data_path='./datasets/dgraphfin', data_name='dgraphfin'):
    """加载DGraphFin数据集"""
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
    
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    graph.ndata['feature'] = features
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    return graph, features, labels, train_mask, val_mask, test_mask


def load_model(model_path, in_feats, hid_dim, num_classes, graph, order, homo, device):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if homo:
        model = BWGNN(in_feats, hid_dim, num_classes, graph, d=order)
    else:
        model = BWGNN_Hetero(in_feats, hid_dim, num_classes, graph, d=order)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint


def evaluate_test_set(model, graph, features, labels, test_mask, best_threshold, batch_size=1024, gpu_id=0):
    """在测试集上进行评估"""
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
    
    inference_progress = tqdm(test_batches, desc='测试集推理', unit='批次', leave=False)
    
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
    
    return auc, ap, f1, rec, pre, inference_time, inference_peak_allocated, y_pred, all_probs[:, 1]


def main():
    parser = argparse.ArgumentParser(description='BWGNN Test Inference')
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset for this model")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    # 模型路径参数
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--save_predictions", action="store_true", 
                       help="Save predictions to file")
    parser.add_argument("--predictions_path", type=str, default="./predictions.npz",
                       help="Path to save predictions")
    
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
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        exit(1)
    
    # 开始计时：加载数据到推理完成的总时间
    total_time_start = time.time()
    
    # 加载数据
    graph, features, labels, train_mask, val_mask, test_mask = load_dgraphfin_data(
        data_path=data_path, 
        data_name=dataset_name
    )
    
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)
    
    in_feats = features.shape[1]
    num_classes = 2
    
    # 获取GPU总显存
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(args.gpu)
        gpu_total_memory = props.total_memory / 1024 / 1024
        print(f'GPU总显存: {gpu_total_memory:,.2f} MB')
    
    # 加载模型
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    
    model, checkpoint = load_model(
        args.model_path, in_feats, args.hid_dim, num_classes, 
        graph, args.order, args.homo, device
    )
    
    best_threshold = checkpoint.get('best_threshold', 0.5)
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    best_epoch = checkpoint.get('best_epoch', 0)
    
    print(f"模型加载成功!")
    print(f"  最佳验证F1: {best_val_f1:.4f}")
    print(f"  最佳阈值: {best_threshold:.4f}")
    print(f"  最佳轮次: {best_epoch}")
    
    # 在测试集上评估
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)
    
    test_auc, test_ap, test_f1, test_rec, test_pre, test_inf_time, test_peak_memory, test_pred, test_probs = evaluate_test_set(
        model, graph, features, labels, test_mask, best_threshold,
        batch_size=args.eval_batch_size, gpu_id=args.gpu
    )
    
    # 结束计时
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    
    # 输出结果
    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)
    print(f'AUC:  {test_auc:.4f}')
    print(f'AP:   {test_ap:.4f}')
    print(f'F1:   {test_f1:.4f}')
    print(f'Recall:   {test_rec:.4f}')
    print(f'Precision: {test_pre:.4f}')
    print(f'\n推理时间: {test_inf_time:.4f}s')
    print(f'加载数据到推理完成总时间: {total_time:.4f}s')
    print(f'峰值显存: {test_peak_memory:,.2f} MB')
    print("="*60)
    
    # 结果格式
    result = f'REC {test_rec*100:.2f} PRE {test_pre*100:.2f} MF1 {test_f1*100:.2f} AUC {test_auc*100:.2f}'
    print(f'\n结果格式: {result}')
    
    # 保存结果
    with open('test_result.txt', 'a+') as f:
        f.write(f'{result}\n')
    
    # 保存预测结果
    if args.save_predictions:
        np.savez(args.predictions_path, 
                 predictions=test_pred, 
                 probabilities=test_probs,
                 labels=labels[test_mask].cpu().numpy())
        print(f"\n预测结果已保存至: {args.predictions_path}")
    
    return {
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'test_rec': test_rec,
        'test_pre': test_pre,
        'inference_time': test_inf_time,
        'peak_memory': test_peak_memory
    }


if __name__ == '__main__':
    main()

