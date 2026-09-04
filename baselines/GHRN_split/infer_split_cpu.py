#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理脚本 - 加载模型进行推理
使用方法:
python infer_split_cpu.py --dataset amazon --model_path ../models/xxx.pt
"""

import torch
import argparse
import numpy as np
import os
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from BWGNN import *
import dgl


def load_dgraphfin_data(data_path, data_name):
    """加载数据"""
    npz_path = os.path.join(data_path, '{}.npz'.format(data_name))
    npz = np.load(npz_path)

    features = torch.from_numpy(npz['x']).float()
    labels = torch.from_numpy(npz['y']).long()
    edge_index = torch.from_numpy(npz['edge_index']).long()
    test_idx = npz['test_mask']

    num_nodes = features.shape[0]
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if isinstance(test_idx, np.ndarray):
        test_mask[test_idx] = True

    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    
    return graph, features, labels, test_mask


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


def main():
    parser = argparse.ArgumentParser(description='BWGNN Inference')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_path", type=str, default='./data', help="Data path")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--order", type=int, default=2, help="Order")
    parser.add_argument("--homo", type=int, default=1, help="Homo")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Batch size")
    
    args = parser.parse_args()
    
    # 固定使用CPU
    device = torch.device('cpu')
    
    # 加载数据
    graph, features, labels, test_mask = load_dgraphfin_data(
        data_path=args.data_path, 
        data_name=args.dataset
    )
    
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)
    
    in_feats = features.shape[1]
    num_classes = 2
    
    # 加载模型
    model, checkpoint = load_model(
        args.model_path, in_feats, args.hid_dim, num_classes, 
        graph, args.order, args.homo, device
    )
    
    best_threshold = checkpoint.get('best_threshold', 0.5)
    
    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = logits.softmax(1)
        
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
        test_probs = probs[test_indices].cpu().numpy()
        test_labels = labels[test_indices].cpu().numpy()
    
    # 计算指标
    y_pred = np.zeros_like(test_labels)
    y_pred[test_probs[:, 1] > best_threshold] = 1
    
    auc = roc_auc_score(test_labels, test_probs[:, 1])
    ap = average_precision_score(test_labels, test_probs[:, 1])
    f1 = f1_score(test_labels, y_pred, average='macro')
    
    # 输出结果（供shell脚本获取）
    print(f"AUC:{auc:.4f}")
    print(f"AP:{ap:.4f}")
    print(f"F1:{f1:.4f}")
    print(f"TEST_SIZE:{len(test_labels)}")


if __name__ == '__main__':
    main()

