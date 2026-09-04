# The file in which the model is trained

import torch
import torch.nn.functional as F
import dgl
import sklearn.metrics as skmetrics
from torch.utils.data import DataLoader, TensorDataset
from models.base import BaseModel, get_thres
from models.DSGAD import DSGAD
import os
import sys
import time
import json
import argparse
import warnings
import random
import numpy as np

# 添加 datasets 目录到 Python 路径
datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
if datasets_path not in sys.path:
    sys.path.insert(0, datasets_path)

import datasets


def seed_everything(seed, strengthen=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strengthen:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ['PYTHONHASHSEED'] = str(seed)


def setup_multigpu():
    """设置多GPU环境"""
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"检测到 {num_gpus} 个GPU，使用多GPU训练")
        return True, num_gpus, list(range(num_gpus))
    return False, 1, [0]


def train_model(args):
    """训练模型并返回最佳模型和结果"""
    # 自动检测GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    # 打印详细的GPU信息
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # 处理model_config
    if hasattr(args, 'model_config') and args.model_config:
        model_config = args.model_config.copy()
    else:
        model_config = {}
    
    # 确保有必要的配置
    model_config['h_feats'] = model_config.get('h_feats', 32)
    model_config['d'] = model_config.get('d', 2)
    model_config['mix_beta'] = model_config.get('mix_beta', 2)
    model_config['use_reduced_memory'] = model_config.get('use_reduced_memory', True)
    
    print(f"\n模型配置: {model_config}")
    
    # Load the graph dataset
    dataset = eval(f'datasets.{args.dataset}')()
    g: dgl.DGLGraph = dataset.g
    
    # 打印图的详细信息
    print(f"\n数据集: {args.dataset}")
    print(f"图结构: {g}")
    print(f"节点数: {g.num_nodes():,}")
    print(f"边数: {g.num_edges():,}")
    print(f"节点特征维度: {g.ndata['feature'].shape[1]}")
    
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    print(f'\n训练集/验证集/测试集样本数: {train_mask.sum().item():,} / {val_mask.sum().item():,} / {test_mask.sum().item():,}')

    features = g.ndata['feature']
    labels = g.ndata['label']

    # Number of Normals / Abnormalities
    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print(f'\n类别权重（交叉熵）: {weight:.5f}')

    # ========== 真正的模型并行 ==========
    if num_gpus > 1:
        print(f"\n使用 {num_gpus} 个GPU进行真正的模型并行训练")
        print("="*60)
        
        # 为每个GPU创建独立的模型实例
        models = []
        devices = []
        
        for device_id in range(num_gpus):
            device = torch.device(f'cuda:{device_id}')
            devices.append(device)
            
            # 复制模型配置，设置不同的device_id
            gpu_config = model_config.copy()
            gpu_config['device_id'] = device_id
            gpu_config['num_gpus'] = num_gpus
            
            # 创建模型并移动到对应GPU
            model = DSGAD(features.shape[0], features.shape[1], **gpu_config).to(device)
            models.append(model)
            
            print(f"GPU {device_id}: 模型已创建并移动到 {device}")
        
        # 主模型（用于评估和保存）
        main_model = models[0]
        main_device = devices[0]
        
        # 打印总的节点分配情况
        print(f"\n节点分配总结:")
        total_nodes = features.shape[0]
        nodes_per_gpu = total_nodes // num_gpus
        for device_id in range(num_gpus):
            start = device_id * nodes_per_gpu
            end = (device_id + 1) * nodes_per_gpu if device_id < num_gpus - 1 else total_nodes
            print(f"  GPU {device_id}: 节点 {start:,} - {end:,} (共 {end-start:,} 个节点)")
        
        # 将图和特征复制到各个GPU
        # 主GPU上有完整的数据
        g_main = g.to(main_device)
        features_main = features.to(main_device)
        labels_main = labels.to(main_device)
        
        # 其他GPU也需要数据（用于前向传播）
        g_list = [g_main]
        features_list = [features_main]
        labels_list = [labels_main]
        
        for device_id in range(1, num_gpus):
            device = devices[device_id]
            g_list.append(g.to(device))
            features_list.append(features.to(device))
            labels_list.append(labels.to(device))
        
        print(f"\n数据已分发到各个GPU")
        
    else:
        # 单GPU情况
        device = args.device if hasattr(args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_config['device_id'] = model_config.get('device_id', 0)
        model_config['num_gpus'] = model_config.get('num_gpus', 1)
        
        main_model = DSGAD(features.shape[0], features.shape[1], **model_config).to(device)
        models = [main_model]
        devices = [device]
        main_device = device
        
        g_main = g.to(device)
        features_main = features.to(device)
        labels_main = labels.to(device)
        g_list = [g_main]
        features_list = [features_main]
        labels_list = [labels_main]
        
        print(f"\n使用单GPU训练: {device}")
    
    # 创建训练数据的索引
    train_idx = torch.where(train_mask)[0].to(main_device)
    
    # 设置batch size
    batch_size = getattr(args, 'batch_size', 256)
    print(f'\n使用Batch Size: {batch_size:,}')
    
    # 创建训练数据加载器
    train_dataset = TensorDataset(train_idx.cpu())
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False
    )
    print(f'训练数据加载器创建完成，共 {len(train_loader)} 个batch')

    # ========== 为每个GPU创建独立的优化器 ==========
    optimizers = []
    for i, model in enumerate(models):
        params = []
        for name, param in model.named_parameters():
            if 'weights' in name: 
                params.append({'params': param, 'lr': getattr(args, 'lr_weights', 1e-1)})
            else: 
                params.append({'params': param, 'lr': getattr(args, 'lr', 1e-3)})
        optimizer = torch.optim.Adam(params)
        optimizers.append(optimizer)
        print(f"GPU {i}: 优化器已创建")

    # 初始化最佳模型记录
    best_val_auc = 0
    best_val_ap = 0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    patience = getattr(args, 'patience', 10)
    
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")
    
    # ========== 训练循环 ==========
    for epoch in range(args.epoch):
        # 设置所有模型为训练模式
        for model in models:
            model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_num, batch_idx in enumerate(train_loader):
            batch_idx_main = batch_idx[0].to(main_device)
            
            # ========== 多GPU并行训练 ==========
            if len(models) > 1:
                # 每个GPU处理自己的部分
                for device_id, (model, optimizer, g_gpu, features_gpu, labels_gpu) in enumerate(
                    zip(models, optimizers, g_list, features_list, labels_list)
                ):
                    device = devices[device_id]
                    
                    # 将batch数据复制到当前GPU
                    batch_idx_gpu = batch_idx_main.to(device)
                    
                    # 前向传播（只处理当前GPU负责的节点）
                    out = model(g_gpu, features_gpu, node_indices=batch_idx_gpu)
                    
                    # 计算loss
                    loss = F.cross_entropy(
                        out,
                        labels_gpu[batch_idx_gpu],
                        weight=torch.tensor([1., weight], device=device),
                    )
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 只在主GPU上记录loss
                    if device_id == 0:
                        total_loss += loss.item()
                        
            else:
                # 单GPU训练
                out = main_model(g_main, features_main, node_indices=batch_idx_main)
                
                loss = F.cross_entropy(
                    out,
                    labels_main[batch_idx_main],
                    weight=torch.tensor([1., weight], device=main_device),
                )
                
                optimizers[0].zero_grad()
                loss.backward()
                optimizers[0].step()
                
                total_loss += loss.item()
            
            num_batches += 1
            
            # 打印训练进度
            if (batch_num + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch+1}, Batch {batch_num+1}/{len(train_loader)}, Loss: {avg_loss:.5f}')
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # ========== 验证集评估（使用批次） ==========
        main_model.eval()
        
        # 创建验证集数据加载器
        val_idx = torch.where(val_mask)[0].to(main_device)
        val_dataset = TensorDataset(val_idx.cpu())
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # 验证时可以用更大的batch
            shuffle=False,
            drop_last=False
        )
        
        all_val_probs = []
        all_val_labels = []
        
        with torch.no_grad():
            for val_batch_idx in val_loader:
                val_batch_idx = val_batch_idx[0].to(main_device)
                
                # 批次前向传播
                out = main_model(g_main, features_main, node_indices=val_batch_idx)
                
                # 收集概率和标签
                batch_probs = out.softmax(1)[:, 1].cpu()
                batch_labels = labels_main[val_batch_idx].cpu()
                
                all_val_probs.append(batch_probs)
                all_val_labels.append(batch_labels)
            
            # 合并所有批次的结果
            val_probs = torch.cat(all_val_probs)
            val_labels = torch.cat(all_val_labels)
            
            # 在验证集上找最佳阈值
            val_thres = get_thres(val_labels, val_probs)
            val_y_pred = torch.zeros_like(val_probs)
            val_y_pred[val_probs >= val_thres] = 1
            
            # 计算验证集指标
            val_auc = skmetrics.roc_auc_score(val_labels, val_probs)
            val_ap = skmetrics.average_precision_score(val_labels, val_probs)
            val_recall = skmetrics.recall_score(val_labels, val_y_pred)
            val_precision = skmetrics.precision_score(val_labels, val_y_pred)
            val_f1_macro = skmetrics.f1_score(val_labels, val_y_pred, average='macro')
            val_f1_micro = skmetrics.f1_score(val_labels, val_y_pred, average='micro')
            val_accuracy = skmetrics.accuracy_score(val_labels, val_y_pred)
        
        print("\n" + "="*60)
        print(f"Epoch {epoch+1}/{args.epoch} - 训练完成")
        print("="*60)
        print(f"训练集平均Loss: {avg_train_loss:.5f}")
        
        print("\n验证集指标:")
        print('auc     ap      recall  precision  f1_macro  f1_micro  accuracy')
        print(f'{val_auc:.5f}  {val_ap:.5f}  {val_recall:.5f}  {val_precision:.5f}  {val_f1_macro:.5f}  {val_f1_micro:.5f}  {val_accuracy:.5f}')
        print(f"验证集最佳阈值: {val_thres:.5f}")
        
        # ========== 检查是否保存最佳模型 ==========
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_ap = val_ap
            best_epoch = epoch + 1
            best_model_state = main_model.state_dict().copy()
            patience_counter = 0
            print(f"\n✅ 发现新的最佳模型! AUC: {best_val_auc:.5f}, AP: {best_val_ap:.5f}")
        else:
            patience_counter += 1
            print(f"\n⏳ 验证集AUC未提升，耐心值: {patience_counter}/{patience}")
        
        # ========== 早停检查 ==========
        if patience_counter >= patience:
            print(f"\n🚨 早停触发! 在 epoch {epoch+1} 停止训练")
            print(f"最佳模型在 epoch {best_epoch}，验证集AUC: {best_val_auc:.5f}")
            break
    
    print("\n" + "="*60)
    print("训练完成，开始验证集最终评估")
    print("="*60)
    
    # ========== 加载最佳模型 ==========
    if best_model_state is not None:
        main_model.load_state_dict(best_model_state)
        print(f"已加载 epoch {best_epoch} 的最佳模型")
    else:
        print("⚠️ 未找到最佳模型，使用最后一个epoch的模型")
    
    # ========== 最终验证集评估 ==========
    main_model.eval()
    
    # 创建验证集数据加载器
    val_idx = torch.where(val_mask)[0].to(main_device)
    val_dataset = TensorDataset(val_idx.cpu())
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False
    )
    
    all_val_probs = []
    all_val_labels = []
    
    with torch.no_grad():
        for val_batch_idx in val_loader:
            val_batch_idx = val_batch_idx[0].to(main_device)
            
            out = main_model(g_main, features_main, node_indices=val_batch_idx)
            
            batch_probs = out.softmax(1)[:, 1].cpu()
            batch_labels = labels_main[val_batch_idx].cpu()
            
            all_val_probs.append(batch_probs)
            all_val_labels.append(batch_labels)
        
        val_probs = torch.cat(all_val_probs)
        val_labels = torch.cat(all_val_labels)
        
        val_thres = get_thres(val_labels, val_probs)
        val_y_pred = torch.zeros_like(val_probs)
        val_y_pred[val_probs >= val_thres] = 1
        
        val_auc = skmetrics.roc_auc_score(val_labels, val_probs)
        val_ap = skmetrics.average_precision_score(val_labels, val_probs)
        val_recall = skmetrics.recall_score(val_labels, val_y_pred)
        val_precision = skmetrics.precision_score(val_labels, val_y_pred)
        val_f1_macro = skmetrics.f1_score(val_labels, val_y_pred, average='macro')
        val_f1_micro = skmetrics.f1_score(val_labels, val_y_pred, average='micro')
        val_accuracy = skmetrics.accuracy_score(val_labels, val_y_pred)
    
    print("\n验证集最终指标:")
    print('auc     ap      recall  precision  f1_macro  f1_micro  accuracy')
    print(f'{val_auc:.5f}  {val_ap:.5f}  {val_recall:.5f}  {val_precision:.5f}  {val_f1_macro:.5f}  {val_f1_micro:.5f}  {val_accuracy:.5f}')
    print(f"最佳阈值: {val_thres:.5f}")
    
    print("\n" + "="*60)
    print("训练总结:")
    print("="*60)
    print(f"最佳模型在 epoch {best_epoch}")
    print(f"验证集最佳AUC: {best_val_auc:.5f}")
    print(f"验证集最佳AP: {best_val_ap:.5f}")
    print("="*60)
    
    return main_model, {
        'best_epoch': best_epoch,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'threshold': float(val_thres),
        'val_recall': val_recall,
        'val_precision': val_precision,
        'val_f1_macro': val_f1_macro,
        'val_f1_micro': val_f1_micro,
        'val_accuracy': val_accuracy,
    }


def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)  # Random number seeds
    parser.add_argument('--run', type=int, default=1)  # How many times you run during training

    parser.add_argument('--dataset', type=str, default='dgraph')  # dataset
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.4, 0.3, 0.3])  # The ratio of the training/validation/testing set
    parser.add_argument('--epoch', type=int, default=2)  # epoch
    parser.add_argument('--device', type=str, default='cuda')  # train device: cuda or cpu

    parser.add_argument('--model', type=str, default='DSGAD')  # model
    parser.add_argument('--model_config', type=json.loads, default='{}')  # The setting of model
    parser.add_argument('--save', type=str, default='', help='save weights path')  # Model weights save paths
    parser.add_argument('--model_name', type=str, default='best_model', help='model name for saving')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='directory to save models')
    
    # 多GPU相关参数
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_weights', type=float, default=1e-1, help='learning rate for weights')
    parser.add_argument('--data_name', type=str, default='', help='data name')
    
    args = parser.parse_args()
    
    # 检查多GPU
    use_multigpu, num_gpus, device_ids = setup_multigpu()
    
    seed_everything(args.seed)
    
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'model config: {args.model_config}')
    print(f'device: {args.device}')
    if use_multigpu:
        print(f'使用多GPU训练，GPU数量: {num_gpus}')
    print()
    
    # 创建模型保存目录
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # 保存模型配置信息
    config_info = {
        'dataset': args.dataset,
        'model_config': args.model_config,
        'h_feats': args.model_config.get('h_feats', 32),
        'd': args.model_config.get('d', 2),
        'mix_beta': args.model_config.get('mix_beta', 2),
        'use_reduced_memory': args.model_config.get('use_reduced_memory', True),
    }
    
    metrics = ['auc', 'recall', 'precision', 'f1_macro']
    performance = {}
    for metric in metrics:
        performance[metric] = []
    
    start_time = time.time()
    for t in range(args.run):  # Train the model and collect metrics
        if args.device == 'cuda': 
            torch.cuda.empty_cache()

        print(f'trial: {t+1}/{args.run}')

        m, p = train_model(args)
        
        if m is not None and p is not None:
            # 保存模型
            if args.save:
                model_path = args.save
            else:
                model_path = os.path.join(args.model_save_dir, f"{args.model_name}.pt")
            
            checkpoint = {
                'model_state_dict': m.state_dict(),
                'best_epoch': p['best_epoch'],
                'val_auc': p['val_auc'],
                'val_ap': p['val_ap'],
                'threshold': p['threshold'],
                'config': config_info,
            }
            torch.save(checkpoint, model_path)
            print(f"\n模型已保存至: {model_path}")
            
            for metric in metrics:
                if metric in p:
                    performance[metric].append(p[metric])

        print()
    
    end_time = time.time()
    total_time = end_time - start_time

    # Finally, all metrics for each training session are output in a unified manner
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'model config: {args.model_config}')
    print(f'总训练时间: {total_time:.2f}s')
    
    print("\n验证集指标:")
    for metric in metrics:
        print(f'{metric} ', end='')
    print()
    for t in range(args.run):
        print(f'{t+1:<2}: ', end='')
        for metric in metrics:
            if t < len(performance[metric]):
                print(f'{performance[metric][t]:.5f} ', end='')
            else:
                print(f'0.00000 ', end='')
        print()
    
    print(f"\n模型保存路径: {os.path.join(args.model_save_dir, args.model_name + '.pt')}")


if __name__ == '__main__':
    main()

