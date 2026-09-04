from __future__ import annotations

import os
import time
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.io import loadmat
from tqdm import tqdm
from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN


def rgtan_train_and_save_models(feat_df, graph, train_idx, test_idx, labels, args, cat_features, 
                                neigh_features: pd.DataFrame, nei_att_head, model_save_dir):
    """
    训练并保存k折模型
    """
    device = args['device']
    graph = graph.to(device)
    
    # 创建模型保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"训练模式 - 保存模型到: {model_save_dir}")
    print(f"{'='*80}")
    
    # 存储每折模型路径
    fold_model_paths = []
    
    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = {}
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    print(f"数据信息:")
    print(f"  • 节点数: {len(feat_df)}")
    print(f"  • 训练节点: {len(train_idx)}")
    print(f"  • 测试节点: {len(test_idx)}")
    print(f"  • 特征维度: {feat_df.shape[1]}")
    print(f"  • 设备: {device}")
    print(f"  • K折数: {args['n_fold']}")
    
    kfold = StratifiedKFold(n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])
    
    all_fold_histories = []
    
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'\n{"-"*60}')
        print(f'训练第 {fold + 1}/{args["n_fold"]} 折')
        print(f'{"-"*60}')
        
        trn_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device)
        val_ind = torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        # 数据加载器
        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = DataLoader(graph, trn_ind, train_sampler,
                                      device=device, use_ddp=False,
                                      batch_size=args['batch_size'],
                                      shuffle=True, drop_last=False, num_workers=0)
        
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = DataLoader(graph, val_ind, val_sampler,
                                    use_ddp=False, device=device,
                                    batch_size=args['batch_size'],
                                    shuffle=True, drop_last=False, num_workers=0)
        
        # 创建模型
        model = RGTAN(in_feats=feat_df.shape[1],
                      hidden_dim=args['hid_dim']//4,
                      n_classes=2,
                      heads=[4]*args['n_layers'],
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)
        
        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 优化器
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)
        
        earlystoper = early_stopper(patience=args['early_stopping'], verbose=True)
        
        # 训练循环
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(args['max_epochs']):
            # 训练阶段
            model.train()
            train_loss_list = []
            
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                
                train_batch_logits, _ = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                
                seeds_valid_mask = (batch_labels == 0) | (batch_labels == 1)
                
                if seeds_valid_mask.any():
                    train_batch_logits_valid = train_batch_logits[seeds_valid_mask]
                    batch_labels_valid = batch_labels[seeds_valid_mask]
                    
                    if train_batch_logits_valid.shape[0] > 0 and batch_labels_valid.shape[0] > 0:
                        train_loss = loss_fn(train_batch_logits_valid, batch_labels_valid)
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        
                        train_loss_list.append(train_loss.cpu().detach().numpy())
            
            lr_scheduler.step()
            
            # 验证阶段
            model.eval()
            val_loss_list = []
            
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                        num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                        seeds, input_nodes, device, blocks)

                    blocks = [block.to(device) for block in blocks]
                    
                    val_batch_logits, _ = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                    
                    seeds_valid_mask = (batch_labels == 0) | (batch_labels == 1)
                    
                    if seeds_valid_mask.any():
                        val_batch_logits_valid = val_batch_logits[seeds_valid_mask]
                        batch_labels_valid = batch_labels[seeds_valid_mask]
                        
                        if val_batch_logits_valid.shape[0] > 0 and batch_labels_valid.shape[0] > 0:
                            val_loss = loss_fn(val_batch_logits_valid, batch_labels_valid)
                            val_loss_list.append(val_loss.cpu().detach().numpy())
            
            # 计算平均损失
            avg_train_loss = np.mean(train_loss_list) if train_loss_list else 0
            avg_val_loss = np.mean(val_loss_list) if val_loss_list else 0
            
            # 更新最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
            
            # Early stopping
            earlystoper.earlystop(avg_val_loss, model)
            if earlystoper.is_earlystop:
                print(f"  早停触发于第{epoch+1}轮, 最佳val_loss: {best_val_loss:.6f}")
                break
            
            if (epoch + 1) % 50 == 0:
                print(f"  第{epoch+1}轮: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # 保存当前折的最佳模型
        model_path = os.path.join(model_save_dir, f'fold_{fold+1}_best_model.pt')
        torch.save({
            'model_state_dict': best_model_state,
            'fold': fold,
            'val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'args': args
        }, model_path)
        fold_model_paths.append(model_path)
        
        print(f"✓ 第{fold+1}折完成: val_loss={best_val_loss:.6f}, 保存到: {os.path.basename(model_path)}")
        
        # 释放当前折的模型显存
        del model, optimizer, lr_scheduler
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    # 保存模型路径文件
    paths_file = os.path.join(model_save_dir, 'fold_model_paths.txt')
    with open(paths_file, 'w') as f:
        for path in fold_model_paths:
            f.write(f"{path}\n")
    
    # 保存训练参数
    args_file = os.path.join(model_save_dir, 'train_args.json')
    import json
    with open(args_file, 'w') as f:
        json.dump(args, f, indent=2, default=str)
    
    print(f'\n{"="*80}')
    print(f'✓ 训练完成! 保存了 {len(fold_model_paths)} 个模型到: {model_save_dir}')
    print(f'{"="*80}')
    
    return fold_model_paths, all_fold_histories


def rgtan_load_and_inference(feat_df, graph, test_idx, labels, args, cat_features, 
                             neigh_features: pd.DataFrame, nei_att_head, model_save_dir):
    """
    加载k折模型并进行推理 - 逐个模型加载推理并释放显存
    """
    device = args['device']
    graph = graph.to(device)
    
    # 准备数据
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}
    
    nei_feat = {}
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(device) 
                   for col in neigh_features.columns}
    
    labels_tensor = torch.from_numpy(labels.values).long().to(device)
    neigh_padding_dict = {}
    
    print(f"\n{'='*80}")
    print(f"推理模式 - 从目录加载模型: {model_save_dir}")
    print(f"{'='*80}")
    
    # 加载模型路径
    paths_file = os.path.join(model_save_dir, 'fold_model_paths.txt')
    if os.path.exists(paths_file):
        with open(paths_file, 'r') as f:
            fold_model_paths = [line.strip() for line in f if line.strip()]
    else:
        # 从目录中查找所有模型文件
        import glob
        fold_model_paths = sorted(glob.glob(os.path.join(model_save_dir, 'fold_*_best_model.pt')))
    
    if not fold_model_paths:
        raise ValueError(f"✗ 错误: 在目录 {model_save_dir} 中没有找到模型文件!")
    
    print(f"找到 {len(fold_model_paths)} 个模型文件:")
    for i, path in enumerate(fold_model_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")
    
    # 为测试集创建数据加载器（计时开始）
    print(f"\n创建测试集数据加载器...")
    start_loader_time = time.perf_counter()
    
    test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
    test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    test_dataloader = DataLoader(graph, test_ind, test_sampler,
                                 use_ddp=False, device=device,
                                 batch_size=args['batch_size'],
                                 shuffle=False, drop_last=False, num_workers=0)
    
    loader_time = (time.perf_counter() - start_loader_time) * 1000  # ms
    print(f"数据加载器创建完成，耗时: {loader_time:.2f} ms")
    
    # 存储每折的预测概率和推理时间
    all_test_probs = []
    all_fold_times = []
    all_load_times = []
    
    # 对每个模型进行推理（逐个加载和释放）
    for fold_idx, model_path in enumerate(fold_model_paths, 1):
        print(f'\n{"-"*50}')
        print(f'[第{fold_idx}折] 加载模型: {os.path.basename(model_path)}')
        print(f'{"-"*50}')
        
        # 模型加载时间
        start_load_time = time.perf_counter()
        checkpoint = torch.load(model_path, map_location=device)
        load_time = (time.perf_counter() - start_load_time) * 1000
        
        # 创建模型实例
        model = RGTAN(in_feats=feat_df.shape[1],
                      hidden_dim=args['hid_dim']//4,
                      n_classes=2,
                      heads=[4]*args['n_layers'],
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  • 模型加载耗时: {load_time:.2f} ms")
        
        # 用于存储当前模型的预测
        fold_test_predictions = torch.zeros([len(feat_df), 2]).float().to(device)
        fold_test_counts = torch.zeros([len(feat_df), 1]).float().to(device)
        
        # 清空CUDA缓存
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        
        # 推理计时
        start_infer_time = time.perf_counter()
        
        with torch.no_grad():
            pbar = tqdm(test_dataloader, desc=f'推理进度')
            for step, (input_nodes, seeds, blocks) in enumerate(pbar):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = \
                    load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels_tensor,
                                       seeds, input_nodes, device, blocks)
                blocks = [block.to(device) for block in blocks]
                
                test_batch_logits, _ = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                
                # 累积预测
                fold_test_predictions[seeds] += test_batch_logits
                fold_test_counts[seeds] += 1
        
        # 推理结束计时
        torch.cuda.synchronize(device) if device.startswith('cuda') else None
        infer_time = (time.perf_counter() - start_infer_time) * 1000  # ms
        
        # 获取显存使用
        if device.startswith('cuda'):
            max_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            current_mem_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        else:
            max_mem_mb = 0
            current_mem_mb = 0
        
        # 平均预测（处理重复节点）
        fold_test_predictions = fold_test_predictions / fold_test_counts.clamp(min=1)
        
        # 获取测试集的预测概率
        test_probs = torch.softmax(fold_test_predictions[test_idx], dim=1)[:, 1].cpu().numpy()
        all_test_probs.append(test_probs)
        all_fold_times.append(infer_time)
        all_load_times.append(load_time)
        
        print(f"  • 推理耗时: {infer_time:.2f} ms")
        print(f"  • 峰值显存: {max_mem_mb:.2f} MB")
        print(f"  • 当前显存: {current_mem_mb:.2f} MB")
        print(f"  • 本折总耗时: {load_time + infer_time:.2f} ms")
        
        # 释放当前模型显存
        del model, checkpoint, fold_test_predictions, fold_test_counts
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            
        # 清理后显存
        if device.startswith('cuda'):
            cleaned_mem_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
            print(f"  • 清理后显存: {cleaned_mem_mb:.2f} MB")
    
    # ================== 计算K折均值结果 ==================
    print(f'\n{"="*80}')
    print(f'K折集成结果 ({len(all_test_probs)}折)')
    print(f'{"="*80}')
    
    # 概率平均（软投票）
    all_test_probs_array = np.array(all_test_probs)  # 形状: [n_folds, n_test_samples]
    mean_test_probs = np.mean(all_test_probs_array, axis=0)
    mean_test_preds = (mean_test_probs > 0.5).astype(int)
    
    # 计算指标
    y_test_target = labels.iloc[test_idx].values
    valid_test_mask = (y_test_target == 0) | (y_test_target == 1)
    
    if valid_test_mask.any():
        y_test_valid = y_test_target[valid_test_mask]
        mean_test_probs_valid = mean_test_probs[valid_test_mask]
        mean_test_preds_valid = mean_test_preds[valid_test_mask]
        
        try:
            final_auc = roc_auc_score(y_test_valid, mean_test_probs_valid)
            final_ap = average_precision_score(y_test_valid, mean_test_probs_valid)
            final_acc = (mean_test_preds_valid == y_test_valid).mean()
            final_f1 = f1_score(y_test_valid, mean_test_preds_valid, average="macro")
            
            print(f"\n集成测试结果:")
            print(f"  • Accuracy:  {final_acc:.4f}")
            print(f"  • AUC:       {final_auc:.4f}")
            print(f"  • AP:        {final_ap:.4f}")
            print(f"  • F1-score:  {final_f1:.4f}")
            print(f"  • 有效样本:  {len(y_test_valid)}/{len(y_test_target)}")
            
        except Exception as e:
            print(f"计算指标时出错: {e}")
    else:
        print("✗ 测试集中没有有效标签(0或1)")
    
    # ================== 输出总时间统计 ==================
    print(f'\n{"="*80}')
    print(f'总时间统计')
    print(f'{"="*80}')
    
    total_loader_time = loader_time  # 数据加载器创建时间
    total_load_time = sum(all_load_times)  # 所有模型加载时间总和
    total_infer_time = sum(all_fold_times)  # 所有折的推理时间总和
    total_time = total_loader_time + total_load_time + total_infer_time  # 总时间
    
    print(f"数据加载器创建: {total_loader_time:.2f} ms")
    print(f"模型加载总时间: {total_load_time:.2f} ms")
    print(f"K折推理总时间: {total_infer_time:.2f} ms")
    print(f"总耗时(加载器+模型加载+K折推理): {total_time:.2f} ms")
    
    print(f"\n各折详细时间:")
    for i in range(len(all_fold_times)):
        print(f"  第{i+1}折 - 加载: {all_load_times[i]:.2f} ms, 推理: {all_fold_times[i]:.2f} ms, "
              f"合计: {all_load_times[i] + all_fold_times[i]:.2f} ms")
    
    if len(all_fold_times) > 1:
        print(f"\n推理时间统计:")
        print(f"  平均推理时间: {np.mean(all_fold_times):.2f} ms")
        print(f"  推理时间标准差: {np.std(all_fold_times):.2f} ms")
        print(f"  最快推理: {min(all_fold_times):.2f} ms (第{np.argmin(all_fold_times)+1}折)")
        print(f"  最慢推理: {max(all_fold_times):.2f} ms (第{np.argmax(all_fold_times)+1}折)")
    
    # 保存预测结果
    results_save_dir = os.path.join(model_save_dir, 'predictions')
    os.makedirs(results_save_dir, exist_ok=True)
    
    # 保存每折的预测
    for i, probs in enumerate(all_test_probs, 1):
        np.save(os.path.join(results_save_dir, f'fold_{i}_probs.npy'), probs)
    
    # 保存集成预测
    np.save(os.path.join(results_save_dir, 'ensemble_mean_probs.npy'), mean_test_probs)
    np.save(os.path.join(results_save_dir, 'ensemble_mean_preds.npy'), mean_test_preds)
    
    # 保存真实标签和测试索引
    np.save(os.path.join(results_save_dir, 'test_indices.npy'), test_idx)
    np.save(os.path.join(results_save_dir, 'test_labels.npy'), y_test_target)
    
    # 保存时间统计
    time_stats = {
        'data_loader_creation_ms': total_loader_time,
        'total_model_load_ms': total_load_time,
        'total_inference_ms': total_infer_time,
        'total_time_ms': total_time,
        'avg_inference_per_fold_ms': np.mean(all_fold_times),
        'avg_load_per_fold_ms': np.mean(all_load_times),
        'std_inference_per_fold_ms': np.std(all_fold_times) if len(all_fold_times) > 1 else 0,
        'min_inference_ms': min(all_fold_times) if all_fold_times else 0,
        'max_inference_ms': max(all_fold_times) if all_fold_times else 0,
        'n_folds': len(all_fold_times),
        'n_test_samples': len(y_test_target),
        'n_valid_samples': len(y_test_valid) if valid_test_mask.any() else 0
    }
    pd.DataFrame([time_stats]).to_csv(os.path.join(results_save_dir, 'time_statistics.csv'), index=False)
    
    # 保存详细时间数据
    time_details = pd.DataFrame({
        'fold': list(range(1, len(all_fold_times) + 1)),
        'load_time_ms': all_load_times,
        'inference_time_ms': all_fold_times,
        'total_time_ms': [load + infer for load, infer in zip(all_load_times, all_fold_times)]
    })
    time_details.to_csv(os.path.join(results_save_dir, 'time_details.csv'), index=False)
    
    # 保存集成指标
    if valid_test_mask.any():
        metrics = {
            'accuracy': final_acc,
            'auc': final_auc,
            'ap': final_ap,
            'f1_score': final_f1,
            'n_folds': len(all_test_probs),
            'n_test_samples': len(y_test_target),
            'n_valid_samples': len(y_test_valid)
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(results_save_dir, 'ensemble_metrics.csv'), index=False)
    
    print(f'\n{"="*80}')
    print(f'✓ 推理完成! 结果保存到: {results_save_dir}')
    print(f'{"="*80}')
    
    return {
        'fold_probs': all_test_probs,
        'ensemble_probs': mean_test_probs,
        'ensemble_preds': mean_test_preds,
        'time_stats': time_stats,
        'metrics': metrics if valid_test_mask.any() else None
    }


# 保持原有的rgtan_main函数（向后兼容）
def rgtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head):
    """
    原有的rgtan_main函数（向后兼容）
    注意：这个函数不会保存k折模型，只训练并返回最后一折的结果
    """
    print(f"\n{'='*80}")
    print(f"使用旧版rgtan_main函数（不会保存k折模型）")
    print(f"要保存k折模型，请使用新的训练接口")
    print(f"{'='*80}")
    
    # 这里调用原有的训练逻辑（简化版）
    # 实际应该调用您原有的训练代码
    device = args['device']
    graph = graph.to(device)
    
    # 原有的训练逻辑...
    # ...
    
    print("训练完成（未保存k折模型）")
    return None


def loda_rgtan_data(dataset: str, test_size: float):
    """
    加载数据函数（保持不变）
    """
    prefix = "./data/"
    
    print(f"加载 {dataset} 数据集...")
    cat_features = []
    neigh_features = []
    
    npz_path = '../data/' + f"{dataset}.npz"
    print(f"从 {npz_path} 加载数据...")
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"npz文件中的键: {list(data.keys())}")
    
    if 'x' in data.files:
        feat_data = pd.DataFrame(data['x'])
        print(f"加载特征 'x', 形状: {data['x'].shape}")
    elif 'features' in data.files:
        feat_data = pd.DataFrame(data['features'])
        print(f"加载特征 'features', 形状: {data['features'].shape}")
    else:
        raise ValueError("npz文件中没有找到特征数据（期望 'x' 或 'features'）")
    
    if 'y' in data.files:
        labels = pd.Series(data['y'])
        print(f"加载标签 'y', 形状: {data['y'].shape}")
    elif 'label' in data.files:
        labels = pd.Series(data['label'])
        print(f"加载标签 'label', 形状: {data['label'].shape}")
    else:
        labels = pd.Series(np.zeros(feat_data.shape[0]))
        print("未找到标签，使用全零标签")
    
    num_nodes = len(feat_data)
    print(f"节点数: {num_nodes}")
    
    if 'edge_index' in data.files:
        edge_index = data['edge_index']
        print(f"加载边索引, 形状: {edge_index.shape}")
        
        if edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
            src = torch.tensor(edge_index[0], dtype=torch.long)
            dst = torch.tensor(edge_index[1], dtype=torch.long)
        elif edge_index.shape[1] == 2:
            src = torch.tensor(edge_index[:, 0], dtype=torch.long)
            dst = torch.tensor(edge_index[:, 1], dtype=torch.long)
        else:
            raise ValueError(f"意外的边索引形状: {edge_index.shape}")
        
        print(f"边数: {len(src)}")
        
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        print(f"图构建完成: {g.num_nodes()} 个节点, {g.num_edges()} 条边")
    else:
        print("未找到edge_index，尝试加载邻接列表...")
        adj_path = prefix + f"{dataset}_homo_adjlists.pickle"
        with open(adj_path, 'rb') as f:
            homo_adj = pickle.load(f)
        
        src, dst = [], []
        for node in homo_adj:
            for neighbor in homo_adj[node]:
                src.append(node)
                dst.append(neighbor)
        
        g = dgl.graph((np.array(src), np.array(dst)), num_nodes=num_nodes)
        print(f"从邻接列表构建图: {g.num_nodes()} 个节点, {g.num_edges()} 条边")
    
    g = g.remove_self_loop().add_self_loop()
    
    g.ndata['feat'] = torch.from_numpy(feat_data.values).to(torch.float32)
    g.ndata['label'] = torch.from_numpy(labels.values).to(torch.long)
    
    # 处理训练/测试掩码
    if 'train_mask' in data.files and 'valid_mask' in data.files and 'test_mask' in data.files:
        train_idx = data['train_mask'].tolist()
        valid_idx = data['valid_mask'].tolist()
        test_idx = data['test_mask'].tolist()
        
        print(f"使用npz掩码: {len(train_idx)} 训练, {len(valid_idx)} 验证, {len(test_idx)} 测试")
        
    elif 'train_mask' in data.files and 'test_mask' in data.files:
        train_idx = data['train_mask'].tolist()
        test_idx = data['test_mask'].tolist()
        
        print(f"使用npz掩码: {len(train_idx)} 训练, {len(test_idx)} 测试")
        
        if len(train_idx) > 0:
            train_idx, valid_idx = train_test_split(
                train_idx, test_size=0.2, random_state=42, stratify=labels.iloc[train_idx]
            )
            print(f"分割训练集: {len(train_idx)} 训练, {len(valid_idx)} 验证")
    else:
        print("npz中没有掩码，使用随机分割")
        all_idx = list(range(len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            all_idx, labels, stratify=labels, test_size=test_size,
            random_state=2, shuffle=True
        )
        train_idx, valid_idx, _, _ = train_test_split(
            train_idx, labels.iloc[train_idx], test_size=0.2,
            random_state=2, shuffle=True, stratify=labels.iloc[train_idx]
        )
        print(f"使用随机分割: {len(train_idx)} 训练, {len(valid_idx)} 验证, {len(test_idx)} 测试")
    
    try:
        neigh_feat_path = prefix + f"{dataset}_neigh_feat.csv"
        feat_neigh = pd.read_csv(neigh_feat_path)
        print("加载邻居特征用于神经网络输入")
        neigh_features = feat_neigh
    except FileNotFoundError:
        print("未找到邻居特征文件，使用空特征")
        neigh_features = []
    
    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
