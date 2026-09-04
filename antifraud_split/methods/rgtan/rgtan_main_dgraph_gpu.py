from __future__ import annotations

import os
import time  # 新增
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


def rgtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head):
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = []
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    print(f"\n{'='*80}")
    print("DEBUG: Data Info")
    print(f"{'='*80}")
    print(f"Number of nodes: {len(feat_df)}")
    print(f"Number of training nodes: {len(train_idx)}")
    print(f"Feature shape: {feat_df.shape}")
    print(f"Labels distribution: {pd.Series(y.values).value_counts().to_dict()}")
    print(f"Labels in training set: {pd.Series(y_target).value_counts().to_dict()}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # 用于存储每个fold的历史记录
    all_fold_histories = []
    # ===== 新增：记录每折测试总推理时长 =====
    fold_infer_time = []
    # ======================================
    
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'\n{"="*80}')
        print(f'Training fold {fold + 1}/{args["n_fold"]}')
        print(f'{"="*80}')
        
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
            device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = DataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = DataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
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
        
        # 打印模型信息
        print(f"\nModel Info:")
        print(f"Input features: {feat_df.shape[1]}")
        print(f"Hidden dim: {args['hid_dim']//4}")
        print(f"Number of layers: {args['n_layers']}")
        print(f"Heads: {[4]*args['n_layers']}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                   4000, 12000], gamma=0.3)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, 2000
        
        # 用于记录每个epoch的指标
        fold_history = {
            'train_loss': [],
            'train_acc': [],
            'train_auc': [],
            'train_ap': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_ap': [],
            'epochs': []
        }
        
        for epoch in range(start_epoch, args['max_epochs']):
            # 训练阶段
            model.train()
            train_loss_list = []
            train_acc_list = []
            train_auc_list = []
            train_ap_list = []
            train_total_samples = 0
            train_valid_samples = 0
            
            # 创建训练进度条
            train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args["max_epochs"]} [Train]', 
                             bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            
            for step, (input_nodes, seeds, blocks) in enumerate(train_pbar):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                
                # 只获取logits，忽略模型返回的valid_mask
                train_batch_logits, _ = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                
                # 直接使用batch_labels计算valid_mask
                seeds_valid_mask = (batch_labels == 0) | (batch_labels == 1)
                
                train_total_samples += len(batch_labels)
                train_valid_samples += seeds_valid_mask.sum().item()
                
                if seeds_valid_mask.any():
                    train_batch_logits_valid = train_batch_logits[seeds_valid_mask]
                    batch_labels_valid = batch_labels[seeds_valid_mask]
                    
                    if train_batch_logits_valid.shape[0] > 0 and batch_labels_valid.shape[0] > 0:
                        train_loss = loss_fn(train_batch_logits_valid, batch_labels_valid)
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        
                        train_loss_list.append(train_loss.cpu().detach().numpy())
                        
                        # 计算训练指标
                        with torch.no_grad():
                            train_preds = torch.argmax(train_batch_logits_valid, dim=1)
                            train_acc = (train_preds == batch_labels_valid).float().mean().item()
                            train_acc_list.append(train_acc)
                            
                            try:
                                if len(torch.unique(batch_labels_valid)) > 1:
                                    train_scores = torch.softmax(train_batch_logits_valid, dim=1)[:, 1].cpu().numpy()
                                    train_auc = roc_auc_score(batch_labels_valid.cpu().numpy(), train_scores)
                                    train_ap = average_precision_score(batch_labels_valid.cpu().numpy(), train_scores)
                                    train_auc_list.append(train_auc)
                                    train_ap_list.append(train_ap)
                                else:
                                    train_auc_list.append(0.5)
                                    train_ap_list.append(0.5)
                            except Exception as e:
                                train_auc_list.append(0.5)
                                train_ap_list.append(0.5)
                        
                        # 更新进度条
                        if step % 5 == 0 or step == len(train_dataloader) - 1:
                            avg_train_loss = np.mean(train_loss_list[-min(10, len(train_loss_list)):]) if train_loss_list else 0
                            avg_train_acc = np.mean(train_acc_list[-min(10, len(train_acc_list)):]) if train_acc_list else 0
                            avg_train_auc = np.mean(train_auc_list[-min(10, len(train_auc_list)):]) if train_auc_list else 0.5
                            train_pbar.set_postfix({
                                'loss': f'{avg_train_loss:.4f}',
                                'acc': f'{avg_train_acc:.4f}',
                                'auc': f'{avg_train_auc:.4f}',
                                'valid%': f'{100*train_valid_samples/train_total_samples:.1f}%'
                            })
                
                else:
                    if step == 0 and epoch == 0:
                        print(f"WARNING: No valid labels (0 or 1) in first batch!")
                        print(f"Batch labels distribution: {torch.bincount(batch_labels)}")
            
            # 计算训练集平均指标
            avg_train_loss = np.mean(train_loss_list) if train_loss_list else 0
            avg_train_acc = np.mean(train_acc_list) if train_acc_list else 0
            avg_train_auc = np.mean(train_auc_list) if train_auc_list else 0.5
            avg_train_ap = np.mean(train_ap_list) if train_ap_list else 0.5
            
            # 验证阶段
            model.eval()
            val_loss_list = []
            val_acc_list = []
            val_auc_list = []
            val_ap_list = []
            val_total_samples = 0
            val_valid_samples = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{args["max_epochs"]} [Val]',
                               bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
                
                for step, (input_nodes, seeds, blocks) in enumerate(val_pbar):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                           seeds, input_nodes, device, blocks)

                    blocks = [block.to(device) for block in blocks]
                    
                    val_batch_logits, _ = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                    oof_predictions[seeds] = val_batch_logits
                    
                    # 直接使用batch_labels计算valid_mask
                    seeds_valid_mask = (batch_labels == 0) | (batch_labels == 1)
                    
                    val_total_samples += len(batch_labels)
                    val_valid_samples += seeds_valid_mask.sum().item()
                    
                    if seeds_valid_mask.any():
                        val_batch_logits_valid = val_batch_logits[seeds_valid_mask]
                        batch_labels_valid = batch_labels[seeds_valid_mask]
                        
                        if val_batch_logits_valid.shape[0] > 0 and batch_labels_valid.shape[0] > 0:
                            val_loss = loss_fn(val_batch_logits_valid, batch_labels_valid)
                            val_loss_list.append(val_loss.cpu().detach().numpy())
                            
                            # 计算验证指标
                            val_preds = torch.argmax(val_batch_logits_valid, dim=1)
                            val_acc = (val_preds == batch_labels_valid).float().mean().item()
                            val_acc_list.append(val_acc)
                            
                            try:
                                if len(torch.unique(batch_labels_valid)) > 1:
                                    val_scores = torch.softmax(val_batch_logits_valid, dim=1)[:, 1].cpu().numpy()
                                    val_auc = roc_auc_score(batch_labels_valid.cpu().numpy(), val_scores)
                                    val_ap = average_precision_score(batch_labels_valid.cpu().numpy(), val_scores)
                                    val_auc_list.append(val_auc)
                                    val_ap_list.append(val_ap)
                                else:
                                    val_auc_list.append(0.5)
                                    val_ap_list.append(0.5)
                            except Exception as e:
                                val_auc_list.append(0.5)
                                val_ap_list.append(0.5)
                            
                            # 更新验证进度条
                            if step % 5 == 0 or step == len(val_dataloader) - 1:
                                avg_val_loss = np.mean(val_loss_list[-min(10, len(val_loss_list)):]) if val_loss_list else 0
                                avg_val_acc = np.mean(val_acc_list[-min(10, len(val_acc_list)):]) if val_acc_list else 0
                                avg_val_auc = np.mean(val_auc_list[-min(10, len(val_auc_list)):]) if val_auc_list else 0.5
                                val_pbar.set_postfix({
                                    'loss': f'{avg_val_loss:.4f}',
                                    'acc': f'{avg_val_acc:.4f}',
                                    'auc': f'{avg_val_auc:.4f}',
                                    'valid%': f'{100*val_valid_samples/val_total_samples:.1f}%'
                                })
            
            # 计算验证集平均指标
            avg_val_loss = np.mean(val_loss_list) if val_loss_list else 0
            avg_val_acc = np.mean(val_acc_list) if val_acc_list else 0
            avg_val_auc = np.mean(val_auc_list) if val_auc_list else 0.5
            avg_val_ap = np.mean(val_ap_list) if val_ap_list else 0.5
            
            # 记录历史
            fold_history['epochs'].append(epoch + 1)
            fold_history['train_loss'].append(avg_train_loss)
            fold_history['train_acc'].append(avg_train_acc)
            fold_history['train_auc'].append(avg_train_auc)
            fold_history['train_ap'].append(avg_train_ap)
            fold_history['val_loss'].append(avg_val_loss)
            fold_history['val_acc'].append(avg_val_acc)
            fold_history['val_auc'].append(avg_val_auc)
            fold_history['val_ap'].append(avg_val_ap)
            
            # 打印epoch总结
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Train - Loss: {avg_train_loss:.6f}, Acc: {avg_train_acc:.4f}, AUC: {avg_train_auc:.4f}, AP: {avg_train_ap:.4f}')
            print(f'      Valid samples: {train_valid_samples}/{train_total_samples} ({100*train_valid_samples/train_total_samples:.1f}%)')
            print(f'Val   - Loss: {avg_val_loss:.6f}, Acc: {avg_val_acc:.4f}, AUC: {avg_val_auc:.4f}, AP: {avg_val_ap:.4f}')
            print(f'      Valid samples: {val_valid_samples}/{val_total_samples} ({100*val_valid_samples/val_total_samples:.1f}%)')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 如果所有指标都是0，可能是数据有问题
            if epoch == 0 and avg_train_loss == 0 and avg_val_loss == 0:
                print(f"\n⚠️  WARNING: All losses are 0! Possible issues:")
                print(f"  1. No valid labels (0 or 1) in the data")
                print(f"  2. Model output is constant")
                print(f"  3. Loss calculation is incorrect")
                print(f"  4. Gradient flow is broken")
            
            # Early stopping
            earlystoper.earlystop(avg_val_loss, model)
            if earlystoper.is_earlystop:
                print(f"\n⚠️  Early Stopping at epoch {epoch+1}!")
                print(f"Best validation loss: {earlystoper.best_cv:.6f}")
                break
        
        # 保存fold历史
        all_fold_histories.append(fold_history)
        
        # 打印fold总结
        print(f'\n{"="*60}')
        print(f'Fold {fold+1} Completed')
        print(f'Best val_loss: {earlystoper.best_cv:.6f}')
        print(f'Total epochs trained: {len(fold_history["epochs"])}')
        
        # 测试阶段
        print(f'\n{"="*60}')
        print('Testing...')
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = DataLoader(graph,
                                     test_ind,
                                     test_sampler,
                                     use_ddp=False,
                                     device=device,
                                     batch_size=args['batch_size'],
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=0)
        b_model = earlystoper.best_model.to(device)
        b_model.eval()

        test_all_preds = []
        test_all_labels = []
        test_all_scores = []

        # ---------- 新增：累计总时长 & 显存 ----------
        total_infer_ms = 0.0
        max_mem_mb = 0.0
        torch.cuda.reset_peak_memory_stats(device)
        # -------------------------------------------

        with torch.no_grad():
            # 使用 tqdm 显示进度条
            for step, (input_nodes, seeds, blocks) in enumerate(
                tqdm(test_dataloader, desc=f'Testing fold{fold+1}'), start=1
            ):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = \
                    load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                       seeds, input_nodes, device, blocks)
                blocks = [block.to(device) for block in blocks]

                # ---------- 计时 & 显存（每 batch） ----------
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                test_batch_logits, _ = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                
                total_infer_ms += (time.perf_counter() - start) * 1000  # ms
                
                torch.cuda.synchronize(device)
                max_mem_mb = max(max_mem_mb, torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                # ----------------------------------------------

                test_predictions[seeds] = test_batch_logits

                seeds_valid_mask = (batch_labels == 0) | (batch_labels == 1)
                if seeds_valid_mask.any():
                    test_batch_logits_valid = test_batch_logits[seeds_valid_mask]
                    batch_labels_valid = batch_labels[seeds_valid_mask]
                    if test_batch_logits_valid.shape[0] > 0 and batch_labels_valid.shape[0] > 0:
                        test_preds = torch.argmax(test_batch_logits_valid, dim=1)
                        test_scores = torch.softmax(test_batch_logits_valid, dim=1)[:, 1]
                        test_all_preds.append(test_preds.cpu().numpy())
                        test_all_labels.append(batch_labels_valid.cpu().numpy())
                        test_all_scores.append(test_scores.cpu().numpy())

        # ---------- 记录本折总时长 & 峰值显存 ----------
        fold_infer_time.append(total_infer_ms)
        print(f"[Inference] 测试集（fold{fold+1}）总推理时长: {total_infer_ms:.2f} ms  显存峰值: {max_mem_mb:.2f} MB")
        # ----------------------------------------------

        if test_all_preds:
            test_all_preds = np.concatenate(test_all_preds)
            test_all_labels = np.concatenate(test_all_labels)
            test_all_scores = np.concatenate(test_all_scores)
            try:
                test_auc = roc_auc_score(test_all_labels, test_all_scores)
                test_ap = average_precision_score(test_all_labels, test_all_scores)
                test_acc = (test_all_preds == test_all_labels).mean()
                test_f1 = f1_score(test_all_labels, test_all_preds, average="macro")
                print(f'\nTest Results for Fold {fold+1}:')
                print(f'Test Acc:  {test_acc:.4f}')
                print(f'Test AUC:  {test_auc:.4f}')
                print(f'Test AP:   {test_ap:.4f}')
                print(f'Test F1:   {test_f1:.4f}')
            except Exception as e:
                print(f'Error calculating test metrics: {e}')
    
    # 最终汇总
    print(f'\n{"="*60}')
    print('TRAINING COMPLETED')
    print(f'{"="*60}')
    
    # 计算验证集指标
    print(f'\n{"-"*60}')
    print('VALIDATION SET RESULTS')
    print(f'{"-"*60}')
    
    if isinstance(labels, torch.Tensor):
        y_val_target = labels[train_idx].cpu().numpy()
    else:
        y_val_target = labels.iloc[train_idx].cpu().numpy()
        
    val_pred = torch.softmax(oof_predictions, dim=1).cpu()[train_idx, 1].numpy()
    val_pred_labels = torch.argmax(oof_predictions, dim=1).cpu()[train_idx].numpy()
    
    valid_val_mask = (y_val_target == 0) | (y_val_target == 1)
    if valid_val_mask.any():
        y_val_target_valid = y_val_target[valid_val_mask]
        val_pred_valid = val_pred[valid_val_mask]
        val_pred_labels_valid = val_pred_labels[valid_val_mask]
        
        try:
            val_auc = roc_auc_score(y_val_target_valid, val_pred_valid)
            val_ap = average_precision_score(y_val_target_valid, val_pred_valid)
            val_acc = (val_pred_labels_valid == y_val_target_valid).mean()
            val_f1 = f1_score(y_val_target_valid, val_pred_labels_valid, average="macro")
            
            print(f'Val Acc:  {val_acc:.4f}')
            print(f'Val AUC:  {val_auc:.4f}')
            print(f'Val AP:   {val_ap:.4f}')
            print(f'Val F1:   {val_f1:.4f}')
            print(f'Val samples: {len(y_val_target_valid)}/{len(y_val_target)} valid samples')
        except Exception as e:
            print(f'Error calculating validation metrics: {e}')
    else:
        print("No valid labels (0 or 1) in validation set")
    
    if isinstance(labels, torch.Tensor):
        y_target = labels[train_idx].cpu().numpy()
    else:
        y_target = labels.iloc[train_idx].cpu().numpy()
        
    oof_pred = torch.softmax(oof_predictions, dim=1).cpu()[train_idx, 1].numpy()

    valid_train_mask = (y_target == 0) | (y_target == 1)
    if valid_train_mask.any():
        my_ap = average_precision_score(y_target[valid_train_mask], oof_pred[valid_train_mask])
        my_auc = roc_auc_score(y_target[valid_train_mask], oof_pred[valid_train_mask])
        print(f'\n{"-"*60}')
        print('OVERALL OOF RESULTS')
        print(f'{"-"*60}')
        print(f'OOF AUC: {my_auc:.4f}')
        print(f'OOF AP:  {my_ap:.4f}')
    else:
        print("No valid labels (0 or 1) in training set for OOF calculation")
    
    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to('cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    
    if isinstance(labels, torch.Tensor):
        y_target = labels[test_idx].cpu().numpy()
    else:
        y_target = labels.iloc[test_idx].cpu().numpy()
        
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    valid_test_mask = (y_target == 0) | (y_target == 1)
    if valid_test_mask.any():
        test_score = test_score[valid_test_mask]
        y_target_valid = y_target[valid_test_mask]
        test_score1 = test_score1[valid_test_mask]

        final_test_auc = roc_auc_score(y_target_valid, test_score)
        final_test_ap = average_precision_score(y_target_valid, test_score)
        final_test_acc = (test_score1 == y_target_valid).mean()
        final_test_f1 = f1_score(y_target_valid, test_score1, average="macro")
        
        print(f'\n{"-"*60}')
        print('FINAL TEST RESULTS')
        print(f'{"-"*60}')
        print(f'Test Acc:  {final_test_acc:.4f}')
        print(f'Test AUC:  {final_test_auc:.4f}')
        print(f'Test AP:   {final_test_ap:.4f}')
        print(f'Test F1:   {final_test_f1:.4f}')
    else:
        print("No valid labels (0 or 1) in test set")
    
    # ================== 新增：每折推理时长汇总 ==================
    print(f'\n{"="*60}')
    print('TEST INFERENCE TIME SUMMARY')
    print(f'{"="*60}')
    for fold_idx, t in enumerate(fold_infer_time, 1):
        print(f'Fold {fold_idx}: {t:.2f} ms')
    print(f'Total across {len(fold_infer_time)} folds: {sum(fold_infer_time):.2f} ms')
    # ==========================================================
    
    print(f'\n{"-"*60}')
    print('FOLD HISTORY SUMMARY')
    print(f'{"-"*60}')
    
    for fold_idx, fold_history in enumerate(all_fold_histories):
        print(f'\nFold {fold_idx + 1}:')
        if fold_history['epochs']:
            last_epoch = fold_history['epochs'][-1]
            print(f'  Epochs trained: {last_epoch}')
            print(f'  Best val_loss: {min(fold_history["val_loss"]) if fold_history["val_loss"] else "N/A":.6f}')
            print(f'  Final train_loss: {fold_history["train_loss"][-1] if fold_history["train_loss"] else "N/A":.6f}')
            print(f'  Final val_loss: {fold_history["val_loss"][-1] if fold_history["val_loss"] else "N/A":.6f}')
    
    return all_fold_histories


def loda_rgtan_data(dataset: str, test_size: float):
    prefix = "./data/"
    
    if 1:
        print(f"Loading {dataset} dataset...")
        cat_features = []
        neigh_features = []
        
        npz_path = '../data/' + f"{dataset}.npz"
        print(f"Loading data from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"Keys in npz file: {list(data.keys())}")
        
        if 'x' in data.files:
            feat_data = pd.DataFrame(data['x'])
            print(f"Loaded features 'x' with shape: {data['x'].shape}")
        elif 'features' in data.files:
            feat_data = pd.DataFrame(data['features'])
            print(f"Loaded features 'features' with shape: {data['features'].shape}")
        else:
            raise ValueError("No feature data found in npz file (expected 'x' or 'features')")
        
        if 'y' in data.files:
            labels = pd.Series(data['y'])
            print(f"Loaded labels 'y' with shape: {data['y'].shape}")
        elif 'label' in data.files:
            labels = pd.Series(data['label'])
            print(f"Loaded labels 'label' with shape: {data['label'].shape}")
        else:
            labels = pd.Series(np.zeros(feat_data.shape[0]))
            print("No labels found, using zeros")
        
        num_nodes = len(feat_data)
        print(f"Number of nodes: {num_nodes}")
        
        if 'edge_index' in data.files:
            edge_index = data['edge_index']
            print(f"Loaded edge_index with shape: {edge_index.shape}")
            
            if edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
                src = torch.tensor(edge_index[0], dtype=torch.long)
                dst = torch.tensor(edge_index[1], dtype=torch.long)
            elif edge_index.shape[1] == 2:
                src = torch.tensor(edge_index[:, 0], dtype=torch.long)
                dst = torch.tensor(edge_index[:, 1], dtype=torch.long)
            else:
                raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")
            
            print(f"Edges: {len(src)} edges loaded")
            
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            print(f"Graph built: {g.num_nodes()} nodes, {g.num_edges()} edges")
        else:
            print("No edge_index found, trying to load adjacency list...")
            adj_path = prefix + f"{dataset}_homo_adjlists.pickle"
            with open(adj_path, 'rb') as f:
                homo_adj = pickle.load(f)
            
            src, dst = [], []
            for node in homo_adj:
                for neighbor in homo_adj[node]:
                    src.append(node)
                    dst.append(neighbor)
            
            g = dgl.graph((np.array(src), np.array(dst)), num_nodes=num_nodes)
            print(f"Graph built from adjlist: {g.num_nodes()} nodes, {g.num_edges()} edges")
        
        g = g.remove_self_loop().add_self_loop()
        
        g.ndata['feat'] = torch.from_numpy(feat_data.values).to(torch.float32)
        g.ndata['label'] = torch.from_numpy(labels.values).to(torch.long)
        
        # 处理训练/测试掩码 - 假设是索引数组
        if 'train_mask' in data.files and 'valid_mask' in data.files and 'test_mask' in data.files:
            train_idx = data['train_mask'].tolist()
            valid_idx = data['valid_mask'].tolist()
            test_idx = data['test_mask'].tolist()
            
            print(f"Using npz index masks: {len(train_idx)} train, {len(valid_idx)} valid, {len(test_idx)} test")
            
        elif 'train_mask' in data.files and 'test_mask' in data.files:
            train_idx = data['train_mask'].tolist()
            test_idx = data['test_mask'].tolist()
            
            print(f"Using npz index masks: {len(train_idx)} train, {len(test_idx)} test")
            
            if len(train_idx) > 0:
                train_idx, valid_idx = train_test_split(
                    train_idx, test_size=0.2, random_state=42, stratify=labels.iloc[train_idx]
                )
                print(f"Split train set: {len(train_idx)} train, {len(valid_idx)} valid")
        else:
            print("No masks in npz, using random split")
            all_idx = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                all_idx, labels, stratify=labels, test_size=test_size,
                random_state=2, shuffle=True
            )
            train_idx, valid_idx, _, _ = train_test_split(
                train_idx, labels.iloc[train_idx], test_size=0.2,
                random_state=2, shuffle=True, stratify=labels.iloc[train_idx]
            )
            print(f"Using random split: {len(train_idx)} train, {len(valid_idx)} valid, {len(test_idx)} test")
        
        try:
            neigh_feat_path = prefix + f"{dataset}_neigh_feat.csv"
            feat_neigh = pd.read_csv(neigh_feat_path)
            print("Neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except FileNotFoundError:
            print("No neighborhood feature file found, using empty features.")
            neigh_features = []
        
        return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
    
    elif dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        neigh_features = []
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        alls, allt = [], []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        
        g = dgl.graph((np.array(alls), np.array(allt)))
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.6,
                                                                random_state=2, shuffle=True)
        feat_neigh = pd.read_csv(prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src, tgt = [], []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        g = dgl.graph((np.array(src), np.array(tgt)))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")
