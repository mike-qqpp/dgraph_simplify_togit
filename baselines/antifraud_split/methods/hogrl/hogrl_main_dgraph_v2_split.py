import random
import torch
import os
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from .hogrl_model import *
from .hogrl_utils_dgraph_v2 import *
import time
import torch.cuda as cuda
import json
import pickle

def test(idx_eval, y_eval, gnn_model, feat_data, edge_indexs):
    """测试函数"""
    gnn_model.eval()
    logits, _ = gnn_model(feat_data, edge_indexs)
    x_softmax = torch.exp(logits).cpu().detach()
    positive_class_probs = x_softmax[:, 1].numpy()[np.array(idx_eval)]
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)

    return auc_score, ap_score, f1_score_val, g_mean

def test_with_timing_batch(idx_eval, y_eval, gnn_model, feat_data, edge_indexs, device, batch_size):
    """带有时间统计的测试函数 - 按batch_size进行minibatch推理"""
    gnn_model.eval()
    
    # 清空GPU缓存，确保准确测量
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # 记录推理开始时间
    start_time = time.time()
    
    # 执行推理（整个数据集）
    with torch.no_grad():
        logits, _ = gnn_model(feat_data, edge_indexs)
    
    # 提取特定数据集的预测结果
    x_softmax = torch.exp(logits).cpu().detach()
    positive_class_probs = x_softmax[:, 1].numpy()[np.array(idx_eval)]
    
    # 记录推理结束时间
    inference_time_ms = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 计算指标
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)
    
    # 记录峰值GPU内存使用量
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换为MB
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0
    
    return auc_score, ap_score, f1_score_val, g_mean, inference_time_ms, peak_memory

def test_with_minibatch_timing(idx_eval, y_eval, gnn_model, feat_data, edge_indexs, device, batch_size):
    """按minibatch进行推理并统计时间"""
    gnn_model.eval()
    
    # 清空GPU缓存，确保准确测量
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # 记录推理开始时间
    start_time = time.time()
    
    # 按batch进行推理
    all_logits = []
    num_batches = int(len(idx_eval) / batch_size) + 1
    
    for batch in range(num_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(idx_eval))
        if i_start >= i_end:
            continue
            
        batch_nodes = idx_eval[i_start:i_end]
        
        with torch.no_grad():
            # 注意：即使只计算部分节点的输出，GNN仍需要整个图进行计算
            logits, _ = gnn_model(feat_data, edge_indexs)
            batch_logits = logits[batch_nodes]
            all_logits.append(batch_logits.cpu())
    
    # 记录推理结束时间
    inference_time_ms = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 合并所有batch的结果
    all_logits_tensor = torch.cat(all_logits, dim=0)
    x_softmax = torch.exp(all_logits_tensor).cpu().detach()
    positive_class_probs = x_softmax[:, 1].numpy()
    
    # 计算指标
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)
    
    # 记录峰值GPU内存使用量
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换为MB
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0
    
    return auc_score, ap_score, f1_score_val, g_mean, inference_time_ms, peak_memory

def print_evaluation_results(split_name, auc, ap, f1, g_mean, inference_time=None, peak_memory=None):
    """打印评估结果"""
    print(f"  {split_name} AUC: {auc:.4f}")
    print(f"  {split_name} AP:  {ap:.4f}")
    print(f"  {split_name} F1:  {f1:.4f}")
    print(f"  {split_name} G-mean: {g_mean:.4f}")
    if inference_time is not None:
        print(f"  {split_name} 推理时间: {inference_time:.2f}毫秒")
    if peak_memory is not None and peak_memory > 0:
        print(f"  {split_name} 峰值GPU内存: {peak_memory:.2f} MB")

def save_model_checkpoint(model, epoch, val_auc, save_path, optimizer=None):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_auc': val_auc
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"✓ 模型检查点已保存: {save_path}")
    return save_path

def load_model_checkpoint(model, checkpoint_path, device='cuda', optimizer=None):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ 模型已从 {checkpoint_path} 加载")
    print(f"  检查点信息 - 轮次: {checkpoint.get('epoch', 'N/A')}, Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model

def save_training_results(results, save_path):
    """保存训练结果"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ 训练结果已保存: {save_path}")

def hogrl_train(args):
    """HOGRL训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 打印GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"GPU: {gpu_name}, 显存: {gpu_memory:.2f} GB")
    
    print('加载数据...')
    
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    print(f"路径: {prefix}, 内容: {os.listdir(prefix) if os.path.exists(prefix) else '路径不存在'}")
    
    # 设置随机种子
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    
    # 加载数据
    edge_indexs, feat_data, labels = load_data(args['dataset'], args['layers_tree'], prefix)
    
    print(f"加载完成: 特征形状={feat_data.shape}, 标签数={len(labels)}")
    
    # 检查边索引格式
    print(f"\n检查边索引格式...")
    for i, relation_data in enumerate(edge_indexs):
        print(f"  关系{i+1}: 基础边索引形状={relation_data[0].shape}")
        for j, tree in enumerate(relation_data[1]):
            print(f"    树{j+1}形状: {tree.shape}")
    
    # 数据分割
    if args['dataset'] is not None:
        print('load data of way 1 !!!!!!!!!')
        # DGraphFin格式数据集：使用数据集中预定义的掩码
        npz_path = f"../data/{args['dataset']}.npz"
        npz = np.load(npz_path)
        
        # 掩码是索引数组，不是布尔掩码
        idx_train = npz['train_mask'].tolist()
        idx_val = npz['valid_mask'].tolist()
        idx_test = npz['test_mask'].tolist()
        
        # 获取对应的标签
        y_train = labels[idx_train].tolist()
        y_val = labels[idx_val].tolist()
        y_test = labels[idx_test].tolist()
        
        print(f"{args['dataset']}数据集分割:")
        print(f"  训练集: {len(idx_train)} 个样本 ({sum(y_train)} 正样本)")
        print(f"  验证集: {len(idx_val)} 个样本 ({sum(y_val)} 正样本)")
        print(f"  测试集: {len(idx_test)} 个样本 ({sum(y_test)} 正样本)")
    else:
        raise ValueError(f"不支持的数据集: {args['dataset']}. 支持的数据集: 'dgraphfin', 'dgraph_exp130_nona', 'yelp', 'amazon'")
    
    # 训练集正负样本分割
    train_pos, train_neg = pos_neg_split(idx_train, y_train)
    print(f"训练集正负样本: {len(train_pos)} 正, {len(train_neg)} 负")

    # 初始化模型
    gnn_model = multi_HOGRL_Model(
        feat_data.shape[1], 2, len(edge_indexs), args['emb_size'], 
        args['drop_rate'], args['weight'], args['layers'], args['layers_tree']
    ).to(device)
    
    # 将数据移动到设备
    processed_edge_indexs = []
    for relation_data in edge_indexs:
        base_edge_index = relation_data[0].to(device)
        tree_list = [tree.to(device) for tree in relation_data[1]]
        processed_edge_indexs.append([base_edge_index, tree_list])
    
    # 特征数据
    feat_data_tensor = torch.tensor(feat_data).float().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.get('lr', 0.005), weight_decay=args.get('weight_decay', 5e-5))
    batch_size = args['batch_size']
    
    # 存储最佳模型信息
    best_val_auc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # 存储最佳模型的评估结果
    best_train_results = None
    best_val_results = None
    best_test_results = None
    
    # 创建保存目录
    model_save_dir = args.get('model_save_dir', './saved_models/')
    os.makedirs(os.path.join(model_save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, 'results'), exist_ok=True)
    
    print(f'\n开始训练...')
    print(f'模型将保存到: {model_save_dir}')
    
    total_training_time = 0
    start_training_time = time.time()
    
    for epoch in range(args['num_epochs']):
        epoch_start_time = time.time()
        gnn_model.train()
        total_loss = 0
        
        # 随机欠采样负样本
        sampled_idx_train = undersample(train_pos, train_neg, scale=1)
        rd.shuffle(sampled_idx_train)

        num_batches = int(len(sampled_idx_train) / batch_size) + 1
        for batch in range(num_batches):
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(sampled_idx_train))
            if i_start >= i_end:
                continue
                
            batch_nodes = sampled_idx_train[i_start:i_end]
            batch_label = torch.tensor(labels[np.array(batch_nodes)]).long().to(device)
            
            optimizer.zero_grad()
            out, _ = gnn_model(feat_data_tensor, processed_edge_indexs)
            
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=device)
            loss = F.nll_loss(out[batch_nodes_tensor], batch_label)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        # 每10个epoch验证一次
        if epoch % 10 == 9:
            # 验证集评估
            val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs)
            
            print(f'Epoch: {epoch:3d}, Loss: {total_loss/num_batches:.4f}, Val AUC: {val_auc:.4f}, Epoch Time: {epoch_time:.2f}s')
            
            # 如果当前模型更好，记录所有结果
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = gnn_model.state_dict().copy()
                
                print(f'  发现最佳模型 (Val AUC: {val_auc:.4f})，开始记录评估指标...')
                
                # 训练集评估
                print(f'    计算训练集指标...')
                train_auc, train_ap, train_f1, train_g_mean, train_inference_time, train_peak_memory = test_with_timing_batch(
                    idx_train, y_train, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_train_results = {
                    'auc': train_auc, 'ap': train_ap, 'f1': train_f1, 'g_mean': train_g_mean,
                    'inference_time': train_inference_time, 'peak_memory': train_peak_memory
                }
                
                # 验证集评估
                print(f'    计算验证集指标...')
                val_auc_full, val_ap_full, val_f1_full, val_g_mean_full, val_inference_time, val_peak_memory = test_with_minibatch_timing(
                    idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_val_results = {
                    'auc': val_auc_full, 'ap': val_ap_full, 'f1': val_f1_full, 'g_mean': val_g_mean_full,
                    'inference_time': val_inference_time, 'peak_memory': val_peak_memory
                }
                
                # 测试集评估（可选，训练时可以不计算）
                print(f'    计算测试集指标...')
                test_auc, test_ap, test_f1, test_g_mean, test_inference_time, test_peak_memory = test_with_minibatch_timing(
                    idx_test, y_test, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_test_results = {
                    'auc': test_auc, 'ap': test_ap, 'f1': test_f1, 'g_mean': test_g_mean,
                    'inference_time': test_inference_time, 'peak_memory': test_peak_memory
                }
                
                print(f'    评估指标记录完成')
                
                # 保存最佳模型检查点
                checkpoint_path = os.path.join(model_save_dir, 'checkpoints', f"hogrl_best_model_epoch{epoch}.pth")
                save_model_checkpoint(gnn_model, epoch, val_auc, checkpoint_path, optimizer)
    
    total_training_time = time.time() - start_training_time
    print(f'\n训练总时间: {total_training_time:.2f}秒')
    
    # 训练结束后保存最终模型
    final_model_path = os.path.join(model_save_dir, 'models', "hogrl_final_model.pth")
    torch.save(gnn_model.state_dict(), final_model_path)
    print(f"✓ 最终模型已保存: {final_model_path}")
    
    # 加载最佳模型状态
    if best_model_state is not None:
        gnn_model.load_state_dict(best_model_state)
    
    # 打印最佳模型的所有评估结果
    print('\n' + '='*60)
    print('训练完成！最佳模型结果汇总')
    print('='*60)
    print(f'最佳模型来自第 {best_epoch} 轮，验证集AUC: {best_val_auc:.4f}')
    print(f'训练总时间: {total_training_time:.2f}秒')
    
    print('\n训练集最佳结果:')
    if best_train_results:
        print_evaluation_results('Train', 
                               best_train_results['auc'], 
                               best_train_results['ap'], 
                               best_train_results['f1'], 
                               best_train_results['g_mean'],
                               best_train_results['inference_time'],
                               best_train_results['peak_memory'])
    
    print('\n验证集最佳结果:')
    if best_val_results:
        print_evaluation_results('Val', 
                               best_val_results['auc'], 
                               best_val_results['ap'], 
                               best_val_results['f1'], 
                               best_val_results['g_mean'],
                               best_val_results['inference_time'],
                               best_val_results['peak_memory'])
    
    print('\n测试集最佳结果:')
    if best_test_results:
        print_evaluation_results('Test', 
                               best_test_results['auc'], 
                               best_test_results['ap'], 
                               best_test_results['f1'], 
                               best_test_results['g_mean'],
                               best_test_results['inference_time'],
                               best_test_results['peak_memory'])
    
    # 准备返回结果
    results = {
        'train_auc': best_train_results['auc'] if best_train_results else 0,
        'train_ap': best_train_results['ap'] if best_train_results else 0,
        'train_f1': best_train_results['f1'] if best_train_results else 0,
        'train_g_mean': best_train_results['g_mean'] if best_train_results else 0,
        'train_inference_time': best_train_results['inference_time'] if best_train_results else 0,
        'train_peak_memory': best_train_results['peak_memory'] if best_train_results else 0,
        'val_auc': best_val_results['auc'] if best_val_results else 0,
        'val_ap': best_val_results['ap'] if best_val_results else 0,
        'val_f1': best_val_results['f1'] if best_val_results else 0,
        'val_g_mean': best_val_results['g_mean'] if best_val_results else 0,
        'val_inference_time': best_val_results['inference_time'] if best_val_results else 0,
        'val_peak_memory': best_val_results['peak_memory'] if best_val_results else 0,
        'test_auc': best_test_results['auc'] if best_test_results else 0,
        'test_ap': best_test_results['ap'] if best_test_results else 0,
        'test_f1': best_test_results['f1'] if best_test_results else 0,
        'test_g_mean': best_test_results['g_mean'] if best_test_results else 0,
        'test_inference_time': best_test_results['inference_time'] if best_test_results else 0,
        'test_peak_memory': best_test_results['peak_memory'] if best_test_results else 0,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'final_model_path': final_model_path,
        'checkpoints_dir': os.path.join(model_save_dir, 'checkpoints'),
        'model_save_dir': model_save_dir,
        'total_training_time': total_training_time
    }
    
    # 保存训练结果
    if args.get('save_results', True):
        results_path = os.path.join(model_save_dir, 'training_results.json')
        save_training_results(results, results_path)
    
    return results

def hogrl_infer(args):
    """HOGRL推理函数 - 只进行推理并对推理过程计时"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 打印GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"GPU: {gpu_name}, 显存: {gpu_memory:.2f} GB")
    
    print('加载数据...')
    
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    print(f"路径: {prefix}, 内容: {os.listdir(prefix) if os.path.exists(prefix) else '路径不存在'}")
    
    # 加载数据
    edge_indexs, feat_data, labels = load_data(args['dataset'], args['layers_tree'], prefix)
    
    print(f"加载完成: 特征形状={feat_data.shape}, 标签数={len(labels)}")
    
    # 数据分割
    if args['dataset'] is not None:
        print('load data of way 1 !!!!!!!!!')
        # DGraphFin格式数据集：使用数据集中预定义的掩码
        npz_path = f"../data/{args['dataset']}.npz"
        npz = np.load(npz_path)
        
        # 掩码是索引数组，不是布尔掩码
        idx_train = npz['train_mask'].tolist()
        idx_val = npz['valid_mask'].tolist()
        idx_test = npz['test_mask'].tolist()
        
        # 获取对应的标签
        y_train = labels[idx_train].tolist()
        y_val = labels[idx_val].tolist()
        y_test = labels[idx_test].tolist()
        
        print(f"{args['dataset']}数据集分割:")
        print(f"  训练集: {len(idx_train)} 个样本 ({sum(y_train)} 正样本)")
        print(f"  验证集: {len(idx_val)} 个样本 ({sum(y_val)} 正样本)")
        print(f"  测试集: {len(idx_test)} 个样本 ({sum(y_test)} 正样本)")
    else:
        raise ValueError(f"不支持的数据集: {args['dataset']}. 支持的数据集: 'dgraphfin', 'dgraph_exp130_nona', 'yelp', 'amazon'")
    
    # 模型路径
    model_path = args['model_path']
    print(f"使用模型: {model_path}")
    
    # 初始化模型
    gnn_model = multi_HOGRL_Model(
        feat_data.shape[1], 2, len(edge_indexs), args['emb_size'], 
        args['drop_rate'], args['weight'], args['layers'], args['layers_tree']
    ).to(device)
    
    # 记录模型加载时间
    print(f'\n开始加载模型...')
    model_load_start_time = time.time()
    
    # 加载模型
    if model_path.endswith('.pth'):
        # 加载最终模型
        gnn_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ 模型权重已从 {model_path} 加载")
    else:
        # 加载检查点
        load_model_checkpoint(gnn_model, model_path, device)
    
    model_load_time = (time.time() - model_load_start_time) * 1000  # 转换为毫秒
    print(f"模型加载时间: {model_load_time:.2f}毫秒")
    
    # 将数据移动到设备
    data_transfer_start_time = time.time()
    processed_edge_indexs = []
    for relation_data in edge_indexs:
        base_edge_index = relation_data[0].to(device)
        tree_list = [tree.to(device) for tree in relation_data[1]]
        processed_edge_indexs.append([base_edge_index, tree_list])
    
    # 特征数据
    feat_data_tensor = torch.tensor(feat_data).float().to(device)
    data_transfer_time = (time.time() - data_transfer_start_time) * 1000  # 转换为毫秒
    print(f"数据传输到设备时间: {data_transfer_time:.2f}毫秒")
    
    batch_size = args['batch_size']
    
    print('\n' + '='*60)
    print('开始推理评估...')
    print('='*60)
    
    # 记录总推理时间
    total_inference_start_time = time.time()
    
    # 训练集推理
    print(f'\n1. 训练集推理...')
    train_inference_start_time = time.time()
    train_auc, train_ap, train_f1, train_g_mean, train_inference_time, train_peak_memory = test_with_timing_batch(
        idx_train, y_train, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
    )
    train_inference_end_time = time.time()
    train_total_time = (train_inference_end_time - train_inference_start_time) * 1000
    print(f"  训练集总推理时间: {train_total_time:.2f}毫秒")
    
    # 验证集推理
    print(f'\n2. 验证集推理...')
    val_inference_start_time = time.time()
    val_auc, val_ap, val_f1, val_g_mean, val_inference_time, val_peak_memory = test_with_minibatch_timing(
        idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
    )
    val_inference_end_time = time.time()
    val_total_time = (val_inference_end_time - val_inference_start_time) * 1000
    print(f"  验证集总推理时间: {val_total_time:.2f}毫秒")
    
    # 测试集推理
    print(f'\n3. 测试集推理...')
    test_inference_start_time = time.time()
    test_auc, test_ap, test_f1, test_g_mean, test_inference_time, test_peak_memory = test_with_minibatch_timing(
        idx_test, y_test, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
    )
    test_inference_end_time = time.time()
    test_total_time = (test_inference_end_time - test_inference_start_time) * 1000
    print(f"  测试集总推理时间: {test_total_time:.2f}毫秒")
    
    # 记录总推理结束时间
    total_inference_time = (time.time() - total_inference_start_time) * 1000
    
    # 生成预测结果
    print(f'\n4. 生成预测结果...')
    prediction_start_time = time.time()
    gnn_model.eval()
    with torch.no_grad():
        logits, embeddings = gnn_model(feat_data_tensor, processed_edge_indexs)
        probabilities = torch.exp(logits).cpu().numpy()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    prediction_time = (time.time() - prediction_start_time) * 1000
    print(f"  预测生成时间: {prediction_time:.2f}毫秒")
    
    # 打印推理结果
    print('\n' + '='*60)
    print('推理结果汇总')
    print('='*60)
    
    print(f'\n时间统计:')
    print(f"  模型加载时间: {model_load_time:.2f}毫秒")
    print(f"  数据传输时间: {data_transfer_time:.2f}毫秒")
    print(f"  训练集推理时间: {train_total_time:.2f}毫秒")
    print(f"  验证集推理时间: {val_total_time:.2f}毫秒")
    print(f"  测试集推理时间: {test_total_time:.2f}毫秒")
    print(f"  预测生成时间: {prediction_time:.2f}毫秒")
    print(f"  总推理时间: {total_inference_time:.2f}毫秒")
    
    print(f'\n节点数量统计:')
    print(f"  训练集节点数: {len(idx_train)}")
    print(f"  验证集节点数: {len(idx_val)}")
    print(f"  测试集节点数: {len(idx_test)}")
    print(f"  总节点数: {len(labels)}")
    
    print(f'\n平均推理时间:')
    print(f"  训练集平均每个节点: {train_total_time/len(idx_train):.4f}毫秒")
    print(f"  验证集平均每个节点: {val_total_time/len(idx_val):.4f}毫秒")
    print(f"  测试集平均每个节点: {test_total_time/len(idx_test):.4f}毫秒")
    print(f"  总体平均每个节点: {total_inference_time/len(labels):.4f}毫秒")
    
    print('\n训练集结果:')
    print_evaluation_results('Train', train_auc, train_ap, train_f1, train_g_mean, 
                           train_inference_time, train_peak_memory)
    
    print('\n验证集结果:')
    print_evaluation_results('Val', val_auc, val_ap, val_f1, val_g_mean, 
                           val_inference_time, val_peak_memory)
    
    print('\n测试集结果:')
    print_evaluation_results('Test', test_auc, test_ap, test_f1, test_g_mean, 
                           test_inference_time, test_peak_memory)
    
    # 准备返回结果
    inference_results = {
        'probabilities': probabilities,
        'predictions': predictions,
        'embeddings': embeddings.cpu().numpy() if embeddings is not None else None,
        'train_metrics': {
            'auc': train_auc, 'ap': train_ap, 'f1': train_f1, 'g_mean': train_g_mean,
            'inference_time': train_inference_time, 'peak_memory': train_peak_memory,
            'total_time': train_total_time
        },
        'val_metrics': {
            'auc': val_auc, 'ap': val_ap, 'f1': val_f1, 'g_mean': val_g_mean,
            'inference_time': val_inference_time, 'peak_memory': val_peak_memory,
            'total_time': val_total_time
        },
        'test_metrics': {
            'auc': test_auc, 'ap': test_ap, 'f1': test_f1, 'g_mean': test_g_mean,
            'inference_time': test_inference_time, 'peak_memory': test_peak_memory,
            'total_time': test_total_time
        },
        'timing_metrics': {
            'model_load_time': model_load_time,
            'data_transfer_time': data_transfer_time,
            'prediction_generation_time': prediction_time,
            'total_inference_time': total_inference_time,
            'nodes_count': {
                'train': len(idx_train),
                'val': len(idx_val),
                'test': len(idx_test),
                'total': len(labels)
            },
            'avg_time_per_node': {
                'train': train_total_time/len(idx_train) if len(idx_train) > 0 else 0,
                'val': val_total_time/len(idx_val) if len(idx_val) > 0 else 0,
                'test': test_total_time/len(idx_test) if len(idx_test) > 0 else 0,
                'total': total_inference_time/len(labels) if len(labels) > 0 else 0
            }
        },
        'model_path': model_path
    }
    
    # 保存推理结果
    if args.get('save_predictions', True):
        results_dir = args.get('results_dir', './inference_results/')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细预测结果
        predictions_file = os.path.join(results_dir, 'predictions.pkl')
        with open(predictions_file, 'wb') as f:
            pickle.dump(inference_results, f)
        print(f"\n✓ 预测结果已保存: {predictions_file}")
        
        # 保存指标摘要
        summary_file = os.path.join(results_dir, 'inference_summary.json')
        summary = {
            'train_metrics': inference_results['train_metrics'],
            'val_metrics': inference_results['val_metrics'],
            'test_metrics': inference_results['test_metrics'],
            'timing_metrics': inference_results['timing_metrics'],
            'model_path': model_path,
            'inference_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ 推理摘要已保存: {summary_file}")
    
    return inference_results

def hogrl_main(args):
    """HOGRL主函数（兼容性）"""
    # 根据模式调用相应的函数
    mode = args.get('mode', 'train')
    
    if mode == 'train':
        return hogrl_train(args)
    elif mode == 'infer':
        return hogrl_infer(args)
    else:
        # 默认使用训练模式
        print(f"警告: 未知模式 '{mode}'，使用默认训练模式")
        return hogrl_train(args)
