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
import json
import pickle

def test(idx_eval, y_eval, gnn_model, feat_data, edge_indexs):
    """测试函数 - CPU版本"""
    gnn_model.eval()
    logits, _ = gnn_model(feat_data, edge_indexs)
    x_softmax = torch.exp(logits).detach()
    positive_class_probs = x_softmax[:, 1].numpy()[np.array(idx_eval)]
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)

    return auc_score, ap_score, f1_score_val, g_mean

def test_with_timing_batch(idx_eval, y_eval, gnn_model, feat_data, edge_indexs, device, batch_size):
    """带有时间统计的测试函数 - CPU版本"""
    gnn_model.eval()
    
    # 记录推理开始时间
    start_time = time.time()
    
    # 执行推理（整个数据集）
    with torch.no_grad():
        logits, _ = gnn_model(feat_data, edge_indexs)
    
    # 提取特定数据集的预测结果
    x_softmax = torch.exp(logits).detach()
    positive_class_probs = x_softmax[:, 1].numpy()[np.array(idx_eval)]
    
    # 记录推理结束时间
    inference_time_ms = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 计算指标
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)
    
    return auc_score, ap_score, f1_score_val, g_mean, inference_time_ms, 0  # CPU峰值内存为0

def test_with_minibatch_timing(idx_eval, y_eval, gnn_model, feat_data, edge_indexs, device, batch_size):
    """按minibatch进行推理并统计时间 - CPU版本"""
    gnn_model.eval()
    
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
            all_logits.append(batch_logits)
    
    # 记录推理结束时间
    inference_time_ms = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 合并所有batch的结果
    all_logits_tensor = torch.cat(all_logits, dim=0)
    x_softmax = torch.exp(all_logits_tensor).detach()
    positive_class_probs = x_softmax[:, 1].numpy()
    
    # 计算指标
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)
    
    return auc_score, ap_score, f1_score_val, g_mean, inference_time_ms, 0

def print_evaluation_results(split_name, auc, ap, f1, g_mean, inference_time=None, peak_memory=None):
    """打印评估结果"""
    print(f"  {split_name} AUC: {auc:.4f}")
    print(f"  {split_name} AP:  {ap:.4f}")
    print(f"  {split_name} F1:  {f1:.4f}")
    print(f"  {split_name} G-mean: {g_mean:.4f}")
    if inference_time is not None:
        print(f"  {split_name} 推理时间: {inference_time:.2f}毫秒")

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

def load_model_checkpoint(model, checkpoint_path, device='cpu', optimizer=None):
    """加载模型检查点 - CPU版本"""
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
    """HOGRL训练函数 - CPU版本"""
    device = torch.device('cpu')
    print(f"使用设备: CPU")
    
    print('加载数据...')
    
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    print(f"路径: {prefix}, 内容: {os.listdir(prefix) if os.path.exists(prefix) else '路径不存在'}")
    
    # 设置随机种子
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    
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
        npz_path = f"../data/{args['dataset']}.npz"
        npz = np.load(npz_path)
        
        idx_train = npz['train_mask'].tolist()
        idx_val = npz['valid_mask'].tolist()
        idx_test = npz['test_mask'].tolist()
        
        y_train = labels[idx_train].tolist()
        y_val = labels[idx_val].tolist()
        y_test = labels[idx_test].tolist()
        
        print(f"{args['dataset']}数据集分割:")
        print(f"  训练集: {len(idx_train)} 个样本 ({sum(y_train)} 正样本)")
        print(f"  验证集: {len(idx_val)} 个样本 ({sum(y_val)} 正样本)")
        print(f"  测试集: {len(idx_test)} 个样本 ({sum(y_test)} 正样本)")
    else:
        raise ValueError(f"不支持的数据集: {args['dataset']}")
    
    # 训练集正负样本分割
    train_pos, train_neg = pos_neg_split(idx_train, y_train)
    print(f"训练集正负样本: {len(train_pos)} 正, {len(train_neg)} 负")

    # 初始化模型
    gnn_model = multi_HOGRL_Model(
        feat_data.shape[1], 2, len(edge_indexs), args['emb_size'], 
        args['drop_rate'], args['weight'], args['layers'], args['layers_tree']
    ).to(device)
    
    # 将数据移动到设备（CPU）
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
            val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs)
            
            print(f'Epoch: {epoch:3d}, Loss: {total_loss/num_batches:.4f}, Val AUC: {val_auc:.4f}, Epoch Time: {epoch_time:.2f}s')
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = gnn_model.state_dict().copy()
                
                print(f'  发现最佳模型 (Val AUC: {val_auc:.4f})，开始记录评估指标...')
                
                # 训练集评估
                print(f'    计算训练集指标...')
                train_auc, train_ap, train_f1, train_g_mean, train_inference_time, _ = test_with_timing_batch(
                    idx_train, y_train, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_train_results = {
                    'auc': train_auc, 'ap': train_ap, 'f1': train_f1, 'g_mean': train_g_mean,
                    'inference_time': train_inference_time
                }
                
                # 验证集评估
                print(f'    计算验证集指标...')
                val_auc_full, val_ap_full, val_f1_full, val_g_mean_full, val_inference_time, _ = test_with_minibatch_timing(
                    idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_val_results = {
                    'auc': val_auc_full, 'ap': val_ap_full, 'f1': val_f1_full, 'g_mean': val_g_mean_full,
                    'inference_time': val_inference_time
                }
                
                # 测试集评估
                print(f'    计算测试集指标...')
                test_auc, test_ap, test_f1, test_g_mean, test_inference_time, _ = test_with_minibatch_timing(
                    idx_test, y_test, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_test_results = {
                    'auc': test_auc, 'ap': test_ap, 'f1': test_f1, 'g_mean': test_g_mean,
                    'inference_time': test_inference_time
                }
                
                print(f'    评估指标记录完成')
                
                checkpoint_path = os.path.join(model_save_dir, 'checkpoints', f"hogrl_best_model_epoch{epoch}.pth")
                save_model_checkpoint(gnn_model, epoch, val_auc, checkpoint_path, optimizer)
    
    total_training_time = time.time() - start_training_time
    print(f'\n训练总时间: {total_training_time:.2f}秒')
    
    final_model_path = os.path.join(model_save_dir, 'models', "hogrl_final_model.pth")
    torch.save(gnn_model.state_dict(), final_model_path)
    print(f"✓ 最终模型已保存: {final_model_path}")
    
    if best_model_state is not None:
        gnn_model.load_state_dict(best_model_state)
    
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
                               best_train_results['inference_time'])
    
    print('\n验证集最佳结果:')
    if best_val_results:
        print_evaluation_results('Val', 
                               best_val_results['auc'], 
                               best_val_results['ap'], 
                               best_val_results['f1'], 
                               best_val_results['g_mean'],
                               best_val_results['inference_time'])
    
    print('\n测试集最佳结果:')
    if best_test_results:
        print_evaluation_results('Test', 
                               best_test_results['auc'], 
                               best_test_results['ap'], 
                               best_test_results['f1'], 
                               best_test_results['g_mean'],
                               best_test_results['inference_time'])
    
    results = {
        'train_auc': best_train_results['auc'] if best_train_results else 0,
        'train_ap': best_train_results['ap'] if best_train_results else 0,
        'train_f1': best_train_results['f1'] if best_train_results else 0,
        'train_g_mean': best_train_results['g_mean'] if best_train_results else 0,
        'train_inference_time': best_train_results['inference_time'] if best_train_results else 0,
        'val_auc': best_val_results['auc'] if best_val_results else 0,
        'val_ap': best_val_results['ap'] if best_val_results else 0,
        'val_f1': best_val_results['f1'] if best_val_results else 0,
        'val_g_mean': best_val_results['g_mean'] if best_val_results else 0,
        'val_inference_time': best_val_results['inference_time'] if best_val_results else 0,
        'test_auc': best_test_results['auc'] if best_test_results else 0,
        'test_ap': best_test_results['ap'] if best_test_results else 0,
        'test_f1': best_test_results['f1'] if best_test_results else 0,
        'test_g_mean': best_test_results['g_mean'] if best_test_results else 0,
        'test_inference_time': best_test_results['inference_time'] if best_test_results else 0,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'final_model_path': final_model_path,
        'checkpoints_dir': os.path.join(model_save_dir, 'checkpoints'),
        'model_save_dir': model_save_dir,
        'total_training_time': total_training_time
    }
    
    if args.get('save_results', True):
        results_path = os.path.join(model_save_dir, 'training_results.json')
        save_training_results(results, results_path)
    
    return results


def hogrl_infer(args):
    """
    HOGRL推理函数 - CPU版本
    只对测试集进行推理，输出指标，不保存文件
    """
    device = torch.device('cpu')
    print(f"使用设备: CPU")
    
    print('\n' + '='*60)
    print('加载数据...')
    print('='*60)
    
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    print(f"数据路径: {prefix}")
    
    # 加载数据
    try:
        edge_indexs, feat_data, labels = load_data(args['dataset'], args['layers_tree'], prefix)
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise
    
    print(f"加载完成: 特征形状={feat_data.shape}, 标签数={len(labels)}")
    
    # 检查边索引
    print(f"\n图结构信息:")
    for i, relation_data in enumerate(edge_indexs):
        print(f"  关系{i+1}: {relation_data[0].shape[1]} 条边")
        for j, tree in enumerate(relation_data[1]):
            print(f"    树{j+1}: {tree.shape[1]} 条边")
    
    # 数据分割
    if args['dataset'] is not None:
        npz_path = f"../data/{args['dataset']}.npz"
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"数据文件不存在: {npz_path}")
            
        npz = np.load(npz_path, allow_pickle=True)
        
        if 'test_mask' not in npz.files:
            raise KeyError(f"npz文件中没有 'test_mask' 键，可用键: {list(npz.files)}")
            
        idx_test = npz['test_mask'].tolist()
        y_test = labels[idx_test].tolist()
        
        print(f"\n{args['dataset']}数据集测试集:")
        print(f"  测试集样本数: {len(idx_test)}")
        print(f"  正样本数: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
        print(f"  负样本数: {len(y_test) - sum(y_test)} ({(1 - sum(y_test)/len(y_test))*100:.2f}%)")
        print(f"  特征维度: {feat_data.shape[1]}")
    else:
        raise ValueError(f"不支持的数据集: {args['dataset']}")
    
    # 模型路径
    model_path = args['model_path']
    print(f"\n模型路径: {model_path}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 初始化模型
    input_dim = feat_data.shape[1]
    print(f"\n初始化模型...")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层维度: {args['emb_size']}")
    print(f"  关系数量: {len(edge_indexs)}")
    print(f"  层数: {args['layers']}")
    
    gnn_model = multi_HOGRL_Model(
        input_dim,
        2, 
        len(edge_indexs), 
        args['emb_size'], 
        args['drop_rate'], 
        args['weight'], 
        args['layers'], 
        args['layers_tree']
    ).to(device)
    
    # 加载模型
    print(f'\n加载模型权重...')
    model_load_start_time = time.time()
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        gnn_model.load_state_dict(state_dict)
        print(f"  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        raise
    
    model_load_time = (time.time() - model_load_start_time) * 1000
    print(f"  加载时间: {model_load_time:.2f}毫秒")
    
    # 数据预处理
    print(f'\n数据预处理...')
    data_prep_start_time = time.time()
    
    processed_edge_indexs = []
    for relation_data in edge_indexs:
        base_edge_index = relation_data[0].to(device)
        tree_list = [tree.to(device) for tree in relation_data[1]]
        processed_edge_indexs.append([base_edge_index, tree_list])
    
    feat_data_tensor = torch.tensor(feat_data).float().to(device)
    
    data_prep_time = (time.time() - data_prep_start_time) * 1000
    print(f"  预处理时间: {data_prep_time:.2f}毫秒")
    
    batch_size = args['batch_size']
    
    # 测试集推理
    print(f'\n' + '='*60)
    print('测试集推理')
    print('='*60)
    
    test_inference_start_time = time.time()
    
    # 执行推理
    gnn_model.eval()
    with torch.no_grad():
        logits, _ = gnn_model(feat_data_tensor, processed_edge_indexs)
        test_logits = logits[idx_test]
    
    test_inference_time = (time.time() - test_inference_start_time) * 1000
    
    # 计算概率和预测
    test_probs = torch.exp(test_logits).numpy()
    test_preds = np.argmax(test_probs, axis=1)
    test_pos_probs = test_probs[:, 1]
    
    # 计算指标
    print(f'\n计算评估指标...')
    
    auc_score = roc_auc_score(y_test, test_pos_probs)
    ap_score = average_precision_score(y_test, test_pos_probs)
    f1 = f1_score(y_test, test_preds, average='macro')
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算更多指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    g_mean = np.sqrt(recall * specificity)
    
    print(f'\n' + '='*60)
    print('测试集结果')
    print('='*60)
    
    print(f'\n【核心指标】')
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  Average Precision (AP): {ap_score:.4f}")
    print(f"  F1-Score (macro): {f1:.4f}")
    print(f"  G-mean: {g_mean:.4f}")
    
    print(f'\n【详细指标】')
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print(f'\n【混淆矩阵】')
    print(f"             预测正类  预测负类")
    print(f"  实际正类    {tp:6d}     {fn:6d}")
    print(f"  实际负类    {fp:6d}     {tn:6d}")
    
    print(f'\n【时间统计】')
    print(f"  模型加载时间: {model_load_time:.2f}毫秒")
    print(f"  数据预处理时间: {data_prep_time:.2f}毫秒")
    print(f"  推理时间: {test_inference_time:.2f}毫秒")
    print(f"  总时间: {model_load_time + data_prep_time + test_inference_time:.2f}毫秒")
    
    print(f'\n【样本统计】')
    print(f"  测试集总样本: {len(y_test)}")
    print(f"  平均推理时间/样本: {test_inference_time/len(y_test):.4f}毫秒")
    
    # 返回结果但不保存文件
    results = {
        'test_metrics': {
            'auc': auc_score,
            'ap': ap_score,
            'f1': f1,
            'g_mean': g_mean,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'timing': {
            'model_load_ms': model_load_time,
            'data_prep_ms': data_prep_time,
            'inference_ms': test_inference_time,
            'total_ms': model_load_time + data_prep_time + test_inference_time
        },
        'sample_stats': {
            'test_samples': len(y_test),
            'positive_samples': int(sum(y_test)),
            'negative_samples': len(y_test) - int(sum(y_test)),
            'positive_ratio': sum(y_test)/len(y_test)
        }
    }
    
    print(f'\n' + '='*60)
    print('推理完成！')
    print('='*60)
    
    return results


def hogrl_main(args):
    """HOGRL主函数（兼容性）"""
    mode = args.get('mode', 'train')
    
    if mode == 'train':
        return hogrl_train(args)
    elif mode == 'infer':
        return hogrl_infer(args)
    else:
        print(f"警告: 未知模式 '{mode}'，使用默认训练模式")
        return hogrl_train(args)
