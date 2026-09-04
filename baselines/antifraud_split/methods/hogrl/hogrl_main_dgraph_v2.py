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

def hogrl_main(args):
    """主函数"""
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
    
    # ========== 修复边索引格式 ==========
    # 注意：从load_data返回的edge_indexs已经是PyTorch张量格式
    # 格式为：[[relation1, relation1_tree], ...] 或 [[relation1, relation1_tree]]
    
    # 检查边索引格式
    print(f"\n检查边索引格式...")
    for i, relation_data in enumerate(edge_indexs):
        print(f"  关系{i+1}: 基础边索引形状={relation_data[0].shape}")
        for j, tree in enumerate(relation_data[1]):
            print(f"    树{j+1}形状: {tree.shape}")
    
    # ========== 数据分割 ==========
    if args['dataset'] is not None :
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
    # edge_indexs已经是PyTorch张量，直接移动到设备
    processed_edge_indexs = []
    for relation_data in edge_indexs:
        base_edge_index = relation_data[0].to(device)
        tree_list = [tree.to(device) for tree in relation_data[1]]
        processed_edge_indexs.append([base_edge_index, tree_list])
    
    # 特征数据
    feat_data_tensor = torch.tensor(feat_data).float().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=5e-5)
    batch_size = args['batch_size']
    
    # 存储最佳模型信息
    best_val_auc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # 存储最佳模型的评估结果（在训练过程中记录）
    best_train_results = None
    best_val_results = None
    best_test_results = None
    
    print('开始训练...')
    for epoch in range(args['num_epochs']):
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

        # 每10个epoch验证一次
        if epoch % 10 == 9:
            # 验证集评估
            val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs)
            
            # 只在控制台输出损失和验证集AUC，不输出详细评估结果
            print(f'Epoch: {epoch:3d}, Loss: {total_loss/num_batches:.4f}, Val AUC: {val_auc:.4f}')
            
            # 如果当前模型更好，记录所有结果
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = gnn_model.state_dict().copy()
                
                # 分别计算各个数据集的推理时间和指标
                print(f'  发现最佳模型 (Val AUC: {val_auc:.4f})，开始记录评估指标...')
                
                # 训练集评估（一次性推理）
                print(f'    计算训练集指标...')
                train_auc, train_ap, train_f1, train_g_mean, train_inference_time, train_peak_memory = test_with_timing_batch(
                    idx_train, y_train, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_train_results = {
                    'auc': train_auc, 'ap': train_ap, 'f1': train_f1, 'g_mean': train_g_mean,
                    'inference_time': train_inference_time, 'peak_memory': train_peak_memory
                }
                
                # 验证集评估（按minibatch推理）
                print(f'    计算验证集指标（按batch_size={batch_size}进行minibatch推理）...')
                val_auc_full, val_ap_full, val_f1_full, val_g_mean_full, val_inference_time, val_peak_memory = test_with_minibatch_timing(
                    idx_val, y_val, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_val_results = {
                    'auc': val_auc_full, 'ap': val_ap_full, 'f1': val_f1_full, 'g_mean': val_g_mean_full,
                    'inference_time': val_inference_time, 'peak_memory': val_peak_memory
                }
                
                # 测试集评估（按minibatch推理）
                print(f'    计算测试集指标（按batch_size={batch_size}进行minibatch推理）...')
                test_auc, test_ap, test_f1, test_g_mean, test_inference_time, test_peak_memory = test_with_minibatch_timing(
                    idx_test, y_test, gnn_model, feat_data_tensor, processed_edge_indexs, device, batch_size
                )
                best_test_results = {
                    'auc': test_auc, 'ap': test_ap, 'f1': test_f1, 'g_mean': test_g_mean,
                    'inference_time': test_inference_time, 'peak_memory': test_peak_memory
                }
                
                print(f'    评估指标记录完成')
                print(f'    推理时间对比: 训练集={train_inference_time:.2f}ms, '
                      f'验证集={val_inference_time:.2f}ms, '
                      f'测试集={test_inference_time:.2f}ms')

    # 训练结束后打印所有最佳结果
    print('\n' + '='*60)
    print('训练完成！最佳模型结果汇总')
    print('='*60)
    print(f'最佳模型来自第 {best_epoch} 轮，验证集AUC: {best_val_auc:.4f}')
    print(f'注意：训练集为一次性推理，验证集和测试集按batch_size={batch_size}进行minibatch推理')
    
    # 加载最佳模型状态
    if best_model_state is not None:
        gnn_model.load_state_dict(best_model_state)
    
    # 打印最佳模型的所有评估结果
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
    
    # 打印汇总表格
    print('\n' + '='*60)
    print('性能指标汇总')
    print('='*60)
    print(f"{'数据集':<10} {'AUC':<8} {'AP':<8} {'F1':<8} {'G-mean':<8} {'推理时间(ms)':<14} {'GPU内存(MB)':<12}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*14} {'-'*12}")
    
    if best_train_results:
        print(f"{'训练集':<10} {best_train_results['auc']:.4f}  {best_train_results['ap']:.4f}  "
              f"{best_train_results['f1']:.4f}  {best_train_results['g_mean']:.4f}  "
              f"{best_train_results['inference_time']:>12.2f}    {best_train_results['peak_memory']:.2f}")
    
    if best_val_results:
        print(f"{'验证集':<10} {best_val_results['auc']:.4f}  {best_val_results['ap']:.4f}  "
              f"{best_val_results['f1']:.4f}  {best_val_results['g_mean']:.4f}  "
              f"{best_val_results['inference_time']:>12.2f}    {best_val_results['peak_memory']:.2f}")
    
    if best_test_results:
        print(f"{'测试集':<10} {best_test_results['auc']:.4f}  {best_test_results['ap']:.4f}  "
              f"{best_test_results['f1']:.4f}  {best_test_results['g_mean']:.4f}  "
              f"{best_test_results['inference_time']:>12.2f}    {best_test_results['peak_memory']:.2f}")
    
    # 添加推理时间分析
    print('\n推理时间分析:')
    print(f"  总节点数: {len(labels)}")
    print(f"  训练集节点数: {len(idx_train)}")
    print(f"  验证集节点数: {len(idx_val)}")
    print(f"  测试集节点数: {len(idx_test)}")
    
    if best_train_results and best_val_results and best_test_results:
        total_inference_time = best_train_results['inference_time'] + best_val_results['inference_time'] + best_test_results['inference_time']
        print(f"  总推理时间: {total_inference_time:.2f}毫秒")
        print(f"  验证集平均每个节点推理时间: {best_val_results['inference_time']/len(idx_val):.4f}毫秒")
        print(f"  测试集平均每个节点推理时间: {best_test_results['inference_time']/len(idx_test):.4f}毫秒")
    
    # 生成嵌入可视化（注释掉）
    # print('\n生成嵌入可视化...')
    # try:
    #     # 记录可视化时间
    #     viz_start_time = time.time()
    #     out, embedding = gnn_model(feat_data, processed_edge_indexs)
    #     Visualization(labels, embedding.cpu().detach(), prefix)
    #     viz_time = time.time() - viz_start_time
    #     print(f'可视化完成，耗时: {viz_time:.4f}秒')
    # except Exception as e:
    #     print(f'可视化失败: {e}')
    
    # 返回所有指标
    return {
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
        'best_epoch': best_epoch
    }
