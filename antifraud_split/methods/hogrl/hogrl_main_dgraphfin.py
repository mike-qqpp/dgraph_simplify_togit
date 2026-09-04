import random
import torch
import os
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from .hogrl_model import *
from .hogrl_utils_dgraphfin import *

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

def print_evaluation_results(split_name, auc, ap, f1, g_mean):
    """打印评估结果"""
    print(f"  {split_name} AUC: {auc:.4f}")
    print(f"  {split_name} AP:  {ap:.4f}")
    print(f"  {split_name} F1:  {f1:.4f}")
    print(f"  {split_name} G-mean: {g_mean:.4f}")

def hogrl_main(args):
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
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
    if args['dataset'] == 'dgraphfin':
        npz_path = "../data/dgraphfin.npz"
        npz = np.load(npz_path)
        
        # 掩码是索引数组，不是布尔掩码
        idx_train = npz['train_mask'].tolist()
        idx_val = npz['valid_mask'].tolist()
        idx_test = npz['test_mask'].tolist()
        
        # 获取对应的标签
        y_train = labels[idx_train].tolist()
        y_val = labels[idx_val].tolist()
        y_test = labels[idx_test].tolist()
        
        print(f"DGraphFin数据集分割:")
        print(f"  训练集: {len(idx_train)} 个样本 ({sum(y_train)} 正样本)")
        print(f"  验证集: {len(idx_val)} 个样本 ({sum(y_val)} 正样本)")
        print(f"  测试集: {len(idx_test)} 个样本 ({sum(y_test)} 正样本)")
        
    elif args['dataset'] == 'yelp' or args['dataset'] == 'CCFD':
        assert args['dataset'] != 'CCFD', 'Due to confidentiality agreements, we are unable to provide the CCFD data.'
        
        index = list(range(len(labels)))
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(
            index, labels, stratify=labels, test_size=args['test_size'], 
            random_state=args['seed'], shuffle=True
        )
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_train_val, y_train_val, stratify=y_train_val, 
            test_size=args['val_size'], random_state=args['seed'], shuffle=True
        )
        
    elif args['dataset'] == 'amazon':
        # 0-3304 are unlabeled nodes
        index = list(range(3305, len(labels)))
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(
            index, labels[3305:], stratify=labels[3305:], 
            test_size=args['test_size'], random_state=args['seed'], shuffle=True
        )
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_train_val, y_train_val, stratify=y_train_val, 
            test_size=args['val_size'], random_state=args['seed'], shuffle=True
        )
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
    
    # 将数据移动到设备
    # edge_indexs已经是PyTorch张量，直接移动到设备
    processed_edge_indexs = []
    for relation_data in edge_indexs:
        base_edge_index = relation_data[0].to(device)
        tree_list = [tree.to(device) for tree in relation_data[1]]
        processed_edge_indexs.append([base_edge_index, tree_list])
    
    # 特征数据
    feat_data = torch.tensor(feat_data).float().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=5e-5)
    batch_size = args['batch_size']
    
    best_val_auc = 0.0
    best_model_state = None
    best_epoch = 0
    
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
            out, _ = gnn_model(feat_data, processed_edge_indexs)
            
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=device)
            loss = F.nll_loss(out[batch_nodes_tensor], batch_label)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 每10个epoch验证一次
        if epoch % 10 == 9:
            # 验证集评估
            val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data, processed_edge_indexs)
            
            # 测试集评估（仅用于监控，不用于模型选择）
            test_auc, test_ap, test_f1, test_g_mean = test(idx_test, y_test, gnn_model, feat_data, processed_edge_indexs)
            
            print(f'\nEpoch: {epoch:3d}, Loss: {total_loss/num_batches:.4f}')
            print('验证集结果:')
            print_evaluation_results('Val', val_auc, val_ap, val_f1, val_g_mean)
            print('测试集结果:')
            print_evaluation_results('Test', test_auc, test_ap, test_f1, test_g_mean)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = gnn_model.state_dict().copy()
                print(f'✅ 发现最佳模型 (Val AUC: {val_auc:.4f})')

    # 最终评估最佳模型
    print('\n' + '='*60)
    print('最终评估结果')
    print('='*60)
    
    if best_model_state is not None:
        # 加载最佳模型
        gnn_model.load_state_dict(best_model_state)
        print(f'使用第{best_epoch}轮的最佳模型')
        
        # 验证集最终评估
        print('\n验证集最终结果:')
        val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data, processed_edge_indexs)
        print_evaluation_results('Val', val_auc, val_ap, val_f1, val_g_mean)
        
        # 测试集最终评估
        print('\n测试集最终结果:')
        test_auc, test_ap, test_f1, test_g_mean = test(idx_test, y_test, gnn_model, feat_data, processed_edge_indexs)
        print_evaluation_results('Test', test_auc, test_ap, test_f1, test_g_mean)
        
        # 训练集评估（可选）
        print('\n训练集结果:')
        train_auc, train_ap, train_f1, train_g_mean = test(idx_train, y_train, gnn_model, feat_data, processed_edge_indexs)
        print_evaluation_results('Train', train_auc, train_ap, train_f1, train_g_mean)
        
        # 打印汇总表格
        print('\n' + '='*60)
        print('性能指标汇总')
        print('='*60)
        print(f"{'数据集':<10} {'AUC':<8} {'AP':<8} {'F1':<8} {'G-mean':<8}")
        print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'训练集':<10} {train_auc:.4f}  {train_ap:.4f}  {train_f1:.4f}  {train_g_mean:.4f}")
        print(f"{'验证集':<10} {val_auc:.4f}  {val_ap:.4f}  {val_f1:.4f}  {val_g_mean:.4f}")
        print(f"{'测试集':<10} {test_auc:.4f}  {test_ap:.4f}  {test_f1:.4f}  {test_g_mean:.4f}")
        
    else:
        print('警告: 没有找到最佳模型状态')
        # 使用最后训练的模型进行评估
        print('\n使用最后训练的模型进行评估:')
        
        # 验证集评估
        val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data, processed_edge_indexs)
        print_evaluation_results('Val', val_auc, val_ap, val_f1, val_g_mean)
        
        # 测试集评估
        test_auc, test_ap, test_f1, test_g_mean = test(idx_test, y_test, gnn_model, feat_data, processed_edge_indexs)
        print_evaluation_results('Test', test_auc, test_ap, test_f1, test_g_mean)
    
    # 生成嵌入可视化
    print('\n生成嵌入可视化...')
    try:
        out, embedding = gnn_model(feat_data, processed_edge_indexs)
        Visualization(labels, embedding.cpu().detach(), prefix)
        print('可视化完成')
    except Exception as e:
        print(f'可视化失败: {e}')
    
    # 返回所有指标
    return {
        'train_auc': train_auc if 'train_auc' in locals() else 0,
        'train_ap': train_ap if 'train_ap' in locals() else 0,
        'train_f1': train_f1 if 'train_f1' in locals() else 0,
        'train_g_mean': train_g_mean if 'train_g_mean' in locals() else 0,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'val_f1': val_f1,
        'val_g_mean': val_g_mean,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'test_g_mean': test_g_mean,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch
    }
