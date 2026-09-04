import random
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from .hogrl_model import *
from .hogrl_utils import *
import numpy as np
import random as rd
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
import os
from tqdm import tqdm
import time

def test_batch(batch_nodes, batch_labels, gnn_model, feat_data, edge_indexs):
    """测试批次数据性能"""
    gnn_model.eval()
    with torch.no_grad():
        logits, _ = gnn_model(feat_data, edge_indexs)
        x_softmax = torch.exp(logits).cpu().detach()
        
        # 确保索引是整数类型
        batch_nodes_int = [int(idx) for idx in batch_nodes]
        positive_class_probs = x_softmax[:, 1].numpy()[batch_nodes_int]
        
        # 计算指标
        batch_labels = np.array(batch_labels)
        positive_class_probs = np.array(positive_class_probs)
        
        if len(np.unique(batch_labels)) > 1:
            auc_score = roc_auc_score(batch_labels, positive_class_probs)
        else:
            auc_score = 0.0
            
        try:
            ap_score = average_precision_score(batch_labels, positive_class_probs)
        except:
            ap_score = 0.0
            
        label_pred = (positive_class_probs >= 0.5).astype(int)
        f1_score_val = f1_score(batch_labels, label_pred, average='macro')
        
        # 计算准确率
        accuracy = accuracy_score(batch_labels, label_pred)
        
        # 计算召回率
        recall = recall_score(batch_labels, label_pred, average='macro')
        
        # 计算G-mean
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(batch_labels, label_pred)
        if cm.shape[0] >= 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            g_mean = np.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0
        else:
            g_mean = 0.0

    return auc_score, ap_score, f1_score_val, accuracy, recall, g_mean

def test(idx_eval, y_eval, gnn_model, feat_data, edge_indexs):
    """测试模型性能"""
    gnn_model.eval()
    with torch.no_grad():
        logits, _ = gnn_model(feat_data, edge_indexs)
        x_softmax = torch.exp(logits).cpu().detach()
        
        # 确保索引是整数类型
        idx_eval = np.array(idx_eval, dtype=int)
        positive_class_probs = x_softmax[:, 1].numpy()[idx_eval]
        
        y_eval = np.array(y_eval)
        positive_class_probs = np.array(positive_class_probs)
        
        if len(np.unique(y_eval)) > 1:
            auc_score = roc_auc_score(y_eval, positive_class_probs)
        else:
            auc_score = 0.0
            
        try:
            ap_score = average_precision_score(y_eval, positive_class_probs)
        except:
            ap_score = 0.0
            
        label_pred = (positive_class_probs >= 0.5).astype(int)
        f1_score_val = f1_score(y_eval, label_pred, average='macro')
        accuracy = accuracy_score(y_eval, label_pred)
        recall = recall_score(y_eval, label_pred, average='macro')
        
        # 计算G-mean
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_eval, label_pred)
        if cm.shape[0] >= 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            g_mean = np.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0
        else:
            g_mean = 0.0

    return auc_score, ap_score, f1_score_val, accuracy, recall, g_mean

def hogrl_main(args):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print('loading data...')
    # 构建数据路径
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    
    # 加载数据
    edge_indexs, feat_data, labels = load_data(args['dataset'], args['layers_tree'], prefix)
    
    # 设置随机种子
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    
    # 数据分割
    print('splitting data...')
    if args['dataset'] == 'yelp' or args['dataset'] == 'CCFD':
        assert args['dataset'] != 'CCFD', 'Due to confidentiality agreements, we are unable to provide the CCFD data.'
        
        index = list(range(len(labels)))
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(
            index, labels, stratify=labels, 
            test_size=args['test_size'], random_state=args['seed'], shuffle=True
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
        # 默认处理方式
        index = list(range(len(labels)))
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(
            index, labels, stratify=labels, 
            test_size=args['test_size'], random_state=args['seed'], shuffle=True
        )
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_train_val, y_train_val, stratify=y_train_val, 
            test_size=args['val_size'], random_state=args['seed'], shuffle=True
        )
    
    # 确保标签是numpy数组
    labels = np.array(labels)
    
    print(f'Train size: {len(idx_train)}, Val size: {len(idx_val)}, Test size: {len(idx_test)}')
    print(f'Positive ratio in train: {np.sum(y_train)/len(y_train):.3f}, '
          f'val: {np.sum(y_val)/len(y_val):.3f}, test: {np.sum(y_test)/len(y_test):.3f}')
    
    # 正负样本分割
    train_pos, train_neg = pos_neg_split(idx_train, y_train)
    
    # 初始化模型
    print('initializing model...')
    gnn_model = multi_HOGRL_Model(
        feat_data.shape[1], 2, len(edge_indexs), 
        args['emb_size'], args['drop_rate'], 
        args['weight'], args['layers'], args['layers_tree']
    ).to(device)
    
    # 将edge_indexs移到设备
    print('moving data to device...')
    for edge_index in edge_indexs:
        edge_index[0] = edge_index[0].to(device)
        edge_index[1] = [tensor.to(device) for tensor in edge_index[1]]
    
    # 转换特征数据
    feat_data = torch.tensor(feat_data).float().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.get('lr', 0.005), 
                                 weight_decay=args.get('weight_decay', 5e-5))
    
    batch_size = args['batch_size']
    
    best_val_auc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # 用于记录训练历史的列表
    train_history = []
    val_history = []
    
    print('\n' + '='*80)
    print('开始训练...')
    print('='*80)
    
    for epoch in range(args['num_epochs']):
        gnn_model.train()
        epoch_loss = 0
        epoch_metrics = {
            'auc': [], 'ap': [], 'f1': [], 
            'accuracy': [], 'recall': [], 'g_mean': []
        }
        
        # 随机欠采样负样本
        sampled_idx_train = undersample(train_pos, train_neg, scale=1)
        rd.shuffle(sampled_idx_train)
        
        # 确保采样索引是整数类型
        sampled_idx_train = [int(idx) for idx in sampled_idx_train]
        
        num_batches = int(len(sampled_idx_train) / batch_size) + 1
        
        print(f'\nEpoch {epoch+1}/{args["num_epochs"]}')
        print(f"训练样本数: {len(sampled_idx_train)}, 批次大小: {batch_size}, 总批次: {num_batches}")
        print('-'*60)
        
        # 创建进度条
        batch_iter = range(num_batches)
        if args.get('show_progress', True):
            batch_iter = tqdm(batch_iter, desc="训练批次", unit="batch")
        
        start_time = time.time()
        
        for batch in batch_iter:
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(sampled_idx_train))
            batch_nodes = sampled_idx_train[i_start:i_end]
            
            # 确保batch_nodes是整数列表
            batch_nodes = [int(node) for node in batch_nodes]
            
            # 获取批次标签
            batch_label = torch.tensor(labels[batch_nodes]).long().to(device)
            batch_label_np = labels[batch_nodes]
            
            optimizer.zero_grad()
            out, _ = gnn_model(feat_data, edge_indexs)
            
            # 使用整数索引获取输出
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=device)
            loss = F.nll_loss(out[batch_nodes_tensor], batch_label)
            
            loss.backward()
            optimizer.step()
            
            # 计算指标
            train_metrics = test_batch(batch_nodes, batch_label_np, gnn_model, feat_data, edge_indexs)
            
            # 记录指标
            epoch_loss += loss.item()
            epoch_metrics['auc'].append(train_metrics[0])
            epoch_metrics['ap'].append(train_metrics[1])
            epoch_metrics['f1'].append(train_metrics[2])
            epoch_metrics['accuracy'].append(train_metrics[3])
            epoch_metrics['recall'].append(train_metrics[4])
            epoch_metrics['g_mean'].append(train_metrics[5])
            
            # 每N个批次显示一次
            if (batch + 1) % args.get('log_interval', 5) == 0 or batch == num_batches - 1:
                avg_auc = np.mean(epoch_metrics['auc'][-args.get('log_interval', 5):])
                avg_ap = np.mean(epoch_metrics['ap'][-args.get('log_interval', 5):])
                avg_f1 = np.mean(epoch_metrics['f1'][-args.get('log_interval', 5):])
                avg_acc = np.mean(epoch_metrics['accuracy'][-args.get('log_interval', 5):])
                
                if not args.get('show_progress', True):  # 如果没有使用tqdm，则打印
                    print(f'  Batch {batch+1}/{num_batches}: '
                          f'Loss: {loss.item():.4f}, '
                          f'AUC: {avg_auc:.4f}, AP: {avg_ap:.4f}, '
                          f'F1: {avg_f1:.4f}, Acc: {avg_acc:.4f}')
        
        epoch_time = time.time() - start_time
        
        # 计算epoch平均指标
        epoch_avg_loss = epoch_loss / num_batches
        epoch_avg_auc = np.mean(epoch_metrics['auc'])
        epoch_avg_ap = np.mean(epoch_metrics['ap'])
        epoch_avg_f1 = np.mean(epoch_metrics['f1'])
        epoch_avg_acc = np.mean(epoch_metrics['accuracy'])
        epoch_avg_recall = np.mean(epoch_metrics['recall'])
        epoch_avg_gmean = np.mean(epoch_metrics['g_mean'])
        
        # 记录训练历史
        train_history.append({
            'epoch': epoch + 1,
            'loss': epoch_avg_loss,
            'auc': epoch_avg_auc,
            'ap': epoch_avg_ap,
            'f1': epoch_avg_f1,
            'accuracy': epoch_avg_acc,
            'recall': epoch_avg_recall,
            'g_mean': epoch_avg_gmean,
            'time': epoch_time
        })
        
        # 显示epoch总结
        print(f'\nEpoch {epoch+1} 总结:')
        print(f'  时间: {epoch_time:.2f}s, 平均损失: {epoch_avg_loss:.4f}')
        print(f'  训练指标 - AUC: {epoch_avg_auc:.4f}, AP: {epoch_avg_ap:.4f}, '
              f'F1: {epoch_avg_f1:.4f}, Acc: {epoch_avg_acc:.4f}, '
              f'Recall: {epoch_avg_recall:.4f}, G-mean: {epoch_avg_gmean:.4f}')
        
        # 验证
        if (epoch + 1) % args.get('val_interval', 10) == 0 or epoch == args['num_epochs'] - 1:
            # 确保验证索引是整数类型
            idx_val_int = [int(idx) for idx in idx_val]
            y_val_list = [int(y) for y in y_val]
            
            val_auc, val_ap, val_f1, val_acc, val_recall, val_g_mean = test(
                idx_val_int, y_val_list, gnn_model, feat_data, edge_indexs
            )
            
            # 记录验证历史
            val_history.append({
                'epoch': epoch + 1,
                'auc': val_auc,
                'ap': val_ap,
                'f1': val_f1,
                'accuracy': val_acc,
                'recall': val_recall,
                'g_mean': val_g_mean
            })
            
            print(f'\n  验证指标 - AUC: {val_auc:.4f}, AP: {val_ap:.4f}, '
                  f'F1: {val_f1:.4f}, Acc: {val_acc:.4f}, '
                  f'Recall: {val_recall:.4f}, G-mean: {val_g_mean:.4f}')
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = gnn_model.state_dict().copy()
                best_epoch = epoch + 1
                print(f'  ★ 新的最佳模型! (Epoch {best_epoch}, Val AUC: {best_val_auc:.4f})')
    
    print('\n' + '='*80)
    print(f'训练完成! 最佳模型在 Epoch {best_epoch}, Val AUC: {best_val_auc:.4f}')
    print('='*80)
    
    # 打印训练历史摘要
    if train_history:
        print('\n训练历史摘要:')
        print('Epoch | Loss   | AUC    | F1     | Acc    | Time')
        print('-'*50)
        for i, hist in enumerate(train_history):
            if (i + 1) % 10 == 0 or i == len(train_history) - 1:
                print(f'{hist["epoch"]:5d} | {hist["loss"]:.4f} | {hist["auc"]:.4f} | '
                      f'{hist["f1"]:.4f} | {hist["accuracy"]:.4f} | {hist["time"]:.1f}s')
    
    # 在测试集上评估最佳模型
    if best_model_state is not None:
        gnn_model.load_state_dict(best_model_state)
    
    print('\n' + '='*80)
    print('测试集评估:')
    
    # 确保测试索引是整数类型
    idx_test_int = [int(idx) for idx in idx_test]
    y_test_list = [int(y) for y in y_test]
    
    test_auc, test_ap, test_f1, test_acc, test_recall, test_g_mean = test(
        idx_test_int, y_test_list, gnn_model, feat_data, edge_indexs
    )
    
    print(f'\n最终测试结果:')
    print(f'  AUC:      {test_auc:.4f}')
    print(f'  AP:       {test_ap:.4f}')
    print(f'  F1-score: {test_f1:.4f}')
    print(f'  Accuracy: {test_acc:.4f}')
    print(f'  Recall:   {test_recall:.4f}')
    print(f'  G-mean:   {test_g_mean:.4f}')
    print('='*80)
    
    # 生成嵌入可视化
    print('\n生成嵌入可视化...')
    out, embedding = gnn_model(feat_data, edge_indexs)
    
    # 确保标签是整数类型
    labels_int = [int(label) for label in labels]
    
    if args.get('visualize', True):
        Visualization(labels_int, embedding.cpu().detach(), prefix)
    
    # 保存结果
    results = {
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'test_accuracy': test_acc,
        'test_recall': test_recall,
        'test_g_mean': test_g_mean,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'train_history': train_history,
        'val_history': val_history
    }
    
    # 保存模型
    if args.get('save_model', False):
        model_path = os.path.join(prefix, 'saved_models', f'hogrl_model_{args["dataset"]}.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'args': args,
            'results': results
        }, model_path)
        print(f'\n模型已保存至: {model_path}')
    
    return results
