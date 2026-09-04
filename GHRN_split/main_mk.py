import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
from dataset import Dataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score
from BWGNN import *
from sklearn.model_selection import train_test_split
import pickle as pkl


def train_and_evaluate(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    
    # --- 数据集划分 ---
    if dataset_name in ['tfinance', 'tsocial', 'dgraphfin']:
        index = list(range(len(labels)))
        if dataset_name == 'amazon':
            index = list(range(3305, len(labels)))

        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=args.train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=0.67,
                                                                random_state=2, shuffle=True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1
    else:
        train_mask = torch.ByteTensor(g.ndata['train_mask'])
        val_mask = torch.ByteTensor(g.ndata['val_mask'])
        test_mask = torch.ByteTensor(g.ndata['test_mask'])
    
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    
    # --- 训练初始化 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_f1 = 0.0
    best_model_state = None  # 保存最佳模型参数
    best_epoch = 0
    
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    
    time_start = time.time()
    
    # --- 训练循环 ---
    for epoch in range(1, args.epoch + 1):
        # 训练步骤
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证步骤
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask], weight=torch.tensor([1., weight]))
            probs = logits.softmax(1)
            
            # 获取最佳F1对应的阈值
            val_f1, thres = get_best_f1(labels[val_mask].numpy(), probs[val_mask].detach().numpy())
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()  # 深拷贝模型参数
                best_epoch = epoch
                best_threshold = thres  # 保存最佳阈值
        
        print(f'Epoch {epoch}, loss: {loss.item():.4f}, val mf1: {val_f1:.4f} (best: {best_val_f1:.4f})')
        
        # 原有保存预测概率的功能（可选）
        if args.del_ratio == 0 and epoch % 20 == 0:
            with open(f'probs_{dataset_name}_BWGNN_{epoch}_{args.homo}.pkl', 'wb') as f:
                pkl.dump(probs, f)
    
    time_end = time.time()
    print(f'Training time: {time_end - time_start:.2f}s')
    print(f'Best epoch: {best_epoch} with val mf1: {best_val_f1:.4f}')
    
    # --- 评估函数（内部定义，用于统一评估流程） ---
    def evaluate_model(model_to_eval, mask, mask_name='Set'):
        """评估模型在指定掩码上的性能"""
        model_to_eval.eval()
        with torch.no_grad():
            logits = model_to_eval(features)
            probs = logits.softmax(1)
            
            # 使用验证集最佳阈值进行预测
            preds = torch.zeros_like(labels)
            preds[probs[:, 1] > best_threshold] = 1
            
            # 计算各种指标
            y_true = labels[mask].numpy()
            y_prob = probs[mask][:, 1].numpy()
            y_pred = preds[mask].numpy()
            
            # 核心指标：AUC 和 AP
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            
            # 其他分类指标（可选）
            f1 = f1_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            
            print(f'\n{mask_name} Evaluation Results:')
            print(f'  AUC: {auc:.4f}')
            print(f'  AP:  {ap:.4f}')
            print(f'  Macro-F1: {f1:.4f}')
            print(f'  Recall:   {rec:.4f}')
            print(f'  Precision:{pre:.4f}')
            
            return auc, ap
    
    # --- 最终评估：加载最佳模型并分别在验证集和测试集上评估 ---
    print('=' * 60)
    print('Final Evaluation with Best Model:')
    
    # 1. 加载最佳模型参数
    model.load_state_dict(best_model_state)
    
    # 2. 评估验证集
    val_auc, val_ap = evaluate_model(model, val_mask, 'Validation')
    
    # 3. 评估测试集
    test_auc, test_ap = evaluate_model(model, test_mask, 'Test')
    
    print('=' * 60)
    
    # --- 保存结果（与原有代码兼容） ---
    # 注意：这里使用测试集上基于最佳阈值的预测来计算最终指标
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = logits.softmax(1)
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > best_threshold] = 1
        
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
    
    result = 'REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(trec*100, tpre*100, tmf1*100, tauc*100)
    print(f'\nLegacy result format: {result}')
    
    with open('result.txt', 'a+') as f:
        f.write(f'{result}\n')
    
    # 返回最佳模型的验证集和测试集 AUC、AP 供多次运行使用
    return best_val_f1, val_auc, val_ap, test_auc, test_ap


# 最佳F1阈值调整函数
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset for this model")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--del_ratio", type=float, default=0., help="delete ratios")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--load_epoch", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")

    args = parser.parse_args()
    print(args)
    
    dataset_name = args.dataset
    del_ratio = args.del_ratio
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    adj_type = args.adj_type
    load_epoch = args.load_epoch
    data_path = args.data_path
    
    graph = Dataset(load_epoch, dataset_name, del_ratio, homo, data_path, adj_type=adj_type).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    set_random_seed(717)

    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        
        # 调用修改后的训练评估函数
        val_f1, val_auc, val_ap, test_auc, test_ap = train_and_evaluate(model, graph, args)
        
        print('\n' + '='*50)
        print('Final Summary:')
        print(f'Best Validation Macro-F1: {val_f1:.4f}')
        print(f'Validation AUC: {val_auc:.4f}, AP: {val_ap:.4f}')
        print(f'Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}')
        print('='*50)
    else:
        final_val_f1s, final_val_aucs, final_val_aps, final_test_aucs, final_test_aps = [], [], [], [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            
            val_f1, val_auc, val_ap, test_auc, test_ap = train_and_evaluate(model, graph, args)
            
            final_val_f1s.append(val_f1)
            final_val_aucs.append(val_auc)
            final_val_aps.append(val_ap)
            final_test_aucs.append(test_auc)
            final_test_aps.append(test_ap)
        
        # 多次运行统计
        final_val_f1s = np.array(final_val_f1s)
        final_val_aucs = np.array(final_val_aucs)
        final_val_aps = np.array(final_val_aps)
        final_test_aucs = np.array(final_test_aucs)
        final_test_aps = np.array(final_test_aps)
        
        print('\n' + '='*60)
        print('Multiple Runs Summary:')
        print(f'Val F1 - mean: {np.mean(final_val_f1s):.4f}, std: {np.std(final_val_f1s):.4f}')
        print(f'Val AUC - mean: {np.mean(final_val_aucs):.4f}, std: {np.std(final_val_aucs):.4f}')
        print(f'Val AP - mean: {np.mean(final_val_aps):.4f}, std: {np.std(final_val_aps):.4f}')
        print(f'Test AUC - mean: {np.mean(final_test_aucs):.4f}, std: {np.std(final_test_aucs):.4f}')
        print(f'Test AP - mean: {np.mean(final_test_aps):.4f}, std: {np.std(final_test_aps):.4f}')
        print('='*60)
