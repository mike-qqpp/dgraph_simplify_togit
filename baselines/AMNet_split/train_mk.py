# -*- coding: utf-8 -*-
import argparse
import torch
import torch.optim as optim
from model import AMNet
from copy import deepcopy
from config import *
import pickle
import numpy as np
import os
import scipy.sparse as sp
import dgl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NpzGraphData:
    """直接从npz文件加载图数据"""
    def __init__(self, data_path, device='cpu'):
        print("从 {} 加载数据...".format(data_path))
        
        if not os.path.exists(data_path):
            raise FileNotFoundError("数据文件不存在: {}".format(data_path))
        
        data = np.load(data_path, allow_pickle=True)
        
        self.x = torch.FloatTensor(data['x'])
        self.y = torch.LongTensor(data['y'])
        self.edge_index = torch.LongTensor(data['edge_index'])
        
        self.train_mask = torch.BoolTensor(data['train_mask'])
        self.val_mask = torch.BoolTensor(data['valid_mask'])
        self.test_mask = torch.BoolTensor(data['test_mask'])
        
        self.num_nodes = self.x.shape[0]
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()
        self.g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        
        self.g.ndata['feature'] = self.x
        self.g.ndata['label'] = self.y
        
        self.device = torch.device(device)
        
        print("数据加载成功:")
        print("  节点数: {:,}, 特征维度: {}".format(self.num_nodes, self.x.shape[1]))
        print("  边数: {:,}".format(self.g.num_edges()))
        print("  训练集大小: {:,}".format(self.train_mask.sum().item()))
        print("  验证集大小: {:,}".format(self.val_mask.sum().item()))
        print("  测试集大小: {:,}".format(self.test_mask.sum().item()))
        
        unique, counts = torch.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            percentage = cnt/self.num_nodes*100
            print("  类别 {}: {:,} 个节点 ({:.2f}%)".format(cls, cnt, percentage))
        
    def to(self, device):
        self.device = torch.device(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.g = self.g.to(device)
        return self

def filter_labels_01(data):
    """只保留类别0和1的节点"""
    print("\n过滤数据，只保留类别0和1的节点...")
    
    print("原始总节点数: {:,}".format(data.num_nodes))
    
    device = data.device
    data_cpu = data.to('cpu')
    
    print("检查掩码尺寸:")
    print("  data.y 尺寸: {}".format(data_cpu.y.shape))
    print("  data.train_mask 尺寸: {}".format(data_cpu.train_mask.shape))
    print("  data.val_mask 尺寸: {}".format(data_cpu.val_mask.shape))
    print("  data.test_mask 尺寸: {}".format(data_cpu.test_mask.shape))
    
    if data_cpu.train_mask.shape[0] != data_cpu.num_nodes:
        print("警告: 掩码尺寸与节点数不匹配！将重新创建随机掩码...")
        
        num_nodes = data_cpu.num_nodes
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        data_cpu.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data_cpu.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data_cpu.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data_cpu.train_mask[train_idx] = True
        data_cpu.val_mask[val_idx] = True
        data_cpu.test_mask[test_idx] = True
        
        print("重新创建的掩码尺寸:")
        print("  data.train_mask: {:,}".format(data_cpu.train_mask.sum().item()))
        print("  data.val_mask: {:,}".format(data_cpu.val_mask.sum().item()))
        print("  data.test_mask: {:,}".format(data_cpu.test_mask.sum().item()))
    
    mask_01 = (data_cpu.y == 0) | (data_cpu.y == 1)
    print("类别0和1的节点数: {:,}".format(mask_01.sum().item()))
    
    if mask_01.sum().item() == 0:
        print("错误: 没有找到类别0或1的节点！")
        return data
    
    original_num_nodes = data_cpu.num_nodes
    
    data_cpu.x = data_cpu.x[mask_01]
    data_cpu.y = data_cpu.y[mask_01]
    
    if data_cpu.train_mask.shape[0] == original_num_nodes:
        data_cpu.train_mask = data_cpu.train_mask[mask_01]
        data_cpu.val_mask = data_cpu.val_mask[mask_01]
        data_cpu.test_mask = data_cpu.test_mask[mask_01]
    else:
        print("注意: 掩码已经被过滤，保持原样")
    
    old_to_new = torch.full((original_num_nodes,), -1, dtype=torch.long)
    new_indices = torch.arange(mask_01.sum().item())
    old_to_new[mask_01] = new_indices
    
    edge_mask = mask_01[data_cpu.edge_index[0]] & mask_01[data_cpu.edge_index[1]]
    filtered_edges = data_cpu.edge_index[:, edge_mask]
    
    if filtered_edges.shape[1] > 0:
        data_cpu.edge_index = torch.stack([
            old_to_new[filtered_edges[0]],
            old_to_new[filtered_edges[1]]
        ])
    else:
        data_cpu.edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    data_cpu.num_nodes = mask_01.sum().item()
    
    if data_cpu.edge_index.shape[1] > 0:
        src = data_cpu.edge_index[0].numpy()
        dst = data_cpu.edge_index[1].numpy()
        data_cpu.g = dgl.graph((src, dst), num_nodes=data_cpu.num_nodes)
    else:
        data_cpu.g = dgl.graph(([], []), num_nodes=data_cpu.num_nodes)
    
    data_cpu.g.ndata['feature'] = data_cpu.x
    data_cpu.g.ndata['label'] = data_cpu.y
    
    print("\n过滤后数据统计:")
    print("  节点数: {:,}".format(data_cpu.num_nodes))
    print("  边数: {:,}".format(data_cpu.g.num_edges()))
    print("  训练集大小: {:,}".format(data_cpu.train_mask.sum().item()))
    print("  验证集大小: {:,}".format(data_cpu.val_mask.sum().item()))
    print("  测试集大小: {:,}".format(data_cpu.test_mask.sum().item()))
    
    if data_cpu.num_nodes > 0:
        unique, counts = torch.unique(data_cpu.y, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            percentage = cnt/data_cpu.num_nodes*100
            print("  类别 {}: {:,} 个节点 ({:.2f}%)".format(cls, cnt, percentage))
    else:
        print("警告: 过滤后没有节点剩余！")
    
    return data_cpu.to(device)

def train(data, model, criterion, optimizer, label, beta=.5):
    anomaly, normal = label
    idx_train = data.train_mask
    
    model.train()
    optimizer.zero_grad()
    
    result = model(
        data.x, 
        data.edge_index, 
        label=(data.train_mask & anomaly, data.train_mask & normal)
    )
    
    if isinstance(result, tuple):
        output = result[0]
        bias_loss = result[1] if len(result) > 1 else 0.0
    else:
        output = result
        bias_loss = 0.0
    
    loss_train = criterion(output[idx_train], data.y[idx_train]) + bias_loss * beta
    
    loss_train.backward()
    optimizer.step()
    
    return loss_train.item()

def evaluate_model(model, data, mask):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        result = model(data.x, data.edge_index)
        
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        
        if mask is not None and mask.sum().item() > 0:
            output_masked = output[mask]
            y_masked = data.y[mask]
            
            y_pred_prob = torch.softmax(output_masked, dim=1)[:, 1].cpu().numpy()
            y_true = y_masked.cpu().numpy()
            
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            if len(np.unique(y_true)) < 2:
                print("警告: 掩码中只有一个类别，无法计算AUC")
                return 0.0, 0.0
            
            try:
                auc_roc = roc_auc_score(y_true, y_pred_prob)
                auc_pr = average_precision_score(y_true, y_pred_prob)
            except Exception as e:
                print(f"计算指标时出错: {e}")
                return 0.0, 0.0
        else:
            print("警告: 掩码为空或没有节点")
            return 0.0, 0.0
    
    return auc_roc, auc_pr

def main(args, exp_num=0):
    print("\n实验 {}/{}".format(exp_num+1, args.exp_num))
    print("-" * 50)
    
    seed = args.seed + exp_num
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    data = NpzGraphData(args.data_path, device='cpu')
    data = filter_labels_01(data)
    
    if data.num_nodes == 0:
        print("错误: 过滤后数据为空！")
        return 0.0, 0.0, 0.0, 0.0
    
    data = data.to(device)
    
    in_channels = data.x.shape[1]
    hid_channels = args.hidden_channels
    num_class = 2
    K = args.M
    filter_num = args.K

    net = AMNet(
        in_channels=in_channels, 
        hid_channels=hid_channels, 
        num_class=num_class,
        K=K, 
        filter_num=filter_num
    )
    net.to(device)
    
    print("\n模型配置:")
    print("  输入特征: {}".format(in_channels))
    print("  隐藏维度: {}".format(hid_channels))
    print("  滤波器数量: {}".format(K))
    print("  滤波器阶数: {}".format(filter_num))
    print("  总参数量: {:,}".format(sum(p.numel() for p in net.parameters())))

    optimizer = optim.Adam([
        dict(params=net.filters.parameters(), lr=args.lr_f),
        dict(params=net.lin, lr=args.lr, weight_decay=args.weight_decay),
        dict(params=net.attn, lr=args.lr, weight_decay=args.weight_decay)]
    )
    
    print("\n优化器配置:")
    print("  滤波器学习率: {}".format(args.lr_f))
    print("  其他参数学习率: {}".format(args.lr))
    print("  权重衰减: {}".format(args.weight_decay))

    num_normal = (data.y == 0).sum().item()
    num_anomaly = (data.y == 1).sum().item()
    
    if args.use_class_weight:
        weight_normal = 1.0
        weight_anomaly = max(num_normal / max(num_anomaly, 1), 1.0)
        weights = torch.Tensor([weight_normal, weight_anomaly])
        print("类别权重: 正常={:.2f}, 异常={:.2f} (正常:异常 = {:,}:{:,})".format(
            weight_normal, weight_anomaly, num_normal, num_anomaly))
    else:
        weights = torch.Tensor([1., 1.])
        print("不使用类别权重 (正常:异常 = {:,}:{:,})".format(num_normal, num_anomaly))
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    
    anomaly = (data.y == 1)
    normal = (data.y == 0)
    label = (anomaly, normal)

    c = 0
    auc_pr_best = 0
    auc_roc_best = 0
    auc_roc_test_epoch = 0
    auc_pr_test_epoch = 0
    best_net = None
    best_epoch = 0

    print("\n开始训练 (epochs={}, patience={})...".format(args.epochs, args.patience))
    for epoch in range(args.epochs):
        loss = train(data, net, criterion, optimizer, label, beta=args.beta)
        
        auc_roc_val, auc_pr_val = evaluate_model(net, data, data.val_mask)
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == 0:
            print('Epoch:{:04d}\tloss:{:.4f}\tVal AUC-ROC:{:.4f}\tVal AUC-PR:{:.4f}'
                  '\tBest AUC-ROC:{:.4f}\tBest AUC-PR:{:.4f}'
                        .format(epoch + 1, loss, auc_roc_val, auc_pr_val, auc_roc_test_epoch, auc_pr_test_epoch))

        if auc_pr_val >= auc_pr_best:
            auc_pr_best = auc_pr_val
            auc_roc_best = auc_roc_val
            auc_roc_test_epoch, auc_pr_test_epoch = evaluate_model(net, data, data.test_mask)
            best_net = deepcopy(net)
            best_epoch = epoch + 1
            c = 0
        else:
            c += 1
            
        if c == args.patience:
            print('早停触发! 在 epoch {} 停止训练'.format(epoch+1))
            print('最佳模型在 epoch {}'.format(best_epoch))
            break

    print("\n最终评估最佳模型 (来自 epoch {})...".format(best_epoch))
    if best_net is not None:
        auc_roc_val_exp, auc_pr_val_exp = evaluate_model(best_net, data, data.val_mask)
        auc_roc_test_exp, auc_pr_test_exp = evaluate_model(best_net, data, data.test_mask)
    else:
        auc_roc_val_exp, auc_pr_val_exp = evaluate_model(net, data, data.val_mask)
        auc_roc_test_exp, auc_pr_test_exp = evaluate_model(net, data, data.test_mask)
    
    print("实验 {} 结果:".format(exp_num+1))
    print("  验证集 AUC-ROC: {:.5f}".format(auc_roc_val_exp))
    print("  验证集 AUC-PR: {:.5f}".format(auc_pr_val_exp))
    print("  测试集 AUC-ROC: {:.5f}".format(auc_roc_test_exp))
    print("  测试集 AUC-PR: {:.5f}".format(auc_pr_test_exp))
    
    return auc_roc_val_exp, auc_pr_val_exp, auc_roc_test_exp, auc_pr_test_exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMNet 图异常检测训练')

    parser.add_argument('--data_path', type=str, required=True, help='.npz数据文件路径')
    parser.add_argument('--hidden_channels', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--M', type=int, default=8, help='滤波器数量')
    parser.add_argument('--K', type=int, default=2, help='滤波器阶数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--lr_f', type=float, default=1e-1, help='滤波器学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--beta', type=float, default=0.5, help='偏差损失权重')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--exp_num', type=int, default=10, help='实验重复次数')
    parser.add_argument('--eval_interval', type=int, default=10, help='评估间隔')
    parser.add_argument('--use_class_weight', action='store_true', help='使用类别权重')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("AMNet 训练配置")
    print("="*60)
    print("数据文件: {}".format(args.data_path))
    print("模型参数:")
    print("  隐藏维度: {}".format(args.hidden_channels))
    print("  滤波器数量 (M): {}".format(args.M))
    print("  滤波器阶数 (K): {}".format(args.K))
    print("训练参数:")
    print("  学习率: {}".format(args.lr))
    print("  滤波器学习率: {}".format(args.lr_f))
    print("  权重衰减: {}".format(args.weight_decay))
    print("  epochs: {}".format(args.epochs))
    print("  早停耐心值: {}".format(args.patience))
    print("  偏差权重 (beta): {}".format(args.beta))
    print("实验参数:")
    print("  重复次数: {}".format(args.exp_num))
    print("  评估间隔: {}".format(args.eval_interval))
    print("  使用类别权重: {}".format(args.use_class_weight))
    print("  随机种子: {}".format(args.seed))
    print("="*60)

    if not os.path.exists(args.data_path):
        print("错误: 文件不存在 - {}".format(args.data_path))
        exit(1)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("错误: 需要安装scikit-learn库")
        print("请运行: pip install scikit-learn")
        exit(1)

    # 分别记录验证集和测试集的指标
    val_auc_roc_list = []
    val_auc_pr_list = []
    test_auc_roc_list = []
    test_auc_pr_list = []

    for i in range(args.exp_num):
        print("\n" + "="*60)
        print("实验 {}/{}".format(i+1, args.exp_num))
        print("="*60)
        
        auc_roc_val, auc_pr_val, auc_roc_test, auc_pr_test = main(args, exp_num=i)
        
        val_auc_roc_list.append(auc_roc_val)
        val_auc_pr_list.append(auc_pr_val)
        test_auc_roc_list.append(auc_roc_test)
        test_auc_pr_list.append(auc_pr_test)
        
        print("实验 {} 完成:".format(i+1))
        print("  Valid: AUC-ROC={:.5f}, AUC-PR={:.5f}".format(auc_roc_val, auc_pr_val))
        print("  Test:  AUC-ROC={:.5f}, AUC-PR={:.5f}".format(auc_roc_test, auc_pr_test))

    # 输出统计结果
    print("\n" + "="*60)
    print("最终统计结果")
    print("="*60)
    print("数据集: {}".format(args.data_path))
    print("实验次数: {}".format(args.exp_num))
    
    print("\n验证集 (Valid) 结果:")
    print("  AUC-ROC: {:.5f} ± {:.5f}".format(np.mean(val_auc_roc_list), np.std(val_auc_roc_list)))
    print("  AUC-PR:  {:.5f} ± {:.5f}".format(np.mean(val_auc_pr_list), np.std(val_auc_pr_list)))
    
    print("\n测试集 (Test) 结果:")
    print("  AUC-ROC: {:.5f} ± {:.5f}".format(np.mean(test_auc_roc_list), np.std(test_auc_roc_list)))
    print("  AUC-PR:  {:.5f} ± {:.5f}".format(np.mean(test_auc_pr_list), np.std(test_auc_pr_list)))
    
    # 打印详细结果
    print("\n详细结果:")
    print("{:^6} | {:^12} {:^12} | {:^12} {:^12}".format(
        "Exp", "Val AUC-ROC", "Val AUC-PR", "Test AUC-ROC", "Test AUC-PR"))
    print("-" * 62)
    for i in range(args.exp_num):
        print("{:^6} | {:^12.5f} {:^12.5f} | {:^12.5f} {:^12.5f}".format(
            i+1, val_auc_roc_list[i], val_auc_pr_list[i], 
            test_auc_roc_list[i], test_auc_pr_list[i]))
    
    # 保存结果到文件
    if args.exp_num > 1:
        results_file = "results_{}.txt".format(os.path.basename(args.data_path).replace('.npz', ''))
        with open(results_file, 'w') as f:
            f.write("数据集: {}\n".format(args.data_path))
            f.write("实验次数: {}\n\n".format(args.exp_num))
            
            f.write("验证集 (Valid) 结果:\n")
            f.write("  AUC-ROC: {:.5f} ± {:.5f}\n".format(np.mean(val_auc_roc_list), np.std(val_auc_roc_list)))
            f.write("  AUC-PR:  {:.5f} ± {:.5f}\n\n".format(np.mean(val_auc_pr_list), np.std(val_auc_pr_list)))
            
            f.write("测试集 (Test) 结果:\n")
            f.write("  AUC-ROC: {:.5f} ± {:.5f}\n".format(np.mean(test_auc_roc_list), np.std(test_auc_roc_list)))
            f.write("  AUC-PR:  {:.5f} ± {:.5f}\n\n".format(np.mean(test_auc_pr_list), np.std(test_auc_pr_list)))
            
            f.write("详细结果:\n")
            f.write("{:^6} | {:^12} {:^12} | {:^12} {:^12}\n".format(
                "Exp", "Val AUC-ROC", "Val AUC-PR", "Test AUC-ROC", "Test AUC-PR"))
            f.write("-" * 62 + "\n")
            for i in range(args.exp_num):
                f.write("{:^6} | {:^12.5f} {:^12.5f} | {:^12.5f} {:^12.5f}\n".format(
                    i+1, val_auc_roc_list[i], val_auc_pr_list[i], 
                    test_auc_roc_list[i], test_auc_pr_list[i]))
        
        print("\n结果已保存到: {}".format(results_file))
