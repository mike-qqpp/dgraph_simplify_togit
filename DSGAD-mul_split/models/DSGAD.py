from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as dglF
import scipy
import sympy
import sklearn.metrics as skmetrics
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModel, get_thres
import datasets
import os
import time

def calculate_theta2(d: int) -> list[list[float]]:
    '''Calculate the polynomial coefficients of the filter'''
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2)**i * (1 - x / 2)**(d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


def poly_conv(theta: list[float], g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
    '''Polynomial convolution using filter coefficients'''

    def unnLaplacian(feat, D_invsqrt, graph):
        """ Operation Feat * D^-1/2 A D^-1/2 """
        graph.ndata['h'] = feat * D_invsqrt
        graph.update_all(dglF.copy_u('h', 'm'), dglF.sum('m', 'h'))
        return feat - graph.ndata.pop('h') * D_invsqrt

    with g.local_scope():
        D_invsqrt = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(features.device)
        h = theta[0] * features
        for k in range(1, len(theta)):
            features = unnLaplacian(features, D_invsqrt, g)
            h += theta[k] * features
    return h


class DSGAD(BaseModel):

    def __init__(
            self,
            in_nodes: int,
            in_feats: int,
            h_feats: int = 8,
            num_classes: int = 2,
            d=2,
            mix_beta: int = 2,
            device_id: int = 0,
            num_gpus: int = 1,
            use_reduced_memory: bool = True,
    ):
        super().__init__()
        self.device_id = device_id
        self.num_gpus = num_gpus
        self.total_nodes = in_nodes
        self.h_feats = h_feats
        self.use_reduced_memory = use_reduced_memory
        
        # 计算当前GPU负责的节点范围
        nodes_per_gpu = in_nodes // num_gpus
        self.start_idx = device_id * nodes_per_gpu
        self.end_idx = (device_id + 1) * nodes_per_gpu if device_id < num_gpus - 1 else in_nodes
        self.num_local_nodes = self.end_idx - self.start_idx
        
        print(f"GPU {device_id}: 负责节点 {self.start_idx:,} 到 {self.end_idx:,} (共 {self.num_local_nodes:,} 个节点)")

        self.thetas = calculate_theta2(d)
        self.num_filters = len(self.thetas)

        self.input = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
        )

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
        ) for _ in range(self.num_filters + mix_beta)])

        c = self.num_filters + mix_beta
        ks = 3
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(c, c, ks, stride, 'same'),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Conv1d(c, c, ks, stride, 'same'),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(c * h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, num_classes),
        )

        # 内存优化版本
        if use_reduced_memory:
            reduced_dim = 1000
            self.weights = nn.Parameter(
                torch.randn(mix_beta, self.num_filters, reduced_dim, h_feats)
            )
            print(f"GPU {device_id}: 使用内存优化权重，形状 {self.weights.shape}")
        else:
            self.weights = nn.Parameter(
                torch.randn(mix_beta, self.num_filters, self.num_local_nodes, h_feats)
            )
            print(f"GPU {device_id}: 权重形状 {self.weights.shape}")

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor, node_indices=None):
        """
        前向传播
        node_indices: 要处理的节点索引（相对于全局节点）
        """
        # 第一步：输入层处理全图特征
        h_full = self.input(in_feat)
        
        if node_indices is not None:
            # 训练模式：只处理指定的batch节点
            
            # 获取当前GPU负责的节点范围的特征
            local_h = h_full[self.start_idx:self.end_idx]
            
            # 权重计算
            if self.use_reduced_memory:
                num_samples = min(1000, self.num_local_nodes)
                sample_indices = torch.randperm(self.num_local_nodes)[:num_samples]
                sampled_h = local_h[sample_indices]
                
                X = sampled_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            else:
                X = local_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            
            # 图卷积
            h_conv_list = []
            for theta in self.thetas:
                h_conv = poly_conv(theta, g, h_full)
                h_conv_list.append(h_conv)
            
            # 只提取batch节点的卷积结果
            h_batch_list = []
            for h_conv in h_conv_list:
                h_batch = h_conv[node_indices]
                h_batch_list.append(h_batch)
            
            # 混合滤波器
            if self.weights.shape[0] > 0:
                mix_list = []
                for w in weights.softmax(1):
                    mixed = sum([h_batch_list[i] * w[i] for i in range(self.num_filters)])
                    mix_list.append(mixed)
                h_batch_list += mix_list
            
            # 后续处理
            h_fc_list = []
            for i, h_batch in enumerate(h_batch_list):
                h_fc = self.fcs[i](h_batch)
                h_fc_list.append(h_fc)
            
            # 合并通道
            h_concat_list = [h.unsqueeze(1) for h in h_fc_list]
            h_concat = torch.cat(h_concat_list, 1)
            
            # 卷积和分类
            h_conv_out = self.conv(h_concat)
            h_out = self.classifier(h_conv_out)
            
            return h_out
            
        else:
            # 评估模式：处理全图
            
            # 权重计算
            if self.use_reduced_memory:
                local_h = h_full[self.start_idx:self.end_idx]
                num_samples = min(1000, self.num_local_nodes)
                sample_indices = torch.randperm(self.num_local_nodes)[:num_samples]
                sampled_h = local_h[sample_indices]
                
                X = sampled_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            else:
                local_h = h_full[self.start_idx:self.end_idx]
                X = local_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            
            # 全图卷积
            h_conv_list = []
            for theta in self.thetas:
                h_conv = poly_conv(theta, g, h_full)
                h_conv_list.append(h_conv)
            
            # 混合滤波器
            if self.weights.shape[0] > 0:
                mix_list = []
                for w in weights.softmax(1):
                    mixed = sum([h_conv_list[i] * w[i] for i in range(self.num_filters)])
                    mix_list.append(mixed)
                h_conv_list += mix_list
            
            # 后续处理
            h_fc_list = []
            for i, h_conv in enumerate(h_conv_list):
                h_fc = self.fcs[i](h_conv)
                h_fc_list.append(h_fc)
            
            # 合并通道
            h_concat_list = [h.unsqueeze(1) for h in h_fc_list]
            h_concat = torch.cat(h_concat_list, 1)
            
            # 卷积和分类
            h_conv_out = self.conv(h_concat)
            h_out = self.classifier(h_conv_out)
            
            return h_out

    @classmethod
    def trainfit(cls, args):
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
                model = cls(features.shape[0], features.shape[1], **gpu_config).to(device)
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
            
            main_model = cls(features.shape[0], features.shape[1], **model_config).to(device)
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
        print("训练完成，开始最终测试集评估")
        print("="*60)
        
        # ========== 加载最佳模型 ==========
        if best_model_state is not None:
            main_model.load_state_dict(best_model_state)
            print(f"已加载 epoch {best_epoch} 的最佳模型")
        else:
            print("⚠️ 未找到最佳模型，使用最后一个epoch的模型")
        
        # ========== 最终测试集评估（使用批次） ==========
        main_model.eval()
        
        # 创建测试集数据加载器
        test_idx = torch.where(test_mask)[0].to(main_device)
        test_dataset = TensorDataset(test_idx.cpu())
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            drop_last=False
        )
        
        all_test_probs = []
        all_test_labels = []
        start = time.time()
        with torch.no_grad():
            for test_batch_idx in test_loader:
                test_batch_idx = test_batch_idx[0].to(main_device)
                
                # 批次前向传播
                out = main_model(g_main, features_main, node_indices=test_batch_idx)
                
                # 收集概率和标签
                batch_probs = out.softmax(1)[:, 1].cpu()
                batch_labels = labels_main[test_batch_idx].cpu()
                
                all_test_probs.append(batch_probs)
                all_test_labels.append(batch_labels)
            
            # 合并所有批次的结果
            test_probs = torch.cat(all_test_probs)
            test_labels = torch.cat(all_test_labels)
            
            # 使用验证集找到的最佳阈值
            test_y_pred = torch.zeros_like(test_probs)
            test_y_pred[test_probs >= val_thres] = 1
            
            # 计算测试集指标
            test_auc = skmetrics.roc_auc_score(test_labels, test_probs)
            test_ap = skmetrics.average_precision_score(test_labels, test_probs)
            test_recall = skmetrics.recall_score(test_labels, test_y_pred)
            test_precision = skmetrics.precision_score(test_labels, test_y_pred)
            test_f1_macro = skmetrics.f1_score(test_labels, test_y_pred, average='macro')
            test_f1_micro = skmetrics.f1_score(test_labels, test_y_pred, average='micro')
            test_accuracy = skmetrics.accuracy_score(test_labels, test_y_pred)
        end = time.time()
        dt = round(end-start, 4)
        print('dt is {}'.format(dt) )    

        print("\n测试集最终指标:")
        print('auc     ap      recall  precision  f1_macro  f1_micro  accuracy')
        print(f'{test_auc:.5f}  {test_ap:.5f}  {test_recall:.5f}  {test_precision:.5f}  {test_f1_macro:.5f}  {test_f1_micro:.5f}  {test_accuracy:.5f}')
        print(f"使用的阈值: {val_thres:.5f} (来自验证集)")
        
        print("\n" + "="*60)
        print("训练总结:")
        print("="*60)
        print(f"最佳模型在 epoch {best_epoch}")
        print(f"验证集最佳AUC: {best_val_auc:.5f}")
        print(f"验证集最佳AP: {best_val_ap:.5f}")
        print(f"测试集最终AUC: {test_auc:.5f}")
        print(f"测试集最终AP: {test_ap:.5f}")

        return main_model, {
            'best_epoch': best_epoch,
            'val_auc': best_val_auc,
            'val_ap': best_val_ap,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro,
            'test_accuracy': test_accuracy,
            'threshold': float(val_thres),
            'auc': test_auc,
            'ap': test_ap,
            'recall': test_recall,
            'precision': test_precision,
            'f1_macro': test_f1_macro,
            'f1_micro': test_f1_micro,
            'accuracy': test_accuracy,
        }


class DSGAD_hete(DSGAD):
    """异构图像本"""
    
    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor, node_indices=None):
        # 第一步：输入层处理全图特征
        h_full = self.input(in_feat)
        
        if node_indices is not None:
            # 训练模式：只处理指定的batch节点
            
            # 获取当前GPU负责的节点范围的特征
            local_h = h_full[self.start_idx:self.end_idx]
            
            # 权重计算
            if self.use_reduced_memory:
                num_samples = min(1000, self.num_local_nodes)
                sample_indices = torch.randperm(self.num_local_nodes)[:num_samples]
                sampled_h = local_h[sample_indices]
                
                X = sampled_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            else:
                X = local_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            
            h_all = []
            for relation in g.canonical_etypes:
                # 图卷积
                h_conv_list = []
                for theta in self.thetas:
                    h_conv = poly_conv(theta, g[relation], h_full)
                    h_conv_list.append(h_conv)
                
                # 只提取batch节点的卷积结果
                h_batch_list = []
                for h_conv in h_conv_list:
                    h_batch = h_conv[node_indices]
                    h_batch_list.append(h_batch)
                
                # 混合滤波器
                if self.weights.shape[0] > 0:
                    mix_list = []
                    for w in weights.softmax(1):
                        mixed = sum([h_batch_list[i] * w[i] for i in range(self.num_filters)])
                        mix_list.append(mixed)
                    h_batch_list += mix_list
                
                # 后续处理
                h_fc_list = []
                for i, h_batch in enumerate(h_batch_list):
                    h_fc = self.fcs[i](h_batch)
                    h_fc_list.append(h_fc)
                
                # 合并通道
                h_concat_list = [h.unsqueeze(1) for h in h_fc_list]
                h_concat = torch.cat(h_concat_list, 1)
                
                # 卷积和分类
                h_conv_out = self.conv(h_concat)
                h_out = self.classifier(h_conv_out)
                
                h_all.append(h_out)
            
            # 平均所有关系的输出
            h = sum(h_all) / len(h_all)
            return h
            
        else:
            # 评估模式：处理全图
            
            # 权重计算
            if self.use_reduced_memory:
                local_h = h_full[self.start_idx:self.end_idx]
                num_samples = min(1000, self.num_local_nodes)
                sample_indices = torch.randperm(self.num_local_nodes)[:num_samples]
                sampled_h = local_h[sample_indices]
                
                X = sampled_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            else:
                local_h = h_full[self.start_idx:self.end_idx]
                X = local_h.unsqueeze(0).unsqueeze(0)
                X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
                weights = self.weights * X
                weights = weights.sum(dim=(2, 3))
            
            h_all = []
            for relation in g.canonical_etypes:
                # 全图卷积
                h_conv_list = []
                for theta in self.thetas:
                    h_conv = poly_conv(theta, g[relation], h_full)
                    h_conv_list.append(h_conv)
                
                # 混合滤波器
                if self.weights.shape[0] > 0:
                    mix_list = []
                    for w in weights.softmax(1):
                        mixed = sum([h_conv_list[i] * w[i] for i in range(self.num_filters)])
                        mix_list.append(mixed)
                    h_conv_list += mix_list
                
                # 后续处理
                h_fc_list = []
                for i, h_conv in enumerate(h_conv_list):
                    h_fc = self.fcs[i](h_conv)
                    h_fc_list.append(h_fc)
                
                # 合并通道
                h_concat_list = [h.unsqueeze(1) for h in h_fc_list]
                h_concat = torch.cat(h_concat_list, 1)
                
                # 卷积和分类
                h_conv_out = self.conv(h_concat)
                h_out = self.classifier(h_conv_out)
                
                h_all.append(h_out)
            
            # 平均所有关系的输出
            h = sum(h_all) / len(h_all)
            return h
