# 内置
import os
import warnings
# dgl
import dgl
import hydra
# 机器学习
import numpy as np
# torch
import torch
import torch.nn.functional as F
# 工程化、自建和其他
import wandb
from dgl import function as fn
from dgl.data.utils import load_graphs
from dgl.utils import expand_as_pair, dgl_warning
from omegaconf import DictConfig, OmegaConf
# pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.metrics import (roc_auc_score, average_precision_score)
from torch import nn
from tqdm import tqdm

from myutils import describe, mask_to_index, set_all_seed, cal_metrics, bin_encoding2

warnings.filterwarnings("ignore")
print(os.getcwd())


class IntraConv_single(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 add_self=True,
                 bias=True,
                 norm=None,
                 activation=None):
        super(IntraConv_single, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype=None, edge_weight=None):
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.srcdata['degree'] = torch.ones((graph.num_src_nodes(), 1)).to(feat.device)
                graph.edata['_edge_weight'] = edge_weight
                msg_fn1 = fn.u_mul_e('h', '_edge_weight', 'm')
                msg_fn2 = fn.u_mul_e('degree', '_edge_weight', 'degree')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            if edge_weight is not None:
                graph.update_all(msg_fn1, fn.sum('m', 'neigh'))
                graph.update_all(msg_fn2, fn.sum('degree', 'degree'))
                h_neigh = graph.dstdata['neigh'] / (graph.dstdata['degree'] + torch.FloatTensor([1e-8]).to(feat.device))
            else:
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']

            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
            h_self = self.fc_self(h_self)
            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh
            # bias term
            if self.bias is not None:
                rst = rst + self.bias
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class IntraConv_multi(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 add_self=True,
                 bias=True,
                 norm=None,
                 activation=None):
        super(IntraConv_multi, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype, edge_weight=None):
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges(etype=etype)
                graph.srcdata['degree'] = torch.ones((graph.num_src_nodes(), 1)).to(feat.device)
                graph.edata['_edge_weight'] = {etype: edge_weight}
                msg_fn1 = fn.u_mul_e('h', '_edge_weight', 'm')
                msg_fn2 = fn.u_mul_e('degree', '_edge_weight', 'degree')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats
            msg_fn = fn.copy_u('h', 'm')
            # Message Passing
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            if edge_weight is not None:
                graph.multi_update_all({
                    etype: (msg_fn1, fn.sum('m', 'neigh'))
                },
                    'sum'
                )
                graph.multi_update_all({
                    etype: (msg_fn2, fn.sum('degree', 'degree'))
                },
                    'sum'
                )
                h_neigh = graph.dstdata['neigh'] / (graph.dstdata['degree'] + torch.FloatTensor([1e-8]).to(feat.device))
            else:
                graph.multi_update_all({
                    etype: (msg_fn, fn.mean('m', 'neigh'))
                },
                    'sum'
                )

                h_neigh = graph.dstdata['neigh']

            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
            h_self = self.fc_self(h_self)
            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh
            # bias term
            if self.bias is not None:
                rst = rst + self.bias
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


# 构建dataloader
class DataModule(LightningDataModule):
    def __init__(self, graph, batch_size, n_classes):
        super().__init__()

        trn_sampler = dgl.dataloading.NeighborSampler(
            [-1]
        )
        val_sampler = dgl.dataloading.NeighborSampler(
            [-1]
        )
        self.g = graph
        # 只选择标签为0或1的节点进行训练和验证
        self.trn_idx = mask_to_index(graph.ndata['trn_msk'])
        self.val_idx = mask_to_index(graph.ndata['val_msk'])
        self.tst_idx = mask_to_index(graph.ndata['tst_msk'])
        
        # 过滤训练集，只保留标签为0,1的节点
        label_mask = (graph.ndata['label'][self.trn_idx] == 0) | (graph.ndata['label'][self.trn_idx] == 1)
        self.trn_idx = self.trn_idx[label_mask]
        
        self.trn_sampler = trn_sampler
        self.val_sampler = val_sampler
        self.batch_size = batch_size
        self.n_classes = n_classes

    def train_dataloader(self):
        loader = dgl.dataloading.DataLoader(
            self.g,
            self.trn_idx.to('cuda'),
            self.trn_sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )
        return loader

    def val_dataloader(self):
        # 验证时使用所有节点，但在验证步骤中会过滤
        loader = dgl.dataloading.DataLoader(
            self.g,
            torch.arange(self.g.num_nodes()).to('cuda'),
            self.val_sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )
        return loader


class DGA(nn.Module):
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes, p=0.3, n_head=1,
                  unclear_up=0.1, unclear_down=0.1):
        """Initialize the SAGE model with the given parameters."""
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('super_mask', torch.ones((num_nodes, self.n_classes)))
        self.n_head = n_head

        # 特征编码
        hidden_units = [n_hidden, n_hidden]
        input_size = in_feats
        hidden_unit = input_size
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(nn.Dropout(p))
            all_layers.append(layer)
            all_layers.append(nn.BatchNorm1d(hidden_unit))  # 加入bn
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
            self.last_dim = hidden_unit
        self.emb_layer = nn.Sequential(*all_layers)

        # 分组器损失
        all_layers = []
        all_layers.append(nn.Linear(n_hidden, self.n_classes))
        self.emb_layer_fc = nn.Sequential(*all_layers)

        # 输出层
        all_layers = []
        all_layers.append(nn.Linear(self.n_head * n_hidden, n_hidden // 2))
        all_layers.append(nn.ReLU())
        all_layers.append(nn.Linear(n_hidden // 2, self.n_classes))
        self.final_fc_layer = nn.Sequential(*all_layers)

        self.attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)
        self.W_x = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)

        self.reset_parameters()
        #
        if n_etypes == 1:
            intra_conv = IntraConv_single
        else:
            intra_conv = IntraConv_multi

        dgas = []
        for r in range(self.n_etypes):
            m = nn.ModuleDict({
                'all': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(),
                                  bias=False),
                'gp0': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(),
                                  bias=False, add_self=False),
                'gp1': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(),
                                  bias=False, add_self=False)
            })
            dgas.append(m)
        self.dgas = nn.ModuleList(dgas)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)

    def dynamic_grouping(self, mask, block, unclear_down, unclear_up):
        mask0 = (mask[:, 1] <= unclear_down)[block.srcdata[dgl.NID][block.edges()[0]]].float()
        mask1 = (mask[:, 1] > unclear_up)[block.srcdata[dgl.NID][block.edges()[0]]].float()
        return mask0, mask1

    def forward(self, blocks, x):
        """Forward pass of the SAGE model."""
        batch_size = blocks[-1].dstdata['feat'].shape[0]
        x = self.emb_layer(x)
        emb_out = self.emb_layer_fc(x)

        mask0_dict = {}
        mask1_dict = {}
        block = blocks[0]

        for etype in block.etypes:
            mask0_dict[etype], mask1_dict[etype] = \
                self.dynamic_grouping(self.super_mask, block.edge_type_subgraph(etypes=[etype]),
                                      self.unclear_up, self.unclear_down)

        h_list = []
        for idx, etype in enumerate(block.etypes):
            h_list.append(self.dgas[idx]['all'](block, x, etype))
            h_list.append(self.dgas[idx]['gp0'](block, x, etype, mask0_dict[etype]))
            h_list.append(self.dgas[idx]['gp1'](block, x, etype, mask1_dict[etype]))

        s_len = len(h_list)
        h_list = torch.stack(h_list, dim=1)

        h_list_proj = self.W_f(h_list).view(batch_size, s_len, self.n_head, self.n_hidden)
        h_list_proj = h_list_proj.permute(0, 2, 1, 3).contiguous().view(-1, s_len, self.n_hidden)

        x_proj = self.W_x(x[:batch_size]).view(batch_size, self.n_head, self.n_hidden, 1)
        x_proj = x_proj.view(-1, self.n_hidden, 1)

        attention_logit = torch.bmm(h_list_proj, x_proj)
        soft_attention = F.softmax(attention_logit, dim=1).transpose(1, 2)
        h_list_rep = h_list.repeat([self.n_head, 1, 1])
        weighted_features = torch.bmm(soft_attention, h_list_rep).squeeze(-2)
        h = weighted_features.view(batch_size, -1)
        o = self.final_fc_layer(h)

        return o, emb_out[:batch_size]


# 构建网络结构
class pl_DGA(LightningModule):
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes,
                 lr=1e-3, weight_decay=5e-4, p=0.3, n_head=1, w=None,
                 unclear_up=0.1, unclear_down=0.1,
                 trn_idx=None, val_idx=None, tst_idx=None):
        super().__init__()
        self.save_hyperparameters()
        self.n_etypes = n_etypes
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dga = DGA(in_feats[0], n_hidden, num_nodes, n_classes, n_etypes, p, n_head, unclear_up, unclear_down)
        self.trn_idx = trn_idx
        self.val_idx = val_idx
        self.tst_idx = tst_idx
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('w', torch.FloatTensor(w))
        self.ps = []
        self.output_nodes_history = []  # 保存历史输出节点

    def forward(self, blocks, x):
        o, emb_out = self.dga(blocks, x)
        return o, emb_out

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]  # 使用真实标签
        
        # 只选择标签为0或1的节点进行训练
        valid_mask = (y == 0) | (y == 1)
        if not valid_mask.any():
            return None
            
        valid_output_nodes = output_nodes[valid_mask]
        valid_y = y[valid_mask]
        
        # 如果这个batch中没有有效标签，跳过
        if len(valid_y) == 0:
            return None
            
        logits, emb_logits = self(blocks, x)
        valid_logits = logits[valid_mask]
        
        loss = F.cross_entropy(valid_logits, valid_y, self.w)
        self.log("trn_loss0", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(valid_output_nodes))
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        
        # 只选择标签为0或1的节点进行验证
        valid_mask = (y == 0) | (y == 1)
        if not valid_mask.any():
            return None, None, None
            
        valid_output_nodes = output_nodes[valid_mask]
        valid_y = y[valid_mask]
        
        logits, emb_logits = self(blocks, x)
        valid_logits = logits[valid_mask]
        
        loss = F.cross_entropy(valid_logits, valid_y, self.w)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(valid_output_nodes))
        return valid_logits, valid_y, valid_output_nodes

    def validation_epoch_end(self, outs):
        """Calculate validation AUC at the end of the epoch."""
        # 过滤掉None值
        outs = [x for x in outs if x is not None and x[0] is not None]
        if len(outs) == 0:
            return
            
        # 收集所有验证结果
        all_logits = torch.cat([x[0] for x in outs])
        all_y = torch.cat([x[1] for x in outs]).cpu().numpy()
        all_output_nodes = torch.cat([x[2] for x in outs]).cpu().numpy()
        
        # 获取全局节点ID
        all_probs = all_logits.softmax(-1).cpu().numpy()
        all_prob_1 = all_probs[:, 1]
        
        if self.trainer.sanity_checking:
            # 如果是sanity check，不进行完整计算
            prob = all_prob_1
        else:
            # 保存当前epoch的结果
            self.ps.append((all_probs, all_output_nodes))
            
            # 计算翻转率
            if len(self.ps) > 1:
                prev_probs, prev_nodes = self.ps[-2]
                prev_prob_1 = prev_probs[:, 1]
                
                # 找到共同的节点
                current_nodes_set = set(all_output_nodes)
                prev_nodes_set = set(prev_nodes)
                common_nodes = list(current_nodes_set & prev_nodes_set)
                
                if len(common_nodes) > 0:
                    # 创建映射
                    curr_node_to_idx = {node: idx for idx, node in enumerate(all_output_nodes)}
                    prev_node_to_idx = {node: idx for idx, node in enumerate(prev_nodes)}
                    
                    curr_indices = [curr_node_to_idx[node] for node in common_nodes]
                    prev_indices = [prev_node_to_idx[node] for node in common_nodes]
                    
                    flip_rate = ((prev_prob_1[prev_indices] <= self.unclear_down) ^ 
                               (all_prob_1[curr_indices] <= self.unclear_down)).mean()
                    self.log('flip_r', flip_rate, prog_bar=True, on_step=False, on_epoch=True)
            else:
                self.log('flip_r', 0.0, prog_bar=True, on_step=False, on_epoch=True)
            
            # 更新super_mask - 只更新当前验证过的节点
            current_super_mask = self.dga.super_mask.clone().cpu().numpy()
            current_super_mask[all_output_nodes] = all_probs
            self.dga.super_mask.copy_(torch.FloatTensor(current_super_mask))
            
            # 计算指标
            prob = all_prob_1
            
            # 找到当前batch中属于训练/验证/测试集的节点
            batch_trn_mask = np.isin(all_output_nodes, self.trn_idx.cpu().numpy())
            batch_val_mask = np.isin(all_output_nodes, self.val_idx.cpu().numpy())
            batch_tst_mask = np.isin(all_output_nodes, self.tst_idx.cpu().numpy())
            
            if batch_trn_mask.any():
                trn_auc = roc_auc_score(all_y[batch_trn_mask], prob[batch_trn_mask])
                trn_aps = average_precision_score(all_y[batch_trn_mask], prob[batch_trn_mask])
                self.log("trn_auc", trn_auc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("trn_aps", trn_aps, prog_bar=True, on_step=False, on_epoch=True)
            
            if batch_val_mask.any():
                val_auc = roc_auc_score(all_y[batch_val_mask], prob[batch_val_mask])
                val_aps = average_precision_score(all_y[batch_val_mask], prob[batch_val_mask])
                self.log("val_auc", val_auc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_aps", val_aps, prog_bar=True, on_step=False, on_epoch=True)
            
            if batch_tst_mask.any():
                tst_auc = roc_auc_score(all_y[batch_tst_mask], prob[batch_tst_mask])
                tst_aps = average_precision_score(all_y[batch_tst_mask], prob[batch_tst_mask])
                self.log("tst_auc", tst_auc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("tst_aps", tst_aps, prog_bar=True, on_step=False, on_epoch=True)
            
            # 记录分组统计
            self.log('g0', (prob <= self.unclear_down).sum() / len(prob), prog_bar=True, on_step=False, on_epoch=True)
            self.log('g1', (prob > self.unclear_up).sum() / len(prob), prog_bar=True, on_step=False, on_epoch=True)
            self.log("pmean", prob.mean(), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        optimizer = torch.optim.Adam(self.dga.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        """Inference function for the model."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size * 5,
            shuffle=False,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )

        if buffer_device is None:
            buffer_device = device

        y = torch.zeros(
            g.num_nodes(),
            self.n_classes,
            device=buffer_device,
        )
        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            x = blocks[0].srcdata["feat"]
            logits, emb_logits = self(blocks, x)
            y[output_nodes] = logits.to(buffer_device)
        return y


@hydra.main(config_path="configs", config_name="amazon", version_base=None)
def run(args: DictConfig):
    # 设置GPU, 设置随机种子，修改模型名称
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
    accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
    set_all_seed(args.seed)
    args.model = f'{args.model}'
    suffix = '_bin' if args.bin_encoding else ''
    args.model = args.model + suffix

    # logger设置在读取数据之前，可以保存describe的信息
    mode = 'offline' #if args.nowandb else 'online'
    wandb_run = wandb.init(project=args.project, config=OmegaConf.to_container(args), mode=mode)

    # 读取数据，输出图的统计信息
    DATA_PATH = '../data/processed/'
    graph, split_dict = load_graphs(DATA_PATH + args.dname + '.dgldata')
    graph = graph[0]
    y = graph.ndata['label'].cpu().numpy()
    
    # 确保只使用0,1标签
    valid_label_mask = (y == 0) | (y == 1)
    print(f"有效标签节点数量: {valid_label_mask.sum()}/{len(y)}")
    print(f"标签分布 - 0: {(y == 0).sum()}, 1: {(y == 1).sum()}, 其他: {len(y) - valid_label_mask.sum()}")
    
    graph.ndata['aux_label'] = graph.ndata['label']
    w = [1, 1]
    n_classes = 2

    n_etypes = len(graph.etypes)

    trn_idx, val_idx, tst_idx = mask_to_index(graph.ndata['trn_msk']), mask_to_index(
        graph.ndata['val_msk']), mask_to_index(graph.ndata['tst_msk'])
    
    # 过滤训练集，只保留标签为0,1的节点
    trn_label_mask = (graph.ndata['label'][trn_idx] == 0) | (graph.ndata['label'][trn_idx] == 1)
    trn_idx = trn_idx[trn_label_mask]
    
    # 信息打印
    print("==" * 20)
    print("数据名称", args.dname)
    describe(graph)
    print("==" * 20)
    print("超参数设定：")
    print(OmegaConf.to_yaml(args))
    print("n_etypes", n_etypes)
    print("n_classes", n_classes)
    print(f"训练节点数: {len(trn_idx)}, 验证节点数: {len(val_idx)}, 测试节点数: {len(tst_idx)}")
    print("==" * 20)

    if args.bin_encoding:
        feature = bin_encoding2(graph, trn_idx, n_bins=args.k)
        graph.ndata['feat'] = torch.FloatTensor(feature.values).contiguous()
        print("after bin_encoding：", feature.shape)
        print("==" * 20)
    else:
        feature = graph.ndata['feat']
        graph.ndata['feat'] = feature.contiguous()
        print("原始特征维度：", feature.shape)

    in_feats = [graph.ndata['feat'].shape[1]]
    unclear_up = unclear_down = args.z
    wandb.log({
        'unclear_up': unclear_up,
        'unclear_down': unclear_down
    })
    print({'unclear_up': unclear_up, 'unclear_down': unclear_down})
    
    # 构建 datamodule 和 model
    datamodule = DataModule(graph, args.bs, n_classes)
    model = pl_DGA(in_feats, args.n_hidden, graph.num_nodes(), n_classes=n_classes,
                   n_etypes=n_etypes, lr=args.lr, weight_decay=args.weight_decay, p=args.p,
                   n_head=args.n_head, w=w,
                   unclear_up=unclear_up, unclear_down=unclear_down,
                   trn_idx=trn_idx, val_idx=val_idx, tst_idx=tst_idx)

    # 进行训练前的设置，timer，checkpoint，early_stopping，logger
    timer = Timer()
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, verbose=False)
    early_stopping = EarlyStopping('val_loss', verbose=False, mode='min', patience=args.patience)
    logger = WandbLogger(wandb=wandb_run)

    trainer = Trainer(
        accelerator=accelerator, devices=[args.gpuid],
        max_epochs=args.max_epochs,
        logger=logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, early_stopping, timer],
    )
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    print(f'training time elapsed {timer.time_elapsed("train"):.2f}s')

    # 读取最优checkpoint 并且进行推理
    print("Evaluating model in", trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        y_hat = model.inference(graph, device, 5120, 0, "cpu")
    prob = y_hat.softmax(-1).cpu().numpy()[:, 1]
    
    # 只对标签为0,1的节点进行评估
    valid_label_mask = (y == 0) | (y == 1)
    valid_trn_idx = trn_idx.cpu().numpy() if torch.is_tensor(trn_idx) else trn_idx
    valid_val_idx = val_idx.cpu().numpy() if torch.is_tensor(val_idx) else val_idx
    valid_tst_idx = tst_idx.cpu().numpy() if torch.is_tensor(tst_idx) else tst_idx
    
    # 确保评估的节点都有有效标签
    valid_trn_idx = [idx for idx in valid_trn_idx if valid_label_mask[idx]]
    valid_val_idx = [idx for idx in valid_val_idx if valid_label_mask[idx]]
    valid_tst_idx = [idx for idx in valid_tst_idx if valid_label_mask[idx]]
    
    dic = cal_metrics(prob, y, valid_trn_idx, valid_val_idx, valid_tst_idx, verbose=True)
    wandb.log(dic)
    wandb.finish()
    print("===" * 10)
    print("===" * 10)


if __name__ == "__main__":
    run()
