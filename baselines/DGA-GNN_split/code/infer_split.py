# Built-in
import os
import sys
import warnings
import time
import argparse
# dgl
import dgl
import hydra
# ML
import numpy as np
# torch
import torch
import torch.nn.functional as F
# Engineering
from dgl import function as fn
from dgl.data.utils import load_graphs
from dgl.utils import expand_as_pair, dgl_warning
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (roc_auc_score, average_precision_score)
from torch import nn
from tqdm import tqdm

from myutils import describe, mask_to_index, set_all_seed, cal_metrics, bin_encoding2

warnings.filterwarnings("ignore")
print(os.getcwd())

# 在导入后立即解析命令行参数（Hydra之前）
_CLI_ARGS = {}
def _parse_cli_args():
    """在Hydra处理之前解析自定义命令行参数"""
    parser = argparse.ArgumentParser(description='Inference with DGA Model', add_help=False)
    # 注意：--config-name 不在这里定义，让 Hydra 自己处理
    parser.add_argument('--model_path', type=str, required=True, help='Path to load model .pt file')
    parser.add_argument('--gpuid', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (use config value if not specified)')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU instead of GPU')
    
    # 只解析自定义参数，其余的留给Hydra（包括--config-name）
    known, unknown = parser.parse_known_args()
    _CLI_ARGS['model_path'] = known.model_path
    _CLI_ARGS['gpuid'] = known.gpuid
    _CLI_ARGS['batch_size'] = known.batch_size
    _CLI_ARGS['cpu'] = known.cpu
    
    return known, unknown

# 解析自定义参数
_CLI_KNOWN, _CLI_UNKNOWN = _parse_cli_args()
# 将未知参数（包含--config-name等Hydra参数）放回sys.argv
sys.argv = [sys.argv[0]] + _CLI_UNKNOWN


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
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
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

            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            lin_before_mp = self._in_src_feats > self._out_feats

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
            if self.bias is not None:
                rst = rst + self.bias
            if self.activation is not None:
                rst = self.activation(rst)
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
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
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

            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            lin_before_mp = self._in_src_feats > self._out_feats
            msg_fn = fn.copy_u('h', 'm')
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
            if self.bias is not None:
                rst = rst + self.bias
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class DGA(nn.Module):
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes, p=0.3, n_head=1,
                  unclear_up=0.1, unclear_down=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('super_mask', torch.ones((num_nodes, self.n_classes)))
        self.n_head = n_head

        hidden_units = [n_hidden, n_hidden]
        input_size = in_feats
        hidden_unit = input_size
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(nn.Dropout(p))
            all_layers.append(layer)
            all_layers.append(nn.BatchNorm1d(hidden_unit))
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
            self.last_dim = hidden_unit
        self.emb_layer = nn.Sequential(*all_layers)

        all_layers = []
        all_layers.append(nn.Linear(n_hidden, self.n_classes))
        self.emb_layer_fc = nn.Sequential(*all_layers)

        all_layers = []
        all_layers.append(nn.Linear(self.n_head * n_hidden, n_hidden // 2))
        all_layers.append(nn.ReLU())
        all_layers.append(nn.Linear(n_hidden // 2, self.n_classes))
        self.final_fc_layer = nn.Sequential(*all_layers)

        self.attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)
        self.W_x = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)

        self.reset_parameters()
        
        if n_etypes == 1:
            intra_conv = IntraConv_single
        else:
            intra_conv = IntraConv_multi

        dgas = []
        for r in range(self.n_etypes):
            m = nn.ModuleDict({
                'all': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(), bias=False),
                'gp0': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(), bias=False, add_self=False),
                'gp1': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(), bias=False, add_self=False)
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


def inference(model, g, device, batch_size, buffer_device=None):
    total_inference_time = 0.0
    inference_times = []
    use_gpu = device.type == 'cuda'
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.num_nodes()).to(device),
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        use_uva=use_gpu,
        num_workers=0,
    )

    if buffer_device is None:
        buffer_device = device

    y = torch.zeros(
        g.num_nodes(),
        model.n_classes,
        device=buffer_device,
    )
    
    print(f"Start inference, total {len(dataloader)} batches...")
    
    for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(tqdm(dataloader)):
        x = blocks[0].srcdata["feat"]
        
        if use_gpu:
            torch.cuda.synchronize()
        inference_start = time.time()
        
        logits, emb_logits = model(blocks, x)
        
        if use_gpu:
            torch.cuda.synchronize()
        
        inference_end = time.time()
        if use_gpu:
            torch.cuda.synchronize()
        
        batch_inference_time = inference_end - inference_start
        inference_times.append(batch_inference_time)
        total_inference_time += batch_inference_time
        
        y[output_nodes] = logits.to(buffer_device)
        
        if (batch_idx + 1) % 10 == 0:
            avg_time = np.mean(inference_times[-10:])
            print(f"  Batch {batch_idx+1}: inference time {batch_inference_time:.4f}s, avg {avg_time:.4f}s")
    
    print("\n" + "="*60)
    print("Inference Statistics")
    print("="*60)
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Avg batch time: {np.mean(inference_times):.4f} ± {np.std(inference_times):.4f} seconds")
    print(f"Min batch time: {np.min(inference_times):.4f} seconds")
    print(f"Max batch time: {np.max(inference_times):.4f} seconds")
    
    if use_gpu:
        print(f"\nGPU Memory Statistics:")
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")
    
    print(f"Total nodes: {g.num_nodes()}")
    print(f"Nodes per second: {g.num_nodes() / total_inference_time:.2f} nodes/s")
    print("="*60 + "\n")
    
    return y


@hydra.main(config_path="configs", config_name="amazon", version_base=None)
def run(args: DictConfig):
    # 设备选择逻辑：如果指定了--cpu则使用CPU，否则使用配置文件中的usegpu设置
    force_cpu = _CLI_ARGS.get('cpu', False)
    use_gpu = not force_cpu and torch.cuda.is_available() and args.usegpu
    device = torch.device(f'cuda:{_CLI_ARGS["gpuid"]}' if use_gpu else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Force CPU mode: {force_cpu}")
    
    set_all_seed(args.seed)
    args.model = f'{args.model}'
    suffix = '_bin' if args.bin_encoding else ''
    args.model = args.model + suffix

    model_path = _CLI_ARGS['model_path']
    print(f"Loading model file: {model_path}")
    
    DATA_PATH = '../data/processed/'
    graph, split_dict = load_graphs(DATA_PATH + args.dname + '.dgldata')
    graph = graph[0]
    y_true = graph.ndata['label'].cpu().numpy()
    
    valid_label_mask = (y_true == 0) | (y_true == 1)
    print(f"Valid label nodes: {valid_label_mask.sum()}/{len(y_true)}")
    print(f"Label distribution - 0: {(y_true == 0).sum()}, 1: {(y_true == 1).sum()}, other: {len(y_true) - valid_label_mask.sum()}")
    
    n_classes = 2

    n_etypes = len(graph.etypes)

    trn_idx, val_idx, tst_idx = mask_to_index(graph.ndata['trn_msk']), mask_to_index(
        graph.ndata['val_msk']), mask_to_index(graph.ndata['tst_msk'])
    
    trn_label_mask = (graph.ndata['label'][trn_idx] == 0) | (graph.ndata['label'][trn_idx] == 1)
    trn_idx = trn_idx[trn_label_mask]
    
    print("==" * 20)
    print("Data name", args.dname)
    describe(graph)
    print("==" * 20)
    print("Hyperparameters:")
    print(OmegaConf.to_yaml(args))
    print("n_etypes", n_etypes)
    print("n_classes", n_classes)
    print(f"Train nodes: {len(trn_idx)}, Val nodes: {len(val_idx)}, Test nodes: {len(tst_idx)}")
    print("==" * 20)

    if args.bin_encoding:
        feature = bin_encoding2(graph, trn_idx, n_bins=args.k)
        graph.ndata['feat'] = torch.FloatTensor(feature.values).contiguous()
        print("after bin_encoding:", feature.shape)
        print("==" * 20)
    else:
        feature = graph.ndata['feat']
        graph.ndata['feat'] = feature.contiguous()
        print("Original feature dim:", feature.shape)

    in_feats = [graph.ndata['feat'].shape[1]]
    unclear_up = unclear_down = args.z
    print({'unclear_up': unclear_up, 'unclear_down': unclear_down})
    
    model = DGA(in_feats[0], args.n_hidden, graph.num_nodes(), n_classes=n_classes,
                n_etypes=n_etypes, p=args.p, n_head=args.n_head,
                unclear_up=unclear_up, unclear_down=unclear_down)
    
    print(f"Loading model weights: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 确定batch_size，命令行参数优先
    batch_size = _CLI_ARGS['batch_size'] if _CLI_ARGS['batch_size'] is not None else args.bs
    print(f"Batch size: {batch_size}")
    
    # 包装模型以添加n_classes属性
    class WrappedModel:
        def __init__(self, model, n_classes):
            self.model = model
            self.n_classes = n_classes
        def __call__(self, blocks, x):
            return self.model(blocks, x)
    
    wrapped_model = WrappedModel(model, n_classes)
    
    print("\n" + "="*60)
    print("Start Inference Timer")
    print("="*60)
    total_inference_start = time.time()
    
    with torch.no_grad():
        y_hat = inference(wrapped_model, graph, device, batch_size, device)
    
    total_inference_end = time.time()
    total_inference_duration = total_inference_end - total_inference_start
    
    print("\n" + "="*60)
    print("Total Inference Statistics")
    print("="*60)
    print(f"Total inference time after model load: {total_inference_duration:.4f} seconds")
    if use_gpu:
        total_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU memory during inference: {total_peak_memory_mb:.2f} MB")
    print("="*60 + "\n")
    
    prob = y_hat.softmax(-1).cpu().numpy()[:, 1]
    
    valid_label_mask = (y_true == 0) | (y_true == 1)
    valid_trn_idx = trn_idx.cpu().numpy() if torch.is_tensor(trn_idx) else trn_idx
    valid_val_idx = val_idx.cpu().numpy() if torch.is_tensor(val_idx) else val_idx
    valid_tst_idx = tst_idx.cpu().numpy() if torch.is_tensor(tst_idx) else tst_idx
    
    valid_trn_idx = [idx for idx in valid_trn_idx if valid_label_mask[idx]]
    valid_val_idx = [idx for idx in valid_val_idx if valid_label_mask[idx]]
    valid_tst_idx = [idx for idx in valid_tst_idx if valid_label_mask[idx]]
    
    dic = cal_metrics(prob, y_true, valid_trn_idx, valid_val_idx, valid_tst_idx, verbose=True)
    print("===" * 10)
    print("===" * 10)

if __name__ == "__main__":
    run()

