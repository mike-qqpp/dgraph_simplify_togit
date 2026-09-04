import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import numpy as np
import pandas as pd
import sys
import pickle
import dgl
import yaml
import time
import json
import torch

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", required=True, choices=['rgtan', 'hogrl'], 
                       help="选择方法: rgtan 或 hogrl")
    parser.add_argument("--mode", default="train", choices=['train', 'infer'],
                       help="运行模式: train(训练并保存k折模型), infer(加载k折模型进行推理)")
    parser.add_argument("--model_dir", default="./saved_models/",
                       help="模型保存/加载目录")
    parser.add_argument("--model_name", default=None,
                       help="模型名称(可选), 不指定则使用dataset+时间戳")
    parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'],
                       help="运行设备: cpu 或 cuda (默认: cpu)")
    
    # 获取方法参数
    parsed_args = parser.parse_args()
    method = parsed_args.method

    # 根据方法加载对应的yaml配置文件
    if method == 'rgtan':
        yaml_file = "config/rgtan_cfg.yaml"
    elif method == 'hogrl':
        yaml_file = "config/hogrl_cfg.yaml"
    else:
        raise NotImplementedError("只支持 rgtan 和 hogrl 方法")

    # 加载yaml配置
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    
    # 添加命令行参数到args
    args['mode'] = parsed_args.mode
    args['model_dir'] = parsed_args.model_dir
    args['model_name'] = parsed_args.model_name
    args['device'] = parsed_args.device
    
    return args


def ensure_cpu_device(args):
    """确保所有设备设置为CPU - 兼容旧版PyTorch"""
    print(f"\n设置运行设备为: CPU")
    args['device'] = 'cpu'
    
    # 对于旧版PyTorch，不需要设置默认设备
    # 我们会在后续代码中通过显式地将张量移动到CPU来处理
    
    # 禁用CUDA相关功能（如果有）
    if torch.cuda.is_available():
        print("注意: CUDA可用但强制使用CPU运行")
        # 设置环境变量禁用CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    return args


def main(args):
    # 强制使用CPU
    args = ensure_cpu_device(args)
    
    method = args['method']
    
    if method == 'rgtan':
        from methods.rgtan.rgtan_main_dgraph_cpu_split import rgtan_train_and_save_models, rgtan_load_and_inference, loda_rgtan_data
        
        print(f"\n{'='*80}")
        print(f"RGTAN 模型 (CPU模式)")
        print(f"数据集: {args['dataset']}")
        print(f"模式: {args['mode']}")
        print(f"模型目录: {args['model_dir']}")
        print(f"模型名称: {args['model_name']}")
        print(f"{'='*80}")
        
        # 加载数据
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        
        # 确保图在CPU上
        g = g.to('cpu')
        
        if args['mode'] == 'train':
            # 训练模式：保存k折模型
            if args['model_name']:
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], args['model_name'])
            else:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], f"{args['dataset']}_{timestamp}")
            
            print(f"\n训练模式 - 保存k折模型")
            print(f"模型将保存到: {model_save_dir}")
            
            fold_model_paths, fold_histories = rgtan_train_and_save_models(
                feat_data, g, train_idx, test_idx, labels, args,
                cat_features, neigh_features, 
                nei_att_head=args['nei_att_heads'][args['dataset']],
                model_save_dir=model_save_dir
            )
            
            print(f"\n✓ 训练完成!")
            print(f"  • 保存了 {len(fold_model_paths)} 个k折模型")
            print(f"  • 模型目录: {model_save_dir}")
            print(f"  • 要推理请运行:")
            print(f"    python main_split_cpu.py --method rgtan --mode infer --model_dir {model_save_dir}")
            
        elif args['mode'] == 'infer':
            # 推理模式 - 只输出指标，不保存文件
            print(f"\n推理模式 - 加载k折模型进行推理")
            print(f"从目录加载模型: {args['model_dir']}")
            
            model_save_dir = args['model_dir']
            if not os.path.exists(model_save_dir):
                print(f"✗ 错误: 模型目录不存在: {model_save_dir}")
                return
            
            # 调用推理函数（已修改为不保存文件）
            inference_results = rgtan_load_and_inference(
                feat_data, g, test_idx, labels, args,
                cat_features, neigh_features,
                nei_att_head=args['nei_att_heads'][args['dataset']],
                model_save_dir=model_save_dir
            )
            
            # 只输出结果，不保存文件
            print(f"\n✓ 推理完成!")
    
    elif method == 'hogrl':
        from methods.hogrl.hogrl_main_dgraph_cpu_split import hogrl_train, hogrl_infer, load_data
        
        print(f"\n{'='*80}")
        print(f"HOGRL 模型 (CPU模式)")
        print(f"数据集: {args['dataset']}")
        print(f"模式: {args['mode']}")
        print(f"模型目录: {args['model_dir']}")
        print(f"模型名称: {args['model_name']}")
        print(f"{'='*80}")
        
        # 加载数据
        prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
        print(f"数据路径: {prefix}")
        
        try:
            edge_indexs, feat_data, labels = load_data(args['dataset'], args['layers_tree'], prefix)
        except Exception as e:
            print(f"加载数据时出错: {e}")
            print("请确保数据文件存在且路径正确")
            return
        
        # 加载数据分割掩码
        npz_path = f"../data/{args['dataset']}.npz"
        if not os.path.exists(npz_path):
            print(f"✗ 错误: 数据文件不存在: {npz_path}")
            return
            
        npz = np.load(npz_path, allow_pickle=True)
        
        # 检查掩码是否存在
        if 'train_mask' not in npz.files or 'valid_mask' not in npz.files or 'test_mask' not in npz.files:
            print(f"✗ 错误: npz文件中缺少掩码信息")
            print(f"文件中的键: {list(npz.files)}")
            return
            
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
        
        if args['mode'] == 'train':
            # 训练模式
            if args['model_name']:
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], args['model_name'])
            else:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], f"{args['dataset']}_{timestamp}")
            
            print(f"\n训练模式 - 保存最佳模型")
            print(f"模型将保存到: {model_save_dir}")
            
            os.makedirs(model_save_dir, exist_ok=True)
            
            train_args = args.copy()
            train_args.update({
                'model_save_dir': model_save_dir,
                'save_checkpoint': True,
                'save_final_model': True,
                'save_results': True,
                'model_dir': os.path.join(model_save_dir, 'models'),
                'checkpoint_dir': os.path.join(model_save_dir, 'checkpoints'),
                'results_dir': os.path.join(model_save_dir, 'results'),
                'idx_train': idx_train,
                'idx_val': idx_val,
                'idx_test': idx_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feat_data': feat_data,
                'edge_indexs': edge_indexs,
                'labels': labels
            })
            
            train_results = hogrl_train(train_args)
            
            results_file = os.path.join(model_save_dir, 'training_results.json')
            with open(results_file, 'w') as f:
                serializable_results = {}
                for key, value in train_results.items():
                    if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                        serializable_results[key] = value
                    elif isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    else:
                        serializable_results[key] = str(value)
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n✓ 训练完成!")
            print(f"  • 模型保存到: {model_save_dir}")
            print(f"  • 结果文件: {results_file}")
            print(f"  • 要推理请运行:")
            print(f"    python main_split_cpu.py --method hogrl --mode infer --model_dir {model_save_dir}")
            
        elif args['mode'] == 'infer':
            # 推理模式 - 只输出指标，不保存文件
            print(f"\n推理模式 - 加载模型进行推理")
            print(f"从目录加载模型: {args['model_dir']}")
            
            model_save_dir = args['model_dir']
            if not os.path.exists(model_save_dir):
                print(f"✗ 错误: 模型目录不存在: {model_save_dir}")
                return
            
            # 查找模型文件
            model_path = None
            models_dir = os.path.join(model_save_dir, 'models')
            checkpoints_dir = os.path.join(model_save_dir, 'checkpoints')
            
            # 首先检查是否有最终模型
            final_model_path = os.path.join(models_dir, 'hogrl_final_model.pth')
            if os.path.exists(final_model_path):
                model_path = final_model_path
                print(f"  • 找到最终模型: {final_model_path}")
            # 检查是否有其他.pth文件在models目录
            elif os.path.exists(models_dir):
                pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
                if pth_files:
                    model_path = os.path.join(models_dir, pth_files[0])
                    print(f"  • 找到模型: {model_path}")
            # 检查checkpoints目录
            elif os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
                    model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
                    print(f"  • 找到检查点模型: {model_path}")
            
            if not model_path:
                print(f"✗ 错误: 在目录中未找到模型文件: {model_save_dir}")
                print(f"搜索的目录: {models_dir}, {checkpoints_dir}")
                return
            
            # 推理参数 - 设置save_predictions为False
            infer_args = args.copy()
            infer_args.update({
                'model_path': model_path,
                'save_predictions': False,  # 不保存预测结果
                'idx_train': idx_train,
                'idx_val': idx_val,
                'idx_test': idx_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feat_data': feat_data,
                'edge_indexs': edge_indexs,
                'labels': labels
            })
            
            # 运行推理（hogrl_infer已经修改为不保存文件）
            inference_results = hogrl_infer(infer_args)
            
            print(f"\n✓ 推理完成!")
    
    else:
        raise NotImplementedError("只支持 rgtan 和 hogrl 方法")


if __name__ == "__main__":
    main(parse_args())
