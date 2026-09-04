import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml
import time

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
    
    return args


def main(args):
    method = args['method']
    
    if method == 'rgtan':
        from methods.rgtan.rgtan_main_dgraph_gpu_split import rgtan_train_and_save_models, rgtan_load_and_inference, loda_rgtan_data
        
        print(f"\n{'='*80}")
        print(f"RGTAN 模型")
        print(f"数据集: {args['dataset']}")
        print(f"模式: {args['mode']}")
        print(f"模型目录: {args['model_dir']}")
        print(f"模型名称: {args['model_name']}")
        print(f"{'='*80}")
        
        # 加载数据
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        
        if args['mode'] == 'train':
            # 训练模式：保存k折模型
            # 确定模型保存目录
            if args['model_name']:
                # 使用指定的模型名称
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], args['model_name'])
            else:
                # 使用时间戳作为模型名称
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
            print(f"    python main.py --method rgtan --mode infer --model_dir {model_save_dir}")
            
        elif args['mode'] == 'infer':
            # 推理模式：加载k折模型进行推理
            print(f"\n推理模式 - 加载k折模型")
            print(f"从目录加载模型: {args['model_dir']}")
            
            # 直接从指定目录加载模型
            model_save_dir = args['model_dir']
            if not os.path.exists(model_save_dir):
                print(f"✗ 错误: 模型目录不存在: {model_save_dir}")
                return
            
            inference_results = rgtan_load_and_inference(
                feat_data, g, test_idx, labels, args,
                cat_features, neigh_features,
                nei_att_head=args['nei_att_heads'][args['dataset']],
                model_save_dir=model_save_dir
            )
            
            print(f"\n✓ 推理完成!")
            print(f"  • 结果保存到: {os.path.join(model_save_dir, 'predictions')}")
    
    elif method == 'hogrl':
        from methods.hogrl.hogrl_main_dgraph_v2_split import hogrl_train, hogrl_infer, load_data
        
        print(f"\n{'='*80}")
        print(f"HORGL 模型")
        print(f"数据集: {args['dataset']}")
        print(f"模式: {args['mode']}")
        print(f"模型目录: {args['model_dir']}")
        print(f"模型名称: {args['model_name']}")
        print(f"{'='*80}")
        
        # 加载数据（与RGTAN不同，需要单独的加载函数）
        prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
        edge_indexs, feat_data, labels = load_data(args['dataset'], args['layers_tree'], prefix)
        
        # 加载数据分割掩码
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
        
        if args['mode'] == 'train':
            # 训练模式
            # 确定模型保存目录
            if args['model_name']:
                # 使用指定的模型名称
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], args['model_name'])
            else:
                # 使用时间戳作为模型名称
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                model_save_dir = os.path.join(args['model_dir'], args['dataset'], f"{args['dataset']}_{timestamp}")
            
            print(f"\n训练模式 - 保存最佳模型")
            print(f"模型将保存到: {model_save_dir}")
            
            # 创建保存目录
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 训练参数
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
            
            # 运行训练
            train_results = hogrl_train(train_args)
            
            # 保存训练结果
            results_file = os.path.join(model_save_dir, 'training_results.json')
            with open(results_file, 'w') as f:
                import json
                # 处理不能序列化的数据
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
            print(f"    python main.py --method hogrl --mode infer --model_dir {model_save_dir}")
            
        elif args['mode'] == 'infer':
            # 推理模式
            print(f"\n推理模式 - 加载模型进行推理")
            print(f"从目录加载模型: {args['model_dir']}")
            
            # 直接从指定目录加载模型
            model_save_dir = args['model_dir']
            if not os.path.exists(model_save_dir):
                print(f"✗ 错误: 模型目录不存在: {model_save_dir}")
                return
            
            # 查找最佳模型
            model_path = None
            models_dir = os.path.join(model_save_dir, 'models')
            
            # 首先检查是否有最终模型
            final_model_path = os.path.join(models_dir, 'hogrl_final_model.pth')
            if os.path.exists(final_model_path):
                model_path = final_model_path
                print(f"  • 找到最终模型: {final_model_path}")
            else:
                # 检查checkpoints目录
                checkpoints_dir = os.path.join(model_save_dir, 'checkpoints')
                if os.path.exists(checkpoints_dir):
                    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
                    if checkpoint_files:
                        # 按epoch排序，选择最新的
                        checkpoint_files.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]) if 'epoch' in x else 0, reverse=True)
                        model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
                        print(f"  • 找到检查点模型: {model_path}")
            
            if not model_path:
                print(f"✗ 错误: 在目录中未找到模型文件: {model_save_dir}")
                return
            
            # 推理参数
            infer_args = args.copy()
            infer_args.update({
                'model_path': model_path,
                'save_predictions': True,
                'results_dir': os.path.join(model_save_dir, 'inference_results'),
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
            
            # 运行推理
            inference_results = hogrl_infer(infer_args)
            
            # 保存推理结果
            os.makedirs(infer_args['results_dir'], exist_ok=True)
            
            # 保存详细推理结果
            predictions_file = os.path.join(infer_args['results_dir'], 'predictions.pkl')
            with open(predictions_file, 'wb') as f:
                pickle.dump(inference_results, f)
            
            # 保存简化版结果（JSON格式）
            summary_file = os.path.join(infer_args['results_dir'], 'inference_summary.json')
            summary = {
                'train_metrics': inference_results.get('train_metrics', {}),
                'val_metrics': inference_results.get('val_metrics', {}),
                'test_metrics': inference_results.get('test_metrics', {}),
                'model_path': model_path,
                'inference_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(summary_file, 'w') as f:
                import json
                json.dump(summary, f, indent=2)
            
            print(f"\n✓ 推理完成!")
            print(f"  • 使用的模型: {model_path}")
            print(f"  • 预测结果: {predictions_file}")
            print(f"  • 结果摘要: {summary_file}")
    
    else:
        raise NotImplementedError("只支持 rgtan 和 hogrl 方法")


if __name__ == "__main__":
    main(parse_args())
