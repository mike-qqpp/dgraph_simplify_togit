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
        from methods.hogrl.hogrl_main_dgraph_v2_split import hogrl_main
        
        print(f"\n{'='*80}")
        print(f"HORGL 模型")
        print(f"数据集: {args['dataset']}")
        print(f"{'='*80}")
        
        # 运行hogrl
        hogrl_main(args)
    
    else:
        raise NotImplementedError("只支持 rgtan 和 hogrl 方法")


if __name__ == "__main__":
    main(parse_args())
