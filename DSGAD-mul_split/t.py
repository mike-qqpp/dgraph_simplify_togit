'''The file in which the model is trained'''

import torch
import argparse
import json
import models
import warnings
import random
import numpy as np
import os
import time


def seed_everything(seed, strengthen=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strengthen:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ['PYTHONHASHSEED'] = str(seed)


def setup_multigpu():
    """设置多GPU环境"""
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"检测到 {num_gpus} 个GPU，使用多GPU训练")
        return True, num_gpus, list(range(num_gpus))
    return False, 1, [0]


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)  # Random number seeds
    parser.add_argument('--run', type=int, default=10)  # How many times you run during training

    parser.add_argument('--dataset', type=str, default='yelp')  # dataset
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.4, 0.3, 0.3])  # The ratio of the training/validation/testing set
    parser.add_argument('--epoch', type=int, default=2)  # epoch
    parser.add_argument('--device', type=str, default='cuda')  # train device: cuda or cpu

    parser.add_argument('--model', type=str, default='DSGAD')  # model
    parser.add_argument('--model_config', type=json.loads, default='{}')  # The setting of model
    parser.add_argument('--save', type=str, default='', help='save weights path')  # Model weights save paths
    
    # 多GPU相关参数
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_weights', type=float, default=1e-1, help='learning rate for weights')
    parser.add_argument('--data_name', type=str, default='', help='data name')
    
    args = parser.parse_args()
    
    # 检查多GPU
    use_multigpu, num_gpus, device_ids = setup_multigpu()
    
    seed_everything(args.seed)
    
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'model config: {args.model_config}')
    print(f'device: {args.device}')
    if use_multigpu:
        print(f'使用多GPU训练，GPU数量: {num_gpus}')
    print()

    metrics = ['auc', 'recall', 'precision', 'f1_macro']
    performance = {}
    for metric in metrics:
        performance[metric] = []

    #start = time.time()
    for t in range(args.run):  # Train the model and collect metrics
        if args.device == 'cuda': 
            torch.cuda.empty_cache()

        print(f'trial: {t+1}/{args.run}')

        m, p = eval(f'models.{args.model}').trainfit(args)
        
        if m is not None and p is not None:
            if args.save: 
                torch.save(m.state_dict(), args.save)  # Save the weights
            for metric in metrics:
                performance[metric].append(p[metric])

        print()
    
    #end = time.time()
    #dt = round(end - start, 4)  # Training time

    # Finally, all metrics for each training session are output in a unified manner
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'model config: {args.model_config}')
    #print(f'time: {dt}')
    #print(start, end)
    for metric in metrics:
        print(f'{metric} ', end='')
    print()
    for t in range(args.run):
        print(f'{t+1:<2}: ', end='')
        for metric in metrics:
            if t < len(performance[metric]):
                print(f'{performance[metric][t]:.5f} ', end='')
            else:
                print(f'0.00000 ', end='')
        print()
