import numpy as np
import pandas as pd
from typing import Dict
import argparse
from scipy import stats

def process_base_features(x: np.ndarray, 
                         train_mask: np.ndarray = None,
                         output_path: str = None) -> pd.DataFrame:
    """
    基础特征处理：
    1. 原始特征
    2. 统计变换
    3. 交互特征
    4. 分桶特征
    """
    
    n_nodes = x.shape[0]
    n_features = x.shape[1]
    
    # 初始化特征列表
    feature_dict = {}
    
    # 1. 原始特征
    for i in range(n_features):
        feature_dict[f'feature_{i}'] = x[:, i].astype(np.float32)
    
    # 2. 统计变换
    # 对数变换（处理偏态分布）
    for i in range(n_features):
        col = x[:, i].copy()
        # 处理非正值
        min_val = np.min(col)
        if min_val <= 0:
            col = col - min_val + 1e-6
        log_col = np.log1p(col)
        feature_dict[f'feature_{i}_log'] = log_col.astype(np.float32)
    
    # 平方根变换
    for i in range(n_features):
        col = x[:, i].copy()
        col = np.maximum(col, 0)  # 确保非负
        sqrt_col = np.sqrt(col)
        feature_dict[f'feature_{i}_sqrt'] = sqrt_col.astype(np.float32)
    
    # 3. 分位数分桶（10个分桶）
    for i in range(n_features):
        col = x[:, i]
        quantiles = np.quantile(col, np.linspace(0, 1, 11)[1:-1])
        binned = np.digitize(col, quantiles)
        feature_dict[f'feature_{i}_bucket'] = binned.astype(np.float32)
    
    # 4. 标准化和归一化
    # Z-score标准化
    x_standardized = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
    for i in range(n_features):
        feature_dict[f'feature_{i}_standardized'] = x_standardized[:, i].astype(np.float32)
    
    # Min-Max归一化
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    x_normalized = (x - min_vals) / range_vals
    for i in range(n_features):
        feature_dict[f'feature_{i}_normalized'] = x_normalized[:, i].astype(np.float32)
    
    # 5. 交互特征（特征相乘）
    if n_features >= 2:
        for i in range(n_features):
            for j in range(i+1, min(i+3, n_features)):  # 限制交互特征数量
                interaction = x[:, i] * x[:, j]
                feature_dict[f'interaction_{i}_{j}'] = interaction.astype(np.float32)
    
    # 6. 多项式特征（2阶）
    for i in range(min(5, n_features)):  # 限制多项式特征数量
        poly_2 = x[:, i] ** 2
        feature_dict[f'feature_{i}_poly2'] = poly_2.astype(np.float32)
    
    # 7. 排名特征
    for i in range(n_features):
        rank = stats.rankdata(x[:, i]) / n_nodes
        feature_dict[f'feature_{i}_rank'] = rank.astype(np.float32)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    # 如果指定了输出路径，保存特征
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    return features_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入npz文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出特征文件路径')
    args = parser.parse_args()
    
    # 加载数据
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    x = data['x']
    train_mask = data['train_mask'] if 'train_mask' in data else None
    
    # 处理基础特征
    features_df = process_base_features(x, train_mask)
    
    # 保存结果
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"基础特征处理完成，特征维度: {features_df.shape}")
    print(f"特征保存至: {args.output_path}")

if __name__ == "__main__":
    main()
