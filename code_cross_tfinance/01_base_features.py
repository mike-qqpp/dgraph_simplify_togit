import numpy as np
import pandas as pd
from typing import Dict
import argparse
from scipy import stats

def process_base_features(x: np.ndarray, 
                         train_mask: np.ndarray = None,
                         output_path: str = None) -> pd.DataFrame:
    """Basic features processing:
1. Original features
2. Statistical transformations
3. Interaction features
4. Binned features"""
    
    n_nodes = x.shape[0]
    n_features = x.shape[1]
    
    # List of initialised features
    feature_dict = {}
    
    # 1. Original features
    for i in range(n_features):
        feature_dict[f'feature_{i}'] = x[:, i].astype(np.float32)
    
    # 2. Statistical transformations
    # Logarithmic transformation (handling skewed distributions)
    for i in range(n_features):
        col = x[:, i].copy()
        # Processing non-positive values
        min_val = np.min(col)
        if min_val <= 0:
            col = col - min_val + 1e-6
        log_col = np.log1p(col)
        feature_dict[f'feature_{i}_log'] = log_col.astype(np.float32)
    
    # Square root transformation
    for i in range(n_features):
        col = x[:, i].copy()
        col = np.maximum(col, 0)  # Make sure it's not negative.
        sqrt_col = np.sqrt(col)
        feature_dict[f'feature_{i}_sqrt'] = sqrt_col.astype(np.float32)
    
    # 3. Binned features (10 bins)
    for i in range(n_features):
        col = x[:, i]
        quantiles = np.quantile(col, np.linspace(0, 1, 11)[1:-1])
        binned = np.digitize(col, quantiles)
        feature_dict[f'feature_{i}_bucket'] = binned.astype(np.float32)
    
    # Standardization and normalization
    # Z-score standardization
    x_standardized = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
    for i in range(n_features):
        feature_dict[f'feature_{i}_standardized'] = x_standardized[:, i].astype(np.float32)
    
    # Min-Max normalized
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    x_normalized = (x - min_vals) / range_vals
    for i in range(n_features):
        feature_dict[f'feature_{i}_normalized'] = x_normalized[:, i].astype(np.float32)
    
    # 5. Interaction features (feature multiplication)
    if n_features >= 2:
        for i in range(n_features):
            for j in range(i+1, min(i+3, n_features)):  # Limit the number of interactive features
                interaction = x[:, i] * x[:, j]
                feature_dict[f'interaction_{i}_{j}'] = interaction.astype(np.float32)
    
    # 6. Polynomial features (2 steps)
    for i in range(min(5, n_features)):  # Limit the number of multiple features
        poly_2 = x[:, i] ** 2
        feature_dict[f'feature_{i}_poly2'] = poly_2.astype(np.float32)
    
    # 7. Features of ranking
    for i in range(n_features):
        rank = stats.rankdata(x[:, i]) / n_nodes
        feature_dict[f'feature_{i}_rank'] = rank.astype(np.float32)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_dict)
    
    # Save feature if output path is specified
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
    
    return features_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, required=True, help='Output Profile Path')
    args = parser.parse_args()
    
    # Loading data
    import numpy as np
    data = np.load(args.input_path, allow_pickle=True)
    x = data['x']
    train_mask = data['train_mask'] if 'train_mask' in data else None
    
    # Deal with underlying features
    features_df = process_base_features(x, train_mask)
    
    # Save Results
    with open(args.output_path, 'wb') as f:
        import pickle
        pickle.dump(features_df, f)
    
    print(f"Basic feature processing complete. Feature dimension:{features_df.shape}")
    print(f"Features saved to:{args.output_path}")

if __name__ == "__main__":
    main()
