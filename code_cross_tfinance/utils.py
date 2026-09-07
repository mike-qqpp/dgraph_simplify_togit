import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Dict, List, Tuple
import scipy.sparse as sp
from scipy import stats
import numba

warnings.filterwarnings('ignore')

def load_npz_data(filepath: str) -> Dict:
    """Load npz file"""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}

def save_features(features_df: pd.DataFrame, filepath: str):
    """Save feature DataFrame as pkl file"""
    with open(filepath, 'wb') as f:
        pickle.dump(features_df, f)

def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe division, avoid zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = 0
    return result

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to the [0, 1] range."""
    if features.shape[1] == 0:
        return features
    
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    
    # Avoid zero.
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    normalized = (features - min_vals) / range_vals
    return normalized.astype(np.float32)

def standardize_features(features: np.ndarray) -> np.ndarray:
    """Standardized features"""
    if features.shape[1] == 0:
        return features
    
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1
    
    standardized = (features - mean) / std
    return standardized.astype(np.float32)

def winsorize_features(features: np.ndarray, limits: tuple = (0.01, 0.99)) -> np.ndarray:
    """Winsorize handles anomalies"""
    if features.shape[1] == 0:
        return features
    
    q_low, q_high = np.quantile(features, limits, axis=0)
    
    # Process one column at a time.
    for i in range(features.shape[1]):
        col = features[:, i]
        col[col < q_low[i]] = q_low[i]
        col[col > q_high[i]] = q_high[i]
    
    return features.astype(np.float32)

@numba.jit(nopython=True)
def fast_neighbor_aggregation(indices: np.ndarray, values: np.ndarray, 
                            shape: Tuple[int, int], node_features: np.ndarray,
                            agg_func: str = 'mean') -> np.ndarray:
    """Quick Neighborhood Profile"""
    n_nodes, n_features = node_features.shape
    result = np.zeros((n_nodes, n_features), dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.int32)
    
    if agg_func == 'mean':
        for idx, (i, j) in enumerate(indices):
            result[i] += node_features[j]
            counts[i] += 1
        
        for i in range(n_nodes):
            if counts[i] > 0:
                result[i] /= counts[i]
    
    elif agg_func == 'sum':
        for idx, (i, j) in enumerate(indices):
            result[i] += node_features[j]
    
    return result
