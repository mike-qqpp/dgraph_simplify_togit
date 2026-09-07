"""5-box feature engineering module - KbinsDiscretizer based on sklearn
Simplicitly and efficiently load data from npz files and create box tabs for nodal features"""
import numpy as np
import pandas as pd
import argparse
from typing import List
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')


def add_box_features(
    input_path: str,
    output_path: str,
    n_bins: int = 5,
    strategy: str = "quantile"
) -> pd.DataFrame:
    """5-minute box for nodal feature in npz file

Parameters:
Input path: Enter npz file path
output path: output pickle file path
n bins: Number of boxes, default 5
strategy: boxing policy, \"quantile\" (species) or \"uniform\" (separability)

Return:
DataFrame with box features"""
    print(f"Loading data: {input_path}")
    
    # Load from npz file
    data = np.load(input_path, allow_pickle=True)
    x = data['x']
    
    n_samples, n_features = x.shape
    print(f"  Number of nodes: {n_samples}, Feature shape: {n_features}")
    
    # Original feature listing
    cols = [f"feature_{i}" for i in range(n_features)]
    
    # Build DataFrame
    df = pd.DataFrame(x.astype(np.float32), columns=cols)
    
    # Initialised Boxer
    kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    
    # fit
    print(f"Boxfit (strategy={strategy})...")
    kbin.fit(df[cols])
    
    # Print box boundary examples
    bin_edges = kbin.bin_edges_
    print(f"\nExample of box boundary (Front3Features):")
    for i in range(min(3, n_features)):
        edges = bin_edges[i]
        print(f"  feature_{i}: [{', '.join([f'{e:.4f}' for e in edges[:n_bins+1]])}]")
    
    # transform
    print(f"\nBoxtransform...")
    box_vals = kbin.transform(df[cols]).astype(int)
    
    # Box listing (simplified name)
    box_cols = [f"bin_{i}" for i in range(n_features)]
    
    # Only keep the boxes and discard the original feature column
    df = pd.DataFrame(box_vals, columns=box_cols, index=df.index)
    
    # Print Box Distribution
    print(f"\nExample of box distribution (Front5Features):")
    for i in range(min(5, n_features)):
        col = f"bin_{i}"
        dist = df[col].value_counts().sort_index()
        dist_str = ", ".join([f"Box{int(k)}:{int(v)}" for k, v in dist.items()])
        print(f"  {col}: {dist_str}")
    
    # Save
    with open(output_path, 'wb') as f:
        pd.to_pickle(df, f)
    print(f"\nSave: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  New: {len(box_cols)}  items (bin_i)")
    
    return df


def add_box_features_multi(
    input_paths: List[str],
    output_paths: List[str],
    n_bins: int = 5,
    strategy: str = "quantile"
) -> List[pd.DataFrame]:
    """Harmonized 5-minute boxes of multiple npz files (first table fit, other tables transform)

Parameters:
Input paths: Enter the list of paths for npz files
Output paths: list of output pickle file paths (corresponds to one input)
n bins: Number of boxes, default 5
strategy: boxing policy, \"quantile\" (species) or \"uniform\" (separability)

Return:
DataFrame List"""
    if len(input_paths) != len(output_paths):
        raise ValueError("Input paths and output paths must be equal in length")
    
    print(f"Load the first file to performfit: {input_paths[0]}")
    
    # Load first file
    data = np.load(input_paths[0], allow_pickle=True)
    x = data['x']
    
    n_samples, n_features = x.shape
    cols = [f"feature_{i}" for i in range(n_features)]
    
    print(f"  Number of nodes: {n_samples}, Feature shape: {n_features}")
    
    # Build DataFrame
    df_first = pd.DataFrame(x.astype(np.float32), columns=cols)
    
    # Initializing Boxer and Fit
    kbin = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    print(f"Boxfit (strategy={strategy})...")
    kbin.fit(df_first[cols])
    
    # Print box boundary
    bin_edges = kbin.bin_edges_
    print(f"\nExample of box boundary (Front3Features):")
    for i in range(min(3, n_features)):
        edges = bin_edges[i]
        print(f"  feature_{i}: [{', '.join([f'{e:.4f}' for e in edges[:n_bins+1]])}]")
    
    box_cols = [f"bin_{i}" for i in range(n_features)]
    results = []
    
    for idx, input_path in enumerate(input_paths):
        print(f"\nProcessing [{idx+1}/{len(input_paths)}]: {input_path}")
        
        # Loading data
        data = np.load(input_path, allow_pickle=True)
        x = data['x']
        n_samples = x.shape[0]
        
        # transform
        box_vals = kbin.transform(x).astype(int)
        
        # Only keep boxes
        df = pd.DataFrame(box_vals, columns=box_cols, index=range(n_samples))
        
        # Print Distribution
        print(f"  Number of nodes: {n_samples}")
        print(f"  Box Distribution (Front3 items):")
        for i in range(min(3, n_features)):
            col = f"bin_{i}"
            dist = df[col].value_counts().sort_index()
            dist_str = ", ".join([f"Box{int(k)}:{int(v)}" for k, v in dist.items()])
            print(f"    {col}: {dist_str}")
        
        # Save
        with open(output_paths[idx], 'wb') as f:
            pd.to_pickle(df, f)
        print(f"  Save: {output_paths[idx]}, Shape: {df.shape}")
        
        results.append(df)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='5-box feature project')
    parser.add_argument('--input_path', type=str, default=None,
                       help='Enter the npz file path (sing file mode)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output pickle file path (single file mode)')
    parser.add_argument('--input_paths', type=str, nargs='+', default=None,
                       help='Enter the list of npz file paths (multi-file mode, flat box)')
    parser.add_argument('--output_paths', type=str, nargs='+', default=None,
                       help='List of output pickle file paths (corresponds to one input)')
    parser.add_argument('--n_bins', type=int, default=5,
                       help='Number of boxes, default 5')
    parser.add_argument('--strategy', type=str, default='quantile',
                       choices=['quantile', 'uniform'],
                       help='Boxing policy, quantile (spectrum) or unit (separate), default quantile')
    args = parser.parse_args()
    
    # Mode judgement
    if args.input_paths and args.output_paths:
        # Multifile Mode
        add_box_features_multi(
            input_paths=args.input_paths,
            output_paths=args.output_paths,
            n_bins=args.n_bins,
            strategy=args.strategy
        )
    elif args.input_path and args.output_path:
        # Single File Mode
        add_box_features(
            input_path=args.input_path,
            output_path=args.output_path,
            n_bins=args.n_bins,
            strategy=args.strategy
        )
    else:
        print("Specify --input path and --output path (single file mode)")
        print("or --input paths and --output paths (multi-file mode, flat box)")
    
    print("We're done with the subbox.")


if __name__ == "__main__":
    main()
