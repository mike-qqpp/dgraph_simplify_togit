import argparse

def get_args():
    parser = argparse.ArgumentParser(description='T-Finance Feature Project')
    
    # File path parameters
    parser.add_argument('--input_path', type=str, default='./data/tfinance.npz',
                       help='Enter the path to the npz file')
    parser.add_argument('--output_path', type=str, default='./output/features.pkl',
                       help='Output Profile Path')
    
    # Neighbour sampling parameters
    parser.add_argument('--max_neighbors_1hop', type=int, default=50,
                       help='Maximum sample of 1-hop neighbours')
    parser.add_argument('--max_neighbors_2hop', type=int, default=500,
                       help='Maximum sample of 2-hop neighbours')
    
    # Feature Selection Parameters
    parser.add_argument('--use_base_features', type=bool, default=True,
                       help='Whether to use underlying features')
    parser.add_argument('--use_structural_features', type=bool, default=True,
                       help='Whether to use structural features')
    parser.add_argument('--use_neighbor_features', type=bool, default=True,
                       help='Use of neighbourhood features')
    parser.add_argument('--use_spectral_features', type=bool, default=False,
                       help='Whether to use spectrometric features (large number of calculations)')
    parser.add_argument('--use_embedding_features', type=bool, default=True,
                       help='Whether or not to use graph embedded features')
    
    # Figure embedded parameters
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Figure embedded dimensions')
    parser.add_argument('--walk_length', type=int, default=40,
                       help='Random length of walking')
    parser.add_argument('--num_walks', type=int, default=10,
                       help='Number of random trips per node')
    
    # Calculate parameters
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel work')
    
    return parser.parse_args()
