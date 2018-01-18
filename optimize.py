import argparse
from data import get_data_by_name
from itertools import product
import numpy as np

def get_data(args):
    if not args.topological:
        x_vals, y_vals = get_data_by_name(args.path, args.data)
    else:
        x_vals, y_vals = get_data_by_name(args.path, args.data, False)

    x_vals= x_vals[:, :-args.feature_dim]
    x_vals = (x_vals - np.mean(x_vals,0)) / np.std(x_vals,0)

    return x_vals, y_vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Network on Planted CLique Data')
    # Add arguments
    parser.add_argument(
        '--network', required=True)
    parser.add_argument(
        '--data', help='dataset used', required=True)
    parser.add_argument(
        '--data_path', help='datset path', required=False, default='data/')
    parser.add_argument(
        '--topological', help='Use topological features', required=False)
    parser.add_argument(
        '-td', '--trunc_dim', type=int, help='Truncate the size of feature dimension', required=False, default=0)

    args = parser.parse_args()

    x_vals, y_vals = get_data(args)
    input_dim = x_vals.shape[-1]

    ###
    ### Search space
    ###

    

