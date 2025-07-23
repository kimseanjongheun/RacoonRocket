import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part_hfgnet import forward
import time

    
def parse():
    parser = argparse.ArgumentParser(description='Train HFGnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')

    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_pools', type=int, default=4, help='Number of downsampling/upsampling layers for sensitivity map U-Net')
    parser.add_argument('--pools', type=int, default=4, help='Number of downsampling/upsampling layers for cascade U-Net')
    parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels (default: 1 for single-coil)')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--gamma', type=float, default=0.01, help='Weight for loss2')
    parser.add_argument('--l1_weight', type=float, default=0.1, help='L1 loss의 가중치(alpha)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'

    start_time = time.time()
    
    # acc4
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')