import argparse
from train import train_only as train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='13C', choices=['13C', '1H'])
    parser.add_argument('--graph_representation', type=str, default='sparsified', choices=['sparsified', 'fully_connected'])
    parser.add_argument('--fold_seed', type=int, default=0)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
