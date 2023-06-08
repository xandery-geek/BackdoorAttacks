import argparse

def add_argument(parser: argparse.ArgumentParser):
    group =  parser.add_argument_group('dataset', 'arguments for dataset')
    group.add_argument('--data_path', type=str, default='../data')
    group.add_argument('--dataset', type=str, default='cifar-10', 
                        choices=['mnist', 'cifar-10', 'imagenet', 'tiny'])
    group.add_argument('--bs', type=int, default=256, help='batch size')