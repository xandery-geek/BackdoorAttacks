import os
import random
import argparse
import torch
import numpy as np
from process.classifier import Classifier
from utils.utils import str2bool


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0, help='gpu id')
    parser.add_argument('--method', type=str, default='BadNets', 
                        choices=['Clean', 'BadNets', 'SIG', 'FIBA', 'FTrojan'], 
                        help='Backdoor attack methods')

    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='cifar-10', 
                        choices=['mnist', 'cifar-10', 'imagenet', 'tiny'])
    parser.add_argument('--bs', type=int, default=128, help='batch size')

    parser.add_argument('--model', type=str, default='ResNet18', 
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'])
    parser.add_argument('--ckpt_path', type=str, default='checkpoint', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='log', help='log path')
    parser.add_argument('--ckpt', type=str, default='', help='load state dict of model from ckpt')
    parser.add_argument('--train', type=str2bool, default='True')

    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.025, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--cosine', type=str2bool, default=True, help='using cosine annealing')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,80,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--trigger', type=str, default='patch', choices=['patch', 'pixel'], help='tirgger type')
    parser.add_argument('--target', type=int, default=0, help='poisoned target')
    parser.add_argument('--percentage', type=float, default=0.01, help='poisoned percentage')

    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    opt = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
    set_seed(1024)

    Processer = Classifier(opt)
    
    if opt.train:
        Processer.train()
    else:
        Processer.eval()
