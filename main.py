import os
import random
import argparse
import torch
import numpy as np
import data
import attacks
import victims
from victims.classifier import Classifier
from utils.utils import str2bool


def add_argument(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('basic', 'basic arguments')
    group.add_argument('--device', type=int, default=0, help='gpu id')
    group.add_argument('--attack', type=str, default='BadNets', 
                        choices=['Clean', 'BadNets', 'SIG', 'FIBA', 'FTrojan'], 
                        help='Backdoor attack methods')
    
    group.add_argument('--ckpt_path', type=str, default='checkpoint', help='checkpoint path')
    group.add_argument('--log_path', type=str, default='log', help='log path')
    group.add_argument('--ckpt', type=str, default='', help='load state dict of model from ckpt')
    group.add_argument('--train', type=str2bool, default=True)

    group.add_argument('--seed', type=int, default=1024, help='random seed')
    group.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_argument(parser)
    data.add_argument(parser)
    attacks.add_argument(parser)
    victims.add_argument(parser)
    
    cfg = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    set_seed(cfg.seed)

    Processer = Classifier(cfg)
    
    if cfg.train:
        Processer.train()
    else:
        Processer.eval()
