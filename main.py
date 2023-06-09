import os
import random
import argparse
import torch
import numpy as np
import data
import attacks
import victims
import victims.classifier as classifier
from utils.utils import str2bool


def add_argument(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('basic', 'basic arguments')
    group.add_argument('--device', type=str, default="0", help='gpu id')
    group.add_argument('--attack', type=str, default='BadNets', 
                        choices=['Clean', 'BadNets', 'SIG', 'FIBA', 'FTrojan'], 
                        help='Backdoor attack methods')
    
    group.add_argument('--ckpt_path', type=str, default='checkpoint', help='checkpoint path')
    group.add_argument('--log_path', type=str, default='log', help='log path')
    group.add_argument('--ckpt', type=str, default='', help='load state dict of model from ckpt')
    group.add_argument('--train', type=str2bool, default=True)

    group.add_argument('--seed', type=int, default=1024, help='random seed')
    group.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    group.add_argument('--enable_tb', type=str2bool, default=True, help='enable tensorboard')


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def set_env(device):
    os.environ['CUDA_VISIBLE_DEVICES'] = device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_argument(parser)
    data.add_argument(parser)
    attacks.add_argument(parser)
    victims.add_argument(parser)
    
    cfg = parser.parse_args()
    device = cfg.device

    cfg.device = [int(i.strip()) for i in device.split(',')]
    
    set_env(device)
    set_seed(cfg.seed)

    classifier.run(cfg)
