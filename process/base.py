import os
import time
import yaml
import math
import torch
import numpy as np
import torch.optim as optim
from abc import abstractclassmethod
from networks.backbone import ResNet
from utils.utils import check_path
from data.utils import get_num_classes


class BaseProcess(object):
    r"""Base class for backdoor attack

    Args:
        opt: arguments
        enable_tb: enable tensorboard, default is `True`
    """
    def __init__(self, opt, enable_tb=True) -> None:

        opt.classes = get_num_classes(opt.dataset)

        if not opt.cosine:
            iterations = opt.lr_decay_epochs.split(',')
            opt.lr_decay_epochs = list([])
            for it in iterations:
                opt.lr_decay_epochs.append(int(it))

        cur_time = time.strftime('%y-%m-%d-%H-%M-%S', time.localtime())

        self.model_name = '{}_{}_{}_{}'.format(opt.dataset, opt.model, cur_time, opt.trial)
        self.ckpt_path = os.path.join(opt.ckpt_path, opt.method, self.model_name)
        check_path(self.ckpt_path)

        self.log_path = os.path.join(opt.log_path, opt.method, self.model_name)
        check_path(self.log_path)

        if enable_tb:
            self.tb_path = os.path.join(self.log_path, 'tensorboard')
            check_path(self.tb_path)
        
        self.opt = opt

        self._save_config_parameters()

    def _load_model(self):
        if 'ResNet' in self.opt.model:
            model = ResNet(self.opt.model, self.opt.classes)
        else:
            raise NotImplementedError('model {} is not supported!'.format(self.opt.model))
        
        if self.opt.ckpt != '' and os.path.exists(self.opt.ckpt):
            print('==> Loading state dict from {}'.format(self.opt.ckpt))
            state_dict = torch.load(self.opt.cpkt)
            model.load_state_dict(state_dict['model'])
        return model

    def _load_optimizer(self, parameters):
        optimizer = optim.SGD(parameters,
                          lr=self.opt.lr,
                          momentum=self.opt.momentum,
                          weight_decay=self.opt.weight_decay)
        return optimizer

    def _adjust_lr(self, optimizer, epoch):
        lr = self.opt.lr
        if self.opt.cosine:
            eta_min = lr * (self.opt.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * ( 1 + math.cos(math.pi * epoch / self.opt.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(self.opt.lr_decay_epochs))
            if steps > 0:
                lr = lr * (self.opt.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _save_model(self, obj, filename):
        ckpt = os.path.join(self.ckpt_path, filename)
        print("Saving checkpint to {}".format(ckpt))
        torch.save(obj, ckpt)

    def _save_config_parameters(self):
        with open(os.path.join(self.log_path, 'parameters.yaml'), 'w') as f:
            yaml.dump(self.opt, f, indent=2)

    @abstractclassmethod
    def train(self):
        pass
    
    @abstractclassmethod
    def eval(self):
        pass
