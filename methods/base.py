import os
import math
import torch
import numpy as np
import torch.optim as optim
from abc import abstractclassmethod
from networks.backbone import SimpleCNN, ResNet
from utils.utils import check_path
from data.dataset import load_data, get_num_classes, create_backdoor_data_loader


class BaseAttack(object):
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

        self.model_name = '{}_{}'.format(opt.dataset, opt.trial)
        self.model_path = os.path.join(opt.save, '{}_models'.format(opt.method), self.model_name)
        check_path(self.model_path)

        if enable_tb:
            self.tb_path = os.path.join(opt.save, '{}_tensorboard'.format(opt.method), self.model_name)
            check_path(self.tb_path)
        
        self.opt = opt

    def _load_model(self):
        if self.opt.model == 'CNN':
            model = SimpleCNN(3, self.opt.classes)
        elif 'ResNet' in self.opt.model:
            model = ResNet(self.opt.model, self.opt.classes)
        else:
            raise NotImplementedError('model {} is not supported!'.format(self.opt.model))
        
        if self.opt.ckpt != '' and os.path.exists(self.opt.ckpt):
            print('==> Loading state dict from {}'.format(self.opt.ckpt))
            state_dict = torch.load(self.opt.cpkt)
            model.load_state_dict(state_dict['model'])
        return model

    def _load_data(self, transform, trigger, poisoned_target, train_p=0.1, mode='replace', **kwargs):
        train_dataset = load_data(self.opt.data_path, self.opt.dataset, train=True)
        test_dataset = load_data(self.opt.data_path, self.opt.dataset, train=False)

        train_loader = create_backdoor_data_loader(train_dataset, trigger, poisoned_target, p=train_p, mode=mode,
                                                    transform=transform, shuffle=True, **kwargs)
        ori_test_loader = create_backdoor_data_loader(test_dataset, trigger, poisoned_target, p=0, mode=mode, 
                                                    transform=transform, shuffle=False, **kwargs)
        poi_test_loader = create_backdoor_data_loader(test_dataset, trigger, poisoned_target, p=1.0, mode=mode, 
                                                    transform=transform, shuffle=False, **kwargs)
        
        return train_loader, ori_test_loader, poi_test_loader

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
        torch.save(obj, os.path.join(self.model_path, filename))

    @abstractclassmethod
    def train(self):
        pass
    
    @abstractclassmethod
    def eval(self):
        pass
