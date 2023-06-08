import os
import time
import yaml
import torch
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
    def __init__(self, cfg, enable_tb=True) -> None:

        cfg.classes = get_num_classes(cfg.dataset)

        cur_time = time.strftime('%y-%m-%d-%H-%M-%S', time.localtime())

        self.model_name = '{}_{}_{}_{}'.format(cfg.dataset, cfg.model, cur_time, cfg.trial)
        self.ckpt_path = os.path.join(cfg.ckpt_path, cfg.attack, self.model_name)
        check_path(self.ckpt_path)

        self.log_path = os.path.join(cfg.log_path, cfg.attack, self.model_name)
        check_path(self.log_path)

        if enable_tb:
            self.tb_path = os.path.join(self.log_path, 'tensorboard')
            check_path(self.tb_path)
        
        self.cfg = cfg

        self._save_config_parameters()

    def _load_model(self):
        if 'ResNet' in self.cfg.model:
            model = ResNet(self.cfg.model, self.cfg.classes)
        else:
            raise NotImplementedError('model {} is not supported!'.format(self.cfg.model))
        
        if self.cfg.ckpt != '' and os.path.exists(self.cfg.ckpt):
            print('==> Loading state dict from {}'.format(self.cfg.ckpt))
            state_dict = torch.load(self.cfg.cpkt)
            model.load_state_dict(state_dict['model'])
        return model

    def _load_optimizer(self, parameters):

        if self.cfg.optim == 'SGD':
            optimizer = optim.SGD(parameters,
                            lr=self.cfg.lr,
                            momentum=self.cfg.momentum,
                            weight_decay=self.cfg.weight_decay)
        elif self.cfg.optim == 'Adam':
            optimizer = optim.Adam(parameters,
                                   lr=self.cfg.lr,
                                   betas=self.cfg.betas,
                                   weight_decay=self.cfg.weight_decay)
        else:
            raise NotImplementedError()
        
        lr_scheduler = None
        
        if self.cfg.lr_scheduler:
            lr_scheduler_type = self.cfg.lr_scheduler_type
            lr_warmup_epochs = self.cfg.lr_warmup_epochs

            if lr_scheduler_type == 'StepLR':
                main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)
            elif lr_scheduler_type == 'CosineAnnealingLR':
                main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.cfg.epochs - lr_warmup_epochs, eta_min=self.cfg.lr_min)
            elif lr_scheduler_type == 'ExponentialLR':
                main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.lr_gamma)
            else:
                    raise NotImplementedError(
                        f'Invalid {lr_scheduler_type=}.'
                        'Only "StepLR", "CosineAnnealingLR" and "ExponentialLR" '
                        'are supported.')
                
            if lr_warmup_epochs > 0:
                lr_warmup_method = self.cfg.lr_warmup_method
                if lr_warmup_method == 'linear':
                    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=self.cfg.lr_warmup_decay,
                        total_iters=lr_warmup_epochs)
                elif lr_warmup_method == 'constant':
                    warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer, factor=self.cfg.lr_warmup_decay,
                        total_iters=lr_warmup_epochs)
                else:
                    raise NotImplementedError(
                        f'Invalid {lr_warmup_method=}.'
                        'Only "linear" and "constant" are supported.')
                
                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                    milestones=[lr_warmup_epochs])
            else:
                lr_scheduler = main_lr_scheduler

        return optimizer, lr_scheduler

    def _save_model(self, obj, filename):
        ckpt = os.path.join(self.ckpt_path, filename)
        print("Saving checkpint to {}".format(ckpt))
        torch.save(obj, ckpt)

    def _save_config_parameters(self):
        with open(os.path.join(self.log_path, 'parameters.yaml'), 'w') as f:
            yaml.dump(self.cfg, f, indent=2)

    @abstractclassmethod
    def train(self):
        pass
    
    @abstractclassmethod
    def eval(self):
        pass
