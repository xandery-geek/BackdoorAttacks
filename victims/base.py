import os
import yaml
import torch
import torch.optim as optim
from networks.backbone import ResNet
import pytorch_lightning as pl
from data.utils import get_num_classes


class BaseProcess(pl.LightningModule):
    r"""Base class for backdoor attack

    Args:
        opt: arguments
        enable_tb: enable tensorboard, default is `True`
    """
    def __init__(self, cfg) -> None:        
        super().__init__()

        self.cfg = cfg
        
        if cfg.train:
            self.save_hyperparameters(self.cfg)

    def _load_model(self):
        classes = get_num_classes(self.cfg.dataset)

        if 'ResNet' in self.cfg.model:
            model = ResNet(self.cfg.model, classes)
        else:
            raise NotImplementedError('model {} is not supported!'.format(self.cfg.model))
        
        return model

    def configure_optimizers(self):
        parameters = self.model.parameters()

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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch"
                }
            }
    
    def _save_config_parameters(self):
        with open(os.path.join(self.log_path, 'parameters.yaml'), 'w') as f:
            yaml.dump(self.cfg, f, indent=2)


    @staticmethod
    def collect_outputs(outputs, key_list):
        """
        Collect outoputs of pytorchlighting
        """
        output_list = [[] for _ in range(len(key_list))]

        for out in outputs:
            for i, key in enumerate(key_list):
                output_list[i].append(out[key])
        return output_list
