import argparse
from utils.utils import str2bool


def add_argument(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('victim', 'arguments for victim model')

    group.add_argument('--model', type=str, default='ResNet18', 
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'])
    group.add_argument('--epochs', type=int, default=100, help='train epoches')
    group.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer name')
    group.add_argument('--lr', type=float, default=0.025, help='learning rate')
    group.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    group.add_argument('--weight_decay', type=float, default=5e-4, 
                       help='weight decay')
    group.add_argument('--betas', type=tuple, default=(0.5, 0.999), 
                       help='betas for Adam optimizer')
    group.add_argument('--lr_scheduler', type=str2bool, default=True, 
                       help='enable lr scheduler')
    group.add_argument('--lr_scheduler_type', default='CosineAnnealingLR',
                        choices=['StepLR', 'CosineAnnealingLR',
                                'ExponentialLR'],
                        help='the lr scheduler '
                        '(default: CosineAnnealingLR)')
    group.add_argument('--lr_min', type=float, default=0.0,
                        help='min learning rate for `eta_min` '
                        'in CosineAnnealingLR (default: 0.0)')
    group.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='the number of epochs to warmup (default: 0)')
    group.add_argument('--lr_warmup_method', default= 'constant',
                        choices=['constant', 'linear'],
                        help='the warmup method (default: constant)')
    group.add_argument('--lr_step_size', type=int, default=30,
                        help='decrease lr every step-size epochs '
                        '(default: 30)')
    group.add_argument('--lr_gamma', type=float, default=0.1,
                        help='decrease lr by a factor of lr-gamma '
                        '(default: 0.1)')
    group.add_argument('--sample_batch', type=int, default=0,
                        help='sample the clean and poisoned images at `sample_batch`-th batch'
                        '(default: 0, -1 for no sample)')
    group.add_argument('--every_n_epoch', type=int, default=10,
                        help='interval for validation of the model'
                        '(default: 10)')
