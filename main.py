import os
import argparse
from process.classifier import Classifier
from utils.utils import str2bool


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0, help='gpu id')
    parser.add_argument('--method', type=str, default='BadNets', choices=['BadNets', 'CleanLabel'], help='Backdoor attack methods')

    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['mnist', 'cifar-10'])
    parser.add_argument('--regenerate', type=str2bool, default=False)
    parser.add_argument('--bs', type=int, default=128, help='batch size')

    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'])
    parser.add_argument('--save', type=str, default='checkpoint', help='save path')
    parser.add_argument('--ckpt', type=str, default='', help='load state dict of model from ckpt')
    parser.add_argument('--train', type=str2bool, default='True')

    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--cosine', type=str2bool, default=True, help='using cosine annealing')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,80,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--trigger', type=str, default='patch', choices=['patch', 'pixel'], help='tirgger type')
    parser.add_argument('--target', type=int, default=0, help='poisoned target')
    parser.add_argument('--percentage', type=float, default=0.1, help='poisoned percentage')

    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    
    # For CleanLabel
    parser.add_argument('--pre_ckpt', type=str, default='', help='pretrained state dict for CleanLabel')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)

    Processer = Classifier(opt)
    
    if opt.train:
        Processer.train()
    Processer.eval()
