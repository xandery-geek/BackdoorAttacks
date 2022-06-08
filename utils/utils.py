import os
import torch
import argparse


class AverageMeter(object):
    r"""Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])

    # return mod.comp1.comp2...
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def check_path(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def unnormalize(arr, mean, std):
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1, 1)
    if std.ndim == 1:
        std = std.reshape(-1, 1, 1)

    return arr * std + mean


def save_images(writer, tag, images, step=0):

    device = (torch.device('cuda') if images.is_cuda else torch.device('cpu'))
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    mean = torch.tensor(list(mean)).to(device)
    std = torch.tensor(list(std)).to(device)

    images = unnormalize(images, mean, std)
    writer.add_images(tag, images, global_step=step)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res