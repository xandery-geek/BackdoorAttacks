import torch
import numpy as np
from torchvision import transforms
from methods.base import BaseAttack
from data.trigger import PatchTrigger
from data.dataset import get_image_size
from utils.utils import AverageMeter, accuracy
from utils.utils import save_images

from torch.utils.tensorboard import SummaryWriter


class BadNets(BaseAttack):
    def __init__(self, opt) -> None:
        super().__init__(opt)

        # load mode
        self.model = self._load_model()
        self.model = self.model.cuda()

        # set trigger
        image_size = get_image_size(self.opt.dataset)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        patch = np.zeros((image_size, image_size), dtype=np.uint8)
        
        mask[image_size-4: image_size-1, image_size-4: image_size-1] = 1
        patch[image_size-4: image_size-1, image_size-4: image_size-1] = 255

        if self.opt.dataset == 'mnist':
            self.trigger = PatchTrigger(mask, patch, mode='CHW')
        else:
            self.trigger = PatchTrigger(mask, patch, mode='HWC')

        # load data
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if self.opt.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.div(255)),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        data_loader = self._load_data(transform, self.trigger, self.opt.target, self.opt.percentage, 
                                        batch_size=self.opt.bs, num_workers=16)
        self.train_loader, self.ori_test_loader, self.poi_test_loader = data_loader
        
        # load optimizer
        self.optimizer = self._load_optimizer(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()

        for _, (images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(), labels.cuda()
            output = self.model(images)
            loss = self.criterion(output, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), labels.shape[0])

        return loss_meter.avg

    def train(self):
        writer = SummaryWriter(log_dir=self.tb_path)

        for epoch in range(self.opt.epochs):
            self._adjust_lr(self.optimizer, epoch)
            loss = self._train(epoch)
            
            print('Epoch: {} \t loss={:.5f}'.format(epoch, loss))
            writer.add_scalar('train_loss', loss, epoch)

        self._save_model({
            'model': self.model.state_dict()
        }, 'last.pth')

        writer.close()

    @torch.no_grad()
    def _eval(self, ori=True):
        self.model.eval()
        writer = SummaryWriter(log_dir=self.tb_path)
        top1 = AverageMeter()

        data_loader = self.ori_test_loader if ori else self.poi_test_loader
        for idx, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = self.model(images)

            acc1, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], labels.shape[0])

            if idx == 0:
                save_images(writer, 'oir_images' if ori else 'poi_images', 
                            images[:10], step=0)
        writer.close()
        return top1.avg

    def eval(self):
        ori_acc = self._eval(ori=True)
        poi_acc = self._eval(ori=False)

        print("Original Acc@1: {:.5f}".format(ori_acc))
        print("Poisoned Acc@1: {:.5f}".format(poi_acc))

