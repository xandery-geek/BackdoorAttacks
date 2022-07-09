import torch
from torchvision import transforms
from process.base import BaseProcess
from utils.utils import AverageMeter, accuracy, import_class
from utils.utils import save_images

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Classifier(BaseProcess):
    def __init__(self, opt) -> None:
        super().__init__(opt)

        # load mode
        self.model = self._load_model()
        self.model = self.model.cuda()

        self.train_loader, self.ori_test_loader, self.poi_test_loader = self._load_data()
        
        # load optimizer
        self.optimizer = self._load_optimizer(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()


    def _load_data(self):
        attack_method = '.'.join(['methods', self.opt.method.lower(), self.opt.method])
        attack = import_class(attack_method)(self.opt)

        # load data
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if self.opt.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.div(255)),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif self.opt.dataset == 'cifar-10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        # self, poisoned_target, train, p=0.1, mode='replace', transform=None
        dataset = attack.get_poisoned_data(self.opt.target, train=True, p=self.opt.percentage, transform=transform)
        train_loader = DataLoader(dataset=dataset, batch_size=self.opt.bs, num_workers=16, shuffle=True)
        
        dataset = attack.get_poisoned_data(self.opt.target, train=False, p=0, transform=transform)
        ori_test_loader = DataLoader(dataset=dataset, batch_size=self.opt.bs, num_workers=16, shuffle=False)
        
        dataset = attack.get_poisoned_data(self.opt.target, train=False, p=1, transform=transform)
        poi_test_loader = DataLoader(dataset=dataset, batch_size=self.opt.bs, num_workers=16, shuffle=False)
        
        return train_loader, ori_test_loader, poi_test_loader


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

