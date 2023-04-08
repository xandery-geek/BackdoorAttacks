import torch
from process.base import BaseProcess
from utils.utils import AverageMeter, accuracy, import_class
from utils.utils import save_images

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


class Classifier(BaseProcess):
    def __init__(self, opt) -> None:
        super().__init__(opt)

        # load model
        self.model = self._load_model()
        self.model = self.model.cuda()
        
        # load optimizer
        self.optimizer = self._load_optimizer(self.model.parameters())
        self.scheduler = CosineAnnealingLR(self.optimizer, self.opt.epochs, eta_min=0)
        self.criterion = torch.nn.CrossEntropyLoss()

        # load attack method
        attack_method = '.'.join(['methods', self.opt.method.lower(), self.opt.method])
        self.attacker = import_class(attack_method)(self.opt)

        # load data
        self.train_loader, self.ori_test_loader, self.poi_test_loader = self._load_data()

        self._print_info()

    def _print_info(self):
        print('-' * 30)
        print("Datast: {}".format(self.opt.dataset))
        print("Model: {}".format(self.opt.model))
        print("Attack: {}".format(self.opt.method))
        print('-' * 30)

    def _load_data(self):
        # load data
        dataset = self.attacker.get_poisoned_data(train=True, p=self.opt.percentage)
        train_loader = DataLoader(dataset=dataset, batch_size=self.opt.bs, num_workers=16, shuffle=True)
        
        dataset = self.attacker.get_poisoned_data(train=False, p=0)
        ori_test_loader = DataLoader(dataset=dataset, batch_size=self.opt.bs, num_workers=16, shuffle=False)
        
        dataset = self.attacker.get_poisoned_data(train=False, p=1)
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
            loss = self._train(epoch)
            self.scheduler.step()

            print('Epoch: {} \t loss={:.5f}'.format(epoch, loss))
            writer.add_scalar('train_loss', loss, epoch)

            if (epoch + 1) % 10 == 0:
                self.eval()

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
                            images[:16], step=idx)
        writer.close()
        return top1.avg

    def eval(self):
        ori_acc = self._eval(ori=True)
        poi_acc = self._eval(ori=False)

        print("Original Acc@1: {:.5f}".format(ori_acc))
        print("Poisoned Acc@1: {:.5f}".format(poi_acc))

