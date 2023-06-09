import torch
from victims.base import BaseProcess
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import AverageMeter, accuracy, import_class
from utils.utils import save_images
from utils.output import output_iter, ansi


class Classifier(BaseProcess):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # load model
        self.model = self._load_model()
        self.model = self.model.cuda()
        
        # load optimizer
        self.optimizer, self.scheduler = self._load_optimizer(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

        # load attack method
        attack_method = '.'.join(['attacks', self.cfg.attack.lower(), self.cfg.attack])
        self.attacker = import_class(attack_method)(self.cfg)

        # load data
        self.train_loader, self.ori_test_loader, self.poi_test_loader = self._load_data()

        # init writer
        self.writer = SummaryWriter(log_dir=self.tb_path)
        self._print_info()

    
    def __del__(self):
        self.writer.close()

    def _print_info(self):
        print('-' * 50)
        print("Datast: {}".format(self.cfg.dataset))
        print("Model: {}".format(self.cfg.model))
        print("Attack: {}".format(self.cfg.attack))
        print("Checkpoint path: {}".format(self.ckpt_path))
        print("Log path: {}".format(self.log_path))
        print('-' * 50)

    def _load_data(self):
        # load data
        dataset = self.attacker.get_poisoned_data(train=True, p=self.cfg.percentage)
        train_loader = DataLoader(dataset=dataset, batch_size=self.cfg.bs, num_workers=16, shuffle=True)
        
        dataset = self.attacker.get_poisoned_data(train=False, p=0)
        ori_test_loader = DataLoader(dataset=dataset, batch_size=self.cfg.bs, num_workers=16, shuffle=False)
        
        dataset = self.attacker.get_poisoned_data(train=False, p=1)
        poi_test_loader = DataLoader(dataset=dataset, batch_size=self.cfg.bs, num_workers=16, shuffle=False)
        
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
        for epoch in range(self.cfg.epochs):
            loss = self._train(epoch)
            if self.scheduler:
                self.scheduler.step()

            print('{blue_light}Epoch:{reset} {}\t {green}loss{reset}: {:.5f}'
                  .format(output_iter(epoch, self.cfg.epochs), loss, **ansi))
            self.writer.add_scalar('train_loss', loss, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            if (epoch + 1) % self.cfg.eval_every_epochs == 0:
                self.eval(epoch=epoch)

        self._save_model({
            'model': self.model.state_dict()
        }, 'last.pth')

    @torch.no_grad()
    def _eval_on_clean(self, epoch=None):
        self.model.eval()
        loss_meter = AverageMeter()
        top1 = AverageMeter()
        
        for idx, (images, labels) in enumerate( self.ori_test_loader):
            images, labels = images.cuda(), labels.cuda()
            output = self.model(images)

            loss = self.criterion(output, labels)
            acc1, _ = accuracy(output, labels, topk=(1, 5))

            loss_meter.update(loss.item(), labels.shape[0])
            top1.update(acc1[0], labels.shape[0])

            if epoch == 0 and idx == self.cfg.sample_batch:
                save_images(self.writer, 'ori_images', images[:16], step=epoch)

        if epoch:
            print('{yellow}Validation \t loss{reset}: {:.5f}'
                  .format(loss_meter.avg, **ansi))
            self.writer.add_scalar('ori_top1', top1.avg, epoch)
            self.writer.add_scalar('eval_loss', loss_meter.avg, epoch)

        return top1.avg
    
    @torch.no_grad()
    def _eval_on_poison(self, epoch=None):
        self.model.eval()
        top1 = AverageMeter()
        
        for idx, (images, labels) in enumerate(self.poi_test_loader):
            images, labels = images.cuda(), labels.cuda()
            output = self.model(images)

            acc1, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], labels.shape[0])

            if epoch == 0 and idx == self.cfg.sample_batch:
                save_images(self.writer, 'poi_images', images[:16], step=epoch)

        if epoch:
            self.writer.add_scalar('poi_top1', top1.avg, epoch)

        return top1.avg

    def eval(self, epoch=None):
        ori_acc = self._eval_on_clean(epoch=epoch)
        print('{yellow}Original Acc@1{reset}: {:.5f}'
                  .format(ori_acc, **ansi))

        if self.cfg.attack != 'Clean':
            poi_acc = self._eval_on_poison(epoch=epoch)
            print('{red}Poisoning Acc@1{reset}: {:.5f}'
                  .format(poi_acc, **ansi))
