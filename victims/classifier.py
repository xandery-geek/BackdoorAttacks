import torch
import pytorch_lightning as pl
from victims.base import BaseProcess
from torch.utils.data import DataLoader
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import AverageMeter, accuracy, import_class, unnormalize
from utils.output import ansi


class Classifier(BaseProcess):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # load model
        self.model = self._load_model()
        self.model = self.model.cuda()
        
        # load criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # load attack method
        attack_method = '.'.join(['attacks', self.cfg.attack.lower(), self.cfg.attack])
        self.attacker = import_class(attack_method)(self.cfg)

        # load data
        self.train_loader, self.ori_test_loader, self.poi_test_loader = self._load_data()
        
        # init global variable
        self.top1_meter = AverageMeter()
        self.poi_top1_meter = AverageMeter()

        self._print_info()

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
        train_loader = DataLoader(dataset=dataset, batch_size=self.cfg.bs, num_workers=8, shuffle=True, persistent_workers=True)
        
        dataset = self.attacker.get_poisoned_data(train=False, p=0)
        ori_test_loader = DataLoader(dataset=dataset, batch_size=self.cfg.bs, num_workers=8, shuffle=False, persistent_workers=True)
        
        dataset = self.attacker.get_poisoned_data(train=False, p=1)
        poi_test_loader = DataLoader(dataset=dataset, batch_size=self.cfg.bs, num_workers=8, shuffle=False, persistent_workers=True)
        
        return train_loader, ori_test_loader, poi_test_loader
    
    def _sample_images(self, img_data, step):
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)

        mean = torch.tensor(list(mean))
        std = torch.tensor(list(std))

        for data in img_data:
            if data['step'] == step or data['step'] == -1:
                name, img = data['name'], data['img']
                img = unnormalize(img, mean, std)
                self.logger.experiment.add_image(name, img, step, dataformats='NCHW')
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        loss = self.criterion(output, labels)

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = self.collect_outputs(outputs, ['loss'])[0]

        batch_num = len(outputs)
        loss = sum(loss).item() / batch_num

        if self.cfg.enable_tb:
            # lr = self.optimizers().param_groups[0]['lr']
            # self.log("lr", lr, on_epoch=True, sync_dist=True)
            self.log("loss", loss, sync_dist=True, on_epoch=True)
    
    def on_validation_epoch_start(self):
        self.top1_meter.reset()
        self.poi_top1_meter.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        images, labels = batch
        output = self.model(images)
        loss = self.criterion(output, labels)
        acc1, _ = accuracy(output, labels, topk=(1, 5))

        if dataloader_idx == 0:
            self.top1_meter.update(acc1[0], labels.shape[0])
        else:
            self.poi_top1_meter.update(acc1[0], labels.shape[0])

        if self.global_rank == 0 and batch_idx == self.cfg.sample_batch:
            name = "ori_images" if dataloader_idx == 0 else "poi_images"
            self._sample_images([{
                "name": name, "img": images[:8].cpu(), "step":0
            }], self.current_epoch)

        return {"loss": loss}
    
    def validation_epoch_end(self, outputs):
        # only collect losses of the first dataloader, i.e., the clean dataloader
        loss = self.collect_outputs(outputs[0], ['loss'])[0]

        batch_num = len(outputs[0])
        loss = sum(loss).item() / batch_num
        
        if self.cfg.enable_tb:
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('top1', self.top1_meter.avg, on_epoch=True, sync_dist=True)
            self.log('poi_top1', self.poi_top1_meter.avg, on_epoch=True, sync_dist=True)
        
        if self.global_rank == 0:
            print('{yellow}Clean Acc@1{reset}: {:.5f}'.format(self.top1_meter.avg, **ansi))
            print('{red}Poisoned Acc@1{reset}: {:.5f}'.format(self.poi_top1_meter.avg, **ansi))

    def on_test_epoch_start(self):
        self.top1_meter.reset()
        self.poi_top1_meter.reset()

    def test_step(self, batch, batch_idx, dataloader_idx):
        images, labels = batch
        output = self.model(images)
        acc1, _ = accuracy(output, labels, topk=(1, 5))

        if dataloader_idx == 0:
            self.top1_meter.update(acc1[0], labels.shape[0])
        else:
            self.poi_top1_meter.update(acc1[0], labels.shape[0])

        return None
    
    def test_epoch_end(self, outputs):
        if self.global_rank == 0:
            print('{yellow}Clean Acc@1{reset}: {:.5f}'.format(self.top1_meter.avg, **ansi))
            print('{red}Poisoned Acc@1{reset}: {:.5f}'.format(self.poi_top1_meter.avg, **ansi))


def run(cfg):

    classifier = Classifier(cfg)

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='top1',
        dirpath=classifier.ckpt_path,
        save_last=True,
        mode='max')

    tb_logger = TensorBoardLogger(classifier.log_path) if cfg.enable_tb else None

    trainer = pl.Trainer(
        devices=len(cfg.device),
        accelerator='gpu',
        max_epochs=cfg.epochs,
        log_every_n_steps=30,
        check_val_every_n_epoch=cfg.every_n_epoch,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        strategy="ddp_find_unused_parameters_false"
    )
    
    if cfg.train:
        trainer.fit(model=classifier, train_dataloaders=classifier.train_loader, 
                    val_dataloaders=[classifier.ori_test_loader, classifier.poi_test_loader])

    else:
        trainer.test(model=classifier, ckpt_path=cfg.ckpt, 
                    dataloaders=[classifier.ori_test_loader, classifier.poi_test_loader])
