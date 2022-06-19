import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from data.trigger import PatchTrigger
from methods.base import BaseAttack
from data.dataset import PoisonedDataset, MergeDataset
from networks.backbone import ResNetFeature
from data.dataset import pil_loader
from data.utils import get_image_size
from data.utils import load_data, dataset_dict


class PerturbationDataset(Dataset):
    def __init__(self, imgs, target, transform=None) -> None:
        super().__init__()
        self.imgs = imgs
        self.target = target
        self.transform = transform
        self.loader = pil_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, self.target, index


class CleanLabel(BaseAttack):
    def __init__(self, opt) -> None:
        super().__init__(opt)

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
        
        # set model
        self.model = ResNetFeature(self.opt.model, pretrained=False)

        if self.opt.pre_ckpt != '' and os.path.exists(self.opt.pre_ckpt):
            print('==> Initial ResNetFeature with {}'.format(self.opt.pre_ckpt))
            state_dict = torch.load(self.opt.pre_ckpt)
            self.model.copy_from_resnet(state_dict['model'])
        else:
            print("Please specify (correct) `pre_ckpt`, not {}!".format(self.opt.pre_ckpt))
        
        self.model = self.model.cuda()
        self.model.eval()

        model_name = '{}_{}_{}'.format(self.opt.dataset, self.opt.model, self.opt.trial)
        self.dataset_path = os.path.join(self.opt.save, '{}_datasets'.format(self.opt.method), model_name)
    
    @staticmethod
    def adversarial_loss(feature):
        batch_size = feature.shape[0]
        device = (torch.device('cuda') if feature.is_cuda else torch.device('cpu'))
        
        feature = F.normalize(feature, dim=1)
        inner_product = feature @ feature.t()  # [batch, batch]
        mask = (1 - torch.eye(batch_size)).to(device)
        inner_product = mask * inner_product
        # norm_2 = torch.norm(feature, p=2, dim=1)
        # loss = inner_product / torch.clamp_min(norm_2 @ norm_2, 1e-08)
        loss = torch.mean(inner_product)
        return loss
        
    def adversarial_perturbation(self, img, iteration=10, step=1, epsilon=8/255.0):
        img = img.cuda()
        delta = torch.zeros_like(img).cuda()
        delta.requires_grad = True

        for _ in range(iteration):
            feature = self.model(img + delta)
            loss = self.adversarial_loss(feature)
            loss.backward()

            delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (img.data + delta.data).clamp(0, 1) - img.data
            delta.grad.zero_()
        return img + delta.detach()

    def generate_poisoned_dataset(self, dataset, poisoned_target):
        if os.path.exists(self.dataset_path):
            print('==> Removing old poisoned dataset :{}!'.format(self.dataset_path))
            shutil.rmtree(self.dataset_path)

        print('==> Generating new poisoned dataset!')
        ori_images = np.array(dataset.imgs)
        ori_target = np.array(dataset.targets)
            
        targeted_idx = np.argwhere(ori_target == poisoned_target)
        targeted_idx = targeted_idx.squeeze()
        new_imgs = ori_images[targeted_idx]

        if self.opt.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.div(255)),
                transforms.Lambda(lambda x: x.repeat(3,1,1))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        new_dataset = PerturbationDataset(new_imgs, poisoned_target, transform=transform)
        data_loader = DataLoader(dataset=new_dataset, batch_size=self.opt.bs, num_workers=16, shuffle=True)

        for img, _, idx in data_loader:
            # perturb semantic
            adv_img = self.adversarial_perturbation(img)
            
            # convert tentor to ndarray
            adv_img = (adv_img * 255).cpu().numpy().astype(np.uint8)
            adv_img = adv_img.transpose(0, 2, 3, 1)

            # add trigger
            poisoned_img = self.trigger(adv_img)
            self.save_poisoned_img(new_dataset, poisoned_img, idx)

    def get_poisoned_data(self, poisoned_target, train, p=0.1, transform=None):
        if train:
            benign_dataset = load_data(self.opt.data_path, self.opt.dataset, train=True)

            if self.opt.regenerate or not os.path.exists(self.dataset_path):
                self.generate_poisoned_dataset(benign_dataset, poisoned_target)

            target_transform = lambda _: poisoned_target
            poisoned_dataset = load_data(self.dataset_path, target_transform=target_transform)
            
            # merge benign dataset and poisoned dataset
            poisoned_data = MergeDataset(benign_dataset, poisoned_dataset, transform)
        else:
            dataset = load_data(self.opt.data_path, self.opt.dataset, train=False)
            poisoned_data = PoisonedDataset(dataset, self.trigger, poisoned_target, p, transform=transform)
        return poisoned_data

    def save_poisoned_img(self, ori_dataset, poisoned_img, index):
        for i in range(len(index)):
            img = poisoned_img[i]
            img = Image.fromarray(img)
            
            ori_path, _ = ori_dataset.imgs[index[i]]
            ori_path_list = ori_path.split('/')

            class_dir, filename = ori_path_list[-2], ori_path_list[-1]
            path = os.path.join(self.dataset_path, class_dir)
            if not os.path.exists(path):
                os.makedirs(path)
            
            img.save(os.path.join(path, filename))
