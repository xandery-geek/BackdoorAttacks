import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from data.trigger import PatchTrigger
from methods.base import BaseAttack
from data.utils import get_image_size
from data.utils import load_data
from data.dataset import PoisonedDataset, NormalDataset
from networks.backbone import ResNetFeature


class PerturbationDataset(Dataset):
    def __init__(self, data, target, transform=None) -> None:
        super().__init__()
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
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
        self.model = ResNetFeature(self.opt.model, pretrained=True)
        # self.model.copy_from_resnet(self.opt.ckpt)
        self.model = self.model.cuda()
        self.model.eval()
    
    def adversarial_loss(self, feature):
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


    def perturb_feature(self, dataset, poisoned_target):
        if isinstance(dataset.targets, list):
            dataset.targets = np.array(dataset.targets)
            
        targeted_idx = np.argwhere(dataset.targets == poisoned_target)
        targeted_idx = targeted_idx.squeeze()
        new_data = copy.deepcopy(dataset.data[targeted_idx])

        if self.opt.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.div(255)),
                transforms.Lambda(lambda x: x.repeat(3,1,1))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        new_dataset = PerturbationDataset(new_data, poisoned_target, transform=transform)
        data_loader = DataLoader(dataset=new_dataset, batch_size=self.opt.bs, num_workers=16, shuffle=True)

        for img, _, idx in data_loader:
            # perturb semantic
            adv_img = self.adversarial_perturbation(img)
            
            # convert tentor to ndarray
            adv_img = (adv_img * 255).cpu().numpy().astype(dataset.data.dtype)
            adv_img = adv_img.transpose(0, 2, 3, 1)

            # add trigger
            poisoned_img = self.trigger(adv_img)
            new_data[idx] = poisoned_img
        
        # combine poisoned images with benign images
        dataset.data = np.concatenate((dataset.data, new_data), axis=0)
        dataset.targets = np.concatenate((dataset.targets, np.array([poisoned_target] * len(new_data))), axis=0)
        return dataset

    def get_poisoned_data(self, poisoned_target, train, p=0.1, mode='replace', transform=None):
        if train:
            dataset = load_data(self.opt.data_path, self.opt.dataset, train=True)
            poisoned_dataset = self.perturb_feature(dataset, poisoned_target)
            poisoned_data = NormalDataset(poisoned_dataset, transform=transform)
        else:
            dataset = load_data(self.opt.data_path, self.opt.dataset, train=False)
            poisoned_data = PoisonedDataset(dataset, self.trigger, poisoned_target, p, mode=mode, transform=transform)
        return poisoned_data


    def save_poisoned_data(self):
        pass