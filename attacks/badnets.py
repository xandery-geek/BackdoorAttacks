import numpy as np
from attacks.base import BaseAttack
from data.trigger import PatchTrigger
from data.utils import get_image_size
from data.dataset import PoisonedDataset
from data.utils import load_data, get_transform
from torchvision import transforms


class BadNetsDataset(PoisonedDataset):
    def __init__(self, dataset, poi_param, p, transform=None, pre_transform=None) -> None:
        super().__init__(dataset, poi_param, p, transform)
        
        self.pre_transform = pre_transform
        self.poi_param = poi_param

        # set trigger
        image_size = self.poi_param['image_size']
        patch_size = self.poi_param['patch_size']
        
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        patch = np.zeros((image_size, image_size), dtype=np.uint8)
        
        mask[image_size-patch_size: image_size, image_size-patch_size: image_size] = 1
        patch[image_size-patch_size: image_size, image_size-patch_size: image_size] = 255

        self.trigger = PatchTrigger(mask, patch, mode='HWC')

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.pre_transform is not None:
            img = self.pre_transform(img)

        # add trigger
        if index in self.poisoned_index:
            img= self.trigger(img)
            target = self.poisoned_target

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class BadNets(BaseAttack):
    """
    Attack: BadNets

    Paper:
        BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
        https://arxiv.org/abs/1708.06733
    """
    def __init__(self, opt) -> None:
        super().__init__('BadNets', opt)

        image_size = get_image_size(self.opt.dataset)
        patch_size = int(image_size // 10 + 1)

        self.poi_param = {
            'target': self.opt.target,
            'image_size' : image_size,
            'patch_size': patch_size
        }

    def get_poisoned_data(self, train, p=0.1):
        transforms_list = get_transform(self.opt.dataset, train=train).transforms
        pre_transform = transforms.Compose(transforms_list[:-2])
        transform = transforms.Compose(transforms_list[-2:])

        dataset = load_data(self.opt.data_path, self.opt.dataset, train=train)
        poisoned_data = BadNetsDataset(dataset, self.poi_param, p, transform=transform, 
                                        pre_transform=pre_transform)
        return poisoned_data
