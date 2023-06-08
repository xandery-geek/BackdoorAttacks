import numpy as np
from PIL import Image
from attacks.base import BaseAttack
from data.dataset import PoisonedDataset
from data.utils import load_data, get_transform
from torchvision import transforms


def sig(img, delta, freq):
    overlay = np.zeros(img.shape, np.float64)
    _, m, _ = overlay.shape
    for i in range(m):
        overlay[:, i] = delta * np.sin(2 * np.pi * i * freq/m)
    overlay = np.clip(overlay + img, 0, 255).astype(np.uint8)
    return overlay


class SIGDataset(PoisonedDataset):
    def __init__(self, dataset, poi_param, p, transform=None, pre_transform=None) -> None:
        super().__init__(dataset, poi_param, p, transform)
        
        self.pre_transform = pre_transform
        self.poi_param = poi_param

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.pre_transform is not None:
            img = self.pre_transform(img)

        # add trigger
        if index in self.poisoned_index:
            img_arr = np.array(img)
            img_arr = sig(img_arr, self.poi_param['delta'], self.poi_param['frequency'])
            img = Image.fromarray(img_arr)
            target = self.poisoned_target

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class SIG(BaseAttack):
    """
    Attack: SIG

    Paper:
        A New Backdoor Attack in CNNS by Training Set Corruption Without Label Poisoning
        https://arxiv.org/abs/1902.11237
    """
    def __init__(self, opt) -> None:
        super().__init__('SIG', opt)

        self.poi_param = {
            'target': self.opt.target,
            'delta' : 20,
            'frequency': 6
        }

    def get_poisoned_data(self, train, p=0.1):
        transforms_list = get_transform(self.opt.dataset, train=train).transforms
        pre_transform = transforms.Compose(transforms_list[:-2])
        transform = transforms.Compose(transforms_list[-2:])

        dataset = load_data(self.opt.data_path, self.opt.dataset, train=train)
        poisoned_data = SIGDataset(dataset, self.poi_param, p, transform=transform, 
                                        pre_transform=pre_transform)
        return poisoned_data
