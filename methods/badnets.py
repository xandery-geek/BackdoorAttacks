import numpy as np
from methods.base import BaseAttack
from data.trigger import PatchTrigger
from data.utils import get_image_size
from data.dataset import PoisonedDataset
from data.utils import load_data


class BadNets(BaseAttack):
    def __init__(self, opt) -> None:
        super().__init__(opt)

        # set trigger
        image_size = get_image_size(self.opt.dataset)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        patch = np.zeros((image_size, image_size), dtype=np.uint8)
        
        mask[image_size-4: image_size-1, image_size-4: image_size-1] = 1
        patch[image_size-4: image_size-1, image_size-4: image_size-1] = 255

        self.trigger = PatchTrigger(mask, patch, mode='HWC')

    def get_poisoned_data(self, poisoned_target, train, p=0.1, mode='replace', transform=None):        
        dataset = load_data(self.opt.data_path, self.opt.dataset, train=train)
        poisoned_data = PoisonedDataset(dataset, self.trigger, poisoned_target, p, mode=mode, transform=transform)
        return poisoned_data
    