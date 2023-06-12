import random
import numpy as np
from abc import abstractclassmethod
from PIL import Image

class BaseTrigger(object):
    def __init__(self, name, mode='CHW') -> None:
        self.name = name
        self.mode = mode
    
    @abstractclassmethod
    def __call__(self, img):
        pass
    
    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class PixelTrigger(BaseTrigger):
    def __init__(self, position, value, mode='CHW') -> None:
        super().__init__('Pixel Trigger', mode=mode)
        self.position = position
        self.value = value

    def __call__(self, img):
        img_arr = np.array(img)
        img_arr[:, self.position[0], self.position[1]] = self.value
        return Image.fromarray(img_arr)


class PatchTrigger(BaseTrigger):
    def __init__(self, patch_img, image_size, patch_size, pos_strategy='fixed', mode='HWC') -> None:
        super().__init__('Patch Trigger', mode=mode)
        assert pos_strategy in ['fixed', 'random']

        self.patch_img = patch_img
        self.image_size = image_size
        self.patch_size = patch_size
        self.pos_strategy = pos_strategy

    def __call__(self, img):

        img_arr = np.array(img)

        if self.pos_strategy == 'random':
            pos = [random.randint(0, self.image_size - self.patch_size),
                   random.randint(0, self.image_size - self.patch_size)]
        else:
            pos = [self.image_size - self.patch_size, self.image_size - self.patch_size]

        img_arr[pos[0]: pos[0] + self.patch_size, pos[1]: pos[1] + self.patch_size, :] = self.patch_img
        return Image.fromarray(img_arr)
