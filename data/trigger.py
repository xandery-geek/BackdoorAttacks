import numpy as np
from abc import abstractclassmethod


class BaseTrigger(object):
    def __init__(self, name, mode='CHW') -> None:
        self.name = name
        self.mode = mode
    
    @abstractclassmethod
    def __call__(self, data):
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

    def __call__(self, data):
        data[:, self.position[0], self.position[1]] = self.value


class PatchTrigger(BaseTrigger):
    def __init__(self, mask, patch, mode='CHW') -> None:
        super().__init__('Patch Trigger', mode=mode)
        assert mask.shape == patch.shape
        self.mask = mask
        self.patch = patch

    def __call__(self, data):
        if self.mode == 'HWC':
            channel = data.shape[-1]
            if len(self.patch.shape) < 3:
                self.mask = np.expand_dims(self.mask, 2)
                self.patch = np.expand_dims(self.patch, 2)
            if self.patch.shape[2] == 1:
                self.mask = np.repeat(self.mask, channel, axis=2)
                self.patch = np.repeat(self.patch, channel, axis=2)

        return data * (1 - self.mask) + self.patch * self.mask
