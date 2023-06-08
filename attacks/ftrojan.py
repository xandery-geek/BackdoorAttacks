import cv2
import numpy as np
from PIL import Image
from attacks.base import BaseAttack
from data.dataset import PoisonedDataset
from data.utils import load_data, get_transform
from torchvision import transforms


def RGB2YUV(x_rgb):
    return cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)


def YUV2RGB(x_yuv):
    return cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def DCT(x_train, window_size):
    x_train = np.transpose(x_train, (2, 0, 1))
    x_dct = np.zeros(x_train.shape, dtype=np.float64)

    for ch in range(x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_dct = cv2.dct(x_train[ch][w:w+window_size, h:h+window_size].astype(np.float64))
                x_dct[ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct  # x_dct: (ch, w, h)


def IDCT(x_train, window_size):
    x_idct = np.zeros(x_train.shape, dtype=np.float64)

    for ch in range(0, x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_idct = cv2.idct(x_train[ch][w:w+window_size, h:h+window_size].astype(np.float64))
                x_idct[ch][w:w+window_size, h:h+window_size] = sub_idct

    x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct  # x_idct: (w, h, ch)


def poison_frequency(x_train, param):

    x_train = x_train * 255.
    if param["YUV"]:
        # transfer to YUV channel
        x_train = RGB2YUV(x_train)
        
    # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (ch, w, h)

    # plug trigger frequency
    for ch in param["channel_list"]:
        for w in range(0, x_train.shape[1], param["window_size"]):
            for h in range(0, x_train.shape[2], param["window_size"]):
                for pos in param["pos_list"]:
                    w_pos = w + pos[0] if w + pos[0] < x_train.shape[1] else x_train.shape[1] - 1
                    h_pos = h + pos[1] if h + pos[1] < x_train.shape[2] else x_train.shape[2] - 1
                    x_train[ch][w_pos][h_pos] += param["magnitude"]
                        
    x_train = IDCT(x_train, param["window_size"])  # (w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)
    x_train = x_train / 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


class FTrojanDataset(PoisonedDataset):
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
            img_arr = np.array(img, dtype=np.float64)/255.
            img_arr = poison_frequency(img_arr, self.poi_param)
            img = Image.fromarray((img_arr * 255).astype(np.uint8))
            target = self.poisoned_target

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class FTrojan(BaseAttack):
    """
    Attack: FTrojan

    Paper:
        An Invisible Black-Box Backdoor Attack Through Frequency Domain
        https://link.springer.com/chapter/10.1007/978-3-031-19778-9_23
    """
    def __init__(self, opt) -> None:
        super().__init__('FTrojan', opt)

        self.poi_param = {
            'target': self.opt.target,
            "channel_list": [1, 2], # [0,1,2] means YUV channels, [1,2] means UV channels
            "degree": 0,
            "magnitude": 50,
            "YUV": True,
            "window_size": 32,
            "pos_list": [(31, 31)],
        }

    def get_poisoned_data(self, train, p=0.1):
        transforms_list = get_transform(self.opt.dataset, train=train).transforms
        pre_transform = transforms.Compose(transforms_list[:-2])
        transform = transforms.Compose(transforms_list[-2:])

        dataset = load_data(self.opt.data_path, self.opt.dataset, train=train)
        poisoned_data = FTrojanDataset(dataset, self.poi_param, p, transform=transform, 
                                        pre_transform=pre_transform)
        return poisoned_data