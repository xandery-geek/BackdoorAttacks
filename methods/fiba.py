import numpy as np
from PIL import Image
from methods.base import BaseAttack
from data.dataset import PoisonedDataset
from data.utils import load_data, get_image_size, get_transform
from torchvision import transforms


def fourier_pattern(img, target_img, beta, ratio):

    #  get the amplitude and phase spectrum of trigger image
    fft_trg_cp = np.fft.fft2(target_img, axes=(-2, -1))  
    amp_target, _ = np.abs(fft_trg_cp), np.angle(fft_trg_cp)  
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-2, -1))
    #  get the amplitude and phase spectrum of source image
    fft_source_cp = np.fft.fft2(img, axes=(-2, -1))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-2, -1))

    # swap the amplitude part of local image with target amplitude spectrum
    c, h, w = img.shape
    b = (np.floor(np.amin((h, w)) * beta)).astype(int)  
    # 中心点
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    amp_source_shift[:, h1:h2, w1:w2] = amp_source_shift[:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,h1:h2, w1:w2]) * ratio
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-2, -1))

    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg


def poison(img, target_img, beta=0.1, alpha=0.15):
    img, target_img = np.transpose(img, (2, 0, 1)), np.transpose(target_img, (2, 0, 1))
    poi_img = fourier_pattern(img, target_img ,beta, alpha)
    poi_img = np.transpose(poi_img, (1, 2, 0))
    poi_img = np.clip(poi_img, 0, 255).astype(np.uint8)

    return poi_img


class FIBADataset(PoisonedDataset):
    def __init__(self, dataset, poi_param, p, transform=None, pre_transform=None) -> None:
        super().__init__(dataset, poi_param, p, transform)
        
        self.pre_transform = pre_transform
        self.poi_param = poi_param
        self.target_img = self.load_target_img()
    
    def load_target_img(self):
        image_size = self.poi_param['image_size']
        target_img = Image.open(self.poi_param['target_image'])
        target_img = target_img.resize((image_size, image_size))
        target_img = np.array(target_img)
        return target_img
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.pre_transform is not None:
            img = self.pre_transform(img)

        # add trigger
        if index in self.poisoned_index:
            img_arr = np.array(img)
            img_arr = poison(img_arr, self.target_img)
            img = Image.fromarray(img_arr)
            target = self.poisoned_target

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class FIBA(BaseAttack):
    """
    Attack: FIBA

    Paper:
        FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis
        https://arxiv.org/abs/2112.01148
    """
    def __init__(self, opt) -> None:
        super().__init__('FIBA', opt)

        self.poi_param = {
            'target': self.opt.target,
            'image_size': get_image_size(self.opt.dataset),
            'target_image' : 'methods/source/fiba_target.jpg',
        }

    def get_poisoned_data(self, train, p=0.1):
        transforms_list = get_transform(self.opt.dataset, train=train).transforms
        pre_transform = transforms.Compose(transforms_list[:-2])
        transform = transforms.Compose(transforms_list[-2:])

        dataset = load_data(self.opt.data_path, self.opt.dataset, train=train)
        poisoned_data = FIBADataset(dataset, self.poi_param, p, transform=transform, 
                                        pre_transform=pre_transform)
        return poisoned_data
