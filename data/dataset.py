import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class NormalDataset(Dataset):
    def __init__(self, dataset, transform=None) -> None:
        """
        datasetdata: original dataset
        """
        super().__init__()
        self.transform = transform

        self.imgs = copy.deepcopy(dataset.imgs)
        self.targets = copy.deepcopy(dataset.targets)

        self.loader = pil_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger, poisoned_target, p, transform=None, pre_transform=None) -> None:
        """
        dataset: `VisionDataset` instance
        trigger: trigger for backdoor
        poisoned_target: target label for poisoned sample
        p: poisoned percentage
        mode: poisoned mode, option from ['replace', 'merge']
        """
        super().__init__()
        self.imgs = copy.deepcopy(dataset.imgs)
        self.targets = copy.deepcopy(dataset.targets)

        self.transform = transform
        self.pre_transform = pre_transform
        self.loader = pil_loader

        self.p = p
        self.trigger = trigger
        self.poisoned_target = poisoned_target

        num_data = len(self.imgs)
        self.poisoned_index = np.random.permutation(num_data)[0: int(num_data * self.p)]
        

    def __len__(self):
        return len(self.imgs)

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


class MergeDataset(Dataset):
    def __init__(self, dataset1, dataset2, transform=None) -> None:
        """
        datasetdata: original dataset
        """
        super().__init__()
        self.transform = transform

        self.imgs = copy.deepcopy(dataset1.imgs)
        self.imgs.extend(copy.deepcopy(dataset2.imgs))

        self.targets = copy.deepcopy(dataset1.targets)
        self.targets.extend(copy.deepcopy(dataset2.targets))

        self.loader = pil_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
                
        if self.transform is not None:
            img = self.transform(img)

        return img, target