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


class PoisonedDataset(NormalDataset):
    def __init__(self, dataset, poi_param, p, transform=None) -> None:
        """
        dataset: `VisionDataset` instance
        poi_param: poisoned parameters
        p: poisoned percentage
        mode: poisoned mode, option from ['replace', 'merge']
        """
        super().__init__(dataset, transform)

        self.p = p
        self.poisoned_target = poi_param['target']

        num_data = len(self.imgs)
        self.poisoned_index = self.get_random_indices(range(num_data), int(num_data * self.p))

    @staticmethod
    def get_random_indices(a, num, seed=1):
        rng = np.random.RandomState(seed=seed)
        indices = rng.choice(a, num, replace=False)
        return indices
    
    def __len__(self):
        return len(self.imgs)


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