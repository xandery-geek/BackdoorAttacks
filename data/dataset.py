import copy
from statistics import mode
import numpy as np
from torch.utils.data import Dataset


class NormalDataset(Dataset):
    def __init__(self, dataset, transform=None) -> None:
        """
        datasetdata: original dataset
        """
        super().__init__()
        self.transform = transform

        self.classes = dataset.classes
        self.data = copy.deepcopy(dataset.data)
        self.targets = copy.deepcopy(dataset.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger, poisoned_target, p, mode='replace', transform=None) -> None:
        """
        dataset: original dataset
        trigger: trigger for backdoor
        poisoned_target: target label for poisoned sample
        p: poisoned percentage
        mode: poisoned mode, option from ['replace', 'merge']
        """
        super().__init__()
        self.classes = dataset.classes
        self.trigger = trigger
        self.poisoned_target = poisoned_target
        self.mode = mode
        self.p = p
        self.transform = transform
        self.data, self.targets = self._add_trigger(dataset.data, dataset.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    def _add_trigger(self, data, targets):
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        if isinstance(new_targets, list):
            new_targets = np.array(new_targets)
        num_data = len(new_data)

        perm = np.random.permutation(num_data)[0: int(num_data * self.p)]

        if self.mode == 'replace':
            new_data[perm] = self.trigger(new_data[perm])
            new_targets[perm] = self.poisoned_target
        elif self.mode == 'merge':
            merge_data = copy.deepcopy(new_data[perm])
            merge_targets = copy.deepcopy(new_targets[perm])
            
            merge_data = self.trigger(merge_data)
            merge_targets = self.poisoned_target

            new_data = np.concatenate((new_data, merge_data), axis=0)
            new_targets = np.concatenate((new_targets, merge_targets), axis=0)
        else:
            raise NotImplementedError('mode {} is not supported!'.format(mode))
        
        return new_data, new_targets
