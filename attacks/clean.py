from attacks.base import BaseAttack
from data.dataset import NormalDataset
from data.utils import load_data, get_transform


class Clean(BaseAttack):
    """
    Without Backdoor Attack
    """
    def __init__(self, opt) -> None:
        super().__init__('Clean', opt)

    def get_poisoned_data(self, train, p=0.1):
        transform = get_transform(self.opt.dataset, train=train)

        dataset = load_data(self.opt.data_path, self.opt.dataset, train=train)
        poisoned_data = NormalDataset(dataset, transform=transform)
        return poisoned_data
