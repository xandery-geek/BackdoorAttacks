from abc import abstractmethod


class BaseAttack(object):
    def __init__(self, name, opt) -> None:
        self.name = name
        self.opt = opt

    @abstractmethod
    def get_poisoned_data(self, *args):
        pass
