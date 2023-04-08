from abc import abstractclassmethod


class BaseAttack(object):
    def __init__(self, name, opt) -> None:
        self.name = name
        self.opt = opt

    @abstractclassmethod
    def get_poisoned_data(self, *args):
        pass
