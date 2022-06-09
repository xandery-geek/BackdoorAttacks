from abc import abstractclassmethod


class BaseAttack(object):
    def __init__(self, opt) -> None:
        self.opt = opt

    @abstractclassmethod
    def get_poisoned_data(self, *args):
        pass
