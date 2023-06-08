import argparse

from . import badnets
from . import clean
from . import sig
from . import fiba
from . import ftrojan


def add_argument(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('attack', 'arguments for attack method')
    group.add_argument('--target', type=int, default=0, help='poisoned target')
    group.add_argument('--percentage', type=float, default=0.01, help='poisoned percentage')
