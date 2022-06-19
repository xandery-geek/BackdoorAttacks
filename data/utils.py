import os
from torchvision import datasets

dataset_dict = {
    'mnist': 'MNIST',
    'cifar-10': 'CIFAR-10'
}


def load_data(data_path, dataset=None, train=True, target_transform=None):

    if dataset is None:
        dataset_path = data_path
    else:
        try:
            dataset_path = os.path.join(data_path, dataset_dict[dataset], 'train' if train else 'test')
        except KeyError:
            raise NotImplementedError('dataset {} is not supported!'.format(dataset))

    dataset = datasets.ImageFolder(root=dataset_path, target_transform=target_transform)
    return dataset


def get_num_classes(dataset):
    classes_dict = {'mnist': 10, 'cifar-10': 10}
    return classes_dict[dataset]


def get_image_size(dataset):
    size_dict = {'mnist': 28, 'cifar-10': 32}
    return size_dict[dataset]
