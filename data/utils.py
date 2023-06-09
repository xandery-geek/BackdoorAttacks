import os
from torchvision import datasets
from torchvision import transforms

dataset_dict = {
    'mnist': 'MNIST',
    'cifar-10': 'CIFAR-10',
    'tiny': 'Tiny-ImageNet',
    'imagenet': 'ImageNet'
}


def load_data(data_path, dataset=None, train=True, transform=None, target_transform=None):

    if dataset is None:
        dataset_path = data_path
    else:
        try:
            dataset_path = os.path.join(data_path, dataset_dict[dataset], 'train' if train else 'test')
        except KeyError:
            raise NotImplementedError('dataset {} is not supported!'.format(dataset))

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform, target_transform=target_transform)
    return dataset


def get_num_classes(dataset):
    classes_dict = {'mnist': 10, 'cifar-10': 10, 'imagenet': 100, 'tiny': 200}
    return classes_dict[dataset]


def get_image_size(dataset):
    size_dict = {'mnist': 28, 'cifar-10': 32, 'imagenet': 224, 'tiny': 64}
    return size_dict[dataset]


def get_padding_size(dataset):
    size_dict = {'mnist': 4, 'cifar-10': 4, 'imagenet': 16, 'tiny': 4}
    return size_dict[dataset]


def get_transform(dataset, train=True):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)

    image_size = get_image_size(dataset)
    padding_size = get_padding_size(dataset)
    if train:
        transform = [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=padding_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]

    transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform = transforms.Compose(transform)
    return transform
