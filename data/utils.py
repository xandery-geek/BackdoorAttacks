from torchvision import datasets


def load_data(data_path, dataset, train=True):
    if dataset == 'mnist':
        return datasets.MNIST(root=data_path, train=train, download=True)
    elif dataset == 'cifar-10':
        return datasets.CIFAR10(root=data_path, train=train, download=True)
    else:
        raise NotImplementedError('dataset {} is not supported!'.format(dataset))


def get_num_classes(dataset):
    classes_dict = {'mnist': 10, 'cifar-10': 10}
    return classes_dict[dataset]


def get_image_size(dataset):
    size_dict = {'mnist': 28, 'cifar-10': 32}
    return size_dict[dataset]