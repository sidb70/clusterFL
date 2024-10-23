from typing import List, Tuple
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler    




def load_selected_classes(dataset: Dataset, selected_classes: List[int]) -> Dataset:
    selected_data = []
    for i in range(len(dataset)):
        if dataset[i][1] in selected_classes:
            selected_data.append(dataset[i])
    return selected_data

def load_cifar10(rotated: bool = False) -> Tuple[Dataset, Dataset]:
    # Define a transform to normalize the data
    if rotated:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomRotation(degrees=(89.999, 90.001)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    # Download and load the training data
    trainset = datasets.CIFAR10('./datasets/cifar-10/', download=True, train=True, transform=transform)
    testset = datasets.CIFAR10('./datasets/cifar-10/', download=True, train=False, transform=transform)
    return trainset, testset

def load_cifar100(rotated: bool = False) -> Tuple[Dataset, Dataset]:
    raise NotImplementedError

def load_mnist(rotated: bool = False) -> Tuple[Dataset, Dataset]:
    raise NotImplementedError

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_global_dataset(dataset_name: str) -> Dataset:
    if dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'cifar100':
        return load_cifar100()

if __name__=='__main__':
    train_loader, val_loader, test_loader = load_cifar10(32, 0.2)
    print("Train size: ", len(train_loader) * 32)
    print("Val size: ", len(val_loader) * 32)
    print("Test size: ", len(test_loader) * 32)