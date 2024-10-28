from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler    


def Transform(rotation: float = 0.0) -> transforms.Compose:
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(rotation, rotation)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform
class RotatedDataset(Dataset):
    def __init__(self, dataset: Dataset, rotation: float = 0.0):
        self.dataset = dataset
        self.rotation = rotation
        if hasattr(dataset, 'transform'):
            self.original_transform = dataset.transform
            dataset.transform = None 
        self.transform = Transform(rotation)
    def __getitem__(self, index: int):
        data, label = self.dataset[index]
        if isinstance(data, torch.Tensor):
            return self.transform(data), label
        return self.transform(data), label
    def __len__(self):
        return len(self.dataset)

class SelectedClassesDataset(Dataset):
    def __init__(self, dataset: Dataset, selected_classes: List[int]):
        selected_classes_data = []
        for i in range(len(dataset)):
            if dataset[i][1] in selected_classes:
                selected_classes_data.append(dataset[i])
        self.selected_classes_data = selected_classes_data
    def __getitem__(self, index: int):
        return self.selected_classes_data[index]
    def __len__(self):
        return len(self.selected_classes_data)     
def load_cifar10() -> Tuple[Dataset, Dataset]:
    transform = Transform()
    # Download and load the training data
    trainset = datasets.CIFAR10('./datasets/cifar-10/', download=True, train=True, transform=transform)
    testset = datasets.CIFAR10('./datasets/cifar-10/', download=True, train=False, transform=transform)
    return trainset, testset

def load_cifar100() -> Tuple[Dataset, Dataset]:
    transform = Transform()
    # Download and load the training data
    trainset = datasets.CIFAR100('./datasets/cifar-100/', download=True, train=True, transform=transform)
    testset = datasets.CIFAR100('./datasets/cifar-100/', download=True, train=False, transform=transform)
    return trainset, testset

def load_mnist() -> Tuple[Dataset, Dataset]:
    transform = Transform()
    # Download and load the training data
    trainset = datasets.MNIST('./datasets/MNIST/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('./datasets/MNIST/', download=True, train=False, transform=transform)
    return trainset, testset

def rotate_dataset(dataset: Dataset, rotation: float = 0.0) -> Dataset:
    return RotatedDataset(dataset, rotation)

def load_global_dataset(dataset_name: str) -> Dataset:
    if dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'cifar100':
        return load_cifar100()
    elif dataset_name == 'mnist':
        return load_mnist()

if __name__=='__main__':
    train_loader, val_loader, test_loader = load_cifar10(32, 0.2)
    print("Train size: ", len(train_loader) * 32)
    print("Val size: ", len(val_loader) * 32)
    print("Test size: ", len(test_loader) * 32)