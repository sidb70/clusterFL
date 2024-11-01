from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Dict, Any


def Transform(rotation: float = 0.0) -> transforms.Compose:
    """
    Create a transformation for datasets.

    Args:
        rotation (float): The rotation to apply to the images.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(rotation, rotation)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform


class RotatedDataset(Dataset):
    """
    A dataset that rotates the images by the given rotation.
    """

    def __init__(self, dataset: Dataset, rotation: float = 0.0):
        """
        Initialize the RotatedDataset.

        Args:
            dataset (Dataset): The dataset to rotate
            rotation (float): The rotation to apply to the images.
        """
        self.dataset = dataset
        self.rotation = rotation
        if hasattr(dataset, "transform"):
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
    """
    A dataset that only contains data from the selected classes of the full dataset.
    """

    def __init__(self, dataset: Dataset, selected_classes: List[int]):
        """
        Initialize the SelectedClassesDataset.

        Args:
            dataset (Dataset): The dataset to select classes from
            selected_classes (List[int]): The classes to select from the dataset.
        """
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
    """
    Load the CIFAR-10 dataset.
    """
    transform = Transform()
    # Download and load the training data
    trainset = datasets.CIFAR10(
        "./datasets/cifar-10/", download=True, train=True, transform=transform
    )
    testset = datasets.CIFAR10(
        "./datasets/cifar-10/", download=True, train=False, transform=transform
    )
    return trainset, testset


def load_cifar100() -> Tuple[Dataset, Dataset]:
    """
    Load the CIFAR-100 dataset.
    """
    transform = Transform()
    # Download and load the training data
    trainset = datasets.CIFAR100(
        "./datasets/cifar-100/", download=True, train=True, transform=transform
    )
    testset = datasets.CIFAR100(
        "./datasets/cifar-100/", download=True, train=False, transform=transform
    )
    return trainset, testset


def load_mnist() -> Tuple[Dataset, Dataset]:
    """
    Load the MNIST dataset.
    """
    transform = Transform()
    # Download and load the training data
    trainset = datasets.MNIST(
        "./datasets/MNIST/", download=True, train=True, transform=transform
    )
    testset = datasets.MNIST(
        "./datasets/MNIST/", download=True, train=False, transform=transform
    )
    return trainset, testset


def load_global_dataset(dataset_name: str) -> Dataset:
    """
    Load the dataset specified by the dataset_name.

    Args:
        dataset_name (str): The name of the dataset to load
    """
    if dataset_name == "cifar10":
        return load_cifar10()
    elif dataset_name == "cifar100":
        return load_cifar100()
    elif dataset_name == "mnist":
        return load_mnist()


def create_clustered_dataset(
    dataset: Dataset, num_clusters: int, cluster_type: str
) -> List[Dataset]:
    """
    Create a clustered dataset from the given dataset based on the partitioning strategy specified by cluster_type.

    Args:
        dataset (Dataset): The dataset to cluster
        num_clusters (int): The number of clusters to create
        cluster_type (str): The type of clustering to perform on the dataset
    """
    if cluster_type == "rotation":
        datasets = []
        for i in range(num_clusters):
            rotation = (i / num_clusters) * 360
            datasets.append(RotatedDataset(dataset=dataset, rotation=rotation))
    elif cluster_type == "selected_classes":
        datasets = []
        classes_per_cluster = len(dataset.classes) // num_clusters
        for cluster_id in range(num_clusters):
            selected_classes = dataset.classes[
                cluster_id
                * classes_per_cluster : (cluster_id + 1)
                * classes_per_cluster
            ]
            datasets.append(SelectedClassesDataset(dataset, selected_classes))
    else:
        raise ValueError(f"Cluster type {cluster_type} not supported")
    return datasets


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_cifar10(32, 0.2)
    print("Train size: ", len(train_loader) * 32)
    print("Val size: ", len(val_loader) * 32)
    print("Test size: ", len(test_loader) * 32)
