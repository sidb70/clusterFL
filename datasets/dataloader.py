from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Dict, Any
from PIL import Image
def Transform(rotation: float = 0.0, num_dims: int = 3) -> transforms.Compose:
    """
    Create a transformation for datasets.

    Args:
        rotation (float): The rotation to apply to the images.
    """
    if num_dims == 1:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Then convert to tensor
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Then convert to tensor
                transforms.RandomRotation(degrees=(rotation, rotation)),  # Apply rotation first
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return transform


class ClusterDataset(Dataset):
    """
    A dataset that selects only the specified classes.
    """

    def __init__(self, dataset: List[Tuple[Any, int]], transform: transforms.Compose = None):
        """
        Initialize the ClusterDataset.

        Args:
            dataset (Dataset): The dataset to cluster
            transform (transforms.Compose): The transformation to apply to the images.
        """
        super(ClusterDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        data, label = self.dataset[index]
        # Apply the transformation if it's defined
        if self.transform:
            #cvt to pil if not already
            if not isinstance(data, Image.Image):
                data = transforms.ToPILImage()(data)
            data = self.transform(data)
        
        return data, label

    def __len__(self):
        return len(self.dataset)

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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Then convert to tensor
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    # Download and load the training data
    trainset = datasets.MNIST(
        "./datasets/MNIST/", download=True, train=True, transform=transform
    )
    testset = datasets.MNIST(
        "./datasets/MNIST/", download=True, train=False, transform=transform
    )
    return trainset, testset

def load_global_dataset(task: str) -> Dataset:
    """
    Load the dataset specified by the dataset_name.

    Args:
        task (str): The name of the dataset to load
    """
    if task == "cifar10":
        return load_cifar10()
    elif task == "cifar100":
        return load_cifar100()
    elif task == "mnist":
        return load_mnist()
    else:
        raise ValueError(f"Task {task} not supported. Must be one of ['cifar10', 'cifar100', 'mnist']")

def create_clustered_dataset(
    dataset: Dataset, num_clusters: int, cluster_type: str
) -> List[Dataset]:
    """
    Create a clustered dataset from the given dataset based on the partitioning strategy specified by cluster_type.

    Args:
        dataset (Dataset): The dataset to cluster
        num_clusters (int): The number of clusters to create
        cluster_type (str): The type of clustering to perform on the dataset

    Returns:
        List[Dataset]: A list of datasets, each representing a cluster
    """
    if cluster_type == "rotation":
        datasets = []
        for i in range(num_clusters):
            rotation = (i / num_clusters + 1) * 360
            transform = Transform(rotation, num_dims = dataset[0][0].shape[0])
            datasets.append(ClusterDataset(dataset=dataset, transform=transform))
    elif cluster_type == "selected_classes":
        datasets = []
        cluster_classes = [classes.tolist() for classes in np.array_split(list(range(len(dataset.classes))), num_clusters)]
        print(cluster_classes)
        datasets = {i: [] for i in range(num_clusters)}
        for i in range(len(dataset)):
            data, label = dataset[i]
            for j, classes in enumerate(cluster_classes):
                if label in classes:
                    datasets[j].append((data, label))
        datasets = [ClusterDataset(data) for data in datasets.values()]
    else:
        raise ValueError(f"Cluster type {cluster_type} not supported")
    return datasets

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_cifar10(32, 0.2)
    print("Train size: ", len(train_loader) * 32)
    print("Val size: ", len(val_loader) * 32)
    print("Test size: ", len(test_loader) * 32)