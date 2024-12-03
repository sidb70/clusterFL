"""
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Cifar10CNN(nn.Module):
    """
    A simple CNN for classifying CIFAR-10 images.
    """

    def __init__(self):
        super(Cifar10CNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.nn(x)
class Cifar100CNN(nn.Module):
    """
    A simple CNN for classifying CIFAR-100 images.
    """

    def __init__(self, out_dim=10):
        super(Cifar100CNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.nn(x)

class MnistCNN(nn.Module):
    """
    A simple CNN for classifying MNIST images.
    """

    def __init__(self):
        super(MnistCNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(1,8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*5*5, 64),
            nn.ReLU(),
            nn.Linear(64, 10),

        )
        self.apply(init_weights)

    def forward(self, x):
        return self.nn(x)
    