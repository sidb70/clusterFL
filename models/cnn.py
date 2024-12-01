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


class CifarCNN(nn.Module):
    """
    A simple CNN for classifying CIFAR-10 images.
    """

    def __init__(self):
        super(CifarCNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
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
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.nn(x)
    