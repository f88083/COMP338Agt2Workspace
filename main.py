import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()

        # First hidden layer
        self.conv1 = nn.Conv2d(kernel_size=[7, 7], stride=2, padding=3,in_channels=3, out_channels=64)

        # Second hidden layer
        self.conv2 = nn.MaxPool2d(kernel_size=[3, 3], stride=2, padding=0)

        # Third hidden layer
        self.conv3 = nn.Conv2d(kernel_size=[3, 3], stride=1, padding=1, in_channels=3, out_channels=64)

        # Fourth hidden layer
        self.conv4 = nn.MaxPool2d(kernel_size=[3 , 3], stride=2, padding=0)

        # Fifth hidden layer
        self.fc1 = nn.Linear(in_features=5, out_features=5)


my_cnn = CNN()
print(my_cnn)
