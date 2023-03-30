import torch
import torch.nn as nn


class MNIST_CNN (nn.Module):
    def __init__(self):
        super (MNIST_CNN, self).__init__ ()
        self.conv1 = nn.Conv2d (in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU ()
        self.pool1 = nn.MaxPool2d (kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d (in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU ()
        self.pool2 = nn.MaxPool2d (kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d (in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU ()
        self.pool3 = nn.MaxPool2d (kernel_size=2, stride=2)
        self.fc1 = nn.Linear (in_features=576, out_features=500)
        nn.init.xavier_uniform_ (self.fc1.weight)
        self.relu4 = nn.ReLU ()
        self.fc2 = nn.Linear (in_features=500, out_features=10)
        nn.init.xavier_uniform_ (self.fc2.weight)

    def forward(self, x):
        x = self.conv1 (x)
        x = self.relu1 (x)
        x = self.pool1 (x)
        x = self.conv2 (x)
        x = self.relu2 (x)
        x = self.pool2 (x)
        x = self.conv3 (x)
        x = self.relu3 (x)
        x = self.pool3 (x)
        x = torch.flatten(x, 1)
        x = self.fc1 (x)
        x = self.relu4 (x)
        x = self.fc2 (x)
        return x