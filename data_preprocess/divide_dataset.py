
import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

# Load the MNIST dataset
mnist_train = datasets.FashionMNIST (root='./data', train=True, download=True)
x_train, y_train = mnist_train.data, mnist_train.targets

# Divide the dataset into 10 parts
num_parts = 10
num_classes = 10

# Create lists to store the divided data and labels
x_train_parts = [[] for _ in range (num_parts)]
y_train_parts = [[] for _ in range (num_parts)]

# Divide the dataset into 10 parts with equal number of each class in each part
class_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
for i in range(num_classes):
    np.random.shuffle(class_indices[i])

part_size = len(x_train) // num_parts
class_size = part_size // num_classes

for part in range(num_parts):
    for class_idx in range(num_classes):
        start_idx = part * class_size
        end_idx = start_idx + class_size
        indices = class_indices[class_idx][start_idx:end_idx]
        x_train_parts[part].extend(x_train[indices])
        y_train_parts[part].extend(y_train[indices])
    print("##########")
    print(len(x_train_parts[part]))
    # Shuffle the samples within each part
    combined = list(zip(x_train_parts[part], y_train_parts[part]))
    np.random.shuffle(combined)
    x_train_parts[part], y_train_parts[part] = zip(*combined)

for i in y_train_parts:
    print(sum(i))
# Check if the directory exists, if not, create it
if not os.path.exists ("./data/divide_fmnist"):
    os.makedirs ("./data/divide_fmnist")

# Convert lists to tensors and save them to separate files
for i in range (num_parts):
    x_train_part = torch.stack (x_train_parts[i])
    y_train_part = torch.tensor (y_train_parts[i])
    torch.save (x_train_part, f"./data/divide_fmnist/x_train_part_{i}.pt")
    torch.save (y_train_part, f"./data/divide_fmnist/y_train_part_{i}.pt")



# Load saved data and create a DataLoader for each data file
dataloaders = []
for i in range (num_parts):
    x_train_part = torch.load (f"./data/divide_fmnist/x_train_part_{i}.pt")
    y_train_part = torch.load (f"./data/divide_fmnist/y_train_part_{i}.pt")

    # Create a TensorDataset and DataLoader for each part
    dataset = TensorDataset (x_train_part, y_train_part)
    dataloader = DataLoader (dataset, batch_size=32, shuffle=True)
    dataloaders.append (dataloader)