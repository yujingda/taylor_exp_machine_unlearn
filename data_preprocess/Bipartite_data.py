import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.model_selection import StratifiedShuffleSplit

dataset_name = 'mnist'
percent_extracted_list = [5,10,20,30,40,50,60,70]
# Load the mnist dataset
mnist_data = MNIST (root='../data', train=True, download=True, transform=ToTensor ())

for percent_extracted in percent_extracted_list:
    # Initialize the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit (n_splits=1, test_size=percent_extracted/100.0, random_state=42)

    # Split the data into 10% extracted data and 90% remaining data
    for train_index, test_index in sss.split (mnist_data.data, mnist_data.targets):
        extracted_data = Subset (mnist_data, test_index)
        remaining_data = Subset (mnist_data, train_index)
    # Calculate the number of each class of data and print
    num_classes = len (mnist_data.classes)
    remaining_data_classes = [0] * num_classes
    extracted_data_classes = [0] * num_classes

    for _, target in remaining_data:
        remaining_data_classes[target] += 1

    for _, target in extracted_data:
        extracted_data_classes[target] += 1

    print ("Number of each class in remaining data:")
    for i in range (num_classes):
        print (f"{mnist_data.classes[i]}: {remaining_data_classes[i]}")

    print ("\nNumber of each class in extracted data:")
    for i in range (num_classes):
        print (f"{mnist_data.classes[i]}: {extracted_data_classes[i]}")

    # Save the extracted and remaining data to files

    folder_path = f'../data/un_{dataset_name}/{percent_extracted}/'
    if not os.path.exists (folder_path):
        os.makedirs (folder_path)

    torch.save (remaining_data, folder_path + 'remaining_data.pt')
    torch.save (extracted_data, folder_path + 'extracted_data.pt')

