import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor

dataset_name = 'mnist'
percent_extracted = 10
folder_path = f'../data/un_{dataset_name}/{percent_extracted}/'
# Read the stored data
remaining_data = torch.load(folder_path + 'remaining_data.pt')
extracted_data = torch.load(folder_path + 'extracted_data.pt')

# Implement a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))])

# Create dataset instances with the stored data and apply the transform
remaining_dataset = CustomDataset(remaining_data, transform=transform)
extracted_dataset = CustomDataset(extracted_data, transform=transform)

# Implement dataloaders for the datasets
remaining_dataloader = DataLoader(remaining_dataset, batch_size=64, shuffle=True)
extracted_dataloader = DataLoader(extracted_dataset, batch_size=64, shuffle=True)





# # Test the dataloaders
# for data, labels in remaining_dataloader:
#     # Check that the shape of the data tensor is correct
#     assert data.shape == torch.Size([64, 1, 28, 28])
#     # Check that the shape of the labels tensor is correct
#     assert labels.shape == torch.Size([64])
#
# for data, labels in extracted_dataloader:
#     # Check that the shape of the data tensor is correct
#     assert data.shape == torch.Size([64, 1, 28, 28])
#     # Check that the shape of the labels tensor is correct
#     assert labels.shape == torch.Size([64])