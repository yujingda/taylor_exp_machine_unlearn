import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tqdm
# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    gradients = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        gradients.append([p.grad.clone() for p in model.parameters()])
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return gradients

# Define the test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
# Set up the data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=1000, shuffle=True)

# Set up the device and the model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Set up the optimizer and the scheduler
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
from torch.autograd import Variable

def compute_hessian(model, data, target, device):
    model.zero_grad()
    output = model(data)
    loss = nn.functional.nll_loss(output, target)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    hessian = []
    for g in grads:
        grad_params = torch.flatten(g)
        hessian_row = []
        for g_param in tqdm.tqdm(grad_params):
            hessian_row.append(torch.autograd.grad(g_param, model.parameters(), retain_graph=True))
        hessian.append(hessian_row)
    return hessian

for epoch in range(1, 11):
    gradients = train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        hessian_matrix = compute_hessian(model, data, target, device)
    optimizer.step()