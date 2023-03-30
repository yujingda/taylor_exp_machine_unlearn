# Set up the data loaders
import copy
import datetime

import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from model.MNIST_CNN import MNIST_CNN
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler
from utils import intersection,list_difference
import argparse

#模型的极限在97~98之间
parser = argparse.ArgumentParser(description='Extract a percentage of the data with the same distribution as the original dataset')
parser.add_argument('--percentage', type=float, default=0.1, help='the percentage of data to extract')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--drop_last', type=bool, default=False, help='drop last')
parser.add_argument('--device', type=int, default=1, help='device of cuda')
parser.add_argument('--lr', type=float, default=0.01, help='lr of SGD')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')

args = parser.parse_args()


# Set a fixed random seed
torch.manual_seed(42)
# Define transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# Load MNIST dataset
#trainset和testset内数据的顺序是不会改变的
trainset = datasets.MNIST('~/data/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/data/MNIST_data/', download=True, train=False, transform=transform)

# Extract 10% of the data with the same distribution as the original dataset
train_size = len(trainset)
indices = list(range(train_size))
np.random.shuffle(indices)
split = int(np.floor(args.percentage * train_size))
train_idx, valid_idx = indices[split:], indices[:split]
np.random.shuffle (indices)
torch.save((train_idx, valid_idx), 'train_valid_idx1.pt')


# temp_list = []
# for i in train_sampler:
#    temp_list.append(i)

#https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset
# Use the original dataset to train the model
#每一个epoch都固定顺序是没有好果子吃的
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, drop_last=args.drop_last)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=2, drop_last=args.drop_last)

testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=True)

# Store the id of the data in each batch
# TODO: this code could be instead with batch sampler
train_data_ids = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)

# Define loss function and optimizer
#如果loss函数里边的模式设置成sum，那么计算出的loss和梯度就真的是总和
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
# Train the model
epochs = args.epochs
valid_loss_min = np.Inf


# Test the model
def test_model(model, testloader, criterion,device):
    model.eval()
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss/len(testloader.dataset)
    print('Test Accuracy: {:.2f}%'.format(100 * np.sum(class_correct) / np.sum(class_total)))

nable_grad_u_list = []
hvp_nable_grad_list = []
unlearn_data_id_each_batch = []
retain_data_id_each_batch = []
model_param_list = []

def calculate_grad_unlearn_data(total_grad, unlearn_data_id_list, model, device, optimizer, B, delta_B, now_lr, args):
    # with torch.no_grad():
    unlearn_sampler = SubsetSeqSampler (unlearn_data_id_list)
    unlearnloader = torch.utils.data.DataLoader (trainset, batch_size=args.batch_size, sampler=unlearn_sampler,
                                               shuffle=False, drop_last=False)
    model.train ()
    unlearn_criterion = torch.nn.CrossEntropyLoss (reduction='sum')
    for data, target in unlearnloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad ()
        output = model (data)
        loss = unlearn_criterion (output, target)
        loss.backward ()
        unlearn_grad=[param.grad.clone().detach() for param in model.parameters ()]
        with torch.no_grad():
            for i, layer_grad in enumerate(unlearn_grad):
                # print(now_lr)
                unlearn_grad[i] = (now_lr/(B-delta_B))*layer_grad
                unlearn_grad[i] -= (now_lr*delta_B/(B-delta_B))*total_grad[i]
    return unlearn_grad

def calculate_hessian(batch_model, retain_data_id_last_batch, nable_grad_list, now_lr):
    # # initialize Hessian matrix with zeros
    # hessian_matrix2 = [torch.zeros_like (param).cuda () for param in uk_model.parameters ()]
    retain_sampler = SubsetSeqSampler (retain_data_id_last_batch)
    retain_dataloader = torch.utils.data.DataLoader (trainset, batch_size=args.batch_size, sampler=retain_sampler,
                                                 shuffle=False, drop_last=False)
    model = copy.deepcopy(batch_model)
    model = model.cuda()
    device = torch.device("cuda")
    for data, target in retain_dataloader:
        # forward pass
        data = data.cuda()
        target = target.cuda()
        criteria = torch.nn.CrossEntropyLoss ()
        model.zero_grad ()
        output = model (data)
        loss = criteria (output, target)
        params = [p for p in model.parameters ()]
        nable_grad_list = hvp_2t(loss, params, nable_grad_list, now_lr)
        # backward pass to calculate gradients
    model.zero_grad ()
    return nable_grad_list

def hvp_2t(y,w,v, lr):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""
    if len (w) != len (v[0]):
        raise (ValueError ("w and v must have the same length."))

    # First backprop
    first_grads = grad (y, w, retain_graph=True, create_graph=True)
    # Second backprop
    # return_grads = grad (elemwise_products, w, create_graph=True, retain_graph=True)
    for i, v_i in enumerate(v):
        for j, v_ele in enumerate (v_i):
            v[i][j] = v_ele.cuda ()
        if i < len(v) - 1:
            return_grads = grad (first_grads, w, grad_outputs=v_i, retain_graph=True)
        else:
            return_grads = grad (first_grads, w, grad_outputs=v_i)
        for j, v_ele in enumerate (v[i]):
            v[i][j] = (v_i[j] - lr*return_grads[j]).cpu()
    return v


for epoch in range(epochs):
    np.random.shuffle (indices)
    # Store the id of these data into the map
    train_sampler = SubsetSeqSampler (indices)
    trainloader = torch.utils.data.DataLoader (trainset, batch_size=args.batch_size, sampler=train_sampler,
                                               shuffle=False, drop_last=args.drop_last, num_workers=2)
    train_loss = 0.0
    model.train()
    correct = 0
    total = 0
    batch_idx = 0
    for data, target in trainloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        gradients = [param.grad.clone().detach() for param in model.parameters ()]
        model_param_list.append([param.data.clone().detach().cpu() for param in model.parameters ()])
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        train_data_ids.append (
            train_sampler.indices[batch_idx * trainloader.batch_size:(batch_idx + 1) * trainloader.batch_size])
        unlearn_data_id_each_batch.append(intersection(train_data_ids[-1], valid_idx))
        retain_data_id_each_batch.append(list_difference(train_data_ids[-1], unlearn_data_id_each_batch[-1]))
        delta_B = len (retain_data_id_each_batch[-1])
        B = len(train_data_ids[-1])
        now_lr = scheduler.get_lr()[-1]
        grad_u = calculate_grad_unlearn_data(gradients, unlearn_data_id_each_batch[-1], model,
                                             device, optimizer, B, delta_B, now_lr, args)
        nable_grad_u_list.append(grad_u)
        if len(nable_grad_u_list)>1:
            hvp_nable_grad_list = calculate_hessian(model, retain_data_id_each_batch[-1], hvp_nable_grad_list, now_lr)
        hvp_nable_grad_list.append(copy.deepcopy(grad_u))
        batch_idx+=1
    train_loss = train_loss/len(trainloader.sampler)
    test_model(model, testloader, criterion, device)
    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(
        epoch+1,
        train_loss,
        100 * correct / total))
    scheduler.step()
    print (datetime.datetime.now ().strftime ("%Y-%m-%d %H:%M:%S"))
    torch.save (model.state_dict (), 'model1.pt')
    torch.save(model_param_list,'model_param_list.pt')
    torch.save (scheduler, 'scheduler1.pt')
    torch.save (nable_grad_u_list, 'nable_grad_u_list1.pt')
    torch.save (hvp_nable_grad_list, 'hvp_nable_grad_list1.pt')
    torch.save (train_data_ids, 'train_data_ids1.pt')





