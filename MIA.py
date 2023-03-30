# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
import time

import numpy as np
# %%
import torch

# ourself libs
from torchvision import datasets, transforms

from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from membership_inference import train_attack_model, attack
from model.MNIST_CNN import MNIST_CNN
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler
"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments ():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 20
        self.N_client = 20
        self.data_name = 'mnist'  # purchase, cifar10, mnist, adult
        # Federated Unlearning Settings
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 1  # If want to forget, change None to the client index



def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments ()
    torch.manual_seed (42)
    # kwargs for data loader
    print (60 * '=')
    print ("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))

    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print (60 * '=')
    print ("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据
    init_global_model = MNIST_CNN()
    model_para = torch.load ('model.pt')
    init_global_model.load_state_dict(model_para)
    transform = transforms.Compose ([transforms.ToTensor (),
                                     transforms.Normalize ((0.1307,), (0.3081,))])
    #构建数据集和dataloader
    trainset = datasets.MNIST ('~/data/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST ('~/data/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=True)

    train_idx, valid_idx = torch.load ('train_valid_idx.pt')
    train_sampler = SubsetRandomSampler (train_idx)
    unlearn_sampler = SubsetRandomSampler (valid_idx)
    trainloader = torch.utils.data.DataLoader (trainset, batch_size=512, sampler=train_sampler,
                                               shuffle=False, drop_last=False, num_workers=2)
    unlearnloader = torch.utils.data.DataLoader (trainset, batch_size=512, sampler=unlearn_sampler,
                                               shuffle=False, drop_last=False, num_workers=2)
    client_loaders = [trainloader, unlearnloader]
    #载入存储好的中间参数
    # scheduler = torch.load ('scheduler.pt')
    nable_grad_u_list = torch.load ('nable_grad_u_list.pt')
    hvp_nable_grad_list = torch.load ('hvp_nable_grad_list.pt')
    # train_data_ids = torch.load ('train_data_ids.pt')
    last_model_param = [i for i in init_global_model.parameters()]


    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print (60 * '=')
    print ("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model (old_GM, client_loaders, test_loader, FL_params)

    print ("\nEpoch  = {}".format (T_epoch))
    print ("Attacking against FL Standard  ")
    target_model = old_GMs[T_epoch]
    (ACC_old, PRE_old) = attack (target_model, attack_model, client_loaders, test_loader, FL_params)

    if (FL_params.if_retrain == True):
        print ("Attacking against FL Retrain  ")
        target_model = retrain_GMs[T_epoch]
        (ACC_retrain, PRE_retrain) = attack (target_model, attack_model, client_loaders, test_loader, FL_params)

    print ("Attacking against FL Unlearn  ")
    target_model = unlearn_GMs[T_epoch]
    (ACC_unlearn, PRE_unlearn) = attack (target_model, attack_model, client_loaders, test_loader, FL_params)
    print ("Attacking against FL Unlearn  ")
    target_model = unlearn_GMs[T_epoch - 1]
    (ACC_unlearn, PRE_unlearn) = attack (target_model, attack_model, client_loaders, test_loader, FL_params)


if __name__ == '__main__':
    Federated_Unlearning ()