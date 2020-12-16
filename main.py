# -*- coding: utf-8 -*-
# Copyright © 2020 Varun Sivashankar, Nikil Roashan Selvam. All rights reserved.

"""
Source code for our work on Role of Local Linearity, Orthogonality and Double Descent in Catastrophic Overfitting.
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import progressbar

# Obtain datset
root_dir='/home'
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
testset= torch.utils.data.Subset(testset, range(1000))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

# Simple neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    # Weak regularizer for successive linear layers ( |W1W2| + !W2W3| + ... |W_n-1.W_n|
    def pairwise(self):
        tot_loss = 0
        reg = 1
        fc_layers=[]
        fc_layer_norms=[]
        for m in net.modules():
          if isinstance(m,nn.Linear):
            fc_layers.append(m)
            fc_layer_norms.append(torch.norm(m.weight))
        
        for i in range(len(fc_layers)-1):
          temp_loss= torch.norm(torch.matmul(fc_layers[i].weight.T, fc_layers[i+1].weight.T))
          temp_loss= temp_loss/(fc_layer_norms[i] *fc_layer_norms[i+1])
          tot_loss = tot_loss + reg*temp_loss
        return tot_loss

    # Orthogonality measure ||W^T.W-I||_F summed across all linear layers
    def orthogonal(self):
        reg = 1e-2
        orth_loss = torch.zeros(1).to(device)
        for m in net.modules():
          if isinstance(m,nn.Linear):
            sym = torch.mm(m.weight, m.weight.T)
            sym -= torch.eye(m.weight.shape[0]).to(device)
            orth_loss = orth_loss + reg * torch.norm(sym)
        return orth_loss
  
# Sample random vector uniformly in range [-eps,+eps]
def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).to(device)
    delta.uniform_(-eps, eps).to(device)
    delta.requires_grad = requires_grad
    return delta

def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms
    
# Test adversarial robustness
def get_adversarial_robustness(testloader, net, num_epochs, epsilon, reg):
  alpha=1/num_epochs
  total=0
  correct=0
  for i, data in enumerate(testloader, 0):
     # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      # zero the parameter gradients
      inputs.requires_grad=True
      adv_inputs=inputs
      adv_inputs.requires_grad=True

      # get adv inputs using pgd-10
      for epoch in range(num_epochs):
        outputs = net(adv_inputs)
        loss = criterion(outputs, labels)
        if reg == "pairwise":
            loss= loss + net.pairwise()
        if reg == "orthogonal":
            loss= loss + net.orthogonal()
        loss.backward(retain_graph=True)
        grad=torch.autograd.grad(loss, inputs)[0]
        adv_delta = alpha* epsilon* torch.sign(grad)
        adv_delta= torch.max(torch.min(adv_delta, torch.FloatTensor(1).fill_(epsilon).to(device)), torch.FloatTensor(1).fill_(-epsilon).to(device))
        adv_inputs = torch.max(torch.min(inputs + adv_delta, torch.FloatTensor(1).fill_(1).to(device)), torch.FloatTensor(1).fill_(0).to(device))

      # test with labels
      outputs = net(adv_inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return (100 * correct) / total

# Test standard accuracy
def get_standard_accuracy(testloader, net, reg): # reg unnecessary
  total=0
  correct=0
  for i, data in enumerate(testloader, 0):
     # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      # test with labels
      outputs = net(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return (100 * correct) / total
   
   
# Main Script
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epsilon=16/255
reg="orthogonal"
num_epochs=100
setting=1

setting_name={1:"Standard Training", 2:"Adversarial Training", 3:"Adversarial Training with GradAlign"}


if setting==1: #Std
    cosines=[]
    losses=[]
    adv_robustness=[]
    test_accuracy=[]
    orthos=[]
    print("epsilon: ", epsilon)
    print("regularization: ", reg)
    pb=progressbar.ProgressBar()
    for epoch in pb(range(1,num_epochs+1)): # loop over the dataset multiple times
        mini_cosines=[]
        mini_orthos=[]
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels= inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            inputs.requires_grad=True

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad=torch.autograd.grad(loss, inputs)[0]
            if epoch!=1:
              optimizer.step()
            
            # sample point in the epsilon ball
            optimizer.zero_grad()
            delta = get_uniform_delta(inputs.shape, eps=epsilon, requires_grad=True)
            new_inputs=inputs+delta
            outputs = net(new_inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad2= torch.autograd.grad(loss, new_inputs)[0]

            # calculate the local linearity measure
            grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
            grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
            grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
            grad1_normalized = grad1 / grad1_norms[:, None, None, None]
            grad2_normalized = grad2 / grad2_norms[:, None, None, None]
            cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
            mini_cosines.append(cos.mean().item())
            mini_orthos.append(net.orthogonal().item())
            running_loss += loss.item()
            
        cosines.append(np.mean(mini_cosines))
        orthos.append(np.mean(mini_orthos))
        losses.append(running_loss/len(trainset))
        adv_robustness.append(get_adversarial_robustness(testloader, net, 10, epsilon, reg))
        test_accuracy.append(get_standard_accuracy(testloader, net, reg))

    print('Finished Training')
    print("losses: ", losses)
    print("cosines: ",cosines)
    print("orthos: ",orthos)
    print("adversarial robustness: ", adv_robustness)
    print("standard accuracy: ", test_accuracy)

elif setting==2: # AT
    cosines=[]
    losses=[]
    adv_robustness=[]
    test_accuracy=[]
    orthos=[]
    print("epsilon: ", epsilon)
    print("regularization: ", reg)
    pb=progressbar.ProgressBar()
    for epoch in pb(range(1,num_epochs+1)): # loop over the dataset multiple times
        mini_cosines=[]
        mini_orthos=[]
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels= inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            inputs.requires_grad=True

            # get adv inputs using fgsm
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad=torch.autograd.grad(loss, inputs)[0]
            adv_delta = epsilon* torch.sign(grad)
            adv_inputs = torch.max(torch.min(inputs + adv_delta, torch.FloatTensor(1).fill_(1).to(device)), torch.FloatTensor(1).fill_(0).to(device))
            
            # sample other input points
            optimizer.zero_grad()
            delta = get_uniform_delta(inputs.shape, eps=epsilon, requires_grad=True)
            new_inputs=inputs+delta
            outputs = net(new_inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad2= torch.autograd.grad(loss, new_inputs)[0]

            # calculate local linearity measure
            grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
            grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
            grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
            grad1_normalized = grad1 / grad1_norms[:, None, None, None]
            grad2_normalized = grad2 / grad2_norms[:, None, None, None]
            cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
            mini_cosines.append(cos.mean().item())
            mini_orthos.append(net.orthogonal().item())

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad=torch.autograd.grad(loss, inputs)[0]
            if epoch!=1:
              optimizer.step()

            running_loss += loss.item()

        cosines.append(np.mean(mini_cosines))
        orthos.append(np.mean(mini_orthos))
        losses.append(running_loss/len(trainset))
        adv_robustness.append(get_adversarial_robustness(testloader, net, 10, epsilon, reg))
        test_accuracy.append(get_standard_accuracy(testloader, net, reg))
        
    print('Finished Training')
    print("losses: ", losses)
    print("cosines: ",cosines)
    print("orthos: ",orthos)
    print("adversarial robustness: ", adv_robustness)
    print("standard accuracy: ", test_accuracy)

elif setting==3:  # AT + GradAlign
    cosines=[]
    losses=[]
    adv_robustness=[]
    test_accuracy=[]
    orthos=[]
    print("epsilon: ", epsilon)
    print("regularization: ", reg+" + GradAlign")
    pb=progressbar.ProgressBar()
    for epoch in pb(range(1,num_epochs+1)): # loop over the dataset multiple times
        mini_cosines=[]
        mini_orthos=[]
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels= inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            inputs.requires_grad=True

            # get adv inputs using fgsm
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad=torch.autograd.grad(loss, inputs,create_graph=True)[0]
            adv_delta = epsilon* torch.sign(grad)
            adv_inputs = torch.max(torch.min(inputs + adv_delta, torch.FloatTensor(1).fill_(1).to(device)), torch.FloatTensor(1).fill_(0).to(device))
            
            # sample other input points
            optimizer.zero_grad()
            delta = get_uniform_delta(inputs.shape, eps=epsilon, requires_grad=True)
            new_inputs=inputs+delta
            outputs = net(new_inputs)
            loss = criterion(outputs, labels)
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad2= torch.autograd.grad(loss, new_inputs, create_graph=True)[0]

            # local linearity
            grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
            grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
            grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
            grad1_normalized = grad1 / grad1_norms[:, None, None, None]
            grad2_normalized = grad2 / grad2_norms[:, None, None, None]
            cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
            mini_cosines.append(cos.mean().item())
            mini_orthos.append(net.orthogonal().item())


            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels) + 0.2 * (1-cos.mean())
            if reg == "pairwise":
                loss= loss + net.pairwise()
            if reg == "orthogonal":
                loss= loss + net.orthogonal()
            loss.backward(retain_graph=True)
            grad=torch.autograd.grad(loss, inputs)[0]
            if epoch!=1:
              optimizer.step()

            running_loss += loss.item()

        cosines.append(np.mean(mini_cosines))
        orthos.append(np.mean(mini_orthos))
        losses.append(running_loss/len(trainset))
        adv_robustness.append(get_adversarial_robustness(testloader, net, 10, epsilon, reg))
        test_accuracy.append(get_standard_accuracy(testloader, net, reg))
        
    print('Finished Training')
    print("losses: ", losses)
    print("cosines: ",cosines)
    print("orthos: ",orthos)
    print("adversarial robustness: ", adv_robustness)
    print("standard accuracy: ", test_accuracy)
        
else:
  print("Check setting")
  
model_path = '/home/models/model.pth'
torch.save(net.state_dict(), model_path)
  
"""
Implementation Detail:
  We currently average over intermediate cosine values and ortho values as we iterate through the batch (due to computational resource countraints).
  Ideally, we would want to run through the entire dataset once again after each iteration to calculate the cosine and ortho values.
"""
