import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from autoencoder import *
from decoder import *
from encoder import *
from loading_bar import *
from data.dataloader_mnist import *
from data.download_mnist import *
from alexnet import * 
from data.dataloader_cifar10 import *

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
alexnet = AlexNet()
model = AbstractAutoEncoder(encoder=Encoder(alexnet), decoder=Decoder()).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss or Binary Cross Entropy loss 
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.BCELoss(reduction='mean')

loading_bar = LoadingBar(length=20)

# Load MNIST dataset
# train_images, train_labels, test_images, test_labels = download_mnist(url_root, file_dict)

# train_dataset = MNISTCustomDataset(train_images, train_labels,transform=data_transform)
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load CIFAR10 dataset
# CIFAR10 dataset 
train_dataloader, valid_dataloader = get_train_valid_loader(data_dir = './data',                                      batch_size = 64,
                       augment = False,                             		     random_seed = 1)

test_dataloader = get_test_loader(data_dir = './data',
                              batch_size = 64)

# Training data
num_epochs = 2
len_dataset = len(train_dataloader)

model.train()
for epoch in range(num_epochs): 
    for idx, B in enumerate(train_dataloader):
        # load data to the actice device
        inputs, targets = (b.to(device) for b in B)

        # compute reconstruction data 
        recon = model(inputs)
        
        # compute training loss 
        loss = criterion(recon, inputs)
        
        # clear gradient and backpro
        model.zero_grad()
        loss.backward()
        
        # optimizer update 
        optimizer.step()
        
        # Print process
        print(
            f"\r┃{epoch+1:12d}/{num_epochs:2d} ┃{loss:12.4f}   {loading_bar(idx/len_dataset)}",
            end="",
            flush=True,
        )
        