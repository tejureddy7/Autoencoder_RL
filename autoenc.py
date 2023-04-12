import numpy as np
import torch
import cv2#Final Autoencoder implementation. Loss=0.010

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2


import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

#from skimage.transform import resize

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 32, 2, padding=1)
        self.fc1 = nn.Linear(32*20*15, 132)
        self.fc2 = nn.Linear(132, 32*20*15)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(32, 128, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(32, 1, 4, stride=2)



    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def decode(self,y):
        x = self.fc2(y)
        x = x.view(x.size(0), 32, 20, 15)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.sigmoid(self.t_conv4(x))
        return x

    def forward(self, x):
        ## encode ##
        x = self.encode(x)
        x = self.decode(x)
        return x
