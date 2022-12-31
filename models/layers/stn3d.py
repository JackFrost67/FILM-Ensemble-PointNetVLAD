# Spatial Transformer Network in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models.layers.CBN import CBN2D, CBN1D


# I don't think conditional batch norm is necessary for the STN
class STN3D(nn.Module):
    def __init__(self, num_points=4096, n_ensemble=1, k=3, use_bn=True, is_cbn_trainable=True, cbn_gain=1.0):
        super(STN3D, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.cbn1 = CBN2D(n_ensemble=n_ensemble, num_features=64, name='cbn1', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
            self.cbn2 = CBN2D(n_ensemble=n_ensemble, num_features=128, name='cbn2', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
            self.cbn3 = CBN2D(n_ensemble=n_ensemble, num_features=1024, name='cbn3', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
            self.cbn4 = CBN1D(n_ensemble=n_ensemble, num_features=512, name='cbn4', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
            self.cbn5 = CBN1D(n_ensemble=n_ensemble, num_features=256, name='cbn5', trainable=is_cbn_trainable, cbn_gain=cbn_gain) 
            #self.bn1 = nn.BatchNorm2d(64)
            #self.bn2 = nn.BatchNorm2d(128)
            #self.bn3 = nn.BatchNorm2d(1024)
            #self.bn4 = nn.BatchNorm1d(512)
            #self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.cbn1(self.conv1(x)))
            x = F.relu(self.cbn2(self.conv2(x)))
            x = F.relu(self.cbn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.cbn4(self.fc1(x)))
            x = F.relu(self.cbn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x