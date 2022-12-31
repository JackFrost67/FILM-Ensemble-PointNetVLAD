# PointNetVLAD with FILM-Ensemble 

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.netvlad import NetVLADLoupe
from models.layers.stn3d import STN3D
from models.layers.CBN import CBN2D

# PointNet feature extractor
class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, n_ensemble=1, cbn_gain=0.0, is_cbn_trainable=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3D(num_points=num_points, k=3, use_bn=False, n_ensemble=n_ensemble, is_cbn_trainable=is_cbn_trainable, cbn_gain=cbn_gain)
        self.feature_trans = STN3D(num_points=num_points, k=64, use_bn=False, n_ensemble=n_ensemble, is_cbn_trainable=is_cbn_trainable, cbn_gain=cbn_gain)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.cbn1 = CBN2D(n_ensemble=n_ensemble, num_features=64, name='cbn1', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
        self.cbn2 = CBN2D(n_ensemble=n_ensemble, num_features=64, name='cbn2', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
        self.cbn3 = CBN2D(n_ensemble=n_ensemble, num_features=64, name='cbn3', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
        self.cbn4 = CBN2D(n_ensemble=n_ensemble, num_features=128, name='cbn4', trainable=is_cbn_trainable, cbn_gain=cbn_gain)
        self.cbn5 = CBN2D(n_ensemble=n_ensemble, num_features=1024, name='cbn5', trainable=is_cbn_trainable, cbn_gain=cbn_gain)          
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = F.relu(self.cbn1(self.conv1(x)))
        x = F.relu(self.cbn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.cbn3(self.conv3(x)))
        x = F.relu(self.cbn4(self.conv4(x)))
        x = self.cbn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans

# PointNetVLAD model
class PointNetVLAD(nn.Module):
    def __init__(self, num_points=2500, n_ensemble=1, cbn_gain=0.0, global_feat=True, is_cbn_trainable=True, 
                 feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVLAD, self).__init__()
        self.n_ensemble = n_ensemble
        self.pointnet = PointNetfeat(num_points=num_points, global_feat=global_feat, n_ensemble=n_ensemble, 
                                 cbn_gain=cbn_gain, is_cbn_trainable=is_cbn_trainable, 
                                 feature_transform=feature_transform, max_pool=max_pool)
        self.netvlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                 output_dim=output_dim, gating=True, add_batch_norm=True, is_training=True)

    def forward(self, x):
        B = x.shape[0]
        x = x.repeat_interleave(self.n_ensemble, dim=0)
        x = self.pointnet(x)
        x = self.netvlad(x)
        x = x.view(B, self.n_ensemble, -1)
        return x

def PointNetVLAD_FILM(num_points=4096, n_ensemble=1, cbn_gain=0.0, global_feat=True, is_cbn_trainable=True, 
                     feature_transform=False, max_pool=True, output_dim=256):
    return PointNetVLAD(num_points=num_points, n_ensemble=n_ensemble, cbn_gain=cbn_gain, global_feat=global_feat, 
                        is_cbn_trainable=is_cbn_trainable, feature_transform=feature_transform, max_pool=max_pool, 
                        output_dim=output_dim)