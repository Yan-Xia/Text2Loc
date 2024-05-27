"""PyG implementation of PointNet++ adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.nn as gnn
import torch_geometric.transforms as T
from models.language_encoder import get_mlp

import numpy as np
from easydict import EasyDict


class SetAbstractionLayer(nn.Module):
    def __init__(self, ratio, radius, mlp):
        super(SetAbstractionLayer, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.point_conv = gnn.PointConv(local_nn=mlp)

    def forward(self, x, pos, batch):
        subset_indices = gnn.fps(pos, batch, self.ratio)

        sparse_indices, dense_indices = gnn.radius(
            pos, pos[subset_indices], self.radius, batch_x=batch, batch_y=batch[subset_indices]
        )
        edge_index = torch.stack(
            (dense_indices, sparse_indices), dim=0
        )  # TODO/CARE: Indices are propagated internally? Care edge direction: a->b <=> a is in N(b)

        x = self.point_conv(x, (pos, pos[subset_indices]), edge_index)

        return x, pos[subset_indices], batch[subset_indices]


class GlobalAbstractionLayer(nn.Module):
    def __init__(self, mlp):
        super(GlobalAbstractionLayer, self).__init__()
        self.mlp = mlp

    def forward(self, x, pos, batch):
        x = torch.cat((x, pos), dim=1)
        x = self.mlp(x)
        x = gnn.global_max_pool(x, batch)
        return x


class PointNet2(nn.Module):
    def __init__(self, num_classes, num_colors, args):
        super(PointNet2, self).__init__()
        assert args.pointnet_layers == 3 and args.pointnet_variation == 0

        self.sa1 = SetAbstractionLayer(0.5, 0.2, get_mlp([3 + 3, 32, 64]))
        self.sa2 = SetAbstractionLayer(0.5, 0.3, get_mlp([64 + 3, 128, 128]))
        self.sa3 = SetAbstractionLayer(0.5, 0.4, get_mlp([128 + 3, 256, 256]))
        self.ga = GlobalAbstractionLayer(get_mlp([256 + 3, 512, 1024]))

        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.class_classifier = nn.Linear(256, num_classes)
        self.color_classifier = nn.Linear(256, num_colors)

        self.dim0 = 1024
        self.dim1 = 512
        self.dim2 = 256

        # Slightly better but larger:
        # self.sa1 = SetAbstractionLayer(0.5, 0.2, get_mlp([3 + 3, 32, 64], add_batchnorm=True))
        # self.sa2 = SetAbstractionLayer(0.5, 0.3, get_mlp([64 + 3, 128, 256], add_batchnorm=True))
        # self.sa3 = SetAbstractionLayer(0.5, 0.4, get_mlp([256 + 3, 512, 512], add_batchnorm=True))
        # self.ga = GlobalAbstractionLayer(get_mlp([512 + 3, 1024, 2048], add_batchnorm=True))
        # self.lin1 = nn.Linear(2048, 1024)
        # self.lin2 = nn.Linear(1024, 512)
        # self.lin3 = nn.Linear(512, num_classes)

    def forward(self, data):
        data.to(self.device)

        x, pos, batch = self.sa1(data.x, data.pos, data.batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)
        features0 = self.ga(x, pos, batch)

        # Dropout did not seem helpful
        features1 = F.relu(self.lin1(features0))
        features2 = F.relu(self.lin2(features1))
        class_pred = self.class_classifier(features2)
        color_pred = self.color_classifier(features2)

        return EasyDict(
            features0=features0,
            features1=features1,
            features2=features2,
            class_pred=class_pred,
            color_pred=color_pred,
        )

    @property
    def device(self):
        return next(self.lin1.parameters()).device


if __name__ == "__main__":
    transform = T.Compose([T.NormalizeScale(), T.RandomRotate(180, axis=2)])
    pos = torch.rand(10, 3)
    print(pos)
    print(transform(EasyDict(pos=pos, num_nodes=10)).pos)

    quit()

    x = torch.rand(10, 3)
    pos = torch.rand(10, 3)
    batch = torch.zeros(10, dtype=torch.long)

    model = PointNet2(10, EasyDict(pointnet_layers=3, pointnet_variation=0))

    out = model(EasyDict(x=x, pos=pos, batch=batch))
