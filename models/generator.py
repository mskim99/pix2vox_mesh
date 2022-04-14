# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import gc
import torch
import torchvision.models


class Generator(torch.nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg

        self.e_layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(65536, 8),
            torch.nn.Linear(8, 12288)  # (4096 // 4) * 64
        )

        self.gc_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.gc_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.Conv2d(16, 16, kernel_size=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
        )
        self.gc_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )


    def forward(self, images):

        images = images.permute(1, 0, 3, 2, 4, 5).contiguous()
        images = torch.split(images, 1, dim=0)

        features = images[0].squeeze(dim=0)
        # print(features.size()) # torch.Size([1, 3, 128, 128, 128])
        features = self.e_layer1(features)
        # print(features.size()) # torch.Size([1, 64, 64, 64, 64])
        features = self.e_layer2(features)
        # print(features.size()) # torch.Size([1, 128, 32, 32, 32])
        features = self.e_layer3(features)
        # print(features.size()) # torch.Size([1, 256, 16, 16, 16])
        features = self.e_layer4(features)
        # print(features.size()) # torch.Size([1, 512, 8, 8, 8])
        features = self.e_layer5(features)
        # print(features.size()) # torch.Size([1, 1024, 4, 4, 4])

        features = features.view([65536])
        features = self.fc_layer(features)
        features = features.view([3, 256, 4, 4])
        # print(features.size())  # torch.Size([3, 256, 4, 4])

        features = self.gc_layer1(features)
        # print(features.size())  # torch.Size([3, 64, 8, 8])
        features = self.gc_layer2(features)
        # print(features.size())  # torch.Size([3, 16, 16, 16])
        features = self.gc_layer3(features)
        # print(features.size())  # torch.Size([3, 1, 32, 32])
        gen_mesh = features.view([3, 1024])
        # print(gen_mesh.size())  # torch.Size([3, 1024])

        return gen_mesh
