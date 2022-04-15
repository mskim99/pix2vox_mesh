# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import gc
import torch
import torchvision.models


class Discriminator(torch.nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg

        # resolution 32 / Volume
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.3),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.3),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout2d(0.3),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout3d(0.3),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout3d(0.3),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout3d(0.3),
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # torch.nn.Dropout3d(0.3),
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Linear(4096, 1),
            torch.nn.Sigmoid(),
        )
        '''
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 512, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm3d(512)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1024, kernel_size=1, stride=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm3d(1024)
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv3d(1024, 1024, kernel_size=1, stride=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Sigmoid()
        )
        '''
    def forward(self, volume):

        features = volume.view(-1, 3, 32, 32)
        # print(features.size()) # torch.Size([1, 3, 32, 32])
        features = self.layer1(features)
        # print(features.size()) # torch.Size([1, 64, 30, 30])
        features = self.layer2(features)
        # print(features.size()) # torch.Size([1, 128, 28, 28])
        features = self.layer3(features)
        # print(features.size()) # torch.Size([1, 128, 14, 14])
        features = self.layer4(features)
        # print(features.size()) # torch.Size([1, 256, 12, 12])
        features = self.layer5(features)
        # print(features.size()) # torch.Size([1, 512, 6, 6])
        features = self.layer6(features)
        # print(features.size()) # torch.Size([1, 512, 4, 4])
        features = self.layer7(features)
        # print(features.size()) # torch.Size([1, 1024, 2, 2])
        features = features.view(-1, 4096)
        features = self.layer8(features)
        # print(features.size()) # torch.Size([1, 1])

        return features