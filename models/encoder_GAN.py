# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import gc
import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # resolution 32 / Volume
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 32, kernel_size=3),
            torch.nn.BatchNorm3d(32),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3),
            torch.nn.BatchNorm3d(64),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=3),
            torch.nn.BatchNorm3d(128),
            torch.nn.ELU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3),
            torch.nn.BatchNorm3d(128),
            torch.nn.ELU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=3),
            torch.nn.BatchNorm3d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 256, kernel_size=3),
            torch.nn.BatchNorm3d(256),
            torch.nn.ELU(),
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=3),
            torch.nn.BatchNorm3d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 512, kernel_size=3),
            torch.nn.BatchNorm3d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.Softmax(dim=1),
        )

        # Don't update params in VGG16 & only use in 2D image processing
        '''
        for param in vgg16_bn.parameters():
            param.requires_grad = False
            '''

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 3, 2, 4, 5).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:

            # For 32 resolution / Volume
            features = img.squeeze(dim=0)
            # print(features.size()) # torch.Size([1, 3, 112, 112, 112])
            features = self.layer1(features)
            # print(features.size()) # torch.Size([1, 32, 110, 110, 110])
            features = self.layer2(features)
            # print(features.size()) # torch.Size([1, 64, 54, 54, 54])
            features = self.layer3(features)
            # print(features.size()) # torch.Size([1, 128, 52, 52, 52])
            features = self.layer4(features)
            # print(features.size()) # torch.Size([1, 128, 50, 50, 50])
            features = self.layer5(features)
            # print(features.size()) # torch.Size([1, 256, 24, 24, 24])
            features = self.layer6(features)
            # print(features.size()) # torch.Size([1, 256, 22, 22, 22])
            features = self.layer7(features)
            # print(features.size()) # torch.Size([1, 512, 10, 10, 10])
            features = self.layer8(features)
            # print(features.size()) # torch.Size([1, 512, 4, 4, 4])

            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8]) / torch.Size([batch_size, n_views, 512, 16, 16])
        return image_features
