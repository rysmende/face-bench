# -*- coding: utf-8 -*-
# @Author: Alfred Xiang Wu
# @Date:   2022-02-09 14:45:31
# @Breif: 
# @Last Modified by:   Alfred Xiang Wu
# @Last Modified time: 2022-02-09 14:48:34

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_models.lightcnn_modules import ResBlock, MFM

class LightCNN(nn.Module):
    def __init__(self, block=ResBlock, layers=[1, 2, 3, 4]):
        super(LightCNN, self).__init__()

        self.conv1 = MFM(3, 48, 3, 1, 1)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.conv2  = MFM(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.conv3  = MFM(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.conv4  = MFM(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.conv5  = MFM(128, 128, 3, 1, 1)

        self.fc = nn.Linear(8*8*128, 256)
        nn.init.normal_(self.fc.weight, std=0.001)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.conv4(x)
        x = self.block4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        fc = self.fc(x)

        return fc
