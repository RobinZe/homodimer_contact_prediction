# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import torch
import torch.nn as nn
import torch.nn.functional as F


def n_list(n):
    num_list = []
    for i in range(n):
        num_list.append(3 * min(4*i+3, 4*(n-i)-3))
    return torch.tensor(num_list)


class PreActBlock1d(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock1d, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=17, stride=stride, padding=8, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=17, stride=1, padding=8, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.elu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else out
        out = self.conv1(out)
        out = self.conv2(F.elu(self.bn2(out)))
        out += shortcut
        return out


class PreActBlock2d(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock2d, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.elu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else out
        out = self.conv1(out)
        out = self.conv2(F.elu(self.bn2(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block1, block2, layers1, layers2):
        super(ResNet, self).__init__()
        self.in_planes_1 = 31
        self.in_planes_2 = 10
        self.bn_mid = nn.BatchNorm1d(60) ###
        self.bn_end = nn.BatchNorm2d(32)
        self.conv_mid = nn.Conv1d(60, 2, kernel_size=17, stride=1, padding=8, bias=False) ###
        self.conv_end = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer_1(block1, layers1[0],  stride=1)
        self.layer2 = self._make_layer_1(block1, layers1[1],  stride=1)
        self.layer3 = self._make_layer_1(block1, layers1[2],  stride=1)
        self.layer4 = self._make_layer_1(block1, layers1[3],  stride=1)
        self.layer5 = self._make_layer_1(block1, layers1[4],  stride=1)
        self.layer6 = self._make_layer_1(block1, layers1[5],  stride=1)

        self.layers1 = self._make_layer_2(block2, layers2[0], 4, stride=1)
        self.layers2 = self._make_layer_2(block2, layers2[1], 4, stride=1)
        self.layers3 = self._make_layer_2(block2, layers2[2], 4, stride=1)
        self.layers4 = self._make_layer_2(block2, layers2[3], 4, stride=1)
        self.layers5 = self._make_layer_2(block2, layers2[4], 4, stride=1)
        self.layers6 = self._make_layer_2(block2, layers2[5], 4, stride=1)
        self.layers7 = self._make_layer_2(block2, layers2[6], 4, stride=1)
        self.layers8 = self._make_layer_2(block2, layers2[7], 4, stride=1)
        self.layers9 = self._make_layer_2(block2, layers2[8], 4, stride=1)

    def _make_layer_1(self, block, planes, stride):
        hid_layer = block(self.in_planes_1, planes, stride)
        self.in_planes_1 = planes
        return nn.Sequential(hid_layer)

    def _make_layer_2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        hid_layers = []
        for stride in strides:
            hid_layers.append(block(self.in_planes_2, planes, stride))
            self.in_planes_2 = planes
        return nn.Sequential(*hid_layers)

    def forward(self, x, y):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        out1 = self.conv_mid(F.elu(self.bn_mid(x)))
        sl = out1.size(-1)
        o0 = out1/(n_list(sl).to(out1.device))
        out1 = out1.unsqueeze(3)
        o1 = out1.expand(-1, -1, -1, sl)/(3 * sl)
        o2 = o1.transpose(3, 2)
        o3 = o0.unsqueeze(3).expand(-1, -1, -1, 2).reshape(1, 2, -1).unfold(2, x.size()[-1], 1)[:, :, :-1]
        y = torch.cat([o1, o2, o3, y], dim=1)
        y = self.layers1(y)
        y = self.layers2(y)
        y = self.layers3(y)
        y = self.layers4(y)
        y = self.layers5(y)
        y = self.layers6(y)
        y = self.layers7(y)
        y = self.layers8(y)
        y = self.layers9(y)

        out2 = torch.sigmoid(self.conv_end(F.elu(self.bn_end(y))))
        return out2


# model = ResNet(PreActBlock1d, PreActBlock2d, [35, 40, 45, 50], [32, 48, 64, 48, 32])
model = ResNet(PreActBlock1d, PreActBlock2d, [35, 40, 45, 50, 55, 60], [32, 32, 48, 64, 64, 64, 48, 32, 32])
# model.eval() !!!

x = torch.rand(1, 31, 30)
y = torch.rand(1, 4, 30, 30)
z = model(x, y)
print(z.squeeze())
