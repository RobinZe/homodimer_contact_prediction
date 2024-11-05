# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper


def n_list(n):
    num_list = []
    for i in range(n):
        num_list.append(3 * min(4*i+3, 4*(n-i)-3))
    return torch.tensor(num_list)


class Bottle2neck1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck1d, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottle2neck2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck2d, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Res2Net(nn.Module):
    def __init__(self, block1d, block2d, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes1d = 31
        self.inplanes2d = 10
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.bn_mid = nn.BatchNorm1d(60)  ###
        self.bn_end = nn.BatchNorm2d(32)
        self.conv_mid = nn.Conv1d(60, 2, kernel_size=17, stride=1, padding=8, bias=False)  ###
        self.conv_end = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer01 = self._make_layer1d(block1d, 35)
        self.layer02 = self._make_layer1d(block1d, 40)
        self.layer03 = self._make_layer1d(block1d, 45)
        self.layer04 = self._make_layer1d(block1d, 50)
        self.layer05 = self._make_layer1d(block1d, 55)
        self.layer06 = self._make_layer1d(block1d, 60)

        self.layer1 = self._make_layer2d(block2d, 32, 4)
        self.layer2 = self._make_layer2d(block2d, 32, 4)
        self.layer3 = self._make_layer2d(block2d, 48, 4)
        self.layer4 = self._make_layer2d(block2d, 64, 4)
        self.layer5 = self._make_layer2d(block2d, 64, 4)
        self.layer6 = self._make_layer2d(block2d, 64, 4)
        self.layer7 = self._make_layer2d(block2d, 48, 4)
        self.layer8 = self._make_layer2d(block2d, 32, 4)
        self.layer9 = self._make_layer2d(block2d, 32, 4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block2d.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer1d(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes1d != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes1d, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        hid_layer = block(self.inplanes1d, planes, stride, downsample)
        self.inplanes1d = planes * block.expansion
        return nn.Sequential(hid_layer)

    def _make_layer2d(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes2d != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes2d, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        hid_layers = []
        hid_layers.append(block(self.inplanes2d, planes, stride, downsample=downsample,
                                stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes2d = planes * block.expansion
        for i in range(1, num_blocks):
            hid_layers.append(block(self.inplanes2d, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*hid_layers)

    @torchsnooper.snoop()
    def forward(self, x, y):
        x = self.layer01(x)
        x = self.layer02(x)
        x = self.layer03(x)
        x = self.layer04(x)
        x = self.layer05(x)
        x = self.layer06(x)
        out1 = self.conv_mid(F.elu(self.bn_mid(x)))
        sl = out1.size(-1)
        o0 = out1/(n_list(sl).to(out1.device))
        out1 = out1.unsqueeze(3)
        o1 = out1.expand(-1, -1, -1, sl)/(3 * sl)
        o2 = o1.transpose(3, 2)
        o3 = o0.unsqueeze(3).expand(-1, -1, -1, 2).reshape(1, 2, -1).unfold(2, x.size()[-1], 1)[:, :, :-1]
        y = torch.cat([o1, o2, o3, y], dim=1)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.layer6(y)
        y = self.layer7(y)
        y = self.layer8(y)
        y = self.layer9(y)

        out2 = torch.sigmoid(self.conv_end(F.elu(self.bn_end(y))))
        return out2

# model = ResNet(PreActBlock1d, PreActBlock2d, [35, 40, 45, 50], [32, 48, 64, 48, 32])
model = Res2Net(Bottle2neck1d, Bottle2neck2d)
# model.eval() !!!

x = torch.rand(1, 31, 30)
y = torch.rand(1, 4, 30, 30)
z = model(x, y)
print(z.squeeze().size())
