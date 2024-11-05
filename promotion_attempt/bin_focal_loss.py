# !/usr/bin/python3
#-*- coding : utf-8 -*-
__author__ = 'Robin'

import torch
import torch.nn as nn


class Focal_Loss(nn.Module):
    def __init__(self, cal_dev='cpu'):
        super(Focal_Loss, self).__init__()
        self.ma_alpha = torch.tensor(0.75).to(cal_dev)

    def forward(self, y_true, y_pred, gamma=2):
        eps = 1e-7
        y_pred = y_pred.clamp(eps, 1 - eps)

        label_cont = torch.sum(y_true[:4], dim=(0,))
        alpha = torch.where(label_cont == 1, self.ma_alpha, 1 - self.ma_alpha)  # alpha or 1-alpha ?
        pt = torch.where(y_true == 1, 1 - y_pred, y_pred)

        loss = - torch.sum(alpha * torch.pow(pt, gamma) * torch.log(1 - pt), dim=(1, 2))
        return torch.mean(loss)
