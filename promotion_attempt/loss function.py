# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import torch
import torch.nn as nn


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    eps = 1e-7
    y_pred.clamp_(eps, 1-eps)

    ma_alpha = torch.where(y_true == 1, alpha, 1-alpha)
    pt = torch.where(y_true == 1, 1-y_pred, y_pred)

    loss = - torch.sum( ma_alpha * torch.pow(pt, gamma) * torch.log(1-pt) )
    return loss
