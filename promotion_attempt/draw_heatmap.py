# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

base_path = '/public/home/jiangyz/dataset/3PWF/'
intra_dis = np.load(base_path + '3PWF-A_intra_dis.npy')
inter_dis = np.load(base_path + '3PWF-AB_inter_dis.npy')
pred_prob = np.loadtxt(base_path + 'pred.npy')

intra_map = np.where(intra_dis < 0, 100, intra_dis)
intra_map = np.where(intra_map > 8., -1, 1)
inter_map = np.where(inter_dis < 0, 100, inter_dis)
inter_map = np.where(inter_map > 20., -1, 1)

intra_pare = np.triu(pred_prob, 0) + np.tril(intra_map, -1)
inter_pare = np.triu(pred_prob, 1) + np.tril(inter_map, 0)
inter_paree = np.triu(pred_prob, 0) + np.tril(inter_map, -1)

sn.set()
ax = sn.heatmap(intra_pare, vmin=-1, vmax=1)
plt.savefig(base_path + 'pred_intra_map.jpg')

sn.set()
ax = sn.heatmap(inter_pare, vmin=-1, vmax=1)
plt.savefig(base_path + 'inter_pred_map.jpg')

sn.set()
ax = sn.heatmap(inter_paree, vmin=-1, vmax=1)
plt.savefig(base_path + 'pred_inter_map.jpg')
