# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import numpy as np
import torch
import math

from model.ResNet_V2 import model as test_model


def load_pair(prefix):
    # read the DCA DI scores
    di_data = np.loadtxt(prefix + "_di.mat")
    di_data = (di_data + di_data.T) * .5
    di_data = np.reshape(di_data, (1,) + di_data.shape)
    # read the DCA APC scores
    apc_data = np.loadtxt(prefix + "_apc.mat")
    apc_data = (apc_data + apc_data.T) * .5
    apc_data = np.reshape(apc_data, (1,) + apc_data.shape)
    # read the distance map
    dismap_data = np.loadtxt(prefix + "_dismap.mat")
    dismap_data = np.reshape(dismap_data, (1,) + dismap_data.shape)
    # read the docking map
    dockmap_data = np.loadtxt(prefix + "_dockmap.mat")
    dockmap_data = np.reshape(dockmap_data, (1,) + dockmap_data.shape)
    # check the shape
    assert di_data.shape[-1] == apc_data.shape[-1] == dockmap_data.shape[-1] == dismap_data.shape[-1]

    pair_data = np.concatenate((di_data, apc_data, dismap_data, dockmap_data), axis=0)
    # add the batch dimension
    pair_data = np.reshape(pair_data, ((1,) + pair_data.shape))
    return pair_data


def top_index(m, n):
    sl = m.shape[-1]
    val_sort = m.reshape(-1, 1).squeeze().tolist()
    ind = []
    for i in range(n):
        inde = val_sort.index(max(val_sort))
        ind.append([int(inde/sl), inde % sl])
        val_sort[inde] = -1
    return ind


f_test = open('/dl/jiangyz/data/C2valid.lst', 'r')
pro_test = f_test.readlines()
data_a, data_b, data_c, data_n = [], [], [], []
for pro_name in pro_test:
    pro_name = pro_name.rstrip("\n")
    pro_name = pro_name.rstrip("\r")
    data_n.append(pro_name)
    load_a = np.loadtxt('/dl/jiangyz/data/huang_C2_ok/%s/%s_f1d.npy' % (pro_name, pro_name))
    data_a.append(np.reshape(load_a, (1,)+load_a.shape))
    data_b.append(load_pair('/dl/jiangyz/data/huang_C2_ok/%s/%s' % (pro_name, pro_name)))
    data_c.append(np.loadtxt('/dl/jiangyz/data/huang_C2_ok/%s/%s_inter_pred.npy' % (pro_name, pro_name)))
f_test.close()


for epoch in range(10, 21):
    test_model.load_state_dict(torch.load('/dl/jiangyz/model/hC2-%s.pkl' % epoch))
    total_loss = []
    acc_1, acc_2, acc_3 = 0, 0, 0
    sur_2, sur_3, n_total = 0, 0, 0
    for n in range(len(data_a)):
        p_name = data_n[n]
        input_x, input_y = data_a[n], data_b[n]
        input_x = torch.tensor(input_x, dtype=torch.float)
        input_y = torch.tensor(input_y, dtype=torch.float)
        val_true = torch.tensor(data_c[n])
        val_pred = test_model(input_x, input_y)
        val_pred = torch.squeeze(val_pred)
        # val_pred = val_pred.to('cpu')
        top_ind = top_index(val_pred, 100)

        acc1, acc_10, acc_90 = val_true[top_ind[0][0], top_ind[0][1]], 0, 0
        for ind in top_ind[:10]:
            acc_10 += val_true[ind[0], ind[1]]
        for ind in top_ind[10:]:
            acc_90 += val_true[ind[0], ind[1]]
        acc10 = acc_10/10
        acc100 = (acc_10 + acc_90)/100
        # print('    ' + p_name + " accuracy %8f  %8f  %8f\n" % (acc1, acc10, acc100))
        acc_1 += acc1
        acc_2 += acc10
        acc_3 += acc100
        sur_2 += math.ceil(acc10)
        sur_3 += math.ceil(acc100)
        n_total += 1
    # print('  Acc1     Acc2     Acc100         Suc_rate1     Suc_rate10     Suc_rate100')
    print(epoch)
    print(' ', acc_1/n_total, acc_2/n_total, acc_3/n_total, acc_1/n_total, sur_2/n_total, sur_3/n_total)
