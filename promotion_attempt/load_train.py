# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import numpy as np
import math
import torch
from model.ResNet_V2 import model as training_model
import torch.optim as optim
training_model.eval()


# read pair feature data
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


def focal_loss(y_true, y_pred, alpha=0.75, gamma=2):
    eps = 1e-7
    y_pred = y_pred.clamp(eps, 1-eps)
    alpha = torch.tensor([alpha])
    alpha = alpha.to(device)

    ma_alpha = torch.where(y_true == 1, alpha, 1-alpha)   # alpha or 1-alpha ?
    pt = torch.where(y_true == 1, 1-y_pred, y_pred)

    loss = - torch.sum( ma_alpha * torch.pow(pt, gamma) * torch.log(1-pt) )
    return loss


def top_index(m, n):
    sl = m.shape[-1]
    val_sort = m.reshape(-1, 1).squeeze().tolist()
    ind = []
    for i in range(n):
        inde = val_sort.index(max(val_sort))
        ind.append([int(inde/sl), inde % sl])
        val_sort[inde] = -1
    return ind


device = torch.device('cpu')
training_model.to(device)
optimizer = optim.Adam(training_model.parameters(), lr=1e-3, weight_decay=1e-4)

f_train = open('/dl/jiangyz/data/C2train.lst', 'r')
pro_train = f_train.readlines()
data_x, data_y, data_z = [], [], []
for pro_name in pro_train:
    pro_name = pro_name.rstrip("\n")
    pro_name = pro_name.rstrip("\r")
    load_x = np.loadtxt('/dl/jiangyz/data/huang_C2_ok/%s/%s_f1d.npy' % (pro_name, pro_name))
    data_x.append(np.reshape(load_x, (1,)+load_x.shape))
    data_y.append(load_pair('/dl/jiangyz/data/huang_C2_ok/%s/%s' % (pro_name, pro_name)))
    data_z.append(np.loadtxt('/dl/jiangyz/data/huang_C2_ok/%s/%s_inter_pred.npy' % (pro_name, pro_name)))
f_train.close()

f_valid = open('/dl/jiangyz/data/C2valid.lst', 'r')
pro_valid = f_valid.readlines()
data_a, data_b, data_c, data_n = [], [], [], []
for pro_name in pro_valid:
    pro_name = pro_name.rstrip("\n")
    pro_name = pro_name.rstrip("\r")
    data_n.append(pro_name)
    load_a = np.loadtxt('/dl/jiangyz/data/huang_C2_ok/%s/%s_f1d.npy' % (pro_name, pro_name))
    data_a.append(np.reshape(load_a, (1,)+load_a.shape))
    data_b.append(load_pair('/dl/jiangyz/data/huang_C2_ok/%s/%s' % (pro_name, pro_name)))
    data_c.append(np.loadtxt('/dl/jiangyz/data/huang_C2_ok/%s/%s_inter_pred.npy' % (pro_name, pro_name)))
f_valid.close()

for epoch in range(20):
    total_loss = []
    for n in range(len(data_x)):
        input_x, input_y = data_x[n], data_y[n]
        val_true = data_z[n]
        seq_len = input_x.shape[-1]
        if seq_len > 400:
            reduced = np.random.randint(0, seq_len-400)
            input_x = input_x[:, :, reduced:(reduced+400)]
            input_y = input_y[:, :, reduced:(reduced+400), reduced:(reduced+400)]
            val_true = val_true[reduced:(reduced+400), reduced:(reduced+400)]
        input_x = torch.tensor(input_x, dtype=torch.float).to(device)
        input_y = torch.tensor(input_y, dtype=torch.float).to(device)
        val_true = torch.tensor(val_true).to(device)
        # input_x, input_y, val_true = input_x.to(device), input_y.to(device), val_true.to(device)
        val_pred = training_model(input_x, input_y)
        val_pred = torch.squeeze(val_pred)
        loss = focal_loss(val_true, val_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # total_loss.append(loss.to('cpu'))
        total_loss.append(loss)
    print("Training Epoch %2d :      Loss  %8f\n" % (epoch+1, sum(total_loss)/len(total_loss)))
    torch.save(training_model.state_dict(), '/dl/jiangyz/model/hC2-%s.pkl' % (epoch+1))
    acc_1, acc_2, acc_3 = 0, 0, 0
    sur_2, sur_3, n_total = 0, 0, 0
    for n in range(len(data_a)):
        p_name = data_n[n]
        input_x, input_y = data_a[n], data_b[n]
        input_x = torch.tensor(input_x, dtype=torch.float).to(device)
        input_y = torch.tensor(input_y, dtype=torch.float).to(device)
        val_true = data_c[n]
        # input_x, input_y = input_x.to(device), input_y.to(device)
        val_pred = training_model(input_x, input_y)
        val_pred = torch.squeeze(val_pred)
        # val_pred = val_pred.to('cpu')
        top_ind = top_index(val_pred, 100)

        for ind in top_ind[:10]:
            acc_10 += val_true[ind[0], ind[1]]
        for ind in top_ind[10:]:
            acc_90 += val_true[ind[0], ind[1]]
        acc10 = acc_10/10
        acc100 = (acc_10 + acc_90)/100
        print('    ' + p_name + " accuracy %8f  %8f  %8f\n" % (acc1, acc10, acc100))
        acc_1 += acc1
        acc_2 += acc10
        acc_3 += acc100
        sur_2 += math.ceil(acc10)
        sur_3 += math.ceil(acc100)
        n_total += 1
    print('  Acc1     Acc2     Acc100         Suc_rate1     Suc_rate10     Suc_rate100')
    print(' ', acc_1/n_total, acc_2/n_total, acc_3/n_total, acc_1/n_total, sur_2/n_total, sur_3/n_total)

# torch.save( training_model.state_dict(), '/dl/jiangyz/model/150_model.pkl' )
