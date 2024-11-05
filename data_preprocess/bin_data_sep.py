# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import csv
import pickle
import numpy as np
import torch
from model.ResNet_bin import model as test_model
test_model.load_state_dict(torch.load('/dl/jiangyz/model/bin_expect1/hC-19.pkl'))


def bin_count(m):
    n = np.inner(*m.shape)
    n0 = np.count_nonzero(m)
    m = np.where(m < 2.0001, 0, m)
    n1 = np.count_nonzero(m)
    m = np.where(m < 4.0001, 0, m)
    n2 = np.count_nonzero(m)
    m = np.where(m < 6.0001, 0, m)
    n3 = np.count_nonzero(m)
    m = np.where(m < 8.0001, 0, m)
    n4 = np.count_nonzero(m)
    m = np.where(m < 10.0001, 0, m)
    n5 = np.count_nonzero(m)
    m = np.where(m < 12.0001, 0, m)
    n6 = np.count_nonzero(m)
    m = np.where(m < 14.0001, 0, m)
    n7 = np.count_nonzero(m)
    m = np.where(m < 16.0001, 0, m)
    n8 = np.count_nonzero(m)
    m = np.where(m < 18.0001, 0, m)
    n9 = np.count_nonzero(m)
    m = np.where(m < 20.0001, 0, m)
    n10 = np.count_nonzero(m)
    return [n0 - n1, n1 - n2, n2 - n3, n3 - n4, n4 - n5, n5 - n6, n6 - n7, n7 - n8, n8 - n9, n9 - n10, n10, n - n0, n]


def conc_data(ite):
    seq_list = list(ite['Sequence'])
    seq_len = len(seq_list)

    acc_dict = {'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
                'G': 78.7, 'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
                'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
                'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7,
                'X': 168.2}
    for i in range(seq_len):
        if seq_list[i] not in acc_dict:
            seq_list[i] = 'X'

    hydro_data = np.squeeze(ite['Hydro'])
    hydro_data = np.reshape(hydro_data, (1,)+hydro_data.shape)
    pssm_data = ite['PSSM'].transpose()
    dssp_data = ite['DSSP'].transpose()
    acc_data = np.squeeze(ite['ACC'])
    rsa_data = np.zeros((seq_len,))
    for i in range(seq_len):
        rsa_data[i] = acc_data[i] / acc_dict[seq_list[i]]
    acc_data = np.reshape(acc_data, (1,)+acc_data.shape)
    rsa_data = np.reshape(rsa_data, (1,)+rsa_data.shape)

    di_data = ite['DI']
    apc_data = ite['APC']
    dismap_data = ite['Dismap']
    dockmap_data = ite['Dockmap']

    dist_data = ite['Distance4label']
    dist_data = np.where(dist_data == 0., dist_data.T, dist_data)
    dist_data = (dist_data + dist_data.T) / 2

    assert dist_data.shape[-1] == seq_len == hydro_data.shape[-1] == pssm_data.shape[-1] \
           == dssp_data.shape[-1] == acc_data.shape[-1] == rsa_data.shape[-1] \
           == di_data.shape[0] == apc_data.shape[0] == dockmap_data.shape[0] == dismap_data.shape[0]

    seq_data = np.concatenate((pssm_data, dssp_data, acc_data, rsa_data, hydro_data), axis=0)
    pair_data = np.array([di_data, apc_data, dismap_data, dockmap_data])
    return seq_len, seq_data, pair_data, dist_data


def dist2label(dist_data):
    dist_data = np.where(dist_data == 0, 30., dist_data)
    dist_data = torch.tensor(dist_data // 2, dtype=torch.float)
    dist_data = torch.where(dist_data > 10, torch.tensor(10.), dist_data)
    label_data = torch.zeros(11, *dist_data.shape)
    label_data.scatter_(0, torch.unsqueeze(dist_data.long(), 0), 1)
    return label_data


''' fil = open('expect_count.csv', 'w')
writer = csv.writer(fil)
writer.writerow(['name', 'bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10',
                 'zero', 'total']) '''
fie = open('sum_count.csv', 'w')
writor = csv.writer(fie)
writor.writerow(['name', 'bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10',
                 'total'])


def process(ite):
    seq_len, input_x, input_y, val_dist = conc_data(ite)
    input_x = np.reshape(input_x, (1,) + input_x.shape)
    input_y = np.reshape(input_y, (1,) + input_y.shape)
    input_x = torch.tensor(input_x, dtype=torch.float)
    input_y = torch.tensor(input_y, dtype=torch.float)

    with torch.no_grad():
        val_pred = test_model(input_x, input_y)
        val_pred = torch.squeeze(val_pred).numpy()
        avg_pred = (val_pred + val_pred.transpose(0, 2, 1)) / 2
        max_pred = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                max_pred[i, j] = np.argmax(avg_pred[:, i, j])
        ''' dist_weight = np.array(range(1, 23, 2))
        dist_weight = np.reshape(dist_weight, (-1, 1, 1))
        pred = np.sum(val_pred * dist_weight, axis=0)
        avg_pred = (pred + pred.T) / 2.0
        sum_pred = np.sum(val_pred, axis=(1, 2))
    writer.writerow([ite['Name']] + bin_count(avg_pred)) '''
    writor.writerow([ite['Name']] + max_pred.tolist() + [np.sum(max_pred)])


training_file = open('/dl/jiangyz/data/DeepHomo_data/train.pkl', 'rb')
train_data = pickle.load(training_file, encoding='iso-8859-1')
# writer.writerow(['training set'])
writor.writerow(['training set'])
for item in train_data:
    process(item)
training_file.close()

valid_file = open('/dl/jiangyz/data/DeepHomo_data/valid.pkl', 'rb')
valid_data = pickle.load(valid_file, encoding='iso-8859-1')
# writer.writerow(['validation set'])
writor.writerow(['validation set'])
for item in valid_data:
    process(item)
valid_file.close()

test_file = open('/dl/jiangyz/data/DeepHomo_data/test.pkl', 'rb')
test_data = pickle.load(test_file, encoding='iso-8859-1')
# writer.writerow(['test set'])
writor.writerow(['test set'])
for item in test_data:
    process(item)
test_file.close()


# fil.close()
fie.close()
