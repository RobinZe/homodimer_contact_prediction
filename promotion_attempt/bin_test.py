# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import csv
import pickle
import numpy as np
import torch
import torch.optim as optim
from model.ResNet_bin import model as test_model
from model.bin_focal_loss import Focal_Loss

# device = torch.device('cuda:0')
# training_model.load_state_dict(torch.load('/dl/jiangyz/model/accurate_train1/hC-5.pkl'))
acc_dict = {'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
                'G': 78.7, 'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
                'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
                'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7,
                'X': 168.2}


# data concatenating
def conc_data(ite):
    seq_list = list(ite['Sequence'])
    seq_len = len(seq_list)

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


def expect_contact(y_true, y_pred):
    true = torch.sum(y_true[:4], axis=0).numpy()
    dist_weight = np.array(range(1, 23, 2))
    dist_weight = np.reshape(dist_weight, (-1, 1, 1))
    # z_norm = np.sum(y_pred, axis=0)
    pred = np.sum(y_pred * dist_weight, axis=0)
    # pred = pred / z_norm
    avg_pred = (pred + pred.T) / 2.0

    seqlen = pred.shape[-1]
    ind = np.zeros((seqlen, seqlen, 2))
    for i in range(seqlen):
        for j in range(seqlen):
            ind[i, j, 0] = i
            ind[i, j, 1] = j
    pred_index = np.dstack((avg_pred, ind))
    M1s = np.ones_like(pred, dtype=np.int16)
    mask = np.triu(M1s, 0)

    res = pred_index[(mask > 0)]
    res_sorted = res[(res[:, 0]).argsort()]
    ''' print "#The top", top, " predictions:"
    print "Number  Residue1  Residue2  Predicted_Score"
    print "%-8d%-10d%-10d%-10.4f" % (i + 1, int(res_sorted[i, 1]) + 1, int(res_sorted[i, 2]) + 1, res_sorted[i, 0]) '''
    top_list = []
    for i in range(len(res_sorted)):
        ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
        # if avg_pred[ind1, ind2] > 8.:
        #     break
        ct = true[ind1, ind2]
        # assert ct in (0, 1)
        top_list.append(ct)

    ''' fn = 0
    while i < len(res_sorted):
        ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
        fn += true[ind1, ind2]
        i += 1 '''
    return top_list


def top_contact(y_true, y_pred):
    true = torch.sum(y_true[:4], axis=0).numpy()
    pred = np.sum(y_pred[:4], axis=0)
    avg_pred = (pred + pred.T) / 2.0

    seqlen = pred.shape[-1]
    ind = np.zeros((seqlen, seqlen, 2))
    for i in range(seqlen):
        for j in range(seqlen):
            ind[i, j, 0] = i
            ind[i, j, 1] = j
    pred_index = np.dstack((avg_pred, ind))
    M1s = np.ones_like(pred, dtype=np.int16)
    mask = np.triu(M1s, 0)

    res = pred_index[(mask > 0)]
    res_sorted = res[(-res[:, 0]).argsort()]
    ''' print "#The top", top, " predictions:"
    print "Number  Residue1  Residue2  Predicted_Score"
    print "%-8d%-10d%-10d%-10.4f" % (i + 1, int(res_sorted[i, 1]) + 1, int(res_sorted[i, 2]) + 1, res_sorted[i, 0]) '''
    top_list = []
    for i in range(len(res_sorted)):
        ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
        ct = true[ind1, ind2]
        top_list.append(ct)

    ''' fn = 0
    while i < len(res_sorted):
        ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
        fn += true[ind1, ind2]
        i += 1 '''
    return top_list


result_file = open('/dl/jiangyz/result/bin_test.csv', 'a')
writer = csv.writer(result_file)
writer.writerow(['evaluation', 'precision top 1', 'precision top 10', 'precision top 100', 'accuracy top 10',
                 'accuracy top 100', 'precision top L/20', 'precision top L/10', 'precision top L/5',
                 'precision top L/2'])

# test data
test_file = open('/dl/jiangyz/data/DeepHomo_data/test.pkl', 'rb')
test_data = pickle.load(test_file, encoding='iso-8859-1')
test_file.close()

# last_precision = -1
# training_model.to(device)
test_model.load_state_dict(torch.load('/dl/jiangyz/model/bin_expect1/hC-19.pkl'))
# loss_fn = Focal_Loss()
''' optimizer = optim.Adam(training_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.5)
print(device) '''


def resul(plist, seq_len):
    ind_20th, ind_10th, ind_5th, ind_2nd = int(seq_len / 20), int(seq_len / 10), int(seq_len / 5), int(seq_len / 2)
    while len(plist) < max(100, ind_2nd):
            plist.append(0)
    precision1 = plist[0]
    precision2 = sum(plist[:10]) / 10
    precision3 = sum(plist[:100]) / 100
    accuracy2 = max(plist[:10])
    accuracy3 = max(plist[:100])
    p20th = sum(plist[:ind_20th]) / ind_20th
    p10th = sum(plist[:ind_10th]) / ind_10th
    p5th = sum(plist[:ind_5th]) / ind_5th
    p2nd = sum(plist[:ind_2nd]) / ind_2nd
    return [precision1, precision2, precision3, accuracy2, accuracy3, p20th, p10th, p5th, p2nd]


def add_list(list1, list2):
    assert len(list2) == len(list1)
    for i in range(len(list1)):
        list1[i] = list1[i] + list2[i]
    return list1


def main():
    # training_model.eval()
    # plen = 0
    # precision_1, precision_10, precision_100, accuracy_10, accuracy_100 = 0, 0, 0, 0, 0
    test_num = 0
    # precision_10th, precision_5th, precision_2nd = 0, 0, 0
    # mean_precision, mean_recall = 0, 0
    rst1, rst2 = [0] * 9, [0] * 9
    for test_item in test_data:
        seq_len, input_x, input_y, val_dist = conc_data(test_item)
        input_x = np.reshape(input_x, (1,)+input_x.shape)
        input_y = np.reshape(input_y, (1,)+input_y.shape)
        input_x = torch.tensor(input_x, dtype=torch.float)
        input_y = torch.tensor(input_y, dtype=torch.float)

        with torch.no_grad():
            val_pred = test_model(input_x, input_y)
            # loss = loss_fn(val_true, val_pred)
            val_pred = torch.squeeze(val_pred).numpy()
            # val_pred = val_pred.to('cpu')
            val_true = dist2label(val_dist)
            precision_list1 = top_contact(val_true, val_pred)
            precision_list2 = expect_contact(val_true, val_pred)

        top_result = resul(precision_list1, seq_len)
        expect_result = resul(precision_list2, seq_len)
        test_num += 1
        rst1 = add_list(rst1, top_result)
        rst2 = add_list(rst2, expect_result)

        # print(precision_list[:100])

    # print('Precision:', round(mean_precision/test_num, 6))  # "\nRecall:", round(mean_recall/test_num, 6))
    for i in range(len(rst1)):
        rst1[i] = rst1[i] / test_num
        rst2[i] = rst2[i] / test_num
    writer.writerow(['top'] + rst1)
    writer.writerow(['expect'] + rst2)


main()
result_file.close()
