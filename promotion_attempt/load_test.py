# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import numpy as np
import torch
# import pickle as pkl
# import csv

from model.Res2Net import model as test_model
test_model.load_state_dict(torch.load('/dl/jiangyz/model/res2net_3/r2n-8.pkl'))


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


def top_contact(true, pred):
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
    print "%-8d%-10d%-10d%-10.4f" % (i + 1, int(res_sorted[i, 1]) + 1, int(res_sorted[i, 2]) + 1, res_sorted[i, 0])
    '''
    top_list = []
    for i in range(len(res_sorted)):
        if i <= max(seqlen, 100):
            ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
            ct = max(true[ind1, ind2], true[ind2, ind1])
            top_list.append(ct)
        else:
            break
    return top_list


''' prec_1, prec_10, prec_100, acc_10, acc_100 = 0, 0, 0, 0, 0
prec_10th, prec_5th, prec_2nd, prec_1st = 0, 0, 0, 0

f_test = open('/dl/jiangyz/data/my_h2.lst', 'r')
pro_test = f_test.readlines()
data_a, data_b, data_c, data_n = [], [], [], []
pik_data = []
f_result = open('/dl/jiangyz/result/my_h2_test.csv', 'w')
writer = csv.writer(f_result)
writer.writerow(['Pro name', 'precision top1', 'precision top10', 'precision top100', 'accuracy top10',
                 'accuracy top100', 'precision top L/10', 'precision top L/5', 'precision top L/2', 'precision top L'])
'''

# for pro_name in pro_test:
def test_fc(pro_name):
    pro_name = pro_name.rstrip("\n")
    pro_name = pro_name.rstrip("\r")
    # data_n.append(pro_name)
    load_a = np.loadtxt('/dl/jiangyz/data/%s/%s_f1d.npy' % (pro_name, pro_name))
    load_b = load_pair('/dl/jiangyz/data/%s/%s' % (pro_name, pro_name))
    # load_c = np.loadtxt('/dl/jiangyz/data/3PWF/%s/%s_inter_pred.npy' % (pro_name, pro_name))
    seq_len = load_b.shape[-1]
    # pik_data.append({'name': pro_name, 'single_feature': load_a, 'pair_feature': load_b, 'label': load_c})

    with torch.no_grad():
        input_x = np.reshape(load_a, (1,) + load_a.shape)
        input_x = torch.tensor(input_x, dtype=torch.float)
        input_y = torch.tensor(load_b, dtype=torch.float)
        val_pred = test_model(input_x, input_y)
        val_pred = torch.squeeze(val_pred).numpy()
        # res_list = top_contact(load_c, val_pred)
    return val_pred
    ''' l_10th = int(seq_len / 10)
    l_5th = int(seq_len / 5)
    l_2nd = int(seq_len / 2)

    prec_1 = res_list[0].item()
    prec_10 = sum(res_list[:10]).item() / 10
    prec_100 = sum(res_list[:100]).item() / 100
    acc_10 = max(res_list[:10]).item()
    acc_100 = max(res_list[:100]).item()
    prec_10th = sum(res_list[:l_10th]).item() / l_10th
    prec_5th = sum(res_list[:l_5th]).item() / l_5th
    prec_2nd = sum(res_list[:l_2nd]).item() / l_2nd
    prec_1st = sum(res_list[:seq_len]).item() / seq_len

    writer.writerow([pro_name, prec_1, prec_10, prec_100, acc_10, acc_100, prec_10th, prec_5th, prec_2nd, prec_1st])


f_test.close()
f_result.close()
fil = open('/dl/jiangyz/data/Deep_data/my_h2_test.pkl', 'wb')
pkl.dump(pik_data, fil)
fil.close() '''


pred_mat = test_fc('3PWF')
np.savetxt('/dl/jiangyz/data/3PWF/pred.npy', pred_mat, fmt='%f')


''' total_loss = []
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
print('  Acc1     Acc2     Acc100         Suc_rate1     Suc_rate10     Suc_rate100')
print(' ', acc_1/n_total, acc_2/n_total, acc_3/n_total, acc_1/n_total, sur_2/n_total, sur_3/n_total) '''
