# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import csv
import pickle
import numpy as np
import torch
from model.Res2Net import model as test_model

# test_model.load_state_dict(torch.load('/dl/jiangyz/model/res2net_3/r2n-8.pkl'))
# device = 'cpu'
# test_model.to(device)


# data concatenating
def conc_data(ite):
    seq_list = list(ite['Sequence'])
    seq_len = len(seq_list)

    hydro_data = np.squeeze(ite['Hydro'])
    hydro_data = np.reshape(hydro_data, (1,)+hydro_data.shape)
    pssm_data = ite['PSSM'].transpose()
    dssp_data = ite['DSSP'].transpose()
    acc_data = np.squeeze(ite['ACC'])
    acc_data = np.reshape(acc_data, (1,)+acc_data.shape)
    rsa_data = np.squeeze(ite['RSA'])
    rsa_data = np.reshape(rsa_data, (1,)+rsa_data.shape)

    di_data = ite['DI']
    apc_data = ite['APC']
    dismap_data = ite['Dismap']
    dockmap_data = ite['Dockmap']

    label_data = ite['Distance4label']
    label_data = np.where(label_data > 8., 0, label_data)
    label_data = np.where(label_data == 0, 0, 1)

    assert label_data.shape[0] == seq_len == hydro_data.shape[-1] == pssm_data.shape[-1] \
           == dssp_data.shape[-1] == acc_data.shape[-1] == rsa_data.shape[-1] \
           == di_data.shape[0] == apc_data.shape[0] == dockmap_data.shape[0] == dismap_data.shape[0]

    seq_data = np.concatenate((pssm_data, dssp_data, acc_data, rsa_data, hydro_data), axis=0)
    pair_data = np.array([di_data, apc_data, dismap_data, dockmap_data])
    return seq_len, seq_data, pair_data, label_data


# evaluating functions
''' def focal_loss(y_true, y_pred, alpha=0.75, gamma=2):
    eps = 1e-7
    y_pred = y_pred.clamp(eps, 1-eps)
    alpha = torch.tensor([alpha])
    # alpha = alpha.to(device)

    ma_alpha = torch.where(y_true == 1, alpha, 1-alpha)   # alpha or 1-alpha ?
    pt = torch.where(y_true == 1, 1-y_pred, y_pred)

    loss = - torch.sum( ma_alpha * torch.pow(pt, gamma) * torch.log(1-pt) )
    return loss '''


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
        ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
        ct = max(true[ind1, ind2], true[ind2, ind1])
        top_list.append(ct)
    return top_list


test_file = open('/dl/jiangyz/data/DeepHomo_data/valid.pkl', 'rb')
test_data = pickle.load(test_file, encoding='iso-8859-1')
test_file.close()
# result_list = []


def test_process(test_model):
    precision_1, precision_10, precision_100, accuracy_10, accuracy_100 = 0, 0, 0, 0, 0
    precision_10th, precision_5th, precision_2nd = 0, 0, 0
    # recall_1, recall_10, recall_100 = 0, 0, 0
    # recall_10th, recall_5th, recall_2nd = 0, 0, 0
    test_num = 0

    for test_item in test_data:
        seq_len: int
        seq_len, input_x, input_y, val_true = conc_data(test_item)
        input_x = np.reshape(input_x, (1,)+input_x.shape)
        input_y = np.reshape(input_y, (1,)+input_y.shape)
        input_x = torch.tensor(input_x, dtype=torch.float)
        input_y = torch.tensor(input_y, dtype=torch.float)
        val_true = torch.tensor(val_true, dtype=torch.float)

        with torch.no_grad():
            # val_pred = torch.zeros_like(val_true)
            # for itm in mod_list:
            #   test_model.load_state_dict(torch.load('/dl/jiangyz/model/accurate_train3-%s/hC-%s.pkl' % (itm[0], itm[1]
            val_pred = test_model(input_x, input_y)
            '''    val_pred += torch.squeeze(val__pred)
                except RuntimeError:
                    reduced = np.random.randint(0, seq_len - 400)
                    input_x = input_x[:, :, reduced:(reduced + 400)]
                    input_y = input_y[:, :, reduced:(reduced + 400), reduced:(reduced + 400)]
                    val_true = val_true[reduced:(reduced + 400), reduced:(reduced + 400)]
                    val_pred = test_model(input_x, input_y) '''
            val_pred = torch.squeeze(val_pred).numpy()  # / len(mod_list)  # .to('cpu')
            # val_pred = val_pred.to('cpu')
            precision_list = top_contact(val_true, val_pred)

        precision1 = precision_list[0]
        precision2 = sum(precision_list[:10])/10
        precision3 = sum(precision_list[:100])/100
        # accuracy2 = max(precision_list[:10])
        # accuracy3 = max(precision_list[:100])
        ind_10th, ind_5th, ind_2nd = int(seq_len/10), int(seq_len/5), int(seq_len/2)
        # sum_list = sum(precision_list)
        p10th = sum(precision_list[:ind_10th])
        p5th = sum(precision_list[:ind_5th])
        p2nd = sum(precision_list[:ind_2nd])
        # p1st = sum(precision_list[:seq_len]) / seq_len
        ''' r1 = precision_list[0] / sum_list
        r10 = sum(precision_list[:10]) / sum_list
        r100 = sum(precision_list[:100]) / sum_list
        r10th = p10th / sum_list
        r5th = p5th / sum_list
        r2nd = p2nd / sum_list '''
        precision10th = p10th / ind_10th
        precision5th = p5th / ind_5th
        precision2nd = p2nd / ind_2nd

        precision_1 += precision1.item()
        precision_10 += precision2.item()
        precision_100 += precision3.item()
        # accuracy_10 += accuracy2.item()
        # accuracy_100 += accuracy3.item()
        precision_10th += precision10th.item()
        precision_5th += precision5th.item()
        precision_2nd += precision2nd.item()
        # precision_1st += p1st.item()
        ''' recall_1 += r1.item()
        recall_10 += r10.item()
        recall_100 += r100.item()
        recall_10th += r10th.item()
        recall_5th += r5th.item()
        recall_2nd += r2nd.item() '''
        test_num += 1
    return precision_1/test_num, precision_10/test_num, precision_100/test_num, precision_10th/test_num, \
           precision_5th/test_num, precision_2nd/test_num
    ''' , recall_1/test_num, recall_10/test_num, recall_100/test_num, \
           recall_10th/test_num, recall_5th/test_num, recall_2nd/test_num
    return [round(precision_1/test_num, 6), round(precision_10/test_num, 6), round(precision_100/test_num, 6),
            round(precision_10th/test_num, 6), round(precision_5th/test_num, 6), round(precision_2nd/test_num, 6),
            round(recall_1/test_num, 6), round(recall_10/test_num, 6), round(recall_100/test_num, 6),
            round(recall_10th/test_num, 6), round(recall_5th/test_num, 6), round(recall_2nd/test_num, 6)] '''


for i in range(5, 13):
    test_model.load_state_dict(torch.load('/dl/jiangyz/model/res2net_3/r2n-%s.pkl' % i))
    final_list = test_process(test_model)
    ''' result_file = open('/dl/jiangyz/result/final_test.csv', 'a')
    writer = csv.writer(result_file)
    writer.writerow(['precision top1', 'precision top10', 'precision top100','accuracy top10', 'accuracy top100',
                     'precision top L/10', 'precision top L/5', 'precision top L/2', 'precision top L'])
    for lists in result_list:
        writer.writerow(lists) '''
    print('item', "precision\t\tr2n-%s(3)" % i)
    print('top 1       %6f' % final_list[0])
    print('top 10      %6f' % final_list[1])
    print('top 100     %6f' % final_list[2])
    print('top L/10    %6f' % final_list[3])
    print('top L/5     %6f' % final_list[4])
    print('top L/2     %6f' % final_list[5])
    # result_file.close()
