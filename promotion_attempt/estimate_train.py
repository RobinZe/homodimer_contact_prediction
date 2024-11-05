# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import csv
import pickle
import numpy as np
import torch
import torch.optim as optim
from model.Res2Net import model as training_model

device = torch.device('cuda:2')
# training_model.load_state_dict(torch.load('/dl/jiangyz/model/estimate1/hC2-18.pkl'))


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
def focal_loss(y_true, y_pred, alpha=0.75, gamma=2, device=device):
    eps = 1e-7
    y_pred = y_pred.clamp(eps, 1-eps)
    alpha = torch.tensor([alpha])
    alpha = alpha.to(device)

    ma_alpha = torch.where(y_true == 1, alpha, 1-alpha)   # alpha or 1-alpha ?
    pt = torch.where(y_true == 1, 1-y_pred, y_pred)

    loss = - torch.sum( ma_alpha * torch.pow(pt, gamma) * torch.log(1-pt) )
    return loss


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
    top_list = []
    for i in range(min(seqlen, 100)):
        ind1, ind2 = int(res_sorted[i, 1]), int(res_sorted[i, 2])
        ct = max(true[ind1, ind2], true[ind2, ind1])
        top_list.append(ct)
    return top_list


result_file = open('/dl/jiangyz/result/res2net.csv', 'w')
writer = csv.writer(result_file)
writer.writerow(['epoch', 'precision top 1', 'precision top 10', 'precision top 100',
                 'accuracy top 1', 'accuracy top 10', 'accuracy top 100'])
# training process
training_file = open('/dl/jiangyz/data/DeepHomo_data/train.pkl', 'rb')
training_data = pickle.load(training_file, encoding='iso-8859-1')
valid_file = open('/dl/jiangyz/data/DeepHomo_data/valid.pkl', 'rb')
valid_data = pickle.load(valid_file, encoding='iso-8859-1')
training_file.close()
valid_file.close()

# last_precision = -1
training_model.to(device)
optimizer = optim.Adam(training_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-6)
print(device)

for epoch in range(5):
    training_model.train()
    training_model.to(device)
    training_num, training_loss = 0, 0
    # optimizer = optim.Adam(training_model.parameters(), lr=lrate, weight_decay=1e-4)
    for training_item in training_data:
        seq_len, input_x, input_y, val_true = conc_data(training_item)
        if seq_len > 400:
            reduced = np.random.randint(0, seq_len-400)
            input_x = input_x[:, reduced:(reduced+400)]
            input_y = input_y[:, reduced:(reduced+400), reduced:(reduced+400)]
            val_true = val_true[reduced:(reduced+400), reduced:(reduced+400)]
        input_x = np.reshape(input_x, (1,)+input_x.shape)
        input_y = np.reshape(input_y, (1,)+input_y.shape)

        input_x = torch.tensor(input_x, dtype=torch.float).to(device)
        input_y = torch.tensor(input_y, dtype=torch.float).to(device)
        val_true = torch.tensor(val_true).to(device)

        val_pred = training_model(input_x, input_y)
        val_pred = torch.squeeze(val_pred)
        loss = focal_loss(val_true, val_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_num += 1
        training_loss += loss.to('cpu')
    print("Training Epoch %2d    Loss  %8f" % (epoch+1, training_loss/training_num))
    torch.save(training_model.state_dict(), '/dl/jiangyz/model/r2n-%s.pkl' % (epoch+1))

    training_model.to('cpu')
    training_model.eval()
    precision_1, precision_10, precision_100, accuracy_10, accuracy_100 = 0, 0, 0, 0, 0
    valid_num, valid_loss = 0, 0
    for valid_item in valid_data:
        seq_len, input_x, input_y, val_true = conc_data(valid_item)
        input_x = np.reshape(input_x, (1,)+input_x.shape)
        input_y = np.reshape(input_y, (1,)+input_y.shape)
        input_x = torch.tensor(input_x, dtype=torch.float)
        input_y = torch.tensor(input_y, dtype=torch.float)

        with torch.no_grad():
            try:
                val_pred = training_model(input_x, input_y)
            except RuntimeError:
                reduced = np.random.randint(0, seq_len - 400)
                input_x = input_x[:, :, reduced:(reduced + 400)]
                input_y = input_y[:, :, reduced:(reduced + 400), reduced:(reduced + 400)]
                val_true = val_true[reduced:(reduced + 400), reduced:(reduced + 400)]
                val_pred = training_model(input_x, input_y)
            val_pred = torch.squeeze(val_pred)
            loss = focal_loss(torch.tensor(val_true), val_pred, device='cpu')
            # val_pred = val_pred.to('cpu')
            precision_list = top_contact(val_true, val_pred.numpy())

        precision1 = precision_list[0]
        precision2 = sum(precision_list[:10])/10
        precision3 = sum(precision_list)/100
        accuracy2 = max(precision_list[:10])
        accuracy3 = max(precision_list)

        precision_1 += precision1
        precision_10 += precision2
        precision_100 += precision3
        accuracy_10 += accuracy2
        accuracy_100 += accuracy3
        valid_num += 1
        valid_loss += loss
    print("          Validation Loss  %8f\n" % (valid_loss/valid_num))
    scheduler.step(precision_1/valid_num)
    writer.writerow([epoch+1, round(precision_1/valid_num, 6), round(precision_10/valid_num, 6),
                    round(precision_100/valid_num, 6), round(precision_1/valid_num, 6),
                    round(accuracy_10/valid_num, 6), round(accuracy_100/valid_num, 6)])

result_file.close()
