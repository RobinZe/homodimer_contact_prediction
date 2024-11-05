# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import os
import sys
import numpy as np
import pandas as pd

''' if len(sys.argv) == 1:
    print('Usage: python dist.py list_file')
    exit(2) '''
base_tran = np.hstack((np.eye(3), np.zeros((3, 1))))


def read_pdb(name):
    f = open(base_path + '/' + name[:4] + '/' + name + '.pdb', 'r')
    fs = f.readlines()
    ordins = []
    for row in fs:
        if 'ATOM  ' == row[:6]:
            atom_name = row[12:16].replace(' ', '')
            res_num = int(row[22:26])
            x = eval(row[30:38])
            y = eval(row[38:46])
            z = eval(row[46:54])
            ordins.append([atom_name, res_num, x, y, z])
    f.close()
    ordin = pd.DataFrame(ordins, columns=['atom_name', 'res_num', 'x', 'y', 'z'])
    return ordin


def read_tran(name, asse):
    f = open(base_path + '/' + name + '/' + name + '_tran.txt', 'r')
    fs = f.readlines()
    bool_write = False
    for row in fs:
        row = row.strip()
        if 0 < len(row) < 5:
            if row[0] == asse:
                bool_write = True
                tran_ordin = []
            else:
                bool_write = False
        elif bool_write and len(row) > 0 and row[0] != ' ':
            tran_ord = row.split()
            tran_ord = list(map(eval, tran_ord))
            tran_ordin.append(tran_ord)
    f.close()
    try:
        return np.array(tran_ordin)
    except UnboundLocalError:
        return base_tran


def read_len(name):
    f = open(base_path + '/' + name[:4] + '/' + name + '.fasta', 'r')
    fs = f.readlines()
    sequence = fs[1].strip()
    return len(sequence)


def intra_dis(dataframe, l):
    d_mat = np.zeros((l, l))
    for i in range(l):
        for j in range(i+1, l):
            d1 = dataframe[dataframe['res_num'] == i+1]
            d2 = dataframe[dataframe['res_num'] == j+1]
            if 'CA' in list(d1['atom_name']) and 'CA' in list(d2['atom_name']):
                dd1 = np.array(d1[d1['atom_name'] == 'CA'][['x', 'y', 'z']])[0]
                dd2 = np.array(d2[d2['atom_name'] == 'CA'][['x', 'y', 'z']])[0]
                if 'CB' in list(d1['atom_name']):
                    dd1 = np.array(d1[d1['atom_name'] == 'CB'][['x', 'y', 'z']])[0]
                if 'CB' in list(d2['atom_name']):
                    dd2 = np.array(d2[d2['atom_name'] == 'CB'][['x', 'y', 'z']])[0]
                d_mat[i, j] = np.sqrt(np.sum((dd1 - dd2)**2))
            else:
                d_mat[i, j] = -1
    intra_mat = d_mat + d_mat.T
    return intra_mat


def min_atom(mat1, mat2):
    dis_list = []
    for i in mat1:
        for j in mat2:
            dis_list.append(np.sqrt(np.sum((i-j)**2)))
    min_dist = min(dis_list)
    return min_dist


def inter_dis(ord1, ord2, l, tran_mat1=base_tran, tran_mat2=base_tran):
    d_mat = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            d1 = ord1[(ord1['res_num'] == i+1) & (ord1['atom_name'] != 'H')]
            d2 = ord2[(ord2['res_num'] == j+1) & (ord2['atom_name'] != 'H')]
            dd1 = np.array(d1[['x', 'y', 'z']])
            dd2 = np.array(d2[['x', 'y', 'z']])
            try:
                dd1 = np.matmul(dd1, tran_mat1[:, :3].T) + tran_mat1[:, 3].T
                dd2 = np.matmul(dd2, tran_mat2[:, :3].T) + tran_mat2[:, 3].T
                d_mat[i, j] = min_atom(dd1, dd2)
            except ValueError:
                d_mat[i, j] = -1

    d_mat = (d_mat + d_mat.T) / 2
    return d_mat


''' def check_tran(pro, chains):
    check_list = set()
    tran_fil = open(base_path + '/' + pro + '/' + pro + '_tran.txt', 'r')
    tran_row = tran_fil.readlines()
    for row in tran_row:
        if len(row) < 5:
            check_list.add(row[0])
    if set(chains) == check_list - {' ', '\n'}:
        check_result = True
    else:
        check_result = False
    tran_fil.close()
    return check_result '''


base_path = '/public/home/jiangyz/dataset/transform_c2'
in_fil = open(base_path + '/prepared_assembly.lst', 'r')
# in_fil = open(base_path + '/c2_try.lst', 'r')
fil_row = in_fil.readlines()
for row in fil_row:
    row = row.strip()
    l = read_len(row[:6].replace('-', '_'))

    if '_' in row:
        ord1 = read_pdb(row)
        tran_mat1 = read_tran(row[:4], row[5])
        tran_mat2 = tran_mat1[3:]
        tran_mat1 = tran_mat1[:3]
        ord2 = ord1.copy(deep=True)

    elif '-' in row:
        chai = row[5:]
        ord1 = read_pdb(row[:4] + '_' + chai[0])
        ord2 = read_pdb(row[:4] + '_' + chai[-1])
        if '-' in chai:
            tran_mat1 = read_tran(row[:4], chai[0])
            tran_mat2 = read_tran(row[:4], chai[-1])
        else:
            tran_mat1 = base_tran
            tran_mat2 = base_tran

    intra_dist = intra_dis(ord1, l)
    inter_dist = inter_dis(ord1, ord2, l, tran_mat1, tran_mat2)
    np.save(base_path + '/' + row[:4] + '/' + row[:6] + '_intra_dis.npy', intra_dist)
    np.save(base_path + '/' + row[:4] + '/' + row + '_inter_dis.npy', inter_dist)
