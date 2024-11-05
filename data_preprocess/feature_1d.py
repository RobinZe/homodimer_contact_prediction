# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import numpy as np
from bin.LoadHHM import *
import math


def unit_8d(n):
    arr = [0] * 7
    arr.insert(n-1, 1)
    return arr


def load_data_from_file(prefix):
    # read the sequence
    sequence = ""
    with open(prefix + ".seq") as f:
        lines = f.readlines()
        if lines[0][0] != ">":
            print("Sequence must start with > !")
            exit(2)
        for line in lines[1:]:
            sequence = sequence + line.strip().replace("/", "")

    seq_list = list(sequence)
    seq_len = len(sequence)

    # produce the hydrophobic feature
    hydrophobic_dict = {"A": 0.50, "C": -0.02, "D": 3.64, "E": 3.63, "F": -1.71,
                        "G": 1.15, "H": 2.33, "I": -1.12, "K": 2.80, "L": -1.25,
                        "M": -0.67, "N": 0.85, "P": 0.14, "Q": 0.77, "R": 1.81,
                        "S": 0.46, "T": 0.25, "V": -0.46, "W": -2.09, "Y": -0.71,
                        "X": 0.50}
    hydro_data = np.zeros((seq_len,))
    for i in range(len(seq_list)):
        if seq_list[i] not in hydrophobic_dict:
            seq_list[i] = 'X'
        hydro_data[i] = hydrophobic_dict[seq_list[i]]
    hydro_data = np.reshape(hydro_data, (1,)+hydro_data.shape)

    # produce the PSSM from HHM file using Xu's script LoadHHM.py
    pssm_data = load_hmm(prefix + ".hhm")['PSSM']
    # check the PSSM shape
    if seq_len != pssm_data.shape[0]:
        print("Wrong shape for PSSM!")
    pssm_data = pssm_data.transpose()

    # read the dssp data
    dssp_data = [[]] * seq_len
    acc_data = np.zeros((seq_len,))
    ra_data = np.zeros((seq_len,))

    dssp_dict = {"H": 1, "B": 2, "E": 3, "G": 4, "I": 5, "T": 6, "S": 7, " ": 8}
    acc_dict = {'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
                'G': 78.7,  'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
                'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
                'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7}
    nres = 0
    flag = False
    with open(prefix + ".dssp") as f:
        lines = f.readlines()
        for line in lines:
            line_ = line.strip()
            if line_[0] == "#":
                flag = True
                continue
            if flag:
                if seq_list[nres] != line[13]:
                    if seq_list[nres + 1] == line[13]:
                        dssp_data[nres] = [] * 8
                        acc_data[nres] = 0
                        ra_data[nres] = 0
                        nres += 1
                    else:
                        print(prefix.split('/')[-1], "Inconsistent sequnece for dssp!", nres + 1, line[13], seq_list[nres])
                        continue
                dssp_data[nres] = unit_8d(dssp_dict[line[16]])
                acc_data[nres] = int(line[35:38])
                ra_data[nres] = acc_data[nres]/acc_dict[seq_list[nres]]
                nres += 1
    dssp_data = np.array(dssp_data).transpose()
    acc_data = np.reshape(acc_data, (1,)+acc_data.shape)
    ra_data = np.reshape(ra_data, (1,)+ra_data.shape)

    # check the shape
    assert hydro_data.shape[-1] == pssm_data.shape[-1] == dssp_data.shape[-1] == acc_data.shape[-1] \
        == ra_data.shape[-1] == seq_len
    ''' data = {"Name": prefix, "Sequence": sequence,
            "Hydro": hydro_data, "PSSM": pssm_data, "DSSP": dssp_data, "ACC": acc_data, "RA": ra_data}
    return data '''

    # sequential feature
    seq_data = np.concatenate((pssm_data, dssp_data, acc_data, ra_data, hydro_data), axis=0)
    '''# pair feature
    pair_data = np.concatenate((data['DI'], data['APC'], data['Dismap'], data['Dockmap']), axis=0)
    if seq_len > 400:
        reduced = np.random.randint(0, seq_len, 400).sort()
        seq_data = seq_data[:, reduced]
    # add the batch dimension
    seq_data = np.reshape(seq_data, ((1,) + seq_data.shape))
    pair_data = np.reshape(pair_data, ((1,) + pair_data.shape)) '''
    return seq_data


''' fi = open('/public/home/jiangyz/dataset/h3_C2.lst', 'r')
profile = fi.readlines()
ok_set = set() '''
profile = ['3PWF']

for perfix in profile:
    perfix = perfix.strip()
    # try:
    x = load_data_from_file('/dl/jiangyz/data/' + perfix + '/' + perfix)
    ''' except FileNotFoundError:
        print('No such feature data:', perfix)
        continue
    except IndexError:
        print(perfix, 'DSSP length out of range!')
        continue
    except TypeError:
        print('Data Wrong with:', perfix)
        continue
    except AssertionError:
        print('Dimension inconsistence with:', perfix)
        continue '''
    np.savetxt('/dl/jiangyz/data/3PWF/%s_1d.npy' % perfix, x, fmt='%f')
    # np.savetxt('/public/home/jiangyz/dataset/dssp/%s_pair.npy', y, fmt='%f')
    # ok_set.add(perfix)
    # print(perfix, 'data loaded.')

''' fi.close()
fo = open('/public/home/jiangyz/dataset/C2_1d.lst', 'w')
for perfix in ok_set:
    fo.writelines(perfix + "\n")
fo.close() '''
