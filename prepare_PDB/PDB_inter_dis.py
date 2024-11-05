# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import sys
import numpy as np

if len(sys.argv) < 2:
    print('Usage :  Python PDB_process.py $filename')
    exit(1)

tran_name = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
             'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'PRO': 'P',
             'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q', 'LYS': 'K',
             'ARG': 'R', 'HIS': 'H', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',

             'ASX': 'N', 'GLX': 'Q', 'UNK': 'G', 'HSD': 'H'}

if len(sys.argv) < 2:
    print('Usage :  Python PDB_process.py $filename')
    exit(1)

# This process will read a PDB file and output a PDB file and a FASTA file
# and finally return coordinate matrix.


for name in sys.argv[1:]:
    # files preparation
    filtr = False
    names = name.split('_')
    pro, chain = names[0], ''
    if len(names) == 2:
        filtr = True
        chain = set(names[1])
        # fo_fasta = open(name + '.seq', 'w')
        fo_pdb = open(name + '.pdb', 'w')  # output another PDB file if filter needed
        fo_pdb.writelines('PRO  NAME :  ' + pro + '   CHAIN :  ' + ' '.join(chain) + "\n")
    # elif len(names) == 1:
    #     fo_fasta = open(pro + '.seq', 'w')
    else:
        print('Input name error!')
        exit(2)
    f_pdb = open(pro + '.pdb', 'r')
    pdb_info = f_pdb.readlines()
    # processing procedure
    seq, cor, cor_cb, n_v = '', [], [0, 0, 0], set()
    tr_mat, tr_bia = [], []
    atom__num = '00'
    for lines in pdb_info:
        if 'ATOM' not in lines[0:6]:
            if lines[0:10] == 'REMARK 350' and len(tr_bia) < 3 and lines[22] == '2':
                tr_mat.append([eval(lines[24:33]), eval(lines[34:43]), eval(lines[44:53])])
                tr_bia.append(eval(lines[57:]))
        elif lines[21] in chain or chain == '':
            if filter:
                fo_pdb.writelines(lines)
            atom_num = lines[22:26].replace(' ', '')
            if atom_num != atom__num:
                # seq += tran_name[lines[17:20]]
                if [0, 0, 0] != cor_cb and [0, 0, 0] != cor[-1]:
                    cor[-1] = cor_cb
                cor.append([0, 0, 0])
                cor_cb = [0, 0, 0]
                atom__num = atom_num
            if 'CA' in lines[12:16]:
                cor[-1] = [eval(lines[30:38]), eval(lines[38:46]), eval(lines[46:54])]
                n_v.add(len(seq)-1)
            elif 'CB' in lines[12:16]:
                car_cb = [eval(lines[30:38]), eval(lines[38:46]), eval(lines[46:54])]
    if filtr:
        fo_pdb.writelines('END')
        fo_pdb.close()
        ''' fo_fasta.writelines('> ' + pro + ' || chain ' + ' '.join(list(chain)) + "\n")
    else:
        # fo_fasta.writelines('> ' + pro + "\n")
    while len(seq) > 60:
        fo_fasta.writelines(seq[:60] + "\n")
        seq = seq[60:]
    fo_fasta.writelines(seq)
    fo_fasta.close() '''
    f_pdb.close()

    D_tra, D_ter, m1, tform, tbias = [], [], np.array(cor), np.array(tr_mat), np.array(tr_bia)
    m2 = np.matmul(m1, tform) + tbias # .reshape((1, 3))
    # print(m2.shape)
    for i in range(len(cor)):
        d1 = np.sum((m1 - m1[i]) * (m1 - m1[i]), 1)
        d2 = np.sum((m1 - m2[i]) * (m1 - m2[i]), 1)
        D_tra.append(np.sqrt(d1))
        D_ter.append(np.sqrt(d2))
    n_iv = set(range(len(cor))) - n_v
    D_tra, D_ter = np.array(D_tra), np.array(D_ter)
    for i in n_iv:
        D_tra[i, :] = -1.
        D_tra[:, i] = -1.
        D_ter[i, :] = -1.
        D_ter[:, i] = -1.

    con_pred = np.where(D_ter > 8, 0, D_ter)
    con_pred = np.where(D_tra <12, 0, con_pred)
    con_pred = np.where(con_pred <= 0, 0, 1)
    # np.savetxt(name + '_dis.npy', D, fmt='%f')
    np.savetxt(name + '_pred.npy', con_pred, fmt='%d')
