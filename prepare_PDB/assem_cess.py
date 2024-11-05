# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import numpy as np

tran_name = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
             'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'PRO': 'P',
             'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q', 'LYS': 'K',
             'ARG': 'R', 'HIS': 'H', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',
             'ASX': 'N', 'GLX': 'Q', 'UNK': 'G', 'HSD': 'H'}

''' if len(sys.argv) < 2:
    print('Usage :  Python assem_cess.py $XXXX.pdbN')
    exit(1) '''

# This process will read a PDB file and output a PDB file and a FASTA file
# and finally return coordinate matrix.
f = open('/public/home/jiangyz/dataset/h3_C2.lst', 'r')
f_lines = f.readlines()

for name in f_lines:
    name = name.rstrip("\n")
    name = name.rstrip("\r")
    # files preparation
    pro, assem = name.split('-')
    filename = pro + '.pdb' + assem

    f_pdb = open('/public/home/jiangyz/dataset/huang_C2/%s/%s' % (name, filename), 'r')
    pdb_info = f_pdb.readlines()
    # processing procedure
    ''' s = [set()]
    for lines in pdb_info:
        if lines[:6] == 'ATOM  ' and lines[77] != 'H':
            s[-1].add((eval(lines[22:27]), tran_name[lines[17:20]]))
        elif lines[:6] == 'TER   ':
            s.append(set())

    # fa = open('/public/home/jiangyz/dataset/assertion.txt', 'w')
    while set() in s:
        s.remove(set())
    assert len(s) == 2
    s0 = s[0]
    if s[0] != s[1]:
        s0 = s[0].intersection(s[1])
        print('Inconsistent assemblies!')
    if len(s0)/len(s[1]) < 0.8:
        print("It's not a homo-dimer for the too-low consistence!")
        f_pdb.close()
        exit(2)

    fo_fasta = open(tag + '.seq', 'w')
    fo_fasta.writelines('>' + pro + ' ||assembly' + assem + "\n") '''
    # fo_pdb = open('/public/home/jiangyz/dataset/huang_C2/%s/%s.pdb' % (name, name), 'w')
    # fo_pdb.writelines('HEADER    PRO NAME: ' + pro + '  ASSEMBLY: ' + assem + "\n")
    ca, na, nan = [[]], set(), set()
    model = True
    seq, cor, cor_cb, n_v = '', [], [0, 0, 0], set()

    for lines in pdb_info:
        if lines[:6] == 'ATOM  ' and lines[17:20] in tran_name:
            aa_num = lines[22:27].strip()
            if aa_num not in nan:
                nan.add(aa_num)
                ca[-1].append([])
            if lines[77] != 'H':
                ca[-1][-1].append([eval(lines[30:38]), eval(lines[38:46]), eval(lines[46:54])])
            if model:
                # fo_pdb.writelines(lines)
                if aa_num not in na:
                    na.add(aa_num)
                    # seq += seq_name
                    if [0, 0, 0] != cor_cb and [0, 0, 0] != cor[-1]:
                        cor[-1] = cor_cb
                    cor.append([0, 0, 0])
                    cor_cb = [0, 0, 0]
                if ' CA ' == lines[12:16]:
                    cor[-1] = [eval(lines[30:38]), eval(lines[38:46]), eval(lines[46:54])]
                    n_v.add(len(na) - 1)
                elif ' CB ' == lines[12:16]:
                    cor_cb = [eval(lines[30:38]), eval(lines[38:46]), eval(lines[46:54])]
        elif lines[:6] == 'TER   ':
            if model and len(cor) > 0:
                # fo_pdb.writelines('TER')
                # fo_pdb.close()
                model = False
            ca.append([])
            nan = set()

    ''' while len(seq) > 60:
        fo_fasta.writelines(seq[:60] + "\n")
        seq = seq[60:]
    # fo_fasta.writelines(seq)
    fo_fasta.close() '''
    f_pdb.close()

    D, m = [], np.array(cor)
    for i in range(len(m)):
        d = np.sum((m - m[i]) * (m - m[i]), 1)
        D.append(np.sqrt(d))
    D = np.array(D)
    n_iv = set(range(len(m))) - n_v
    for i in n_iv:
        D[i, :] = -1.
        D[:, i] = -1.
    diameter = np.max(D)
    np.savetxt('/public/home/jiangyz/dataset/huang_C2/%s/%s_intra_dis.npy' % (name, name), D, fmt='%f')
    D = np.where(D <= 12., D, 0.)
    D = np.where(D > 0., 1., 0.)
    np.savetxt('/public/home/jiangyz/dataset/huang_C2/%s/%s_intra_pred.npy' % (name, name), D, fmt='%d')

    f_blast = open('/public/home/jiangyz/dataset/huang_C2/%s/%s.blast' % (name, name), 'r')
    info_blast = f_blast.readlines()
    seq1, seq2, st1, ed1, st2, ed2 = '', '', set(), set(), set(), set()
    for row in info_blast:
        if 'Identities' in row:
            if seq1 == '':
                continue
            else:
                break
        if row[:6] == 'Query:':
            rows = row.split(' ')
            seq1 += rows[-2]
            st1.add(eval(rows[1]))
            ed1.add(eval(rows[-1]))
        elif row[:6] == 'Sbjct:':
            rows = row.split(' ')
            seq2 += rows[-2]
            st2.add(eval(rows[1]))
            ed2.add(eval(rows[-1]))
    f_blast.close()
    seq1, seq2 = list(seq1), list(seq2)
    assert len(seq1) == len(seq2)

    while [] in ca:
        ca.remove([])
    imp1, imp2 = ca[0], ca[1]
    # print(len(imp1), len(imp2), len(seq1), len(seq2))
    at1, at2 = min(st1)-1, min(st2)-1
    am1, am2 = max(ed1), max(ed2)
    seq1 = [0]*at1 + seq1 + [0]*(len(imp1)-am1)
    seq2 = [0]*at2 + seq2 + [0]*(len(imp2)-am2)
    try:
        for atomk in range(len(seq2)):
            if seq2[atomk] == '-':
                imp2.insert(atomk, [])
        assert len(imp2) == len(seq2)
        while '-' in seq1:
            ai = seq1.index('-')
            seq1.pop(ai)
            seq2.pop(ai-at1+at2)
            imp2.pop(ai-at1+at2)
        assert len(seq1) == len(imp1)
    except AssertionError as ae:
        print(name, len(seq1), len(imp1))
    # D = np.zeros((len(imp1), len(imp2)))
    M = np.ones((len(imp1), len(imp1)))*diameter
    for atomi in range(len(imp1)):
        for atomj in range(len(imp2)):
            if atomj-at1+at2 < len(imp1) and at1 >= at2:
                dis = diameter
                cor1, cor2 = np.array(imp1[atomi]), np.array(imp2[atomj])
                for row in cor2:
                    d2 = np.sum((row - cor1) * (row - cor1), axis=1)
                    dis = min(np.sqrt(np.min(d2)), dis)
                M[atomi, atomj-at1+at2] = dis
    con_pred = np.where(M <= 8, 1., 0.)

    np.savetxt('/public/home/jiangyz/dataset/huang_C2/%s/%s_inter_dis.npy' % (name, name), M, fmt='%f')
    np.savetxt('/public/home/jiangyz/dataset/huang_C2/%s/%s_inter_pred.npy' % (name, name), con_pred, fmt='%d')

f.close()
