# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

import os
import sys

base_path = '/public/home/jiangyz/dataset/transform_c_else'
f_lst = open(base_path + '/' + sys.argv[1], 'r')

''' tran_name = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
             'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'PRO': 'P',
             'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q', 'LYS': 'K',
             'ARG': 'R', 'HIS': 'H', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',
             'ASX': 'N', 'GLX': 'Q', 'UNK': 'G', 'HSD': 'H'} '''


def identify(path, chai):
    a1, a2 = chai[0], chai[1]
    ident_result, bool_global = False, False
    os.system("~/bin/blast/bl2seq -p blastp -i %s_%s.fasta -j %s_%s.fasta -e 0.01 -F F -o %s_%s.blast"
              % (path, a1, path, a2, path, chai))

    try:
        f_bl = open('%s_%s.blast' % (path, chai), 'r')
    except FileNotFoundError:
        return ident_result
    f_bls = f_bl.readlines()
    for row in f_bls:
        if 'letters)' in row:
            total_length = eval(row.split()[0].split('(')[-1])
            # print(total_length)
        if 'Length =' in row:
            align_length = eval(row.split()[-1])
            align_ratio = align_length / total_length
            # print(align_length)
            if align_ratio > 0.8:
                bool_global = True
            else:
                bool_global = False
        if bool_global and 'Identities' in row:
            iden = row.replace(' ', '')
            iden = iden.split('%')[0]
            iden = iden.split('(')[-1]
            iden = eval(iden)
            bool_global = False
            # print(iden)
            if iden > 98.9999:
                ident_result = True
    f_bl.close()
    return ident_result

'''
def refind(p, a):
    fil = open(base_path + '/' + p + '/' + p + '.pdb', 'r')
    pdb_rows = fil.readlines()
    remain_line, rigid_line = 0, 0

    for pdb_row in pdb_rows:
        if 'REMARK 350' in pdb_row and 'APPLY THE FOLLOWING' in pdb_row and ', '.join(a) in pdb_row:
            remain_line = 6
        if remain_line > 0 and 'BIOMT' in pdb_row[10:20]:
            tran_str, tran_num = pdb_row[24:].split(), []
            for ts in tran_str:
                tran_num.append(eval(ts))
            if set(tran_num) == {0., 1.}:
                rigid_line += 1
            remain_line -= 1

        if 'ATOM' in pdb_row[:6]:
            break
    fil.close()
    if rigid_line >= 6:
        return True
    else:
        return False '''


f_lists = f_lst.readlines()
homo_list = []

for name in f_lists:
    name = name.strip()
    ''' if '_' in name:
        names = name.split('_')
        pro = names[0]
        ass = names[1]
        if len(ass) > 1:
            double_chain = refind(pro, ass)
            if double_chain:
                assems = []
                for a1 in range(len(ass)):
                    for a2 in range(a1+1, len(ass)):
                        a_ss = ass[a1] + ass[a2]
                        similarity = identify(base_path + '/' + pro + '/' + pro, a_ss)
                        if similarity:
                            assems.append(a_ss)
                if len(assems) > 0:
                    name = pro + '-' + '-'.join(assems)
                else:
                    continue '''

    if '-' in name:
        assems = []
        names = name.split('-')
        pro = names[0]
        ass = names[1]
        for a1 in range(len(ass)):
            for a2 in range(a1+1, len(ass)):
                a_ss = ass[a1] + ass[a2]
                similarity = identify(base_path + '/' + pro + '/' + pro, a_ss)
                if similarity:
                    assems.append(a_ss)
        if len(assems) > 0:
            name = pro + '-' + '-'.join(assems)
        else:
            continue
    homo_list.append(name)



'''
    try:
        f = open(base_path + '/' + name + '/' + pdb, 'r')
    except FileNotFoundError:
        print(name, 'not found!')
        continue
    info = f.readlines()
    seqs, aa_num = [''], set()
    for row in info:
        if row[:6] == 'ATOM  ':
            seq_seq = row[17:20]
            aa_aa = row[22:27].replace(' ', '')
            if aa_aa not in aa_num:
                try:
                    seqs[-1] += tran_name[seq_seq]
                except KeyError as e:
                    print(name, 'has key error!')
                    print(e)
                aa_num.add(aa_aa)
        elif row[:6] == 'TER   ':
                seqs.append('')
                aa_num = set()

    while '' in seqs:
        seqs.remove('')
    if len(seqs) != 2:
        print('Unsuitable protein,', name, 'Not 2 seqs!')
        continue
    seq1, seq2 = seqs[0], seqs[1]

    f1 = open(base_path + '/' + name + '/' + name + '.seq1', 'w')
    f2 = open(base_path + '/' + name + '/' + name + '.seq2', 'w')
    while len(seq1) > 60:
        f1.writelines(seq1[:60] + "\n")
        seq1 = seq1[60:]
    while len(seq2) > 60:
        f2.writelines(seq2[:60] + "\n")
        seq2 = seq2[60:]
    f1.writelines(seq1 + "\n")
    f2.writelines(seq2 + "\n")
    f1.close()
    f2.close()
    f.close()
'''

f_lst.close()

f1_lst = open(base_path + '/c_homo.lst', 'w')
for item in homo_list:
    f1_lst.writelines(item + "\n")
f1_lst.close()
