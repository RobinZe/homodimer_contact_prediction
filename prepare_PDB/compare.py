# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__arthor__ = 'Robin'

import os

''' tran_name = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
             'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'PRO': 'P',
             'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q', 'LYS': 'K',
             'ARG': 'R', 'HIS': 'H', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',

             'ASX': 'N', 'GLX': 'Q', 'UNK': 'G', 'HSD': 'H'}

def ext_seq(name, chai):
    print('seq for', name, chai)
    fasta_fil = open(base_path + name + '/' + name + '_' + chai + '.fasta', 'w')
    fasta_fil.writelines(">%s || Chain %s \n" % (name, chai))
    pdb_fil = open(base_path + name + '/' + name + '.pdb', 'r')
    pdb_row = pdb_fil.readlines()
    for row in pdb_row:
        if row[:6] == 'SEQRES' and row[11] == chai:
            row = row.strip()[19:].split()
            for res in row:
                try:
                    fasta_fil.writelines(tran_name[res])
                except KeyError:
                    fasta_fil.writelines('X')
    fasta_fil.writelines("\n")
    pdb_fil.close()
    fasta_fil.close()


def ext_pdb(name, chai):
    print('pdb for', name, chai)
    pdb__fil = open(base_path + name + '/' + name + '_' + chai + '.pdb', 'w')
    pdb_fil = open(base_path + name + '/' + name + '.pdb', 'r')
    pdb_row = pdb_fil.readlines()
    for row in pdb_row:
        if row[:6] == 'HEADER':
            row = row.strip()
            pdb__fil.writelines(row + "\n")
        elif row[:6] == 'ATOM  ' and row[21] == chai:
            row = row.strip()
            pdb__fil.writelines(row + "\n")
        elif row[:3] == 'END':
            pdb__fil.writelines(row + "\n")
    pdb_fil.close()
    pdb__fil.close() '''

base_path = '/public/home/jiangyz/dataset/transform_c2/'
ref_fil = open(base_path + 'surprise_assembly.lst', 'r')
# fil = open(base_path + 'c2_trytry.lst', 'r')
bal_fil = open(base_path + 'c2_lst.lst', 'r')
out_fil = open(base_path + 'balanced_assembly.lst', 'w')
reference = ref_fil.readlines()
candidate = bal_fil.readlines()
pro_ass = dict()

for refer in reference:
    refer = refer.strip()
    pro_ass[refer[:4]] = set()

for refer in reference:
    refer = refer.strip()
    if '_' == refer[5]:
        for rrrefer in refer[5:]:
            pro_ass[refer[:4]].add(rrrefer)
    else:
        pro_ass[refer[:4]].add(refer[5:])

for candi in candidate:
    candi = candi.strip()
    name, chai = candi.split('_')
    # print(name, pro_ass[name])
    is_candi = False
    for chaii in pro_ass[name]:
        if set(chaii) - {'_'} == set(chai):
            is_candi = True
            break
    if is_candi:
        out_fil.writelines(candi + "\n")
    else:
        # print(pro_ass[name])
        for chaii in chai:
            if os.path.exists(base_path + name + '/' + name + '_' + chaii + '.a3m'):
                chai0 = chaii
                break
        # print(chai0, pro_ass[name])
        le_candi = False
        for chaii in pro_ass[name]:
            if chai0 in chaii:
                if len(chaii) == 1:
                    out_fil.writelines(name + '_' + chaii + "\n")
                elif len(chaii) == 2:
                    out_fil.writelines(name + '-' + chaii + "\n")
                le_candi = True
                break
        if not le_candi:
            chais = list(pro_ass[name]).sort(key=lambda i: len(i))
            chaii = chais[0]
            if len(chaii) == 1:
                out_fil.writelines(name + '_' + chaii + "\t" + 'FETRneeded' + "\n")
            else:
                out_fil.writelines(name + '-' + chaii + "\t" + 'FETRneeded' + "\n")


'''     pro.add(candid[:4])
    if candid[4] == '_':
        name, ass = candid.split('_')
        singl_pro.add(name)
        for chai in ass:
            name_ass = name + '_' + chai
            if os.path.exists(base_path + name + '/' + name_ass + '.a3m'):
                out_fil.writelines(name_ass + "\n")
                if not os.path.exists(base_path + name + '/' + name_ass + '.pdb'):
                    ext_pdb(name, chai)
                break
        if os.path.exists(base_path + name + '/' + name_ass + '.a3m'):
            continue
        else:
            name_ass = candid[:6]
            ext_seq(name, name_ass[-1])
            ext_pdb(name, name_ass[-1])
            out_fil.writelines(name_ass + "\t" + 'Features for' + name_ass[-1] + "\n")

doubl_pro = pro - singl_pro
for candid in candidates:
    if candid[:4] in doubl_pro:
        if not '-' in candid:
            print('strange name for', candid)
            continue
        candid = candid.strip()
        name, ass = candid.split('-')
        if '_' in candid:
            ass1, ass2 = ass.split('_')
            for chai in ass:
                name_ass = name + '_' + chai
                if os.path.exists(base_path + name + '/' + name_ass + '.a3m'):
                    if not os.path.exists(base_path + name + '/' + name_ass + '.pdb'):
                        ext_pdb(name, chai)
                    break
            if os.path.exists(base_path + name + '/' + name_ass + '.a3m'):
                if chai in ass1:
                    chaii = ass2[0]
                else:
                    chaii = ass1[0]
                out_fil.writelines(name + '-' + chai + chaii + "\n")
                if not os.path.exists(base_path + name + '/' + name + '_' + chaii + '.pdb'):
                    ext_pdb(name, chaii)
            else:
                name_ass = candid[:5] + ass1[0] + ass2[0]
                chai, chaii = name_ass[5:]
                ext_seq(name, chai)
                ext_pdb(name, chai)
                ext_pdb(name, chaii)
                out_fil.writelines(name_ass + "\t" + 'Features for ' + chai + "\n")
        else:
            for chai in ass:
                name_ass = name + '_' + chai
                if os.path.exists(base_path + name + '/' + name_ass + '.a3m'):
                    if not os.path.exists(base_path + name + '/' + name_ass + '.pdb'):
                        ext_pdb(name, chai)
                    break
            if os.path.exists(base_path + name + '/' + name_ass + '.a3m'):
                chaii = ass.replace(chai, '')[0]
                out_fil.writelines(name + '-' + chai + chaii + "\n")
                if not os.path.exists(base_path + name + '/' + name + '_' + chaii + '.pdb'):
                    ext_pdb(name, chaii)
            else:
                name_ass = candid[:7]
                chai, chaii = name_ass[5:]
                ext_seq(name, chai)
                ext_pdb(name, chai)
                ext_pdb(name, chaii)
                out_fil.writelines(name_ass + "\t" + 'Features for ' + chai + "\n") '''

out_fil.close()
ref_fil.close()
bal_fil.close()
