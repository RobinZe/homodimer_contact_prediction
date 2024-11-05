# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__authoe__ = 'Robin'

import numpy as np

base_path = '/public/home/jiangyz/dataset/transform_c2'
f_list = open(base_path + '/c2_homo.lst', 'r')


def get_itf(name):
    fil = open(base_path + '/' + name + '/' + name + '.pdb', 'r')
    lines = fil.readlines()
    b_dimer, dn, dr, dc = False, 0, 0, set()
    for li in lines:
        if 'REMARK 350' == li[:10]:
            if b_dimer and 'BIOMOLECULE' in li:
                if dn > 0:
                    for ke in d_itf1:
                        if ke[:4] == name and set(ke[5:]).issubset(dc):
                            d_itf1[ke] = max(d_itf1[ke], dr)
                else:
                    for ke in d_itf2:
                        if ke[:4] == name and set(ke[5:]).issubset(dc):
                            d_itf2[ke] = max(d_itf2[ke], dr)
                b_dimer, dn, dr, dc = False, 0, 0, set()
            elif 'DIMERIC' in li:
                b_dimer = True
            elif b_dimer and 'BURIED SURFACE' in li:
                dr = eval(li.split()[-2])
            elif b_dimer and 'APPLY THE FOLLOWING' in li:
                dc = li.split(':')[-1]
                dc = set(dc) - {',', ' '}
            elif b_dimer and 'BIOMT' in li:
                di = li[24:].split()
                dm = set()
                for i in di:
                    dm.add(eval(i))
                if dm != {0., 1.}:
                    dn += 1
        if 'SEQRES' in li[:6] and b_dimer:
            if dn > 0:
                for ke in d_itf1:
                    if ke[:4] == name and set(ke[5:]).issubset(dc):
                        d_itf1[ke] = max(d_itf1[ke], dr)
            else:
                for ke in d_itf2:
                    if ke[:4] == name and set(ke[5:]).issubset(dc):
                        d_itf2[ke] = max(d_itf2[ke], dr)
            break


pro_list = f_list.readlines()
d_itf1, d_itf2, s_pro = dict(), dict(), set()

for pro in pro_list:
    pro = pro.strip()
    if '_' in pro:
        d_itf1[pro] = 0
        s_pro.add(pro[:4])
    elif '-' in pro:
        d_itf2[pro] = 0
        s_pro.add(pro[:4])


for pro in s_pro:
    pro = pro.strip()
    get_itf(pro)

real_ok = set()
prepared_list = []
for pro in d_itf1:
    if pro[:4] not in real_ok:
        if d_itf1[pro] >= 1000:
            real_ok.add(pro[:4])
            prepared_list.append(pro)
        # elif d_itf1[pro] == 0:
        #     print('zero interface:',pro)
for pro in d_itf2:
    if pro[:4] not in real_ok:
        if d_itf2[pro] >= 1000:
            real_ok.add(pro[:4])
            prepared_list.append(pro)
        # elif d_itf2[pro] == 0:
        #     print('zero interface:', pro)

f_list.close()
f_itf = open(base_path + '/redund_interfc.lst', 'w')
for i in prepared_list:
    f_itf.writelines(i + "\n")
f_itf.close()
