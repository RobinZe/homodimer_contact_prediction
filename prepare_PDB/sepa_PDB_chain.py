# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__arthor__ = 'Robin'

import os
base_path = '/public/home/jiangyz/dataset/transform_c2/'


def write_pdb(name, chai):
    wf = open(base_path + name + '/' + name + '_' + chai + '.pdb', 'w')
    rf = open(base_path + name + '/' + name + '.pdb', 'r')
    rfs = rf.readlines()
    for r in rfs:
        if r[:6] == 'HEADER' or r[:3] == 'END':
            wf.writelines(r.strip() + "\n")
        elif r[:3] == 'TER' and r[21] == chai:
            wf.writelines(r.strip() + "\n")
        elif r[:4] == 'ATOM' and r[21] == chai:
            wf.writelines(r.strip() + "\n")
    rf.close()
    wf.close()


fil = open(base_path + 'prepared_assembly.lst', 'r')
fils = fil.readlines()
for pro_ass in fils:
    pro_ass = pro_ass.strip()
    pro, ass = pro_ass[:4], pro_ass[5:]
    for asse in ass:
        if not os.path.exists(base_path + pro + '/' + pro + '_' + asse + '.pdb'):
            write_pdb(pro, asse)
fil.close()
