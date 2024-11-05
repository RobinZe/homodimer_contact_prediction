# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'


base_path = '/public/home/jiangyz/dataset/transform_c2'
base_list = 'c2_homo.lst'

sgl, dbl = set(), set()
fch = open(base_path + '/' + base_list, 'r')
fchs = fch.readlines()
for i in fchs:
    if '_' in i:
        sgl.add(i[:4])
    elif '-' in i:
        dbl.add(i[:4])
dbl = dbl - sgl


def get_rsl(pdb):
    pdb = pdb.strip()
    f_rsl = open(base_path + '/%s/%s.pdb' % (pdb, pdb), 'r')
    rsl = f_rsl.readlines()
    out_rsl = 3.0
    for row in rsl:
        if row[:10] == 'REMARK   2' and 'RESOLUTION' in row:
            rows = row.split()
            try:
                out_rsl = eval(rows[-2])
            finally:
                break
        elif row[:10] == 'REMARK   3' or row[:6] == 'SEQRES':
            break
    f_rsl.close()
    return out_rsl


def comp(s):
    if s in sgl:
        return 0
    elif s in dbl:
        return 10
    else:
        return 20


f = open(base_path + '/c2db_clu.tsv', 'r')
pairs = f.readlines()
h3, clu = [], set()
lpro, lrsl = [], []
min_rsl = 3.0

for pair in pairs:
    pair.strip()
    names = pair.split()
    assert len(names) == 2
    leader = names[0]

    if leader in clu:
        follower = names[1]
        lpro.append(follower + str(comp(follower)))
        lrsl.append(get_rsl(follower) + comp(follower))
        ''' current_rsl = get_rsl(follower)
        if current_rsl < min_rsl:
            min_rsl = current_rsl
            h3.pop()
            h3.append(follower) '''
    else:
        if bool(lpro):
            h3.append(lpro[lrsl.index(min(lrsl))][:-1])
            del lpro[:], lrsl[:]
        clu.add(leader)
        lpro.append(leader + str(comp(leader)))
        lrsl.append(get_rsl(leader) + comp(leader))
        # min_rsl = get_rsl(leader)
if not lpro:
    h3.append(lpro[lrsl.index(min(lrsl))][:-1])

f.close()
fo = open(base_path + '/nonredund.lst', 'w')
for row in h3:
    fo.writelines(row + "\n")
fo.close()
