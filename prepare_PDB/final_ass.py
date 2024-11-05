# !/usr/bin/env_python3
# -*- coding:utf-8 -*-
__author__ = 'Robin'

base_path = '/public/home/jiangyz/dataset/transform_c2'
f1 = open(base_path + '/c2_nored.lst', 'r')
f1s = f1.readlines()
pro1, pro2 = set(), set()
for i in f1s:
    i = i.strip()
    if '-' in i:
        pro2.add(i[:4])
    else:
        pro1.add(i)
f1.close()


def get_bool(p, a):
    f = open(base_path + '/' + p + '/' + p + '.pdb', 'r')
    fs = f.readlines()
    sa1, sa2 = set(), set()
    for row in fs:
        if row[:4] == 'ATOM' and row[21] == a[0]:
            sa1.add(int(row[22:26]))
        elif row[:4] == 'ATOM' and row[21] == a[1]:
            sa2.add(int(row[22:26]))
    f.close()
    if sa1 == sa2:
        return True
    else:
        return False


f2 = open(base_path + '/c2_homo.lst', 'r')
f2s = f2.readlines()
ass = set()
for i in f2s:
    i = i.strip()
    j = i[:4]
    if j in pro1:
        ass.add(i[:6])
        pro1.remove(j)
    elif j in pro2:
        js = i.split('-')[1:]
        pro2.remove(j)
        for jj in js:
            if get_bool(j, jj):
                ass.add(j + '-' + jj)
                break

assert len(pro1) == len(pro2) == 0
f2.close()

f3 = open(base_path + '/c2_final.lst', 'w')
for i in ass:
    f3.writelines(i + "\n")
f3.close()
