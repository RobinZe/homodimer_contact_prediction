# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__aothor__ = 'Robin'

f = open('/public/home/jiangyz/dataset/h99_C2.txt', 'r')
fo = open('/public/home/jiangyz/dataset/mm/C2.fasta', 'w')
f_name = f.readlines()

for name in f_name:
    name = name.rstrip("\n")
    name = name.rstrip("\r")
    fo.writelines('>' + name + " || Sequence 1\n")
    f_seq = open('/public/home/jiangyz/dataset/huang_C2/%s/%s.seq' % (name, name), 'r')
    seqs = f_seq.readlines()
    for seq in seqs[1:]:
        seq = seq.rstrip("\n")
        seq = seq.rstrip("\r")
        fo.writelines(seq)
    fo.writelines("\n")
    f_seq.close()

fo.close()
f.close()
