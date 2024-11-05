# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__arthor__ = 'Robin'

import os
fr=open('/dl/jiangyz/data/ddd','r')
# fo=open('/dl/jiangyz/data/huang_C0/remain','w')
lst=fr.readlines()

# split for assembly files
''' for i in lst:
    ii=i.split('-')
    pro=ii[0]
    # ass=ii[1]
    pro_lst.add(pro)
    try:
        os.system('curl -s -f https://files.rcsb.org/download/%s.pdb%s.gz -o ~/data/huang_C0/%s.pdb%s.gz'%(pro,ass,pro,ass))
    except:
        fo.writelines(i+"\n")
    if os.path.exists('~/data/huang_C0/%s.pdb%s.gz'%(pro,ass)):
        continue
    else:
        fo.writelines(i+"\n") '''

for pro in lst:
    # pro = pro.rstrip("\n")
    # pro = pro.rstrip("\r")
    pro=pro.strip()
    os.system('curl -s -f https://files.rcsb.org/download/%s.pdb.gz -o ~/data/ddd_file/%s.pdb.gz' % (pro, pro))
    if os.path.exists('~/data/ddd_file/%s.pdb.gz' % pro):
        continue
    else:
        os.system('curl -s -f https://files.rcsb.org/download/%s.pdb -o ~/data/ddd_file/%s.pdb' % (pro, pro))

# fo.close()
fr.close()
