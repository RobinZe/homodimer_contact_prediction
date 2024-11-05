# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import h5py

f=h5py.File('model/model.h5', 'r')
f_out = open('network_archtecture.txt', 'w')

def  read_h5(file, stri):
    if hasattr(file[stri], 'keys'):
        f.write(file[stri].name + "\n")
        for key in file[stri].keys():
            read_h5(file[stri], key)
    else:
        f.write(file[stri].name + '    ' + str(file[stri].shape) + "\n")

for key in f.keys():
    read_h5(f, key)

f.close()
f_out.close()