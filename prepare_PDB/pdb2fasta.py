# !/usr/bin/env_python3
#-*- coding:utf-8 -*-
__author__ = 'Robin'

import sys

tran_name = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
             'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'PRO': 'P',
             'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q', 'LYS': 'K',
             'ARG': 'R', 'HIS': 'H', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',

             'ASX': 'N', 'GLX': 'Q', 'UNK': 'G', 'HSD': 'H'}

if len(sys.argv) != 2:
    print('Usage :  Python pdb2fasta.py list_filename')
    exit(1)

# Extract sequence from ATOM lines
''' for file in sys.argv[1:]:
    if file[-4:]!='.pdb':
        file = file+'.pdb'
    f=open(file,'r')
    f_line=f.readlines()
    seq, cha, at_ta = ['0'],['0'], '0'
    for atom in f_line:
        if f_line[:4]=='ATOM':
            atom_num=f_line[22:26].replace(' ','')
            if f_line[21] != cha[-1]:
                cha.append(f_line[21])
                seq.append('')
            if atom_num != at_na:
                at_na = atom_num
                seq[-1] += tran_name[f_line[17:20]]
    seq.pop(0)
    cha.pop(0)
    f.close() '''

base_path = '/public/home/jiangyz/dataset/transform_c2/'
list_file = open(base_path + sys.argv[1], 'r')
names = list_file.readlines()
assem_file = open(base_path + 'surprise_assembly.lst', 'w')


def diff_write(chains, coord, counr, counn):
    f_tran = open(base_path + name + '/' + name + '_tran.txt', 'w')
    print(chains, counr, counn)
    if len(chains) == 1 and counn[-1] == 0:
        assem_file.writelines(name + '-' + chains[0] + "\n")
    elif len(chains) == 1 and len(coord) == 6 and counn[-1] in (1, 2, 3):
        assem_file.writelines(name + '_' + chains[0] + "\n")
        f_tran.writelines(chains[0] + "\n")
        for row_tran in coord:
            f_tran.writelines(row_tran + "\n")
    elif len(chains) == 2:
        assem_file.writelines(name + '-' + chains[0][0] + '_' + chains[1][0] + "\n")
        f_tran.writelines(chains[0][0] + "\n")
        for row_tran in coord[:counr[1]]:
            f_tran.writelines(row_tran + "\n")
        f_tran.writelines("\n" + chains[1][0] + "\n")
        for row_tran in coord[counr[1]:counr[2]]:
            f_tran.writelines(row_tran + "\n")
    else:
        print(name, 'with strange coordinates!')
    f_tran.close()


# extract SEQRES and ATOM lines for assembly chains
for name in names:
    name = name[:4]
    try:
        pdb_file = open(base_path + name + '/' + name + '.pdb', 'r')
    except FileNotFoundError:
        print('No PDB file for', name, '!')
        continue
    pdb_content = pdb_file.readlines()
    # out_seq = dict()
    # out_pdb = open(name + '/' + name + '_assembly.pdb', 'w')

    chains, biom, dime, sfc = [], False, False, False
    for pdb_row in pdb_content:
        if pdb_row[:10] == 'REMARK 350':
            if 'BIOMOLECULE' in pdb_row:
                chains, coord, counr, counn = [], [], [], []
                biom = True
            elif pdb_row[10:].strip().replace(' ', '') == '':
                biom = False
                dime = False
                if chains:
                    counr.append(nrow)
                    counn.append(n_noneval)
                    diff_write(chains, coord, counr, counn)
            elif biom and 'DIMERIC' in pdb_row:
                dime = True
                biom = False
                nrow, n_noneval = 0, 0
            elif dime:
                if 'APPLY THE FOLLOWING' in pdb_row:
                    pdb_row = pdb_row.strip()
                    chain = pdb_row.split(':')[-1]
                    chain = chain.replace(',', '')
                    chain = chain.replace(' ', '')
                    chains.append(chain)
                    counr.append(nrow)
                    counn.append(n_noneval)
                elif 'BIOMT' in pdb_row[10:20]:
                    nrow += 1
                    n_noneval += 1
                    tran_str = pdb_row[24:].strip()
                    tran_num = list(map(eval, tran_str.split()))
                    coord.append(tran_str)
                    if set(tran_num) == {0., 1.}:
                        n_noneval -= 1
        elif dime and chains:
            counr.append(nrow)
            counn.append(n_noneval)
            diff_write(chains, coord, counr, counn)
            break
        ''' if sfc and pdb_row[:6] == 'REMARK' and len(pdb_row.strip().split()) == 2:
            if nrow == 0:
                assem_file.writelines(name + '-' + chains + "\n")
            elif nrow > 0:
                assem_file.writelines(name + '_' + chains + "\n")
            for chain in chains:
                out_seq[chain] = ''
            sfc = False
        if pdb_row[:10] == 'REMARK 350':
            if sfc and 'BIOMOLECULE' in pdb_row:
                if nrow == 0:
                    assem_file.writelines(name + '-' + chains + "\n")
                elif nrow > 0:
                    assem_file.writelines(name + '_' + chains + "\n")
                for chain in chains:
                    out_seq[chain] = ''
                sfc = False
                chains = ''
            if 'DIMERIC' in pdb_row:
                dimer = True
                nrow = 0
            if dimer and 'BURIED SURFACE' in pdb_row:
                dimer = False
                dr = eval(pdb_row.split()[-2])
                if dr > 999.999:
                    sfc = True
                else:
                    continue
            if sfc and 'BIOMT' in pdb_row[10:20]:
                nrow += 1
                tran_str, tran_num = pdb_row[24:].split(), []
                for ts in tran_str:
                    tran_num.append(eval(ts))
                if set(tran_num) == {0., 1.}:
                    nrow -= 1
            if sfc and 'APPLY THE FOLLOWING' in pdb_row:
                chains = pdb_row.split(':')[-1]
                chains = chains.strip()
                chains = chains.replace(',', '')
                chains = chains.replace(' ', '')

        if pdb_row[:6] == 'SEQRES' and pdb_row[11] in out_seq:
            chain = pdb_row[11]
            seq = pdb_row[17:].strip().split()
            for s in seq:
                try:
                    out_seq[chain] += tran_name[s]
                except KeyError:
                    out_seq[chain] += 'X'
            break '''

    pdb_file.close()
    ''' for chain in out_seq:
        out_fasta = open(name + '/' + name + '_%s.fasta' % chain, 'w')
        out_fasta.writelines(">%s || Chain %s \n" % (name, chain))
        out_fasta.writelines(out_seq[chain] + "\n")
        out_fasta.close() '''

list_file.close()
assem_file.close()
# 如何选择有限'_'的list
# 把seq和pure_pdb提取出来
# write fasta files for 60 residues per line
''' f = open(file[:-4] + '.seq', 'w')
boo_chain = False
f.write('>' + file + "\n")
if len(seq)>1:
    boo_chain=True
for i in len(cha):
    output=seq[i]
    if boo_chain:
        f.write('Chain' + cha[i] + "\n")
    while len(output)>60:
        f.write(output[:60] + "\n")
        [ output.pop(0) for _ in range(60) ]
    f.write(output + "\n") '''
