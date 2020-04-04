#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 2:04 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : sim.user.py
import numpy as np
import faiss
import sys


def load_emb(file):
    ret = list()
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                ret.append([float(item) for item in line.split("\t")[1].split(",")])
    return ret


def load_uid_map(file):
    ret = dict()
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                ps = line.split("\t")
                ret[int(ps[1])] = ps[0]
    return ret


sim_size = 50
nlist = 100
m = 8
k = 4
d = 64
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
uid_map_file = "/Users/wizardholy/project/recsys_learning/datas/info/info.uid.map.txt"
emb_file = "/Users/wizardholy/project/recsys_learning/emb.txt"
out_file = "/Users/wizardholy/project/recsys_learning/sim_user.txt"
print(sys.argv)
if len(sys.argv) > 4:
    d = int(sys.argv[1])
    uid_map_file = sys.argv[2]
    emb_file = sys.argv[3]
    out_file = sys.argv[4]

print(d, uid_map_file, emb_file, out_file)

uid_map = load_uid_map(uid_map_file)
df = load_emb(emb_file)
df = np.array(df).astype('float32')
index.train(df)
index.add(df)
D, I = index.search(df, sim_size + 1)
sim = I.tolist()
weights = D.tolist()
print("len sim", len(sim))
with open(out_file, mode="w") as f:
    count = 0
    for i in range(len(sim)):
        uid = uid_map[i]
        outs = []
        for j in range(1, len(sim[i])):
            try:
                suid = uid_map[sim[i][j]]
                weight = weights[i][j]
                outs.append("" + suid + "#" + str(weight))
            except:
                print(suid, i, j)
                pass
        f.write(uid + " " + (",".join(outs)) + "\n")
        count += 1
    print("total user:", count)
