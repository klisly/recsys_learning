#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 2:04 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : sim.user.py
import numpy as np


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def user_sim(vect0, vect1):
    vector0 = [float(item) for item in vect0.split(",")]
    vector1 = [float(item) for item in vect1.split(",")]
    return float(cosine_similarity(vector0, vector1))


emb_file = "emb.txt"
items = dict()
with open(emb_file, encoding="utf-8") as f:
    for line in f:
        ds = line.strip().split("\t")
        items[ds[0]] = ds[1]
with open("sim.user.rate.txt", mode='w', encoding="utf8") as f:
    dts = dict()
    count = 0
    f.write("user1,user2,rate")
    for key1 in items.keys():
        for key2 in items.keys():
            if key1 == key2:
                continue
            rate = 0
            if key2 + ":" + key1 in dts.keys():
                rate = dts[key2 + ":" + key1]
            else:
                rate = user_sim(items[key1], items[key2])
            f.write(key1 + "," + key2 + "," + str(rate) + "\n")
            count += 1
            if count % 10000:
                f.flush()
