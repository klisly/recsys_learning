#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 5:11 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : DataLoader.py
import sys
import math
import random

uids_idx = dict()
items_idx = dict()
uid_count = 0
item_count = 0
train_datas = list()
test_datas = list()
uid_itemids = dict()

file = '../../datas/info/user_read_docs.csv'
with open(file, encoding='utf8') as f:
    count = 0
    item = None
    for line in f:
        line = line.strip()
        pts = line.split(",")
        uid = pts[0]
        uid_idx = uid_count
        if uid in uids_idx.keys():
            uid_idx = uids_idx[uid]
        else:
            uids_idx[uid] = uid_idx
            uid_count += 1
        item_id = pts[1]
        item_idx = item_count
        if item_id in items_idx.keys():
            item_idx = items_idx[item_id]
        else:
            items_idx[item_id] = item_idx
            item_count += 1
        if uid_idx not in uid_itemids.keys():
            uid_itemids[uid_idx] = set()

        uid_itemids[uid_idx].add(item_idx)
        report = int(pts[2])
        try:
            rating = int(int(math.ceil(int(pts[3]) * 10 / (float(pts[-3]) * 8))))
        except:
            rating = 1
            print("error", line)
        if rating > 10:
            rating = 10
        new_item = [uid_idx, item_idx, rating, report]
        if item:
            if item and item[0] == new_item[0]:
                train_datas.append(item)
            else:
                test_datas.append(item)
        item = new_item
        if count % 1000 == 0:
            print(count)
        count += 1

test_datas.append(item)
with open("../../datas/info/info.train.rating", encoding="utf8", mode="w") as f:
    for item in train_datas:
        out = str(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\t" + str(item[3])
        f.write(out + "\n")

with open("../../datas/info/info.test.rating", encoding="utf8", mode="w") as f:
    with open("../../datas/info/info.test.negative", encoding="utf8", mode="w") as nf:
        for item in test_datas[:6000]:
            line = str(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\t" + str(item[3])
            f.write(line + "\n")
            nge = '(' + str(item[0]) + ',' + str(item[1]) + ')'
            itemset = uid_itemids[item[0]]
            for i in range(1, 100):
                idx = random.randint(0, item_count - 1)
                while idx in itemset:
                    idx = random.randint(1, item_count - 1)
                nge += "\t" + str(idx)
                itemset.add(idx)
            nf.write(nge.strip() + "\n")

with open("../../datas/info/info.uid.map.txt", encoding="utf8", mode="w") as f:
    for item in uids_idx.keys():
        f.write(item + "\t" + str(uids_idx[item]) + "\n")

with open("../../datas/info/info.item.map.txt", encoding="utf8", mode="w") as f:
    for item in items_idx.keys():
        f.write(str(item) + "\t" + str(items_idx[item]) + "\n")
