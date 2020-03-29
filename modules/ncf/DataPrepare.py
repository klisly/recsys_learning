#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 5:11 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : DataLoader.py
import sys

file = '../../datas/info/user_read_docs.csv'
with open(file, encoding='utf8') as f:
    count = 0
    for line in f:
        line = line.strip()
        pts = line.split(",")
        print(pts)
        count += 1
        if count > 10:
            break
