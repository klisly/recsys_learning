#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/28 8:47 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : DataLoader.py
import scipy.sparse as sp
import numpy as np


class DataLoader(object):
    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line != None and line != "":
                    arr = line.split("\t")
                    user, item = int(arr[0]), int(arr[1])
                    ratingList.append([user, item])
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line != None and line != "":
                    arr = line.split("\t")
                    negatives = []
                    for x in arr[1:]:
                        negatives.append(int(x))
                    negativeList.append(negatives)
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != None and line != "":
                    arr = line.split("\t")
                    u, i = int(arr[0]), int(arr[1])
                    num_users = max(num_users, u)
                    num_items = max(num_items, i)
        # Construct matrix
        count = 0
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != None and line != "":
                    arr = line.split("\t")
                    user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                    if (rating > 3):
                        mat[user, item] = 1.0
                    count += 1
                    if count > 10000:
                        break

        return mat
