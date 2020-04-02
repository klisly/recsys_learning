#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/1 5:55 下午
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : pool.txt.py
import os
from multiprocessing import Process, Pool
import time

def func(n):
    return n ** 2


if __name__ == '__main__':
    pool = Pool()
    # obj_lst = []
    # for i in range(6):
    #     p_obj = pool.apply_async(func, args=(i,))  # 异步执行进程
    #     obj_lst.append(p_obj)
    # pool.close()  # 不再向进程池提交新的任务了
    # pool.join()  # 进程池中的进程都执行完了
    # print([p_obj.get() for p_obj in obj_lst])
    num = 1000
    start_pool_time = time.time()
    pool = Pool(processes=5)
    pool.map(func, range(num))  # map是异步执行的，并且自带close和join
    pool.close()
    pool.join()
    print("通过进程池执行的时间:", time.time() - start_pool_time)

    std_start_time = time.time()
    for i in range(num):
        pass
    print("正常执行的执行时间:", time.time() - std_start_time)

    pro_start_time = time.time()
    p_lst = []
    for i in range(num):
        p = Process(target=func, args=(i,))
        p.start()
        p_lst.append(p)

    [pp.join() for pp in p_lst]
    print("多进程的执行时间:", time.time() - pro_start_time)
