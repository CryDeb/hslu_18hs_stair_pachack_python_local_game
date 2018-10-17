import multiprocessing
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool, TimeoutError
import time
import os

def f(v,t):
    print(v)
    time.sleep(v)
    return t*t

def merge_names_unpack(args):
    print args[1]
    return args

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == '__main__':
    pool = Pool(processes=4)              # start 4 worker processes
    print(range(10))
    # print "[0, 1, 4,..., 81]"
    values = (1,2,3,4,5,6,7,8)
    results = pool.map_async(merge_names_unpack, [(1, x) for x in range(10)])
    a = results.get(10)
    print a[0]