'''
Author: LOTEAT
Date: 2024-08-16 09:43:05
'''
import numpy as np
def cal_dist(p1, p2):
    diff = p2 - p1
    diff = np.square(diff)
    dist = np.sqrt(np.sum(diff))
    return dist
    
    