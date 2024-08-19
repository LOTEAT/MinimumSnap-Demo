'''
Author: LOTEAT
Date: 2024-08-16 09:42:44
'''
import numpy as np
from .distance import cal_dist


def time_allocation(waypoints: np.ndarray, time: float,
                    allocation_type: str) -> np.ndarray:
    """allocate time
    
    Args:
        waypoints (np.ndarray): nodes that the robot passes through
        time (float): total time
        allocation_type (str): allocation type

    Returns:
        np.ndarray: time for each segment
    """
    n_seg = waypoints.shape[0] - 1
    seg_time = np.zeros(n_seg)
    if allocation_type == 'average':
        seg_time[:] = time / n_seg
    else:
        dist = np.zeros(n_seg)
        for i in range(n_seg):
            dist[i] = cal_dist(waypoints[i], waypoints[i + 1])
        dist_sum = np.sum(dist)
        seg_time = dist / dist_sum * time
    return seg_time
