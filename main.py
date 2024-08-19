'''
Author: LOTEAT
Date: 2024-08-16 09:17:34
'''
import argparse
from matplotlib import pyplot as plt
import numpy as np
from tools.time_allocation import time_allocation
from solver.minimum_snap_solver import MinimumSnapSolver
from tools.plot import plot_motion


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters of minimum snap.')
    parser.add_argument('-n',
                        type=int,
                        default=8,
                        help='the number of waypoints')
    parser.add_argument('-t',
                        type=float,
                        default=8.0,
                        help='the sum of the time in seconds')
    parser.add_argument('-n_order',
                        type=float,
                        default=7,
                        help='polynomial order')
    parser.add_argument('-alloc',
                        type=str,
                        default='average',
                        choices=['average', 'proportion'],
                        help='the sum of the time in seconds')
    parser.add_argument('--show',
                        action='store_true',
                        help='whether to show')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # limit the x y coordinates
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # input n points
    waypoints = plt.ginput(args.n, show_clicks=True)
    plt.close()
    waypoints = np.array(waypoints)
    # allocate time
    seg_time = time_allocation(waypoints, args.t, args.alloc)
    x_solver = MinimumSnapSolver(args.n_order, waypoints[:, 0], seg_time)
    x_coefs = x_solver.solve()
    y_solver = MinimumSnapSolver(args.n_order, waypoints[:, 1], seg_time)
    y_coefs = y_solver.solve()
    plot_motion(x_coefs, y_coefs, seg_time, waypoints, args.show)
