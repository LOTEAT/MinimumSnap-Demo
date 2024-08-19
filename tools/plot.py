'''
Author: LOTEAT
Date: 2024-07-31 15:47:31
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import threading
import numpy as np


def plot_motion(x_coefs, y_coefs, seg_time, waypoints, is_show=True):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    robot, = ax.plot([], [], 'bo', label='Robot')
    connections, = ax.plot([], [], 'k-')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', label='Waypoints')
    idx = 0
    trajectory = []
    for time_stamp in np.arange(0, np.sum(seg_time), 0.05):
        past_time = np.sum(seg_time[:idx])
        time_upper = np.sum(seg_time[:idx + 1])
        if time_stamp < time_upper:
            time = time_stamp - past_time
            x_coef = x_coefs[8*idx: 8*(idx + 1)]
            y_coef = y_coefs[8*idx: 8*(idx + 1)]
            trajectory.append(dict(
                time = time,
                x_coef = x_coef,
                y_coef = y_coef
            ))
        else:
            idx += 1
        x_trajectory = []
        y_trajectory = []
    

    def init():
        robot.set_data([], [])
        connections.set_data(x_trajectory, y_trajectory)
        return [robot, connections]

    def update(frame):
        x_coef = frame['x_coef']
        y_coef = frame['y_coef']
        time = frame['time']
        x_poly_func = np.poly1d(x_coef)
        y_poly_func = np.poly1d(y_coef)
        robot_x = x_poly_func(time)
        robot_y = y_poly_func(time)
        robot.set_data(robot_x, robot_y)
        x_trajectory.append(robot_x)
        y_trajectory.append(robot_y)
        connections.set_data(x_trajectory, y_trajectory)
        if len(y_trajectory) == len(trajectory):
            threading.Timer(1, plt.close, [fig]).start()
        return [robot, connections]

    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=trajectory,
                                  init_func=init,
                                  blit=True,
                                  interval=50,
                                  repeat=False)
    ax.legend()
    writer = PillowWriter(fps=20)
    if is_show:
        plt.show()
    else:
        ani.save("motion.gif", writer=writer)
