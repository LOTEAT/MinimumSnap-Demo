'''
Author: LOTEAT
Date: 2024-08-16 09:50:53
'''

import numpy as np
from scipy.linalg import block_diag
from qpsolvers import solve_qp


class MinimumSnapSolver:
    """ Minimum Snap Solver

    This solver optimizes polynomial coefficients.
    """

    def __init__(self, n_order, waypoints, seg_time):
        # start condition: p=start, v=0, a=0, j=0
        self.start_cond = np.zeros(4)
        self.start_cond[0] = waypoints[0]
        # end condition: p=end, v=0, a=0, j=0
        self.end_cond = np.zeros(4)
        self.end_cond[0] = waypoints[-1]
        self.n_order = n_order
        self.n_poly = n_order + 1
        self.seg_time = seg_time
        self.n_seg = waypoints.shape[0] - 1
        self.waypoints = waypoints
        # self.coef_mat, self.scale_mat are used to calculate Q matrix
        self.coef_mat, self.scale_mat = self.initialize()

    def initialize(self):
        coef_func = lambda x: x * (x - 1) * (x - 2) * (x - 3)
        coef_mat = np.zeros((self.n_order + 1, self.n_order + 1))
        scale_mat = np.ones((self.n_order + 1, self.n_order + 1))

        coef_vec1 = np.array(
            [coef_func(i) for i in range(self.n_order, 3, -1)]).reshape(-1, 1)
        coef_vec2 = np.array(
            [coef_func(i) for i in range(self.n_order, 3, -1)]).reshape(1, -1)
        coef_mat[:self.n_order - 3, :self.n_order - 3] = coef_vec1 @ coef_vec2

        scale_mat_list = []
        for _ in range(self.n_order - 3):
            if len(scale_mat_list) == 0:
                scale_mat_list.append(
                    np.array(list(range(2 * self.n_order - 7, self.n_order - 4, -1))))
            else:
                scale_mat_list.append(scale_mat_list[-1] - 1)
        scale_mat[:self.n_order - 3, :self.n_order - 3] = np.array(scale_mat_list)

        return coef_mat, scale_mat

    def solve(self):
        Q = self.get_quadratic_matrix()
        Aeq, beq = self.get_constraints()
        q = np.zeros(Q.shape[0])
        x = solve_qp(Q, q, None, None, Aeq, beq, solver='cvxopt')
        return x

    def get_quadratic_matrix(self):
        Q = []
        for i in range(self.n_seg):
            seg_time = self.seg_time[i]
            T_mat = seg_time**self.scale_mat
            seg_Q = self.coef_mat * T_mat / self.scale_mat
            Q.append(seg_Q)
        return block_diag(*Q)

    def get_motion_coef(self, t):
        motion_coef = []
        poly_coef = np.ones(self.n_order + 1)
        t_power = np.arange(self.n_order, -1, -1)
        t_value = t**t_power
        motion_coef.append(poly_coef * t_value)
        for i in range(3):
            poly_coef = np.poly1d(poly_coef).deriv().coeffs
            poly_coef_pad = np.hstack([poly_coef, np.array([0, ] * (i + 1))])
            t_power -= 1
            # avoid 0^i where i < 0
            t_power[t_power < 0] = 0
            t_value = t**t_power
            motion_coef.append(poly_coef_pad * t_value)
        return np.array(motion_coef)

    def get_constraints(self):
        Aeq = []
        beq = []
        # start condition
        n_var = self.n_seg * (self.n_order + 1)
        Aeq_start = np.zeros((4, n_var))
        Aeq_start[:4, :self.n_poly] = self.get_motion_coef(0)
        beq_start = self.start_cond
        Aeq.append(Aeq_start)
        beq.append(beq_start)
        # end condition
        Aeq_end = np.zeros((4, n_var))
        Aeq_end[:4, -self.n_poly:] = self.get_motion_coef(self.seg_time[-1])
        beq_end = self.end_cond
        Aeq.append(Aeq_end)
        beq.append(beq_end)

        # position constraints
        Aeq_pos = np.zeros((self.n_seg - 1, n_var))
        beq_pos = np.zeros(self.n_seg - 1)
        for i in range(self.n_seg - 1):
            Aeq_pos[i, self.n_poly * i:self.n_poly * (i + 1)] = self.get_motion_coef(self.seg_time[i])[0]
            beq_pos[i] = self.waypoints[i + 1]
        Aeq.append(Aeq_pos)
        beq.append(beq_pos)

        # continuity constraints
        # p v a j
        Aeq_continue = np.zeros((4 * (self.n_seg - 1), n_var))
        beq_continue = np.zeros(4 * (self.n_seg - 1))
        for i in range(self.n_seg - 1):
            # the end of last segment
            last_coef_mat_end = self.get_motion_coef(self.seg_time[i])
            # the start of the next segment
            next_coef_mat_start = self.get_motion_coef(0)
            Aeq_continue[4 * i:4 * (i + 1),
                         self.n_poly * i:self.n_poly * (i + 1)] = last_coef_mat_end
            Aeq_continue[4 * i:4 * (i + 1),
                         self.n_poly * (i + 1):self.n_poly * (i + 2)] = -next_coef_mat_start
        Aeq.append(Aeq_continue)
        beq.append(beq_continue)
        Aeq = np.concatenate(Aeq)
        beq = np.concatenate(beq)
        return Aeq, beq
