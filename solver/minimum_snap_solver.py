'''
Author: LOTEAT
Date: 2024-08-16 09:50:53
'''

import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import block_diag


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
                    np.array(
                        list(range(2 * self.n_order - 7, self.n_order - 4,
                                   -1))))
            else:
                scale_mat_list.append(scale_mat_list[-1] - 1)
        scale_mat[:self.n_order - 3, :self.n_order -
                  3] = np.array(scale_mat_list)

        return coef_mat, scale_mat

    def solve(self):
        Q = self.get_quadratic_matrix()
        M = self.get_mapping_matrix()
        CT = self.get_selecting_matrix()
        C = CT.T
        R = C @ inv(M).T @ Q @ inv(M) @ (CT)
        R_pp = R[self.n_seg + 7:, self.n_seg + 7:]
        R_fp = R[:self.n_seg + 7, self.n_seg + 7:]
        dF = np.zeros(8 + self.n_seg - 1)
        dF[:4] = self.start_cond
        dF[4:-4] = self.waypoints[1:-1]
        dF[-4:] = self.end_cond
        dP = -pinv(R_pp).dot(R_fp.T).dot(dF)
        d = np.hstack([dF, dP])
        res = inv(M).dot(CT).dot(d)
        return res

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
            poly_coef_pad = np.hstack([poly_coef,
                                       np.array([
                                           0,
                                       ] * (i + 1))])
            t_power -= 1
            # avoid 0^i where i < 0
            t_power[t_power < 0] = 0
            t_value = t**t_power
            motion_coef.append(poly_coef_pad * t_value)
        return np.array(motion_coef)

    def get_mapping_matrix(self):
        M = []
        for i in range(self.n_seg):
            seg_time = self.seg_time[i]
            start_mapping_mat = self.get_motion_coef(0)
            end_mapping_mat = self.get_motion_coef(seg_time)
            seg_M = np.vstack([start_mapping_mat, end_mapping_mat])
            M.append(seg_M)
        return block_diag(*M)

    def get_selecting_matrix(self):
        selecting_mat = []
        n_var = (self.n_seg + 1) * 4
        # start selection
        start_selecting_mat = np.zeros((4, n_var))
        start_selecting_mat[:4, :4] = np.eye(4)
        selecting_mat.append(start_selecting_mat)
        # waypoint selection
        for i in range(self.n_seg - 1):
            seg_end_selecting_mat = np.zeros((4, n_var))
            p_idx = 4 + i
            v_idx = 4 + 4 + self.n_seg - 1
            a_idx = v_idx + 1
            j_idx = a_idx + 1
            seg_end_selecting_mat[0, p_idx] = 1
            seg_end_selecting_mat[1, v_idx] = 1
            seg_end_selecting_mat[2, a_idx] = 1
            seg_end_selecting_mat[3, j_idx] = 1
            seg_start_selecting_mat = seg_end_selecting_mat.copy()
            selecting_mat.extend(
                [seg_end_selecting_mat, seg_start_selecting_mat])
        # end selection
        end_selecting_mat = np.zeros((4, n_var))
        end_selecting_mat[:4, list(range(4 + self.n_seg - 1, 4 + self.n_seg - 1 +
                                 4))] = np.eye(4)
        selecting_mat.append(end_selecting_mat)
        return np.concatenate(selecting_mat, axis=0)
