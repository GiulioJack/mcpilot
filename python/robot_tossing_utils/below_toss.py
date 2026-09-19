# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
from math import pi, dist, fabs, cos, sqrt, atan2, sin, cos, tan

import numpy as np

from robot_tossing_utils.panda_utils import *

correction_angle = atan2(0.088 * 2, upper_arm + 0.384 + 0.107)  # "correction" to the angle of joint 4
arm_charging_config = [0.0, 30 * pi / 180, 0, -145 * pi / 180, 0, 105 * pi / 180, pi / 4]
q_2_post_toss = pi / 4
q_4_post_toss = -pi / 2 - correction_angle  # + 20 * pi / 180
q_6_post_toss = pi
arm_toss_config = [0.0, q_2_post_toss, 0.0, q_4_post_toss, 0.0, q_6_post_toss, pi / 4] # 45° of incline of toss cartesian vel

ALPHA = pi/4 # incline of toss cartesian vel

release_position = [0.7551644570703442, 0.0, 1.107557285230802]
pickup_position = [0.4, 0.4, 1.0]



def toss_from_below(move_group, v_norm, yaw_target, dt=0.001, use_jacobian=True, vel_trj_type='smooth'):
    R = wrist_len  # approx launch with semicircle
    omega = v_norm / R  # angular velocity
    print('Angular Velocity: {}'.format(omega))

    if use_jacobian:
        q_launch = copy.deepcopy(arm_toss_config)
        q_launch[0] = yaw_target  # to align to frames

        J = move_group.get_jacobian_matrix(q_launch, [0.0, 0.0, bullet_grasp_dist])  # velocity at ~gripping location
        # J = move_group.get_jacobian_matrix(q_launch)  # velocity at ~gripping location

        print('\nq: {}'.format(q_launch))

        print('\nJ * dq (only q_4 and q_6 moving max vel)')
        print(np.matmul(J[:3, :], np.array([0.0, 0.0, 0.0, max_dq[3], 0.0, 0.0, 0.0])))

        print('\nPseudoinverse')
        print(np.linalg.pinv(J[:3, :]))

        print('\nPseudoinverse * v')
        # print(np.matmul(np.linalg.pinv(J[:3, :]), np.array([cos(ALPHA) * v_norm, 0.0, sin(ALPHA) * v_norm])))
        dq_ref = np.matmul(np.linalg.pinv(J[:3, :]), np.array([v_norm * cos(ALPHA), 0.0, v_norm * sin(ALPHA)]))
        print(dq_ref)

        # print(np.matmul(np.linalg.pinv(J[:3,:], np.array([cos(ALPHA) * v_norm, 0.0, sin(ALPHA) * v_norm]))))

        A = np.array([[J[0, 3], J[0, 5]], [J[2, 3], J[2, 5]]])

        print('Restrict J: {}'.format(A))
        print('Restrict J^-1 : {}'.format(np.linalg.inv(A)))
        print('Restrict J^-1 * d_x : {}'.format(np.matmul(np.linalg.inv(A), np.array([cos(ALPHA), sin(ALPHA)])) * v_norm))

        dq_2 = dq_ref[1]
        dq_4 = dq_ref[3]
        dq_6 = dq_ref[5]
    else:
        if omega <= max_dq[3]:  # if joint4 can give enough "push"
            # set joint 4 velocity to omega
            dq_4 = omega
            dq_6 = 0.0
        else:
            # set joint 4 velocity to max and joint 6 velocity to omega - max dq joint 4
            if omega - max_dq[3] > max_dq[5]:
                print('#### Required velocity exceeds joints limits ####')
            dq_4 = max_dq[3]
            dq_6 = min(omega - max_dq[3], max_dq[5])  # new required angular velocity for 6th joint

    if vel_trj_type == 'smooth':
        print('Tossing with smooth trajectory')
        positions, velocities, accelerations, times, pre_toss_joint_position, trj_time = (
            smooth_trj(arm_charging_config, arm_toss_config, dq_2, dq_4, dq_6, dt, final_damping=True))
    else:
        # step_trajectory
        print('Tossing with step trajectory')
        positions, velocities, times, pre_toss_joint_position, trj_time = step_trj(arm_charging_config, arm_toss_config,
                                                                         dq_4, dq_6, dt)
        accelerations = None

    arm_charging_config[0] = yaw_target

    return times, positions, velocities, accelerations, arm_charging_config, arm_toss_config, trj_time

