# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi

q_release = [0.0, -25 * pi / 180, 0.0, -pi / 4, 0.0, pi, 0]
q_charge = [0.0, pi / 4, 0.0, -pi / 2, 0.0, 135 * pi / 180, 0]
max_q = [1.7628, -0.0698, 3.7525]
min_q = [-1.7628, -3.0718, -0.0175]
max_dq = [2.1750] * 2 + [2.6100]

def settling_time(delta_q, target_vel):
    """
    :param delta_q: required value for signal at settling time
    :param target_vel: required value for signal velocity at settling time
    :return:
    """
    return delta_q * 12 / (target_vel * 7)


def signal(t, target_vel, delta_q):
    """
        Returns value of the signal (joint position) at time t, such that trajectory gets to desidered postion with
        desired velocity

    :param t: time
    :param a: required velocity at settling time
    :param delta_q: required signal value at settling time
    """
    a = target_vel
    tf = settling_time(delta_q, a)
    return -a * (t ** 4) / (4 * (tf ** 3)) + (a * (t ** 3)) / (3 * (tf ** 2)) + (a * t ** 2) / (2 * tf)


def signal_vel(t, target_vel, delta_q):
    """
        Smooth velocity signal such that it gets to target velocity with acceleration = 0 and its integral is equal to
        delta_q
        :param t: time
        :param a: required velocity at settling time
        :param delta_q: required signal value at settling time
    """
    a = target_vel
    tf = settling_time(delta_q, a)
    return - (a * pow(t, 3)) / pow(tf, 3) + (a * pow(t, 2)) / pow(tf, 2) + (a * t) / tf


def required_delta_q(settling_time, required_vel):
    """
    :param settling_time: fixed settling time to reach required vel at null acceleration
    :param required_vel: required value for signal velocity at settling time
    :return:
    """
    return 7 * settling_time * required_vel / 12


def f_kin(q1, q2, q3, q4, q5, q6, q7):
    """
        With:
        gripper_prosthesis = 0.18
        gripper_x_shift = 0.135
    """
    return np.array([[0.407 * (
            ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
        q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(q1) * sin(
        q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (sin(q1) * cos(
        q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * cos(q6) + 0.223 * (
                              -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
                          q1) * cos(q4)) * sin(q6) - 0.407 * (
                              -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
                          q1) * cos(q4)) * cos(q6) - 0.384 * (
                              -sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) - 0.0825 * (
                              -sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q1) * sin(
        q3) - 0.0825 * sin(q2) * sin(q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4) + 0.316 * sin(q2) * cos(
        q1) + 0.0825 * cos(q1) * cos(q2) * cos(q3)],
                     [0.407 * (((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(
                         q4)) * cos(q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(
                         q6) + 0.223 * (((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(
                         q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(
                         q6) + 0.223 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(
                         q2) * cos(q4)) * sin(q6) - 0.407 * (
                              -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(
                          q2) * cos(q4)) * cos(q6) - 0.384 * (
                              sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) - 0.0825 * (
                              sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) - 0.0825 * sin(q1) * sin(
                         q2) * sin(q4) + 0.384 * sin(q1) * sin(q2) * cos(q4) + 0.316 * sin(q1) * sin(q2) + 0.0825 * sin(
                         q1) * cos(q2) * cos(q3) + 0.0825 * sin(q3) * cos(q1)],
                     [0.407 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
                         q5)) * sin(q6) + 0.223 * (
                              (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(
                          q3) * sin(q5)) * cos(q6) + 0.223 * (
                              sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                              sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * cos(q6) + 0.384 * sin(q2) * sin(
                         q4) * cos(q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q2) * cos(
                         q3) - 0.0825 * sin(q4) * cos(q2) + 0.384 * cos(q2) * cos(q4) + 0.316 * cos(q2) + 0.333]])


def J(q1, q2, q3, q4, q5, q6, q7):
    """
        With:
        gripper_prosthesis = 0.18
        gripper_x_shift = 0.135
    """

    return np.array([
        [-0.407 * (((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) - 0.223 * (
                 ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
             q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) - 0.223 * (
                 -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * sin(
            q6) + 0.407 * (
                 -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * cos(
            q6) + 0.384 * (sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + 0.0825 * (
                 sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + 0.0825 * sin(q1) * sin(q2) * sin(
            q4) - 0.384 * sin(q1) * sin(q2) * cos(q4) - 0.316 * sin(q1) * sin(q2) - 0.0825 * sin(q1) * cos(q2) * cos(
            q3) - 0.0825 * sin(q3) * cos(q1), (0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                       sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * cos(
            q6) + 0.384 * sin(q2) * sin(q4) * cos(q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q2) * cos(
            q3) - 0.0825 * sin(q4) * cos(q2) + 0.384 * cos(q2) * cos(q4) + 0.316 * cos(q2)) * cos(q1), (0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                                                sin(q2) * sin(
                                                                                                            q4) * cos(
                                                                                                            q3) + cos(
                                                                                                            q2) * cos(
                                                                                                            q4)) * cos(
            q6) + 0.384 * sin(q2) * sin(q4) * cos(q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q2) * cos(
            q3) - 0.0825 * sin(q4) * cos(q2) + 0.384 * cos(q2) * cos(q4)) * sin(q1) * sin(q2) - (0.407 * (
                ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((sin(q1) * cos(
            q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(
            q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) + 0.223 * (-(
                sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * sin(
            q6) - 0.407 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(
            q4)) * cos(q6) - 0.384 * (sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) - 0.0825 * (
                                                                                                         sin(q1) * cos(
                                                                                                     q2) * cos(
                                                                                                     q3) + sin(
                                                                                                     q3) * cos(
                                                                                                     q1)) * cos(
            q4) - 0.0825 * sin(q1) * sin(q2) * sin(q4) + 0.384 * sin(q1) * sin(q2) * cos(q4) + 0.0825 * sin(q1) * cos(
            q2) * cos(q3) + 0.0825 * sin(q3) * cos(q1)) * cos(q2), (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * (
                 0.407 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
             q5)) * sin(q6) + 0.223 * (
                         (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
                     q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(
             q6) - 0.407 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * cos(q6) + 0.384 * sin(q2) * sin(
             q4) * cos(q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q4) * cos(q2) + 0.384 * cos(
             q2) * cos(q4)) + (0.407 * (
                ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((sin(q1) * cos(
            q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(
            q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) + 0.223 * (
                                       -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(
                                   q1) * sin(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                       -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(
                                   q1) * sin(q2) * cos(q4)) * cos(q6) - 0.384 * (
                                       sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) - 0.0825 * (
                                       sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(
            q4) - 0.0825 * sin(q1) * sin(q2) * sin(q4) + 0.384 * sin(q1) * sin(q2) * cos(q4)) * sin(q2) * sin(q3),
         (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * (0.407 * (
                 (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
             q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
             q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                                                 sin(q2) * sin(
                                                                                                             q4) * cos(
                                                                                                             q3) + cos(
                                                                                                             q2) * cos(
                                                                                                             q4)) * cos(
             q6)) - (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * (0.407 * (
                 ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
             q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((
                                                                                                              sin(q1) * cos(
                                                                                                          q2) * cos(
                                                                                                          q3) + sin(
                                                                                                          q3) * cos(
                                                                                                          q1)) * cos(
             q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(
             q5)) * cos(q6) + 0.223 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(
             q2) * cos(q4)) * sin(q6) - 0.407 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(
             q1) * sin(q2) * cos(q4)) * cos(q6)), (
                 ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * sin(
             q5) + (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * cos(q5)) * (0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                           sin(q2) * sin(
                                                                                       q4) * cos(q3) + cos(
                                                                                       q2) * cos(q4)) * cos(
            q6)) - ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * sin(q5) - sin(q2) * sin(q3) * cos(q5)) * (
                 0.407 * (((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(
             q4)) * cos(q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((
                                                                                                                         sin(q1) * cos(
                                                                                                                     q2) * cos(
                                                                                                                     q3) + sin(
                                                                                                                     q3) * cos(
                                                                                                                     q1)) * cos(
             q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(
             q5)) * cos(q6) + 0.223 * (
                         -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(
                     q4)) * sin(q6) - 0.407 * (
                         -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(
                     q4)) * cos(q6)), 0],
        [0.407 * (((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (
                 ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
             q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * cos(q6) + 0.223 * (
                 -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(q1) * cos(q4)) * sin(
            q6) - 0.407 * (
                 -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(q1) * cos(q4)) * cos(
            q6) - 0.384 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) - 0.0825 * (
                 -sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q1) * sin(
            q3) - 0.0825 * sin(q2) * sin(q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4) + 0.316 * sin(q2) * cos(
            q1) + 0.0825 * cos(q1) * cos(q2) * cos(q3), (0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                 sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(
                                                             q4)) * cos(q6) + 0.384 * sin(q2) * sin(q4) * cos(
            q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q2) * cos(q3) - 0.0825 * sin(q4) * cos(
            q2) + 0.384 * cos(q2) * cos(q4) + 0.316 * cos(q2)) * sin(q1), -(0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                    sin(q2) * sin(q4) * cos(
                                                                                q3) + cos(q2) * cos(q4)) * cos(
            q6) + 0.384 * sin(q2) * sin(q4) * cos(q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q2) * cos(
            q3) - 0.0825 * sin(q4) * cos(q2) + 0.384 * cos(q2) * cos(q4)) * sin(q2) * cos(q1) + (0.407 * (
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
            q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                            sin(q1) * cos(
                                                                                                        q3) + sin(
                                                                                                        q3) * cos(
                                                                                                        q1) * cos(
                                                                                                        q2)) * sin(
            q5)) * cos(q6) + 0.223 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
            q1) * cos(q4)) * sin(q6) - 0.407 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(
            q2) * cos(q1) * cos(q4)) * cos(q6) - 0.384 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(
            q4) - 0.0825 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q1) * sin(
            q3) - 0.0825 * sin(q2) * sin(q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4) + 0.0825 * cos(q1) * cos(
            q2) * cos(q3)) * cos(q2), -(sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * (0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                                    sin(q2) * sin(
                                                                                                q4) * cos(q3) + cos(
                                                                                                q2) * cos(
                                                                                                q4)) * cos(
            q6) + 0.384 * sin(q2) * sin(q4) * cos(q3) + 0.0825 * sin(q2) * cos(q3) * cos(q4) - 0.0825 * sin(q4) * cos(
            q2) + 0.384 * cos(q2) * cos(q4)) - (0.407 * (
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
            q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                            sin(q1) * cos(
                                                                                                        q3) + sin(
                                                                                                        q3) * cos(
                                                                                                        q1) * cos(
                                                                                                        q2)) * sin(
            q5)) * cos(q6) + 0.223 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
            q1) * cos(q4)) * sin(q6) - 0.407 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(
            q2) * cos(q1) * cos(q4)) * cos(q6) - 0.384 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(
            q4) - 0.0825 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q2) * sin(
            q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4)) * sin(q2) * sin(q3),
         -(-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(q1) * cos(q4)) * (0.407 * (
                 (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
             q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
             q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                                                   sin(q2) * sin(
                                                                                                               q4) * cos(
                                                                                                               q3) + cos(
                                                                                                               q2) * cos(
                                                                                                               q4)) * cos(
             q6)) + (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * (0.407 * (
                 ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
             q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
             q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                             sin(q1) * cos(
                                                                                                         q3) + sin(
                                                                                                         q3) * cos(
                                                                                                         q1) * cos(
                                                                                                         q2)) * sin(
             q5)) * cos(q6) + 0.223 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
             q1) * cos(q4)) * sin(q6) - 0.407 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(
             q2) * cos(q1) * cos(q4)) * cos(q6)), -(
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * sin(
            q5) + (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * cos(q5)) * (0.407 * (
                (-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(q5)) * sin(
            q6) + 0.223 * ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * cos(q5) + sin(q2) * sin(q3) * sin(
            q5)) * cos(q6) + 0.223 * (sin(q2) * sin(q4) * cos(q3) + cos(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                                                          sin(q2) * sin(
                                                                                      q4) * cos(q3) + cos(
                                                                                      q2) * cos(q4)) * cos(
            q6)) + ((-sin(q2) * cos(q3) * cos(q4) + sin(q4) * cos(q2)) * sin(q5) - sin(q2) * sin(q3) * cos(q5)) * (
                 0.407 * (((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(
             q1)) * cos(q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((
                                                                                                                         -sin(
                                                                                                                             q1) * sin(
                                                                                                                     q3) + cos(
                                                                                                                     q1) * cos(
                                                                                                                     q2) * cos(
                                                                                                                     q3)) * cos(
             q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(
             q5)) * cos(q6) + 0.223 * (
                         -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
                     q1) * cos(q4)) * sin(q6) - 0.407 * (
                         -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
                     q1) * cos(q4)) * cos(q6)), 0],
        [0, -(0.407 * (
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
            q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                            sin(q1) * cos(
                                                                                                        q3) + sin(
                                                                                                        q3) * cos(
                                                                                                        q1) * cos(
                                                                                                        q2)) * sin(
            q5)) * cos(q6) + 0.223 * (
                      -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(q1) * cos(
                  q4)) * sin(q6) - 0.407 * (
                      -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(q1) * cos(
                  q4)) * cos(q6) - 0.384 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) - 0.0825 * (
                      -sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q1) * sin(
            q3) - 0.0825 * sin(q2) * sin(q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4) + 0.316 * sin(q2) * cos(
            q1) + 0.0825 * cos(q1) * cos(q2) * cos(q3)) * cos(q1) - (0.407 * (
                ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((sin(q1) * cos(
            q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(
            q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) + 0.223 * (-(
                sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * sin(
            q6) - 0.407 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(
            q4)) * cos(q6) - 0.384 * (sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) - 0.0825 * (
                                                                             sin(q1) * cos(q2) * cos(q3) + sin(
                                                                         q3) * cos(q1)) * cos(q4) - 0.0825 * sin(
            q1) * sin(q2) * sin(q4) + 0.384 * sin(q1) * sin(q2) * cos(q4) + 0.316 * sin(q1) * sin(q2) + 0.0825 * sin(
            q1) * cos(q2) * cos(q3) + 0.0825 * sin(q3) * cos(q1)) * sin(q1), -(0.407 * (
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
            q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                            sin(q1) * cos(
                                                                                                        q3) + sin(
                                                                                                        q3) * cos(
                                                                                                        q1) * cos(
                                                                                                        q2)) * sin(
            q5)) * cos(q6) + 0.223 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
            q1) * cos(q4)) * sin(q6) - 0.407 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(
            q2) * cos(q1) * cos(q4)) * cos(q6) - 0.384 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(
            q4) - 0.0825 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q1) * sin(
            q3) - 0.0825 * sin(q2) * sin(q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4) + 0.0825 * cos(q1) * cos(
            q2) * cos(q3)) * sin(q1) * sin(q2) + (0.407 * (
                ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((sin(q1) * cos(
            q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(
            q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) + 0.223 * (
                                                          -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(
                                                      q4) + sin(q1) * sin(q2) * cos(q4)) * sin(q6) - 0.407 * (
                                                          -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(
                                                      q4) + sin(q1) * sin(q2) * cos(q4)) * cos(q6) - 0.384 * (
                                                          sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(
            q4) - 0.0825 * (sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) - 0.0825 * sin(q1) * sin(
            q2) * sin(q4) + 0.384 * sin(q1) * sin(q2) * cos(q4) + 0.0825 * sin(q1) * cos(q2) * cos(q3) + 0.0825 * sin(
            q3) * cos(q1)) * sin(q2) * cos(q1), (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * (0.407 * (
                ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((sin(q1) * cos(
            q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(
            q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) + 0.223 * (-(
                sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * sin(
            q6) - 0.407 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(
            q4)) * cos(q6) - 0.384 * (sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) - 0.0825 * (
                                                                                                             sin(q1) * cos(
                                                                                                         q2) * cos(
                                                                                                         q3) + sin(
                                                                                                         q3) * cos(
                                                                                                         q1)) * cos(
            q4) - 0.0825 * sin(q1) * sin(q2) * sin(q4) + 0.384 * sin(q1) * sin(q2) * cos(q4)) - (
                 sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * (0.407 * (
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
            q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                            sin(q1) * cos(
                                                                                                        q3) + sin(
                                                                                                        q3) * cos(
                                                                                                        q1) * cos(
                                                                                                        q2)) * sin(
            q5)) * cos(q6) + 0.223 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
            q1) * cos(q4)) * sin(q6) - 0.407 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(
            q2) * cos(q1) * cos(q4)) * cos(q6) - 0.384 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(
            q4) - 0.0825 * (-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) - 0.0825 * sin(q2) * sin(
            q4) * cos(q1) + 0.384 * sin(q2) * cos(q1) * cos(q4)),
         (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(q1) * cos(q4)) * (0.407 * (
                 ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
             q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((
                                                                                                              sin(q1) * cos(
                                                                                                          q2) * cos(
                                                                                                          q3) + sin(
                                                                                                          q3) * cos(
                                                                                                          q1)) * cos(
             q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(
             q5)) * cos(q6) + 0.223 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(
             q2) * cos(q4)) * sin(q6) - 0.407 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(
             q1) * sin(q2) * cos(q4)) * cos(q6)) - (
                 -(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * (
                 0.407 * (((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(
             q1)) * cos(q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((
                                                                                                                         -sin(
                                                                                                                             q1) * sin(
                                                                                                                     q3) + cos(
                                                                                                                     q1) * cos(
                                                                                                                     q2) * cos(
                                                                                                                     q3)) * cos(
             q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(
             q5)) * cos(q6) + 0.223 * (
                         -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
                     q1) * cos(q4)) * sin(q6) - 0.407 * (
                         -(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
                     q1) * cos(q4)) * cos(q6)), (
                 ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * sin(
             q5) + (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * cos(q5)) * (0.407 * (
                ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(
            q5) - (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * sin(q6) + 0.223 * (((sin(q1) * cos(
            q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * cos(q5) - (sin(q1) * sin(
            q3) * cos(q2) - cos(q1) * cos(q3)) * sin(q5)) * cos(q6) + 0.223 * (-(
                sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(q4)) * sin(
            q6) - 0.407 * (-(sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * sin(q4) + sin(q1) * sin(q2) * cos(
            q4)) * cos(q6)) - (
                 ((sin(q1) * cos(q2) * cos(q3) + sin(q3) * cos(q1)) * cos(q4) + sin(q1) * sin(q2) * sin(q4)) * sin(
             q5) + (sin(q1) * sin(q3) * cos(q2) - cos(q1) * cos(q3)) * cos(q5)) * (0.407 * (
                ((-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(
            q5) - (sin(q1) * cos(q3) + sin(q3) * cos(q1) * cos(q2)) * sin(q5)) * sin(q6) + 0.223 * (((-sin(
            q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * cos(q4) + sin(q2) * sin(q4) * cos(q1)) * cos(q5) - (
                                                                                                            sin(q1) * cos(
                                                                                                        q3) + sin(
                                                                                                        q3) * cos(
                                                                                                        q1) * cos(
                                                                                                        q2)) * sin(
            q5)) * cos(q6) + 0.223 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(q2) * cos(
            q1) * cos(q4)) * sin(q6) - 0.407 * (-(-sin(q1) * sin(q3) + cos(q1) * cos(q2) * cos(q3)) * sin(q4) + sin(
            q2) * cos(q1) * cos(q4)) * cos(q6)), 0]])


def get_release_config(v_norm, gamma, t_delay):
    dq_2 = -0.40308587 * abs(v_norm)
    dq_4 = 0.40308587 * abs(v_norm)
    dq_6 = 0.40308587 * abs(v_norm)

    trj_time_q2 = settling_time(q_release[1] - q_charge[1], dq_2)
    trj_time_q4 = settling_time(q_release[3] - q_charge[3], dq_4)
    trj_time_q6 = settling_time(q_release[3] - q_charge[3], dq_6)

    trj_time = max([trj_time_q2, trj_time_q4, trj_time_q6])

    delta_q_6 = q_release[5] - q_charge[5]  # max deltas available
    delta_q_4 = q_release[3] - q_charge[3]
    delta_q_2 = q_release[1] - q_charge[1]

    delta_time_q2 = max(0.0, trj_time - trj_time_q2)
    delta_time_q4 = max(0.0, trj_time - trj_time_q4)
    delta_time_q6 = max(0.0, trj_time - trj_time_q6)

    # print('trj_times:', trj_time_q2, trj_time_q4, trj_time_q6)
    # print('delta_times:', delta_time_q2, delta_time_q4, delta_time_q6)

    q_charge[0] = q_release[0] = gamma  # align yaw angle

    damping = 2
    dt = 0.001  # 10000Hz

    if t_delay >= 0:
        t = 0
        q = q_release
        dq = None

        while t <= t_delay:
            dq = [0.0, signal_vel(trj_time - damping * t - delta_time_q2, dq_2, delta_q_2) * (trj_time - damping * t > delta_time_q2),
                  0.0, signal_vel(trj_time - damping * t - delta_time_q4, dq_4, delta_q_4) * (trj_time - damping * t > delta_time_q4),
                  0.0, signal_vel(trj_time - damping * t - delta_time_q6, dq_6, delta_q_6) * (trj_time - damping * t > delta_time_q6),
                  0.0]

            q = [q[0], q[1] + dq[1] * dt,
                 q[2], q[3] + dq[3] * dt,
                 q[4], q[5] + dq[5] * dt,
                 q[6]]

            t += dt

        # while t <= t_delay:
        #     dq = [0.0, signal_vel(trj_time - t - delta_time_q2, dq_2, delta_q_2) * (trj_time - t > delta_time_q2),
        #           0.0, signal_vel(trj_time - t - delta_time_q4, dq_4, delta_q_4) * (trj_time - t > delta_time_q4),
        #           0.0, signal_vel(trj_time - t - delta_time_q6, dq_6, delta_q_6) * (trj_time - t > delta_time_q6),
        #           0.0]
        #
        #     q = [q[0], q[1] + dq[1] * dt,
        #          q[2], q[3] + dq[3] * dt,
        #          q[4], q[5] + dq[5] * dt,
        #          q[6]]
        #
        #     t += damping * dt
        #     t += dt
        #
        # print(q, dq)

    else:
        # t = 0
        # q = q_charge
        # dq = [0.0] * 7

        t_rel = trj_time + t_delay  # t_delay is <0

        # while t <= t_rel:
        #     dq = [0.0, signal_vel(t - delta_time_q2, dq_2, delta_q_2) * (t > delta_time_q2),
        #           0.0, signal_vel(t - delta_time_q4, dq_4, delta_q_4) * (t > delta_time_q4),
        #           0.0, signal_vel(t - delta_time_q6, dq_6, delta_q_6) * (t > delta_time_q6),
        #           0.0]
        #     q = [q[0], q[1] + dq[1] * dt,
        #          q[2], q[3] + dq[3] * dt,
        #          q[4], q[5] + dq[5] * dt,
        #          q[6]]
        #     t += dt
        #
        # print(q, dq)

        q = q_charge

        q = [q[0], q[1] + signal(t_rel - delta_time_q2, dq_2, delta_q_2) * (t_rel > delta_time_q2),
             q[2], q[3] + signal(t_rel - delta_time_q4, dq_4, delta_q_4) * (t_rel > delta_time_q4),
             q[4], q[5] + signal(t_rel - delta_time_q6, dq_6, delta_q_6) * (t_rel > delta_time_q6),
             q[6]]

        dq = [0.0, signal_vel(t_rel - delta_time_q2, dq_2, delta_q_2) * (t_rel > delta_time_q2),
              0.0, signal_vel(t_rel - delta_time_q4, dq_4, delta_q_4) * (t_rel > delta_time_q4),
              0.0, signal_vel(t_rel - delta_time_q6, dq_6, delta_q_6) * (t_rel > delta_time_q6),
              0.0]

        # print(q, dq)

    return q, dq


def get_initial_condition(v_norm, gamma, t_delay):
    q, dq = get_release_config(v_norm, gamma, t_delay)

    J_rel = J(*q)

    v_rel = np.dot(J_rel, np.array(dq))
    v_rel[0] = - v_rel[0]  # Rotate the reference frame
    v_rel[1] = - v_rel[1]

    p_rel = f_kin(*q)
    p_rel[0, 0] = - p_rel[0, 0]
    p_rel[1, 0] = - p_rel[1, 0]

    return p_rel, v_rel, np.linalg.norm(v_rel)

def tossing_init(v_norm, yaw, t_delay):
    """
    Return the delayed release state in the Panda base frame.

    The x-y components are already rotated to the frame used by the
    backward tossing experiment. The z component is still relative to
    the Panda base and therefore must not be compared directly with
    Gazebo world-frame trajectories when the robot is spawned above the
    ground.
    """
    p_rel, v_rel, _ = get_initial_condition(v_norm, yaw, t_delay)

    x_0 = list(p_rel[:, 0])  # column to row
    x_0_dot = list(v_rel)
    return x_0 + x_0_dot


RELEASE_RADIAL_VELOCITY_COMPENSATION_COEFFICIENTS = (
    0.0029248370612070633,
    -0.0010445877502908132,
    0.0024409739564510468,
)


def release_radial_velocity_compensation(v_norm, coefficients=None):
    """Return the calibrated outward radial-velocity correction [m/s].

    The default cubic-through-the-origin model was fitted to the 40
    exploration releases (five per seed) from ``100_energy_score`` through
    ``107_energy_score``.  Coefficients are ordered by increasing power, so
    the returned value is ``c1*v + c2*v**2 + c3*v**3``.
    """
    if coefficients is None:
        coefficients = RELEASE_RADIAL_VELOCITY_COMPENSATION_COEFFICIENTS
    speed = float(v_norm)
    return sum(float(coefficient) * speed ** power
               for power, coefficient in enumerate(coefficients, start=1))


def tossing_init_gazebo(v_norm, yaw, t_delay, base_z=1.03,
                        compensate_radial_velocity=False):
    """Return the delayed release state in the Gazebo world frame.

    The tossing launch files spawn the Panda with ``z=1.03``. Recorded
    projectile trajectories are consequently expressed in the Gazebo
    world frame, while :func:`tossing_init` returns a state relative to
    the robot base. This helper applies the missing vertical translation
    while preserving the existing x-y frame rotation.

    Parameters
    ----------
    v_norm : float
        Commanded Cartesian release-speed magnitude.
    yaw : float
        Throwing direction around the world z axis.
    t_delay : float
        Delay between the nominal and actual release instants.
    base_z : float, optional
        Panda-base height used by the Gazebo launch file.
    compensate_radial_velocity : bool, optional
        Add the calibrated speed-dependent outward radial velocity.  Disabled
        by default to preserve historical experiment behavior.
    """
    state = tossing_init(v_norm, yaw, t_delay)
    state[2] += float(base_z)
    if compensate_radial_velocity:
        delta_v = release_radial_velocity_compensation(v_norm)
        state[3] += delta_v * np.cos(float(yaw))
        state[4] += delta_v * np.sin(float(yaw))
    return state


def get_profiles(v_norm):
    dq_2 = -0.40308587 * abs(v_norm)
    dq_4 = 0.40308587 * abs(v_norm)
    dq_6 = 0.40308587 * abs(v_norm)

    trj_time_q2 = settling_time(q_release[1] - q_charge[1], dq_2)
    trj_time_q4 = settling_time(q_release[3] - q_charge[3], dq_4)
    trj_time_q6 = settling_time(q_release[3] - q_charge[3], dq_6)

    trj_time = max([trj_time_q2, trj_time_q4, trj_time_q6])

    delta_q_6 = q_release[5] - q_charge[5]  # max deltas available
    delta_q_4 = q_release[3] - q_charge[3]
    delta_q_2 = q_release[1] - q_charge[1]

    delta_time_q2 = max(0.0, trj_time - trj_time_q2)
    delta_time_q4 = max(0.0, trj_time - trj_time_q4)
    delta_time_q6 = max(0.0, trj_time - trj_time_q6)

    times = np.linspace(0.0, trj_time, math.ceil(trj_time * 1000))
    dt = 0.001

    profiles_dq = [
        [signal_vel(t - delta_time_q2, dq_2, delta_q_2) * int(t >= delta_time_q2) for t in times],
        [signal_vel(t - delta_time_q4, dq_4, delta_q_4) * int(t >= delta_time_q4) for t in times],
        [signal_vel(t - delta_time_q6, dq_6, delta_q_6) * int(t >= delta_time_q6) for t in times]
    ]

    for j in range(len(profiles_dq)):
        profiles_dq[j] = profiles_dq[j] + profiles_dq[j][::-2]

    times = np.linspace(0.0, trj_time * 1.5, math.ceil(trj_time * 1.5 * 1000))

    profiles_dq = np.array(profiles_dq)
    profiles_q = [[q_charge[1]], [q_charge[3]], [q_charge[5]]]
    for j in range(len(profiles_dq)):
        for k in range(profiles_dq.shape[1]):
            profiles_q[j].append(profiles_q[j][-1] + dt * profiles_dq[j, k])
    profiles_q = np.array(profiles_q)

    return profiles_q, profiles_dq, times, trj_time
