# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import numpy as np

try:
    from math import pi, dist, fabs, cos, sqrt, atan2, sin, cos, tan
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt, sin, tan

from robot_tossing_utils.panda_utils import *

# arm_charging_config = [0.0,  pi/4,  0.0, -pi/2,          0.0, 135 * pi / 180, pi/4]
# arm_toss_config =     [0.0, -pi/4,  0.0, -30 * pi / 180, 0.0, 150 * pi / 180, pi/4] # 30° of incline of toss cartesian vel

arm_charging_config = [0.0, pi/4,             0.0,   -pi/2, 0.0, 135 * pi / 180, pi/4]
arm_toss_config =     [0.0, - 25 * pi / 180,  0.0,   -pi/4, 0.0, 180 * pi / 180, pi/4] # 20° of incline of toss cartesian vel

# release_position = [0.3186459511431726, -0.0008579069982752416, 2.2] #cartesian (w.r.t. robot)
release_position = [0.07, 0.0, 2.55]

#pickup_position = [-0.7, 0.0, 1.0]
pickup_position = [-0.8, 0.0, 1.0]

ALPHA = -1 * np.pi / 180

def toss_back_catapult(move_group, v_norm, yaw_target=00, dt=0.001):
    q_launch = copy.deepcopy(arm_toss_config)
    q_charge = copy.deepcopy(arm_charging_config)

    q_launch[0] = q_charge[0] = yaw_target  # to align to frames

    # dq_ref = v_norm * np.array([-0.54647482] + [0.54647482] * 2)
    dq_ref = v_norm * np.array([-0.40308587] + [0.40308587] * 2)

    dq_2 = dq_ref[0]
    dq_4 = dq_ref[1]
    dq_6 = dq_ref[2]

    positions, velocities, accelerations, times, pre_toss_joint_position, trj_time = (
        smooth_trj(q_charge, q_launch, dq_2, dq_4, dq_6, dt, final_damping=True))


    return times, positions, velocities, accelerations, pre_toss_joint_position, q_launch, trj_time
