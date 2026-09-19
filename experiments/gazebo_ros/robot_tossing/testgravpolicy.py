#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
import argparse
import json
import os
from distutils.util import strtobool

import rospy
import geometry_msgs.msg
import torch
import math
import numpy
import numpy as np
import pickle as pkl

from matplotlib import pyplot as plt

import policy_learning.Policy_Tossing as Policy
from mcpilot.srv import TossExperiment
from nav_msgs.msg import Path, Odometry
import tossing_utils

"""
    
    Notes:
    
    Seed 100 - simulation without friction, policy without friction
    
"""

bullet_name = 'red_ball_friction'
seed = 10
v_mult = 1.0
model_drag = False
max_target_dist = 0.5
target_radius = 0.05
load_poly_file = 'toss_exp_results/polyfit_toss_noise.pkl'
start_idx_test_trials = 0
num_test_trials = 10
evaluation_seed = -1
num_toss_test = 100
output_dir = ''
release_z_world = 2.53
min_target_dist = 0.75
flg_world_radius_targets = False
release_anticipation_seconds = 0.0
physical_delay_min_ms = 0.0
physical_delay_max_ms = 0.0

p = argparse.ArgumentParser('test Tossing w/ Panda Robot')
p.add_argument('-seed', type=int, default=10, help='seed')
p.add_argument('-target_radius', type=float, default=0.05, help='target radius in test')
p.add_argument('-bullet_name', type=str, default='red_ball_friction', help='name of the bullet in simulation')
p.add_argument('-load_poly_file', type=str, default='no_file',
               help='name of the file where the coefficients for the poly are saved')
p.add_argument('-start_idx_test_trials', type=int, default=0, help='index of first test trials')
p.add_argument('-num_test_trials', type=int, default=10, help='index of first test trials')
p.add_argument('-num_toss_test', type=int, default=100,
               help='number of analytical-policy evaluation throws per test trial')
p.add_argument('-evaluation_seed', type=int, default=-1,
               help='Base paired-evaluation seed. Negative preserves the legacy seed+513 convention.')
p.add_argument('-output_dir', type=str, default='',
               help='Optional output directory. Empty preserves results_tmp/<seed>.')
p.add_argument('-release_z_world', type=float, default=2.53,
               help='Analytical world-frame release height [m].')
p.add_argument('-min_target_dist', type=float, default=0.75,
               help='Minimum world-frame target radius [m].')
p.add_argument('-max_target_dist', type=float, default=0.5,
               help='Width of the world-frame target-radius interval [m].')
p.add_argument('-flg_world_radius_targets', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='Interpret min/max target distances as world radii and subtract the release radius before tossing_init. Enable for the current MC-PILCO protocol.')
p.add_argument('-model_drag', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true uses stokes drag model in the equations')
p.add_argument('-release_anticipation_seconds', type=float, default=0.0,
               help='Fixed release-command anticipation [s], recorded for reproducibility.')
p.add_argument('-physical_delay_min_ms', type=float, default=0.0,
               help='Lower physical release-delay bound [ms], recorded for reproducibility.')
p.add_argument('-physical_delay_max_ms', type=float, default=0.0,
               help='Upper physical release-delay bound [ms], recorded for reproducibility.')
locals().update(vars(p.parse_known_args()[0]))

# Set the seed. The current MC-PILCO evaluation uses base+algorithm seed.
paired_eval_seed = seed + 513 if evaluation_seed < 0 else evaluation_seed + seed
torch.manual_seed(paired_eval_seed)
np.random.seed(paired_eval_seed)
experiment_srv_name = 'tossing_experiment_testing'
red_ball_state_topic = 'red_ball_friction_odom'

num_trials = num_toss_test
# release_position = tossing_utils.release_position
release_position = [0.07, 0.0, release_z_world]
ALPHA = -1 * np.pi / 180
target_altitude = .1
V_MAX = 3.5


if model_drag:
    print('Test policy modeling stokes drag')
    policy_model = Policy.gravityPolicyFriction(state_dim=3, target_indeces=[6, 7, 8], alpha=ALPHA,
                                        release_pos=release_position, u_max=3.5,
                                        flg_squash=True, v_mult=v_mult)
else:
    print('Test gravity policy')
    print(load_poly_file)
    policy_model = Policy.gravityPolicy(state_dim=3, target_indeces=[6, 7, 8], alpha=ALPHA,
                                    release_pos=release_position, u_max=3.5,
                                    flg_squash=True, load_file=load_poly_file, v_mult=v_mult)

rospy.init_node('MC_PILCO_experiment_test_policy')

if evaluation_seed >= 0:
    rospy.set_param('/tossing/release_delay/seed', int(paired_eval_seed))
    reset_id = int(rospy.get_param('/tossing/release_delay/reset_id', 0)) + 1
    rospy.set_param('/tossing/release_delay/reset_id', reset_id)
    synthetic_prefix = '/tossing/release_delay/synthetic_ground_truth'
    rospy.set_param(synthetic_prefix + '/seed', int(paired_eval_seed))
    synthetic_reset_id = int(rospy.get_param(
        synthetic_prefix + '/reset_id', 0
    )) + 1
    rospy.set_param(synthetic_prefix + '/reset_id', synthetic_reset_id)


def sample_target_state():
    target_radius_world = min_target_dist + np.random.rand() * max_target_dist
    model_distance = (
        target_radius_world - release_position[0]
        if flg_world_radius_targets else target_radius_world
    )
    yaw = 2 * (np.random.rand() - 0.5) * np.pi / 6
    return tossing_utils.tossing_init(
        model_distance,
        yaw,
        target_altitude,
        release_pos=release_position,
    )


f_init_particles = sample_target_state

if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_config.json'), 'w') as f:
        json.dump({
            'seed': int(seed),
            'base_evaluation_seed': int(evaluation_seed),
            'paired_evaluation_seed': int(paired_eval_seed),
            'num_toss_test': int(num_toss_test),
            'target_radius_m': float(target_radius),
            'target_radial_range_m': [
                float(min_target_dist),
                float(min_target_dist + max_target_dist),
            ],
            'release_position_world_m': [float(x) for x in release_position],
            'bullet_name': bullet_name,
            'model_drag': bool(model_drag),
            'release_anticipation_seconds': float(release_anticipation_seconds),
            'physical_release_delay_ms': [
                float(physical_delay_min_ms),
                float(physical_delay_max_ms),
            ],
            'effective_residual_delay_ms': [
                float(physical_delay_min_ms - 1000.0 * release_anticipation_seconds),
                float(physical_delay_max_ms - 1000.0 * release_anticipation_seconds),
            ],
        }, f, indent=2)


for i in range(start_idx_test_trials, num_test_trials):
    results = tossing_utils.test_policy(policy_model, num_trials, seed, f_init_particles, True,
                                        'analytical_policy_{}'.format(i+1), bullet_name=bullet_name,
                                        save_dir=(output_dir if output_dir else None))

tossing_utils.plot_policy_test_results(results, target_radius=target_radius)
# plt.show()
plt.close('all')
