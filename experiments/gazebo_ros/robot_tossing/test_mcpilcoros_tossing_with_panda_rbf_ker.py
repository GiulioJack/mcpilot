#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
    Test MC-PILCO on a simulated Panda Robot system for tossing a ball to reach a target
    (GPs equipped with square-exponential kernels)

    Seed:



"""

from distutils.util import strtobool

import rospy
import torch
import numpy as np
import policy_learning.Policy as Policy
import policy_learning.MC_PILCO_ros_envs as MC_PILCO_ros_envs
import policy_learning.Cost_function as Cost_function
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import matplotlib.pyplot as plt
import dill as pkl
import argparse
import csv
import hashlib
import io
import json
import os
import time

import tossing_utils
import panda_differential_kinematics_utils

release_position = [0.07, 0.0, 2.55]
ALPHA = -1 * np.pi / 180
target_altitude = .1
V_MAX = 3.5
max_target_dist = 1.65


seed = 0
num_threads = 0
device_name = ''
model = ''
cost_type = ''
target_radius = 0.0
ctrl_noise_type = ''
centers_init_type = ''
num_toss_test = 0.0
use_dropout = True
num_particles = -1
bullet_name = 'red_ball'
num_trials = num_exp = num_test_trials = 1
verbose = False
std_ctrl_noise = 0.1
mean_ctrl_noise = 0.0
# flg_GP_mean = True  # if true use the GP mean
flg_GP_mean = False
flg_use_cache = False
flg_do_test = False
flg_simulate_diff_targets = True
poly_model_toss = True
flg_load_t_delay_dist = False
flg_model_release_delay = False
flg_apply_release_anticipation = False
flg_compensate_release_radial_velocity = False
evaluation_seed = 20260825
release_anticipation_seconds = 0.10
physical_delay_lower_seconds = 0.12
physical_delay_upper_seconds = 0.13
synthetic_ground_truth = True
shared_delay_calibration_file = ''
flg_delay_calibration_only = False
delay_calibration_throws = 50
flg_delay_diagnostic_only = False
delay_calibration_cache_dir = ''
delay_calibration_gp_checkpoint = ''
delay_diagnostic_bo_file = ''
delay_diagnostic_a_points = 21
delay_diagnostic_b_points = 21
delay_diagnostic_seed = 314159
flg_release_state_diagnostic_only = False
release_state_diagnostic_max_delay_seconds = 0.15
release_state_diagnostic_step_ms = 1.0
policy_only_source_dir = ''
policy_only_output_dir = ''
manual_delay_lower_seconds = None
manual_delay_upper_seconds = None
manual_delay_source = 'manual_exact_distribution'
policy_optimization_seed = -1
policy_only_run_bo = False
flg_bo_only = False
run_output_dir = ''
data_augmentation = 0
data_augmentation_max_angle_degrees = 45.0
std_meas_noise = 0.01
std_velocity_meas_noise = 0.0
flg_apply_meas_noise_to_training_data = False
training_measurement_noise_seed = -1
flg_real_style_data_augmentation = False
flg_sample_gp_posterior = True
delay_bo_a_lower_seconds = 0.0
delay_bo_a_upper_seconds = 0.10
delay_bo_b_lower_seconds = 0.001
delay_bo_b_upper_seconds = 0.10
delay_bo_init_points = 20
delay_bo_iterations = 100
delay_bo_seed = -1
delay_bo_particles = 10
delay_loss = 'trajectory_energy_score'
# Load parameters
p = argparse.ArgumentParser('test Tossing w/ Panda Robot')
p.add_argument('-seed', type=int, default=2, help='seed')
p.add_argument('-num_threads', type=int, default=8, help='number of computational threads')
p.add_argument('-num_particles', type=int, default=400, help='number of particles in MC gradient computation')
p.add_argument('-device_name', type=str, default='cuda', help='device name')
p.add_argument('-model', type=str, default='Speed', help='type of model learning (Speed, Speed_poly, delta)')
p.add_argument('-cost_type', type=str, default='last_pos', help='type of cost function for Tossbot')
p.add_argument('-centers_init_type', type=str, default='sparse', help='type of cost function for Tossbot (sparse, line, focus)')
p.add_argument('-bullet_name', type=str, default='red_ball_friction', help='bullet to load in the simulation (ball, cube, ..)')
p.add_argument('-ctrl_noise_type', type=str, default='zero_mean', help='type of control noise to apply in simulations (positive, negative, gauss, none)')
p.add_argument('-target_radius', type=float, default=0.05, help='Nominal target radius, same of the gazebo target model')
p.add_argument('-num_toss_test', type=int, default=100, help='number of toss in test trial (after training)')
p.add_argument('-num_exp', type=int, default=5, help='number of exploration trials (before first training)')
p.add_argument('-exploration_velocity_schedule', type=str, default='',
               help='Optional comma-separated absolute velocities [m/s], one per exploration throw. Example: 1.0,1.7,2.4,3.0,3.4. Empty retains Gaussian residual exploration.')
p.add_argument('-num_trials', type=int, default=1, help='number of total trials')
p.add_argument('-num_test_trials', type=int, default=10, help='number of total test trials')
p.add_argument('-use_dropout', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True,
               help='If true applies standard dropout params in reinforce policy (0.25, 0.125, 0.0)')
p.add_argument('-verbose', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true shows plots of data collection')
p.add_argument('-flg_GP_mean', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true uses mean function in GP models')
p.add_argument('-flg_load_t_delay_dist', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true loads the optimized t delay distribution parameters')
p.add_argument('-flg_model_release_delay', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True,
               help='If true loads the optimized t delay distribution parameters')
p.add_argument('-flg_apply_release_anticipation', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true applies the fixed coarse opening-command anticipation')
p.add_argument('-release_anticipation_seconds', type=float, default=0.10,
               help='Fixed coarse opening-command anticipation [s]. With the default U(0.12,0.13) s physical delay, 0.10 s leaves U(0.02,0.03) s residual timing uncertainty.')
p.add_argument('-flg_compensate_release_radial_velocity', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true, apply the radial release-velocity correction fitted on the 40 exploration throws from seeds 100_energy_score through 107_energy_score. Disabled by default.')
p.add_argument('-physical_delay_lower_seconds', type=float, default=0.12,
               help='Lower bound of the simulated physical release delay [s].')
p.add_argument('-physical_delay_upper_seconds', type=float, default=0.13,
               help='Upper bound of the simulated physical release delay [s].')
p.add_argument('-synthetic_ground_truth', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True,
               help='Select the Gazebo synthetic residual-delay path (default: True). Pass False explicitly to run the physical actuator-delay path.')
p.add_argument('-shared_delay_calibration_file', type=str, default='',
               help='Optional shared residual-delay calibration pickle. When set, MC-PILOT loads this calibration instead of re-estimating delay from its five exploration throws.')
p.add_argument('-flg_delay_calibration_only', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true, collect a standalone delay-calibration dataset, estimate the shared residual-delay distribution, save it, and exit before MC-PILOT policy learning.')
p.add_argument('-delay_calibration_throws', type=int, default=50,
               help='Number of standalone throws used for the one-time shared delay calibration.')
p.add_argument('-flg_delay_diagnostic_only', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true, reuse an existing 50-throw calibration cache and completed BO result, load the saved calibration GP if available (otherwise train it once), run only the delay-identifiability diagnostic, and exit. No robot throws and no BO are performed.')
p.add_argument('-delay_calibration_cache_dir', type=str, default='',
               help='Directory containing cached calibration trial_*.pkl files. Empty uses results_tmp/<seed>/cache_tossing_trials/.')
p.add_argument('-delay_calibration_gp_checkpoint', type=str, default='',
               help='Calibration GP checkpoint. Empty uses results_tmp/<seed>/delay_calibration_gp_checkpoint.pt. If it exists and matches the cached throws it is loaded; otherwise the GP is trained and a checkpoint is written.')
p.add_argument('-delay_diagnostic_gp_source_dir', type=str, default='',
               help='Existing run directory containing log.pkl. In diagnostic-only mode, load its frozen GP snapshot 4 instead of retraining or requiring a checkpoint.')
p.add_argument('-delay_diagnostic_bo_file', type=str, default='',
               help='Completed delay-BO pickle to reuse in diagnostic-only mode. Empty first tries -shared_delay_calibration_file and then the local delay-result pickle in results_tmp/<seed>.')
p.add_argument('-delay_diagnostic_a_points', type=int, default=21,
               help='Number of diagnostic grid points for residual-delay lower bound a in [0,0.10] s.')
p.add_argument('-delay_diagnostic_b_points', type=int, default=21,
               help='Number of diagnostic grid points for residual-delay width b in [0.001,0.10] s.')
p.add_argument('-delay_diagnostic_a_max_seconds', type=float, default=0.10,
               help='Upper diagnostic grid bound for lower endpoint a [s].')
p.add_argument('-delay_diagnostic_b_max_seconds', type=float, default=0.10,
               help='Upper diagnostic grid bound for width b [s]; zero is always included.')
p.add_argument('-delay_diagnostic_skip_plots', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='Write diagnostic CSV/NPZ/JSON without plotting; useful on memory-constrained machines.')
p.add_argument('-delay_diagnostic_seed', type=int, default=314159,
               help='RNG seed reset before every diagnostic candidate to enforce common random numbers.')
p.add_argument('-flg_quick_trajectory_loss_only', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true, reuse cached calibration throws, saved calibration GP and completed BO result to compare early-trajectory loss for the true residual distribution and the BO-estimated distribution only. No robot throws, GP training (if checkpoint is compatible), BO, or grid search are performed.')
p.add_argument('-flg_release_state_diagnostic_only', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true, compare the first measured free-flight position/velocity in the cached calibration throws directly against the analytical release-state mapping over a delay grid. No GP, BO, or robot throws are used.')
p.add_argument('-release_state_diagnostic_max_delay_seconds', type=float, default=0.15,
               help='Maximum residual delay [s] tested by the direct release-state diagnostic.')
p.add_argument('-release_state_diagnostic_step_ms', type=float, default=1.0,
               help='Delay-grid spacing [ms] for the direct release-state diagnostic.')
p.add_argument('-policy_only_source_dir', type=str, default='',
               help='Existing completed run containing log.pkl with the frozen GP snapshot. When set, skip exploration, GP optimization, delay BO, and on-system training rollout.')
p.add_argument('-policy_only_output_dir', type=str, default='',
               help='Separate output directory for a policy-only rerun. Required with -policy_only_source_dir and must differ from the source directory.')
p.add_argument('-manual_delay_lower_seconds', type=float, default=None,
               help='Lower bound [s] of the manually imposed policy-rollout delay distribution in policy-only mode.')
p.add_argument('-manual_delay_upper_seconds', type=float, default=None,
               help='Upper bound [s] of the manually imposed policy-rollout delay distribution in policy-only mode.')
p.add_argument('-manual_delay_source', type=str, default='manual_exact_distribution',
               help='Provenance label stored with a manually supplied rollout-delay distribution.')
p.add_argument('-policy_optimization_seed', type=int, default=-1,
               help='RNG seed reset immediately before policy-only optimization. Negative uses -seed.')
p.add_argument('-policy_only_run_bo', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='In policy-only mode, rerun delay BO with the frozen source GP and five source exploration conditions before optimizing the policy.')
p.add_argument('-flg_bo_only', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='With policy-only BO, save BO execution timing and exit before policy optimization.')
p.add_argument('-run_output_dir', type=str, default='',
               help='Optional output directory for a normal run. Empty uses results_tmp/<seed>.')
p.add_argument('-data_augmentation', type=int, default=0,
               help='Number of random Z-rotated copies added for every observed trajectory. Zero disables augmentation.')
p.add_argument('-data_augmentation_max_angle_degrees', type=float, default=45.0,
               help='Symmetric maximum random Z-rotation angle for data augmentation, in degrees.')
p.add_argument('-std_meas_noise', type=float, default=0.01,
               help='Position measurement-noise standard deviation in meters. Target dimensions remain noise-free.')
p.add_argument('-std_velocity_meas_noise', type=float, default=0.0,
               help='Velocity measurement-noise standard deviation in meters per second.')
p.add_argument('-flg_apply_meas_noise_to_training_data', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true, perturb measured training positions with Gaussian noise before interpolation/model learning.')
p.add_argument('-training_measurement_noise_seed', type=int, default=-1,
               help='RNG seed for training measurement-noise injection. Negative uses -seed.')
p.add_argument('-flg_real_style_data_augmentation', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='Inject noise into positions; augmented copies share the noisy position trajectory and velocities are obtained by numerical differentiation after filtering.')
p.add_argument('-flg_sample_gp_posterior', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True,
               help='Sample GP predictive posteriors during policy rollouts. False propagates the GP predictive mean.')
p.add_argument('-delay_bo_a_lower_seconds', type=float, default=0.0,
               help='BO lower bound [s] for uniform-delay lower endpoint a.')
p.add_argument('-delay_bo_a_upper_seconds', type=float, default=0.10,
               help='BO upper bound [s] for uniform-delay lower endpoint a.')
p.add_argument('-delay_bo_b_lower_seconds', type=float, default=0.001,
               help='BO lower bound [s] for uniform-delay width b.')
p.add_argument('-delay_bo_b_upper_seconds', type=float, default=0.10,
               help='BO upper bound [s] for uniform-delay width b.')
p.add_argument('-delay_bo_init_points', type=int, default=20,
               help='Number of random initial BO evaluations.')
p.add_argument('-delay_bo_iterations', type=int, default=100,
               help='Number of guided BO iterations after initialization.')
p.add_argument('-delay_bo_seed', type=int, default=-1,
               help='Random seed for delay BO. Negative uses -seed.')
p.add_argument('-delay_bo_particles', type=int, default=10,
               help='Number of GP rollout particles used to score each trajectory for every delay-BO candidate. Independent of -num_particles used for policy optimization.')
p.add_argument('-delay_loss', type=str, default='trajectory_energy_score',
               choices=['mse', 'energy_score', 'trajectory_energy_score'],
               help='Loss used by release-delay BO. Default mse preserves the historical landing objective exactly.')
p.add_argument('-evaluation_seed', type=int, default=20260825,
               help='Base seed for paired policy evaluation targets and physical delay samples')
p.add_argument('-flg_use_cache', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true uses cached exploration data')
p.add_argument('-flg_do_test', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False,
               help='If true performs test')
p.add_argument('-flg_simulate_diff_targets', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True,
               help='If true, at each particles simulation in a policy gradient step, each particle is initialized independently, '
                    'instead if false, in a policy gradient step each particles are initialized the same.')


locals().update(vars(p.parse_known_args()[0]))

if exploration_velocity_schedule.strip():
    exploration_velocity_schedule = [
        float(value.strip())
        for value in exploration_velocity_schedule.split(',')
        if value.strip()
    ]
    if len(exploration_velocity_schedule) != int(num_exp):
        raise ValueError(
            '-exploration_velocity_schedule must contain exactly {} values '
            '(one for each exploration throw); received {}'.format(
                int(num_exp), len(exploration_velocity_schedule)
            )
        )
    if any(value <= 0.0 or value > V_MAX for value in exploration_velocity_schedule):
        raise ValueError(
            'Every prescribed exploration velocity must be in (0, {:.3f}] m/s'.format(V_MAX)
        )
else:
    exploration_velocity_schedule = None

if exploration_velocity_schedule is not None and flg_use_cache:
    raise ValueError(
        '-exploration_velocity_schedule requires fresh exploration data; '
        'run with -flg_use_cache False'
    )

if (
        delay_bo_a_lower_seconds < 0.0
        or delay_bo_a_upper_seconds < delay_bo_a_lower_seconds
        or delay_bo_b_lower_seconds <= 0.0
        or delay_bo_b_upper_seconds < delay_bo_b_lower_seconds
):
    raise ValueError('Invalid delay-BO parameter bounds')
if delay_bo_init_points < 0 or delay_bo_iterations < 0:
    raise ValueError('BO evaluation counts must be non-negative')
if delay_bo_init_points + delay_bo_iterations <= 0:
    raise ValueError('BO requires at least one evaluation')
if delay_bo_particles <= 0:
    raise ValueError('-delay_bo_particles must be positive')
effective_delay_bo_seed = int(seed if delay_bo_seed < 0 else delay_bo_seed)
if data_augmentation < 0:
    raise ValueError('-data_augmentation must be non-negative')
if data_augmentation_max_angle_degrees < 0.0:
    raise ValueError('-data_augmentation_max_angle_degrees must be non-negative')
if std_meas_noise < 0.0:
    raise ValueError('-std_meas_noise must be non-negative')
if std_velocity_meas_noise < 0.0:
    raise ValueError('-std_velocity_meas_noise must be non-negative')
effective_training_measurement_noise_seed = int(
    seed if training_measurement_noise_seed < 0 else training_measurement_noise_seed
)

manual_delay_mode = (
    manual_delay_lower_seconds is not None
    or manual_delay_upper_seconds is not None
)
if manual_delay_mode:
    if (
            manual_delay_lower_seconds is None
            or manual_delay_upper_seconds is None
            or manual_delay_lower_seconds < 0.0
            or manual_delay_upper_seconds < manual_delay_lower_seconds
    ):
        raise ValueError('Both manual delay bounds must define a valid interval')

if run_output_dir:
    run_output_dir = os.path.abspath(os.path.expanduser(run_output_dir))

policy_only_mode = bool(policy_only_source_dir)
if policy_only_run_bo and not policy_only_mode:
    raise ValueError(
        '-policy_only_run_bo requires -policy_only_source_dir'
    )
if flg_bo_only and not (policy_only_mode and policy_only_run_bo):
    raise ValueError(
        '-flg_bo_only requires -policy_only_source_dir and '
        '-policy_only_run_bo True'
    )
if policy_only_mode:
    if not policy_only_output_dir:
        raise ValueError(
            '-policy_only_output_dir is required with -policy_only_source_dir'
        )
    if (
            not policy_only_run_bo
            and (
                manual_delay_lower_seconds is None
                or manual_delay_upper_seconds is None
            )
    ):
        raise ValueError(
            'Both manual delay bounds are required unless '
            '-policy_only_run_bo True is selected'
        )
    if not policy_only_run_bo and (
            manual_delay_lower_seconds < 0.0
            or manual_delay_upper_seconds < manual_delay_lower_seconds
    ):
        raise ValueError('Invalid manual policy delay interval')
    if int(num_exp) != 5:
        raise ValueError(
            'Policy-only mode currently requires -num_exp 5 to select the '
            'seed GP trained on trial_0 through trial_4'
        )

    policy_only_source_dir = os.path.abspath(
        os.path.expanduser(policy_only_source_dir)
    )
    policy_only_output_dir = os.path.abspath(
        os.path.expanduser(policy_only_output_dir)
    )
    if policy_only_source_dir == policy_only_output_dir:
        raise ValueError('Policy-only source and output directories must differ')
    if os.path.exists(policy_only_output_dir):
        if not os.path.isdir(policy_only_output_dir):
            raise ValueError('Policy-only output path exists and is not a directory')
        unexpected_output_files = [
            name for name in os.listdir(policy_only_output_dir)
            if name != 'run.log'
        ]
        if unexpected_output_files:
            raise FileExistsError(
                'Policy-only output directory is not empty: {}'.format(
                    policy_only_output_dir
                )
            )
    if not os.path.isfile(os.path.join(policy_only_source_dir, 'log.pkl')):
        raise FileNotFoundError(
            'Missing source GP log: {}'.format(
                os.path.join(policy_only_source_dir, 'log.pkl')
            )
        )
    for exploration_index in range(5):
        exploration_path = os.path.join(
            policy_only_source_dir,
            'cache_tossing_trials',
            'trial_{}.pkl'.format(exploration_index),
        )
        if not os.path.isfile(exploration_path):
            raise FileNotFoundError(
                'Missing source exploration trajectory: {}'.format(
                    exploration_path
                )
            )

# Set the seed
torch.manual_seed(seed)
np.random.seed(seed)


def reset_physical_delay_rng(delay_seed):
    """Restart the ROS-side physical and synthetic delay RNGs reproducibly."""
    rospy.set_param('/tossing/release_delay/seed', int(delay_seed))
    reset_id = int(rospy.get_param('/tossing/release_delay/reset_id', 0)) + 1
    rospy.set_param('/tossing/release_delay/reset_id', reset_id)

    # Synthetic-ground-truth mode has an independent RNG and reset counter.
    # Reset it as well so that a multi-seed runner does not merely continue the
    # random stream selected when roslaunch was started.
    synthetic_prefix = '/tossing/release_delay/synthetic_ground_truth'
    rospy.set_param(synthetic_prefix + '/seed', int(delay_seed))
    synthetic_reset_id = int(
        rospy.get_param(synthetic_prefix + '/reset_id', 0)
    ) + 1
    rospy.set_param(synthetic_prefix + '/reset_id', synthetic_reset_id)


def configure_physical_release_delay(
        lower_seconds, upper_seconds, use_synthetic_ground_truth=True):
    """Configure the ROS delay and explicitly select its realization path."""
    lower_seconds = float(lower_seconds)
    upper_seconds = float(upper_seconds)
    if lower_seconds < 0.0 or upper_seconds < lower_seconds:
        raise ValueError(
            'Invalid physical delay interval [{}, {}] s'.format(
                lower_seconds, upper_seconds
            )
        )

    lower_ms = int(round(1000.0 * lower_seconds))
    upper_ms = int(round(1000.0 * upper_seconds))
    midpoint_ms = 0.5 * (lower_ms + upper_ms)

    # The ROS parameter server outlives each Python seed in the batch runners.
    # Explicitly select the requested path here; otherwise an earlier run can
    # silently determine whether every later seed is physical or synthetic.
    rospy.set_param(
        '/tossing/release_delay/synthetic_ground_truth/enabled',
        bool(use_synthetic_ground_truth),
    )
    rospy.set_param('/tossing/release_delay/enabled', True)
    rospy.set_param('/tossing/release_delay/distribution', 'uniform')
    rospy.set_param('/tossing/release_delay/min_steps', lower_ms)
    rospy.set_param('/tossing/release_delay/max_steps', upper_ms)
    rospy.set_param('/tossing/release_delay/fixed_steps', int(round(midpoint_ms)))
    rospy.set_param('/tossing/release_delay/mean_steps', float(midpoint_ms))
    # Only used if the distribution is manually switched to Gaussian.
    rospy.set_param(
        '/tossing/release_delay/std_steps',
        max(0.0, float(upper_ms - lower_ms) / 4.0),
    )

    print(
        'Release realization mode: {}'.format(
            'synthetic_ground_truth'
            if use_synthetic_ground_truth else 'physical_actuator'
        )
    )
    print(
        'Physical actuator delay configured as U({:.1f}, {:.1f}) ms'.format(
            1000.0 * lower_seconds, 1000.0 * upper_seconds
        )
    )


def save_shared_delay_calibration(path, payload, metadata):
    """Save the BO payload plus a human-readable metadata sidecar."""
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'wb') as calibration_file:
        pkl.dump(payload, calibration_file)

    metadata_path = os.path.splitext(path)[0] + '.json'
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return path, metadata_path


def load_shared_delay_calibration(path):
    """Load the shared calibration in the same format used by legacy runs."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError('Shared delay calibration file not found: {}'.format(path))
    with open(path, 'rb') as calibration_file:
        return pkl.load(calibration_file)


def make_stratified_delay_calibration_states(num_throws, calibration_seed):
    """Create a reproducible calibration set spanning the radial workspace.

    Five radial strata are used.  Targets are placed at the center of each
    stratum and paired with yaw angles sampled uniformly in the normal task
    range.  The resulting schedule is shuffled so that velocity regimes are
    interleaved in time.
    """
    if num_throws <= 0:
        raise ValueError('delay_calibration_throws must be positive')
    if num_throws % 5 != 0:
        raise ValueError(
            'delay_calibration_throws must be a multiple of 5 because the '
            'current delay-estimation helper consumes trajectories in groups of 5'
        )

    rng = np.random.RandomState(int(calibration_seed) + 1701)
    radial_edges = np.linspace(
        tossing_utils.min_distance,
        tossing_utils.min_distance + max_target_dist,
        6,
    )
    radial_levels = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    repetitions = num_throws // len(radial_levels)

    schedule = []
    for radius in radial_levels:
        for _ in range(repetitions):
            yaw = rng.uniform(-np.pi / 6.0, np.pi / 6.0)
            schedule.append((float(radius), float(yaw)))
    rng.shuffle(schedule)

    states = [
        tossing_utils.tossing_init(
            radius - release_position[0],
            yaw,
            target_altitude,
            release_pos=release_position,
        )
        for radius, yaw in schedule
    ]
    return schedule, states


def _calibration_trial_paths(cache_dir, num_throws):
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    paths = [os.path.join(cache_dir, 'trial_{}.pkl'.format(i)) for i in range(int(num_throws))]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            'Calibration cache is incomplete: {} of {} trial files are missing. '
            'First missing file: {}'.format(len(missing), len(paths), missing[0])
        )
    return cache_dir, paths


def _calibration_cache_fingerprint(trial_paths):
    """SHA256 over the exact raw 50-throw cache used for GP calibration."""
    digest = hashlib.sha256()
    for path in trial_paths:
        digest.update(os.path.basename(path).encode('utf-8'))
        with open(path, 'rb') as trial_file:
            while True:
                block = trial_file.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _file_sha256(path):
    """Return a reproducibility fingerprint for one immutable source file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source_file:
        while True:
            block = source_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_calibration_conditions_from_cache(PL_obj, trial_paths):
    """Load BO observations, including each complete free-flight trajectory."""
    conditions = []
    for path in trial_paths:
        with open(path, 'rb') as trial_file:
            noisy_samples, input_samples = pkl.load(trial_file)
        trajectory_target_point = np.asarray(noisy_samples[0, 7:], dtype=float)
        # Keep NumPy's scalar type here.  simulate_system() calls .item() on
        # the stored policy output, matching the type produced by the original
        # delay-calibration path.  Casting to Python float breaks that contract.
        trajectory_policy_output = np.linalg.norm(
            np.asarray(input_samples[0, :], dtype=float)
        )
        trajectory_landing = np.asarray(noisy_samples[-1, 1:4], dtype=float)
        trajectory_times = np.asarray(noisy_samples[:, 0], dtype=float)
        trajectory_times = trajectory_times - trajectory_times[0]
        conditions.append([
            trajectory_target_point,
            trajectory_policy_output,
            trajectory_landing,
            {
                'times': trajectory_times,
                'positions': np.asarray(noisy_samples[:, 1:4], dtype=float),
            },
        ])
    PL_obj.trajectories_initial_conditions = conditions
    return conditions


def _populate_calibration_gp_data_from_cache(PL_obj, trial_paths):
    """Rebuild the GP training dataset from cached raw trajectories.

    This path is used only when no compatible trained-GP checkpoint exists.
    It performs no ROS interaction and no BO.
    """
    if int(PL_obj.model_learning.num_samples) != 0:
        raise RuntimeError(
            'Refusing to append cached calibration data to a non-empty GP dataset '
            '(num_samples={}). Start diagnostic-only mode in a fresh Python process.'.format(
                PL_obj.model_learning.num_samples
            )
        )
    for idx, path in enumerate(trial_paths):
        with open(path, 'rb') as trial_file:
            noisy_samples, input_samples = pkl.load(trial_file)
        state_samples, input_samples_resamp = PL_obj.interpolate_states(
            np.array(noisy_samples, copy=True),
            np.array(input_samples, copy=True),
        )
        PL_obj.model_learning.add_data(
            new_state_samples=state_samples,
            new_input_samples=input_samples_resamp,
        )
        print('Loaded calibration GP data {}/{} from {}'.format(
            idx + 1, len(trial_paths), os.path.basename(path)))


def _checkpoint_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def save_calibration_gp_checkpoint(PL_obj, checkpoint_path, cache_fingerprint, num_throws):
    """Save the exact trained GP hyperparameters and GP training tensors."""
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    model = PL_obj.model_learning
    payload = {
        'format_version': 1,
        'cache_fingerprint_sha256': str(cache_fingerprint),
        'num_calibration_throws': int(num_throws),
        'num_gp': int(model.num_gp),
        'approximation_mode': model.approximation_mode,
        'model_state_dict': _checkpoint_to_cpu(model.state_dict()),
        # Save the *actual* GP tensors used during training.  This avoids the
        # tiny interpolation jitter changing when a checkpoint is reloaded.
        'gp_inputs': model.gp_inputs.detach().cpu().clone(),
        'gp_output_list': [item.detach().cpu().clone() for item in model.gp_output_list],
        'num_samples': int(model.num_samples),
        'evolution_indices': list(model.evolution_indices) if model.evolution_indices is not None else None,
        'dim_state': int(model.dim_state),
        'dim_input': int(model.dim_input),
        'norm_mean_coef_list': list(model.norm_mean_coef_list),
        'norm_scale_coef_list': list(model.norm_scale_coef_list),
    }
    torch.save(payload, checkpoint_path)
    print('Saved calibration GP checkpoint: {}'.format(checkpoint_path))
    return checkpoint_path


def try_load_calibration_gp_checkpoint(PL_obj, checkpoint_path, cache_fingerprint, num_throws):
    """Load a compatible calibration GP; return False safely on any mismatch."""
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        print('Calibration GP checkpoint not found: {}'.format(checkpoint_path))
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = PL_obj.model_learning

        expected = {
            'format_version': 1,
            'cache_fingerprint_sha256': str(cache_fingerprint),
            'num_calibration_throws': int(num_throws),
            'num_gp': int(model.num_gp),
            'approximation_mode': model.approximation_mode,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    'checkpoint metadata mismatch for {}: saved={!r}, expected={!r}'.format(
                        key, checkpoint.get(key), value
                    )
                )

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.gp_inputs = checkpoint['gp_inputs'].to(device=model.device, dtype=model.dtype)
        model.gp_output_list = [
            item.to(device=model.device, dtype=model.dtype)
            for item in checkpoint['gp_output_list']
        ]
        model.num_samples = int(checkpoint['num_samples'])
        model.evolution_indices = checkpoint.get('evolution_indices')
        model.dim_state = int(checkpoint['dim_state'])
        model.dim_input = int(checkpoint['dim_input'])
        model.norm_mean_coef_list = list(checkpoint['norm_mean_coef_list'])
        model.norm_scale_coef_list = list(checkpoint['norm_scale_coef_list'])

        # alpha/K^{-1}/SOD caches are not parameters in state_dict.  Rebuild
        # them deterministically from the exact saved GP inputs/outputs.  This
        # is much cheaper than the 1001-epoch hyperparameter optimization.
        with torch.no_grad():
            for gp_index in range(model.num_gp):
                model.pretrain_gp(gp_index)

        print('Loaded compatible calibration GP checkpoint: {}'.format(checkpoint_path))
        return True
    except Exception as exc:
        print(
            'WARNING: could not safely load calibration GP checkpoint {}: {}\n'
            'The checkpoint will be ignored and the GP will be trained from the cached throws.'.format(
                checkpoint_path, repr(exc)
            )
        )
        return False


def ensure_calibration_gp(PL_obj, checkpoint_path, cache_fingerprint, trial_paths,
                          model_optimization_opt_list, model_data_already_loaded):
    """Load a matching GP checkpoint, otherwise train once and save it."""
    if try_load_calibration_gp_checkpoint(
            PL_obj, checkpoint_path, cache_fingerprint, len(trial_paths)):
        return 'loaded'

    if not model_data_already_loaded:
        _populate_calibration_gp_data_from_cache(PL_obj, trial_paths)

    print('\n\n----- REINFORCE THE CALIBRATION MODEL -----')
    t_model_learning_start = __import__('time').time()
    PL_obj.model_learning.reinforce_model(
        optimization_opt_list=model_optimization_opt_list
    )
    elapsed = __import__('time').time() - t_model_learning_start
    print('\nCalibration model optimization time: {}'.format(elapsed))
    save_calibration_gp_checkpoint(
        PL_obj, checkpoint_path, cache_fingerprint, len(trial_paths)
    )
    return 'trained'


def load_completed_delay_bo_payload(explicit_path, shared_path, local_path):
    """Load a completed BO result without ever launching BO."""
    candidates = []
    for candidate in (explicit_path, shared_path, local_path):
        candidate = str(candidate).strip() if candidate is not None else ''
        if candidate:
            absolute = os.path.abspath(os.path.expanduser(candidate))
            if absolute not in candidates:
                candidates.append(absolute)

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        with open(candidate, 'rb') as result_file:
            payload = pkl.load(result_file)
        if not isinstance(payload, (list, tuple)) or len(payload) < 4:
            raise ValueError('Unexpected delay-BO payload format in {}'.format(candidate))
        trials_results = payload[3]
        if not isinstance(trials_results, (list, tuple)) or len(trials_results) < 2:
            raise ValueError('Missing BO optimum in {}'.format(candidate))
        optimum = trials_results[1]
        if 'params' not in optimum or 'a' not in optimum['params'] or 'b' not in optimum['params']:
            raise ValueError('Invalid BO optimum structure in {}'.format(candidate))
        print('Reusing completed delay BO result: {}'.format(candidate))
        return payload, candidate

    raise FileNotFoundError(
        'No completed delay-BO result was found. Checked: {}. '
        'Let the current BO finish first; diagnostic-only mode will never rerun BO.'.format(
            candidates
        )
    )


def run_quick_trajectory_loss_comparison(
        PL_obj, init_function, trial_paths, a_hat, b_hat,
        true_residual_lower, true_residual_upper,
        diagnostic_seed, windows_seconds=(0.05, 0.10, 0.20, 0.30)):
    """Compare early free-flight trajectory loss for true vs BO delay models.

    This is intentionally a minimal diagnostic. It keeps exactly 10 simulated
    particles per observed throw and uses the same particle-to-observation
    squared-position loss as the existing landing objective, but averages it
    over the initial trajectory window instead of only at landing.
    """
    num_particles = 10
    windows_seconds = tuple(float(w) for w in windows_seconds)
    if not windows_seconds or min(windows_seconds) <= 0.0:
        raise ValueError('Trajectory-loss windows must be positive')
    max_window = max(windows_seconds)
    num_steps = int(np.floor(max_window / PL_obj.T_sampling)) + 1
    sim_times = np.arange(num_steps, dtype=float) * float(PL_obj.T_sampling)

    def evaluate_case(case_name, dist_fun):
        np.random.seed(int(diagnostic_seed))
        torch.manual_seed(int(diagnostic_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(diagnostic_seed))

        sums = {w: 0.0 for w in windows_seconds}
        counts = {w: 0 for w in windows_seconds}
        per_throw = []

        for throw_index, path in enumerate(trial_paths):
            with open(path, 'rb') as trial_file:
                noisy_samples, input_samples = pkl.load(trial_file)
            noisy_samples = np.asarray(noisy_samples, dtype=float)
            input_samples = np.asarray(input_samples, dtype=float)

            target_point = noisy_samples[0, 7:10]
            target_yaw = np.arctan2(target_point[1], target_point[0])
            initial_speed = np.linalg.norm(input_samples[0, :])

            obs_times = noisy_samples[:, 0] - noisy_samples[0, 0]
            obs_pos = noisy_samples[:, 1:4]
            if obs_times[-1] <= 0.0:
                raise RuntimeError('Non-positive trajectory duration in {}'.format(path))

            f_init_particles = lambda v: init_function(v, target_yaw, dist_fun())
            particles_sequence = PL_obj.get_particles_evolution_robot_kin(
                f_init_particles,
                num_particles,
                num_steps,
                initial_vel=initial_speed.item(),
                particle_pred=True,
            )
            sim_pos = np.stack([
                particles_sequence[t][:, :3].detach().cpu().numpy()
                for t in range(num_steps)
            ], axis=0)  # [time, particle, xyz]

            throw_row = {'throw_index': int(throw_index)}
            for window in windows_seconds:
                valid_end = min(float(window), float(obs_times[-1]), float(sim_times[-1]))
                mask = sim_times <= valid_end + 1e-12
                times_this = sim_times[mask]
                if len(times_this) == 0:
                    continue
                obs_interp = np.column_stack([
                    np.interp(times_this, obs_times, obs_pos[:, axis])
                    for axis in range(3)
                ])
                errors = sim_pos[mask, :, :] - obs_interp[:, None, :]
                # Same squared Euclidean position discrepancy, now over time.
                mse = float(np.mean(np.sum(errors ** 2, axis=2)))
                sums[window] += mse
                counts[window] += 1
                throw_row['mse_{:.0f}ms'.format(1000.0 * window)] = mse
            per_throw.append(throw_row)

        means = {
            w: (sums[w] / counts[w] if counts[w] else float('nan'))
            for w in windows_seconds
        }
        return {'name': case_name, 'mean_mse': means, 'per_throw': per_throw}

    true_case = evaluate_case(
        'ground_truth_uniform',
        lambda: np.random.uniform(true_residual_lower, true_residual_upper),
    )
    estimated_case = evaluate_case(
        'bo_estimated_uniform',
        lambda: np.random.uniform(a_hat, a_hat + b_hat),
    )

    print('\n==========================================================')
    print('QUICK EARLY-TRAJECTORY LOSS COMPARISON')
    print('Particles per observed throw: {} (UNCHANGED)'.format(num_particles))
    print('Throws: {}'.format(len(trial_paths)))
    print('Common-random-number seed: {}'.format(diagnostic_seed))
    print('True residual: U({:.1f}, {:.1f}) ms'.format(
        1000.0 * true_residual_lower, 1000.0 * true_residual_upper))
    print('BO estimate:  U({:.1f}, {:.1f}) ms'.format(
        1000.0 * a_hat, 1000.0 * (a_hat + b_hat)))
    print('----------------------------------------------------------')
    for window in windows_seconds:
        true_mse = true_case['mean_mse'][window]
        est_mse = estimated_case['mean_mse'][window]
        winner = 'TRUE' if true_mse < est_mse else 'BO'
        ratio = true_mse / est_mse if est_mse > 0 else float('inf')
        print('{:>4.0f} ms: true={:.8g} | BO={:.8g} | better={} | true/BO={:.3f}'.format(
            1000.0 * window, true_mse, est_mse, winner, ratio))
    print('==========================================================\n')

    return {'ground_truth': true_case, 'bo_estimated': estimated_case}



def run_direct_release_state_diagnostic(
        init_function, trial_paths, output_dir,
        max_delay_seconds=0.15, step_ms=1.0,
        bo_a=None, bo_b=None):
    """Compare cached first free-flight states directly with h(P, v, delay).

    The cached tossing trajectories produced by the current ROS tracker start
    at the first odometry sample whose timestamp is not earlier than the
    physical detach-service call.  Therefore noisy_samples[0,1:7] is the first
    measured *free-flight* state, not a sample collected while the projectile
    is still attached to the gripper.

    Position and velocity are deliberately diagnosed separately so this check
    introduces no arbitrary weighting between metres and metres/second.
    No GP prediction is involved.
    """
    max_delay_seconds = float(max_delay_seconds)
    step_ms = float(step_ms)
    if max_delay_seconds <= 0.0:
        raise ValueError('release-state diagnostic max delay must be positive')
    if step_ms <= 0.0:
        raise ValueError('release-state diagnostic step must be positive')

    step_seconds = step_ms / 1000.0
    delays = np.arange(
        0.0, max_delay_seconds + 0.5 * step_seconds, step_seconds,
        dtype=float,
    )
    if len(delays) < 2:
        raise ValueError('release-state diagnostic grid has fewer than two points')

    pos_sq_by_throw = []
    vel_sq_by_throw = []
    per_throw_rows = []
    first_sample_dt = []

    for throw_index, path in enumerate(trial_paths):
        with open(path, 'rb') as trial_file:
            noisy_samples, input_samples = pkl.load(trial_file)
        noisy_samples = np.asarray(noisy_samples, dtype=float)
        input_samples = np.asarray(input_samples, dtype=float)
        if noisy_samples.ndim != 2 or noisy_samples.shape[0] < 1 or noisy_samples.shape[1] < 10:
            raise ValueError('Unexpected cached trajectory shape {} in {}'.format(
                noisy_samples.shape, path))
        if input_samples.ndim != 2 or input_samples.shape[0] < 1:
            raise ValueError('Unexpected cached input shape {} in {}'.format(
                input_samples.shape, path))

        obs_pos = np.asarray(noisy_samples[0, 1:4], dtype=float)
        obs_vel = np.asarray(noisy_samples[0, 4:7], dtype=float)
        target = np.asarray(noisy_samples[0, 7:10], dtype=float)
        target_yaw = float(np.arctan2(target[1], target[0]))
        target_radius = float(np.linalg.norm(target[:2]))
        commanded_speed = float(np.linalg.norm(input_samples[0, :]))

        if noisy_samples.shape[0] > 1:
            first_dt = float(noisy_samples[1, 0] - noisy_samples[0, 0])
            if first_dt > 0.0:
                first_sample_dt.append(first_dt)
        else:
            first_dt = float('nan')

        pos_sq = np.empty(len(delays), dtype=float)
        vel_sq = np.empty(len(delays), dtype=float)
        for delay_index, delay_seconds in enumerate(delays):
            state = np.asarray(
                init_function(commanded_speed, target_yaw, float(delay_seconds)),
                dtype=float,
            ).reshape(-1)
            if state.size < 6:
                raise ValueError('Release-state initializer returned {} elements'.format(state.size))
            pos_sq[delay_index] = float(np.sum((state[:3] - obs_pos) ** 2))
            vel_sq[delay_index] = float(np.sum((state[3:6] - obs_vel) ** 2))

        pos_sq_by_throw.append(pos_sq)
        vel_sq_by_throw.append(vel_sq)
        pos_best_idx = int(np.argmin(pos_sq))
        vel_best_idx = int(np.argmin(vel_sq))
        per_throw_rows.append({
            'throw_index': int(throw_index),
            'commanded_speed_mps': commanded_speed,
            'target_radius_m': target_radius,
            'target_yaw_rad': target_yaw,
            'first_freeflight_dt_to_next_sample_s': first_dt,
            'obs_x_m': float(obs_pos[0]),
            'obs_y_m': float(obs_pos[1]),
            'obs_z_m': float(obs_pos[2]),
            'obs_vx_mps': float(obs_vel[0]),
            'obs_vy_mps': float(obs_vel[1]),
            'obs_vz_mps': float(obs_vel[2]),
            'best_position_delay_ms': float(1000.0 * delays[pos_best_idx]),
            'best_position_error_m': float(np.sqrt(pos_sq[pos_best_idx])),
            'best_velocity_delay_ms': float(1000.0 * delays[vel_best_idx]),
            'best_velocity_error_mps': float(np.sqrt(vel_sq[vel_best_idx])),
        })

    pos_sq_by_throw = np.asarray(pos_sq_by_throw, dtype=float)
    vel_sq_by_throw = np.asarray(vel_sq_by_throw, dtype=float)
    mean_pos_mse = np.mean(pos_sq_by_throw, axis=0)
    mean_vel_mse = np.mean(vel_sq_by_throw, axis=0)
    global_pos_idx = int(np.argmin(mean_pos_mse))
    global_vel_idx = int(np.argmin(mean_vel_mse))

    def grid_metrics(delay_seconds):
        idx = int(np.argmin(np.abs(delays - float(delay_seconds))))
        return {
            'grid_delay_ms': float(1000.0 * delays[idx]),
            'position_rmse_m': float(np.sqrt(mean_pos_mse[idx])),
            'velocity_rmse_mps': float(np.sqrt(mean_vel_mse[idx])),
        }

    comparisons = {
        'zero_ms': grid_metrics(0.0),
        'true_midpoint_25ms': grid_metrics(0.025),
        'fixed_65ms': grid_metrics(0.065),
    }
    if bo_a is not None and bo_b is not None:
        comparisons['bo_midpoint'] = grid_metrics(float(bo_a) + 0.5 * float(bo_b))

    per_pos_best = np.asarray([row['best_position_delay_ms'] for row in per_throw_rows], dtype=float)
    per_vel_best = np.asarray([row['best_velocity_delay_ms'] for row in per_throw_rows], dtype=float)
    summary = {
        'description': 'Direct comparison of first cached free-flight state against analytical release-state mapping; no GP/BO loss is involved.',
        'num_throws': int(len(trial_paths)),
        'delay_grid_min_ms': float(1000.0 * delays[0]),
        'delay_grid_max_ms': float(1000.0 * delays[-1]),
        'delay_grid_step_ms': step_ms,
        'global_position_best_delay_ms': float(1000.0 * delays[global_pos_idx]),
        'global_position_best_rmse_m': float(np.sqrt(mean_pos_mse[global_pos_idx])),
        'global_velocity_best_delay_ms': float(1000.0 * delays[global_vel_idx]),
        'global_velocity_best_rmse_mps': float(np.sqrt(mean_vel_mse[global_vel_idx])),
        'per_throw_position_best_delay_ms': {
            'mean': float(np.mean(per_pos_best)),
            'median': float(np.median(per_pos_best)),
            'std': float(np.std(per_pos_best)),
            'min': float(np.min(per_pos_best)),
            'max': float(np.max(per_pos_best)),
        },
        'per_throw_velocity_best_delay_ms': {
            'mean': float(np.mean(per_vel_best)),
            'median': float(np.median(per_vel_best)),
            'std': float(np.std(per_vel_best)),
            'min': float(np.min(per_vel_best)),
            'max': float(np.max(per_vel_best)),
        },
        'first_freeflight_sample_spacing_s': (
            {
                'mean': float(np.mean(first_sample_dt)),
                'median': float(np.median(first_sample_dt)),
                'min': float(np.min(first_sample_dt)),
                'max': float(np.max(first_sample_dt)),
            }
            if first_sample_dt else None
        ),
        'reference_delay_comparisons': comparisons,
    }

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    grid_csv = os.path.join(output_dir, 'release_state_delay_grid.csv')
    per_throw_csv = os.path.join(output_dir, 'release_state_per_throw.csv')
    summary_json = os.path.join(output_dir, 'release_state_summary.json')

    with open(grid_csv, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=[
            'delay_ms', 'position_mse_m2', 'position_rmse_m',
            'velocity_mse_m2ps2', 'velocity_rmse_mps',
        ])
        writer.writeheader()
        for idx, delay_seconds in enumerate(delays):
            writer.writerow({
                'delay_ms': float(1000.0 * delay_seconds),
                'position_mse_m2': float(mean_pos_mse[idx]),
                'position_rmse_m': float(np.sqrt(mean_pos_mse[idx])),
                'velocity_mse_m2ps2': float(mean_vel_mse[idx]),
                'velocity_rmse_mps': float(np.sqrt(mean_vel_mse[idx])),
            })

    with open(per_throw_csv, 'w', newline='') as output_file:
        fieldnames = list(per_throw_rows[0].keys()) if per_throw_rows else []
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_throw_rows)

    with open(summary_json, 'w') as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)

    print('\n==========================================================')
    print('DIRECT RELEASE-STATE DELAY DIAGNOSTIC')
    print('No GP, no BO, no robot throws.')
    print('Throws: {}'.format(len(trial_paths)))
    print('Delay grid: {:.1f}--{:.1f} ms in {:.1f} ms steps'.format(
        1000.0 * delays[0], 1000.0 * delays[-1], step_ms))
    print('Observed state: first cached odometry sample after physical detachment')
    if first_sample_dt:
        print('Typical free-flight odometry spacing: median {:.2f} ms'.format(
            1000.0 * float(np.median(first_sample_dt))))
    print('----------------------------------------------------------')
    print('POSITION: global best delay = {:.1f} ms | RMSE = {:.4f} m'.format(
        1000.0 * delays[global_pos_idx], np.sqrt(mean_pos_mse[global_pos_idx])))
    print('VELOCITY: global best delay = {:.1f} ms | RMSE = {:.4f} m/s'.format(
        1000.0 * delays[global_vel_idx], np.sqrt(mean_vel_mse[global_vel_idx])))
    print('Per-throw best delay (position): median {:.1f} ms, mean {:.1f} ms, std {:.1f} ms'.format(
        np.median(per_pos_best), np.mean(per_pos_best), np.std(per_pos_best)))
    print('Per-throw best delay (velocity): median {:.1f} ms, mean {:.1f} ms, std {:.1f} ms'.format(
        np.median(per_vel_best), np.mean(per_vel_best), np.std(per_vel_best)))
    print('----------------------------------------------------------')
    for name, metrics in comparisons.items():
        print('{:>20}: grid {:.1f} ms | pos RMSE {:.4f} m | vel RMSE {:.4f} m/s'.format(
            name,
            metrics['grid_delay_ms'],
            metrics['position_rmse_m'],
            metrics['velocity_rmse_mps'],
        ))
    print('----------------------------------------------------------')
    print('Summary: {}'.format(summary_json))
    print('Grid: {}'.format(grid_csv))
    print('Per-throw: {}'.format(per_throw_csv))
    print('==========================================================\n')
    return summary


def run_delay_identifiability_diagnostic(
        PL_obj, init_function, a_hat, b_hat,
        true_residual_lower, true_residual_upper,
        output_dir, diagnostic_seed, a_points=21, b_points=21,
        delay_loss='trajectory_energy_score', num_particles=10, a_max_seconds=0.10,
        b_max_seconds=0.10, delay_diagnostic_skip_plots=False):
    """Evaluate the selected delay objective using common random numbers."""
    if int(a_points) < 2 or int(b_points) < 2:
        raise ValueError('delay diagnostic grid requires at least 2 points per axis')
    conditions = getattr(PL_obj, 'trajectories_initial_conditions', None)
    if not conditions:
        raise RuntimeError('No cached calibration landing conditions are available')

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    target_radii = np.asarray([
        np.linalg.norm(np.asarray(item[0], dtype=float)[:2]) for item in conditions
    ], dtype=float)
    commanded_speeds = np.asarray([float(item[1]) for item in conditions], dtype=float)
    radius_edges = np.linspace(float(np.min(target_radii)) - 1e-12,
                               float(np.max(target_radii)) + 1e-12, 6)
    speed_edges = np.linspace(float(np.min(commanded_speeds)) - 1e-12,
                              float(np.max(commanded_speeds)) + 1e-12, 6)

    def bin_means(values, errors, edges):
        output = []
        values = np.asarray(values)
        errors = np.asarray(errors)
        for idx in range(len(edges) - 1):
            if idx == len(edges) - 2:
                mask = (values >= edges[idx]) & (values <= edges[idx + 1])
            else:
                mask = (values >= edges[idx]) & (values < edges[idx + 1])
            output.append({
                'lower': float(edges[idx]),
                'upper': float(edges[idx + 1]),
                'count': int(np.sum(mask)),
                'mean_mse': float(np.mean(errors[mask])) if np.any(mask) else None,
            })
        return output

    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def evaluate_distribution(name, dist_fun):
        np.random.seed(int(diagnostic_seed))
        torch.manual_seed(int(diagnostic_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(diagnostic_seed))
        per_throw_loss, mean_loss, _, loss_diagnostics = PL_obj.simulate_system(
            dist_fun, conditions, init_function, particle_pred=True,
            num_particles_test=num_particles, delay_loss=delay_loss,
            return_diagnostics=True,
        )
        per_throw_loss = np.asarray(per_throw_loss, dtype=float)
        per_throw_mse = np.asarray(
            loss_diagnostics['per_throw_landing_mse'], dtype=float
        )
        return {
            'name': name,
            'loss_name': delay_loss,
            'selected_loss_value': float(mean_loss),
            'per_throw_selected_loss': [float(x) for x in per_throw_loss],
            'mean_landing_mse': float(loss_diagnostics['mean_landing_mse']),
            'per_throw_mse': [float(x) for x in per_throw_mse],
            'by_target_radius': bin_means(target_radii, per_throw_loss, radius_edges),
            'by_commanded_speed': bin_means(commanded_speeds, per_throw_loss, speed_edges),
        }

    try:
        true_midpoint = 0.5 * (true_residual_lower + true_residual_upper)
        reference_cases = [
            evaluate_distribution('ground_truth_uniform',
                                  lambda: np.random.uniform(true_residual_lower, true_residual_upper)),
            evaluate_distribution('fixed_true_midpoint', lambda: true_midpoint),
            evaluate_distribution('fixed_60ms', lambda: 0.060),
            evaluate_distribution('bo_estimated_uniform',
                                  lambda: np.random.uniform(a_hat, a_hat + b_hat)),
            evaluate_distribution('no_residual_delay', lambda: 0.0),
        ]

        a_values = np.linspace(0.0, float(a_max_seconds), int(a_points))
        b_values = np.linspace(0.0, float(b_max_seconds), int(b_points))
        grid_rows = []
        best_row = None

        print('\n----- Delay identifiability diagnostic (common random numbers) -----')
        print('Grid: {} x {} = {} candidates'.format(
            len(a_values), len(b_values), len(a_values) * len(b_values)))
        print('Selected delay loss: {}'.format(delay_loss))
        print('Particles per calibration throw: {}'.format(num_particles))
        print('Calibration throws per candidate: {}'.format(len(conditions)))
        print('Common-random-number seed: {}'.format(diagnostic_seed))

        for a_idx, a_value in enumerate(a_values):
            for b_value in b_values:
                result = evaluate_distribution(
                    'grid',
                    lambda a=float(a_value), b=float(b_value): np.random.uniform(a, a + b),
                )
                row = {
                    'a_seconds': float(a_value),
                    'b_seconds': float(b_value),
                    'upper_seconds': float(a_value + b_value),
                    'loss_name': delay_loss,
                    'selected_loss_value': float(result['selected_loss_value']),
                    'mean_landing_mse': float(result['mean_landing_mse']),
                    'by_target_radius': result['by_target_radius'],
                    'by_commanded_speed': result['by_commanded_speed'],
                }
                grid_rows.append(row)
                if best_row is None or row['selected_loss_value'] < best_row['selected_loss_value']:
                    best_row = row
            print('Diagnostic grid row {}/{} complete'.format(a_idx + 1, len(a_values)))

        csv_path = os.path.join(output_dir, 'delay_identifiability_grid.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            fieldnames = ['a_seconds', 'b_seconds', 'upper_seconds',
                          'loss_name', 'selected_loss_value', 'mean_landing_mse'] + [
                'radius_bin_{}_mse'.format(i) for i in range(5)
            ] + ['speed_bin_{}_mse'.format(i) for i in range(5)]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in grid_rows:
                flat = {
                    'a_seconds': row['a_seconds'], 'b_seconds': row['b_seconds'],
                    'upper_seconds': row['upper_seconds'],
                    'loss_name': row['loss_name'],
                    'selected_loss_value': row['selected_loss_value'],
                    'mean_landing_mse': row['mean_landing_mse'],
                }
                for i, entry in enumerate(row['by_target_radius']):
                    flat['radius_bin_{}_mse'.format(i)] = entry['mean_mse']
                for i, entry in enumerate(row['by_commanded_speed']):
                    flat['speed_bin_{}_mse'.format(i)] = entry['mean_mse']
                writer.writerow(flat)

        per_throw_path = os.path.join(output_dir, 'delay_reference_cases_per_throw.csv')
        with open(per_throw_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['case', 'throw_index', 'target_radius_m',
                             'commanded_speed_mps', 'landing_mse'])
            for case in reference_cases:
                for idx, mse in enumerate(case['per_throw_mse']):
                    writer.writerow([case['name'], idx, target_radii[idx],
                                     commanded_speeds[idx], mse])

        summary = {
            'diagnostic_seed': int(diagnostic_seed),
            'common_random_numbers': True,
            'loss_name': delay_loss,
            'particles_per_throw': int(num_particles),
            'num_calibration_throws': int(len(conditions)),
            'grid': {
                'a_range_seconds': [0.0, float(a_max_seconds)], 'a_points': int(a_points),
                'b_range_seconds': [0.0, float(b_max_seconds)], 'b_points': int(b_points),
                'num_candidates': int(len(grid_rows)), 'best': best_row,
            },
            'reference_parameters': {
                'ground_truth_uniform': [float(true_residual_lower), float(true_residual_upper)],
                'fixed_true_midpoint_seconds': float(true_midpoint),
                'fixed_60ms_seconds': 0.060,
                'bo_estimated_uniform': [float(a_hat), float(a_hat + b_hat)],
            },
            'target_radius_bin_edges_m': [float(x) for x in radius_edges],
            'commanded_speed_bin_edges_mps': [float(x) for x in speed_edges],
            'reference_cases': reference_cases,
        }
        loss_grid = np.asarray([row['selected_loss_value'] for row in grid_rows], dtype=float).reshape(
            len(a_values), len(b_values)).T
        min_loss = float(np.min(loss_grid))
        true_a = float(true_residual_lower)
        true_b = float(true_residual_upper - true_residual_lower)
        true_i = int(np.argmin(np.abs(a_values - true_a)))
        true_j = int(np.argmin(np.abs(b_values - true_b)))
        cell_area_ms2 = (1000.0 * (a_values[1] - a_values[0]) *
                         1000.0 * (b_values[1] - b_values[0]))
        summary['grid_analysis'] = {
            'minimum': {'a_seconds': float(a_values[np.unravel_index(np.argmin(loss_grid), loss_grid.shape)[1]]),
                        'b_seconds': float(b_values[np.unravel_index(np.argmin(loss_grid), loss_grid.shape)[0]]),
                        'loss': min_loss},
            'true_grid_point': {'a_seconds': float(a_values[true_i]), 'b_seconds': float(b_values[true_j]),
                                'loss': float(loss_grid[true_j, true_i]),
                                'relative_excess': float((loss_grid[true_j, true_i] - min_loss) / max(abs(min_loss), 1e-15))},
            'near_minimum': {str(pct) + '_percent': {
                'count': int(np.sum(loss_grid <= min_loss * (1.0 + pct / 100.0))),
                'area_ms2': float(np.sum(loss_grid <= min_loss * (1.0 + pct / 100.0)) * cell_area_ms2),
                'a_range_ms': [float(1000*a_values[np.where(np.any(loss_grid <= min_loss * (1.0 + pct / 100.0), axis=0))[0][0]]), float(1000*a_values[np.where(np.any(loss_grid <= min_loss * (1.0 + pct / 100.0), axis=0))[0][-1]])],
                'b_range_ms': [float(1000*b_values[np.where(np.any(loss_grid <= min_loss * (1.0 + pct / 100.0), axis=1))[0][0]]), float(1000*b_values[np.where(np.any(loss_grid <= min_loss * (1.0 + pct / 100.0), axis=1))[0][-1]])],
            } for pct in (1, 5, 10)},
        }
        np.savez_compressed(os.path.join(output_dir, 'delay_identifiability_grid.npz'),
                            a_seconds=a_values, b_seconds=b_values, loss=loss_grid,
                            relative_excess=(loss_grid-min_loss)/max(abs(min_loss), 1e-15))
        summary_path = os.path.join(output_dir, 'delay_identifiability_summary.json')
        with open(summary_path, 'w') as summary_file:
            json.dump(summary, summary_file, indent=2)
        if delay_diagnostic_skip_plots:
            return summary
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, values, title in ((axes[0], loss_grid, 'loss'),
                                  (axes[1], (loss_grid-min_loss)/max(abs(min_loss), 1e-15), 'relative excess loss')):
            image = ax.contourf(1000*a_values, 1000*b_values, values, levels=30, cmap='viridis')
            fig.colorbar(image, ax=ax, label=title)
            best_j, best_i = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
            ax.plot(1000*true_a, 1000*true_b, 'w*', ms=11, mec='k', label='true U(20,30)')
            ax.plot(1000*a_values[best_i], 1000*b_values[best_j], 'rx', ms=9, mew=2, label='grid minimum')
            ax.plot(1000*a_hat, 1000*b_hat, 'co', ms=6, label='BO estimate')
            ax.set(xlabel='a [ms]', ylabel='b [ms]', title=title)
            ax.legend(fontsize=8)
        fig.savefig(os.path.join(output_dir, 'delay_identifiability_heatmaps.png'), dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        axes[0].plot(1000*a_values, loss_grid[true_j, :]); axes[0].set(xlabel='a [ms] (b=10 ms)', ylabel='loss')
        axes[1].plot(1000*b_values, loss_grid[:, true_i]); axes[1].set(xlabel='b [ms] (a=20 ms)', ylabel='loss')
        fig.savefig(os.path.join(output_dir, 'delay_identifiability_slices.png'), dpi=180)
        plt.close(fig)
        print('\nReference cases:')
        for case in reference_cases:
            print('  {:>24s}: selected={:.8f}, landing MSE={:.8f}'.format(
                case['name'], case['selected_loss_value'], case['mean_landing_mse']))
        print('Best grid candidate: U({:.1f}, {:.1f}) ms, {}={:.8f}'.format(
            1000.0 * best_row['a_seconds'], 1000.0 * best_row['upper_seconds'],
            delay_loss, best_row['selected_loss_value']))
        print('Diagnostic summary: {}'.format(summary_path))
        print('Diagnostic grid: {}'.format(csv_path))
        print('Reference per-throw errors: {}'.format(per_throw_path))
        return summary
    finally:
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


# Configure the physical simulator explicitly for runs that interact with ROS.
# Diagnostic-only mode uses only cached trajectories + the learned GP, so it
# deliberately avoids touching the ROS parameter server and can run offline.
if not flg_delay_diagnostic_only:
    configure_physical_release_delay(
        physical_delay_lower_seconds,
        physical_delay_upper_seconds,
        use_synthetic_ground_truth=synthetic_ground_truth,
    )
    rospy.set_param('/tossing/release_delay/anticipation_seconds', 0.0)
    reset_physical_delay_rng(seed)
else:
    print('Diagnostic-only mode: skipping ROS physical-delay configuration')
# model = 'delta_state'

# Default data type
dtype = torch.float64

# Set the device
device = torch.device(device_name)

# Set number of computational threads
torch.set_num_threads(num_threads)
print('---- Set environment parameters ----')

# Set task parameters
u_max = V_MAX


dz = target_altitude - release_position[2]  # Fix target at 0.10m from ground


#T_sampling = 0.015
# T_sampling = 0.02
T_sampling = 0.01
T_exploration = 1.5
T_control = 1.0
state_dim = 3 * 2  # position + velocity
target_dim = 3  # target is a postion
input_dim = 3  # policy controls a cartesian (initial) velocity
num_gp = int(state_dim / 2)  # Model velocities only
gp_input_dim = int(state_dim / 2)
std_position_noise = float(std_meas_noise)
std_velocity_noise = float(std_velocity_meas_noise)
std_list = (
    [std_position_noise] * int(state_dim / 2)
    + [std_velocity_noise] * int(state_dim / 2)
    + [0] * target_dim
)

print('---- Set model learning parameters ----')
## set approximation
# None --> Exact GP
# SOD --> Subset of data
# SOR --> Subset of regressors
# nystrom --> Nystrom approximation
flg_approximation = 'SOD'  # None 'SOD' 'SOR' 'nystrom'
## set GP mean

## set normalization flag
flg_norm = False  # if True normalize data to train the GP
flg_norm_mean = False  # if True and flg_norm=True mean is removed
# set particles sampling strategy
# flg_parametric_model = False --> rollout are performed sampling from the Y posterior distribution
# flg_parametric_model = True -->  GP models are PHI(X)*W, MC-PILCO samples only one time from the W posterir
#                                  instead of sampling from the Y posterior
#                                  can be used with (nystrom, SOR)
flg_parametric_model = False

## set model
if model == 'Speed':
    print('Speed model RBF')
    num_gp = 3
    f_model_learning, model_learning_par = tossing_utils.get_speed_model(num_gp, T_sampling, device, dtype, flg_GP_mean,
                                                                         flg_approximation=flg_approximation)
elif model == 'Speed_poly':
    print('Speed model RBF+MPK')
    num_gp = 3
    f_model_learning, model_learning_par = tossing_utils.get_speed_model(num_gp, T_sampling, device, dtype, flg_GP_mean,
                                                                         add_MPK=True)
else:
    print('delta model RBF')
    num_gp = 6
    f_model_learning, model_learning_par = tossing_utils.get_delta_state_model(num_gp, T_sampling, device, dtype, flg_GP_mean)



print('\n---- Set exploration policy ----')
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = {}
rand_exploration_policy_par['state_dim'] = state_dim
rand_exploration_policy_par['input_dim'] = input_dim
rand_exploration_policy_par['u_max'] = u_max
rand_exploration_policy_par['dtype'] = dtype
rand_exploration_policy_par['device'] = device

print('\n---- Set control policy ----')
policy_reinit_dict, control_policy_par, f_control_policy = tossing_utils.getPolicy(u_max, state_dim, target_dim,
                                                                                   device, dtype, max_target_dist,
                                                                                   centers_init=centers_init_type)


print('\n---- Set cost function ---- \n Type: {}'.format(cost_type))
if cost_type == 'cumulative':
    f_cost_function = Cost_function.Tossing_bot_cost_cumulattive
    cost_function_par = {'dtype': dtype, 'device': device,
                         'target_indeces': list(range(state_dim, state_dim + target_dim)),
                         'lengthscales': torch.tensor(tossing_utils.cumulative_cost_lengthscales, dtype=dtype, device=device),
                         'pos_indeces': list(range(int(state_dim / 2)))}
else:
    f_cost_function = Cost_function.Tossing_bot_cost
    cost_function_par = {'dtype': dtype, 'device': device,
                         'target_indeces': list(range(state_dim, state_dim + target_dim)),
                         'lengthscales': torch.tensor(tossing_utils.last_pos_cost_lengthscales, dtype=dtype, device=device),
                         'pos_indeces': list(range(int(state_dim / 2)))}


Cost_function.Tossing_bot_cost(**cost_function_par)

print('\n---- Init policy learning object ----')
MC_PILCO_init_dict = {}
MC_PILCO_init_dict['f_sim'] = None
MC_PILCO_init_dict['T_sampling'] = T_sampling
MC_PILCO_init_dict['state_dim'] = state_dim + target_dim
MC_PILCO_init_dict['input_dim'] = input_dim
MC_PILCO_init_dict['std_meas_noise'] = np.array(std_list)


MC_PILCO_init_dict['f_model_learning'] = f_model_learning
MC_PILCO_init_dict['model_learning_par'] = model_learning_par
MC_PILCO_init_dict['f_rand_exploration_policy'] = f_rand_exploration_policy
MC_PILCO_init_dict['rand_exploration_policy_par'] = rand_exploration_policy_par
MC_PILCO_init_dict['f_control_policy'] = f_control_policy
MC_PILCO_init_dict['control_policy_par'] = control_policy_par

MC_PILCO_init_dict['log_path'] = (
    policy_only_output_dir
    if policy_only_mode
    else (
        run_output_dir
        if run_output_dir
        else 'results_tmp/' + str(seed)
    )
)
os.makedirs(MC_PILCO_init_dict['log_path'], exist_ok=True)
os.makedirs(os.path.join(MC_PILCO_init_dict['log_path'], 'cache_tossing_trials'), exist_ok=True)
MC_PILCO_init_dict['dtype'] = dtype
MC_PILCO_init_dict['device'] = device
MC_PILCO_init_dict['data_augmentation'] = data_augmentation
MC_PILCO_init_dict['data_augmentation_seed'] = seed
MC_PILCO_init_dict['data_augmentation_max_angle'] = np.deg2rad(
    data_augmentation_max_angle_degrees
)
MC_PILCO_init_dict['flg_apply_meas_noise_to_training_data'] = (
    flg_apply_meas_noise_to_training_data
)
MC_PILCO_init_dict['training_measurement_noise_seed'] = (
    effective_training_measurement_noise_seed
)
MC_PILCO_init_dict['flg_real_style_data_augmentation'] = (
    flg_real_style_data_augmentation
)

print('\n---- Set MC-PILCO options ----')
# Model optimization options
model_optimization_opt_dict = {}
model_optimization_opt_dict['f_optimizer'] = 'lambda p : torch.optim.Adam(p, lr=0.02)'
model_optimization_opt_dict['criterion'] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict['N_epoch'] = 1001
model_optimization_opt_dict['N_epoch_print'] = 500
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp
# Policy optimization options
policy_optimization_dict = {}
policy_optimization_dict['num_particles'] = num_particles
policy_optimization_dict['opt_steps_list'] = [1500] * (num_trials + num_exp) #[1001 + 100*i for i in range(num_trials+num_exp)]
policy_optimization_dict['lr_list'] = [0.01] * (num_trials+num_exp)
policy_optimization_dict['f_optimizer'] = 'lambda p, lr : torch.optim.Adam(p, lr)'
policy_optimization_dict['num_step_print'] = 100
if use_dropout:
    policy_optimization_dict['p_dropout_list'] = [.25] * (num_trials+num_exp)
    policy_optimization_dict['p_drop_reduction'] = 0.25 / 2
else:
    policy_optimization_dict['p_dropout_list'] = [0.0] * (num_trials+num_exp)
    policy_optimization_dict['p_drop_reduction'] = 0.0
policy_optimization_dict['alpha_diff_cost'] = 0.99
policy_optimization_dict['min_diff_cost'] = 0.04
policy_optimization_dict['num_min_diff_cost'] = 200
policy_optimization_dict['min_step'] = 200
policy_optimization_dict['lr_reduction_ratio'] = 0.5
policy_optimization_dict['lr_min'] = 0.0025
policy_optimization_dict['policy_reinit_dict'] = policy_reinit_dict
# Options for method reinforce
reinforce_param_dict = {}

reinforce_param_dict['initial_state_var'] = np.array([0.0] * (state_dim+target_dim))
reinforce_param_dict['T_exploration'] = T_exploration
reinforce_param_dict['T_control'] = T_control
reinforce_param_dict['model_optimization_opt_list'] = model_optimization_opt_list
reinforce_param_dict['policy_optimization_dict'] = policy_optimization_dict
reinforce_param_dict['num_explorations'] = num_exp

reinforce_param_dict['flg_init_func'] = True
def sample_task_initial_state():
    # Sample the *world-frame target radius* directly in [0.75, 2.40] m.
    # tossing_init() expects the displacement from the release radius, hence
    # subtract release_position[0] before passing the distance.
    target_radius_world = tossing_utils.min_distance + np.random.rand() * max_target_dist
    yaw = 2 * (np.random.rand() - 0.5) * np.pi / 6
    return tossing_utils.tossing_init(
        target_radius_world - release_position[0],
        yaw,
        target_altitude,
        release_pos=release_position,
    )

reinforce_param_dict['init_func'] = sample_task_initial_state
reinforce_param_dict['initial_state'] = [0]*(state_dim+target_dim)
reinforce_param_dict['initial_state_var'] = [0]*(state_dim+target_dim)

# Particles initialization dict
particles_initial_distribution_dict = {'initial_distribution': 'custom_function',
                                       'init_func': reinforce_param_dict['init_func'],
                                       'particles_initial_state_var': torch.tensor([10e-6]*(state_dim+target_dim),
                                                                                   device=device,
                                                                                   dtype=dtype)}  # very low variance for the gaussian

reinforce_param_dict['particles_initial_distribution_dict'] = particles_initial_distribution_dict

# Start the learning algorithm


# target_indeces, lengthscales, pos_indeces

MC_PILCO_init_dict['f_cost_function'] = f_cost_function
MC_PILCO_init_dict['cost_function_par'] = cost_function_par
MC_PILCO_init_dict['ros_write_proxy_name'] = 'tossing_exp_proxy_in'
MC_PILCO_init_dict['ros_read_proxy_name'] = 'tossing_exp_proxy_out'
MC_PILCO_init_dict['pos_indeces'] = list(range(3))
MC_PILCO_init_dict['vel_indeces'] = [3 + i for i in range(3)]
MC_PILCO_init_dict['target_dim'] = target_dim
MC_PILCO_init_dict['target_indeces'] = list(range(state_dim, state_dim+target_dim))
#filtering_dict
MC_PILCO_init_dict['filtering_dict'] = {'fc': 0.7}
MC_PILCO_init_dict['skip_fist_samples'] = 0
MC_PILCO_init_dict['verbose_plots'] = verbose
MC_PILCO_init_dict['bullet_name'] = bullet_name
MC_PILCO_init_dict['trials_data_save_path'] = MC_PILCO_init_dict['log_path'] + '/cache_tossing_trials/'
MC_PILCO_init_dict['load_cached_trials'] = flg_use_cache
MC_PILCO_init_dict['exploration_velocity_schedule'] = exploration_velocity_schedule
release_radial_velocity_compensation_coefficients = (
    panda_differential_kinematics_utils.RELEASE_RADIAL_VELOCITY_COMPENSATION_COEFFICIENTS
    if flg_compensate_release_radial_velocity else None
)
MC_PILCO_init_dict['release_radial_velocity_compensation_coefficients'] = (
    release_radial_velocity_compensation_coefficients
)


def tossing_init_gazebo_configured(v_norm, yaw, t_delay):
    return panda_differential_kinematics_utils.tossing_init_gazebo(
        v_norm,
        yaw,
        t_delay,
        compensate_radial_velocity=flg_compensate_release_radial_velocity,
    )


print(
    'Release radial-velocity compensation: {}'.format(
        'enabled, coefficients={}'.format(release_radial_velocity_compensation_coefficients)
        if flg_compensate_release_radial_velocity else 'disabled'
    )
)
if exploration_velocity_schedule is not None:
    print(
        'Absolute exploration velocity schedule [m/s]: {}'.format(
            exploration_velocity_schedule
        )
    )
MC_PILCO_init_dict['flg_simulate_diff_targets'] = flg_simulate_diff_targets
MC_PILCO_init_dict['release_position'] = release_position
MC_PILCO_init_dict['flg_save_all_particles_to_log'] = True


def jacobian_(q):
    """
        To account the rotation of 180° w.r.t. z (robot throws behind itself)
    """
    J = panda_differential_kinematics_utils.J(*q)
    J[0:2, :] = -J[0:2, :]
    return J

MC_PILCO_init_dict['jacobian'] = lambda q: jacobian_(q)

def forward_kinematics(q):
    """
        To account the rotation of 180° w.r.t. z (robot throws behind itself)
    """
    p = panda_differential_kinematics_utils.f_kin(*q)
    p[0:2, :] = -p[0:2, :]
    p[2, :] += 1.03
    return p

MC_PILCO_init_dict['forward_kinematics'] = lambda q: forward_kinematics(q)
MC_PILCO_init_dict['initial_cond_func'] = panda_differential_kinematics_utils.get_release_config

def get_targets(N):
    dist = tossing_utils.min_distance + torch.rand((N, 1)) * max_target_dist
    gamma = 2*(torch.rand((N, 1))-0.5) * np.pi/6

    targets = torch.zeros((N, state_dim+target_dim), device=device, dtype=dtype)
    targets[:, state_dim+0] = (torch.cos(gamma) * dist).reshape((N, ))
    targets[:, state_dim+1] = (torch.sin(gamma) * dist).reshape((N, ))
    targets[:, state_dim+2] = target_altitude

    return targets

MC_PILCO_init_dict['targets_dist_function'] = lambda N: get_targets(N)
MC_PILCO_init_dict['t_delay_dist'] = lambda: 0.0

if ctrl_noise_type == 'zero_mean':
    MC_PILCO_init_dict['sigma_dist_function'] = lambda N: 0.0
elif ctrl_noise_type == 'none':
    MC_PILCO_init_dict['sigma_dist_function'] = lambda N: 0.0 # TODO
else:
    MC_PILCO_init_dict['sigma_dist_function'] = lambda N: 0.0 # TODO


print('\n---- Save test configuration ----')
config_log_dict = {}
config_log_dict['MC_PILCO_init_dict'] = MC_PILCO_init_dict
config_log_dict['reinforce_param_dict'] = reinforce_param_dict

# Remember to use dill to save and load lambda
pkl.dump(
    config_log_dict,
    open(
        os.path.join(MC_PILCO_init_dict['log_path'], 'config_log.pkl'),
        'wb',
    ),
)

PL_obj = MC_PILCO_ros_envs.MC_PILCO_ROS_Tossing_Experiment(**MC_PILCO_init_dict)
PL_obj.sample_gp_posterior = flg_sample_gp_posterior

if flg_GP_mean:
    filename = 'tossing_t_delay_opt_results_real_data_speed_integration_with_mean_input_velocities_resamp.pkl'
else:
    filename = 'tossing_t_delay_opt_results_real_data_speed_integration_input_velocities_resamp.pkl'

# The default simulated actuator delay is U(0.12, 0.13) s. To mirror the
# real setup, apply a fixed 0.10 s coarse command anticipation before the five
# exploration throws. The true residual timing offset is then U(0.02,0.03) s.
#
# BO is therefore calibrated directly on compensated exploration data and
# estimates an effective residual distribution U(a_res, a_res+b_res).
# MC-PILOT propagates that entire fitted residual distribution.  The paired
# ablation uses the same physical delay and the same fixed anticipation but
# assumes zero residual delay during policy optimization.
a_hat = 0.0
b_hat = 0.0
anticipation_seconds = (
    float(release_anticipation_seconds)
    if flg_apply_release_anticipation
    else 0.0
)

if anticipation_seconds < 0.0:
    raise ValueError('release_anticipation_seconds must be non-negative')

# Apply the coarse anticipation *before* collecting exploration throws.  In
# diagnostic-only mode this value is used only analytically; no ROS call is
# needed because no physical/simulated throw is executed.
if not (flg_delay_diagnostic_only or flg_quick_trajectory_loss_only or flg_release_state_diagnostic_only):
    rospy.set_param(
        '/tossing/release_delay/anticipation_seconds',
        anticipation_seconds,
    )
print(
    "Fixed physical opening anticipation: {:.1f} ms".format(
        1000.0 * anticipation_seconds
    )
)

shared_delay_calibration_file = str(shared_delay_calibration_file).strip()

# -------------------------------------------------------------------------
# Direct release-state diagnostic: cached throws only, no GP and no BO
# -------------------------------------------------------------------------
if flg_release_state_diagnostic_only:
    if flg_delay_calibration_only or flg_delay_diagnostic_only or flg_quick_trajectory_loss_only:
        raise ValueError(
            '-flg_release_state_diagnostic_only is mutually exclusive with the other delay diagnostic/calibration modes'
        )

    release_cache_dir = str(delay_calibration_cache_dir).strip()
    if not release_cache_dir:
        release_cache_dir = MC_PILCO_init_dict['trials_data_save_path']
    release_cache_dir, release_trial_paths = _calibration_trial_paths(
        release_cache_dir, int(delay_calibration_throws)
    )

    release_bo_a = None
    release_bo_b = None
    try:
        local_bo_file = os.path.join(MC_PILCO_init_dict['log_path'], filename)
        release_bo_payload, release_bo_source = load_completed_delay_bo_payload(
            explicit_path=delay_diagnostic_bo_file,
            shared_path=shared_delay_calibration_file,
            local_path=local_bo_file,
        )
        _, _, _, release_trials_results = release_bo_payload
        _, release_opt = release_trials_results
        release_bo_a = float(release_opt['params']['a'])
        release_bo_b = float(release_opt['params']['b'])
        print('Reference BO estimate: U({:.1f}, {:.1f}) ms from {}'.format(
            1000.0 * release_bo_a,
            1000.0 * (release_bo_a + release_bo_b),
            release_bo_source,
        ))
    except FileNotFoundError:
        print('No completed BO result found; direct release-state diagnostic will continue without a BO reference.')

    print('\nRelease-state diagnostic -- CACHED FREE-FLIGHT DATA ONLY')
    print('Calibration cache: {}'.format(release_cache_dir))
    run_direct_release_state_diagnostic(
        init_function=tossing_init_gazebo_configured,
        trial_paths=release_trial_paths,
        output_dir=MC_PILCO_init_dict['log_path'],
        max_delay_seconds=float(release_state_diagnostic_max_delay_seconds),
        step_ms=float(release_state_diagnostic_step_ms),
        bo_a=release_bo_a,
        bo_b=release_bo_b,
    )
    print('Direct release-state diagnostic complete.')
    sys.exit(0)

# -------------------------------------------------------------------------
# Quick trajectory-loss comparison: cached throws + saved GP + completed BO
# -------------------------------------------------------------------------
if flg_quick_trajectory_loss_only:
    if flg_delay_calibration_only or flg_delay_diagnostic_only:
        raise ValueError(
            '-flg_quick_trajectory_loss_only is mutually exclusive with calibration/diagnostic modes'
        )

    quick_cache_dir = str(delay_calibration_cache_dir).strip()
    if not quick_cache_dir:
        quick_cache_dir = MC_PILCO_init_dict['trials_data_save_path']
    quick_cache_dir, quick_trial_paths = _calibration_trial_paths(
        quick_cache_dir, int(delay_calibration_throws)
    )
    cache_fingerprint = _calibration_cache_fingerprint(quick_trial_paths)
    _load_calibration_conditions_from_cache(PL_obj, quick_trial_paths)

    quick_checkpoint = str(delay_calibration_gp_checkpoint).strip()
    if not quick_checkpoint:
        quick_checkpoint = os.path.join(
            MC_PILCO_init_dict['log_path'], 'delay_calibration_gp_checkpoint.pt'
        )
    gp_source = ensure_calibration_gp(
        PL_obj=PL_obj,
        checkpoint_path=quick_checkpoint,
        cache_fingerprint=cache_fingerprint,
        trial_paths=quick_trial_paths,
        model_optimization_opt_list=model_optimization_opt_list,
        model_data_already_loaded=False,
    )

    local_bo_file = os.path.join(MC_PILCO_init_dict['log_path'], filename)
    bo_payload, bo_source = load_completed_delay_bo_payload(
        explicit_path=delay_diagnostic_bo_file,
        shared_path=shared_delay_calibration_file,
        local_path=local_bo_file,
    )
    _, _, _, quick_trials_results = bo_payload
    _, quick_opt = quick_trials_results
    quick_a = float(quick_opt['params']['a'])
    quick_b = float(quick_opt['params']['b'])
    true_residual_lower = physical_delay_lower_seconds - effective_anticipation_seconds
    true_residual_upper = physical_delay_upper_seconds - effective_anticipation_seconds

    print('\nQuick trajectory diagnostic -- NO ROBOT THROWS, NO BO, NO GRID')
    print('Calibration cache: {}'.format(quick_cache_dir))
    print('Calibration GP: {}'.format(gp_source))
    print('BO result: {}'.format(bo_source))
    print('Selected delay loss: {}'.format(delay_loss))

    run_quick_trajectory_loss_comparison(
        PL_obj=PL_obj,
        init_function=tossing_init_gazebo_configured,
        trial_paths=quick_trial_paths,
        a_hat=quick_a,
        b_hat=quick_b,
        true_residual_lower=true_residual_lower,
        true_residual_upper=true_residual_upper,
        diagnostic_seed=int(delay_diagnostic_seed),
        windows_seconds=(0.05, 0.10, 0.20, 0.30),
    )
    print('Quick trajectory-loss diagnostic complete.')
    sys.exit(0)

# -------------------------------------------------------------------------
# Standalone diagnostic: cached throws + completed BO only (never rerun BO)
# -------------------------------------------------------------------------
if flg_delay_diagnostic_only:
    if flg_delay_calibration_only:
        raise ValueError(
            '-flg_delay_diagnostic_only and -flg_delay_calibration_only are mutually exclusive'
        )

    diagnostic_cache_dir = str(delay_calibration_cache_dir).strip()
    if not diagnostic_cache_dir:
        diagnostic_cache_dir = MC_PILCO_init_dict['trials_data_save_path']
    diagnostic_cache_dir, diagnostic_trial_paths = _calibration_trial_paths(
        diagnostic_cache_dir, int(delay_calibration_throws)
    )
    cache_fingerprint = _calibration_cache_fingerprint(diagnostic_trial_paths)
    _load_calibration_conditions_from_cache(PL_obj, diagnostic_trial_paths)

    diagnostic_checkpoint = str(delay_calibration_gp_checkpoint).strip()
    if not diagnostic_checkpoint:
        diagnostic_checkpoint = os.path.join(
            MC_PILCO_init_dict['log_path'], 'delay_calibration_gp_checkpoint.pt'
        )

    diagnostic_gp_source_dir = str(delay_diagnostic_gp_source_dir).strip()
    if diagnostic_gp_source_dir:
        # Snapshot 4 is the GP immediately after the five saved exploration
        # throws.  This is the normal calibration GP, not a retrained proxy.
        # The archived snapshots were written on CUDA hosts.  Their tensors
        # are inference-only here, so map serialized storages to CPU before
        # dill unpickles log.pkl on a CPU-only offline workstation.
        torch.storage._load_from_bytes = lambda blob: torch.load(
            io.BytesIO(blob), map_location='cpu'
        )
        PL_obj.load_model_snapshot_from_log(
            data_collection_index=4, num_data_collections=5,
            folder=os.path.abspath(os.path.expanduser(diagnostic_gp_source_dir)),
        )
        PL_obj.model_learning.set_eval_mode()
        gp_source = 'saved snapshot: ' + diagnostic_gp_source_dir
    else:
        gp_source = ensure_calibration_gp(
            PL_obj=PL_obj,
            checkpoint_path=diagnostic_checkpoint,
            cache_fingerprint=cache_fingerprint,
            trial_paths=diagnostic_trial_paths,
            model_optimization_opt_list=model_optimization_opt_list,
            model_data_already_loaded=False,
        )

    local_bo_file = os.path.join(MC_PILCO_init_dict['log_path'], filename)
    bo_payload, bo_source = load_completed_delay_bo_payload(
        explicit_path=delay_diagnostic_bo_file,
        shared_path=shared_delay_calibration_file,
        local_path=local_bo_file,
    )
    _, _, _, diagnostic_trials_results = bo_payload
    _, diagnostic_opt = diagnostic_trials_results
    diagnostic_a = float(diagnostic_opt['params']['a'])
    diagnostic_b = float(diagnostic_opt['params']['b'])
    true_residual_lower = physical_delay_lower_seconds - anticipation_seconds
    true_residual_upper = physical_delay_upper_seconds - anticipation_seconds

    print('\n==========================================================')
    print('DELAY DIAGNOSTIC ONLY -- NO ROBOT THROWS, NO BO')
    print('Calibration cache: {}'.format(diagnostic_cache_dir))
    print('Calibration GP: {}'.format(gp_source))
    print('BO result: {}'.format(bo_source))
    print('BO estimate: U({:.1f}, {:.1f}) ms'.format(
        1000.0 * diagnostic_a, 1000.0 * (diagnostic_a + diagnostic_b)))
    print('==========================================================\n')

    run_delay_identifiability_diagnostic(
        PL_obj=PL_obj,
        init_function=tossing_init_gazebo_configured,
        a_hat=diagnostic_a,
        b_hat=diagnostic_b,
        true_residual_lower=true_residual_lower,
        true_residual_upper=true_residual_upper,
        output_dir=MC_PILCO_init_dict['log_path'],
        diagnostic_seed=int(delay_diagnostic_seed),
        a_points=int(delay_diagnostic_a_points),
        b_points=int(delay_diagnostic_b_points),
        delay_loss=delay_loss,
        num_particles=delay_bo_particles,
        a_max_seconds=float(delay_diagnostic_a_max_seconds),
        b_max_seconds=float(delay_diagnostic_b_max_seconds),
        delay_diagnostic_skip_plots=bool(delay_diagnostic_skip_plots),
    )
    print('Diagnostic-only mode complete.')
    sys.exit(0)

# -------------------------------------------------------------------------
# One-time shared delay calibration
# -------------------------------------------------------------------------
# This stage is deliberately separate from MC-PILOT learning.  The 50
# calibration trajectories are used only to train a temporary free-flight GP
# for the BO objective and to estimate the effective residual release-time
# distribution.  They are never reused as GP training data by the ten
# MC-PILOT seeds, which still receive exactly num_exp=5 exploration throws.
if flg_delay_calibration_only:
    if not shared_delay_calibration_file:
        raise ValueError(
            '-shared_delay_calibration_file is required when '
            '-flg_delay_calibration_only=True'
        )

    calibration_schedule, calibration_states = (
        make_stratified_delay_calibration_states(
            int(delay_calibration_throws),
            int(seed),
        )
    )
    calibration_state_index = {'value': 0}

    def calibration_initial_state():
        idx = calibration_state_index['value']
        if idx >= len(calibration_states):
            raise RuntimeError(
                'Calibration initial-state schedule exhausted at throw {}'.format(idx)
            )
        calibration_state_index['value'] += 1
        return calibration_states[idx]

    calibration_reinforce_par = dict(reinforce_param_dict)
    calibration_reinforce_par['num_explorations'] = int(delay_calibration_throws)
    calibration_reinforce_par['init_func'] = calibration_initial_state

    print('\n==========================================================')
    print('ONE-TIME SHARED DELAY CALIBRATION')
    print('Selected delay loss: {}'.format(delay_loss))
    print('Calibration throws: {}'.format(delay_calibration_throws))
    print('These throws will NOT be used by the MC-PILOT seed GPs.')
    print('==========================================================\n')

    reset_physical_delay_rng(seed)
    PL_obj.reinforce(
        **calibration_reinforce_par,
        random_initial_state=False,
        num_trials=0,
    )

    # Preserve the same BO search domain used by the previous simulation
    # protocol.  With 50 observations the estimator is better constrained,
    # but it is still allowed to return an effective residual that absorbs
    # moderate release-state/model mismatch.
    t_delay_left_lim = float(delay_bo_a_lower_seconds)
    t_delay_right_lim = float(delay_bo_a_upper_seconds)
    delta_delay = 0.10
    resolution = 100

    calibration_cache_dir, calibration_trial_paths = _calibration_trial_paths(
        MC_PILCO_init_dict['trials_data_save_path'], int(delay_calibration_throws)
    )
    cache_fingerprint = _calibration_cache_fingerprint(calibration_trial_paths)
    _load_calibration_conditions_from_cache(PL_obj, calibration_trial_paths)

    calibration_checkpoint = str(delay_calibration_gp_checkpoint).strip()
    if not calibration_checkpoint:
        calibration_checkpoint = os.path.join(
            MC_PILCO_init_dict['log_path'], 'delay_calibration_gp_checkpoint.pt'
        )

    # reinforce(..., num_trials=0) already added all 50 fresh trajectories to
    # model_learning.  Reuse a matching checkpoint if one exists; otherwise
    # train once and checkpoint the exact GP *before* starting BO.  Therefore
    # even an interrupted BO no longer loses the expensive GP optimization.
    ensure_calibration_gp(
        PL_obj=PL_obj,
        checkpoint_path=calibration_checkpoint,
        cache_fingerprint=cache_fingerprint,
        trial_paths=calibration_trial_paths,
        model_optimization_opt_list=model_optimization_opt_list,
        model_data_already_loaded=True,
    )

    print('\n\n----- Optimize delay distribution -----')
    pbounds = {
        'a': [t_delay_left_lim, t_delay_right_lim],
        'b': [
            float(delay_bo_b_lower_seconds),
            float(delay_bo_b_upper_seconds),
        ],
    }
    bo_res, bo_opt = PL_obj.bayes_optimization_t_delay_distribution(
        tossing_init_gazebo_configured,
        particle_pred=True,
        pbounds=pbounds,
        init_points=delay_bo_init_points,
        n_iter=delay_bo_iterations,
        random_state=effective_delay_bo_seed,
        num_particles_test=delay_bo_particles,
        delay_loss=delay_loss,
        common_random_seed=effective_delay_bo_seed,
    )
    trials_results = [bo_res, bo_opt]

    calibration_payload = [
        t_delay_left_lim, delta_delay, resolution, trials_results
    ]
    # Keep a local copy in the calibration run as well as the shared copy.
    with open(os.path.join(MC_PILCO_init_dict['log_path'], filename), 'wb') as f:
        pkl.dump(calibration_payload, f)

    _, calibration_opt = trials_results
    calibration_a = float(calibration_opt['params']['a'])
    calibration_b = float(calibration_opt['params']['b'])
    true_residual_lower = physical_delay_lower_seconds - anticipation_seconds
    true_residual_upper = physical_delay_upper_seconds - anticipation_seconds

    calibration_metadata = {
        'calibration_seed': int(seed),
        'num_calibration_throws': int(delay_calibration_throws),
        'calibration_target_sampling': {
            'type': 'five_equal_radial_strata_with_uniform_yaw',
            'target_radial_range_m': [
                float(tossing_utils.min_distance),
                float(tossing_utils.min_distance + max_target_dist),
            ],
            'yaw_range_rad': [-float(np.pi / 6.0), float(np.pi / 6.0)],
            'schedule': [
                {'target_radius_m': radius, 'yaw_rad': yaw}
                for radius, yaw in calibration_schedule
            ],
        },
        'physical_actuator_delay_seconds': {
            'distribution': 'uniform',
            'lower': float(physical_delay_lower_seconds),
            'upper': float(physical_delay_upper_seconds),
        },
        'fixed_coarse_anticipation_seconds': float(anticipation_seconds),
        'true_residual_delay_seconds': {
            'distribution': 'uniform',
            'lower': float(true_residual_lower),
            'upper': float(true_residual_upper),
        },
        'estimated_effective_residual_distribution': {
            'distribution': 'uniform',
            'a': calibration_a,
            'b': calibration_b,
            'lower': calibration_a,
            'upper': calibration_a + calibration_b,
        },
        'bo_search': {
            'loss_name': delay_loss,
            'selected_loss_value': float(-calibration_opt['target']),
            'a_bounds': [float(t_delay_left_lim), float(t_delay_right_lim)],
            'b_bounds': [
                float(delay_bo_b_lower_seconds),
                float(delay_bo_b_upper_seconds),
            ],
            'init_points': int(delay_bo_init_points),
            'guided_iterations': int(delay_bo_iterations),
            'bo_seed': effective_delay_bo_seed,
            'common_random_numbers': True,
            'particles_per_trajectory': int(delay_bo_particles),
        },
        'data_usage': {
            'temporary_calibration_gp_trajectories': int(delay_calibration_throws),
            'mcpilot_seed_gp_trajectories': 0,
            'reused_as_mcpilot_exploration': False,
        },
    }

    saved_calibration_path, saved_metadata_path = save_shared_delay_calibration(
        shared_delay_calibration_file,
        calibration_payload,
        calibration_metadata,
    )
    with open(
        os.path.join(MC_PILCO_init_dict['log_path'], 'shared_delay_calibration.json'),
        'w',
    ) as local_metadata_file:
        json.dump(calibration_metadata, local_metadata_file, indent=2)

    print(
        'Shared effective residual delay: U({:.1f}, {:.1f}) ms'.format(
            1000.0 * calibration_a,
            1000.0 * (calibration_a + calibration_b),
        )
    )
    print('Saved shared calibration to {}'.format(saved_calibration_path))
    print('Saved calibration metadata to {}'.format(saved_metadata_path))
    print('Calibration-only mode complete; MC-PILOT policy learning was not run.')
    sys.exit(0)

needs_delay_calibration = (
    (flg_model_release_delay or flg_load_t_delay_dist)
    and not policy_only_mode
    and not manual_delay_mode
)

if policy_only_mode:
    # The fitted or manually imposed distribution affects only MC-PILOT
    # rollouts.  The ROS/Gazebo physical residual remains independent.
    rospy.set_param(
        '/tossing/release_delay/anticipation_seconds',
        anticipation_seconds,
    )
    if policy_only_run_bo:
        # Installed after the frozen GP has been loaded and constrained BO has
        # completed below.
        MC_PILCO_init_dict['t_delay_dist'] = lambda: 0.0
        policy_delay_configuration = {
            'source': 'frozen_gp_constrained_bayesian_optimization',
            'a_bounds_seconds': [
                float(delay_bo_a_lower_seconds),
                float(delay_bo_a_upper_seconds),
            ],
            'b_bounds_seconds': [
                float(delay_bo_b_lower_seconds),
                float(delay_bo_b_upper_seconds),
            ],
            'init_points': int(delay_bo_init_points),
            'guided_iterations': int(delay_bo_iterations),
            'bo_seed': effective_delay_bo_seed,
            'common_random_numbers': True,
            'particles_per_trajectory': int(delay_bo_particles),
        }
    else:
        manual_delay_lower_seconds = float(manual_delay_lower_seconds)
        manual_delay_upper_seconds = float(manual_delay_upper_seconds)
        MC_PILCO_init_dict['t_delay_dist'] = lambda: np.random.uniform(
            low=manual_delay_lower_seconds,
            high=manual_delay_upper_seconds,
        )
        policy_delay_configuration = {
            'source': manual_delay_source,
            'distribution': 'uniform',
            'lower': manual_delay_lower_seconds,
            'upper': manual_delay_upper_seconds,
        }
    MC_PILCO_init_dict['load_cached_trials'] = False

    source_trial_paths = [
        os.path.join(
            policy_only_source_dir,
            'cache_tossing_trials',
            'trial_{}.pkl'.format(i),
        )
        for i in range(5)
    ]
    policy_only_metadata = {
        'seed': int(seed),
        'mode': (
            'frozen_gp_delay_bo_only'
            if flg_bo_only
            else 'frozen_gp_delay_bo_then_policy_optimization'
            if policy_only_run_bo
            else 'policy_optimization_only'
        ),
        'source_run_dir': policy_only_source_dir,
        'source_gp_log': os.path.join(policy_only_source_dir, 'log.pkl'),
        'source_gp_log_sha256': _file_sha256(
            os.path.join(policy_only_source_dir, 'log.pkl')
        ),
        'source_gp_snapshot_index': 4,
        'source_exploration_indices': [0, 1, 2, 3, 4],
        'source_exploration_fingerprint_sha256': (
            _calibration_cache_fingerprint(source_trial_paths)
        ),
        'gp_training_performed': False,
        'delay_bayesian_optimization_performed': bool(policy_only_run_bo),
        'on_system_training_rollout_performed': False,
        'policy_delay_configuration': policy_delay_configuration,
        'physical_actuator_delay_seconds': {
            'distribution': 'uniform',
            'lower': float(physical_delay_lower_seconds),
            'upper': float(physical_delay_upper_seconds),
        },
        'fixed_coarse_anticipation_seconds': float(anticipation_seconds),
        'physical_residual_delay_seconds': {
            'distribution': 'uniform',
            'lower': float(
                physical_delay_lower_seconds - anticipation_seconds
            ),
            'upper': float(
                physical_delay_upper_seconds - anticipation_seconds
            ),
        },
        'policy_optimization_seed': int(
            seed if policy_optimization_seed < 0 else policy_optimization_seed
        ),
    }
    with open(
        os.path.join(
            MC_PILCO_init_dict['log_path'],
            'policy_only_config.json',
        ),
        'w',
    ) as policy_only_config_file:
        json.dump(policy_only_metadata, policy_only_config_file, indent=2)

    # Rewrite the serialized configuration after installing the manual delay
    # sampler; the earlier generic snapshot still contained the zero-delay
    # placeholder used during object construction.
    config_log_dict['MC_PILCO_init_dict'] = MC_PILCO_init_dict
    pkl.dump(
        config_log_dict,
        open(
            os.path.join(MC_PILCO_init_dict['log_path'], 'config_log.pkl'),
            'wb',
        ),
    )

    if policy_only_run_bo:
        print(
            'Policy-only mode: frozen seed GP, constrained delay BO with b in [{:.1f}, {:.1f}] ms'.format(
                1000.0 * delay_bo_b_lower_seconds,
                1000.0 * delay_bo_b_upper_seconds,
            )
        )
    else:
        print(
            'Policy-only mode: frozen seed GP, manual rollout delay U({:.1f}, {:.1f}) ms'.format(
                1000.0 * manual_delay_lower_seconds,
                1000.0 * manual_delay_upper_seconds,
            )
        )

if needs_delay_calibration:
    print('\nDelay calibration loss: {}'.format(delay_loss))
    print('Delay calibration particles per throw: {}'.format(delay_bo_particles))
    using_shared_delay_calibration = bool(shared_delay_calibration_file)

    if using_shared_delay_calibration:
        # Shared calibration was obtained once from the independent 50-throw
        # calibration batch.  Do not collect or train anything here: each
        # MC-PILOT seed will collect its own five exploration throws below.
        t_delay_left_lim, delta_delay, resolution, trials_results = (
            load_shared_delay_calibration(shared_delay_calibration_file)
        )
        print(
            'Loaded shared delay calibration from {}'.format(
                os.path.abspath(os.path.expanduser(shared_delay_calibration_file))
            )
        )
    elif not flg_load_t_delay_dist:
        # Legacy behavior retained for backwards compatibility: estimate delay
        # from this seed's own exploration cache.  The paper runner no longer
        # uses this path when a shared calibration file is supplied.
        if not flg_use_cache:
            reset_physical_delay_rng(seed)
            PL_obj.reinforce(
                **reinforce_param_dict,
                random_initial_state=False,
                num_trials=0,
            )

        t_delay_left_lim = float(delay_bo_a_lower_seconds)
        t_delay_right_lim = float(delay_bo_a_upper_seconds)
        delta_delay = 0.10
        resolution = 100

        local_bo_bounds = {
            'a': [t_delay_left_lim, t_delay_right_lim],
            'b': [
                float(delay_bo_b_lower_seconds),
                float(delay_bo_b_upper_seconds),
            ],
        }

        print('Selected delay loss: {}'.format(delay_loss))
        trials_results = PL_obj.test_delay_dist_opitimization_by_trials(
            num_trials,
            model_optimization_opt_list=model_optimization_opt_list,
            f_init_particles=tossing_init_gazebo_configured,
            t_delay_left_lim=t_delay_left_lim,
            t_delay_right_lim=t_delay_right_lim,
            delta_delay=delta_delay,
            resolution=resolution,
            opt_type='bayes',
            verbose=True,
            particle_pred=True,
            add_data_model=flg_use_cache,
            pbounds=local_bo_bounds,
            bo_init_points=delay_bo_init_points,
            bo_iterations=delay_bo_iterations,
            bo_num_particles=delay_bo_particles,
            delay_loss=delay_loss,
            bo_random_state=effective_delay_bo_seed,
            bo_common_random_seed=effective_delay_bo_seed,
        )
        pkl.dump(
            [t_delay_left_lim, delta_delay, resolution, trials_results],
            open(MC_PILCO_init_dict['log_path'] + '/' + filename, 'wb'),
        )
    else:
        calibration_path = MC_PILCO_init_dict['log_path'] + '/' + filename
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(
                "Residual-delay calibration file not found: {}.".format(
                    calibration_path
                )
            )
        t_delay_left_lim, delta_delay, resolution, trials_results = pkl.load(
            open(calibration_path, 'rb')
        )

    res, opt = trials_results
    print('--------------------------')
    print(opt)
    print('--------------------------')

    a_hat = float(opt['params']['a'])
    b_hat = float(opt['params']['b'])

    print(
        "Estimated effective residual delay: U({:.1f}, {:.1f}) ms".format(
            1000.0 * a_hat,
            1000.0 * (a_hat + b_hat),
        )
    )

    # The delay calibration is performed after the initial deterministic
    # anticipation.  Its lower bound is therefore the additional anticipation
    # to apply.  Policy optimization propagates only the remaining width.
    estimated_full_lower = anticipation_seconds + a_hat
    estimated_full_upper = anticipation_seconds + a_hat + b_hat
    effective_anticipation_seconds = (
        estimated_full_lower if flg_model_release_delay else anticipation_seconds
    )
    rospy.set_param(
        '/tossing/release_delay/anticipation_seconds',
        effective_anticipation_seconds,
    )

    if flg_model_release_delay:
        MC_PILCO_init_dict['t_delay_dist'] = lambda: np.random.uniform(
            low=0.0,
            high=b_hat,
        )
        # Exploration uses the explicit residual U(20,30) ms synthetic path.
        # After estimating the lower bound, switch the synthetic simulator to
        # the equivalent full physical U(120,130) ms representation so that
        # the newly added anticipation changes the actual release instant too.
        if synthetic_ground_truth:
            synthetic_prefix = '/tossing/release_delay/synthetic_ground_truth'
            rospy.set_param(synthetic_prefix + '/min_ms', 1000.0 * physical_delay_lower_seconds)
            rospy.set_param(synthetic_prefix + '/max_ms', 1000.0 * physical_delay_upper_seconds)
            rospy.set_param(synthetic_prefix + '/compensate_command_anticipation', True)
    else:
        MC_PILCO_init_dict['t_delay_dist'] = lambda: 0.0

    if using_shared_delay_calibration:
        # The shared 50-throw calibration is independent of policy learning.
        # Proposed runs collect five fresh exploration throws; paired ablations
        # reuse only those five throws through flg_use_cache=True.
        MC_PILCO_init_dict['load_cached_trials'] = flg_use_cache
    else:
        # Legacy per-seed calibration reused the same exploration throws for BO
        # and subsequent MC-PILOT learning.
        MC_PILCO_init_dict['load_cached_trials'] = True

    true_residual_lower = physical_delay_lower_seconds - anticipation_seconds
    true_residual_upper = physical_delay_upper_seconds - anticipation_seconds

    compensation_log = {
        'seed': int(seed),
        'physical_actuator_delay_seconds': {
            'distribution': 'uniform',
            'lower': physical_delay_lower_seconds,
            'upper': physical_delay_upper_seconds,
        },
        'initial_anticipation_seconds': anticipation_seconds,
        'effective_anticipation_seconds': effective_anticipation_seconds,
        'estimated_full_effective_delay_seconds': {
            'distribution': 'uniform',
            'lower': estimated_full_lower,
            'upper': estimated_full_upper,
        },
        'true_residual_delay_seconds': {
            'distribution': 'uniform',
            'lower': true_residual_lower,
            'upper': true_residual_upper,
        },
        'estimated_effective_residual_distribution': {
            'a': a_hat,
            'b': b_hat,
            'lower': a_hat,
            'upper': a_hat + b_hat,
        },
        'release_radial_velocity_compensation': {
            'enabled': bool(flg_compensate_release_radial_velocity),
            'model': 'cubic_through_origin_mps',
            'training_runs': [
                '{}_energy_score'.format(value) for value in range(100, 108)
            ],
            'training_throws_per_run': 5,
            'coefficients_in_increasing_power_order': (
                list(release_radial_velocity_compensation_coefficients)
                if release_radial_velocity_compensation_coefficients is not None
                else list(
                    panda_differential_kinematics_utils.RELEASE_RADIAL_VELOCITY_COMPENSATION_COEFFICIENTS
                )
            ),
        },
        'delay_calibration_source': (
            {
                'type': 'shared_50_throw_calibration',
                'file': os.path.abspath(os.path.expanduser(shared_delay_calibration_file)),
            }
            if using_shared_delay_calibration
            else {'type': 'seed_exploration_legacy'}
        ),
        'delay_bayesian_optimization': {
            'loss_name': delay_loss,
            'selected_loss_value': float(-opt['target']),
            'a_bounds_seconds': [
                float(delay_bo_a_lower_seconds),
                float(delay_bo_a_upper_seconds),
            ],
            'b_bounds_seconds': [
                float(delay_bo_b_lower_seconds),
                float(delay_bo_b_upper_seconds),
            ],
            'init_points': int(delay_bo_init_points),
            'guided_iterations': int(delay_bo_iterations),
            'bo_seed': effective_delay_bo_seed,
            'common_random_numbers': True,
            'particles_per_trajectory': int(delay_bo_particles),
        },
        'model_residual_distribution': (
            {
                'distribution': 'uniform',
                'lower': 0.0,
                'upper': b_hat,
            }
            if flg_model_release_delay
            else {'distribution': 'fixed', 'value': 0.0}
        ),
    }
    with open(
        os.path.join(
            MC_PILCO_init_dict['log_path'],
            'delay_compensation_config.json',
        ),
        'w',
    ) as compensation_file:
        json.dump(compensation_log, compensation_file, indent=2)

    # Diagnostic evaluation uses the same residual-timing convention as BO:
    # all five exploration throws were already collected with the fixed coarse
    # anticipation active.
    if hasattr(PL_obj, 'trajectories_initial_conditions'):
        def evaluate_delay_distribution(name, dist_fun, evaluation_seed_local):
            np.random.seed(evaluation_seed_local)
            torch.manual_seed(evaluation_seed_local)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(evaluation_seed_local)
            per_throw_loss, mean_loss, _, loss_diagnostics = PL_obj.simulate_system(
                dist_fun,
                PL_obj.trajectories_initial_conditions,
                tossing_init_gazebo_configured,
                particle_pred=True,
                num_particles_test=delay_bo_particles,
                delay_loss=delay_loss,
                return_diagnostics=True,
            )
            return {
                'name': name,
                'loss_name': delay_loss,
                'selected_loss_value': float(mean_loss),
                'per_throw_selected_loss': [float(v) for v in per_throw_loss],
                'mean_landing_mse': float(loss_diagnostics['mean_landing_mse']),
                'per_throw_mse': [
                    float(v) for v in loss_diagnostics['per_throw_landing_mse']
                ],
            }

        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        delay_diagnostics = {
            'seed': int(seed),
            'loss_name': delay_loss,
            'selected_loss_value': float(-opt['target']),
            'fixed_coarse_anticipation_seconds': anticipation_seconds,
            'estimated_residual': {
                'a': a_hat,
                'b': b_hat,
                'lower': a_hat,
                'upper': a_hat + b_hat,
            },
            'ground_truth_residual': {
                'lower': true_residual_lower,
                'upper': true_residual_upper,
            },
            'objective_checks': [
                evaluate_delay_distribution(
                    'estimated_residual',
                    lambda: np.random.uniform(a_hat, a_hat + b_hat),
                    seed + 1000,
                ),
                evaluate_delay_distribution(
                    'ground_truth_residual',
                    lambda: np.random.uniform(
                        true_residual_lower,
                        true_residual_upper,
                    ),
                    seed + 1000,
                ),
                evaluate_delay_distribution(
                    'no_residual_delay',
                    lambda: 0.0,
                    seed + 1000,
                ),
                evaluate_delay_distribution(
                    'fixed_residual_midpoint',
                    lambda: 0.5 * (
                        true_residual_lower + true_residual_upper
                    ),
                    seed + 1000,
                ),
            ],
        }

        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        diagnostics_path = os.path.join(
            MC_PILCO_init_dict['log_path'],
            'delay_estimation_diagnostics.json',
        )
        with open(diagnostics_path, 'w') as diagnostics_file:
            json.dump(delay_diagnostics, diagnostics_file, indent=2)
        print('Delay diagnostics saved to {}'.format(diagnostics_path))
elif not policy_only_mode:
    # No residual model. Keep the requested fixed coarse anticipation active
    # (or zero for the optional uncompensated diagnostic).
    rospy.set_param(
        '/tossing/release_delay/anticipation_seconds',
        anticipation_seconds,
    )
    if manual_delay_mode and flg_model_release_delay:
        manual_delay_lower_seconds = float(manual_delay_lower_seconds)
        manual_delay_upper_seconds = float(manual_delay_upper_seconds)
        MC_PILCO_init_dict['t_delay_dist'] = lambda: np.random.uniform(
            low=manual_delay_lower_seconds,
            high=manual_delay_upper_seconds,
        )
        print(
            'Using exact/manual policy delay U({:.1f}, {:.1f}) ms; '
            'delay estimation skipped'.format(
                1000.0 * manual_delay_lower_seconds,
                1000.0 * manual_delay_upper_seconds,
            )
        )
        with open(
            os.path.join(MC_PILCO_init_dict['log_path'], 'exact_delay_config.json'),
            'w',
        ) as exact_delay_file:
            json.dump({
                'source': manual_delay_source,
                'distribution': 'uniform',
                'lower_seconds': manual_delay_lower_seconds,
                'upper_seconds': manual_delay_upper_seconds,
                'fixed_coarse_anticipation_seconds': float(
                    anticipation_seconds
                ),
                'estimated_total_delay_seconds': {
                    'lower': float(
                        anticipation_seconds + manual_delay_lower_seconds
                    ),
                    'upper': float(
                        anticipation_seconds + manual_delay_upper_seconds
                    ),
                },
                'delay_estimation_performed': False,
                'data_augmentation_copies': int(data_augmentation),
                'observed_trajectories': int(num_exp),
                'effective_training_trajectories': int(
                    num_exp * (1 + data_augmentation)
                ),
                'data_augmentation_max_angle_degrees': float(
                    data_augmentation_max_angle_degrees
                ),
                'std_meas_noise': float(std_meas_noise),
                'std_velocity_meas_noise': float(std_velocity_meas_noise),
                'training_measurement_noise_applied': bool(
                    flg_apply_meas_noise_to_training_data
                ),
                'training_measurement_noise_channels': (
                    ['position_m']
                    if flg_real_style_data_augmentation
                    else ['position_m', 'velocity_m_per_s']
                ),
                'training_measurement_noise_seed': int(
                    effective_training_measurement_noise_seed
                ),
                'augmentation_noise_mode': (
                    'shared_noisy_position_differentiated_velocity'
                    if flg_real_style_data_augmentation
                    else 'independent_per_copy_before_filtering'
                ),
                'augmentation_rotation_origin': 'world_z_axis',
                'augmentation_velocity_source': (
                    'differentiated_from_filtered_noisy_position'
                    if flg_real_style_data_augmentation
                    else 'filtered_measured_velocity'
                ),
                'policy_particles': int(num_particles),
                'sample_gp_predictive_posterior': bool(
                    flg_sample_gp_posterior
                ),
            }, exact_delay_file, indent=2)
    else:
        MC_PILCO_init_dict['t_delay_dist'] = lambda: 0.0
    MC_PILCO_init_dict['load_cached_trials'] = flg_use_cache


PL_obj = MC_PILCO_ros_envs.MC_PILCO_ROS_Tossing_Experiment(**MC_PILCO_init_dict)
PL_obj.sample_gp_posterior = flg_sample_gp_posterior

if policy_only_mode:
    # Snapshot 4 is the GP trained immediately after the five exploration
    # trajectories.  Load it directly and optimize only a fresh policy: do not
    # call reinforce(), because reinforce() would train the GP again and apply
    # the learned policy once on the robot.
    PL_obj.load_model_snapshot_from_log(
        data_collection_index=4,
        num_data_collections=5,
        folder=policy_only_source_dir,
    )
    PL_obj.model_learning.set_eval_mode()

    if policy_only_run_bo:
        # Reuse the five observed target/command/landing tuples for the BO
        # objective without adding them to or retraining the frozen GP.
        _load_calibration_conditions_from_cache(PL_obj, source_trial_paths)
        np.random.seed(effective_delay_bo_seed)
        torch.manual_seed(effective_delay_bo_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_delay_bo_seed)

        constrained_bo_bounds = {
            'a': [
                float(delay_bo_a_lower_seconds),
                float(delay_bo_a_upper_seconds),
            ],
            'b': [
                float(delay_bo_b_lower_seconds),
                float(delay_bo_b_upper_seconds),
            ],
        }
        print(
            'Running frozen-GP delay BO with bounds {}'.format(
                constrained_bo_bounds
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bo_start_time = time.perf_counter()
        bo_res, bo_opt = PL_obj.bayes_optimization_t_delay_distribution(
            tossing_init_gazebo_configured,
            particle_pred=True,
            pbounds=constrained_bo_bounds,
            init_points=delay_bo_init_points,
            n_iter=delay_bo_iterations,
            random_state=effective_delay_bo_seed,
            num_particles_test=delay_bo_particles,
            delay_loss=delay_loss,
            common_random_seed=effective_delay_bo_seed,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bo_execution_seconds = time.perf_counter() - bo_start_time
        a_hat = float(bo_opt['params']['a'])
        b_hat = float(bo_opt['params']['b'])
        fitted_delay_sampler = lambda: np.random.uniform(
            low=a_hat,
            high=a_hat + b_hat,
        )
        MC_PILCO_init_dict['t_delay_dist'] = fitted_delay_sampler
        PL_obj.t_delay_dist = fitted_delay_sampler

        constrained_bo_payload = [
            float(delay_bo_a_lower_seconds),
            float(delay_bo_a_upper_seconds - delay_bo_a_lower_seconds),
            None,
            [bo_res, bo_opt],
        ]
        with open(
            os.path.join(
                MC_PILCO_init_dict['log_path'],
                'constrained_delay_bo_results.pkl',
            ),
            'wb',
        ) as constrained_bo_file:
            pkl.dump(constrained_bo_payload, constrained_bo_file)

        policy_only_metadata['policy_delay_configuration']['result'] = {
            'a': a_hat,
            'b': b_hat,
            'lower': a_hat,
            'upper': a_hat + b_hat,
            'loss_name': delay_loss,
            'selected_loss_value': float(-bo_opt['target']),
        }
        bo_timing = {
            'seed': int(seed),
            'bo_seed': int(effective_delay_bo_seed),
            'source_run_dir': policy_only_source_dir,
            'source_gp_log_sha256': policy_only_metadata['source_gp_log_sha256'],
            'source_exploration_fingerprint_sha256': policy_only_metadata[
                'source_exploration_fingerprint_sha256'
            ],
            'execution_seconds': float(bo_execution_seconds),
            'init_points': int(delay_bo_init_points),
            'guided_iterations': int(delay_bo_iterations),
            'total_evaluations': int(len(bo_res)),
            'particles_per_trajectory': int(delay_bo_particles),
            'num_trajectories': int(len(PL_obj.trajectories_initial_conditions)),
            'delay_loss': delay_loss,
            'device': str(device),
            'cuda_device_name': (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
            'result': policy_only_metadata['policy_delay_configuration']['result'],
        }
        policy_only_metadata['bo_timing'] = bo_timing
        with open(
            os.path.join(MC_PILCO_init_dict['log_path'], 'bo_execution_time.json'),
            'w',
        ) as bo_timing_file:
            json.dump(bo_timing, bo_timing_file, indent=2)
        with open(
            os.path.join(
                MC_PILCO_init_dict['log_path'],
                'policy_only_config.json',
            ),
            'w',
        ) as policy_only_config_file:
            json.dump(policy_only_metadata, policy_only_config_file, indent=2)
        config_log_dict['MC_PILCO_init_dict'] = MC_PILCO_init_dict
        pkl.dump(
            config_log_dict,
            open(
                os.path.join(
                    MC_PILCO_init_dict['log_path'],
                    'config_log.pkl',
                ),
                'wb',
            ),
        )
        print(
            'Constrained BO result: U({:.3f}, {:.3f}) ms (b={:.3f} ms)'.format(
                1000.0 * a_hat,
                1000.0 * (a_hat + b_hat),
                1000.0 * b_hat,
            )
        )
        print('BO execution time: {:.6f} s'.format(bo_execution_seconds))
        if flg_bo_only:
            print('BO-only mode complete; policy optimization was not run.')
            sys.exit(0)

    effective_policy_optimization_seed = int(
        seed if policy_optimization_seed < 0 else policy_optimization_seed
    )
    np.random.seed(effective_policy_optimization_seed)
    torch.manual_seed(effective_policy_optimization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_policy_optimization_seed)

    print(
        'Starting policy-only optimization with RNG seed {}'.format(
            effective_policy_optimization_seed
        )
    )
    cost_list, std_cost_list, particles_states, particles_inputs, opt_steps = (
        PL_obj.reinforce_policy(
            T_control=T_control,
            trial_index=4,
            particles_initial_distribution_dict=(
                particles_initial_distribution_dict
            ),
            **policy_optimization_dict
        )
    )

    torch.save(
        PL_obj.control_policy.state_dict(),
        os.path.join(MC_PILCO_init_dict['log_path'], 'policy.torch'),
    )
    PL_obj.log_dict['policy_only'] = policy_only_metadata
    PL_obj.log_dict['cost_trial_list'] = [cost_list]
    PL_obj.log_dict['std_cost_trial_list'] = [std_cost_list]
    PL_obj.log_dict['particles_states_list'] = [particles_states]
    PL_obj.log_dict['particles_inputs_list'] = [particles_inputs]
    PL_obj.log_dict['parameters_trial_list'] = [
        PL_obj.control_policy.state_dict()
    ]
    PL_obj.log_dict['opt_step_list'] = [int(opt_steps)]
    pkl.dump(
        PL_obj.log_dict,
        open(
            os.path.join(MC_PILCO_init_dict['log_path'], 'log.pkl'),
            'wb',
        ),
    )
    print('Policy-only optimization complete; no system rollout was collected.')
else:
    PL_obj.reinforce(
        **reinforce_param_dict,
        random_initial_state=False,
        num_trials=num_trials
    )

if flg_do_test:
    # Paired evaluation: both policy variants see exactly the same target RNG
    # sequence and the same physical-delay RNG sequence for a given seed.
    paired_eval_seed = int(evaluation_seed) + int(seed)
    np.random.seed(paired_eval_seed)
    torch.manual_seed(paired_eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(paired_eval_seed)
    reset_physical_delay_rng(paired_eval_seed)

    with open(
        os.path.join(MC_PILCO_init_dict['log_path'], 'evaluation_config.json'),
        'w',
    ) as evaluation_file:
        json.dump(
            {
                'base_evaluation_seed': int(evaluation_seed),
                'paired_evaluation_seed': paired_eval_seed,
                'target_radius_m': float(target_radius),
                'target_radial_range_m': [
                    float(tossing_utils.min_distance),
                    float(tossing_utils.min_distance + max_target_dist),
                ],
            },
            evaluation_file,
            indent=2,
        )

    results = {}
    for i in range(num_test_trials):
        results[i] = tossing_utils.test_policy(
            PL_obj.control_policy,
            num_toss_test,
            seed,
            reinforce_param_dict['init_func'],
            True,
            bullet_name=bullet_name,
            save_post_str='{}'.format(i+1),
            save_dir=MC_PILCO_init_dict['log_path'],
        )

    for i in range(num_test_trials):
        distances = [results[i][j][1] for j in results[i].keys()]
        print(
            'Performance: {}% target reach'.format(
                100 * np.count_nonzero(
                    np.abs(np.array(distances)) < target_radius
                ) / len(distances)
            )
        )

    # print('plot is just the last')
    # tossing_utils.plot_policy_test_results(results[num_test_trials-1], target_radius)

    # plt.show()
    plt.close('all')
