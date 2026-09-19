# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import logging
import math
import time

import mpmath
import numpy
import numpy as np
import std_msgs
import torch
import sys
import rospy
import json

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from policy_learning.MC_PILCO import MC_PILCO, MC_PILCO4PMS
from policy_learning.delay_calibration_losses import (
    DELAY_LOSSES,
    energy_score,
    interpolate_descending_plane_crossing,
    trajectory_energy_score,
)
from policy_learning.release_velocity_compensation import (
    apply_radial_velocity_compensation,
)
import bayes_opt

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import pickle as pkl


def _wait_for_ros_subscriber(publisher, topic_name, timeout_seconds=30.0):
    """Avoid dropping the non-latched one-shot request sent to the ROS proxy."""
    deadline = time.monotonic() + timeout_seconds
    while publisher.get_num_connections() == 0:
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException(
                'ROS shut down while waiting for a subscriber on {}'.format(topic_name)
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'No subscriber connected to {} after {:.0f} s. Start '
                'MC_PILCO_ros_tossing_proxy.py before running MC-PILOT.'.format(
                    topic_name, timeout_seconds
                )
            )
        rospy.sleep(0.1)


class MC_PILCO_ROS_Experiment(MC_PILCO4PMS):
    """
        Class that loads experiment data from a ros service that implements system interaction with simulation (or real env?)

    """

    def __init__(self, T_sampling, state_dim, input_dim, f_sim,
                 f_model_learning, model_learning_par,
                 f_rand_exploration_policy, rand_exploration_policy_par,
                 f_control_policy, control_policy_par,
                 f_cost_function, cost_function_par,
                 ros_write_proxy_name, ros_read_proxy_name,
                 pos_indeces, vel_indeces,
                 target_indeces, target_dim,
                 std_meas_noise=None, log_path=None,
                 filtering_dict={}, std_meas_noise_sim=None,
                 dtype=torch.float64, device=torch.device('cpu'), flg_save_all_particles_to_log=False):
        super(MC_PILCO_ROS_Experiment, self).__init__(T_sampling=T_sampling, state_dim=state_dim, input_dim=input_dim,
                                                      f_sim=f_sim, f_model_learning=f_model_learning,
                                                      model_learning_par=model_learning_par,
                                                      f_rand_exploration_policy=f_rand_exploration_policy,
                                                      rand_exploration_policy_par=rand_exploration_policy_par,
                                                      f_control_policy=f_control_policy,
                                                      control_policy_par=control_policy_par,
                                                      f_cost_function=f_cost_function,
                                                      cost_function_par=cost_function_par,
                                                      pos_indeces=pos_indeces, vel_indeces=vel_indeces,
                                                      std_meas_noise=std_meas_noise, log_path=log_path,
                                                      filtering_dict=filtering_dict,
                                                      std_meas_noise_sim=std_meas_noise_sim,
                                                      dtype=dtype, device=device,
                                                      flg_save_all_particles_to_log=flg_save_all_particles_to_log)
        self.ros_write_topic_name = ros_write_proxy_name
        self.ros_read_topic_name = ros_read_proxy_name
        self.target_indeces = target_indeces
        self.target_dim = target_dim

    def get_data_from_system(self, initial_state, T_exploration, trial_index, flg_exploration=False):
        """
            Forward
        """
        rospy.init_node("MC_PILCO_experiment")
        experiment_publisher = rospy.Publisher(self.ros_write_topic_name, std_msgs.msg.String, queue_size=10)

        policy_path = 0

        # select the policy
        if flg_exploration:
            print("Execute initial exploration policy")
        else:
            print("Export control policy parameters")
            policy_path = self.log_path + '/policy.torch'
            torch.save(self.control_policy.state_dict(), policy_path)

        # Write to topic
        experiment_params = [initial_state, T_exploration, trial_index, flg_exploration, policy_path]
        experiment_params_serialization = json.dumps(experiment_params)
        print('Sending message to ROS proxy topic: {}'.format(self.ros_write_topic_name))

        rate = rospy.Rate(1)  # 1hz
        for i in range(2):
            rate.sleep()
        rospy.loginfo(experiment_params_serialization)

        _wait_for_ros_subscriber(experiment_publisher, self.ros_write_topic_name)
        experiment_publisher.publish(experiment_params_serialization)

        print('Sent message to ROS proxy topic ')
        experiment = rospy.wait_for_message(self.ros_read_topic_name, std_msgs.msg.Float64MultiArray)
        """
            Check correctness of data loading
        """

        print('State dim: {}'.format(self.state_dim))

        # print('DATA ({}): {}'.format(len(experiment.data),experiment.data))

        positions_dim = len(self.pos_indeces)
        num_samples = int(
            len(experiment.data) / (
                    1 + positions_dim + self.input_dim))  # vector contains concatenation of timestamps + states + inputs
        noisy_samples = np.zeros((num_samples, self.state_dim + 1))  # timestamps + dimensions

        noisy_samples[:, 0] = [experiment.data[i] for i in range(num_samples)]
        for j in range(positions_dim):
            noisy_samples[:, 1 + self.pos_indeces[j]] = [experiment.data[i] for i in
                                                         range((j + 1) * num_samples, (j + 2) * num_samples)]

        for j in range(self.target_dim):
            print(initial_state[self.target_indeces[j]])
            noisy_samples[:, 1 + self.target_indeces[j]] = [initial_state[self.target_indeces[j]]] * num_samples

        input_samples = np.zeros((num_samples, self.input_dim))
        for j in range(self.input_dim):
            input_samples[:, j] = [experiment.data[i] for i in
                                   range((positions_dim + j) * num_samples, (positions_dim + j + 1) * num_samples)]

        # v = mpmath.norm(input_samples[0, :])
        # plt.figure()
        # ax = plt.axes(projection='3d')
        #
        # ax.plot3D(noisy_samples[:, self.pos_indeces[0]], noisy_samples[:, self.pos_indeces[1]], noisy_samples[:, self.pos_indeces[2]], label='trj_v={}'.format(v), linewidth=2.0)
        #
        # ax.axes.set_ylim3d(bottom=-0.1, top=0.1)
        # plt.legend()
        #
        # plt.savefig('trj_trial_{}'.format(trial_index+1))

        # print('States with computed velocities: {}'.format(noisy_samples))
        # print('Inputs forwarded to MC PILCO: {}'.format(input_samples))

        print(noisy_samples)
        meas_states = noisy_samples
        noiseless_samples = noisy_samples

        # approximate velocities using position measures
        state_samples, meas_states, input_samples, noiseless_samples, noisy_samples = self.get_velocities(meas_states,
                                                                                                          input_samples,
                                                                                                          noiseless_samples,
                                                                                                          noisy_samples)
        print(state_samples)
        print(input_samples)

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.num_data_collection += 1
        # add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

        def myhook():
            print("shutdown time!")

        rospy.on_shutdown(myhook)


class MC_PILCO_ROS_Tossing_Experiment(MC_PILCO_ROS_Experiment):
    """
        Class that loads experiment data from a ros service that implements system interaction with simulation (or real env?)
        of Tossing experiments.

    """

    def __init__(self, T_sampling, state_dim, input_dim, f_sim,
                 f_model_learning, model_learning_par,
                 f_rand_exploration_policy, rand_exploration_policy_par,
                 f_control_policy, control_policy_par,
                 f_cost_function, cost_function_par,
                 ros_write_proxy_name, ros_read_proxy_name,
                 pos_indeces, vel_indeces,
                 target_indeces, target_dim, release_position,
                 jacobian, forward_kinematics, initial_cond_func,
                 targets_dist_function, sigma_dist_function, t_delay_dist,
                 bullet_name='red_ball_friction',
                 skip_fist_samples=5,
                 std_meas_noise=None, log_path=None, flg_simulate_diff_targets=True,
                 filtering_dict={}, std_meas_noise_sim=None, trials_data_save_path=None, load_cached_trials=False,
                 exploration_velocity_schedule=None,
                 data_augmentation=0, data_augmentation_seed=None,
                 data_augmentation_max_angle=np.pi / 4,
                 flg_apply_meas_noise_to_training_data=False,
                 training_measurement_noise_seed=None,
                 flg_real_style_data_augmentation=False,
                 dtype=torch.float64, device=torch.device('cpu'), verbose_plots=False,
                 flg_save_all_particles_to_log=True,
                 release_radial_velocity_compensation_coefficients=None):
        super(MC_PILCO_ROS_Tossing_Experiment, self).__init__(T_sampling=T_sampling, state_dim=state_dim,
                                                              input_dim=input_dim,
                                                              f_sim=f_sim, f_model_learning=f_model_learning,
                                                              model_learning_par=model_learning_par,
                                                              f_rand_exploration_policy=f_rand_exploration_policy,
                                                              rand_exploration_policy_par=rand_exploration_policy_par,
                                                              f_control_policy=f_control_policy,
                                                              control_policy_par=control_policy_par,
                                                              f_cost_function=f_cost_function,
                                                              cost_function_par=cost_function_par,
                                                              pos_indeces=pos_indeces, vel_indeces=vel_indeces,
                                                              std_meas_noise=std_meas_noise, log_path=log_path,
                                                              filtering_dict=filtering_dict,
                                                              std_meas_noise_sim=std_meas_noise_sim,
                                                              dtype=dtype, device=device,
                                                              ros_write_proxy_name=ros_write_proxy_name,
                                                              ros_read_proxy_name=ros_read_proxy_name,
                                                              target_indeces=target_indeces, target_dim=target_dim,
                                                              flg_save_all_particles_to_log=flg_save_all_particles_to_log)

        self.jacobian = jacobian  # lambda q: np.zeros((3, 7))
        self.forward_kinematics = forward_kinematics  # lambda q: np.zeros((1, 7))
        self.initial_cond_func = initial_cond_func  # lambda v_norm, yaw, t_delay: (np.zeros((7, 1)), np.zeros((7, 1)))
        self.targets_dist_function = targets_dist_function
        # lambda num_particles: torch.tensor(np.random.rand(num_particles, self.target_dim),
        # device=self.device, dtype=self.dtype)
        self.sigma_dist_function = sigma_dist_function
        # lambda num_particles: torch.zeros((num_particles, 1),
        # device=self.device, dtype=self.dtype)
        self.t_delay_dist = t_delay_dist
        # lambda num_particles: torch.zeros((num_particles, 1),
        # device=self.device, dtype=self.dtype)


        if skip_fist_samples is None or skip_fist_samples < 0:
            self.skip_fist_samples = 0
        else:
            self.skip_fist_samples = skip_fist_samples

        self.verbose_plots = verbose_plots
        self.bullet_name = bullet_name

        # can be None
        self.trials_data_save_path = trials_data_save_path
        self.load_cached_trials = load_cached_trials
        self.exploration_velocity_schedule = (
            None if exploration_velocity_schedule is None
            else [float(value) for value in exploration_velocity_schedule]
        )
        if int(data_augmentation) != data_augmentation or data_augmentation < 0:
            raise ValueError('data_augmentation must be a non-negative integer')
        if data_augmentation_max_angle < 0.0:
            raise ValueError('data_augmentation_max_angle must be non-negative')
        self.data_augmentation = int(data_augmentation)
        self.data_augmentation_max_angle = float(data_augmentation_max_angle)
        self.data_augmentation_rng = np.random.RandomState(data_augmentation_seed)
        self.flg_apply_meas_noise_to_training_data = bool(
            flg_apply_meas_noise_to_training_data
        )
        self.training_measurement_noise_rng = np.random.RandomState(
            training_measurement_noise_seed
        )
        self.flg_real_style_data_augmentation = bool(
            flg_real_style_data_augmentation
        )
        self.release_radial_velocity_compensation_coefficients = (
            None if release_radial_velocity_compensation_coefficients is None
            else tuple(float(value) for value in release_radial_velocity_compensation_coefficients)
        )
        self.flg_simulate_diff_targets = flg_simulate_diff_targets

        self.release_position = release_position

    def get_data_from_system(self, initial_state, T_exploration, trial_index, flg_exploration=False):
        """
            Forward
        """
        rospy.init_node("MC_PILCO_experiment")
        experiment_publisher = rospy.Publisher(self.ros_write_topic_name, std_msgs.msg.String, queue_size=10)

        policy_path = ''

        # select the policy
        if flg_exploration:
            print("Execute initial exploration policy")
        else:
            print("Export control policy parameters")
            policy_path = self.log_path + '/policy.torch'
            torch.save(self.control_policy.state_dict(), policy_path)

        if flg_exploration and self.load_cached_trials:
        # if self.load_cached_trials:
            noisy_samples, input_samples = pkl.load(
                open(self.trials_data_save_path + 'trial_{}.pkl'.format(trial_index), 'rb'))
        else:
            # Write to topic
            prescribed_velocity = None
            if flg_exploration and self.exploration_velocity_schedule is not None:
                prescribed_velocity = self.exploration_velocity_schedule[trial_index]
                print(
                    'Prescribed absolute exploration velocity: {:.4f} m/s'.format(
                        prescribed_velocity
                    )
                )
            experiment_params = [initial_state, T_exploration, trial_index, flg_exploration, policy_path,
                                 self.bullet_name, prescribed_velocity]
            experiment_params_serialization = json.dumps(experiment_params)
            print('Sending message to ROS proxy topic: {}'.format(self.ros_write_topic_name))

            rate = rospy.Rate(1)  # 1hz
            for i in range(2):
                rate.sleep()
            rospy.loginfo(experiment_params_serialization)

            _wait_for_ros_subscriber(experiment_publisher, self.ros_write_topic_name)
            experiment_publisher.publish(experiment_params_serialization)

            print('Sent message to ROS proxy topic ')
            experiment = rospy.wait_for_message(self.ros_read_topic_name, std_msgs.msg.Float64MultiArray)

            print('State dim: {}'.format(self.state_dim))

            # print('DATA ({}): {}'.format(len(experiment.data),experiment.data))

            positions_dim = len(self.pos_indeces)
            velocity_dim = len(self.vel_indeces)
            # vector contains concatenation of timestamps + states + inputs
            num_samples = int(len(experiment.data) / (1 + positions_dim + 2 * velocity_dim + self.input_dim))
            noisy_samples = np.zeros((num_samples, self.state_dim + 1))  # timestamps + dimensions

            noisy_samples[:, 0] = [experiment.data[i] for i in range(num_samples)]

            for j in range(positions_dim):
                noisy_samples[:, 1 + self.pos_indeces[j]] = [experiment.data[i] for i in
                                                             range((j + 1) * num_samples, (j + 2) * num_samples)]

            for j in range(velocity_dim):
                noisy_samples[:, 1 + self.vel_indeces[j]] = [experiment.data[i] for i in
                                                             range((j + 4) * num_samples, (j + 5) * num_samples)]

            for j in range(self.target_dim):
                print(initial_state[self.target_indeces[j]])
                noisy_samples[:, 1 + self.target_indeces[j]] = [initial_state[self.target_indeces[j]]] * num_samples

            input_samples = np.zeros((num_samples, self.input_dim))
            for j in range(self.input_dim):
                input_samples[:, j] = [experiment.data[i] for i in
                                       range((1 + positions_dim + 2 * velocity_dim + j) * num_samples,
                                             (1 + positions_dim + 2 * velocity_dim + j + 1) * num_samples)]

            print(noisy_samples)
            meas_states = noisy_samples
            noiseless_samples = noisy_samples

            # Save trial
            if not self.trials_data_save_path is None:
                pkl.dump([noisy_samples, input_samples],
                         open(self.trials_data_save_path + 'trial_{}.pkl'.format(trial_index), 'wb'))

        raw_samples = np.array(noisy_samples, copy=True)
        raw_inputs = np.array(input_samples, copy=True)
        print('Num samples before resamp: {}'.format(raw_samples.shape[0]))
        measured_samples = self.add_training_measurement_noise(
            raw_samples,
            include_velocity=not self.flg_real_style_data_augmentation,
        )

        state_samples, input_samples = self.interpolate_states(
            measured_samples,
            raw_inputs,
            derive_velocities_from_positions=self.flg_real_style_data_augmentation,
            add_interpolation_jitter=not self.flg_real_style_data_augmentation,
        )
        print('Num samples after resamp: {}'.format(state_samples.shape[0]))

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.num_data_collection += 1
        # add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)
        for augmentation_index in range(self.data_augmentation):
            angle = self.data_augmentation_rng.uniform(
                -self.data_augmentation_max_angle,
                self.data_augmentation_max_angle,
            )
            augmentation_source = (
                measured_samples if self.flg_real_style_data_augmentation
                else raw_samples
            )
            augmented_raw_states, augmented_raw_inputs = self.rotate_tossing_data(
                augmentation_source[:, 1:], raw_inputs, angle
            )
            augmented_raw_samples = np.column_stack((
                augmentation_source[:, 0], augmented_raw_states
            ))
            if self.flg_real_style_data_augmentation:
                augmented_measured_samples = augmented_raw_samples
            else:
                augmented_measured_samples = self.add_training_measurement_noise(
                    augmented_raw_samples
                )
            augmented_states, augmented_inputs = self.interpolate_states(
                augmented_measured_samples,
                augmented_raw_inputs,
                derive_velocities_from_positions=self.flg_real_style_data_augmentation,
                add_interpolation_jitter=(
                    not self.flg_real_style_data_augmentation
                ),
            )
            self.model_learning.add_data(
                new_state_samples=augmented_states,
                new_input_samples=augmented_inputs,
            )
            print(
                'Added {} Z-rotation augmentation {}/{} '
                '(angle={:.2f} deg)'.format(
                    (
                        'shared noisy-position/differentiated-velocity'
                        if self.flg_real_style_data_augmentation
                        else 'independently noised'
                    ),
                    augmentation_index + 1,
                    self.data_augmentation,
                    np.degrees(angle),
                )
            )

        def myhook():
            print("shutdown time!")

        rospy.on_shutdown(myhook)

    def add_training_measurement_noise(self, noisy_samples, include_velocity=True):
        """Optionally perturb measured training positions and velocities."""
        if not self.flg_apply_meas_noise_to_training_data:
            return noisy_samples

        noisy_samples = np.array(noisy_samples, copy=True)
        std_meas_noise = np.asarray(self.std_meas_noise, dtype=float)
        position_std = std_meas_noise[self.pos_indeces]
        velocity_std = std_meas_noise[self.vel_indeces]
        if not include_velocity:
            velocity_std = np.zeros_like(velocity_std)
        if not np.any(position_std > 0.0) and not np.any(velocity_std > 0.0):
            return noisy_samples

        position_noise = self.training_measurement_noise_rng.normal(
            loc=0.0,
            scale=position_std.reshape(1, -1),
            size=(noisy_samples.shape[0], len(self.pos_indeces)),
        )
        velocity_noise = self.training_measurement_noise_rng.normal(
            loc=0.0,
            scale=velocity_std.reshape(1, -1),
            size=(noisy_samples.shape[0], len(self.vel_indeces)),
        )
        noisy_samples[:, 1 + np.asarray(self.pos_indeces)] += position_noise
        noisy_samples[:, 1 + np.asarray(self.vel_indeces)] += velocity_noise
        print(
            'Added training measurement noise: position std {} m, '
            'velocity std {} m/s'.format(
                position_std.tolist(), velocity_std.tolist()
            )
        )
        return noisy_samples

    def rotate_tossing_data(self, state_samples, input_samples, angle):
        """Rotate a complete tossing sequence around the world Z axis."""
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        augmented_states = np.array(state_samples, copy=True)
        augmented_inputs = np.array(input_samples, copy=True)
        augmented_states[:, self.pos_indeces] = (
            augmented_states[:, self.pos_indeces] @ rotation.T
        )
        augmented_states[:, self.vel_indeces] = (
            augmented_states[:, self.vel_indeces] @ rotation.T
        )
        if self.target_dim != 3:
            raise ValueError('Z-rotation augmentation requires a 3D target')
        augmented_states[:, self.target_indeces] = (
            augmented_states[:, self.target_indeces] @ rotation.T
        )
        if augmented_inputs.shape[1] != 3:
            raise ValueError('Z-rotation augmentation requires a 3D input')
        augmented_inputs[:, :] = augmented_inputs @ rotation.T
        return augmented_states, augmented_inputs

    def interpolate_states(self, noisy_samples, input_samples,
                           derive_velocities_from_positions=False,
                           add_interpolation_jitter=True):
        """
            Application of interpolation for resampling at constant sampling frequency.
        """
        v_norm_target = np.linalg.norm(input_samples[0, :])
        input_samples = input_samples[self.skip_fist_samples:, :]
        noisy_samples = noisy_samples[self.skip_fist_samples:, :]

        print(input_samples)

        times = noisy_samples[:, 0]
        print('Times: {}'.format(times))
        t_0 = times[0]
        times = times - np.ones_like(times) * t_0
        print('Times: {}'.format(times))
        if derive_velocities_from_positions or not add_interpolation_jitter:
            noisy_samples = np.array(noisy_samples[:, 1:], copy=True)
        else:
            noisy_samples = noisy_samples[:, 1:] + np.random.rand(*noisy_samples[:, 1:].shape) * \
                            np.std(noisy_samples[:, 1], axis=0) * 0.001

        print('Avg samp freq: {} Hz'.format(1 / np.mean(times[1:] - times[:-1])))

        # Build an exact T_sampling grid contained in the measured interval.
        # The previous ceil/linspace construction ended at the *next* sampling
        # boundary, extrapolating as much as one complete sample beyond the
        # measured trajectory and also producing a spacing different from
        # T_sampling.
        num_intervals = int(math.floor(
            (times[-1] - times[0]) / self.T_sampling + 1e-12
        ))
        times_new = times[0] + np.arange(num_intervals + 1) * self.T_sampling
        print('Resampling at: {} Hz'.format(1 / np.mean(times_new[1:] - times_new[:-1])))
        print('Times resampled: {}'.format(times_new))
        input_samples = input_samples[:len(times_new), :]

        noisy_samples_resamp = np.zeros((len(times_new), noisy_samples.shape[1]))
        for j in self.pos_indeces:
            # f = interp1d(times, noisy_samples[:, j], kind='cubic', fill_value="extrapolate")
            f = interp1d(times, noisy_samples[:, j], fill_value="extrapolate")
            noisy_samples_resamp[:, j] = f(times_new)

        for j in self.vel_indeces:
            # f = interp1d(times, noisy_samples[:, j], kind='cubic', fill_value="extrapolate")
            f = interp1d(times, noisy_samples[:, j], fill_value="extrapolate")
            noisy_samples_resamp[:, j] = f(times_new)
        for j in self.target_indeces:
            noisy_samples_resamp[:, j] = noisy_samples[:noisy_samples_resamp.shape[0], j]

        state_samples = np.zeros([noisy_samples_resamp.shape[0], noisy_samples_resamp.shape[1]])
        state_samples[:, self.target_indeces] = noisy_samples_resamp[:, self.target_indeces]

        num_vel_samples = np.zeros([noisy_samples_resamp.shape[0] - 1, noisy_samples_resamp.shape[1]])

        for i in range(len(self.pos_indeces)):  # assuming each position has a relative measured velocity
            b, a = signal.butter(2, 0.5)
            pos = signal.filtfilt(b, a, noisy_samples_resamp[:, self.pos_indeces[i]])
            state_samples[:, self.pos_indeces[i]] = pos
            if not derive_velocities_from_positions:
                vel = signal.filtfilt(
                    b, a, noisy_samples_resamp[:, self.vel_indeces[i]]
                )
                state_samples[:, self.vel_indeces[i]] = vel

        if derive_velocities_from_positions:
            central_velocity = (
                state_samples[2:, self.pos_indeces]
                - state_samples[:-2, self.pos_indeces]
            ) / (2 * self.T_sampling)
            state_samples = state_samples[1:-1, :]
            state_samples[:, self.vel_indeces] = central_velocity
            input_samples = input_samples[1:-1, :]
            print('Derived training velocities from filtered positions')
            filter_edge_trim_samples = 0
        else:
            filter_edge_trim_samples = 2

        # filtfilt is least reliable at the right boundary.  Do not expose
        # those edge-conditioned transitions to the GP: in the tossing data
        # they generated isolated velocity deltas roughly ten times larger
        # than the physical drag.  Two samples are sufficient for the current
        # second-order filter and preserve the useful flight trajectory.
        if state_samples.shape[0] <= filter_edge_trim_samples:
            raise ValueError(
                'Not enough resampled trajectory points ({}) to trim {} '
                'filter-edge samples'.format(
                    state_samples.shape[0], filter_edge_trim_samples
                )
            )
        if filter_edge_trim_samples:
            state_samples = state_samples[:-filter_edge_trim_samples, :]
            input_samples = input_samples[:-filter_edge_trim_samples, :]

        if self.verbose_plots:
            plt.figure('Collected trajectory')
            ax = plt.axes(projection='3d')
            ax.plot3D(noisy_samples[:, 0], noisy_samples[:, 1], noisy_samples[:, 2], 'r--', label='noisy',
                      linewidth=3.5)
            ax.plot3D(state_samples[:, 0], state_samples[:, 1], state_samples[:, 2], label='resampled + filt',
                      color='b',
                      linewidth=2)

            yaw = math.atan2(state_samples[-1, 1], state_samples[-1, 0])
            print('Yaw: {}'.format(yaw))

            rel_pos = [self.release_position[0] * math.cos(yaw), self.release_position[0] * math.sin(yaw),
                       self.release_position[2]]
            print(list(state_samples[0, :]))
            print('Dist resp. theoretical release: {}'.format(math.dist(rel_pos, list(state_samples[0, :3]))))
            ax.scatter(rel_pos[0], rel_pos[1], rel_pos[2], marker='^')

            plt.tight_layout()
            plt.grid()
            plt.legend()

            delta_vel_sim = state_samples[1:, self.vel_indeces] - state_samples[:-1, self.vel_indeces]

            print(input_samples)
            if self.skip_fist_samples == 0:
                vel = np.zeros((state_samples.shape[0] - 1, len(self.vel_indeces)))
                for i in range(len(self.pos_indeces)):
                    pos = signal.filtfilt(b, a, state_samples[:, self.pos_indeces[i]])
                    # velocities are computed by means of central difference
                    vel[0, i] = input_samples[0, i]
                    vel[1:, i] = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)
            else:
                vel = np.zeros((state_samples.shape[0] - 2, len(self.vel_indeces)))
                for i in range(len(self.pos_indeces)):
                    pos = signal.filtfilt(b, a, state_samples[:, self.pos_indeces[i]])
                    # velocities are computed by means of central difference
                    vel[:, i] = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)

            delta_vel_numeric = vel[1:] - vel[:-1]

            for j in range(len(self.vel_indeces)):
                plt.figure('Delta vel of trj dim={}'.format(j + 1))
                plt.plot(delta_vel_sim[:, j], label='simulator')
                plt.plot(delta_vel_numeric[:, j], label='numeric differentiation')
                plt.grid()
                plt.legend()
                plt.tight_layout()

            dimensions_names = ['x', 'y', 'z']

            plt.figure('Velocities', figsize=(4, 3))
            for i in range(3):
                plt.plot(vel[1:, i], label=r'$V_{}$'.format(dimensions_names[i]))
            plt.tight_layout()
            plt.legend()

            plt.figure('Velocity deltas')
            print('Avg deltas: ')
            for i in range(3):
                plt.plot(delta_vel_numeric[1:, i], label='dim={}'.format(dimensions_names[i]))
                print('{}: mean={}, std={}'.format(dimensions_names[i], np.mean(delta_vel_numeric[1:, i]),
                                                   np.std(delta_vel_numeric[1:, i])))
            plt.tight_layout()
            plt.legend()

            plt.figure('Delta vel - statistics', figsize=(4, 3))
            plt.boxplot(delta_vel_numeric, showfliers=False, labels=dimensions_names, patch_artist=True)
            plt.tight_layout()

            vel_norms = np.linalg.norm(state_samples[:, self.vel_indeces], axis=1)
            plt.figure('Velocities norms')
            plt.plot(vel_norms, label='simulator')
            plt.plot([v_norm_target] * len(vel_norms), label='target')
            plt.plot()
            plt.grid()
            plt.legend()
            plt.tight_layout()

            plt.show()

        # vel_norms = np.linalg.norm(state_samples[:, self.vel_indeces], axis=1)
        # #delta_vel = (v_norm_target - vel_norms[0]) / vel_norms[0]
        # delta_vel = (vel_norms[0] - v_norm_target) / v_norm_target
        # self.std_ctrl_noise = delta_vel
        #
        # print('Applied velocity noise: {} %'.format(100*self.std_ctrl_noise))

        return state_samples, input_samples

    def apply_metric(self, particles_states_list, particles_inputs_list):
        # Add to log the data about the ball reaching the target in simulations
        distances_list = []
        for particles_states in particles_states_list:
            last_state_pos = particles_states[-1, :, self.pos_indeces]  # t x Particles x dim
            targets = particles_states[-1, :, self.target_indeces]
            distances = np.linalg.norm(last_state_pos - targets, axis=0)  # dim x particles (for some reason idk)
            distances_list.append(distances)

        distances_arr = np.array(distances_list)
        print(distances_arr.shape)
        # print results of last trial
        print('Success rate in last sim: {}%'.format(
            100 * np.count_nonzero(distances_arr[-1, :] < 0.05) / distances_arr.shape[1]))

        if self.log_path is not None:
            print('\nSave log file...')
            if 'sim_distances_from_targets' in self.log_dict.keys():
                self.log_dict['sim_distances_from_targets'].append(distances_arr)
            else:
                self.log_dict['sim_distances_from_targets'] = [distances_arr]
            pkl.dump(self.log_dict, open(self.log_path + '/log.pkl', 'wb'))

    def get_velocities(self, meas_states, input_samples, noiseless_samples, noisy_samples):
        """
        Offline state filtering for modeling

        Application of interpolation for resampling at constant sampling frequency.

        ###################################################################################
        IMPORTANT: change the code according to how you intend to filter data for the model
        ###################################################################################
        """
        meas_states = meas_states[self.skip_fist_samples:, :]
        input_samples = input_samples[self.skip_fist_samples:, :]
        noiseless_samples = noiseless_samples[self.skip_fist_samples:, :]
        noisy_samples = noisy_samples[self.skip_fist_samples:, :]

        times = noisy_samples[:, 0]
        print('Times (b4 resampling): {}'.format(times))
        t_0 = times[0]
        times = times - np.ones_like(times) * t_0
        print('Times (b4 resampling): {}'.format(times))
        noisy_samples = noisy_samples[:, 1:]

        plt.figure('Collected trajectory')
        ax = plt.axes(projection='3d')
        ax.plot3D(noisy_samples[:, 0], noisy_samples[:, 1], noisy_samples[:, 2], 'r--', label='noisy', linewidth=3.5)

        # need the timestamps + positions
        times_new = np.linspace(times[0], math.ceil(times[-1] / self.T_sampling) * self.T_sampling, num=times.shape[0])
        noisy_samples_resamp = copy.deepcopy(noisy_samples)
        for j in self.pos_indeces:
            # f = interp1d(times, noisy_samples[:, j], kind='cubic', fill_value="extrapolate")
            f = interp1d(times, noisy_samples[:, j], fill_value="extrapolate")
            noisy_samples_resamp[:, j] = f(times_new)

        # idx = np.argmin(np.abs(noisy_samples_resamp[:, 2] - noisy_samples[-1, 2] * np.ones_like(noisy_samples_resamp[:, 2])))
        # noisy_samples_resamp = noisy_samples_resamp[:(idx+1), :]

        if self.skip_fist_samples == 0:
            state_samples = np.zeros([noisy_samples_resamp.shape[0] - 1, noisy_samples_resamp.shape[1]])
            state_samples[:, self.target_indeces] = noisy_samples_resamp[1:, self.target_indeces]
            for i in range(len(self.pos_indeces)):
                b, a = signal.butter(2, 0.5)
                # positions are filtered
                pos = signal.filtfilt(b, a, noisy_samples_resamp[:, self.pos_indeces[i]])
                # velocities are computed by means of central difference
                vel = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)
                # discard first and last samples
                state_samples[:, self.pos_indeces[i]] = pos[:-1]
                state_samples[0, self.vel_indeces[i]] = input_samples[
                    0, i]  # The policy determines the initial velocity
                state_samples[1:, self.vel_indeces[i]] = vel

            input_samples = input_samples[:-1, :]
            noiseless_samples = noiseless_samples[:-1, :]
            meas_states = meas_states[:-1, :]
            noisy_samples_resamp = noisy_samples_resamp[:-1, :]
        else:
            state_samples = np.zeros([noisy_samples_resamp.shape[0] - 2, noisy_samples_resamp.shape[1]])
            state_samples[:, self.target_indeces] = noisy_samples_resamp[1:-1, self.target_indeces]
            for i in range(len(self.pos_indeces)):
                b, a = signal.butter(2, 0.5)
                # positions are filtered
                pos = signal.filtfilt(b, a, noisy_samples_resamp[:, self.pos_indeces[i]])
                # velocities are computed by means of central difference
                vel = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)
                # discard first and last samples
                state_samples[:, self.pos_indeces[i]] = pos[1:-1]
                state_samples[:, self.vel_indeces[i]] = vel

            input_samples = input_samples[1:-1, :]
            noiseless_samples = noiseless_samples[1:-1, :]
            meas_states = meas_states[1:-1, :]
            noisy_samples_resamp = noisy_samples_resamp[1:-1, :]

        if input_samples.shape[0] - 1 < state_samples.shape[0]:
            input_samples = np.vstack(input_samples, np.zeros(
                (state_samples.shape[0] - input_samples.shape[0] - 1, input_samples.shape[1])))
        else:
            input_samples = input_samples[:-(input_samples.shape[0] - state_samples.shape[0] - 1), :]

        ax.plot3D(state_samples[:, 0], state_samples[:, 1], state_samples[:, 2], label='resampled + filt', color='b',
                  linewidth=2)
        plt.grid()
        plt.legend()

        delta_vel_filt = state_samples[1:, self.vel_indeces] - state_samples[:-1, self.vel_indeces]

        vel = np.zeros((noisy_samples.shape[0] - 1, 3))
        for i in range(len(self.pos_indeces)):
            pos = signal.filtfilt(b, a, noisy_samples[:, self.pos_indeces[i]])
            # velocities are computed by means of central difference
            vel[0, i] = input_samples[0, i]
            vel[1:, i] = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)

        delta_vel_noisy = vel[1:] - vel[:-1]

        for j in range(len(self.vel_indeces)):
            plt.figure('Delta vel of trj dim={}'.format(j + 1))
            plt.plot(delta_vel_filt[:, j], label='resamp+filt')
            plt.plot(delta_vel_noisy[:, j], label='noisy')
            plt.grid()
            plt.legend()

        plt.show()

        return state_samples, meas_states, input_samples, noiseless_samples, noisy_samples_resamp

    # def apply_policy(self, num_particles, T_control, p_dropout):
    #     """
    #     Apply the policy in simulation having the possibility to only measure some state
    #     variables (positions) and the necessity to derive the others (velocities) by filtering.
    #     ###################################################################################
    #     IMPORTANT: change the code according to how the measures are taken and filtered!!!
    #     ###################################################################################
    #     """
    #     # initialize variables
    #     states_sequence_list = []
    #     inputs_sequence_list = []
    #
    #     # get initial particles
    #     particles_init_func = self.initial_particles_distribution  # It should be a func
    #
    #     initial_states = torch.zeros((num_particles, self.state_dim), device=self.device, dtype=self.dtype)
    #     if self.flg_simulate_diff_targets:
    #         for j in range(num_particles):
    #             initial_states[j, :] = torch.tensor(particles_init_func(), device=self.device, dtype=self.dtype)
    #     else:
    #         x_0 = torch.tensor(particles_init_func(), device=self.device, dtype=self.dtype)
    #         for j in range(num_particles):
    #             initial_states[j, :] = x_0
    #     states_sequence_list.append(initial_states)
    #
    #     u_0 = self.control_policy(states_sequence_list[0], t=0, p_dropout=p_dropout)
    #     # Sampling noise with beta distribution
    #     if self.ctrl_noise_type == 'none':
    #         sigma = 0.0
    #     elif self.ctrl_noise_type == 'gauss':
    #         sigma = torch.normal(torch.ones(num_particles, self.input_dim, device=self.device,
    #                                         dtype=self.dtype) * self.mean_ctrl_noise,
    #                              torch.ones(num_particles, self.input_dim, device=self.device,
    #                                         dtype=self.dtype) * self.std_ctrl_noise)
    #     else:
    #         beta = torch.distributions.Beta(torch.tensor([1.0], device=self.device, dtype=self.dtype),
    #                                         torch.tensor([3.0], device=self.device, dtype=self.dtype))
    #         if self.ctrl_noise_type == 'positive':
    #             sigma = self.std_ctrl_noise * beta.rsample((num_particles,))
    #         else:  # negative
    #             sigma = - self.std_ctrl_noise * beta.rsample((num_particles,))
    #
    #     # beta = torch.distributions.Beta(torch.tensor([1.0], device=self.device, dtype=self.dtype),
    #     #                                torch.tensor([3.0], device=self.device, dtype=self.dtype))
    #     # u_norm = torch.norm(u_0, dim=1)
    #     # delta_pos = torch.normal(torch.zeros(num_particles, device=self.device, dtype=self.dtype),
    #     #                          0.02 * u_norm)
    #     # sigma = torch.zeros(num_particles, device=self.device, dtype=self.dtype)
    #     # for k in range(num_particles):
    #     #     if delta_pos[k] > 0 and u_norm[k] >= 1.175:
    #     #         sigma[k] = -2*self.std_ctrl_noise * beta.rsample((1,)) + 1.0
    #     #     else:
    #     #         sigma[k] = -self.std_ctrl_noise * beta.rsample((1,)) + 1.0
    #
    #     states_sequence_list[0][:, self.vel_indeces] = u_0 * (1 + sigma)
    #     meas_state_samples = states_sequence_list[0].clone()
    #
    #     # init low-pass filter
    #     # b, a = signal.butter(1, self.filtering_dict['fc'])
    #
    #     # measurement noise
    #     std_pos_noise = torch.tensor(self.std_meas_noise_sim[self.pos_indeces], dtype=self.dtype, device=self.device)
    #
    #     # init measured and noisy state samples
    #     meas_states_sequence_list = []
    #     meas_states_sequence_list.append(meas_state_samples)
    #     noisy_states_sequence_list = []
    #     noisy_states_sequence_list.append(meas_state_samples)
    #
    #     # store initial inputs
    #     inputs_sequence_list.append(u_0)
    #     # print(torch.sum(inputs_sequence_list[0]))
    #
    #     for t in range(1, int(T_control)):
    #         # get next state mean and variance (given the states sampled and the inputs computed)
    #         particles, _, _ = self.model_learning.get_next_state(current_state=states_sequence_list[t - 1],
    #                                                              current_input=inputs_sequence_list[t - 1])
    #         states_sequence_list.append(particles)
    #
    #         # # get the noisy states (add noise to positions)
    #         noisy_state_samples = states_sequence_list[t].clone()
    #         noisy_state_samples[:, self.pos_indeces] += std_pos_noise * torch.randn(num_particles,
    #                                                                                 len(self.pos_indeces),
    #                                                                                 device=self.device,
    #                                                                                 dtype=self.dtype)
    #         noisy_states_sequence_list.append(noisy_state_samples)
    #         #
    #         # # measure velocities by numerical differentiation
    #         noisy_states_sequence_list[t][:, self.vel_indeces] = (noisy_states_sequence_list[t][:, self.pos_indeces] -
    #                                                               noisy_states_sequence_list[t - 1][:,
    #                                                               self.pos_indeces]) / self.T_sampling
    #
    #         # next input is zero
    #         inputs_sequence_list.append(torch.zeros_like(inputs_sequence_list[0], device=self.device, dtype=self.dtype))
    #
    #     # returns states/inputs trajectory
    #     return torch.stack(noisy_states_sequence_list), torch.stack(inputs_sequence_list)

    def apply_policy(self, num_particles, T_control, p_dropout):
        """
        Apply the policy in simulation having the possibility to only measure some state
        variables (positions) and the necessity to derive the others (velocities) by filtering.
        ###################################################################################
        IMPORTANT: change the code according to how the measures are taken and filtered!!!
        ###################################################################################
        """
        # initialize variables
        states_sequence_list = []
        inputs_sequence_list = []

        # get initial particles

        targets = self.targets_dist_function(num_particles)

        initial_vel_ratios = torch.zeros((num_particles, len(self.vel_indeces)),
                                         device=self.device, dtype=self.dtype)
        initial_positions = torch.zeros((num_particles, len(self.pos_indeces)),
                                        device=self.device, dtype=self.dtype)
        # Compute jacobian and dq_release
        with torch.no_grad():
            v_0, gamma = self.control_policy.get_v(targets, t=0.0, p_dropout=0.0)  # output is (N x 1)
            for k in range(num_particles):
                t_delay = self.t_delay_dist()
                q, dq = self.initial_cond_func(v_0[k].item(), gamma[k].item(), t_delay)
                J = self.jacobian(q)
                initial_vel_ratios[k, :] = torch.tensor(np.dot(J, [dq_i / v_0[k].item() for dq_i in dq]),
                                                        device=self.device, dtype=self.dtype)
                initial_positions[k, :] = torch.tensor(self.forward_kinematics(q).reshape(1, -1),
                                                       device=self.device, dtype=self.dtype)
            sigma = self.sigma_dist_function(num_particles)

        v_0, _ = self.control_policy.get_v(targets, t=0.0, p_dropout=p_dropout)
        u_0 = initial_vel_ratios * v_0
        if self.release_radial_velocity_compensation_coefficients is not None:
            u_0 = apply_radial_velocity_compensation(
                u_0,
                v_0,
                gamma,
                self.release_radial_velocity_compensation_coefficients,
            )

        initial_states = targets
        initial_states[:, self.pos_indeces] = initial_positions
        initial_states[:, self.vel_indeces] = u_0 * (1 + sigma)

        states_sequence_list.append(initial_states)

        meas_state_samples = states_sequence_list[0].clone()

        # measurement noise
        std_pos_noise = torch.tensor(self.std_meas_noise_sim[self.pos_indeces], dtype=self.dtype, device=self.device)

        # init measured and noisy state samples
        meas_states_sequence_list = []
        meas_states_sequence_list.append(meas_state_samples)
        noisy_states_sequence_list = []
        noisy_states_sequence_list.append(meas_state_samples)

        # store initial inputs
        inputs_sequence_list.append(u_0)
        # print(torch.sum(inputs_sequence_list[0]))

        for t in range(1, int(T_control)):
            # get next state mean and variance (given the states sampled and the inputs computed)
            particles, _, _ = self.model_learning.get_next_state(current_state=states_sequence_list[t - 1],
                                                                 current_input=inputs_sequence_list[t - 1])
            states_sequence_list.append(particles)

            # # get the noisy states (add noise to positions)
            noisy_state_samples = states_sequence_list[t].clone()
            noisy_state_samples[:, self.pos_indeces] += std_pos_noise * torch.randn(num_particles,
                                                                                    len(self.pos_indeces),
                                                                                    device=self.device,
                                                                                    dtype=self.dtype)
            noisy_states_sequence_list.append(noisy_state_samples)
            #
            # # measure velocities by numerical differentiation
            noisy_states_sequence_list[t][:, self.vel_indeces] = (noisy_states_sequence_list[t][:, self.pos_indeces] -
                                                                  noisy_states_sequence_list[t - 1][:,
                                                                  self.pos_indeces]) / self.T_sampling

            # next input is zero
            inputs_sequence_list.append(torch.zeros_like(inputs_sequence_list[0], device=self.device, dtype=self.dtype))

        # returns states/inputs trajectory
        return torch.stack(noisy_states_sequence_list), torch.stack(inputs_sequence_list)

    def plot_model_learning(self, trials_results, state_samples_history, trial_index, actual_trajectory, rollout_states,
                            num_simulation_samples, plot_noised_trs=False):
        """
            Plot uncertain simulations
        """

        if plot_noised_trs:
            init_state = torch.tensor(state_samples_history[0:1, :], device=self.device, dtype=self.dtype)
            uncertain_trajectories = []  # Uncertain trajectories are generated with uncertainty given by previous error
            for i in range(50):
                sigma = trials_results[trial_index]['target_dist_before_update'] / (
                        num_simulation_samples * self.T_sampling)
                # random versor
                # v = np.random.rand(3)
                v = [state_samples_history[-1, j] - rollout_states[-1, j] for j in range(3)]
                print('Distance: {}'.format(v))
                v = v / np.linalg.norm(v)
                delta_v = np.random.normal(0, sigma) * v
                # delta_v = np.random.rand() * sigma * v
                # delta_v = np.random.rand() * sigma * self.current_gps_mse
                uncertain_init_state = init_state
                uncertain_init_state[0:1, self.vel_indeces] += torch.tensor(delta_v, device=self.device,
                                                                            dtype=self.dtype)
                uncertain_trajectories.append(
                    self.get_rollout_trj(uncertain_init_state, actual_trajectory.shape[0]))
                print(uncertain_trajectories[-1].shape)
        plt.figure('Simulation')

        ax = plt.axes(projection='3d')
        ax.plot3D(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2], 'r--', label='ref',
                  linewidth=3.5)
        ax.plot3D(rollout_states[:, 0], rollout_states[:, 1], rollout_states[:, 2], label='GP prediction',
                  color='b', linewidth=2, marker='o')

        if plot_noised_trs:
            for i in range(len(uncertain_trajectories)):
                trj = uncertain_trajectories[i]
                print('trajectory: {}'.format(trj))
                ax.plot3D(trj[:, 0], trj[:, 1], trj[:, 2], color='b', alpha=0.1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.legend()
        plt.grid()

    def print_test_after_model_update(self, gp_output_mean_list, gp_output_var_list):
        """
            Estimates the gravity vector
        """
        g_direction = np.zeros((3,))
        for i in range(3):
            g_direction[i] = np.mean(gp_output_mean_list[i])
        for i in range(3):
            g_direction[i] = g_direction[i] / np.linalg.norm(g_direction)

        print('Direction of gravity: {}'.format(g_direction))

    def get_rollout_prediction_performance(self, data_collection_index, T_rollout=None, add_name='',
                                           particle_pred=False, save_to_log=True):
        """
        Test rollout prediction
        """
        # simulate rollout with inputs from 'data_collection_index' trial
        rollout_states = self.rollout(data_collection_index=data_collection_index,
                                      T_rollout=T_rollout, particle_pred=particle_pred)
        # get rollout performance
        for state_dim_index in range(self.state_dim):
            print('MSE Rollout dim' + str(state_dim_index) + ': ',
                  ((self.state_samples_history[data_collection_index][:, state_dim_index] - rollout_states[:,
                                                                                            state_dim_index]) ** 2).mean())

        if self.log_path is not None and save_to_log:
            print('\nSave rollout to log file...')
            if 'rollout_states' in self.log_dict.keys():
                self.log_dict['rollout_states'].append(rollout_states)
            else:
                self.log_dict['rollout_states'] = [rollout_states]

        # uncomment to plot rollout performance
        # plt.figure('Ball trajectory plot')
        #
        # ax = plt.axes(projection='3d')
        # ax.plot3D(self.state_samples_history[data_collection_index][:, 0],
        #           self.state_samples_history[data_collection_index][:, 1],
        #           self.state_samples_history[data_collection_index][:, 2], 'r--', label='ref', linewidth=3.5)
        # ax.plot3D(rollout_states[:, 0], rollout_states[:, 1], rollout_states[:, 2], label='GP')
        #
        # plt.legend()
        # plt.grid()
        # plt.savefig('rollout_gp'+'_trial'+str(data_collection_index)+'_'+add_name+'.png')
        # plt.show()
        #
        return rollout_states, self.state_samples_history[data_collection_index], self.input_samples_history[
            data_collection_index]

    def get_rollout_trj(self, initial_state, T_rollout, particle_pred=False):
        print(initial_state)
        rollout_trj = torch.zeros((T_rollout, initial_state.size(dim=1)), device=self.device, dtype=self.dtype)

        rollout_trj[0:1, :] = initial_state[0]
        # simulate system evolution for 'T_rollout' steps
        for t in range(1, T_rollout):
            # get next state
            rollout_trj[t:t + 1, :], _, _ = self.model_learning.get_next_state(current_state=rollout_trj[t - 1:t, :],
                                                                               current_input=torch.zeros((1, 3),
                                                                                                         device=self.device,
                                                                                                         dtype=self.dtype),
                                                                               particle_pred=particle_pred)
        return rollout_trj.detach().cpu().numpy()

    def get_particles_evolution_robot_kin(self, particles_init_func, num_particles, T_control, particle_pred=True,
                                          initial_vel=None):
        if initial_vel is None:
            initial_vel = 1.2 + torch.rand(1, device=self.device, dtype=self.dtype) * 1.6

        states_sequence_list = []
        inputs_sequence_list = []
        initial_states = torch.zeros((num_particles, self.state_dim), device=self.device, dtype=self.dtype)
        for j in range(num_particles):
            initial_states[j, :] = torch.tensor(particles_init_func(initial_vel) + [0.0] * 3, device=self.device,
                                                dtype=self.dtype)
        states_sequence_list.append(initial_states)

        # simulated measurement (at t=0 it is the true state)
        meas_state_samples = states_sequence_list[0].clone()

        # measurement noise
        std_pos_noise = torch.tensor(self.std_meas_noise_sim[self.pos_indeces], dtype=self.dtype, device=self.device)

        # init measured and noisy state samples
        noisy_states_sequence_list = []
        noisy_states_sequence_list.append(meas_state_samples)

        # store initial inputs
        inputs_sequence_list.append(states_sequence_list[0][:, self.vel_indeces])

        return self.simulate_particles(states_sequence_list, inputs_sequence_list, noisy_states_sequence_list,
                                       T_control, std_pos_noise, num_particles, particle_pred)

    def get_particles_evolution(self, particles_init_func, num_particles, T_control, particle_pred=True, alpha=0,
                                policy=None):
        """
        Apply the policy in simulation having the possibility to only measure some state
        variables (positions) and the necessity to derive the others (velocities) by filtering.
        ###################################################################################
        IMPORTANT: change the code according to how the measures are taken and filtered!!!
        ###################################################################################
        """
        # initialize variables
        states_sequence_list = []
        inputs_sequence_list = []

        # get initial particles
        initial_states = torch.zeros((num_particles, self.state_dim), device=self.device, dtype=self.dtype)
        for j in range(num_particles):
            initial_states[j, :] = torch.tensor(particles_init_func(), device=self.device, dtype=self.dtype)
        states_sequence_list.append(initial_states)

        u_0 = torch.zeros((states_sequence_list[0].size(dim=0), 3), device=self.device, dtype=self.dtype)
        if policy is None:
            random_vel = 1.2 + torch.rand(states_sequence_list[0].size(dim=0), device=self.device,
                                          dtype=self.dtype) * 1.6
        else:
            random_vel = policy(states_sequence_list[0])

        angles = torch.atan2(initial_states[:, 1], initial_states[:, 0])
        alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)
        u_0[:, 0] = random_vel * torch.cos(angles) * torch.cos(alpha)
        u_0[:, 1] = random_vel * torch.sin(angles) * torch.cos(alpha)
        u_0[:, 1] = random_vel * torch.sin(alpha)
        states_sequence_list[0][:, self.vel_indeces] = u_0

        # simulated measurement (at t=0 it is the true state)
        meas_state_samples = states_sequence_list[0].clone()

        # measurement noise
        std_pos_noise = torch.tensor(self.std_meas_noise_sim[self.pos_indeces], dtype=self.dtype, device=self.device)

        # init measured and noisy state samples
        noisy_states_sequence_list = []
        noisy_states_sequence_list.append(meas_state_samples)

        # store initial inputs
        inputs_sequence_list.append(u_0)
        # print(torch.sum(inputs_sequence_list[0]))

        return self.simulate_particles(states_sequence_list, inputs_sequence_list, noisy_states_sequence_list,
                                       T_control, std_pos_noise, num_particles, particle_pred)

    def simulate_particles(self, states_sequence_list, inputs_sequence_list, noisy_states_sequence_list, T_control,
                           std_pos_noise, num_particles, particle_pred):

        for t in range(1, int(T_control)):
            # get next state mean and variance (given the states sampled and the inputs computed)
            particles, _, _ = self.model_learning.get_next_state(current_state=states_sequence_list[t - 1],
                                                                 current_input=inputs_sequence_list[t - 1],
                                                                 particle_pred=particle_pred)
            states_sequence_list.append(particles)
            # print(particles)

            # get the noisy states (add noise to positions)
            noisy_state_samples = states_sequence_list[t].clone()
            noisy_state_samples[:, self.pos_indeces] += std_pos_noise * torch.randn(num_particles,
                                                                                    len(self.pos_indeces),
                                                                                    device=self.device,
                                                                                    dtype=self.dtype)
            noisy_states_sequence_list.append(noisy_state_samples)

            # measure velocities by numerical differentiation
            noisy_states_sequence_list[t][:, self.vel_indeces] = (noisy_states_sequence_list[t][:, self.pos_indeces] -
                                                                  noisy_states_sequence_list[t - 1][:,
                                                                  self.pos_indeces]) / self.T_sampling

            # compute next input
            inputs_sequence_list.append(torch.zeros_like(inputs_sequence_list[0], device=self.device, dtype=self.dtype))

        # plt.figure('Particles trajectories')
        # ax = plt.axes(projection='3d')
        # for i in range(num_particles):
        #     particle_pos = np.array([noisy_states_sequence_list[t][i, self.pos_indeces].cpu().detach().numpy() for t in
        #                              range(int(T_control))])
        #     ax.plot3D(particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2])
        #
        # plt.legend()
        # plt.grid()
        # plt.savefig('particles.png')

        return noisy_states_sequence_list

    def test_delay_dist_opitimization_by_trials(self, num_trials, model_optimization_opt_list, f_init_particles,
                                                t_delay_left_lim, t_delay_right_lim, delta_delay, resolution, opt_type='grid',
                                                verbose=False, particle_pred=True, num_exploration=1, add_data_model=True,
                                                pbounds=None, bo_init_points=20, bo_iterations=100,
                                                bo_num_particles=10, delay_loss='trajectory_energy_score',
                                                bo_random_state=None, bo_common_random_seed=None):
        """
        """
        self.trajectories_initial_conditions = []
        print('\n\n\n\n----------------- INITIAL trajectories -----------------')
        for expl_index in range(5*num_exploration):
            # interact with the system
            noisy_samples, input_samples = pkl.load(
                open(self.trials_data_save_path + 'trial_{}.pkl'.format(expl_index), 'rb'))
            print(noisy_samples, input_samples)
            trajectory_target_point = noisy_samples[0, 7:]
            trajectory_policy_output = np.linalg.norm(np.array(input_samples[0, :]))
            trajectory_landing = noisy_samples[-1, 1:4]
            trajectory_times = np.asarray(noisy_samples[:, 0], dtype=float)
            trajectory_times = trajectory_times - trajectory_times[0]
            measured_trajectory = {
                'times': trajectory_times,
                'positions': np.asarray(noisy_samples[:, 1:4], dtype=float),
            }

            print('-------------------------------------------------------------------')
            print(trajectory_target_point, trajectory_policy_output, trajectory_landing)
            print('-------------------------------------------------------------------')

            self.trajectories_initial_conditions.append(
                [trajectory_target_point, trajectory_policy_output, trajectory_landing,
                 measured_trajectory])

            state_samples, input_samples = self.interpolate_states(noisy_samples, input_samples)
            print('Num samples after resamp: {}'.format(state_samples.shape[0]))

            self.state_samples_history.append(state_samples)
            self.input_samples_history.append(input_samples)
            self.num_data_collection += 1
            if add_data_model:
                # add data to model_learning object
                self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

            print('-----------------------')
        print('Exp samples: {}'.format(sum([self.state_samples_history[i].shape[0] for i in range(5)])))


        # train GPs on observed interaction data
        print('\n\n----- REINFORCE THE MODEL -----')
        t_model_learning_start = time.time()
        self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
        t_model_learning_stop = time.time()
        print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
        print('\n\n----- Optimize delay distribution -----')


        if pbounds is None:
            pbounds = {'a': [t_delay_left_lim, t_delay_right_lim],
                       'b': [delta_delay/resolution, delta_delay]}
        res, opt = self.bayes_optimization_t_delay_distribution(
            f_init_particles,
            particle_pred,
            pbounds,
            init_points=bo_init_points,
            n_iter=bo_iterations,
            random_state=bo_random_state,
            num_particles_test=bo_num_particles,
            delay_loss=delay_loss,
            common_random_seed=bo_common_random_seed,
        )
        results = [res, opt]


        return results

    def bayes_optimization_t_delay_distribution(self, init_function, particle_pred, pbounds=None,
                                                init_points=20, n_iter=100, random_state=None,
                                                num_particles_test=10, delay_loss='trajectory_energy_score',
                                                common_random_seed=None):
        """
            Requires the model to be optimized

            init_function: f(initial_vel_norm, yaw, t_delay) --> [p_0 + p_0_dot]
        """

        if delay_loss not in DELAY_LOSSES:
            raise ValueError('Unknown delay calibration loss: {}'.format(delay_loss))
        print('Delay calibration loss: {}'.format(delay_loss))

        def get_fitness(a, b):
            # print('Test U({},{})'.format(a, a + b))

            numpy_rng_state = None
            torch_rng_state = None
            cuda_rng_state = None
            if common_random_seed is not None:
                # Every candidate sees the same base uniform variates, GP
                # particle draws and measurement noise.  This removes Monte
                # Carlo ranking noise without perturbing the caller's RNGs.
                numpy_rng_state = np.random.get_state()
                torch_rng_state = torch.random.get_rng_state()
                if torch.cuda.is_available():
                    cuda_rng_state = torch.cuda.get_rng_state_all()
                np.random.seed(int(common_random_seed))
                torch.manual_seed(int(common_random_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(common_random_seed))
            try:
                t_delay_dist = lambda: np.random.uniform(a, a + b)
                _, mean_loss, _ = (
                    self.simulate_system(
                        t_delay_dist,
                        self.trajectories_initial_conditions,
                        init_function,
                        particle_pred,
                        num_particles_test=num_particles_test,
                        delay_loss=delay_loss,
                    ))
            finally:
                if numpy_rng_state is not None:
                    np.random.set_state(numpy_rng_state)
                    torch.random.set_rng_state(torch_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_state)

            # print('Mean MSE: {}\n\n'.format(mean_mse))
            return -mean_loss

        if pbounds is None:
            pbounds = {'a': [-0.1, 0.1], 'b': [0.01, 0.2]}

        optimizer = bayes_opt.bayesian_optimization.BayesianOptimization(
            get_fitness,
            pbounds,
            allow_duplicate_points=True,
            random_state=random_state,
        )

        optimizer.maximize(
            init_points=int(init_points),
            n_iter=int(n_iter),
        )

        # Bayesian opt search for distribution parameters

        optimization_results = optimizer.res
        optimum = dict(optimizer.max)
        optimum['loss_name'] = delay_loss
        optimum['selected_loss_value'] = float(-optimum['target'])
        return optimization_results, optimum

    def simulate_system(self, dist_fun, trajectories_initial_conditions, init_function, particle_pred,
                        num_particles_test=10, delay_loss='trajectory_energy_score', return_diagnostics=False):
        num_particles_test = int(num_particles_test)
        if num_particles_test <= 0:
            raise ValueError('num_particles_test must be positive')
        if delay_loss not in DELAY_LOSSES:
            raise ValueError('Unknown delay calibration loss: {}'.format(delay_loss))
        # Compensated exploration trajectories can take more than 0.75 s to
        # cross the target plane. A shorter horizon incorrectly assigns MSE=1
        # to otherwise valid delay candidates and biases BO calibration.
        max_simulated_samples = round(1.20 / self.T_sampling)

        # print('max_simulated_samples:', max_simulated_samples)

        trajectories_mse = []
        trajectories_selected_loss = []
        particles_landing_positions = []

        for trajectories_initial_condition in trajectories_initial_conditions:
            trajectory_target_point, trajectory_policy_output, trajectory_landing = trajectories_initial_condition[:3]
            measured_trajectory = (
                trajectories_initial_condition[3]
                if len(trajectories_initial_condition) > 3 else None
            )

            target_point_yaw = np.arctan2(trajectory_target_point[1], trajectory_target_point[0])
            target_altitude = trajectory_target_point[2]

            f_init_particles = lambda v: init_function(v, target_point_yaw, dist_fun())

            simulated_samples_this_throw = max_simulated_samples
            if delay_loss == 'trajectory_energy_score':
                if measured_trajectory is None:
                    raise ValueError(
                        'trajectory_energy_score requires measured free-flight trajectories'
                    )
                measured_times = np.asarray(measured_trajectory['times'], dtype=float)
                finite_times = measured_times[np.isfinite(measured_times)]
                if finite_times.size == 0:
                    raise ValueError('measured free-flight trajectory has no valid times')
                simulated_samples_this_throw = max(
                    max_simulated_samples,
                    int(np.ceil(float(np.max(finite_times)) / self.T_sampling)) + 1,
                )

            particles_sequence_list = self.get_particles_evolution_robot_kin(f_init_particles,
                                                                             num_particles_test,
                                                                             simulated_samples_this_throw,
                                                                             initial_vel=trajectory_policy_output.item(),
                                                                             particle_pred=particle_pred)
            particles_positions_list = []

            for i in range(num_particles_test):
                particle_pos = np.array(
                    [particles_sequence_list[t][i, list(range(3))].cpu().detach().numpy() for t in
                     range(simulated_samples_this_throw)])
                # print(particle_pos)
                particles_positions_list.append(particle_pos)

            landing_positions = []
            for particles_positions in particles_positions_list:
                crossing = interpolate_descending_plane_crossing(
                    particles_positions, target_altitude
                )
                if crossing is not None:
                    landing_positions.append(crossing)
                # print(particles_positions[-1, :3])
            if len(landing_positions) > 0:
                landing_positions = np.array(landing_positions)
                # print(landing_positions.shape)
                particles_errors = landing_positions[:, :2] - trajectory_landing[:2]

                landing_mse = np.mean(particles_errors[:, 0] ** 2 + particles_errors[:, 1] ** 2)
                trajectories_mse.append(landing_mse)
                particles_landing_positions.append(landing_positions)
            else:
                landing_mse = 1.0
                trajectories_mse.append(landing_mse)
                particles_landing_positions.append([])

            if delay_loss == 'mse':
                # This is exactly the historical objective, including its
                # non-landing fallback and averaging convention.
                selected_loss = landing_mse
            elif delay_loss == 'energy_score':
                selected_loss = (
                    energy_score(landing_positions[:, :3], np.asarray(trajectory_landing)[:3])
                    if len(landing_positions) > 0 else 1.0
                )
            else:
                simulated_positions = np.stack(particles_positions_list, axis=1)
                selected_loss = trajectory_energy_score(
                    simulated_positions,
                    self.T_sampling,
                    measured_trajectory['times'],
                    measured_trajectory['positions'],
                )
            trajectories_selected_loss.append(selected_loss)

        trajectories_mse = np.array(trajectories_mse)
        trajectories_selected_loss = np.asarray(trajectories_selected_loss)
        diagnostics = {
            'loss_name': delay_loss,
            'selected_loss_value': float(np.mean(trajectories_selected_loss)),
            'per_throw_selected_loss': trajectories_selected_loss,
            'mean_landing_mse': float(np.mean(trajectories_mse)),
            'per_throw_landing_mse': trajectories_mse,
            'particles_per_throw': num_particles_test,
        }
        self.delay_calibration_diagnostics = diagnostics
        result = (
            trajectories_selected_loss,
            diagnostics['selected_loss_value'],
            particles_landing_positions,
        )
        if return_diagnostics:
            return result + (diagnostics,)
        return result


class MC_PILCO_Tossing_Experiment_from_files(MC_PILCO_ROS_Tossing_Experiment):
    """
        Class that loads experiment data from data files that implements system interaction with of Tossing experiments.

        files_template: '..path/to/files/dir/_{}_trajectory.pkl'    -> template to pick the files
        files_index_list: ['prima', 1: 'seconda']           -> contains the ordered file indices

        Each trial is 10 trajectories,
        each trajectory is rotated by 180 degrees + a random rotation in [-90, 90] deg before being added to the
        model's training dataset

    """

    def __init__(self, T_sampling, state_dim, input_dim, f_sim,
                 f_model_learning, model_learning_par,
                 f_rand_exploration_policy, rand_exploration_policy_par,
                 f_control_policy, control_policy_par,
                 f_cost_function, cost_function_par,
                 ros_write_proxy_name, ros_read_proxy_name,
                 pos_indeces, vel_indeces, files_template, files_index_list,
                 target_indeces, target_dim, release_position,
                 jacobian, forward_kinematics, initial_cond_func,
                 targets_dist_function, sigma_dist_function, t_delay_dist,
                 data_augmentation=2,
                 flg_save_all_particles_to_log=True,
                 skip_fist_samples=5,
                 std_meas_noise=None, log_path=None,
                 filtering_dict={}, std_meas_noise_sim=None,
                 dtype=torch.float64, device=torch.device('cpu'),
                 verbose_plots=False, g_direction=None):
        super(MC_PILCO_Tossing_Experiment_from_files, self).__init__(T_sampling=T_sampling, state_dim=state_dim,
                                                                     input_dim=input_dim,
                                                                     f_sim=f_sim, f_model_learning=f_model_learning,
                                                                     model_learning_par=model_learning_par,
                                                                     f_rand_exploration_policy=f_rand_exploration_policy,
                                                                     rand_exploration_policy_par=rand_exploration_policy_par,
                                                                     f_control_policy=f_control_policy,
                                                                     control_policy_par=control_policy_par,
                                                                     f_cost_function=f_cost_function,
                                                                     cost_function_par=cost_function_par,
                                                                     pos_indeces=pos_indeces, vel_indeces=vel_indeces,
                                                                     std_meas_noise=std_meas_noise, log_path=log_path,
                                                                     filtering_dict=filtering_dict,
                                                                     std_meas_noise_sim=std_meas_noise_sim,
                                                                     dtype=dtype, device=device,
                                                                     ros_write_proxy_name=ros_write_proxy_name,
                                                                     ros_read_proxy_name=ros_read_proxy_name,
                                                                     target_indeces=target_indeces,
                                                                     target_dim=target_dim,
                                                                     skip_fist_samples=skip_fist_samples,
                                                                     verbose_plots=verbose_plots,
                                                                     release_position=release_position,
                                                                     flg_save_all_particles_to_log=flg_save_all_particles_to_log,
                                                                     jacobian=jacobian,
                                                                     forward_kinematics=forward_kinematics,
                                                                     initial_cond_func=initial_cond_func,
                                                                     targets_dist_function=targets_dist_function,
                                                                     sigma_dist_function=sigma_dist_function,
                                                                     t_delay_dist=t_delay_dist
                                                                     )
        self.files_template = files_template
        self.files_index_list = files_index_list
        self.g_direction = g_direction

        self.trajectories_initial_conditions = []

        self.data_augmentation = data_augmentation

    def get_data_from_system(self, initial_state, T_exploration, trial_index, angle_rot=0.0, flg_exploration=False):
        """
            Loads 10 trajectories, each trajectory is rotated by 180 degrees + a random rotation in [-90, 90] deg
        """
        for data_index in range(10):
            print('data_index:', data_index)
            self.get_data_for_model(trial_index=data_index, angle_rot=0.0)
            for _ in range(self.data_augmentation):
                angle = 2 * (np.random.rand() - 0.5) * np.pi / 4
                self.get_data_for_model(trial_index=trial_index * 10 + data_index,
                                        angle_rot=angle, load_initial_conditions=False)

    def get_data_for_model(self, trial_index, angle_rot=0.0, load_initial_conditions=True):
        """
            loads a file, the policy cannot be applied
            Basically all parameters are ignored (except trial index that picks the file)
        """
        state_samples, input_samples = self.load_data_from_files(trial_index=trial_index,
                                                                 angle_rot=angle_rot,
                                                                 load_initial_conditions=load_initial_conditions)
        # add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

    def load_data_from_files(self, trial_index, angle_rot, load_initial_conditions=True):
        # Load file
        trajectory_files_template = self.files_template['trajectory_files_template']
        trajectory = pkl.load(open(trajectory_files_template.format(*self.files_index_list[trial_index]), 'rb'))

        if load_initial_conditions:
            try:
                landing_files_template = self.files_template['landing_files_template']
                trajectory_landing = np.loadtxt(landing_files_template.format(*self.files_index_list[trial_index]),
                                                delimiter=',')
                target_point_files_template = self.files_template['target_point_files_template']
                trajectory_target_point = np.loadtxt(
                    target_point_files_template.format(*self.files_index_list[trial_index]),
                    delimiter=',')
                policy_output_files_template = self.files_template['policy_output_files_template']
                trajectory_policy_output = np.loadtxt(
                    policy_output_files_template.format(*self.files_index_list[trial_index]),
                    delimiter=',')
                self.trajectories_initial_conditions.append(
                    [trajectory_target_point, trajectory_policy_output, trajectory_landing])

                print('Loaded trajectories initial conditions')
            except:
                print('Error loading trajectories initial conditions, trial: ' + str(trial_index))


        print(trajectory)
        # Trajectories might be composed of bouncing segments
        # Remove contact points
        trj = None
        falling = False
        # for k in range(trajectory.shape[0] - 1):
        #     if trajectory[k, 3] > trajectory[k+1, 3]: # to avoid the first part of trj where direction is up
        #         falling = True
        #     if trajectory[k, 3] < trajectory[k+1, 3] and falling:
        #         k_1 = k-1
        #         break
        # trajectory_list.append(random_rotation(trajectory[:k_1, :]))
        # if not falling:
        #     print('Trajectory is strange, no falling')

        # trajectory_list has a single trj
        K = 0
        for k in range(1, trajectory.shape[0]):
            if trajectory[k, 3] < trajectory[k - 1, 3]:  # to avoid the first part of trj where direction is up
                falling = True

            if trajectory[k, 3] > trajectory[k - 1, 3] and falling:
                if k > 10:
                    K = k
                    break
        trj = rotation_z(trajectory[1:K - 2, :], gamma=np.pi + angle_rot)

        assert trj.shape[0]>0  # Fail in loading trajectory

        #for i, trj in enumerate(trajectory_list):
        num_samples = trj.shape[0]

        noisy_samples = np.zeros((num_samples, self.state_dim + 1))  # timestamps + dimensions

        noisy_samples[:, :4] = trj

        target_pos = noisy_samples[-1, 1:4]

        print('Target: {}'.format(target_pos))

        for j in range(self.target_dim):
            noisy_samples[:, 1 + self.target_indeces[j]] = [target_pos[j]] * num_samples

        input_samples = np.zeros((num_samples, self.input_dim))

        # approximate velocities using position measures
        state_samples, meas_states, input_samples, noiseless_samples, noisy_samples = (
            self.get_velocities(input_samples, noisy_samples))

        input_samples[0, :] = state_samples[0, self.vel_indeces]  # just for formalism

        #print(state_samples)
        #print(input_samples)

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.num_data_collection += 1

        return state_samples, input_samples

    def get_velocities(self, input_samples, noisy_samples):
        """
        Offline state filtering for modeling

        Just compute the velocities, trajectories should already be filtered and resampled

        """

        times = noisy_samples[:, 0]
        noisy_samples = noisy_samples[:, 1:]
        state_samples = np.zeros([noisy_samples.shape[0] - 2, self.state_dim])
        state_samples[:, self.pos_indeces] = noisy_samples[1:-1, self.pos_indeces]
        state_samples[:, self.target_indeces] = noisy_samples[1:-1, self.target_indeces]

        for i in range(len(self.pos_indeces)):
            b, a = signal.butter(2, self.filtering_dict['fc'])
            # b, a = signal.butter(2, 0.1)
            # positions are filtered
            # pos = signal.lfilter(b, a, noisy_samples[:, self.pos_indeces[i]],
            #                      zi=signal.lfiltic(b, a, noisy_samples[0, self.pos_indeces[i]]))
            # pos = pos[0]
            # print(pos)
            pos = signal.filtfilt(b, a, noisy_samples[:, self.pos_indeces[i]])

            # velocities are computed by means of central difference
            vel = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)
            # discard first and last samples
            state_samples[:, self.pos_indeces[i]] = pos[1:-1]

            # if i < len(self.pos_indeces) - 1 and True:  # approximate constant velocities
            #     state_samples[:, self.vel_indeces[i]] = np.array([np.mean(vel)] * vel.shape[0])
            # else:
            state_samples[:, self.vel_indeces[i]] = vel

            noisy_samples[1:-1, self.vel_indeces[i]] = (noisy_samples[2:, self.pos_indeces[i]] -
                                                        noisy_samples[:-2, self.pos_indeces[i]]) / (2 * self.T_sampling)

        state_samples = state_samples[self.skip_fist_samples:, :]
        input_samples = input_samples[self.skip_fist_samples + 1:-1, :]

        times = times[self.skip_fist_samples + 1:-1]
        noisy_samples = noisy_samples[self.skip_fist_samples + 1:-1, :]

        ##############################
        # state_samples = noisy_samples
        ##############################

        if self.verbose_plots:
            plt.figure('Collected trajectory', figsize=(4, 3))
            ax = plt.axes(projection='3d')
            ax.plot3D(noisy_samples[:, 0], noisy_samples[:, 1], noisy_samples[:, 2], 'r--', label='noisy',
                      linewidth=3.5)

            ax.plot3D(state_samples[:, 0], state_samples[:, 1], state_samples[:, 2], label='filt',
                      color='b',
                      linewidth=2)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.grid()
            plt.legend()
            plt.tight_layout()

            fig, axs = plt.subplots(nrows=3, ncols=1)
            axs[0].set_title('Positions')
            axs[0].plot(times, state_samples[:, 0], label='x')
            axs[0].grid()
            axs[0].legend()
            axs[1].plot(times, state_samples[:, 1], label='y')
            axs[1].grid()
            axs[1].legend()
            axs[2].plot(times, state_samples[:, 2], label='z')
            axs[2].grid()
            axs[2].legend()

            plt.figure('velocities of trj', figsize=(4, 3))
            col = ['r', 'g', 'b']
            dim = ['x', 'y', 'z']
            for j in range(len(self.vel_indeces)):
                plt.plot(times, state_samples[:, self.vel_indeces[j]], label=r'$V_{}$ filt'.format(dim[j]),
                         color=col[j])
                plt.plot(times, noisy_samples[:, self.vel_indeces[j]], label=r'$V_{}$ noisy'.format(dim[j]),
                         color=col[j],
                         linestyle='dashed')
            plt.grid()
            plt.legend()
            plt.tight_layout()

            delta_vel_filt = state_samples[1:, self.vel_indeces] - state_samples[:-1, self.vel_indeces]
            delta_vel_noisy = noisy_samples[1:, self.vel_indeces] - noisy_samples[:-1, self.vel_indeces]

            #print(delta_vel_filt)
            #print(delta_vel_noisy)

            for j in range(len(self.vel_indeces)):
                plt.figure('Delta vel of trj, dim = {}'.format(dim[j]))
                plt.plot(delta_vel_filt[:, j], label='{} filt'.format(dim[j]), color=col[j])
                plt.plot(delta_vel_noisy[:, j], label='{} noisy'.format(dim[j]), color=col[j],
                         linestyle='dashed')
            plt.grid()
            plt.legend()

            fig, ax = plt.subplots()
            fig.set_figheight(3)
            fig.set_figwidth(4)
            ax2 = ax.twinx()

            ax.get_shared_y_axes().join(ax, ax2)
            #ax2.get_shared_y_axes().join(ax)

            box1 = ax.boxplot(delta_vel_noisy, showfliers=False, labels=dim, patch_artist=True)
            box2 = ax2.boxplot(delta_vel_filt, showfliers=False, labels=dim, patch_artist=True)
            ax2.set_yticks([])

            for item in ['whiskers', 'fliers', 'caps']:
                plt.setp(box1[item], color='b', linewidth=2)
                plt.setp(box2[item], color='g', linewidth=2, alpha=0.5)

            plt.setp(box1["boxes"], facecolor='lightblue')
            plt.setp(box2["boxes"], facecolor='lightgreen', alpha=0.5)

            plt.setp(box1["medians"], linewidth=4, color='blue')
            plt.setp(box2["medians"], linewidth=4, color='green')

            plt.tight_layout()

            vel_norms_filt = np.linalg.norm(state_samples[:, self.vel_indeces], axis=1)
            vel_norms_noisy = np.linalg.norm(noisy_samples[:, self.vel_indeces], axis=1)
            plt.figure('Velocities norms', figsize=(4, 3))
            plt.plot(times, vel_norms_filt, label='v norm filt')
            plt.plot(times, vel_norms_noisy, label='v norm noisy')
            plt.grid()
            plt.legend()
            plt.tight_layout()

            plt.show()

        return state_samples, state_samples, input_samples, state_samples, noisy_samples

    def test_model_learning_by_trials(self, num_trials, model_optimization_opt_list, f_init_particles,
                                      verbose=False, particle_pred=True):
        """
            Model learning test
            10 trajectories -> train model -> test on next 10 trajectories -> re-train model -> test on next 10 trajectories ...
        """
        print('\n\n\n\n----------------- INITIAL 10 trajectories -----------------')
        for expl_index in range(10):
            # interact with the system
            self.get_data_for_model(trial_index=expl_index, angle_rot=0.0)
            for _ in range(self.data_augmentation):
                angle = 2 * (np.random.rand() - 0.5) * np.pi / 4
                self.get_data_for_model(trial_index=expl_index, angle_rot=angle, load_initial_conditions=False)
        print('Exp samples: {}'.format(sum([self.state_samples_history[i].shape[0] for i in range(10)])))

        trials_results = {}

        for trial_index in range(1, num_trials):
            print('\n\n----- TRIAL {} -----'.format(trial_index))
            trials_results[trial_index] = {'state_samples_list': [],
                                           'rollout_states_list': [],
                                           'particles_sequence_list_sequence_list_velocities': []
                                           }
            # train GPs on observed interaction data
            print('\n\n----- REINFORCE THE MODEL -----')
            t_model_learning_start = time.time()
            self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
            t_model_learning_stop = time.time()
            print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
            print('\n\n----- CHECK THE LEARNING PERFORMANCE (before model update) on new 10 trajectories data -----')

            num_particles_test = 100
            particles_sequence_list_velocities = []
            velocities = [1.2, 2.0, 2.8]

            max_simulated_samples = round(.8 / self.T_sampling)
            particles_positions_list_velocities = []
            for vel in velocities:
                particles_sequence_list = self.get_particles_evolution_robot_kin(f_init_particles,
                                                                                 num_particles_test,
                                                                                 max_simulated_samples,
                                                                                 initial_vel=vel,
                                                                                 particle_pred=particle_pred)
                particles_positions_list = []

                for i in range(num_particles_test):
                    particle_pos = np.array(
                        [particles_sequence_list[t][i, list(range(3))].cpu().detach().numpy() for t in
                         range(max_simulated_samples)])
                    particles_positions_list.append(particle_pos)

                if verbose:
                    plt.figure('Particles trajectories V={}'.format(vel))
                    ax = plt.axes(projection='3d')
                    for i in range(len(particles_positions_list)):
                        ax.plot3D(particles_positions_list[i][:, 0],
                                  particles_positions_list[i][:, 1],
                                  particles_positions_list[i][:, 2])
                    plt.grid()

                particles_positions_list_velocities.append(particles_positions_list)
                particles_sequence_list_velocities.append(particles_sequence_list)

            trials_results[trial_index]['particles_sequence_list_velocities'] = particles_sequence_list_velocities
            trials_results[trial_index]['particles_positions_list_velocities'] = particles_positions_list_velocities

            if verbose:
                plt.figure('Simulation trajectories')
                for i in range(len(velocities)):
                    positions = particles_sequence_list_velocities[i][-1][:, list(range(3))].cpu().detach().numpy()
                    plt.scatter([positions[:, 0]], [positions[:, 1]], color='C' + str(i))
                plt.grid()
                plt.scatter([0.0], [0.0], c='r', s=500)
                plt.show()

            for data_index in range(10):
                state_samples, input_samples = self.load_data_from_files(trial_index=trial_index * 10 + data_index,
                                                                         angle_rot=0.0, load_initial_conditions=False)
                rollout_states, state_samples_history, _ = self.get_rollout_prediction_performance(
                    data_collection_index=trial_index * 10 + data_index, add_name='pre_tr', particle_pred=False)

                _, _, _, _ = self.get_model_learning_performance(data_collection_index=trial_index * 10 + data_index,
                                                                 post_str='pre_tr')

                dist = math.dist(state_samples_history[-1, :3], rollout_states[-1, :3])
                print('Error in prediction of last position:', str(dist))

                trials_results[trial_index]['state_samples_list'].append(state_samples_history)
                trials_results[trial_index]['rollout_states_list'].append(rollout_states)

                # add data to model_learning object
                self.get_data_for_model(trial_index=trial_index * 10 + data_index,
                                        angle_rot=0, load_initial_conditions=False)
                for _ in range(self.data_augmentation):
                    angle = 2 * (np.random.rand() - 0.5) * np.pi / 4
                    self.get_data_for_model(trial_index=trial_index * 10 + data_index,
                                            angle_rot=angle)

                #self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

        return trials_results

        #
        # # train GPs on observed interaction data
        # print('\n\n----- REINFORCE THE MODEL -----')
        # t_model_learning_start = time.time()
        # self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
        # t_model_learning_stop = time.time()
        # print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
        #
        # # get model learning performance
        # print('\n\n----- CHECK THE LEARNING PERFORMANCE (after model update) on exploration data -----')
        # _, _, _, _ = self.get_model_learning_performance(data_collection_index=num_explorations - 1)
        # # get policy performance
        # print('\n\n----- CHECK THE ROLLOUT PERFORMANCE (after model update) on exploration data -----')
        # _, _, _ = self.get_rollout_prediction_performance(data_collection_index=num_explorations - 1,
        #                                                   add_name='post_tr', particle_pred=False)
        #
        # first_trial_index = num_explorations
        # last_trial_index = num_trials + num_explorations - 1
        #
        # trials_results = {}
        #
        # for trial_index in range(first_trial_index, last_trial_index):
        #     print('\n\n\n\n----------------- TRIAL ' + str(trial_index) + ' -----------------')
        #     if flg_init_func:
        #         x0 = init_func()
        #     else:
        #         if random_initial_state:  # initial state randomly sampled
        #             if flg_init_uniform:  # uniform initial distribution
        #                 x0 = np.random.uniform(init_low_bound, init_up_bound)
        #             elif flg_init_multi_gauss:  # multimodal gaussians initial distribution
        #                 num_init = np.random.randint(initial_state.shape[0])
        #                 x0 = np.random.normal(initial_state[num_init, :], np.sqrt(initial_state_var[num_init, :]))
        #             else:  # gaussian initial distribution
        #                 x0 = np.random.normal(initial_state, np.sqrt(initial_state_var))
        #         else:  # deterministic initial state
        #             x0 = initial_state
        #
        #     # interact with the system
        #     self.get_data_from_system(initial_state=x0,
        #                               T_exploration=T_exploration,
        #                               flg_exploration=True,  # exploration interaction
        #                               trial_index=trial_index)
        #     if verbose_print:
        #         print(self.input_samples_history[0])
        #
        #     # get model learning performance
        #     print('\n\n----- CHECK THE LEARNING PERFORMANCE (before model update) on trial data -----')
        #     trials_results[trial_index] = {}
        #     trials_results[trial_index]['ML_gp_mse_before_update'] = []
        #     _, gp_outputs_target_list, gp_output_mean_list, _ = self.get_model_learning_performance(
        #         data_collection_index=trial_index)
        #     for gp_index in range(0, self.model_learning.num_gp):
        #         trials_results[trial_index]['ML_gp_mse_before_update'].append(
        #             ((gp_outputs_target_list[gp_index] - gp_output_mean_list[gp_index]) ** 2).mean())
        #
        #     if vel_indices is None:
        #         vel = None
        #     else:
        #         vel = self.state_samples_history[trial_index][:, vel_indices]
        #     # self.plot_GP_inference(gp_outputs_target_list, gp_output_mean_list, vel=vel, title='before model update')
        #     # plt.show()
        #
        #     # get policy performance
        #     print('\n\n----- CHECK THE ROLLOUT PERFORMANCE (before model update) on trial data -----')
        #     _, _, _, _ = self.get_model_learning_performance(data_collection_index=trial_index, post_str='pre_tr')
        #     trials_results[trial_index]['ML_rollout_mse_before_update'] = []
        #     rollout_states, state_samples_history, _ = self.get_rollout_prediction_performance(
        #         data_collection_index=trial_index, add_name='pre_tr', particle_pred=False)
        #     for state_dim_index in range(self.state_dim):
        #         trials_results[trial_index]['ML_rollout_mse_before_update'].append(
        #             ((state_samples_history[:, state_dim_index] - rollout_states[:, state_dim_index]) ** 2).mean())
        #
        #     trials_results[trial_index]['target_dist_before_update'] = math.dist(state_samples_history[-1, :2],
        #                                                                          rollout_states[-1, :2])
        #     print('\n\n------ Target dist (before model update): {}'.format(
        #         trials_results[trial_index]['target_dist_before_update']))
        #
        #     # Testing the model with uncertain rollouts with init_v = policy(x_0) + uncertainty
        #     # uncertainty: wgn with std = (error of predicted landing in previous rollout) / (trajectory time)
        #
        #     # Initial state (random)
        #
        #     # Predicted trajectory
        #     actual_trajectory = state_samples_history
        #
        #     # self.plot_model_learning(trials_results, trial_index, actual_trajectory,
        #     #                          rollout_states, num_simulation_samples, plot_noised_trs)
        #
        #     if verbose_print:
        #         print('Predicted trajectory; {}'.format(rollout_states))
        #
        #     # train GPs on observed interaction data
        #     print('\n\n----- REINFORCE THE MODEL -----')
        #     t_model_learning_start = time.time()
        #     self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
        #     t_model_learning_stop = time.time()
        #     print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
        #
        #     # get model learning performance
        #     print('\n\n----- CHECK THE LEARNING PERFORMANCE (after model update) on trial data -----')
        #     trials_results[trial_index]['ML_gp_mse_after_update'] = []
        #     _, gp_outputs_target_list, gp_output_mean_list, gp_output_var_list = self.get_model_learning_performance(
        #         data_collection_index=trial_index)
        #     for gp_index in range(0, self.model_learning.num_gp):
        #         trials_results[trial_index]['ML_gp_mse_after_update'].append(
        #             ((gp_outputs_target_list[gp_index] - gp_output_mean_list[gp_index]) ** 2).mean())
        #
        #     if vel_indices is None:
        #         vel = None
        #     else:
        #         vel = self.state_samples_history[trial_index][:, vel_indices]
        #     # self.plot_GP_inference(gp_outputs_target_list, gp_output_mean_list, vel=vel, title = 'after model update')
        #     # plt.show()
        #
        #     # get policy performance
        #     print('\n\n----- CHECK THE ROLLOUT PERFORMANCE (after model update) on trial data -----')
        #     trials_results[trial_index]['ML_rollout_mse_after_update'] = []
        #     rollout_states, state_samples_history, _ = self.get_rollout_prediction_performance(
        #         data_collection_index=trial_index, particle_pred=False,
        #         add_name='post_tr')
        #     for state_dim_index in range(self.state_dim):
        #         trials_results[trial_index]['ML_rollout_mse_after_update'].append(
        #             ((state_samples_history[:, state_dim_index] - rollout_states[:, state_dim_index]) ** 2).mean())
        #     trials_results[trial_index]['target_dist_after_update'] = math.dist(state_samples_history[-1, :2],
        #                                                                         rollout_states[-1, :2])
        #     print('\n\n------ Target dist (after model update): {}'.format(
        #         trials_results[trial_index]['target_dist_after_update']))
        #
        #     # self.plot_model_learning(trials_results, trial_index, actual_trajectory,
        #     #                          rollout_states, num_simulation_samples, plot_noised_trs)
        #
        #     self.print_test_after_model_update(gp_output_mean_list, gp_output_var_list)
        #
        #     plt.show()
        # return trials_results

    def test_delay_dist_opitimization_by_trials(self, num_trials, model_optimization_opt_list, f_init_particles,
                                                t_delay_left_lim, t_delay_right_lim, delta_delay, resolution, opt_type='grid',
                                                verbose=False, particle_pred=True, num_exploration=1):
        """
        """
        print('\n\n\n\n----------------- INITIAL 10 trajectories -----------------')
        for expl_index in range(10*num_exploration):
            # interact with the system
            self.get_data_for_model(trial_index=expl_index, angle_rot=0.0)
            for _ in range(self.data_augmentation):
                angle = 2 * (np.random.rand() - 0.5) * np.pi / 4
                self.get_data_for_model(trial_index=expl_index, angle_rot=angle, load_initial_conditions=False)
        print('Exp samples: {}'.format(sum([self.state_samples_history[i].shape[0] for i in range(10)])))

        for trial_index in range(num_exploration, num_trials):
            print('\n\n----- TRIAL {} -----'.format(trial_index))
            for data_index in range(10):
                _, _ = self.load_data_from_files(trial_index=trial_index * 10 + data_index,
                                                 angle_rot=0.0)

        results = {}


        # train GPs on observed interaction data
        print('\n\n----- REINFORCE THE MODEL -----')
        t_model_learning_start = time.time()
        self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
        t_model_learning_stop = time.time()
        print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
        print('\n\n----- Optimize delay distribution -----')

        # t_delay_left_lim = -0.02
        # delta_delay = 0.04
        # resolution = 20
        if opt_type == 'grid':
            dist_grid, fitness_values, dist_particles_landing_positions = (
                self.optimize_t_delay_distribution(f_init_particles, particle_pred, t_delay_left_lim, t_delay_right_lim,
                                                   delta_delay, resolution))
            results = [dist_grid, fitness_values, dist_particles_landing_positions]

        elif opt_type == 'bayes':
            pbounds = {'a': [t_delay_left_lim, t_delay_right_lim],
                       'b': [delta_delay/resolution, delta_delay]}
            res, opt = self.bayes_optimization_t_delay_distribution(f_init_particles, particle_pred, pbounds)
            results = [res, opt]



        # for data_index in range(10):
        #     # state_samples, input_samples = self.load_data_from_files(trial_index=trial_index * 10 + data_index,
        #     #                                                          angle_rot=0.0)
        #     # add data to model_learning object
        #     self.get_data_for_model(trial_index=trial_index * 10 + data_index,
        #                             angle_rot=0)
        #     for _ in range(2):
        #         angle = 2 * (np.random.rand() - 0.5) * np.pi / 4
        #         self.get_data_for_model(trial_index=trial_index * 10 + data_index,
        #                                 angle_rot=angle, load_initial_conditions=False)

            #self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

        return results

        #
        # # train GPs on observed interaction data
        # print('\n\n----- REINFORCE THE MODEL -----')
        # t_model_learning_start = time.time()
        # self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
        # t_model_learning_stop = time.time()
        # print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
        #
        # # get model learning performance
        # print('\n\n----- CHECK THE LEARNING PERFORMANCE (after model update) on exploration data -----')
        # _, _, _, _ = self.get_model_learning_performance(data_collection_index=num_explorations - 1)
        # # get policy performance
        # print('\n\n----- CHECK THE ROLLOUT PERFORMANCE (after model update) on exploration data -----')
        # _, _, _ = self.get_rollout_prediction_performance(data_collection_index=num_explorations - 1,
        #                                                   add_name='post_tr', particle_pred=False)
        #
        # first_trial_index = num_explorations
        # last_trial_index = num_trials + num_explorations - 1
        #
        # trials_results = {}
        #
        # for trial_index in range(first_trial_index, last_trial_index):
        #     print('\n\n\n\n----------------- TRIAL ' + str(trial_index) + ' -----------------')
        #     if flg_init_func:
        #         x0 = init_func()
        #     else:
        #         if random_initial_state:  # initial state randomly sampled
        #             if flg_init_uniform:  # uniform initial distribution
        #                 x0 = np.random.uniform(init_low_bound, init_up_bound)
        #             elif flg_init_multi_gauss:  # multimodal gaussians initial distribution
        #                 num_init = np.random.randint(initial_state.shape[0])
        #                 x0 = np.random.normal(initial_state[num_init, :], np.sqrt(initial_state_var[num_init, :]))
        #             else:  # gaussian initial distribution
        #                 x0 = np.random.normal(initial_state, np.sqrt(initial_state_var))
        #         else:  # deterministic initial state
        #             x0 = initial_state
        #
        #     # interact with the system
        #     self.get_data_from_system(initial_state=x0,
        #                               T_exploration=T_exploration,
        #                               flg_exploration=True,  # exploration interaction
        #                               trial_index=trial_index)
        #     if verbose_print:
        #         print(self.input_samples_history[0])
        #
        #     # get model learning performance
        #     print('\n\n----- CHECK THE LEARNING PERFORMANCE (before model update) on trial data -----')
        #     trials_results[trial_index] = {}
        #     trials_results[trial_index]['ML_gp_mse_before_update'] = []
        #     _, gp_outputs_target_list, gp_output_mean_list, _ = self.get_model_learning_performance(
        #         data_collection_index=trial_index)
        #     for gp_index in range(0, self.model_learning.num_gp):
        #         trials_results[trial_index]['ML_gp_mse_before_update'].append(
        #             ((gp_outputs_target_list[gp_index] - gp_output_mean_list[gp_index]) ** 2).mean())
        #
        #     if vel_indices is None:
        #         vel = None
        #     else:
        #         vel = self.state_samples_history[trial_index][:, vel_indices]
        #     # self.plot_GP_inference(gp_outputs_target_list, gp_output_mean_list, vel=vel, title='before model update')
        #     # plt.show()
        #
        #     # get policy performance
        #     print('\n\n----- CHECK THE ROLLOUT PERFORMANCE (before model update) on trial data -----')
        #     _, _, _, _ = self.get_model_learning_performance(data_collection_index=trial_index, post_str='pre_tr')
        #     trials_results[trial_index]['ML_rollout_mse_before_update'] = []
        #     rollout_states, state_samples_history, _ = self.get_rollout_prediction_performance(
        #         data_collection_index=trial_index, add_name='pre_tr', particle_pred=False)
        #     for state_dim_index in range(self.state_dim):
        #         trials_results[trial_index]['ML_rollout_mse_before_update'].append(
        #             ((state_samples_history[:, state_dim_index] - rollout_states[:, state_dim_index]) ** 2).mean())
        #
        #     trials_results[trial_index]['target_dist_before_update'] = math.dist(state_samples_history[-1, :2],
        #                                                                          rollout_states[-1, :2])
        #     print('\n\n------ Target dist (before model update): {}'.format(
        #         trials_results[trial_index]['target_dist_before_update']))
        #
        #     # Testing the model with uncertain rollouts with init_v = policy(x_0) + uncertainty
        #     # uncertainty: wgn with std = (error of predicted landing in previous rollout) / (trajectory time)
        #
        #     # Initial state (random)
        #
        #     # Predicted trajectory
        #     actual_trajectory = state_samples_history
        #
        #     # self.plot_model_learning(trials_results, trial_index, actual_trajectory,
        #     #                          rollout_states, num_simulation_samples, plot_noised_trs)
        #
        #     if verbose_print:
        #         print('Predicted trajectory; {}'.format(rollout_states))
        #
        #     # train GPs on observed interaction data
        #     print('\n\n----- REINFORCE THE MODEL -----')
        #     t_model_learning_start = time.time()
        #     self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)
        #     t_model_learning_stop = time.time()
        #     print('\nModel optimization time: ' + str(t_model_learning_stop - t_model_learning_start))
        #
        #     # get model learning performance
        #     print('\n\n----- CHECK THE LEARNING PERFORMANCE (after model update) on trial data -----')
        #     trials_results[trial_index]['ML_gp_mse_after_update'] = []
        #     _, gp_outputs_target_list, gp_output_mean_list, gp_output_var_list = self.get_model_learning_performance(
        #         data_collection_index=trial_index)
        #     for gp_index in range(0, self.model_learning.num_gp):
        #         trials_results[trial_index]['ML_gp_mse_after_update'].append(
        #             ((gp_outputs_target_list[gp_index] - gp_output_mean_list[gp_index]) ** 2).mean())
        #
        #     if vel_indices is None:
        #         vel = None
        #     else:
        #         vel = self.state_samples_history[trial_index][:, vel_indices]
        #     # self.plot_GP_inference(gp_outputs_target_list, gp_output_mean_list, vel=vel, title = 'after model update')
        #     # plt.show()
        #
        #     # get policy performance
        #     print('\n\n----- CHECK THE ROLLOUT PERFORMANCE (after model update) on trial data -----')
        #     trials_results[trial_index]['ML_rollout_mse_after_update'] = []
        #     rollout_states, state_samples_history, _ = self.get_rollout_prediction_performance(
        #         data_collection_index=trial_index, particle_pred=False,
        #         add_name='post_tr')
        #     for state_dim_index in range(self.state_dim):
        #         trials_results[trial_index]['ML_rollout_mse_after_update'].append(
        #             ((state_samples_history[:, state_dim_index] - rollout_states[:, state_dim_index]) ** 2).mean())
        #     trials_results[trial_index]['target_dist_after_update'] = math.dist(state_samples_history[-1, :2],
        #                                                                         rollout_states[-1, :2])
        #     print('\n\n------ Target dist (after model update): {}'.format(
        #         trials_results[trial_index]['target_dist_after_update']))
        #
        #     # self.plot_model_learning(trials_results, trial_index, actual_trajectory,
        #     #                          rollout_states, num_simulation_samples, plot_noised_trs)
        #
        #     self.print_test_after_model_update(gp_output_mean_list, gp_output_var_list)
        #
        #     plt.show()
        # return trials_results

    def optimize_t_delay_distribution(self, init_function, particle_pred, t_delay_left_lim, t_delay_right_lim, delta_delay, resolution):
        """
            Requires the model to be optimized

            init_function: f(initial_vel_norm, yaw, t_delay) --> [p_0 + p_0_dot]
        """
        # Grid search for distribution parameters

        dist_grid = []
        fitness_values = []
        dist_particles_landing_positions = []
        for a in np.arange(t_delay_left_lim, t_delay_right_lim, delta_delay / resolution):
            for b in np.arange(delta_delay / resolution, delta_delay + delta_delay / resolution, delta_delay / resolution):
                print('Test U({},{})'.format(a, a + b))
                dist_grid.append([a, b])
                t_delay_dist = lambda: np.random.uniform(a, a + b)
                trajectories_mse, mean_mse, particles_landing_positions = (
                    self.simulate_system(t_delay_dist, self.trajectories_initial_conditions, init_function, particle_pred))

                print('Mean MSE: {}\n\n'.format(mean_mse))

                fitness_values.append(mean_mse)
                dist_particles_landing_positions.append(particles_landing_positions)

        fitness_values = np.array(fitness_values)

        return dist_grid, fitness_values, dist_particles_landing_positions





    def simulate_system(self, dist_fun, trajectories_initial_conditions, init_function, particle_pred):
        num_particles_test = 10
        max_simulated_samples = round(.75 / self.T_sampling)

        # print('max_simulated_samples:', max_simulated_samples)

        trajectories_mse = []
        particles_landing_positions = []

        for trajectories_initial_condition in trajectories_initial_conditions:
            trajectory_target_point, trajectory_policy_output, trajectory_landing = trajectories_initial_condition

            target_point_yaw = np.arctan2(trajectory_target_point[1], trajectory_target_point[0])
            target_altitude = trajectory_target_point[2]

            f_init_particles = lambda v: init_function(v, target_point_yaw, dist_fun())

            particles_sequence_list = self.get_particles_evolution_robot_kin(f_init_particles,
                                                                             num_particles_test,
                                                                             max_simulated_samples,
                                                                             initial_vel=trajectory_policy_output.item(),
                                                                             particle_pred=particle_pred)
            particles_positions_list = []

            for i in range(num_particles_test):
                particle_pos = np.array(
                    [particles_sequence_list[t][i, list(range(3))].cpu().detach().numpy() for t in
                     range(max_simulated_samples)])
                # print(particle_pos)
                particles_positions_list.append(particle_pos)

            landing_positions = []
            for particles_positions in particles_positions_list:
                crossing = interpolate_descending_plane_crossing(
                    particles_positions, target_altitude
                )
                if crossing is not None:
                    landing_positions.append(crossing)
                # print(particles_positions[-1, :3])
            landing_positions = np.array(landing_positions)
            # print(landing_positions.shape)
            particles_errors = landing_positions[:, :2] - trajectory_target_point[:2]

            trajectories_mse.append(np.mean(particles_errors[:, 0] ** 2 + particles_errors[:, 1] ** 2))
            particles_landing_positions.append(landing_positions)

        trajectories_mse = np.array(trajectories_mse)

        return trajectories_mse, np.mean(trajectories_mse), particles_landing_positions




def rotation_z(trj, gamma=np.pi):
    R = np.array([[[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]]])
    R = np.repeat(R, trj.shape[0], axis=0)

    times = trj[:, 0:1]
    points_rot = np.matmul(R, np.expand_dims(trj[:, 1:], axis=-1)).squeeze(-1)

    return np.hstack([times, points_rot])


def random_z_rotation(trj, angle_max=np.pi / 6):
    gamma = np.random.rand() * angle_max
    return rotation_z(trj, gamma=gamma)
