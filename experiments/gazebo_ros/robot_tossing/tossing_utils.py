# SPDX-License-Identifier: AGPL-3.0-or-later
import math
import os
import numpy as np
import rospy
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import model_learning.Model_learning as ML
import simulation_class.tossing_dynamics_prior as f_ode_4control
import copy
from mcpilot.srv import TossExperiment
from nav_msgs.msg import Path, Odometry
import policy_learning.Policy as Policy
import policy_learning.Policy_Tossing as Policy_Tossing
import pickle as pkl

# release_position = [0.754605189645874, 0.0, 1.1079377717134171]  # geometric computed: [0.7482395118728365, 0.0, 1.0916519738370616]
release_position = [0.0753151405590284, 0.0, 1.50204328412483]
V_MAX = 3.5
target_altitude = -.72
alpha = -3 * math.pi / 180 #np.pi / 4
min_distance = 0.75


num_basis = 250
#num_basis = 200
experiment_srv_name = 'tossing_experiment_testing'
learning_experiment_srv_name = 'tossing_experiment_learning'
bullet_state_topic_format = '{}_odom'

f_init_particles_50cm = lambda: tossing_init(0.1 + np.random.rand() * 0.4, 2 * (np.random.rand() - 0.5) * np.pi / 6, target_altitude)
f_init_particles_70cm = lambda: tossing_init(0.1 + np.random.rand() * 0.6, 2 * (np.random.rand() - 0.5) * np.pi / 6, target_altitude)
f_init_particles_90cm = lambda: tossing_init(0.1 + np.random.rand() * 0.8, 2 * (np.random.rand() - 0.5) * np.pi / 6, target_altitude)

f_init_particles = lambda dist, rel: (lambda: tossing_init(min_distance + np.random.rand() * dist,
                                                           2 * (np.random.rand() - 0.5) * np.pi / 6,
                                                           # 0.0,
                                                           target_altitude,
                                                           release_pos=rel))

distances_lengthscales = [0.5]*3
cumulative_cost_lengthscales = [0.5]*2
last_pos_cost_lengthscales = [0.10]*2


def gen_batch_of_targets(target_dist_range, target_z_range, yaw_range, batch_size):
    """
        Generates a batch of uniformly sampled targets at given (planar) distance, altitude, yaw ranges
    """
    gamma = np.arange(yaw_range[0], yaw_range[1], (yaw_range[1] - yaw_range[0]) / batch_size)

    d = target_dist_range[0] + np.random.rand(batch_size) * (target_dist_range[1] - target_dist_range[0])
    z = target_z_range[0] + np.random.rand(batch_size) * (target_z_range[1] - target_z_range[0])

    s = [[release_position[0] * np.cos(gamma[i]), release_position[1] * np.sin(gamma[i]), release_position[2],
          0.0, 0.0, 0.0, d[i] * np.cos(gamma[i]), d[i] * np.sin(gamma[i]), z[i]] for i in range(batch_size)]

    return np.array(s)


def generate_policy_input_tensor(max_x=1.25, max_y=0.625, dx=0.05, target_height=None):
    X = np.arange(release_position[0], max_x, dx)
    Y = np.arange(-max_y, max_y, dx)

    if target_height is None:
        target_height = target_altitude

    Z = np.array([target_height])

    inputs = []

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for z in Z:
                yaw = math.atan2(y, x)
                r_pos = [release_position[0] * math.cos(yaw), release_position[1] * math.sin(yaw), release_position[2]]
                inputs.append(r_pos + [0]*3 + [x, y, z])

    return torch.tensor(inputs), X, Y


def reference_toss_trj(release, target, num_points, alpha=math.pi / 4, g=9.81):
    """

    :param release: Starting position of the trajectory
    :param target:  Target position where the bullet is supposed to land
    :param alpha:   Angle of the initial velocity w.r.t. horizontal plane
    :param g:       Gravity acceleration

    :return:    Reference trajectory of the launch to reach target from release position
    """
    trj = []
    trj.append(release)
    dx = math.sqrt((target[0] - release[0]) ** 2 + (target[1] - release[1]) ** 2)
    dz = target[2] - release[2]
    v_norm = math.sqrt(0.5 * g * (dx ** 2) / ((math.cos(alpha) ** 2) * (math.tan(alpha) * dx - dz)))

    yaw = math.atan2(target[1], target[0])
    v_x = math.cos(alpha) * v_norm
    v_z = math.sin(alpha) * v_norm
    trj_time = dx / v_x
    dt = trj_time / num_points

    times = np.linspace(start=0, stop=(num_points - 1) * dt, num=num_points)
    for t in times:
        trj.append([release[0] + v_x * t * math.cos(yaw),
                    release[1] + v_x * t * math.sin(yaw),
                    release[2] + v_z * t - 0.5 * g * t ** 2])

    return np.array(trj)


def plot_ditances_batches_of_points(points0, points1):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points0[:, 0], points0[:, 1], points0[:, 2], color='r')
    ax.scatter3D(points1[:, 0], points1[:, 1], points1[:, 2], color='b')
    for i in range(points0.shape[0]):
        ax.plot([points0[i, 0], points1[i, 0]], [points0[i, 1], points1[i, 1]], [points0[i, 2], points1[i, 2]],
                color='g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def tossing_init(dist, yaw, altitude, release_pos=None):
    """
        dist: distance projected on the x-y plane
        yaw: angle of toss
        altitude: target alltitude in world frame
    """
    if release_pos is None:
        release_pos = copy.copy(release_position)  # use default

    x_0 = [release_pos[0] * math.cos(yaw), release_pos[0] * math.sin(yaw), release_pos[2]]
    x_0_dot = [0] * 3
    x_T = [(release_pos[0] + dist)*math.cos(yaw), (release_pos[0] + dist)*math.sin(yaw), altitude]

    return x_0 + x_0_dot + x_T


def get_speed_model(num_gp, T_sampling, device, dtype, flg_GP_mean, state_dim=6, target_dim=3, input_dim=3,
                    sigma_n_num=5*(10**(-3)), set_integration_cond=True, add_MPK=False, flg_approximation=None,
                    gp_adaptive_init=False):
    """
        Returns the Speed-integration model with RBF kernel or RBF + MPK
    """
    if add_MPK:
        f_model_learning = ML.Speed_Model_learning_RBF_MPK
    else:
        f_model_learning = ML.Speed_Model_learning_RBF

    print(f_model_learning)
    model_learning_par = {}
    model_learning_par['num_gp'] = num_gp
    model_learning_par['T_sampling'] = T_sampling
    model_learning_par['vel_indeces'] = [3, 4, 5]
    model_learning_par['not_vel_indeces'] = [0, 1, 2]
    model_learning_par['cnst_indices'] = [6, 7, 8]
    if set_integration_cond:
        model_learning_par['integration_cond'] = lambda state: ((state[:, 2] - state[:, 8]) > 0.0).reshape([-1, 1]).float()
    else:
        model_learning_par['integration_cond'] = None

    model_learning_par['device'] = device
    model_learning_par['dtype'] = dtype
    model_learning_par['gp_adaptive_init'] = gp_adaptive_init

    if flg_approximation == 'SOD':
        model_learning_par['approximation_mode'] = 'SOD'
        model_learning_par['approximation_dict'] = {'SOD_threshold_mode': 'absolute',
                                                    'SOD_threshold': [10**(-4)]*num_gp,
                                                    'flg_SOD_permutation': False,
                                                    'criterion': 'error'}  # Set SoD approximation parameters
    elif flg_approximation == 'SOR':
        model_learning_par['approximation_mode'] = 'SOR'
        model_learning_par['approximation_dict'] = {'SOR_threshold_mode': 'absolute',
                                                    'SOR_threshold': [0.003]*num_gp,
                                                    'downsampling_rate': 10,
                                                    'flg_regressors_trainable': True,
                                                    'criterion': 'downsampling'}


    init_dict = {}
    # RBF initial par
    # init_dict['active_dims'] = np.arange(int(state_dim/2)-1, state_dim)  # z, dot_x, dot_y, dot_z
    init_dict['active_dims'] = np.arange(int(state_dim / 2), state_dim)  # dot_x, dot_y, dot_z
    init_dict['lengthscales_init'] = np.ones(init_dict['active_dims'].size)
    init_dict['flg_train_lengthscales'] = True
    init_dict['scale_init'] = np.ones(1)
    init_dict['flg_train_scale'] = True
    init_dict['sigma_n_init'] = 1 * np.ones(1)
    init_dict['sigma_n_num'] = sigma_n_num
    init_dict['flg_train_sigma_n'] = True

    init_dict['dtype'] = dtype
    init_dict['device'] = device

    if flg_GP_mean:
        # set the mean functionsù
        print('\nLoading Gravity mean function!\n')
        f_mean_add_par_dict = {'active_dims_x': list(range(int(state_dim / 2))),
                               'active_dims_u': list(range(state_dim + target_dim, state_dim + target_dim + input_dim)),
                               'active_dims_x_dot': list(range(int(state_dim / 2), state_dim)),
                               'T_sampling': T_sampling}

        init_dict['f_mean'] = f_ode_4control.tossing_delta_theta_dot
        init_dict_list = []
        for i in range(num_gp):
            f_mean_add_par_dict['out_dim'] = i  # 0:x, 1:y, 2: z
            init_dict['f_mean_add_par_dict'] = f_mean_add_par_dict
            init_dict_list.append(copy.deepcopy(init_dict))
    else:
        init_dict_list = [init_dict] * num_gp

    if add_MPK:
        init_dict_MPK = {}
        init_dict_MPK['active_dims'] = np.arange(int(state_dim / 2), state_dim)  # dot_x, dot_y, dot_z
        init_dict_MPK['sigma_n_init'] = 1 * np.ones(1)
        init_dict_MPK['sigma_n_num'] = sigma_n_num
        init_dict_MPK['flg_train_sigma_n'] = True
        init_dict_MPK['dtype'] = dtype
        init_dict_MPK['device'] = device
        init_dict_MPK['poly_deg'] = 2
        print('---------')
        print(init_dict['active_dims'].shape[0])
        print('---------')
        init_dict_MPK['Sigma_pos_par_init'] = \
            [np.ones([deg + 1, init_dict['active_dims'].shape[0] + 1]) if deg == 0 else np.ones([deg + 1, init_dict['active_dims'].shape[0]]) for deg in range(init_dict_MPK['poly_deg'])]
        init_dict_MPK['Sigma_pos_par_init'] = np.ones([init_dict['active_dims'].shape[0] + 1, init_dict['active_dims'].shape[0] + 1])
        print('-----------------------')
        print(init_dict_MPK['Sigma_pos_par_init'])
        print('-----------------------')

        init_dict_MPK['flg_train_Sigma_pos_par'] = True #[True] * init_dict_MPK['poly_deg']

        init_dict_list_MPK = [init_dict_MPK] * num_gp

        model_learning_par['init_dict_list'] = [[init_dict_list[i], init_dict_list_MPK[i]] for i in range(num_gp)]
    else:
        model_learning_par['init_dict_list'] = init_dict_list

    return f_model_learning, model_learning_par


def get_delta_state_model(num_gp, T_sampling, device, dtype, flg_GP_mean, state_dim=6, target_dim=3, input_dim=3):
    f_model_learning = ML.Model_learning_RBF_keep
    print(f_model_learning)
    model_learning_par = {}
    model_learning_par['num_gp'] = num_gp
    # model_learning_par['T_sampling'] = T_sampling
    model_learning_par['state_indeces'] = [0, 1, 2, 3, 4, 5]
    model_learning_par['keep_indeces'] = [6, 7, 8]
    model_learning_par['device'] = device
    model_learning_par['dtype'] = dtype
    init_dict = {}
    # RBF initial par
    # init_dict['active_dims'] = np.arange(int(state_dim/2), state_dim)
    # init_dict['active_dims'] = np.arange(int(state_dim/2)-1, state_dim)
    init_dict['active_dims'] = np.arange(0, state_dim)
    init_dict['lengthscales_init'] = np.ones(init_dict['active_dims'].size)
    init_dict['flg_train_lengthscales'] = True
    init_dict['scale_init'] = np.ones(1)
    init_dict['flg_train_scale'] = False
    init_dict['sigma_n_init'] = 1 * np.ones(1)
    init_dict['sigma_n_num'] = None
    init_dict['flg_train_sigma_n'] = True

    init_dict['dtype'] = dtype
    init_dict['device'] = device

    if flg_GP_mean:
        # set the mean functions
        f_mean_add_par_dict = {'active_dims_x': list(range(state_dim)),
                               'active_dims_pos': list(range(int(state_dim / 2))),
                               'active_dims_vel': list(range(int(state_dim / 2), state_dim)),
                               'T_sampling': T_sampling}

        init_dict['f_mean'] = f_ode_4control.tossing_delta_state
        init_dict_list = []
        for i in range(num_gp):
            f_mean_add_par_dict['out_dim'] = i  # 0:x, 1:y, 2: z, 3: x_dot, 4: y_dot, 5: z_dot
            init_dict['f_mean_add_par_dict'] = f_mean_add_par_dict
            init_dict_list.append(copy.deepcopy(init_dict))
    else:
        init_dict_list = [init_dict] * num_gp

    model_learning_par['init_dict_list'] = init_dict_list

    return f_model_learning, model_learning_par


def getPolicy(u_max, state_dim, target_dim, device, dtype, max_dist=0.5, max_yaw=math.pi/6, centers_init='sparse'):
    """
    Type:
        -simple: (planar dist, vertical dist) -> vel
        -complete: (dx, dy, dz) -> vel
        -target: taget coord -> vel
    """
    control_policy_par = {}
    # state_dim, num_basis, target_indeces, alpha, pos_indeces
    control_policy_par['state_dim'] = int(state_dim / 2)
    control_policy_par['target_indeces'] = list(range(state_dim, state_dim + target_dim))
    control_policy_par['alpha'] = alpha
    control_policy_par['pos_indeces'] = list(range(int(state_dim / 2)))
    # control_policy_par['input_dim'] = input_dim
    control_policy_par['u_max'] = u_max
    control_policy_par['num_basis'] = num_basis
    control_policy_par['dtype'] = dtype
    control_policy_par['device'] = device
    control_policy_par['squash_name'] = 'tanh'

    if centers_init == 'sparse':
        dist = min_distance + max_dist
        centers = np.hstack([dist * np.random.rand(num_basis, 1),
                                       dist * math.sin(max_yaw) * 2 * (np.random.rand(num_basis, 1) - 0.5),
                                       target_altitude * np.ones((num_basis, 1))])
    elif centers_init == 'line':
        x_y = np.random.rand(num_basis, 1) * max_dist + np.array([min_distance, -max_dist/2])
        centers = np.hstack([x_y, target_altitude * np.ones((num_basis, 1))])
    elif centers_init == 'focus':
        sigma = 0.05
        x = np.random.rand(num_basis, 1) * sigma + min_distance + max_dist/2
        y = np.random.rand(num_basis, 1) * sigma
        centers = np.hstack([x, y, target_altitude * np.ones((num_basis, 1))])
    else:
        return None

    f_control_policy = Policy_Tossing.TossingPolicyTarget
    print(f_control_policy)
    control_policy_par['lengthscales_init'] = np.array(distances_lengthscales)  # 1 * np.ones(int(state_dim/2))

    print('Init policy centers: {}'.format(centers))

    # cos_centers = np.cos(angle_centers)
    # sin_centers = np.sin(angle_centers)
    # not_angle_centers = np.pi * 2 * (np.random.rand(num_basis, 3) - 0.5)
    control_policy_par['centers_init'] = np.array(centers)
    control_policy_par['weight_init'] = u_max * (np.random.rand(1, num_basis) - 0.5)
    control_policy_par['flg_train_lengthscales'] = False
    control_policy_par['flg_squash'] = True
    control_policy_par['flg_drop'] = True
    policy_reinit_dict = {}
    policy_reinit_dict['lenghtscales_par'] = control_policy_par['lengthscales_init']
    policy_reinit_dict['centers_par'] = np.array([1.] * 5)
    policy_reinit_dict['weight_par'] = u_max

    return policy_reinit_dict, control_policy_par, f_control_policy


def test_policy(policy_model, num_trials, seed, f_init_particles=f_init_particles_50cm, save=False, save_post_str='1',
                use_policy=True, bullet_name='red_ball', orientation=None, save_dir=None):
    """
        Test the policy on a set of random sampled targets

    """
    if orientation is None:
        orientation = [0] * 3
    print('Summary')
    print(policy_model)

    results = {}

    for i in range(num_trials):
        print('\n{}/{}'.format(i+1, num_trials))
        rospy.wait_for_service(experiment_srv_name)
        srv = rospy.ServiceProxy(experiment_srv_name, TossExperiment)

        initial_state = f_init_particles()
        with torch.no_grad():
            policy_cmd, _ = policy_model.get_v(torch.tensor([initial_state], device=policy_model.device, dtype=policy_model.dtype), t=0, p_dropout=0.0)

        print('V_toss: {}'.format(policy_cmd))

        policy_cmd = max(policy_cmd.detach().cpu().numpy()[0], 0.1)
        try:
            res = srv(initial_state[6:], not use_policy, policy_cmd, bullet_name, orientation)
            if res is None:
                print('Tossing service failed with generic error.')
                continue
        except Exception as exc:
            print('Tossing service failed: {}'.format(repr(exc)))
            continue

        # Use the trajectory returned by the toss service as the source of
        # truth for the landing position.  Waiting for a fresh `<bullet>_odom`
        # message here is unsafe: after the service returns, the next odometry
        # packet can already belong to a reset/repositioned projectile.
        #
        # tossing_experiment_script.py serializes the N x 10 trajectory one
        # complete column at a time:
        #   [time, x, y, z, vx, vy, vz, wx, wy, wz].
        flat_trajectory = np.asarray(res.trajectory, dtype=float)
        num_columns = 10

        if flat_trajectory.size == 0:
            print('Tossing service returned an empty trajectory.')
            continue
        if flat_trajectory.size % num_columns != 0:
            print(
                'Invalid tossing trajectory size: {} is not divisible by {}'.format(
                    flat_trajectory.size, num_columns
                )
            )
            continue

        num_samples = flat_trajectory.size // num_columns
        trajectory = flat_trajectory.reshape(num_columns, num_samples).T

        # read_bullet_state.py terminates the trajectory at the target plane,
        # so the last returned row is the landing/crossing state of this exact
        # throw.  Preserve the historical planar success metric by comparing
        # x-y only (the old code set both z coordinates to zero).
        ball_pos = [trajectory[-1, 1], trajectory[-1, 2], 0.0]

        target = initial_state[6:8] + [0.0]
        dist_from_target = math.dist(ball_pos, target)
        overshoot = (np.linalg.norm(ball_pos) - np.linalg.norm(target)) > 0
        print('dist: {}{}'.format('+' if overshoot else '-', dist_from_target))

        results[i] = (target, dist_from_target if overshoot else -dist_from_target)  # save both the target location and the distance where the ball fell.
        if save:
            if save_dir is None:
                save_dir = 'results_tmp/{}'.format(seed)
            os.makedirs(save_dir, exist_ok=True)
            pkl.dump(
                results,
                open(
                    os.path.join(
                        save_dir,
                        'tossing_test_{}.pkl'.format(save_post_str),
                    ),
                    'wb',
                ),
            )

    return results


def plot_policy_test_results(results, target_radius=0.05):
    """
        use pyplot.show() after the function to plot graphs
    """
    distances = [results[i][1] for i in results.keys()]

    fig, ax = plt.subplots()
    ax.set_title('Tossing performances (distance from target)')
    ax.scatter(list(range(1, len(distances) + 1)), distances, color='b', marker='o')
    ax.plot(list(range(1, len(distances) + 1)), [target_radius] * len(distances), color='red')
    ax.plot(list(range(1, len(distances) + 1)), [-target_radius] * len(distances), color='red')

    print('Performance: {}% target reach'.format(100 * np.count_nonzero(np.abs(np.array(distances)) < target_radius) / len(distances)))

    plt.figure('Tossing perf eval (green: target reach, red: target miss)', figsize=(4,3))
    ax = plt.axes(projection='3d')
    for k in results.keys():
        target, dist = results[k]
        if abs(dist) <= target_radius:
            col = 'g'
            ax.scatter3D(target[0], target[1], target[2], color=col)
        else:
            col = 'r'
            if dist > 0:
                ax.scatter3D(target[0], target[1], target[2], color=col, marker="^")
            else:
                ax.scatter3D(target[0], target[1], target[2], color=col, marker="v")

    ax.scatter3D(release_position[0], release_position[1], release_position[2], color='b', marker='^', s=100)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.grid()


# _, sparse_par, _ = getPolicy(V_MAX, 6, 3, 'cpu', float, centers_init='sparse')
# _, line_par, _ = getPolicy(V_MAX, 6, 3, 'cpu', float, centers_init='line')
# _, focus_par, _ = getPolicy(V_MAX, 6, 3, 'cpu', float, centers_init='focus')
#
# fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
#
# fig.set_figheight(3)
# fig.set_figwidth(6)
#
# axs[0].set_title('sparse')
# axs[0].scatter(sparse_par['centers_init'][:, 1], sparse_par['centers_init'][:, 0], s=10, c=sparse_par['weight_init'])
#
# axs[1].set_title('line')
# axs[1].scatter(line_par['centers_init'][:, 1], line_par['centers_init'][:, 0], s=10, c=line_par['weight_init'])
#
# axs[2].set_title('focus')
# axs[2].scatter(focus_par['centers_init'][:, 1], focus_par['centers_init'][:, 0], s=10, c=focus_par['weight_init'])
#
# plt.tight_layout()
# plt.show()

# plot_ditances_batches_of_points(gen_batch_of_targets([1.2, 1.6], [0.1, 0.1], [-np.pi / 6, np.pi / 6], 10)[:, 6:],
#                                gen_batch_of_targets([1.2, 1.6], [0.1, 0.1], [-np.pi / 6, np.pi / 6], 10)[:, 6:])

# results = pkl.load(open('toss_exp_results/MC_PILCO_tossing_test_50trials_policy__cost_last_pos_200particles_100Hz.pkl', 'rb'))
# plot_policy_test_results(results, 0.05)
# plt.show()
