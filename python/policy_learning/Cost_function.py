# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import torch
import numpy as np
import matplotlib.pyplot as plt


class Expected_cost(torch.nn.modules.loss._Loss):
    """ 
    Expected cost class. The cost is computed through a torch function defined in the initialization
    """

    def __init__(self, cost_function):
        """Initialize the object"""
        super(Expected_cost, self).__init__()
        self.cost_function = cost_function

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """ Computes the global cost applying self.cost_function.
            States_sequence.shape: [num_instants, num_particles, state_dim]
            inputs_sequence.shape: [num_instants, num_particles, input_dim]
        """

        # Returns the sum of the expected costs
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        mean_costs = torch.mean(costs, 1)  # average cost at each time step over particles ...
        std_costs = torch.std(costs.detach(), 1)  # ... and corresponding std

        return torch.sum(mean_costs), torch.sum(std_costs)


class Expected_cost_with_variance(torch.nn.modules.loss._Loss):
    """
    Expected cost class. The cost is computed through a torch function defined in the initialization
    """

    def __init__(self, cost_function, variance_lengthscale):
        """Initialize the object"""
        super(Expected_cost_with_variance, self).__init__()
        self.cost_function = cost_function
        self.l2_variance = 2 * variance_lengthscale ** 2

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """ Computes the global cost applying self.cost_function.
            States_sequence.shape: [num_instants, num_particles, state_dim]
            inputs_sequence.shape: [num_instants, num_particles, input_dim]
        """

        # Returns the sum of the expected costs
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        mean_costs = torch.mean(costs, 1)  # average cost at each time step over particles ...
        std_costs = torch.std(costs.detach(), 1)  # ... and corresponding std

        # compute covariances
        if states_sequence.shape[1] > 1:
            det_cov = torch.tensor(
                [torch.det(torch.cov(states_sequence[t, :, :].T)) for t in range(states_sequence.shape[0])])
            # print('det_cov', det_cov)
            mean_costs = mean_costs * (1 + torch.exp(-det_cov / self.l2_variance))

        return torch.sum(mean_costs), torch.sum(std_costs)


class Min_cost(torch.nn.modules.loss._Loss):
    """
    min cost class. The cost is computed through a torch function defined in the initialization
    """

    def __init__(self, cost_function):
        """Initialize the object"""
        super(Min_cost, self).__init__()
        self.cost_function = cost_function

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """ Computes the global cost applying self.cost_function.
            States_sequence.shape: [num_instants, num_particles, state_dim]
            inputs_sequence.shape: [num_instants, num_particles, input_dim]
        """

        # Returns the sum of the expected costs
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        min_costs = torch.min(torch.sum(costs, 0))  # average cost at each time step over particles ...
        std_costs = torch.zeros_like(min_costs)

        return min_costs, std_costs


class Expected_distance(Expected_cost):
    """
    Cost function given by the sum of the expected distances from target state
    """

    def __init__(self, target_state, lengthscales, active_dims):
        # get the distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: distance_from_target(x, u, trial_index, target_state=target_state,
                                                                lengthscales=lengthscales, active_dims=active_dims)
        # initit the superclass with the lambda function
        super(Expected_distance, self).__init__(f_cost)


def distance_from_target(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, active_dims):
    # normalize states and targets (consider only used states)
    norm_states = states_sequence[:, :, active_dims] / lengthscales
    norm_target = target_state / lengthscales

    # get the square distance
    dist = torch.sum(norm_states ** 2, dim=2, keepdim=True)
    dist = dist + torch.sum(norm_target ** 2, dim=1, keepdim=True).transpose(0, 1)
    dist -= 2 * torch.matmul(norm_states, norm_target.transpose(dim0=0, dim1=1))
    # return the cost
    return dist


class Expected_saturated_distance(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from target state
    """

    def __init__(self, target_state, lengthscales, active_dims):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_target(x, u, trial_index,
                                                                          target_state=target_state,
                                                                          lengthscales=lengthscales,
                                                                          active_dims=active_dims)
        # initit the superclass with the lambda function
        super(Expected_saturated_distance, self).__init__(f_cost)


def saturated_distance_from_target(states_sequence, inputs_sequence, trial_index, target_state, lengthscales,
                                   active_dims):
    """ 
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """

    # get state components evaluated in the cost
    active_states = states_sequence[:, :, active_dims]

    # normalize states and targets
    norm_states = active_states / lengthscales
    norm_target = target_state / lengthscales
    # get the square distance
    # dist = torch.sum(norm_states ** 2, dim=2, keepdim=True)
    # dist = dist + torch.sum(norm_target ** 2, dim=1, keepdim=True).transpose(0, 1)
    # dist -= 2 * torch.matmul(norm_states, norm_target.transpose(dim0=0, dim1=1))
    dist = torch.sum(((norm_states - norm_target)) ** 2, dim=2)

    cost = 1 - torch.exp(-dist)

    return cost


class Expected_saturated_distance_from_trajectory(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from a target trajectory
    """

    def __init__(self, target_traj, lengthscales, flg_var_lengthscales=False, used_indeces=None):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_trajectory(x, u, trial_index,
                                                                              target_traj=target_traj,
                                                                              lengthscales=lengthscales,
                                                                              flg_var_lengthscales=flg_var_lengthscales,
                                                                              used_indeces=used_indeces)
        # initit the superclass with the lambda function
        super(Expected_saturated_distance_from_trajectory, self).__init__(f_cost)


def saturated_distance_from_trajectory(states_sequence, inputs_sequence, trial_index, target_traj, lengthscales,
                                       flg_var_lengthscales, used_indeces):
    """ 
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """
    if used_indeces == None:
        used_indeces = list(range(0, states_sequence.shape[2]))

    # get state components evaluated in the cost
    print(target_traj.shape)
    targets = target_traj.repeat(1, states_sequence.shape[1]).view(states_sequence.shape)
    print(targets.shape)
    exit()
    if flg_var_lengthscales:
        dist = torch.sum(
            ((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales[trial_index]) ** 2,
            dim=2)
    else:
        dist = torch.sum(((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales) ** 2,
                         dim=2)
    cost = 1 - torch.exp(-dist)

    return cost


class Expected_saturated_distance_from_trajectory_in_pieces(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from a target trajectory
    """

    def __init__(self, repeated_trajectory, lengthscales, flg_var_lengthscales=False, used_indeces=None):
        # get the saturated distance function as a function of states and inputs

        f_cost = lambda x, u, trial_index: saturated_distance_from_target_trajectory(x, u, trial_index,
                                                                                     target_traj=repeated_trajectory,
                                                                                     lengthscales=lengthscales,
                                                                                     flg_var_lengthscales=flg_var_lengthscales,
                                                                                     used_indeces=used_indeces)
        # initit the superclass with the lambda function
        super(Expected_saturated_distance_from_trajectory_in_pieces, self).__init__(f_cost)

def repeat_trajectory_in_pieces(target_traj, num_pieces, num_particles):
    points = []
    piece_len = int(target_traj.shape[0] / num_pieces)
    for n in range(num_pieces):
        points.append(target_traj[n * piece_len:(n + 1) * piece_len])

    repeated_trajectory = torch.zeros((int(target_traj.shape[0]/num_pieces), num_particles, target_traj.shape[1]),
                                      dtype=target_traj.dtype, device=target_traj.device)
    for n in range(num_pieces):
        repeated_trajectory[:, n * int(num_particles / num_pieces):(n + 1) * int(num_particles / num_pieces), :] = copy.deepcopy(points[n]).unsqueeze(1).repeat(1, int(num_particles / num_pieces), 1)
    return repeated_trajectory, points


def saturated_distance_from_target_trajectory(states_sequence, inputs_sequence, trial_index, target_traj, lengthscales,
                                       flg_var_lengthscales, used_indeces):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """
    if used_indeces == None:
        used_indeces = list(range(0, states_sequence.shape[2]))

    # get state components evaluated in the cost
    targets = target_traj # repeat(1, states_sequence.shape[1]).view(states_sequence.shape)
    if flg_var_lengthscales:
        dist = torch.sum(
            ((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales[trial_index]) ** 2,
            dim=2)
    else:
        diff = states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]
        dist = torch.sum((diff / lengthscales) ** 2, dim=2)
    cost = 1 - torch.exp(-dist)

    return cost



class Cart_pole_cost(Expected_cost):
    """ Cost for the cart pole system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad], target position [m]).
    """

    def __init__(self, target_state, lengthscales, angle_index, pos_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: cart_pole_cost(x, u, trial_index, target_state=target_state,
                                                          lengthscales=lengthscales,
                                                          angle_index=angle_index,
                                                          pos_index=pos_index)
        # initit the superclass with the lambda function
        super(Cart_pole_cost, self).__init__(f_cost)


def cart_pole_cost(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, angle_index, pos_index):
    """ 
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x = states_sequence[:, :, pos_index]
    theta = states_sequence[:, :, angle_index]

    target_x = target_state[1]
    target_theta = target_state[0]

    return (1 - torch.exp(
        -((torch.abs(theta) - target_theta) / lengthscales[0]) ** 2 - ((x - target_x) / lengthscales[1]) ** 2))


class Cart_pole_mujoco_cost(Expected_cost):
    """ Cost for the cart pole system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad], target position [m]).
    """

    def __init__(self, target_state, lengthscales, angle_index, pos_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: cart_pole_mujoco_cost(x, u, trial_index, target_state=target_state,
                                                          lengthscales=lengthscales,
                                                          angle_index=angle_index,
                                                          pos_index=pos_index)
        # initit the superclass with the lambda function
        super(Cart_pole_mujoco_cost, self).__init__(f_cost)


def cart_pole_mujoco_cost(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, angle_index, pos_index):
    """ 
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x = states_sequence[:, :, pos_index]
    theta = states_sequence[:, :, angle_index] - torch.pi

    target_x = target_state[1]
    target_theta = target_state[0] + torch.pi

    return (1 - torch.exp(
        -((torch.abs(theta) - target_theta) / lengthscales[0]) ** 2 - ((x - target_x) / lengthscales[1]) ** 2))


class Cart_double_pole_mujoco_cost(Expected_cost):
    """ Cost for the cart pole system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad], target position [m]).
    """

    def __init__(self, target_state, lengthscales, angle_index, pos_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: cart_double_pole_mujoco_cost(x, u, trial_index, target_state=target_state,
                                                          lengthscales=lengthscales,
                                                          angle_index=angle_index,
                                                          pos_index=pos_index)
        # initit the superclass with the lambda function
        super(Cart_double_pole_mujoco_cost, self).__init__(f_cost)


def cart_double_pole_mujoco_cost(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, angle_index, pos_index):
    """ 
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x = states_sequence[:, :, pos_index]
    theta1 = states_sequence[:, :, angle_index[0]] - np.pi
    theta2 = states_sequence[:, :, angle_index[1]]
    

    cost = (1 - torch.exp(-((torch.abs(theta1) - np.pi) / lengthscales[1]) ** 2 - ((x) / lengthscales[0]) ** 2 -((torch.abs(theta2)) / lengthscales[2]) ** 2))
    return cost


class Pendulum_cost(Expected_cost):
    """ Cost for the pendulum system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad]]).
    """

    def __init__(self, target_angle, lengthscale, angle_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: pendulum_cost(x, u, trial_index, target_angle=target_angle,
                                                         lengthscale=lengthscale,
                                                         angle_index=angle_index)
        # initit the superclass with the lambda function
        super(Pendulum_cost, self).__init__(f_cost)


def pendulum_cost(states_sequence, inputs_sequence, trial_index, target_angle, lengthscale, angle_index):
    """ 
    Cost function given by the saturated distance between |theta| and target_angle.
    """
    theta = states_sequence[:, :, angle_index]

    return (1 - torch.exp(-((torch.abs(theta) - target_angle) / lengthscale) ** 2))


class Pendulum_cost_mujoco(Expected_cost):
    """ Cost for the pendulum system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad]]).
    """

    def __init__(self, lengthscale, angle_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: pendulum_cost_mujoco(x, u, trial_index, target_angle=np.pi,
                                                         lengthscale=lengthscale,
                                                         angle_index=angle_index)
        # initit the superclass with the lambda function
        super(Pendulum_cost_mujoco, self).__init__(f_cost)


def pendulum_cost_mujoco(states_sequence, inputs_sequence, trial_index, target_angle, lengthscale, angle_index):
    """ 
    Cost function given by the saturated distance between |theta| and target_angle.
    """
    theta = states_sequence[:, :, angle_index] - np.pi  # shift the angle to be in the range [-pi, pi]

    return (1 - torch.exp(-((torch.abs(theta) - target_angle) / lengthscale) ** 2))


class Pendubot_cost(Expected_cost):
    """ Cost for the pendulum system:
        target is assumed in the unstable equilibrium configuration defined in 'target_state' (target angle [rad]]).
    """

    def __init__(self, targets, lengthscales, indices, flg_weight_vel=False):
        if flg_weight_vel:
            f_cost = lambda x, u, trial_index: pendubot_cost_vel(x, u, trial_index, targets=targets,
                                                                 lengthscales=lengthscales,
                                                                 indices=indices)
        else:
            # get the saturated distance function as a function of states and inputs
            f_cost = lambda x, u, trial_index: pendubot_cost(x, u, trial_index, targets=targets,
                                                             lengthscales=lengthscales,
                                                             indices=indices)
        # initit the superclass with the lambda function
        super(Pendubot_cost, self).__init__(f_cost)


class Pendubot_cost_ROA(Expected_cost):
    def __init__(self, targets, S, indices):
        f_cost = lambda x, u, trial_index: pendubot_cost_roa(x, u, trial_index, targets=targets, S=S, indices=indices)
        super(Pendubot_cost_ROA, self).__init__(f_cost)


def pendubot_cost_roa(states_sequence, inputs_sequence, trial_index, targets, S, indices):
    """
    Cost function given by the saturated distance between |theta| and target_angle, weighted by S.
    """
    x = states_sequence[:, :, indices]

    # for i in range(len(targets)):
    #     x[:, :, indices[i]] -= targets[i]
    #
    # norm = torch.einsum("NMi,ij,NMj", x, S, x)

    e_1 = targets[0] - states_sequence[:, :, indices[0]]
    e_2 = targets[1] - states_sequence[:, :, indices[1]]
    de_1 = targets[2] - states_sequence[:, :, indices[2]]
    de_2 = targets[3] - states_sequence[:, :, indices[3]]

    norm = (torch.abs(e_1) / 3.0) ** 2 + (torch.abs(e_1 - e_2) / 3.0) ** 2
    # (torch.abs(e_1) / 3.0) ** 2 + (torch.abs(e_2) / 3.0) ** 2 +

    # norm = S[0, 0] * e_1**2 + S[0, 1] * e_1 * e_2 + S[0, 2] * e_1 * de_1 + S[0, 3] * e_1 * de_2 + \
    #        S[1, 0] * e_1 * e_2 + S[1, 1] * e_2**2 + S[1, 2] * e_2 * de_1 + S[1, 3] * e_2 * de_2 + \
    #        S[2, 0] * e_1 * de_1 + S[2, 1] * e_2 * de_1 + S[2, 2] * de_1**2 + S[2, 3] * de_1 * de_2 + \
    #        S[3, 0] * e_1 * de_2 + S[3, 1] * e_2 * de_2 + S[3, 2] * de_1 * de_2 + S[3, 3] * de_2**2

    # print(norm.size())

    return 1 - torch.exp(-norm)


class Pendubot_cost_gready(Min_cost):
    """ Cost for the pendubot system:
        target is assumed in the unstable equilibrium configuration defined in 'target_state' (target angle [rad]]).
    """

    def __init__(self, targets, lengthscales, indices, flg_weight_vel=False):
        if flg_weight_vel:
            f_cost = lambda x, u, trial_index: pendubot_cost_vel(x, u, trial_index, targets=targets,
                                                                 lengthscales=lengthscales,
                                                                 indices=indices)
        else:
            # get the saturated distance function as a function of states and inputs
            f_cost = lambda x, u, trial_index: pendubot_cost(x, u, trial_index, targets=targets,
                                                             lengthscales=lengthscales,
                                                             indices=indices)
        # initit the superclass with the lambda function
        super(Pendubot_cost_gready, self).__init__(f_cost)


class Pendubot_cost_with_variance(Expected_cost_with_variance):
    """ Cost for the pendubot system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad]]).
    """

    def __init__(self, targets, lengthscales, indices, variance_lengthscale, flg_weight_vel=False):
        if flg_weight_vel:
            f_cost = lambda x, u, trial_index: pendubot_cost_vel(x, u, trial_index, targets=targets,
                                                                 lengthscales=lengthscales,
                                                                 indices=indices)
        else:
            # get the saturated distance function as a function of states and inputs
            f_cost = lambda x, u, trial_index: pendubot_cost(x, u, trial_index, targets=targets,
                                                             lengthscales=lengthscales,
                                                             indices=indices)
        # initit the superclass with the lambda function
        super(Pendubot_cost_with_variance, self).__init__(f_cost, variance_lengthscale)


def pendubot_cost(states_sequence, inputs_sequence, trial_index, targets, lengthscales, indices):
    """
    Cost function given by the saturated distance between |theta| and target_angle.
    """
    theta_1 = states_sequence[:, :, indices[0]]
    theta_2 = states_sequence[:, :, indices[1]]

    return 1 - torch.exp(-((torch.abs(theta_1) - targets[0]) / lengthscales[0]) ** 2
                         - ((torch.abs(theta_2) - targets[1]) / lengthscales[1]) ** 2)


def pendubot_cost_weight_input(states_sequence, inputs_sequence, trial_index, targets, lengthscales, indices):
    """
    Cost function given by the saturated distance between |theta| and target_angle + cost on the input.
    """
    theta_1 = states_sequence[:, :, indices[0]]
    theta_2 = states_sequence[:, :, indices[1]]

    u_1 = inputs_sequence[:, :, indices[0]]
    u_2 = inputs_sequence[:, :, indices[0]]

    return ((1 - torch.exp(-((torch.abs(theta_1) - targets[0]) / lengthscales[0]) ** 2
                           - ((torch.abs(theta_2) - targets[1]) / lengthscales[1]) ** 2)) +
            (1 - torch.exp(-((torch.abs(u_1) - targets[2]) / lengthscales[2]) ** 2
                           - ((torch.abs(u_2) - targets[3]) / lengthscales[3]) ** 2)))


def pendubot_cost_vel(states_sequence, inputs_sequence, trial_index, targets, lengthscales, indices):
    """
    Cost function given by the saturated distance between |theta| and target_angle.
    """
    theta_1 = states_sequence[:, :, indices[0]]
    theta_2 = states_sequence[:, :, indices[1]]
    omega_1 = states_sequence[:, :, indices[2]]
    omega_2 = states_sequence[:, :, indices[3]]

    return 1 - torch.exp(-((torch.abs(theta_1) - targets[0]) / lengthscales[0]) ** 2
                         - ((torch.abs(theta_2) - targets[1]) / lengthscales[1]) ** 2
                         - ((torch.abs(omega_1) - targets[2]) / lengthscales[2]) ** 2
                         - ((torch.abs(omega_2) - targets[3]) / lengthscales[3]) ** 2)


class Extended_state_cost(Expected_cost):
    """
        Cost for the systems defined by extended state: [state, goal]:
    """

    def __init__(self, target_indices, state_indices, lengthscales):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_target(x, u, trial_index, target_indices,
                                                                          state_indices,
                                                                          lengthscales)
        # initit the superclass with the lambda function
        super(Extended_state_cost, self).__init__(f_cost)


def saturated_distance_from_target_state(states_sequence, inputs_sequence, trial_index, target_indices, state_indices,
                                   lengthscales):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """

    # get state components evaluated in the cost
    active_states = states_sequence[:, :, state_indices]
    target_states = states_sequence[:, :, target_indices]

    # normalize states and targets
    norm_states = active_states / lengthscales
    norm_targets = target_states / lengthscales
    # get the square distance
    dist = torch.sum(((norm_states - norm_targets)) ** 2, dim=2)
    cost = 1 - torch.exp(-dist)

    return cost


class Acrobot_cost(Expected_cost):
    """ Cost for the Furuta pendulum system:
        target is assumed in the instable equilibrium configuration [0, pi, 0, 0]
    """

    def __init__(self, target_state, lengthscales, max_penalty,
                 low_bound, up_bound, slope_weight, indices):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: acrobot_cost(x, u,
                                                        trial_index, target_state=target_state,
                                                        lengthscales=lengthscales,
                                                        max_penalty=max_penalty, low_bound=low_bound, up_bound=up_bound,
                                                        slope_weight=slope_weight,
                                                        indices=indices)
        # initit the superclass with the lambda function
        super(Acrobot_cost, self).__init__(f_cost)


def acrobot_cost(states_sequence, inputs_sequence, trial_index,
                 target_state, lengthscales, max_penalty, low_bound, up_bound,
                 slope_weight, indices):
    """ Cost is the saturated distance from the target state plus a penalty if the velocities go outside limits.
    """

    e_1 = target_state[0] - states_sequence[:, :, indices[0]]
    e_2 = target_state[1] - states_sequence[:, :, indices[1]]
    omega_1 = states_sequence[:, :, indices[2]]
    omega_2 = states_sequence[:, :, indices[3]]

    # return

    norm = (torch.abs(e_1) / lengthscales[0]) ** 2 + (torch.abs(e_2) / lengthscales[1]) ** 2

    return (1 - torch.exp(-norm) +
            max_penalty * (1 / (1 + torch.exp(-slope_weight * (low_bound - omega_1))) +
                           1 / (1 + torch.exp(-slope_weight * (omega_1 - up_bound)))) +
            max_penalty * (1 / (1 + torch.exp(-slope_weight * (low_bound - omega_2))) +
                           1 / (1 + torch.exp(-slope_weight * (omega_2 - up_bound))))
            )

class Acrobot_last_pos_cost(Expected_cost):
    """ Cost for the Tossing Bot system:
        target is assumed to be included in the states variables.
            state = (curr_state_pos, curr_state_vel_, target_state_pos)
    """

    def __init__(self, target_state, lengthscales, pos_indeces, num_samples=10):
        # get the saturated distance function as a function of states and inputs
        # num_samples: Number of the last samples to weight
        f_cost = lambda x, u, trial_index: saturated_distance_from_target_state_acrobot(x, u, trial_index,
                                                                                target_state=target_state,
                                                                                lengthscales=lengthscales,
                                                                                active_dims=pos_indeces,
                                                                                num_samples=num_samples)
        # initit the superclass with the lambda function
        super(Acrobot_last_pos_cost, self).__init__(f_cost)


def saturated_distance_from_target_state_acrobot(states_sequence, inputs_sequence, trial_index, target_state, lengthscales,
                                   active_dims, num_samples):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    computed only for the last position
    """
    e_1 = target_state[0] - states_sequence[-num_samples:, :, active_dims[0]]
    e_2 = target_state[1] - states_sequence[-num_samples:, :, active_dims[1]]

    # return

    norm = (torch.abs(e_1) / lengthscales[0]) ** 2 + (torch.abs(e_2) / lengthscales[1]) ** 2

    return 1 - torch.exp(-norm)

class Acrobot_cost_pos_penalty(Expected_cost):
    """ Cost for the Furuta pendulum system:
        target is assumed in the instable equilibrium configuration [0, pi, 0, 0]
    """

    def __init__(self, target_state, lengthscales, max_penalty,
                 low_bound, up_bound, slope_weight, indices):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: acrobot_cost_pos_penalty(x, u,
                                                                    trial_index, target_state=target_state,
                                                                    lengthscales=lengthscales,
                                                                    max_penalty=max_penalty, low_bound=low_bound,
                                                                    up_bound=up_bound,
                                                                    slope_weight=slope_weight,
                                                                    indices=indices)
        # initit the superclass with the lambda function
        super(Acrobot_cost_pos_penalty, self).__init__(f_cost)


def acrobot_cost_pos_penalty(states_sequence, inputs_sequence, trial_index,
                             target_state, lengthscales, max_penalty, low_bound, up_bound,
                             slope_weight, indices):
    """ Cost is the saturated distance from the target state plus a penalty if the velocities go outside limits.
    """
    e_1 = (states_sequence[:, :, indices[0]]) % (2*np.pi) - np.pi
    e_2 = (states_sequence[:, :, indices[1]] + np.pi) % (2*np.pi) - np.pi
    # e_2 = states_sequence[:, :, indices[1]]


    # e_1 = torch.sin(states_sequence[:, :, indices[0]]) + torch.cos(target_state[0]) - torch.cos(states_sequence[:, :, indices[0]])
    # e_2 = torch.sin(states_sequence[:, :, indices[1]]) + torch.cos(target_state[1]) - torch.cos(states_sequence[:, :, indices[1]])
    
    omega_1 = states_sequence[:, :, indices[2]]
    omega_2 = states_sequence[:, :, indices[3]]

    # return

    norm = (torch.abs(e_1) / lengthscales[0]) ** 2 + (torch.abs(e_2) / lengthscales[1]) ** 2


    return (1 - torch.exp(-norm) +
            max_penalty * (torch.relu(slope_weight * (low_bound[0] - states_sequence[:, :, indices[0]])) + torch.relu(slope_weight * (states_sequence[:, :, indices[0]] - up_bound[0]))) +
            max_penalty * (torch.relu(slope_weight * (low_bound[1] - states_sequence[:, :, indices[1]])) + torch.relu(slope_weight * (states_sequence[:, :, indices[0]] - up_bound[0]))) +
            max_penalty * (torch.relu(slope_weight * (low_bound[2] - omega_1)) + torch.relu(slope_weight * (omega_1 - up_bound[2]))) +
            max_penalty * (torch.relu(slope_weight * (low_bound[3] - omega_2)) + torch.relu(slope_weight * (omega_2 - up_bound[3]))))


    # return (1 - torch.exp(-norm) +
    #         max_penalty * (1 / (1 + torch.exp(-slope_weight * (low_bound[0] - states_sequence[:, :, indices[0]]))) +
    #                        1 / (1 + torch.exp(-slope_weight * (states_sequence[:, :, indices[0]] - up_bound[0])))) +
    #         max_penalty * (1 / (1 + torch.exp(-slope_weight * (low_bound[1] - states_sequence[:, :, indices[1]]))) +
    #                        1 / (1 + torch.exp(-slope_weight * (states_sequence[:, :, indices[1]] - up_bound[1])))) +
    #         max_penalty * (1 / (1 + torch.exp(-slope_weight * (low_bound[2] - omega_1))) +
    #                        1 / (1 + torch.exp(-slope_weight * (omega_1 - up_bound[2])))) +
    #         max_penalty * (1 / (1 + torch.exp(-slope_weight * (low_bound[3] - omega_2))) +
    #                        1 / (1 + torch.exp(-slope_weight * (omega_2 - up_bound[3]))))
    #         )


class Tossing_bot_cost(Expected_cost):
    """ Cost for the Tossing Bot system:
        target is assumed to be included in the states variables.
            state = (curr_state_pos, curr_state_vel_, target_state_pos)
    """

    def __init__(self, target_indeces, lengthscales, pos_indeces, dtype, device):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_state_target(x, u, trial_index,
                                                                                target_indeces=target_indeces,
                                                                                lengthscales=lengthscales,
                                                                                pos_indeces=pos_indeces)
        # initit the superclass with the lambda function
        super(Tossing_bot_cost, self).__init__(f_cost)
        self.dtype = dtype
        self.device = device


def saturated_distance_from_state_target(states_sequence, inputs_sequence, trial_index, target_indeces, lengthscales,
                                         pos_indeces):
    states_pos_sequence = states_sequence[-1:, :, pos_indeces[:2]]
    states_targets_sequence = states_sequence[-1:, :, target_indeces[:2]]

    dist = torch.sum(((states_pos_sequence - states_targets_sequence) / lengthscales) ** 2, dim=2)

    return 1 - torch.exp(-dist)


class Tossing_bot_cost_cumulattive(Expected_cost):
    """ Cost for the Tossing Bot system:
        target is assumed to be included in the states variables.
            state = (curr_state_pos, curr_state_vel_, target_state_pos)
    """

    def __init__(self, target_indeces, lengthscales, pos_indeces, dtype, device):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_cumulative_distance_from_state_target(x, u, trial_index,
                                                                                           target_indeces=target_indeces,
                                                                                           lengthscales=lengthscales,
                                                                                           pos_indeces=pos_indeces)
        # initit the superclass with the lambda function
        super(Tossing_bot_cost_cumulattive, self).__init__(f_cost)
        self.dtype = dtype
        self.device = device


def saturated_cumulative_distance_from_state_target(states_sequence, inputs_sequence, trial_index, target_indeces,
                                                    lengthscales, pos_indeces):
    states_pos_sequence = states_sequence[:, :, pos_indeces[:2]]
    states_targets_sequence = states_sequence[:, :, target_indeces[:2]]

    dist = torch.sum(((states_pos_sequence - states_targets_sequence) / lengthscales[:2]) ** 2, dim=2)

    return 1 - torch.exp(-dist)


class Hopper_Cost(Expected_cost):
    """ Cost for the cart pole system:
        target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad], target position [m]).
    """

    def __init__(self, lengthscales, vel_indices, pos_indices):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: hopper_cost(x, u, trial_index, lengthscales=lengthscales,
                                                       vel_indices=vel_indices,
                                                       pos_indices=pos_indices)
        # initit the superclass with the lambda function
        super(Hopper_Cost, self).__init__(f_cost)


def hopper_cost(states_sequence, inputs_sequence, trial_index, lengthscales, vel_indices, pos_indices):
    """
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x_dot = states_sequence[:, :, vel_indices[1]]
    z = states_sequence[:, :, pos_indices[0]]

    return 1 / (1 + torch.exp(-(0.7 - z) / lengthscales[0]))


class Quadrotor_cost(Expected_cost):
    """
    Cost function for the quadrotor system ->
    Given by the distance between the current position (x,y,z) and the target position
    """

    def __init__(self, target_positon, lengthscales, pos_index):
        f_cost = lambda x, u, trial_index: quadrotor_cost(x, u, trial_index, target_state=target_positon,
                                                          lengthscales=lengthscales, pos_index=pos_index)

        super(Quadrotor_cost, self).__init__(f_cost)


def quadrotor_cost(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, pos_index):
    x = states_sequence[:, :, pos_index[0]]
    y = states_sequence[:, :, pos_index[1]]
    z = states_sequence[:, :, pos_index[2]]

    x_t = target_state[0]
    y_t = target_state[1]
    z_t = target_state[2]

    return (1 - torch.exp(
        -((x - x_t) / lengthscales[0]) ** 2 - ((y - y_t) / lengthscales[1]) ** 2 - ((z - z_t) / lengthscales[2]) ** 2))
