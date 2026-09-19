#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
    Script for implementing a proxy to a ROS service for the tossing experiment, run as python script, not with rosrun

"""
import math

import numpy.random
import rospy
import std_msgs
import torch
import json

import policy_learning.Policy_Tossing as Policy
from mcpilot.srv import TossExperiment


import tossing_utils
from tossing_utils import gen_batch_of_targets, V_MAX

experiment_srv_name = 'tossing_experiment_learning'
test_srv_name = 'tossing_experiment_testing'

eps = 0.1  # m/s


def rl_proxy(message):
    """
        In this experiment the Policy only forwards command for the first state
    """
    print('Received: {}'.format(str(message.data)))
    data = json.loads(str(message.data))
    if len(data) == 6:
        # Backwards compatibility with older MC-PILCO clients.
        initial_state, T_exploration, trial_index, flg_exploration, policy_path, bullet_name = data
        prescribed_exploration_velocity = None
    else:
        initial_state, T_exploration, trial_index, flg_exploration, policy_path, bullet_name, prescribed_exploration_velocity = data

    if flg_exploration:
        if prescribed_exploration_velocity is None:
            policy_cmd = numpy.random.normal(0.0, 0.1)
            service_exploration = True
            print('Random toss: delta v={}'.format(policy_cmd))
        else:
            # The tossing service adds exploration commands to its analytical
            # ballistic velocity.  Send this prescribed value as a control
            # command instead so it remains an absolute velocity.
            policy_cmd = float(prescribed_exploration_velocity)
            service_exploration = False
            print('Stratified exploration toss: absolute v={}'.format(policy_cmd))
    else:
        service_exploration = False
        print('Computing toss velocity with policy')
        # policy_model = Policy.TossingPolicy(state_dim=3, target_indeces=[6, 7, 8], alpha=numpy.pi / 4,
        #                                     pos_indeces=[0, 1, 2],
        #                                     u_max=V_MAX, num_basis=500, flg_squash=True)

        #policy_model = Policy.TossingPolicySimple(state_dim=3, target_indeces=[6, 7, 8], alpha=numpy.pi / 4,
        #                                          pos_indeces=[0, 1, 2], u_max=V_MAX, num_basis=500, flg_squash=True)

        policy_model = Policy.TossingPolicyTarget(state_dim=3, target_indeces=[6, 7, 8], alpha=tossing_utils.alpha,
                                          pos_indeces=[0, 1, 2], u_max=V_MAX, num_basis=tossing_utils.num_basis,
                                          flg_squash=True)

        policy_model.load_state_dict(torch.load(policy_path))

        policy_cmd, gamma = policy_model.get_v(torch.tensor([initial_state]), t=0, p_dropout=0.0)
        print(policy_cmd)

        policy_cmd = max(policy_cmd.detach().cpu().numpy()[0][0], eps)
        gamma = gamma.detach().cpu().numpy()[0]

        print('Tossing with: v={}, gamma={}'.format(policy_cmd, gamma))

    rospy.wait_for_service(experiment_srv_name)
    srv = rospy.ServiceProxy(experiment_srv_name, TossExperiment)

    cnt_try = 5
    response = None
    while cnt_try > 0:
        try:
            response = srv(initial_state[6:], service_exploration, policy_cmd, bullet_name, [0] * 3)
            break
        except:
            cnt_try -= 1

    if cnt_try < 0:
        data_msg = std_msgs.msg.Float64MultiArray()
        RL_proxy_publisher.publish(data_msg)
    elif not (response is None):
        states = list(response.trajectory)
        inputs = list(response.inputs)
        print('Num samples: {}'.format(len(states)/7))

        num_samples = int(len(states) / 10)  # times + cartesian positions + linear velocities + angular velocities
        end_position = [states[2*num_samples - 1], states[3*num_samples - 1],
                        states[4*num_samples - 1]]
        target = initial_state[6:]
        print(target)
        dist_from_target = math.dist(target, end_position)
        print('Distance from target: {}'.format(dist_from_target))

        # if not flg_exploration:
        #     rospy.wait_for_service(test_srv_name)
        #     test_srv = rospy.ServiceProxy(test_srv_name, TossExperiment)
        #
        #     global experiments_results
        #     experiments_results[trial_index] = []
        #
        #     batch_states = gen_batch_of_targets([1.0, 1.55], [0.1]*2, [-numpy.pi/6, numpy.pi/6], 1)
        #     print(batch_states)
        #
        #     for i in range(batch_states.shape[0]):
        #         rospy.wait_for_service(test_srv_name)
        #         initial_state = list(batch_states[i,:])
        #         target = initial_state[6:]
        #         policy_cmd, _ = policy_model.get_v(torch.tensor([initial_state]), t=0, p_dropout=0.0)
        #         print('Toss: {}'.format(policy_cmd))
        #         policy_cmd = max(policy_cmd.detach().cpu().numpy()[0][0], eps)
        #         test_srv(target, False, policy_cmd)
        #
        #         rospy.sleep(2.0)
        #         ball_odom = rospy.wait_for_message('red_ball_odom', Odometry)
        #         ball_pose = ball_odom.pose
        #         print(ball_pose)
        #         ball_pos = [ball_pose.pose.position.x, ball_pose.pose.position.y, 0.0]
        #
        #         target[2] = 0.0
        #         dist_from_target = math.dist(ball_pos, target)
        #
        #         print('({}) Distance from target: {}'.format(i+1, dist_from_target))
        #         experiments_results[trial_index].append(dist_from_target)
        # else:
        #     experiments_results[trial_index] = dist_from_target

        data_msg = std_msgs.msg.Float64MultiArray()
        data_msg.data = states + inputs

        RL_proxy_publisher.publish(data_msg)
    else:
        raise Exception('Something broke')

rospy.init_node('MC_PILCO_experiment_proxy')

RL_proxy_subscriber = rospy.Subscriber('tossing_exp_proxy_in', std_msgs.msg.String, rl_proxy)
RL_proxy_publisher = rospy.Publisher('tossing_exp_proxy_out', std_msgs.msg.Float64MultiArray, queue_size=10)
rospy.spin()
