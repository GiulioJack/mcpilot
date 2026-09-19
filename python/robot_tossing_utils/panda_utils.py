# SPDX-License-Identifier: AGPL-3.0-or-later
from math import sqrt, atan2, sin, cos, tan, atan2, pi, floor, copysign
import copy

import numpy as np
from matplotlib import pyplot as plt
from std_msgs.msg import String, Empty
from moveit_commander.conversions import pose_to_list
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
import rospy
from moveit_msgs.msg import RobotTrajectory, Grasp, PlaceLocation, Constraints
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Point, Pose, Quaternion
import tf
import actionlib

group_name_gripper = ["panda_hand", "hand"]
rob_model_name = 'panda'
rob_link_name = 'panda_link7'
wrist_len = sqrt((0.384 + 0.107 + 0.23) ** 2 + (2 * 0.088) ** 2)
alpha_0 = atan2(0.384 + 0.107, 0.088)
upper_arm = 0.316
forearm = 0.333
max_q = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
min_q = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
max_dq = [2.1750] * 4 + [2.6100] * 3
max_ddq = [15, 7.5, 10, 12.5, 15, 20, 20]

bullet_grasp_dist = 0.12  # distance (meters) from the ball reference frame (geometric center) to the hand reference frame of the robot
identity_orient = tf.transformations.quaternion_from_euler(0, 0, 0)
load_orient = Quaternion()
load_orient.x = identity_orient[0]
load_orient.y = identity_orient[1]
load_orient.z = identity_orient[2]
load_orient.w = identity_orient[3]

joint_names = []
for i in range(7):
    joint_names.append('panda_joint{}'.format(i + 1))
gripper_fingers_names = ['panda_finger_joint1', 'panda_finger_joint2']

TOSS_TYPE_below = 'below'
TOSS_TYPE_catapult_back = 'catapult_back'

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
    return - (a * t ** 3) / (tf ** 3) + (a * t ** 2) / (tf ** 2) + (a * t) / tf


def signal_acc(t, target_vel, delta_q):
    """
        Acceleration of Smooth velocity signal
    :param t: time
    :param a: required velocity at settling time
    :param delta_q: required signal value at settling time
    """
    a = target_vel
    tf = settling_time(delta_q, a)
    return - (3 * a * t ** 2) / (tf ** 3) + (2 * a * t) / (tf ** 2) + (a) / tf


def settling_time(delta_q, target_vel):
    """
    :param delta_q: required value for signal at settling time
    :param target_vel: required value for signal velocity at settling time
    :return:
    """
    if abs(target_vel) > 0:
        return delta_q * 12 / (target_vel * 7)
    else:
        return 0.0


def required_delta_q(settling_time, required_vel):
    """
    :param settling_time: fixed settling time to reach required vel at null acceleration
    :param required_vel: required value for signal velocity at settling time
    :return:
    """
    return 7 * settling_time * required_vel / 12


def planar_distance(p1, p2):
    """
    :param p1:  point
    :param p2:  another point
    :return:    distance between the projections on the horizontal plane of the two points
    """
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def altitude_diff(p1, p2):
    """
    :param p1:  point
    :param p2:  another point
    :return:    altitude difference of the two points (positive if second point is higher, negative otherwise)
    """
    return p2[2] - p1[2]


def compute_tossing_v_norm(dx, dz, alpha=pi / 4, g=9.81):
    """
    Computes release velocity norm given distances and inclination (target is assumed aligned with launch trajectory)
    :param dx:      distance from release position in the horizontal plane
    :param dz:      difference of altitude between target and launch (target_z - launch_z)
    :param alpha:   slope of launch
    :param g:       gravity acceleration

    :return:        Release velocity computed with simple ballistic equations
    """
    return sqrt(0.5 * g * (dx ** 2) / ((cos(alpha) ** 2) * (tan(alpha) * dx - dz)))



def vertical_movement(move_group_arm, dz, step):
    """

    :param move_group_arm:  Arm moveit controller
    :param dz:              Vertical displacement w.r.t. world frame
    :param step:            interpolation step
    :return:    True if movement was performed successfully, False otherwise
    """
    waypoints = []
    wpose = copy.deepcopy(move_group_arm.get_current_pose().pose)

    # print(range(int(math.floor(abs(dz / step)))))

    for i in range(int(floor(abs(dz / step)))):  # meters to cm
        wpose.position.z += step
        waypoints.append(copy.deepcopy(wpose))

    if abs(dz / step) - floor(abs(dz / step)) > 0:
        wpose.position.z += copysign(1, step) * (dz - abs(step) * floor(abs(dz / step)))
        waypoints.append(copy.deepcopy(wpose))

    (plan, fraction) = move_group_arm.compute_cartesian_path(waypoints,  # waypoints to follow
                                                             0.01,  # eef_step
                                                             0.0)  # jump_threshold
    if plan is None:
        return False

    if not move_group_arm.execute(plan, wait=True):
        return False
    move_group_arm.stop()
    move_group_arm.clear_pose_targets()

    return True


def descend(move_group_arm, dz):
    """
    :param move_group_arm:  Arm moveit controller
    :param dz:              vertical displacement
    :return:    True if vertical descent was performed successfully, False otherwise
    """
    return vertical_movement(move_group_arm, dz, -0.02)


def depart(move_group_arm, dz):
    """
    :param move_group_arm:  Arm moveit controller
    :param dz:              vertical displacement
    :return:    True if vertical departure was performed successfully, False otherwise
    """
    return vertical_movement(move_group_arm, dz, 0.02)


def grasp_object(robot, obj_name, scene, grasping_group="hand", eef_link='panda_link7'):
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, obj_name, touch_links=touch_links)


def look_for_gripper(moveit_commander):
    i = 0
    for i in range(len(group_name_gripper)):
        try:
            move_group_gripper = moveit_commander.MoveGroupCommander(group_name_gripper[i])
            return move_group_gripper
        except:
            pass
        i += 1
    raise ValueError('No gripper move group found')

def open_gripper(moveit_commander, wait=True):
    move_group_gripper = look_for_gripper(moveit_commander=moveit_commander)
    open_joint = [0.04, 0.04]
    move_group_gripper.go(open_joint, wait=wait)
    move_group_gripper.stop()


def close_gripper(moveit_commander, wait=True):
    move_group_gripper = look_for_gripper(moveit_commander=moveit_commander)
    close_joint = [0.0, 0.0]
    move_group_gripper.go(close_joint, wait=wait)
    move_group_gripper.stop()


def grasp(obj_name, obj_link_name):
    """
    :param obj_name:        Object name in gazebo
    :param obj_link_name:   Object's link name in gazebo
    :return:    True if successfully attached, false otherwise
    """
    rospy.loginfo('Grasping {}'.format(obj_name))
    attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
    attach_srv.wait_for_service()
    req = AttachRequest()
    req.model_name_1 = rob_model_name
    req.link_name_1 = rob_link_name
    req.model_name_2 = obj_name
    req.link_name_2 = obj_link_name

    ret = attach_srv.call(req)

    return ret.ok


def release(obj_name, obj_link_name):
    """
    :param obj_name:        Object name in gazebo
    :param obj_link_name:   Object's link name in gazebo
    :return:    True if successfully detached, false otherwise
    """
    detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
    detach_srv.wait_for_service()
    req = AttachRequest()
    req.model_name_1 = rob_model_name
    req.link_name_1 = rob_link_name
    req.model_name_2 = obj_name
    req.link_name_2 = obj_link_name

    ret = detach_srv.call(req)

    return ret.ok


def plan_open_gripper(delta_time, dt=0.01):
    """
    :param delta_time:
    :return: planned trajectoru to open gripper after delta_time seconds
    """
    num_steps = int(delta_time / dt)
    times = np.linspace(start=0, stop=(num_steps - 1) * dt, num=num_steps)
    positions = []

    for t in times:
        if t < delta_time:
            positions.append([0.0, 0.0])
        else:
            positions.append([0.04, 0.04])

    return follow_timed_joint_trajectory_no_vel(positions, times, gripper_fingers_names)


def compute_ballistics(arm_charging_config, release_position, p_target, alpha=pi / 4):
    """

    :param p_target:    Position of the target (trajectory will be a parabola)
    :param alpha:       Angle of the initial velocity w.r.t. horizontal plane
    :return:        Norm of initial velocity to reach desired target, joint position for pre-launch phase, cartesian
                    position of the bullet (ball) at release time
    """
    yaw = atan2(p_target[1], p_target[0])
    print('Yaw aligment: {}'.format(yaw))

    pre_toss_joint_position = copy.deepcopy(arm_charging_config)
    pre_toss_joint_position[0] = yaw

    release_p = [release_position[0] * cos(yaw), release_position[0] * sin(yaw), release_position[2]]
    print('release pos: {}'.format(release_p))

    dx = planar_distance(release_p, p_target)
    dz = altitude_diff(release_p, p_target)
    print('dx: {}, dy: {}'.format(dx, dz))

    v_norm = compute_tossing_v_norm(dx, dz, alpha)
    print('Velocity norm: {}'.format(v_norm))

    return v_norm, pre_toss_joint_position, release_p, yaw


def step_trj(pre_toss_joint_position, post_toss, dq_4, dq_6, dt):
    """
        Generates a tossing trajectory moving joints 4 and 6 only.
        Velocity is a step, i.e. generated velocity is constant during the trajectory
        (would require infinite instantaneous acceleration, works in practice)

    :param pre_toss_joint_position:     joint position for pre-launch phase
    :param post_toss:                   joint positions for post-launch phase
                                        (namely the configuration reached with the JointTrajectory)
    :param dq_4, dq_6:    Velocity to impose to joints

    :return:    positions, velocities, times, pre_toss_joint_position;
        pre_toss_joint_position is corrected
    """
    trj_time = (post_toss[3] - pre_toss_joint_position[3]) / dq_4  # trajectory time defined by joint 4
    print('Trajectory time: {}'.format(trj_time))

    num_steps = int(trj_time / dt)
    times = np.linspace(start=0, stop=(num_steps - 1) * dt, num=num_steps)

    max_delta6 = post_toss[5] - pre_toss_joint_position[5]
    actual_delta6 = min(dq_6 * trj_time, max_delta6)
    pre_toss_joint_position[5] = post_toss[5] - actual_delta6
    print('Setting trj with dq{}={}, delta_q4={}'.format(4, dq_4, post_toss[3] - pre_toss_joint_position[3]))
    print('Setting trj with dq{}={}, delta_q4={}'.format(6, dq_6, post_toss[5] - pre_toss_joint_position[5]))

    positions = []
    velocities = []
    for t in times:
        q = copy.deepcopy(pre_toss_joint_position)
        q[3] += dq_4 * t
        q[5] += dq_6 * t

        positions.append(q)
        velocities.append([0, 0, 0, dq_4, 0, dq_6, 0])

    return positions, velocities, times, pre_toss_joint_position, trj_time


def smooth_trj(pre_toss_joint_position, post_toss, dq_2, dq_4, dq_6, dt, final_damping=False):
    """
        Generates a tossing trajectory moving joints 4 and 6 only.
        Generates a "smooth" velocity for both joints such that they reach the desired position with desired constant
        velocity.
        i.e. velocity is described by: -a t^3 + b t^2 +c t

        with a=(alpha/tf^3), b=(alpha/tf^2), c=(alpha/tf)

        In this way
            -velocity at time tf is = alpha
            -acceleration at time tf is null
            -integral of velocity from o to tf is equal to the desired change in joint position

        Requires instantaneous positive (finite) acceleration, works in practice

    :param pre_toss_joint_position:
    :param post_toss:
    :param dq_4:
    :param dq_6:
    :return:
    """

    trj_time_q2 = settling_time(post_toss[1] - pre_toss_joint_position[1], dq_2)  # trajectory time defined by joint 2
    trj_time_q4 = settling_time(post_toss[3] - pre_toss_joint_position[3], dq_4)  # trajectory time defined by joint 4
    trj_time_q6 = settling_time(post_toss[5] - pre_toss_joint_position[5], dq_6)  # trajectory time defined by joint 6

    trj_time = max([trj_time_q2, trj_time_q4, trj_time_q6])  # all joints should evolve
    delta_time_q2 = max(0, (trj_time - trj_time_q2))
    delta_time_q4 = max(0, (trj_time - trj_time_q4))
    delta_time_q6 = max(0, (trj_time - trj_time_q6))

    print('Trajectory time: {}'.format(trj_time))

    num_steps = int(trj_time / dt)
    times = np.linspace(start=0, stop=(num_steps - 1) * dt, num=num_steps)

    """
    max_delta_q_6 = post_toss[5] - pre_toss_joint_position[5]  # max deltas available
    max_delta_q_2 = post_toss[1] - pre_toss_joint_position[1]
    actual_delta_q_6 = min(required_delta_q(trj_time, dq_6), max_delta_q_6)
    actual_delta_q_2 = min(required_delta_q(trj_time, dq_2), max_delta_q_2)
    pre_toss_joint_position[5] = post_toss[5] - actual_delta_q_6
    pre_toss_joint_position[1] = post_toss[1] - actual_delta_q_2
    """

    print('Setting trj with dq2={}, delta_q2={}'.format(dq_2, post_toss[1] - pre_toss_joint_position[1]))
    print('Setting trj with dq4={}, delta_q4={}'.format(dq_4, post_toss[3] - pre_toss_joint_position[3]))
    print('Setting trj with dq6={}, delta_q6={}'.format(dq_6, post_toss[5] - pre_toss_joint_position[5]))

    positions = []
    velocities = []
    accelerations = []

    q_last = copy.deepcopy(pre_toss_joint_position)
    dq_last = [0] * 7
    ddq_last = [0] * 7

    for t in times:
        q = copy.deepcopy(q_last)
        dq = [0] * 7
        ddq = [0] * 7

        if t > delta_time_q2:
            q[1] = pre_toss_joint_position[1] + signal(t - delta_time_q2, dq_2, post_toss[1] - pre_toss_joint_position[1])
            dq[1] = signal_vel(t - delta_time_q2, dq_2, post_toss[1] - pre_toss_joint_position[1])
            ddq[1] = signal_acc(t - delta_time_q2, dq_2, post_toss[1] - pre_toss_joint_position[1])

        if t > delta_time_q4:
            q[3] = pre_toss_joint_position[3] + signal(t - delta_time_q4, dq_4, post_toss[3] - pre_toss_joint_position[3])
            dq[3] = signal_vel(t - delta_time_q4, dq_4, post_toss[3] - pre_toss_joint_position[3])
            ddq[3] = signal_acc(t - delta_time_q4, dq_4, post_toss[3] - pre_toss_joint_position[3])

        if t > delta_time_q6:
            q[5] = pre_toss_joint_position[5] + signal(t - delta_time_q6, dq_6, post_toss[5] - pre_toss_joint_position[5])
            dq[5] = signal_vel(t -delta_time_q6, dq_6, post_toss[5] - pre_toss_joint_position[5])
            ddq[5] = signal_acc(t -delta_time_q6, dq_6, post_toss[5] - pre_toss_joint_position[5])

        positions.append(q)
        velocities.append(dq)
        accelerations.append(ddq)
        q_last = q
        dq_last = dq
        ddq_last = ddq

    # num_steps = int(2*(1/release_freq) / dt)
    # times_cnst = np.linspace(start=0, stop=(num_steps - 1) * dt, num=num_steps)
    #
    # for t in times_cnst:
    #     positions.append([positions[-1][k] + dq_last[k]*dt for k in range(7)])
    #     velocities.append(dq_last)
    #     accelerations.append([0]*7)

    if final_damping:
        damp = 2
        for t in reversed(times[::damp]):
            q = copy.deepcopy(positions[-1])
            dq = [0] * 7
            ddq = [0] * 7

            if t > delta_time_q2:
                dq[1] = signal_vel(t - delta_time_q2, dq_2, (post_toss[1] - pre_toss_joint_position[1]) / 1)
                ddq[1] = -damp * signal_acc(t - delta_time_q2, dq_2, (post_toss[1] - pre_toss_joint_position[1]) / 1)

            if t > delta_time_q4:
                dq[3] = signal_vel(t - delta_time_q4, dq_4, (post_toss[3] - pre_toss_joint_position[3]) / 1)
                ddq[3] = -damp * signal_acc(t - delta_time_q4, dq_4, (post_toss[3] - pre_toss_joint_position[3]) / 1)

            if t > delta_time_q6:
                dq[5] = signal_vel(t - delta_time_q6, dq_6, (post_toss[5] - pre_toss_joint_position[5]) / 1)
                ddq[5] = -damp * signal_acc(t - delta_time_q6, dq_6, (post_toss[5] - pre_toss_joint_position[5]) / 1)

            q = [q[i] + dq[i] * dt for i in range(7)]

            positions.append(q)
            velocities.append(dq)
            accelerations.append(ddq)
    ######### DO NOT REMOVE!!!!!!!!
    times = np.linspace(start=0, stop=(len(positions) - 1) * dt, num=len(positions))
    #########

    ######### Lines below can be removed
    # plotpos = np.array(positions)
    # plotvel = np.array(velocities)
    # plotacc = np.array(accelerations)
    #
    # acc_int = np.zeros_like(plotvel)
    # vel_int = np.zeros_like(plotvel)
    #
    # for t in range(plotacc.shape[0]):
    #     acc_int[t, :] = np.sum(plotacc[:t + 1, :], axis=0) * dt
    #     vel_int[t, :] = plotpos[0, :] + np.sum(plotvel[:t + 1, :], axis=0) * dt
    #
    # tf = trj_time
    #
    # fig, axs = plt.subplots(3, 3)
    # axs[0, 0].plot(times, [post_toss[3]] * plotpos.shape[0], label='rel q_{}'.format(4), color='r',
    #                linestyle='dashed')
    # axs[0, 0].plot(times, plotpos[:, 3], label='q_{}'.format(4))
    # axs[0, 0].axvline(x=tf, color='black')
    # axs[0, 0].plot(times, [max_q[3]] * plotpos.shape[0], label='max q_{}'.format(4))
    # axs[0, 0].plot(times, [min_q[3]] * plotpos.shape[0], label='min q_{}'.format(4))
    # axs[0, 0].legend(loc='lower left')
    #
    # axs[1, 0].plot(times, [post_toss[5]] * plotpos.shape[0], label='rel q_{}'.format(6), color='r',
    #                linestyle='dashed')
    # axs[1, 0].plot(times, plotpos[:, 5], label='q_{}'.format(6))
    # axs[1, 0].axvline(x=tf, color='black')
    # axs[1, 0].plot(times, [max_q[5]] * plotpos.shape[0], label='max q_{}'.format(6))
    # axs[1, 0].plot(times, [min_q[5]] * plotpos.shape[0], label='min q_{}'.format(6))
    # # axs[i].plot(vel_int[:, i], label='est q_{}'.format(i + 1))
    # axs[1, 0].legend(loc='lower left')
    #
    # axs[2, 0].plot(times, [post_toss[1]] * plotpos.shape[0], label='rel q_{}'.format(2), color='r',
    #                linestyle='dashed')
    # axs[2, 0].plot(times, plotpos[:, 1], label='q_{}'.format(6))
    # axs[2, 0].axvline(x=tf, color='black')
    # axs[2, 0].plot(times, [max_q[1]] * plotpos.shape[0], label='max q_{}'.format(2))
    # axs[2, 0].plot(times, [min_q[1]] * plotpos.shape[0], label='min q_{}'.format(2))
    # # axs[i].plot(vel_int[:, i], label='est q_{}'.format(i + 1))
    # axs[2, 0].legend(loc='lower left')
    #
    # # axs[0, 1].plot(times, [dq_4] * plotvel.shape[0], label='rel dq_{}'.format(4), color='r', linestyle='dashed')
    # axs[0, 1].plot(times, plotvel[:, 3], label='dq_{}'.format(4))
    # axs[0, 1].axvline(x=tf, color='black')
    # axs[0, 1].plot(times, [max_dq[3]] * plotpos.shape[0], label='max dq_{}'.format(4))
    # axs[0, 1].plot(times, [-max_dq[3]] * plotpos.shape[0], label='min dq_{}'.format(4))
    # axs[0, 1].legend(loc='lower left')
    #
    # # axs[1, 1].plot(times, [dq_6] * plotvel.shape[0], label='rel dq_{}'.format(6), color='r', linestyle='dashed')
    # axs[1, 1].plot(times, plotvel[:, 5], label='dq_{}'.format(6))
    # axs[1, 1].axvline(x=tf, color='black')
    # axs[1, 1].plot(times, [max_dq[5]] * plotpos.shape[0], label='max dq_{}'.format(6))
    # axs[1, 1].plot(times, [-max_dq[5]] * plotpos.shape[0], label='min dq_{}'.format(6))
    # # axs[i].plot(vel_int[:, i], label='est q_{}'.format(i + 1))
    # axs[1, 1].legend(loc='lower left')
    #
    # # axs[2, 1].plot(times, [dq_2] * plotvel.shape[0], label='rel dq_{}'.format(2), color='r', linestyle='dashed')
    # axs[2, 1].plot(times, plotvel[:, 1], label='dq_{}'.format(2))
    # axs[2, 1].axvline(x=tf, color='black')
    # axs[2, 1].plot(times, [max_dq[1]] * plotpos.shape[0], label='max dq_{}'.format(2))
    # axs[2, 1].plot(times, [-max_dq[1]] * plotpos.shape[0], label='min dq_{}'.format(2))
    # # axs[i].plot(vel_int[:, i], label='est q_{}'.format(i + 1))
    # axs[2, 1].legend(loc='lower left')
    #
    # axs[0, 2].plot(times, plotacc[:, 3], label='ddq_{}'.format(4))
    # axs[0, 2].axvline(x=tf, color='black')
    # axs[0, 2].plot(times, [max_ddq[3]] * plotpos.shape[0], label='max ddq_{}'.format(4))
    # axs[0, 2].plot(times, [-max_ddq[3]] * plotpos.shape[0], label='min ddq_{}'.format(4))
    # axs[0, 2].legend(loc='lower left')
    # axs[1, 2].plot(times, plotacc[:, 5], label='ddq_{}'.format(6))
    # axs[1, 2].axvline(x=tf, color='black')
    # axs[1, 2].plot(times, [max_ddq[5]] * plotpos.shape[0], label='max ddq_{}'.format(6))
    # axs[1, 2].plot(times, [-max_ddq[5]] * plotpos.shape[0], label='min ddq_{}'.format(6))
    # # axs[i].plot(vel_int[:, i], label='est q_{}'.format(i + 1))
    # axs[1, 2].legend(loc='lower left')
    # axs[2, 2].plot(times, plotacc[:, 1], label='ddq_{}'.format(2))
    # axs[2, 2].axvline(x=tf, color='black')
    # axs[2, 2].plot(times, [max_ddq[1]] * plotpos.shape[0], label='max ddq_{}'.format(2))
    # axs[2, 2].plot(times, [-max_ddq[1]] * plotpos.shape[0], label='min ddq_{}'.format(2))
    # # axs[i].plot(vel_int[:, i], label='est q_{}'.format(i + 1))
    # axs[2, 2].legend(loc='lower left')
    #
    # fig.subplots_adjust(left=0.05, bottom=0, right=1, top=1, wspace=0.3, hspace=0.3)
    # plt.savefig('plot.pdf')
    # np.savetxt('ref_toss_v_1.0.txt', np.concatenate([plotpos, plotvel, plotacc], 1))

    # plt.show()

    # for i in range(7):
    #     axs[i].plot(plotvel[:, i], label='dq_{}'.format(i + 1))
    #     #axs[i].plot(acc_int[:, i], label='est dq_{}'.format(i + 1))
    #     plt.legend()
    #
    # fig, axs = plt.subplots(7, 1)
    # for i in range(7):
    #     axs[i].plot(plotacc[:, i], label='ddq_{}'.format(i + 1))
    #     plt.legend()

    # for i in range(50):
    #     positions.append([q[j] + dq[j]*i*0.001 for j in range(7)])
    #     velocities.append(dq)
    #     accelerations.append([0]*7)
    #
    # p_end0 = positions[-1]
    #
    # for i in range(200):
    #     velocities.append([dq[j] - (dq[j]/0.2)*i*0.001 for j in range(7)])
    #     positions.append([p_end0[j] + dq[j]*i*0.001 - dq[j]*(i*0.001)**2/0.4 for j in range(7)])
    #     accelerations.append([- (dq[j]/0.2) for j in range(7)])

    return positions, velocities, accelerations, times, pre_toss_joint_position, trj_time


def follow_timed_joint_trajectory_no_vel(positions, times, j_names):
    """
    :param positions:   List of joint positions
    :param times:       List of progressive time
    :param j_names:     joints names
    :return:    JointTrajectory with given positions and time
    """
    return follow_timed_joint_trajectory(positions, [[0] * 7] * len(positions), times, j_names)


def follow_timed_joint_trajectory(positions, velocities, times, j_names):
    """
    :param positions:   List of joint positions
    :param velocities:  List of joint velocities
    :param times:       List of progressive time
    :param j_names:     joints
    :return:    JointTrajectory with given positions and velocities at defined times
    """
    jt = JointTrajectory()
    jt.joint_names = j_names
    # jt.header.stamp = rospy.Time.now() + rospy.Duration(time_offset)
    # don't set this time, only time from start in each point

    for (position, velocity, time) in zip(positions, velocities, times):
        jtp = JointTrajectoryPoint()
        jtp.positions = position
        jtp.velocities = velocity
        jtp.time_from_start = rospy.Duration(time)
        jt.points.append(jtp)

    rt = RobotTrajectory()
    rt.joint_trajectory = jt
    rt.multi_dof_joint_trajectory = MultiDOFJointTrajectory()

    return rt


def send_goal_to_controller(arm_trajectory, post_toss, bullet_name,
                            nominal_release_time=None, release_anticipation=0.0):
    """
    Feed the arm trajectory to the effort joint trajectory controller and issue
    an *opening command* before the nominal release instant.

    The physical actuator/detachment delay is not implemented here.  The
    /tossing/release_delay/open_command topic is consumed by
    read_bullet_state.py, which samples the configured physical delay and
    detaches the projectile only after that delay has elapsed.

    :param arm_trajectory: JointTrajectory fed to the controller.
    :param post_toss: Kept for backward compatibility with existing callers.
    :param bullet_name: Active projectile model.
    :param nominal_release_time: Nominal release time [s] from trajectory
                                 start.  For the smooth tossing trajectory this
                                 is the end of the rising phase, before the
                                 damping segment.
    :param release_anticipation: Time [s] by which the opening command is
                                 advanced relative to nominal_release_time.
    """
    del post_toss  # release is now time-triggered, not configuration-triggered.

    if len(arm_trajectory.points) == 0:
        raise ValueError("Cannot execute an empty tossing trajectory")

    if nominal_release_time is None:
        nominal_release_time = arm_trajectory.points[-1].time_from_start.to_sec()

    nominal_release_time = float(nominal_release_time)
    release_anticipation = max(0.0, float(release_anticipation))
    command_time = max(0.0, nominal_release_time - release_anticipation)

    rospy.loginfo(
        "Nominal release at %.3f s; opening command at %.3f s "
        "(anticipation %.1f ms)",
        nominal_release_time,
        command_time,
        1000.0 * release_anticipation,
    )

    arm_client = actionlib.SimpleActionClient(
        '/effort_joint_trajectory_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction,
    )
    arm_client.wait_for_server()

    open_command_pub = rospy.Publisher(
        '/tossing/release_delay/open_command',
        Empty,
        queue_size=1,
    )
    # The tracker is already running, but allow the publisher/subscriber
    # connection to settle before trajectory feedback starts arriving.
    rospy.sleep(0.05)

    arm_goal = FollowJointTrajectoryGoal()
    arm_goal.trajectory = arm_trajectory
    arm_goal.goal_time_tolerance = rospy.Duration(0)

    command_sent = {'value': False}
    last_feedback = {'msg': None}
    previous_feedback = {'msg': None, 'sim_time': None}
    nominal_logged = {'value': False}
    synthetic_postcheck_logged = {'value': False}

    def _vec(values):
        return [float(x) for x in values]

    def send_open_command(source):
        if command_sent['value']:
            return
        publish_sim_time = rospy.Time.now().to_sec()
        fb = last_feedback['msg']

        # Publish all timing metadata *before* the Empty notification.  The
        # synthetic-ground-truth tracker consumes these parameters immediately
        # on receipt and uses them to schedule nominal_release + sampled_delay.
        rospy.set_param('/tossing/release_delay/last_open_publish_sim_time', publish_sim_time)
        if fb is not None:
            rospy.set_param('/tossing/release_delay/last_open_feedback_desired_elapsed', float(fb.desired.time_from_start.to_sec()))
            rospy.set_param('/tossing/release_delay/last_open_feedback_actual_elapsed', float(fb.actual.time_from_start.to_sec()))
            rospy.set_param('/tossing/release_delay/last_open_feedback_desired_positions', _vec(fb.desired.positions))
            rospy.set_param('/tossing/release_delay/last_open_feedback_desired_velocities', _vec(fb.desired.velocities))
            rospy.set_param('/tossing/release_delay/last_open_feedback_actual_positions', _vec(fb.actual.positions))
            rospy.set_param('/tossing/release_delay/last_open_feedback_actual_velocities', _vec(fb.actual.velocities))
        rospy.set_param(
            '/tossing/release_delay/last_command_anticipation_seconds',
            release_anticipation,
        )
        rospy.set_param(
            '/tossing/release_delay/last_nominal_release_time_seconds',
            nominal_release_time,
        )

        open_command_pub.publish(Empty())
        command_sent['value'] = True
        rospy.loginfo(
            "Opening command issued from %s at trajectory time %.3f s",
            source,
            command_time,
        )

    def tossing_trj_callback(feedback):
        sim_now = rospy.Time.now().to_sec()
        prev_msg = previous_feedback['msg']
        prev_sim = previous_feedback['sim_time']
        last_feedback['msg'] = feedback
        desired_elapsed = feedback.desired.time_from_start.to_sec()
        if (not nominal_logged['value']) and desired_elapsed >= nominal_release_time:
            nominal_logged['value'] = True
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_sim_time', sim_now)
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_desired_elapsed', float(desired_elapsed))
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_actual_elapsed', float(feedback.actual.time_from_start.to_sec()))
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_desired_positions', _vec(feedback.desired.positions))
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_desired_velocities', _vec(feedback.desired.velocities))
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_actual_positions', _vec(feedback.actual.positions))
            rospy.set_param('/tossing/release_delay/last_nominal_feedback_actual_velocities', _vec(feedback.actual.velocities))

            # Convert the nominal release instant into ROS/Gazebo simulation
            # time using the two controller-feedback samples that bracket it.
            # This avoids subtracting a controller trajectory time from a
            # Gazebo simulation-time timestamp.
            nominal_sim = sim_now
            interp_ok = False
            if prev_msg is not None and prev_sim is not None:
                prev_desired = prev_msg.desired.time_from_start.to_sec()
                denom = desired_elapsed - prev_desired
                if prev_desired <= nominal_release_time <= desired_elapsed and denom > 1e-12:
                    alpha = (nominal_release_time - prev_desired) / denom
                    nominal_sim = float(prev_sim) + alpha * (sim_now - float(prev_sim))
                    interp_ok = True
                    rospy.set_param('/tossing/release_delay/last_nominal_bracket_prev_desired_elapsed', float(prev_desired))
                    rospy.set_param('/tossing/release_delay/last_nominal_bracket_prev_sim_time', float(prev_sim))
                    rospy.set_param('/tossing/release_delay/last_nominal_bracket_next_desired_elapsed', float(desired_elapsed))
                    rospy.set_param('/tossing/release_delay/last_nominal_bracket_next_sim_time', float(sim_now))
                    rospy.set_param('/tossing/release_delay/last_nominal_interpolation_alpha', float(alpha))
            rospy.set_param('/tossing/release_delay/last_nominal_release_sim_time_interpolated', float(nominal_sim))
            rospy.set_param('/tossing/release_delay/last_nominal_release_sim_time_interpolation_valid', bool(interp_ok))
            rospy.loginfo(
                'TIMING PROBE nominal feedback: desired=%.6f s actual=%.6f s sim=%.6f; nominal_sim=%.6f (%s)',
                desired_elapsed,
                feedback.actual.time_from_start.to_sec(),
                sim_now,
                nominal_sim,
                'interpolated' if interp_ok else 'fallback',
            )

        # In synthetic-ground-truth mode a 20--30 ms release can occur before
        # the first post-nominal feedback sample is available.  Once both the
        # interpolated nominal timestamp and the actual detach timestamp exist,
        # perform an independent check of the realized residual.
        if (not synthetic_postcheck_logged['value'] and
                rospy.get_param('/tossing/release_delay/synthetic_ground_truth/enabled', True) and
                rospy.has_param('/tossing/release_delay/last_nominal_release_sim_time_interpolated') and
                rospy.has_param('/tossing/release_delay/last_actual_release_sim_time')):
            synthetic_postcheck_logged['value'] = True
            nominal_sim_check = float(rospy.get_param(
                '/tossing/release_delay/last_nominal_release_sim_time_interpolated'
            ))
            detach_sim_check = float(rospy.get_param(
                '/tossing/release_delay/last_actual_release_sim_time'
            ))
            sampled_ms_check = float(rospy.get_param(
                '/tossing/release_delay/synthetic_ground_truth/last_sample_ms'
            ))
            measured_ms_check = 1000.0 * (detach_sim_check - nominal_sim_check)
            error_ms_check = measured_ms_check - sampled_ms_check
            rospy.set_param(
                '/tossing/release_delay/synthetic_ground_truth/last_interpolated_measured_residual_ms',
                measured_ms_check,
            )
            rospy.set_param(
                '/tossing/release_delay/synthetic_ground_truth/last_interpolated_realization_error_ms',
                error_ms_check,
            )
            rospy.loginfo(
                'SYNTHETIC GT POSTCHECK: sampled=%.3f ms | independently interpolated effective=%.3f ms | error=%+.3f ms',
                sampled_ms_check,
                measured_ms_check,
                error_ms_check,
            )

        previous_feedback['msg'] = feedback
        previous_feedback['sim_time'] = sim_now

        # The desired trajectory point is expressed in time from trajectory
        # start, so this trigger stays aligned with simulated/controller time.
        elapsed = desired_elapsed
        if elapsed >= command_time:
            send_open_command('trajectory feedback')

    def tossing_trj_done(state, result):
        print('State: {}'.format(state))
        print('Result: {}'.format(result))
        # Fallback only: never detach directly here, otherwise the configured
        # physical delay could be bypassed.
        if not command_sent['value']:
            rospy.logwarn(
                "Trajectory finished before the opening command was observed; "
                "issuing it now."
            )
            send_open_command('done callback fallback')

    rospy.loginfo('Moving the arm to goal position...')
    arm_client.send_goal(
        arm_goal,
        feedback_cb=tossing_trj_callback,
        done_cb=tossing_trj_done,
    )

    return  # asynchronous, as in the original implementation


def release_ball_at(move_group, pose, bullet_name):
    move_group.set_joint_value_target(pose)
    move_group.go(wait=True)
    move_group.stop()
    rospy.sleep(rospy.Duration(1))
    release(bullet_name, '{}::link'.format(bullet_name))
