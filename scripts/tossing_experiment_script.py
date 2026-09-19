#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import rospkg
import sys
import os
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Point, Quaternion
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from nav_msgs.msg import Path, Odometry

import tf
from tf.transformations import quaternion_matrix, quaternion_from_euler

import matplotlib.pyplot

from mcpilot.msg import ModelStateStamp, ModelTrajectory

import math

try:
    from math import pi, dist, fabs, cos, sqrt, atan2, sin, cos, tan
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt, sin, tan

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
from gazebo_msgs.srv import SpawnModel, DeleteModel  # For deleting models from the environment
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from mcpilot.srv import TossExperiment


from robot_tossing_utils import catapult_toss, below_toss
from robot_tossing_utils.panda_utils import *

TOSS_TYPE = copy.copy(TOSS_TYPE_below)


bullet_delta_grasping = [0.135, 0.18]

# release_position = {'below': [0.7551644570703442, 0.0, 1.107557285230802],
#                     'catapult_back': [0.0]*3}

#pickup_position = {'below': [0.4, 0.4, 1.0], 'catapult_back': [-0.5, 0.0, 1.0]}

load_position = [-3.0, 0.0, 0.5]

# Use the projectile selected by the launch file.  The package ships
# ``red_ball_friction``; keeping the legacy ``red_ball`` name here made
# calibration wait forever for a non-existent ``/red_ball_odom`` topic.
bullet_name = rospy.get_param('bullet_name', 'red_ball_friction')
bullets_names = [bullet_name]

target_item_name = 'target'
bullet_odom_format = '{}_odom'
bullet_odom = bullet_odom_format.format(bullet_name)

rospack = rospkg.RosPack()
target_model_file_hole = rospack.get_path('mcpilot') + '/' + rospy.get_param("target_model_file_hole_path")
target_model_test = 'target_test'
target_model_file_full = rospack.get_path('mcpilot') + '/' + rospy.get_param("target_model_file_full_path")
target_model_train = 'target_train'
targets = [target_model_test, target_model_train]

release_freq = 50.0  # Hz (frequency of the release node)

arm_charging_config = None
arm_toss_config = None

release_position = None

def launch_joint_trj(move_group, v_norm, yaw_target, pre_toss_joint_position, dt=0.001):
    """

    :param v_norm:  Norm of initial velocity to reach desired target
    :param yaw_target:   yaw Angle of the target
    :param pre_toss_joint_position: joint position for pre-launch phase
    :param dt:      time step for the trajectory
    :param use_jacobian:    If true uses Jacobian to compute joint velocity for tossing (Currently not working correctly),
                            if False computes the joint velocities approximating toss to a circular trajectory
                            exploiting the geometry of the Panda Robot
    :param vel_trj_type:    smooth: generates trajectory with func: 'smooth_trj'
    
    :return:    The JointTrajectory object to feed to the Controller, the trajectory time, the joint positions for 
    pre-launch phase, joint positions for post-launch phase (namely the configuration reached with the JointTrajectory) 
    """

    if TOSS_TYPE == 'below':
        times, positions, velocities, accelerations, pre_toss_joint_position, post_toss_joint_position, rising_time = (
            below_toss.toss_from_below(move_group=move_group, v_norm=v_norm,
                                       yaw_target=yaw_target, dt=dt))
    elif TOSS_TYPE == 'catapult_back':
        times, positions, velocities, accelerations, pre_toss_joint_position, post_toss_joint_position, rising_time = (
            catapult_toss.toss_back_catapult(move_group=move_group, v_norm=v_norm, yaw_target=yaw_target))
    else:
        raise NotImplementedError('No ' + str(TOSS_TYPE) + 'implemented (yet)')



    # rising_time is the nominal release instant: the generated trajectory
    # reaches arm_toss_config there.  times[-1] also includes the subsequent
    # damping segment.
    return (follow_timed_joint_trajectory(positions, velocities, times, joint_names),
            times[-1], rising_time, pre_toss_joint_position, post_toss_joint_position)




def spawn_model(item_name, file, position, orientation):
    spawn_model_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    with open(file, "r") as f:
        product_xml = f.read()

    item_pose = Pose(Point(position[0], position[1], position[2]), orientation)
    spawn_model_srv(item_name, product_xml, "", item_pose, "world")

    return


def perform_tossing(target_dist, yaw_target, z_target, v_norm, alpha, pre_toss_joint_position, release_pos, orientation,
                    bullet_model_name, target_model):
    global bullet_name, bullets_names, bullet_odom_format
    bullet_name = bullet_model_name  # Update chosen bullet
    rospy.set_param('bullet_name', bullet_name)
    
    anticipation_seconds = float(
        rospy.get_param(
            "/tossing/release_delay/anticipation_seconds",
            0.0
        )
    )

    rospy.loginfo(
        "Release command anticipation: %.1f ms",
        anticipation_seconds * 1000.0
    )

    x_target = target_dist * cos(yaw_target)
    y_target = target_dist * sin(yaw_target)
    print('Setting target: ({}, {})'.format(x_target, y_target))
    print('Orientation: {}'.format(orientation))

    moveit_commander.roscpp_initialize(sys.argv)
    #Move all bullet models in load position
    for model in bullets_names + targets:
        # In case it was forgotten attached
        release(model, '{}::link'.format(model))
        # Set bullet pose
        state_msg = ModelState()
        state_msg.model_name = model

        if model == bullet_model_name:
            if TOSS_TYPE == TOSS_TYPE_below:
                pickup_position = below_toss.pickup_position
            elif TOSS_TYPE == TOSS_TYPE_catapult_back:
                pickup_position = catapult_toss.pickup_position
            else:
                raise NotImplementedError(TOSS_TYPE + ' not implemented')

            state_msg.pose.position.x = pickup_position[0]
            state_msg.pose.position.y = pickup_position[1]
            state_msg.pose.position.z = pickup_position[2]

            orientation_quat = quaternion_from_euler(orientation[0], orientation[1], orientation[2], axes='rzxz')
            print('####Quaternion######: {}'.format(orientation_quat))
            state_msg.pose.orientation.x = orientation_quat[0]
            state_msg.pose.orientation.y = orientation_quat[1]
            state_msg.pose.orientation.z = orientation_quat[2]
            state_msg.pose.orientation.w = orientation_quat[3]
        elif model == target_model:
            state_msg.pose.position.x = x_target
            state_msg.pose.position.y = y_target
            state_msg.pose.position.z = z_target
            state_msg.pose.orientation = load_orient
        else:
            state_msg.pose.position.x = load_position[0]
            state_msg.pose.position.y = load_position[1]
            state_msg.pose.position.z = load_position[2]
            state_msg.pose.orientation = load_orient


        # Set ball state in pick pose
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))
        #rospy.sleep(1.0)

    rospy.sleep(1.5)

    bullet_odom_msg = rospy.wait_for_message(bullet_odom_format.format(bullet_name), Odometry, timeout=5)
    bullet_pose = bullet_odom_msg.pose
    bullet_pos = [bullet_pose.pose.position.x, bullet_pose.pose.position.y, bullet_pose.pose.position.z]

    print('Grasping {} at {}'.format(bullet_name, bullet_pos))

    robot = moveit_commander.RobotCommander()

    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    print("============ Printing robot state")
    print(robot.get_current_state())

    joints_start = move_group.get_current_joint_values()

    open_gripper(moveit_commander)

    move_group.set_max_velocity_scaling_factor(0.8)

    pose_goal = geometry_msgs.msg.Pose()

    if TOSS_TYPE == TOSS_TYPE_catapult_back:
        quat = quaternion_from_euler(pi, 0, -pi / 4 + pi)
        pose_goal.position.x = bullet_pos[0] + bullet_delta_grasping[0]
        pose_goal.position.y = bullet_pos[1]
        pose_goal.position.z = bullet_pos[2] + bullet_delta_grasping[1] + bullet_grasp_dist
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]

        move_group.set_pose_target(pose_goal)

        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        grasp(bullet_name, '{}::link'.format(bullet_name))

    elif TOSS_TYPE == TOSS_TYPE_below:
        quat = quaternion_from_euler(pi, 0, -pi / 4)
        pose_goal.position.x = bullet_pos[0]
        pose_goal.position.y = bullet_pos[1]
        pose_goal.position.z = bullet_pos[2] + bullet_grasp_dist
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]
        approach_pose = copy.deepcopy(pose_goal)
        approach_pose.position.z += 0.2

        move_group.set_pose_target(approach_pose)

        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()

        p_ee = move_group.get_current_pose()
        print(p_ee)
        D_z = p_ee.pose.position.z - bullet_grasp_dist - bullet_pos[2]
        print(D_z)

        descend(move_group, D_z)

        grasp(bullet_name, '{}::link'.format(bullet_name))
        # close_gripper(moveit_commander)

        depart(move_group, D_z)


    target = [x_target, y_target, z_target]

    print("\nMoving to pre-tossing pose... ")
    # move_group.go(copy.deepcopy(joints_start), wait=True)
    # move_group.stop()
    # move_group.go(copy.deepcopy(joints_start), wait=True)
    # move_group.stop()
    if v_norm == 0.0:
        if TOSS_TYPE == TOSS_TYPE_below:
            q = below_toss.arm_toss_config
        elif TOSS_TYPE == TOSS_TYPE_catapult_back:
            q = catapult_toss.arm_toss_config
        else:
            raise NotImplementedError
        q[0] = yaw_target

        release_ball_at(move_group, q, bullet_name=bullet_name)
    else:
        plan, trj_time, nominal_release_time, pre_toss, post_toss = launch_joint_trj(
            move_group, v_norm, yaw_target, pre_toss_joint_position)

        print("Trajectory time: {}".format(trj_time))
        print("Nominal release time: {}".format(nominal_release_time))
        print("Pre-tossing arm config: {}".format(pre_toss))

        move_group.go(copy.deepcopy(pre_toss), wait=True)
        move_group.stop()
        move_group.set_max_velocity_scaling_factor(1.0)
        move_group.set_max_acceleration_scaling_factor(1.0)

        # print(plan)
        res = send_goal_to_controller(
            plan.joint_trajectory,
            post_toss,
            bullet_name=bullet_name,
            nominal_release_time=nominal_release_time,
            release_anticipation=anticipation_seconds,
        )

    ## Reading Ball trajectory
    print('Waiting trajectory...')
    try:
        trj_ball = rospy.wait_for_message('bullet_trajectory', ModelTrajectory, timeout=30.0)
    except rospy.exceptions.ROSException as e:
        print('ROSException captured: ' + str(e))
        return None

    print('Toss complete!')

    trj = []
    if trj_ball is not None:
        if len(trj_ball.path.poses) == 0:
            rospy.logerr(
                'Projectile tracker aborted the throw before a valid target-plane crossing.'
            )
            return None

        for i in range(len(trj_ball.path.poses)):
            pose_stamp = trj_ball.path.poses[i]
            pose = pose_stamp.pose
            time = pose_stamp.header.stamp
            lin_vel = trj_ball.twists[i].linear
            ang_vel = trj_ball.twists[i].angular
            stamp_pos = [time.secs + time.nsecs * (10 ** (-9)), pose.position.x, pose.position.y, pose.position.z]
            # [x,y,z] + [dot_x, dot_y, dot_z] + [wx, wy, wz]
            trj.append(stamp_pos + [lin_vel.x, lin_vel.y, lin_vel.z, ang_vel.x, ang_vel.y, ang_vel.z])

        print('Distance from target: {}'.format(math.dist(target, trj[-1][1:4])))

        # bullet_trajectory already starts at the actual physical detachment.
        # Do not trim it to the nominal release configuration: after command
        # anticipation, actual detachment can occur before that configuration.
        trj = np.array(trj)

        print('Trajectory size: {} samples'.format(trj.shape[0]))
        return trj
    else:
        return None


def toss_experiment(target, flg_exploration, policy_command, orientation, bullet_model_name, model_file):
    print('Target : {}'.format(target))
    rospy.set_param('/tossing/target_altitude', float(target[2]))

    if TOSS_TYPE == TOSS_TYPE_below:
        alpha = below_toss.ALPHA # np.pi / 4  # const
    elif TOSS_TYPE == TOSS_TYPE_catapult_back:
        alpha = catapult_toss.ALPHA

    # YAW = 0 for experiments
    v_norm, pre_toss_joint_position, release_p, yaw_target = compute_ballistics(arm_charging_config=arm_charging_config,
                                                                                release_position=release_position,
                                                                                p_target=target,
                                                                                alpha=alpha)
    target_dist = planar_distance([0, 0, target[2]], target)
    z_target = target[2]

    if flg_exploration:
        v_norm += policy_command
    else:
        # Load policy for control of tossing
        # v_norm = max(policy_command, 0.1)  # with this small epsilon the ball falls to the ground
        v_norm = policy_command

    ball_path = perform_tossing(target_dist, yaw_target, z_target, max(v_norm, 0), alpha, pre_toss_joint_position,
                                release_p, orientation, bullet_model_name, model_file)

    if ball_path is not None:
        # Build states vector
        # states = [release_p]

        inputs = [[v_norm * cos(alpha) * cos(yaw_target), v_norm * cos(alpha) * sin(yaw_target), v_norm * sin(alpha)]] + \
                 [[0, 0, 0]] * (ball_path.shape[0] - 1)

        states_data = []
        inputs_data = []
        for i in range(ball_path.shape[1]):
            states_data += [ball_path[j, i] for j in range(ball_path.shape[0])]
        for i in range(3):
            inputs_data += [inputs[j][i] for j in range(len(inputs))]

        # print(states_data)
        # print(inputs_data)

        return states_data, inputs_data

    else:
        return None, None


def calib():
    global release_position, bullet_odom_format, arm_charging_config, arm_toss_config
    print('Running Calib -- Do Not send requests!!!!! \n\n\n --------------------------------------------------')
    open_gripper(moveit_commander)
    # Calibration
    if TOSS_TYPE == TOSS_TYPE_below:
        arm_charging_config = below_toss.arm_charging_config
        arm_toss_config = below_toss.arm_toss_config
        pickup_position = below_toss.pickup_position
    elif TOSS_TYPE == TOSS_TYPE_catapult_back:
        arm_charging_config = catapult_toss.arm_charging_config
        arm_toss_config = catapult_toss.arm_toss_config
        pickup_position = catapult_toss.pickup_position

    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    #
    # arm_toss_config = [0.0, - 45 * pi / 180,  0.0,  -pi/4, 0.0, 180 * pi / 180, pi/4]
    #
    # print('Post toss: {}'.format(arm_toss_config))
    #
    # group_name = "panda_arm"
    # # group_name = "panda"
    #
    #
    #
    # print('move_group.get_current_rpy:', move_group.get_current_rpy())
    #
    #
    # # Jacobian computations
    # J = move_group.get_jacobian_matrix(arm_toss_config, [0.0, 0.0, 0.0])
    # print('Jacobian:', J)
    # J_ = np.eye(3)
    # J_[0, 0] = J[0, 1]
    # J_[0, 1] = J[0, 3]
    # J_[0, 2] = J[0, 5]
    #
    # J_[1, 0] = J[1, 1]
    # J_[1, 1] = J[1, 3]
    # J_[1, 2] = J[1, 5]
    #
    # J_[2, 0] = J[2, 1]
    # J_[2, 1] = J[2, 3]
    # J_[2, 2] = J[2, 5]
    #
    # print('J 3x3:', J_)
    #
    # alpha = 0 * pi / 180
    # v_ref = np.array([-1 * cos(alpha), 0.0, 1 * sin(alpha)])
    # # v_ref = np.array([1. , 0.0, 0.0])
    #
    # J__inv = np.linalg.inv(J_)
    #
    # print('J 3x3 inv:', J__inv)
    #
    # print('J 3x3 inv * v_ref:', np.matmul(J__inv, v_ref))
    #
    # print('-------------')
    #
    # J_22 = np.eye(2)
    # J_22[0, 0] = J[0, 1]
    # J_22[0, 1] = J[0, 3]
    # J_22[1, 0] = J[2, 1]
    # J_22[1, 1] = J[2, 3]
    #
    # print('J 2x2:', J_22)
    #
    # J_22_inv = np.linalg.inv(J_22)
    #
    # # alpha = 0
    # print('J 3x3 inv:', J_22_inv)
    # print('J 2x2 inv * v_ref:', np.matmul(J_22_inv, np.array([-2*cos(alpha), 2*sin(alpha)])))
    #
    # exit()

    move_group.set_joint_value_target(arm_toss_config)
    move_group.go(wait=True)
    move_group.stop()
    rate = rospy.Rate(1)
    for i in range(2):
        rate.sleep()
    print(move_group.get_current_pose())
    launch_pose = move_group.get_current_pose()
    pos = launch_pose.pose.position
    hand_pos = np.array([pos.x, pos.y, pos.z])



    # Set bullet pose
    state_msg = ModelState()
    state_msg.model_name = bullet_name
    state_msg.pose.position.x = pickup_position[0]
    state_msg.pose.position.y = pickup_position[1]
    state_msg.pose.position.z = pickup_position[2]
    state_msg.pose.orientation = load_orient

    # Set ball state in pick pose
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))

    rospy.sleep(1.5)

    bullet_odom_msg = rospy.wait_for_message(bullet_odom_format.format(bullet_name), Odometry)
    bullet_pose = bullet_odom_msg.pose
    bullet_pos = [bullet_pose.pose.position.x, bullet_pose.pose.position.y, bullet_pose.pose.position.z]

    pose_goal = geometry_msgs.msg.Pose()

    if TOSS_TYPE == TOSS_TYPE_catapult_back:
        quat = quaternion_from_euler(pi, 0, -pi/4 + pi)
        pose_goal.position.x = bullet_pos[0] + bullet_delta_grasping[0]
        pose_goal.position.y = bullet_pos[1]
        pose_goal.position.z = bullet_pos[2] + bullet_delta_grasping[1] + bullet_grasp_dist
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]

        move_group.set_pose_target(pose_goal)

        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        grasp(bullet_name, '{}::link'.format(bullet_name))


    elif TOSS_TYPE == TOSS_TYPE_below:
        quat = quaternion_from_euler(pi, 0, -pi / 4)
        pose_goal.position.x = bullet_pos[0]
        pose_goal.position.y = bullet_pos[1]
        pose_goal.position.z = bullet_pos[2] + bullet_grasp_dist
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]
        approach_pose = copy.deepcopy(pose_goal)
        approach_pose.position.z += 0.2

        move_group.set_pose_target(approach_pose)

        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()

        p_ee = move_group.get_current_pose()
        print(p_ee)
        D_z = p_ee.pose.position.z - bullet_grasp_dist - bullet_pos[2]
        print(D_z)

        descend(move_group, D_z)

        grasp(bullet_name, '{}::link'.format(bullet_name))
        # close_gripper(moveit_commander)

        depart(move_group, D_z)

    move_group.set_joint_value_target(arm_toss_config)
    move_group.go(wait=True)
    move_group.stop()
    rate = rospy.Rate(1)
    for i in range(2):
        rate.sleep()
    print(move_group.get_current_pose())

    bullet_odom_msg = rospy.wait_for_message(bullet_odom_format.format(bullet_name), Odometry)
    bullet_pose = bullet_odom_msg.pose
    bullet_pos = [bullet_pose.pose.position.x, bullet_pose.pose.position.y, bullet_pose.pose.position.z]

    release_position = [bullet_pos[0], 0.0, bullet_pos[2]]

    print('Actual ball pos at release pose: {}'.format(bullet_pos))

    release(bullet_name, '{}::link'.format(bullet_name))
    rospy.sleep(1.5)


def load_bullets():
    global bullets_names
    bullets_dir = rospack.get_path('mcpilot') + '/' + rospy.get_param('bullet_models_files_path')
    models = os.listdir(bullets_dir)

    bullets_names = []

    for model in models:
        bullets_names.append(model)
        spawn_model(model, rospack.get_path('mcpilot') + '/' + rospy.get_param(
            'bullet_models_files_path') + model + '/model.sdf', load_position, load_orient)
        rospy.loginfo('Loading {} model'.format(model))
    return

def load_targets():
    spawn_model(target_model_train, target_model_file_full, load_position, load_orient)
    spawn_model(target_model_test, target_model_file_hole, load_position, load_orient)

    return

if __name__ == '__main__':
    rospy.init_node("tossing_experiment_node")
    rospy.set_param("/tossing_experiment_ready", False)
    rospy.on_shutdown(lambda: rospy.set_param("/tossing_experiment_ready", False))

    try:
        TOSS_TYPE = rospy.get_param('toss_type')
    except:
        TOSS_TYPE = TOSS_TYPE_below
        print('Setting default launch type: below')

    rate = rospy.Rate(1)
    for i in range(2):
        rate.sleep()

    # Load all bullets
    load_bullets()

    # Load all targets
    load_targets()

    calib()
    rospy.set_param('/tossing_experiment_ready', True)
    rospy.loginfo('Tossing experiment services are ready')
    """
        Calibration is needed for better precision
        If user chooses to run calib, make sure that the callback that is responsible to detach the ball is off, run it later
        
        If the calibration process loses the ball in between the computed ballistics will be wrong!
    """

    s_learn = rospy.Service('tossing_experiment_learning', TossExperiment,
                            lambda r: toss_experiment(r.target, r.flg_exploration, r.policy_command, r.orientation,
                                                      r.bullet_model_name, target_model_train))
    s_test = rospy.Service('tossing_experiment_testing', TossExperiment,
                           lambda r: toss_experiment(r.target, r.flg_exploration, r.policy_command, r.orientation,
                                                     r.bullet_model_name, target_model_test))

    rospy.spin()
