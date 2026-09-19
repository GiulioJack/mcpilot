#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Track the active projectile, apply a configurable release delay, and publish its trajectory."""
import math
import os
import copy
import threading
from collections import deque

import geometry_msgs.msg
import numpy as np
import rospkg
import rospy
import csv
import os
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState

from gazebo_ros_link_attacher.srv import (
    Attach, AttachRequest, ScheduledDetach, ScheduledDetachRequest
)
from mcpilot.msg import ModelTrajectory
from robot_tossing_utils import below_toss, catapult_toss
from robot_tossing_utils.panda_utils import TOSS_TYPE_below, TOSS_TYPE_catapult_back
from robot_tossing_utils.release_delay import DelayConfig, ReconfigurableDelaySampler

TOSS_TYPE = TOSS_TYPE_below
bullet = "red_ball_friction"
model_trj = ModelTrajectory()
bullet_pub = None
release_position = [7.55177987e-01, -2.56263722e-04, 1.1]

rob_model_name = "panda"
rob_link_name = "panda_link7"
epsilon = 0.0
meas_std = 0.0

delay_sampler = ReconfigurableDelaySampler()
release_timer = None
release_command_active = False
release_command_bullet = None
projectile_released = False
release_actual_sim_time = None
delay_reset_id_seen = None
trajectory_watchdog = None
latest_joint_state = None
synthetic_rng = None
synthetic_rng_seed = None
synthetic_reset_id_seen = None

# The scheduled-detach service deliberately blocks until Gazebo reaches the
# requested simulation timestamp.  Odometry callbacks continue to run while
# that service response is in flight.  Retain those samples so service-response
# latency cannot move the cached trajectory's time origin past the detach.
pending_release_samples = deque()
pending_release_deadline = None
pending_release_capture_bullet = None
pending_release_capture_lock = threading.RLock()


def _joint_state_callback(msg):
    global latest_joint_state
    latest_joint_state = msg


def _delay_config_from_ros():
    prefix = "/tossing/release_delay"
    values = {
        "enabled": rospy.get_param(prefix + "/enabled", True),
        "distribution": rospy.get_param(prefix + "/distribution", "uniform"),
        "min_steps": rospy.get_param(prefix + "/min_steps", 120),
        "max_steps": rospy.get_param(prefix + "/max_steps", 130),
        "fixed_steps": rospy.get_param(prefix + "/fixed_steps", 125),
        "mean_steps": rospy.get_param(prefix + "/mean_steps", 125.0),
        "std_steps": rospy.get_param(prefix + "/std_steps", 2.5),
        "seed": rospy.get_param(prefix + "/seed", 0),
    }
    return DelayConfig.from_mapping(values)


def _sample_delay():
    global delay_reset_id_seen

    reset_id = int(rospy.get_param("/tossing/release_delay/reset_id", 0))
    if delay_reset_id_seen != reset_id:
        delay_sampler.reset()
        delay_reset_id_seen = reset_id

    sampled_ms = delay_sampler.sample(_delay_config_from_ros())
    rospy.set_param("/tossing/release_delay/last_sample_steps", int(sampled_ms))
    rospy.set_param("/tossing/release_delay/last_sample_ms", int(sampled_ms))
    rospy.set_param("/tossing/release_delay/last_sample_seconds", float(sampled_ms) / 1000.0)
    rospy.loginfo("Sampled physical release delay: %d ms", sampled_ms)
    return sampled_ms



def _synthetic_ground_truth_enabled():
    return bool(rospy.get_param(
        "/tossing/release_delay/synthetic_ground_truth/enabled", True
    ))


def _sample_synthetic_residual_ms():
    """Sample the known delay value used by the synthetic test.

    This sampler is deliberately independent from the physical actuator-delay
    sampler.  In synthetic-ground-truth mode the actuator model is bypassed:
    by default the value is an effective residual. When
    ``compensate_command_anticipation`` is enabled, it is a physical delay and
    the commanded anticipation is subtracted before scheduling detachment.
    """
    global synthetic_rng, synthetic_rng_seed, synthetic_reset_id_seen

    prefix = "/tossing/release_delay/synthetic_ground_truth"
    seed = int(rospy.get_param(prefix + "/seed", 0))
    reset_id = int(rospy.get_param(prefix + "/reset_id", 0))
    if (synthetic_rng is None or synthetic_rng_seed != seed or
            synthetic_reset_id_seen != reset_id):
        synthetic_rng = np.random.RandomState(seed)
        synthetic_rng_seed = seed
        synthetic_reset_id_seen = reset_id

    distribution = str(rospy.get_param(prefix + "/distribution", "uniform")).lower()
    if distribution == "uniform":
        low_ms = float(rospy.get_param(prefix + "/min_ms", 20.0))
        high_ms = float(rospy.get_param(prefix + "/max_ms", 30.0))
        if high_ms < low_ms:
            raise ValueError("synthetic max_ms must be >= min_ms")
        sampled_ms = float(synthetic_rng.uniform(low_ms, high_ms))
    elif distribution == "fixed":
        sampled_ms = float(rospy.get_param(prefix + "/fixed_ms", 25.0))
    else:
        raise ValueError(
            "Synthetic ground-truth mode supports 'uniform' or 'fixed', got {}".format(
                distribution
            )
        )

    rospy.set_param(prefix + "/last_sample_ms", sampled_ms)
    rospy.set_param(prefix + "/last_sample_seconds", sampled_ms / 1000.0)
    rospy.loginfo("SYNTHETIC GT sampled delay value: %.3f ms", sampled_ms)
    return sampled_ms

def release(obj_name, obj_link_name):
    detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
    detach_srv.wait_for_service()
    req = AttachRequest()
    req.model_name_1 = rob_model_name
    req.link_name_1 = rob_link_name
    req.model_name_2 = obj_name
    req.link_name_2 = obj_link_name
    return detach_srv.call(req).ok


def _release_condition(pose):
    read_x = pose.position.x + meas_std * np.random.normal()
    read_y = pose.position.y + meas_std * np.random.normal()
    read_z = pose.position.z + meas_std * np.random.normal()
    if TOSS_TYPE == TOSS_TYPE_below:
        return (
            read_z >= release_position[2] - epsilon
            and math.dist([pose.position.x, pose.position.y], [0.0, 0.0]) > 0.6
        )
    if TOSS_TYPE == TOSS_TYPE_catapult_back:
        return (
            math.sqrt(read_x * read_x + read_y * read_y) >= release_position[0]
            and read_x >= 0.0
            and read_z >= 2.0
        )
    return False


def _append_state(model_status):
    pose_stamp = geometry_msgs.msg.PoseStamped()
    pose_stamp.pose = model_status.pose.pose
    pose_stamp.header = model_status.header
    model_trj.path.poses.append(pose_stamp)
    model_trj.twists.append(model_status.twist.twist)


def _clear_pending_release_capture():
    global pending_release_deadline, pending_release_capture_bullet
    with pending_release_capture_lock:
        pending_release_samples.clear()
        pending_release_deadline = None
        pending_release_capture_bullet = None


def _arm_pending_release_capture(obj_name, deadline_sec):
    global pending_release_deadline, pending_release_capture_bullet
    with pending_release_capture_lock:
        pending_release_samples.clear()
        pending_release_deadline = rospy.Time.from_sec(float(deadline_sec))
        pending_release_capture_bullet = obj_name


def _buffer_pending_release_sample(model_status, ref_bullet):
    """Retain post-deadline odometry while scheduled_detach is blocked.

    The deadline is a lower bound on the actual detach time.  Final selection
    is deferred until the plugin returns its authoritative detach timestamp.
    """
    with pending_release_capture_lock:
        capture_pending = (
            pending_release_deadline is not None
            and pending_release_capture_bullet == ref_bullet
            and release_command_active
            and release_command_bullet == ref_bullet
        )
        if not capture_pending:
            return False
        # Preserve the earliest samples, not the most recent ones.  The first
        # one or two can share the deadline step, so keep a bounded prefix and
        # choose against the plugin's exact timestamp during finalization.
        if (model_status.header.stamp >= pending_release_deadline and
                len(pending_release_samples) < 256):
            pending_release_samples.append(copy.deepcopy(model_status))
        return True


def _take_samples_after_detach(release_time_sec):
    """Return all buffered samples at/after actual Gazebo detach in time order."""
    global pending_release_deadline, pending_release_capture_bullet
    actual_time = rospy.Time.from_sec(float(release_time_sec))
    with pending_release_capture_lock:
        eligible = [
            sample for sample in pending_release_samples
            if sample.header.stamp >= actual_time
        ]
        buffered_count = len(pending_release_samples)
        pending_release_samples.clear()
        pending_release_deadline = None
        pending_release_capture_bullet = None
    if not eligible:
        return [], buffered_count
    eligible.sort(key=lambda sample: sample.header.stamp.to_sec())
    return eligible, buffered_count


def _begin_trajectory_capture(model_status, source):
    """Append the first valid free-flight sample and record capture latency."""
    global projectile_released
    _append_state(model_status)
    _start_trajectory_watchdog()
    projectile_released = False

    first_time = model_status.header.stamp.to_sec()
    rospy.set_param(
        "/tossing/release_delay/last_first_trajectory_sample_sim_time",
        float(first_time),
    )
    if release_actual_sim_time is not None:
        capture_lag_ms = 1000.0 * (
            first_time - release_actual_sim_time.to_sec()
        )
        rospy.set_param(
            "/tossing/release_delay/last_detach_to_first_sample_ms",
            float(capture_lag_ms),
        )
        rospy.loginfo(
            "FREE-FLIGHT CAPTURE started from %s: detach=%.6f | first=%.6f | lag=%+.3f ms",
            source,
            release_actual_sim_time.to_sec(),
            first_time,
            capture_lag_ms,
        )



def _cancel_release_timer():
    global release_timer
    if release_timer is not None:
        try:
            release_timer.shutdown()
        except Exception:
            pass
        release_timer = None


def _cancel_trajectory_watchdog():
    global trajectory_watchdog
    if trajectory_watchdog is not None:
        try:
            trajectory_watchdog.shutdown()
        except Exception:
            pass
        trajectory_watchdog = None


def _reset_tracking_state(reason=None, publish_empty=False):
    """Clear all per-throw state so the next toss can always start cleanly.

    If ``publish_empty`` is true, publish an empty ModelTrajectory first.  This
    wakes the waiting tossing service and lets it report a failed acquisition
    instead of hanging until its 30 s timeout.
    """
    global model_trj
    global release_command_active, release_command_bullet, projectile_released
    global release_actual_sim_time

    if reason:
        rospy.logwarn("Resetting projectile tracker: %s", reason)

    if publish_empty and bullet_pub is not None:
        failure_msg = ModelTrajectory()
        failure_msg.name = model_trj.name
        bullet_pub.publish(failure_msg)

    model_trj = ModelTrajectory()
    _cancel_release_timer()
    _cancel_trajectory_watchdog()
    release_command_active = False
    release_command_bullet = None
    projectile_released = False
    release_actual_sim_time = None
    _clear_pending_release_capture()


def _trajectory_watchdog_callback(_event):
    """Abort a throw that never reaches the target plane.

    This is a recovery path for contacts/rim hits or other simulator events
    that leave an active trajectory forever.  Without it, all later opening
    commands used to be ignored because ``model_trj`` stayed non-empty.
    """
    global trajectory_watchdog
    trajectory_watchdog = None
    if len(model_trj.path.poses) == 0:
        return

    timeout_s = float(rospy.get_param(
        "/tossing/release_delay/trajectory_timeout_seconds", 3.0
    ))
    _reset_tracking_state(
        "projectile did not cross the target plane within {:.3f} s".format(
            timeout_s
        ),
        publish_empty=True,
    )


def _start_trajectory_watchdog():
    global trajectory_watchdog
    _cancel_trajectory_watchdog()
    timeout_s = float(rospy.get_param(
        "/tossing/release_delay/trajectory_timeout_seconds", 3.0
    ))
    if timeout_s <= 0.0:
        return
    trajectory_watchdog = rospy.Timer(
        rospy.Duration.from_sec(timeout_s),
        _trajectory_watchdog_callback,
        oneshot=True,
    )


def _append_effective_delay_record(record):
    """Append one timing record to a CSV file for later calibration analysis."""
    path = rospy.get_param(
        "/tossing/release_delay/effective_delay_log_file",
        os.path.expanduser("~/.ros/tossing_effective_release_delay.csv"),
    )
    path = os.path.abspath(os.path.expanduser(str(path)))
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    fieldnames = [
        "mode",
        "detach_sim_time",
        "open_publish_sim_time",
        "sampled_physical_delay_ms",
        "sampled_synthetic_residual_ms",
        "synthetic_scheduled_deadline_sim_time",
        "measured_command_to_detach_ms",
        "nominal_release_sim_time",
        "nominal_interpolation_valid",
        "effective_residual_ms",
        "synthetic_realization_error_ms",
        "nominal_release_trajectory_time",
        "open_feedback_trajectory_time",
    ]
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fieldnames})
    rospy.loginfo("TIMING PROBE saved effective-delay record to %s", path)


def _finalize_release_after_detach(obj_name, release_time_sec, synthetic_mode):
    """Record one completed detach and enable free-flight trajectory capture.

    ``release_time_sec`` is returned by the Gazebo update-thread
    scheduled-detach service, so it is the authoritative physics-step timestamp
    and is independent of Python/ROS callback and service-response latency.
    """
    global release_timer, release_command_active, projectile_released
    global release_actual_sim_time

    release_actual_sim_time = rospy.Time.from_sec(float(release_time_sec))
    rospy.set_param(
        "/tossing/release_delay/last_actual_release_sim_time",
        float(release_time_sec),
    )

    prefix = "/tossing/release_delay"
    open_pub_time = rospy.get_param(prefix + "/last_open_publish_sim_time", None)
    open_fb_elapsed = rospy.get_param(prefix + "/last_open_feedback_desired_elapsed", None)
    nominal_time = rospy.get_param(prefix + "/last_nominal_release_time_seconds", None)
    if open_pub_time is not None:
        actual_actuator_delay = float(release_time_sec) - float(open_pub_time)
        rospy.set_param(prefix + "/last_measured_command_to_detach_seconds", actual_actuator_delay)
        if open_fb_elapsed is not None:
            inferred_detach_elapsed = float(open_fb_elapsed) + actual_actuator_delay
            rospy.set_param(prefix + "/last_inferred_detach_trajectory_time_seconds", inferred_detach_elapsed)
            if nominal_time is not None:
                residual = inferred_detach_elapsed - float(nominal_time)
                rospy.set_param(prefix + "/last_inferred_residual_from_feedback_seconds", residual)
                rospy.loginfo(
                    "TIMING PROBE detach: open_fb=%.6f s, command->detach=%.3f ms, inferred traj detach=%.6f s, nominal=%.6f s, residual=%.3f ms",
                    float(open_fb_elapsed), 1000.0 * actual_actuator_delay,
                    inferred_detach_elapsed, float(nominal_time), 1000.0 * residual,
                )

    nominal_sim = rospy.get_param(prefix + "/last_nominal_release_sim_time_interpolated", None)
    interpolation_valid = bool(rospy.get_param(
        prefix + "/last_nominal_release_sim_time_interpolation_valid", False
    ))
    nominal_source = "interpolated" if interpolation_valid else "fallback"
    if synthetic_mode and nominal_sim is None:
        # The synthetic deadline can occur before the first feedback sample
        # after nominal release is available.  The prediction used to schedule
        # the deadline is retained here; panda_utils later provides an
        # independent interpolated post-check.
        nominal_sim = rospy.get_param(
            prefix + "/synthetic_ground_truth/last_predicted_nominal_sim_time",
            None,
        )
        nominal_source = "predicted_from_open_feedback"

    if nominal_sim is not None:
        effective_residual = float(release_time_sec) - float(nominal_sim)
        rospy.set_param(prefix + "/last_effective_residual_seconds", effective_residual)
        sampled_ms = rospy.get_param(prefix + "/last_sampled_physical_delay_ms", float('nan'))
        sampled_synthetic_ms = rospy.get_param(
            prefix + "/synthetic_ground_truth/last_sample_ms", None
        )
        expected_synthetic_residual_ms = rospy.get_param(
            prefix + "/synthetic_ground_truth/last_effective_residual_ms",
            sampled_synthetic_ms,
        )
        scheduled_deadline = rospy.get_param(
            prefix + "/synthetic_ground_truth/last_scheduled_deadline_sim_time", None
        )
        realization_error_ms = ""
        if synthetic_mode and expected_synthetic_residual_ms is not None:
            realization_error_ms = (
                1000.0 * effective_residual
                - float(expected_synthetic_residual_ms)
            )
            rospy.set_param(
                prefix + "/synthetic_ground_truth/last_realization_error_ms",
                float(realization_error_ms),
            )
            rospy.loginfo(
                "SYNTHETIC GT CHECK: sampled=%.3f ms | expected residual=%+.3f ms | measured effective=%.3f ms | error=%+.3f ms | nominal_sim=%.6f | detach_sim=%.6f | interpolation=%s",
                float(sampled_synthetic_ms), float(expected_synthetic_residual_ms),
                1000.0 * effective_residual,
                float(realization_error_ms), float(nominal_sim),
                float(release_time_sec), nominal_source,
            )
        else:
            rospy.loginfo(
                "EFFECTIVE RELEASE DELAY: sampled actuator=%.3f ms | nominal_sim=%.6f | detach_sim=%.6f | effective residual=%.3f ms | interpolation=%s",
                float(sampled_ms), float(nominal_sim), float(release_time_sec),
                1000.0 * effective_residual,
                "valid" if interpolation_valid else "fallback",
            )
        _append_effective_delay_record({
            "mode": "synthetic_ground_truth" if synthetic_mode else "physical_actuator",
            "detach_sim_time": float(release_time_sec),
            "open_publish_sim_time": open_pub_time if open_pub_time is not None else "",
            "sampled_physical_delay_ms": "" if synthetic_mode else sampled_ms,
            "sampled_synthetic_residual_ms": expected_synthetic_residual_ms if synthetic_mode else "",
            "synthetic_scheduled_deadline_sim_time": scheduled_deadline if synthetic_mode else "",
            "measured_command_to_detach_ms": (1000.0 * (float(release_time_sec) - float(open_pub_time))) if open_pub_time is not None else "",
            "nominal_release_sim_time": float(nominal_sim),
            "nominal_interpolation_valid": interpolation_valid,
            "effective_residual_ms": 1000.0 * effective_residual,
            "synthetic_realization_error_ms": realization_error_ms,
            "nominal_release_trajectory_time": nominal_time if nominal_time is not None else "",
            "open_feedback_trajectory_time": open_fb_elapsed if open_fb_elapsed is not None else "",
        })
    else:
        rospy.logwarn("TIMING PROBE: no nominal-release simulation timestamp was available at detachment")

    if latest_joint_state is not None:
        rospy.set_param(prefix + "/last_detach_joint_names", list(latest_joint_state.name))
        rospy.set_param(prefix + "/last_detach_joint_positions", [float(x) for x in latest_joint_state.position])
        rospy.set_param(prefix + "/last_detach_joint_velocities", [float(x) for x in latest_joint_state.velocity])
        rospy.set_param(prefix + "/last_detach_joint_state_stamp", latest_joint_state.header.stamp.to_sec())

    buffered_samples = []
    buffered_count = 0
    # Keep this lock until every retained sample is committed.  Odometry
    # callbacks waiting in _buffer_pending_release_sample() can then resume only
    # after model_trj already contains the complete buffered prefix, preventing
    # newer live samples from being inserted ahead of older buffered samples.
    with pending_release_capture_lock:
        buffered_samples, buffered_count = _take_samples_after_detach(
            release_time_sec
        )

        # If valid samples were retained while the scheduled-detach service was
        # returning, reserve trajectory initialization for this thread.
        # Otherwise preserve the legacy behavior and let the next odometry
        # callback start it.
        projectile_released = not buffered_samples
        release_command_active = False
        release_timer = None

        if buffered_samples:
            rospy.loginfo(
                "Recovered %d post-detach odometry samples from %d buffered samples",
                len(buffered_samples),
                buffered_count,
            )
            _begin_trajectory_capture(
                buffered_samples[0], "scheduled-detach buffer"
            )
            for buffered_sample in buffered_samples[1:]:
                _append_state(buffered_sample)


def _start_physical_release_delay(obj_name):
    """Schedule either the physical actuator delay or a synthetic residual.

    Synthetic-ground-truth mode bypasses the actuator model completely.  The
    opening command is only an *early scheduling notification*.  A known
    residual d is sampled and the detach deadline is set to

        nominal_release_sim_time + d.

    The nominal simulation timestamp is predicted from the controller feedback
    sample that emitted the opening command.  The later timing-probe
    interpolation is retained as an independent realization check.
    """
    global release_timer, release_command_active, release_command_bullet
    global projectile_released, release_actual_sim_time

    _cancel_release_timer()
    _clear_pending_release_capture()

    release_command_bullet = obj_name
    release_command_active = True
    projectile_released = False
    release_actual_sim_time = None

    command_time = rospy.Time.now()
    prefix = "/tossing/release_delay"

    # Never let timing data from the previous throw leak into this throw.
    for key in (
        prefix + "/last_nominal_release_sim_time_interpolated",
        prefix + "/last_nominal_release_sim_time_interpolation_valid",
    ):
        if rospy.has_param(key):
            rospy.delete_param(key)

    if _synthetic_ground_truth_enabled():
        sampled_ms = _sample_synthetic_residual_ms()
        rospy.set_param(prefix + "/last_sampled_physical_delay_ms", -1.0)

        open_pub_time = rospy.get_param(prefix + "/last_open_publish_sim_time", None)
        open_fb_elapsed = rospy.get_param(prefix + "/last_open_feedback_desired_elapsed", None)
        nominal_time = rospy.get_param(prefix + "/last_nominal_release_time_seconds", None)
        if open_pub_time is None or open_fb_elapsed is None or nominal_time is None:
            release_command_active = False
            release_command_bullet = None
            raise RuntimeError(
                "Synthetic ground-truth release requires open publish time, "
                "open feedback elapsed time, and nominal release time. "
                "Make sure the companion panda_utils patch is active."
            )

        # Predict the nominal release timestamp in Gazebo simulation time from
        # the controller feedback sample that emitted the early scheduling
        # notification.  The later feedback interpolation remains an independent
        # post-check of this prediction.
        nominal_sim_pred = (
            float(open_pub_time)
            + float(nominal_time)
            - float(open_fb_elapsed)
        )
        compensate_anticipation = bool(rospy.get_param(
            prefix + "/synthetic_ground_truth/compensate_command_anticipation",
            False,
        ))
        anticipation_seconds = float(rospy.get_param(
            prefix + "/last_command_anticipation_seconds",
            0.0,
        ))
        effective_residual_seconds = float(sampled_ms) / 1000.0
        if compensate_anticipation:
            effective_residual_seconds -= anticipation_seconds
        deadline_sec = nominal_sim_pred + effective_residual_seconds
        now_sec = rospy.Time.now().to_sec()
        _arm_pending_release_capture(obj_name, deadline_sec)

        rospy.set_param(prefix + "/synthetic_ground_truth/last_predicted_nominal_sim_time", nominal_sim_pred)
        rospy.set_param(prefix + "/synthetic_ground_truth/last_scheduled_deadline_sim_time", deadline_sec)
        rospy.set_param(
            prefix + "/synthetic_ground_truth/last_effective_residual_ms",
            1000.0 * effective_residual_seconds,
        )
        rospy.set_param(prefix + "/last_command_sim_time", command_time.to_sec())
        rospy.set_param(prefix + "/last_release_deadline_sim_time", deadline_sec)

        if deadline_sec <= now_sec:
            rospy.logerr(
                "SYNTHETIC GT deadline is already late by %.3f ms. Increase "
                "release anticipation so the scheduling command arrives before "
                "nominal_release + residual.",
                1000.0 * (now_sec - deadline_sec),
            )

        # Register the exact simulation-time deadline with the Gazebo world
        # plugin.  The service blocks until WorldUpdateBegin detaches the joint
        # at the first physics step at/after deadline_sec and returns that exact
        # simulator timestamp.  Python callback latency therefore cannot alter
        # the synthetic ground truth.
        scheduled_srv = rospy.ServiceProxy(
            "/link_attacher_node/scheduled_detach", ScheduledDetach
        )
        scheduled_srv.wait_for_service(timeout=2.0)
        req = ScheduledDetachRequest()
        req.model_name_1 = rob_model_name
        req.link_name_1 = rob_link_name
        req.model_name_2 = obj_name
        req.link_name_2 = "{}::link".format(obj_name)
        req.detach_sim_time = float(deadline_sec)

        rospy.loginfo(
            "SYNTHETIC GT plugin schedule %s: nominal_sim(pred)=%.6f | "
            "sampled delay=%.3f ms | anticipation=%.3f ms | "
            "effective residual=%+.3f ms | detach deadline=%.6f | "
            "lead time=%.3f ms",
            obj_name, nominal_sim_pred, float(sampled_ms),
            1000.0 * anticipation_seconds, 1000.0 * effective_residual_seconds,
            deadline_sec,
            1000.0 * (deadline_sec - now_sec),
        )
        try:
            res = scheduled_srv.call(req)
        except Exception as exc:
            _clear_pending_release_capture()
            release_command_active = False
            release_command_bullet = None
            raise RuntimeError(
                "Gazebo scheduled-detach service failed: {}".format(exc)
            )

        if not res.ok:
            _clear_pending_release_capture()
            rospy.logerr(
                "SYNTHETIC GT scheduled detach failed: %s", res.message
            )
            release_command_active = False
            release_command_bullet = None
            projectile_released = False
            return

        actual_sec = float(res.actual_detach_sim_time)
        overshoot_ms = 1000.0 * (actual_sec - deadline_sec)
        rospy.set_param(
            prefix + "/synthetic_ground_truth/last_plugin_deadline_overshoot_ms",
            overshoot_ms,
        )
        rospy.loginfo(
            "SYNTHETIC GT plugin detach complete: target=%.6f | actual=%.6f | "
            "physics-step overshoot=%+.3f ms",
            deadline_sec, actual_sec, overshoot_ms,
        )
        _finalize_release_after_detach(
            obj_name, actual_sec, synthetic_mode=True
        )
        return

    sampled_ms = _sample_delay()
    rospy.set_param(prefix + "/last_sampled_physical_delay_ms", float(sampled_ms))

    delay_seconds = float(sampled_ms) / 1000.0
    # Define the actuator delay from the publisher's timestamp, not from the
    # later subscriber callback.  This prevents ROS transport/callback latency
    # from being folded into the configured physical delay.
    open_pub_time = float(rospy.get_param(
        prefix + "/last_open_publish_sim_time", command_time.to_sec()
    ))
    deadline_sec = open_pub_time + delay_seconds
    now_sec = rospy.Time.now().to_sec()
    _arm_pending_release_capture(obj_name, deadline_sec)

    rospy.set_param(prefix + "/last_command_sim_time", open_pub_time)
    rospy.set_param(prefix + "/last_release_deadline_sim_time", deadline_sec)

    if deadline_sec <= now_sec:
        rospy.logwarn(
            "Physical detach deadline is already late by %.3f ms; detaching "
            "on the next Gazebo physics step",
            1000.0 * (now_sec - deadline_sec),
        )

    # Schedule the detach in Gazebo's update thread.  A Python rospy.Timer can
    # fire late under load, and the old synchronous detach call recorded time
    # only after its service response returned; both effects biased later runs
    # toward a longer realized delay.
    scheduled_srv = rospy.ServiceProxy(
        "/link_attacher_node/scheduled_detach", ScheduledDetach
    )
    scheduled_srv.wait_for_service(timeout=2.0)
    req = ScheduledDetachRequest()
    req.model_name_1 = rob_model_name
    req.link_name_1 = rob_link_name
    req.model_name_2 = obj_name
    req.link_name_2 = "{}::link".format(obj_name)
    req.detach_sim_time = float(deadline_sec)

    rospy.loginfo(
        "Opening command received for %s; Gazebo detachment scheduled in "
        "%.1f ms (sim-time deadline %.6f)",
        obj_name,
        1000.0 * delay_seconds,
        deadline_sec,
    )
    try:
        res = scheduled_srv.call(req)
    except Exception as exc:
        _clear_pending_release_capture()
        release_command_active = False
        release_command_bullet = None
        raise RuntimeError(
            "Gazebo physical scheduled-detach service failed: {}".format(exc)
        )

    if not res.ok:
        _clear_pending_release_capture()
        rospy.logerr("Physical scheduled detach failed: %s", res.message)
        release_command_active = False
        release_command_bullet = None
        projectile_released = False
        return

    actual_sec = float(res.actual_detach_sim_time)
    rospy.set_param(
        prefix + "/last_physical_plugin_deadline_overshoot_ms",
        1000.0 * (actual_sec - deadline_sec),
    )
    _finalize_release_after_detach(
        obj_name, actual_sec, synthetic_mode=False
    )

def _append_target_plane_crossing(model_status, target_altitude):
    """Append an interpolated state exactly at z == target_altitude."""
    if len(model_trj.path.poses) == 0:
        _append_state(model_status)
        return

    prev_pose_stamp = model_trj.path.poses[-1]
    prev_twist = model_trj.twists[-1]
    curr_pose = model_status.pose.pose
    curr_twist = model_status.twist.twist

    z0 = prev_pose_stamp.pose.position.z
    z1 = curr_pose.position.z
    denom = z1 - z0

    if abs(denom) < 1e-12:
        frac = 1.0
    else:
        frac = (target_altitude - z0) / denom
        frac = min(1.0, max(0.0, frac))

    pose_stamp = geometry_msgs.msg.PoseStamped()
    pose_stamp.header = model_status.header

    t0 = prev_pose_stamp.header.stamp.to_sec()
    t1 = model_status.header.stamp.to_sec()
    if t1 >= t0:
        pose_stamp.header.stamp = rospy.Time.from_sec(t0 + frac * (t1 - t0))

    pose_stamp.pose.position.x = (
        prev_pose_stamp.pose.position.x
        + frac * (curr_pose.position.x - prev_pose_stamp.pose.position.x)
    )
    pose_stamp.pose.position.y = (
        prev_pose_stamp.pose.position.y
        + frac * (curr_pose.position.y - prev_pose_stamp.pose.position.y)
    )
    pose_stamp.pose.position.z = target_altitude
    pose_stamp.pose.orientation = curr_pose.orientation

    twist = geometry_msgs.msg.Twist()
    for attr in ("x", "y", "z"):
        setattr(
            twist.linear,
            attr,
            getattr(prev_twist.linear, attr)
            + frac * (getattr(curr_twist.linear, attr) - getattr(prev_twist.linear, attr)),
        )
        setattr(
            twist.angular,
            attr,
            getattr(prev_twist.angular, attr)
            + frac * (getattr(curr_twist.angular, attr) - getattr(prev_twist.angular, attr)),
        )

    model_trj.path.poses.append(pose_stamp)
    model_trj.twists.append(twist)


def open_command_callback(_msg):
    """Start one physical actuator-delay sample for the active projectile."""
    if not rospy.get_param("/tossing_experiment_ready", False):
        rospy.logwarn("Ignoring release open command while experiment is not ready")
        return

    if len(model_trj.path.poses) != 0:
        # A new opening command can only belong to a new toss (panda_utils
        # sends at most one command per trajectory).  Therefore, a non-empty
        # trajectory here is stale state from a previous throw that never
        # reached the target plane.  Reset it instead of permanently ignoring
        # every subsequent throw.
        _reset_tracking_state(
            "new opening command arrived while a stale trajectory was active"
        )

    _start_physical_release_delay(bullet)


def bullet_callback(model_status, ref_bullet):
    global model_trj
    global release_command_active, release_command_bullet, projectile_released
    global release_actual_sim_time

    if bullet != ref_bullet:
        return

    # The tracker is launched together with the experiment node. This guard
    # prevents accidental detachment during its calibration sequence.
    if not rospy.get_param("/tossing_experiment_ready", False):
        if (len(model_trj.path.poses) != 0 or release_command_active or
                projectile_released):
            _reset_tracking_state("experiment is not ready")
        return

    pose = model_status.pose.pose
    if len(model_trj.path.poses) == 0:
        trigger_mode = rospy.get_param(
            "/tossing/release_delay/trigger_mode", "command"
        )

        if trigger_mode == "command":
            # In synthetic mode, retain odometry arriving while the blocking
            # scheduled-detach service response is in flight.  Finalization
            # selects the first timestamp at/after the actual plugin detach.
            if _buffer_pending_release_sample(model_status, ref_bullet):
                return
            # In legacy physical-delay mode, start at the first odometry sample
            # received after the detach service call.
            if not projectile_released or release_command_bullet != bullet:
                return
            # Never start the free-flight trajectory from an odometry message
            # that predates the detach service call.  This is essential at the
            # 1 kHz publication rate, where rospy can otherwise process a
            # backlog of stale odometry samples.
            if (release_actual_sim_time is not None and
                    model_status.header.stamp < release_actual_sim_time):
                return
            _begin_trajectory_capture(model_status, "odometry callback")
            return

        elif trigger_mode == "condition":
            # Backward-compatible mode: the nominal release condition starts
            # the same simulation-time actuator delay.
            if not release_command_active and not projectile_released:
                if not _release_condition(pose):
                    return
                _start_physical_release_delay(bullet)
                return

            if not projectile_released or release_command_bullet != bullet:
                return
            if (release_actual_sim_time is not None and
                    model_status.header.stamp < release_actual_sim_time):
                return
            _begin_trajectory_capture(model_status, "odometry callback")
            return

        else:
            raise ValueError(
                "Unknown /tossing/release_delay/trigger_mode: {}".format(
                    trigger_mode
                )
            )

    target_altitude = float(rospy.get_param("/tossing/target_altitude", 0.1))
    if pose.position.z < target_altitude:
        # Interpolate to the actual target plane so dropped odometry samples do
        # not turn an arbitrary high-above-ground state into the landing point.
        _append_target_plane_crossing(model_status, target_altitude)
        bullet_pub.publish(model_trj)
        _reset_tracking_state()
        model_trj.name = model_status.header.frame_id
    else:
        _append_state(model_status)


def synch(_clock_msg):
    global bullet
    bullet = rospy.get_param("bullet_name", "red_ball_friction")


if __name__ == "__main__":
    rospy.init_node("bullet_state_node")
    rospy.sleep(2.0)

    TOSS_TYPE = rospy.get_param("toss_type", TOSS_TYPE_below)
    if TOSS_TYPE == TOSS_TYPE_below:
        release_position = below_toss.release_position
    elif TOSS_TYPE == TOSS_TYPE_catapult_back:
        release_position = catapult_toss.release_position
    else:
        raise ValueError("Unknown toss_type: {}".format(TOSS_TYPE))

    rospack = rospkg.RosPack()
    bullets_dir = rospack.get_path("mcpilot") + "/" + rospy.get_param("bullet_models_files_path")
    models = sorted(os.listdir(bullets_dir))
    rospy.loginfo("Tracking: %s", models)

    # Each projectile publishes odometry at up to 1 kHz.  An unbounded rospy
    # subscriber queue accumulates stale messages because this Python callback
    # cannot reliably process all of them.  Keep only the newest sample.
    bullet_subs = [
        rospy.Subscriber(
            "{}_odom".format(model),
            Odometry,
            bullet_callback,
            model,
            queue_size=1,
            tcp_nodelay=True,
        )
        for model in models
    ]
    bullet_pub = rospy.Publisher("bullet_trajectory", ModelTrajectory, queue_size=50)
    clock_sub = rospy.Subscriber("clock", Clock, synch, queue_size=1)
    joint_state_sub = rospy.Subscriber("/joint_states", JointState, _joint_state_callback, queue_size=1)
    open_command_sub = rospy.Subscriber(
        "/tossing/release_delay/open_command",
        Empty,
        open_command_callback,
        queue_size=10,
    )
    rospy.spin()
