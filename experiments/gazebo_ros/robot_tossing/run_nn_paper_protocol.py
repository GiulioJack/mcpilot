#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Paper-ready supervised inverse-model baseline protocol for MC-PILOT tossing.

Protocol
--------
* Seeds: 100..104 by default.
* One shared random-exploration dataset of 75 valid throws per seed.
* The same dataset is reused across all NN architectures.
* Checkpoints use nested prefixes: 5,15,...,75 samples.
* Architecture: Nh=3; hidden width=200; ReLU.
* Every architecture/checkpoint is trained from scratch for 1000 Adam steps.
* One fixed set of 40 evaluation targets per seed is reused at every checkpoint.
* Evaluation throws are never used for training.
* No interactive plotting.
* Fully resumable: partial datasets and completed checkpoints are reused.

This intentionally preserves the inverse-regression data construction of
`test_unsupervised_learning.py`: when a throw misses, the measured/estimated
landing point becomes the NN input and the commanded release velocity is the
regression target.
"""

import argparse
import csv
import json
import math
import os
import pickle as pkl
import sys
import time
from pathlib import Path

# Non-interactive plotting in any transitively imported module.
os.environ.setdefault("MPLBACKEND", "Agg")

# This script is intended to live in experiments/gazebo_ros/robot_tossing.

import numpy as np
import rospy
import torch
from nav_msgs.msg import Odometry
from mcpilot.srv import TossExperiment

import policy_learning.Policy_Tossing as Policy
import tossing_utils


EXPERIMENT_SERVICE = "tossing_experiment_testing"
BULLET_STATE_TOPIC_FORMAT = "{}_odom"
TRAJECTORY_COLUMNS = 10
LANDING_SOURCE = "target_plane_trajectory_endpoint"
LEGACY_LANDING_SOURCE = "legacy_post_service_odometry"
ALPHA = -1.0 * np.pi / 180.0
TARGET_ALTITUDE = 0.1
MIN_TARGET_DIST = 0.75
TARGET_DIST_SPAN = 1.65  # 0.75 + 1.65 = 2.40 m
MAX_YAW = np.pi / 6.0
V_MIN_EXPLORATION = 1.0
V_MAX = 3.5


def parse_service_trajectory(flat_trajectory):
    """Decode the column-major trajectory returned by TossExperiment."""
    flat = np.asarray(flat_trajectory, dtype=float)
    if flat.size == 0:
        raise ValueError("Tossing service returned an empty trajectory")
    if flat.size % TRAJECTORY_COLUMNS != 0:
        raise ValueError(
            "Trajectory length {} is not divisible by {}".format(
                flat.size, TRAJECTORY_COLUMNS
            )
        )
    return flat.reshape(TRAJECTORY_COLUMNS, flat.size // TRAJECTORY_COLUMNS).T


def target_plane_endpoint(flat_trajectory, target_altitude, tolerance=0.05):
    """Return the final xyz sample, rejecting incomplete target-plane paths."""
    trajectory = parse_service_trajectory(flat_trajectory)
    endpoint = trajectory[-1, 1:4].copy()
    if not np.all(np.isfinite(endpoint)):
        raise ValueError("Tossing service returned a non-finite endpoint")
    if abs(float(endpoint[2]) - float(target_altitude)) > tolerance:
        raise ValueError(
            "Trajectory ended before the target plane: target_z={:.4f}, "
            "last_z={:.4f}".format(float(target_altitude), float(endpoint[2]))
        )
    return endpoint


def records_use_target_plane_endpoint(payload):
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return bool(records) and all(
        record.get("landing_source") == LANDING_SOURCE for record in records
    )


def records_match_landing_source(payload, landing_source):
    """Accept unlabelled records only in the explicit legacy resume mode."""
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not records:
        return False
    if landing_source == LEGACY_LANDING_SOURCE:
        return all(
            record.get("landing_source") in (None, LEGACY_LANDING_SOURCE)
            for record in records
        )
    return records_use_target_plane_endpoint(payload)


def summary_matches_landing_source(summary, landing_source):
    if landing_source == LEGACY_LANDING_SOURCE:
        return summary.get("landing_source") in (None, LEGACY_LANDING_SOURCE)
    return summary.get("landing_source") == LANDING_SOURCE


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_state(dist, yaw):
    # `dist` is the world-frame target radius. tossing_init expects the planar
    # displacement from the release point, so remove the 7 cm release radius.
    # This gives every baseline the same [0.75, 2.40] m target domain.
    return tossing_utils.tossing_init(
        dist - float(tossing_utils.release_position[0]), yaw, TARGET_ALTITUDE
    )


def training_plan(seed, n=75):
    """Deterministic plan matching the original seed+25 convention."""
    rng = np.random.RandomState(seed + 25)
    plan = []
    for i in range(n):
        dist = MIN_TARGET_DIST + rng.rand() * TARGET_DIST_SPAN
        yaw = 2.0 * (rng.rand() - 0.5) * MAX_YAW
        speed = V_MIN_EXPLORATION + rng.rand() * (V_MAX - V_MIN_EXPLORATION)
        plan.append({
            "index": i,
            "distance": float(dist),
            "yaw": float(yaw),
            "speed": float(speed),
            "initial_state": make_state(dist, yaw),
        })
    return plan


def evaluation_plan(seed, n=40, evaluation_seed=20260825):
    """Independent fixed target set, shared by all NN configs for a seed."""
    # Separate streams per seed while keeping them independent of training RNG.
    rng = np.random.RandomState((evaluation_seed + seed) % (2**32 - 1))
    plan = []
    for i in range(n):
        dist = MIN_TARGET_DIST + rng.rand() * TARGET_DIST_SPAN
        yaw = 2.0 * (rng.rand() - 0.5) * MAX_YAW
        state = make_state(dist, yaw)
        plan.append({
            "index": i,
            "distance": float(dist),
            "yaw": float(yaw),
            "initial_state": state,
        })
    return plan


def atomic_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pkl.dump(obj, f)
    os.replace(str(tmp), str(path))


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(str(tmp), str(path))


def service_proxy():
    rospy.wait_for_service(EXPERIMENT_SERVICE)
    return rospy.ServiceProxy(EXPERIMENT_SERVICE, TossExperiment)


def reset_release_delay_stream(seed):
    """Restart both physical and synthetic delay streams for paired tests."""
    physical_prefix = "/tossing/release_delay"
    rospy.set_param(physical_prefix + "/seed", int(seed))
    rospy.set_param(
        physical_prefix + "/reset_id",
        int(rospy.get_param(physical_prefix + "/reset_id", 0)) + 1,
    )
    synthetic_prefix = physical_prefix + "/synthetic_ground_truth"
    rospy.set_param(synthetic_prefix + "/seed", int(seed))
    rospy.set_param(
        synthetic_prefix + "/reset_id",
        int(rospy.get_param(synthetic_prefix + "/reset_id", 0)) + 1,
    )


def perform_exploration_throw(initial_state, speed, bullet_name, max_attempts=5):
    """Collect one inverse-model pair from the exact target-plane crossing."""
    srv = service_proxy()
    target = list(initial_state[6:9])

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = srv(target, False, float(speed), bullet_name, [0] * 3)
            if response is None:
                raise RuntimeError("Tossing service returned None")

            endpoint = target_plane_endpoint(response.trajectory, target[2])
            planar_endpoint = endpoint[:2]
            planar_target = np.asarray(target[:2], dtype=float)
            planar_error = float(np.linalg.norm(planar_endpoint - planar_target))
            overshoot = np.linalg.norm(planar_endpoint) > np.linalg.norm(planar_target)

            nn_input = list(initial_state)
            nn_input[6:9] = endpoint.tolist()
            signed_error = planar_error if overshoot else -planar_error

            return {
                "nn_input": nn_input,
                "speed": float(speed),
                "requested_target": target,
                "landing_used_for_regression": endpoint.tolist(),
                "target_plane_endpoint": endpoint.tolist(),
                "landing_source": LANDING_SOURCE,
                "signed_error": float(signed_error),
                "attempts": attempt,
            }

        except Exception as exc:
            last_exc = exc
            print("Exploration throw failed (attempt {}/{}): {}".format(
                attempt, max_attempts, repr(exc)
            ))
            rospy.sleep(0.5)

    raise RuntimeError("Exploration throw failed after {} attempts: {}".format(
        max_attempts, repr(last_exc)
    ))


def collect_dataset(seed, out_dir, bullet_name, n_train=75,
                    landing_source=LANDING_SOURCE):
    seed_dir = Path(out_dir) / "datasets" / "seed_{}".format(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = seed_dir / "training.pkl"
    plan_path = seed_dir / "training_plan.json"

    plan = training_plan(seed, n_train)
    atomic_json(plan, plan_path)

    if dataset_path.exists():
        data = pkl.load(open(dataset_path, "rb"))
        if data.get("X") and not records_match_landing_source(data, landing_source):
            raise RuntimeError(
                "Stored NN training data do not match the selected landing "
                "source: {}".format(dataset_path)
            )
    else:
        data = {"X": [], "Y": [], "records": []}

    print("\nSeed {}: training dataset has {}/{} valid throws".format(
        seed, len(data["X"]), n_train
    ))

    if landing_source == LEGACY_LANDING_SOURCE and len(data["X"]) < n_train:
        raise RuntimeError(
            "Legacy resume mode only completes evaluations from an existing "
            "complete dataset; {} has only {}/{} samples".format(
                dataset_path, len(data["X"]), n_train
            )
        )

    while len(data["X"]) < n_train:
        i = len(data["X"])
        item = plan[i]
        print("\n[seed {}] collecting training throw {}/{}: dist={:.3f}, yaw={:.3f}, v={:.3f}".format(
            seed, i + 1, n_train, item["distance"], item["yaw"], item["speed"]
        ))
        rec = perform_exploration_throw(
            list(item["initial_state"]),
            item["speed"],
            bullet_name,
        )
        data["X"].append(rec["nn_input"])
        data["Y"].append(rec["speed"])
        rec["plan_index"] = i
        data["records"].append(rec)
        atomic_pickle(data, dataset_path)

    return data, seed_dir


def build_policy(num_hidden, hidden_dim, device):
    return Policy.TossingNNPolicy(
        state_dim=9,
        pos_indeces=None,
        target_indeces=[6, 7, 8],
        alpha=ALPHA,
        num_hidden=num_hidden,
        hidden_dim=hidden_dim,
        activation_name="ReLU",
        u_max=V_MAX,
        flg_squash=True,
        device=device,
    )


def train_policy(X_np, Y_np, seed, num_hidden, hidden_dim, opt_steps, batch_size, device):
    # Same seed for all budgets of a seed/architecture: differences primarily
    # reflect the nested training prefix rather than initialization noise.
    torch.manual_seed(seed + 25)
    np.random.seed(seed + 25)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 25)

    model = build_policy(num_hidden, hidden_dim, device)
    X = torch.tensor(X_np, device=device, dtype=model.dtype)
    Y = torch.tensor(Y_np, device=device, dtype=model.dtype)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    last_loss = None
    for _ in range(opt_steps):
        # Preserve original with-replacement mini-batch sampling.
        batch_indices = np.random.choice(range(X.shape[0]), batch_size)
        inputs = X[batch_indices, :]
        labels = Y[batch_indices]
        outputs = model.get_v(inputs)[0]
        loss = loss_fn(outputs, labels.view(-1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    return model, last_loss


def perform_eval_throw(model, initial_state, bullet_name,
                       landing_source=LANDING_SOURCE, max_attempts=5):
    srv = service_proxy()
    target_xyz = list(initial_state[6:9])

    with torch.no_grad():
        state_tensor = torch.tensor(
            [initial_state], device=model.device, dtype=model.dtype
        )
        policy_cmd, _ = model.get_v(state_tensor, t=0, p_dropout=0.0)
        policy_cmd = max(float(policy_cmd.detach().cpu().numpy()[0]), 0.1)

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = srv(target_xyz, False, policy_cmd, bullet_name, [0] * 3)
            if response is None:
                raise RuntimeError("Tossing service returned None")

            target_xy = np.asarray(target_xyz[:2], dtype=float)
            if landing_source == LEGACY_LANDING_SOURCE:
                ball_pose = rospy.wait_for_message(
                    BULLET_STATE_TOPIC_FORMAT.format(bullet_name),
                    Odometry,
                    timeout=5.0,
                ).pose.pose
                endpoint = np.asarray(
                    [ball_pose.position.x, ball_pose.position.y, 0.0],
                    dtype=float,
                )
                target_record = [float(target_xy[0]), float(target_xy[1]), 0.0]
            else:
                endpoint = target_plane_endpoint(
                    response.trajectory, target_xyz[2]
                )
                target_record = [
                    float(target_xy[0]), float(target_xy[1]),
                    float(target_xyz[2]),
                ]

            error = float(np.linalg.norm(endpoint[:2] - target_xy))
            overshoot = np.linalg.norm(endpoint[:2]) > np.linalg.norm(target_xy)

            record = {
                "target": target_record,
                "landing": endpoint.tolist(),
                "landing_source": landing_source,
                "error": float(error),
                "signed_error": float(error if overshoot else -error),
                "velocity": float(policy_cmd),
                "attempts": attempt,
            }
            if landing_source == LANDING_SOURCE:
                record["target_plane_endpoint"] = endpoint.tolist()
            return record
        except Exception as exc:
            last_exc = exc
            print("Evaluation throw failed (attempt {}/{}): {}".format(
                attempt, max_attempts, repr(exc)
            ))
            rospy.sleep(0.5)

    raise RuntimeError("Evaluation throw failed after {} attempts: {}".format(
        max_attempts, repr(last_exc)
    ))


def evaluate_policy(model, eval_plan, result_path, bullet_name, target_radius,
                    landing_source=LANDING_SOURCE):
    result_path = Path(result_path)
    if result_path.exists():
        payload = pkl.load(open(result_path, "rb"))
        if payload.get("records") and not records_match_landing_source(
            payload, landing_source
        ):
            raise RuntimeError(
                "Stored NN evaluation data do not match the selected landing "
                "source: {}".format(result_path)
            )
    else:
        payload = {"records": []}

    records = payload["records"]
    while len(records) < len(eval_plan):
        i = len(records)
        item = eval_plan[i]
        print("  eval {}/{}: dist={:.3f}, yaw={:.3f}".format(
            i + 1, len(eval_plan), item["distance"], item["yaw"]
        ))
        rec = perform_eval_throw(
            model,
            list(item["initial_state"]),
            bullet_name,
            landing_source,
        )
        rec["target_index"] = i
        records.append(rec)
        atomic_pickle(payload, result_path)

    errors = np.asarray([r["error"] for r in records], dtype=float)
    success = errors <= target_radius
    return {
        "n_eval": int(len(errors)),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "std_error": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
        "p90_error": float(np.quantile(errors, 0.90)),
        "max_error": float(errors.max()),
    }


def write_global_summary(out_dir):
    out_dir = Path(out_dir)
    rows = []
    for summary_path in sorted(out_dir.glob("Nh_*/Ntrain_*/seed_*/summary.json")):
        with open(summary_path) as f:
            rows.append(json.load(f))
    if not rows:
        return
    fields = [
        "seed", "num_hidden", "n_train", "n_eval", "successes", "success_rate",
        "mean_error", "median_error", "std_error", "p90_error", "max_error",
        "train_loss", "evaluation_seed", "bullet_name", "target_radius",
        "landing_source",
    ]
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def seed_is_complete(out_dir, seed, depths, budgets, n_eval_intermediate,
                     n_eval_final, landing_source=LANDING_SOURCE):
    """Return true when every requested checkpoint for a seed is reusable."""
    out_dir = Path(out_dir)
    final_budget = max(budgets)
    for depth in depths:
        for budget in budgets:
            required = n_eval_final if budget == final_budget else n_eval_intermediate
            cfg_dir = (
                out_dir / "Nh_{}".format(depth) / "Ntrain_{}".format(budget)
                / "seed_{}".format(seed)
            )
            summary_path = cfg_dir / "summary.json"
            eval_path = cfg_dir / "tossing_test_NN_policy_1.pkl"
            if not summary_path.exists() or not eval_path.exists():
                return False
            try:
                summary = json.load(open(summary_path))
                payload = pkl.load(open(eval_path, "rb"))
                if not summary_matches_landing_source(summary, landing_source):
                    return False
                if not records_match_landing_source(payload, landing_source):
                    return False
                if summary.get("n_eval", 0) < required:
                    return False
                if len(payload.get("records", [])) < required:
                    return False
            except Exception:
                return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="100,101,102,103,104")
    p.add_argument("--depths", default="3")
    p.add_argument("--budgets", default="5,15,25,35,45,55,65,75")
    p.add_argument("--n-train-max", type=int, default=75)
    p.add_argument("--n-eval-intermediate", type=int, default=40)
    p.add_argument("--n-eval-final", type=int, default=40)
    p.add_argument("--evaluation-seed", type=int, default=20260825)
    p.add_argument("--hidden-dim", type=int, default=200)
    p.add_argument("--opt-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--bullet-name", default="red_ball_friction")
    p.add_argument("--target-radius", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="paper_results_2026/supervised_inverse_model")
    p.add_argument(
        "--legacy-odometry-resume",
        action="store_true",
        help=(
            "Explicitly resume an existing legacy NN result directory whose "
            "landing measurements came from post-service odometry. This mode "
            "will not collect or extend training datasets."
        ),
    )
    args = p.parse_args()

    landing_source = (
        LEGACY_LANDING_SOURCE if args.legacy_odometry_resume else LANDING_SOURCE
    )

    seeds = parse_int_list(args.seeds)
    depths = parse_int_list(args.depths)
    budgets = parse_int_list(args.budgets)
    if max(budgets) > args.n_train_max:
        raise ValueError("A requested budget exceeds --n-train-max")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "seeds": seeds,
        "depths": depths,
        "budgets": budgets,
        "n_train_max": args.n_train_max,
        "n_eval_intermediate": args.n_eval_intermediate,
        "n_eval_final": args.n_eval_final,
        "evaluation_seed": args.evaluation_seed,
        "hidden_dim": args.hidden_dim,
        "opt_steps": args.opt_steps,
        "batch_size": args.batch_size,
        "bullet_name": args.bullet_name,
        "target_radius": args.target_radius,
        "landing_source": landing_source,
        "target_domain": {
            "min_distance": MIN_TARGET_DIST,
            "max_distance": MIN_TARGET_DIST + TARGET_DIST_SPAN,
            "yaw_min": -MAX_YAW,
            "yaw_max": MAX_YAW,
            "target_altitude": TARGET_ALTITUDE,
        },
        "device": str(device),
        "notes": [
            "One shared 75-throw random-exploration dataset per seed.",
            "Only the Nh=3 architecture is reported.",
            "Nested dataset prefixes define interaction budgets.",
            "Every budget uses the same 40 paired evaluation targets.",
            "Evaluation throws are not added to training data.",
        ],
    }
    atomic_json(protocol, out_dir / "protocol.json")

    rospy.init_node("nn_paper_protocol_node", anonymous=True)

    for seed in seeds:
        if seed_is_complete(
            out_dir,
            seed,
            depths,
            budgets,
            args.n_eval_intermediate,
            args.n_eval_final,
            landing_source,
        ):
            print("SKIP completed NN seed {}".format(seed))
            continue

        reset_release_delay_stream(seed)
        data, dataset_dir = collect_dataset(
            seed,
            out_dir,
            args.bullet_name,
            args.n_train_max,
            landing_source,
        )

        max_eval = max(args.n_eval_intermediate, args.n_eval_final)
        eval_plan = evaluation_plan(seed, max_eval, args.evaluation_seed)
        atomic_json(eval_plan, dataset_dir / "evaluation_targets.json")

        X_all = np.asarray(data["X"], dtype=float)
        Y_all = np.asarray(data["Y"], dtype=float)

        for depth in depths:
            for budget in budgets:
                n_eval = (
                    args.n_eval_final if budget == max(budgets)
                    else args.n_eval_intermediate
                )
                budget_eval_plan = eval_plan[:n_eval]
                cfg_dir = out_dir / "Nh_{}".format(depth) / "Ntrain_{}".format(budget) / "seed_{}".format(seed)
                cfg_dir.mkdir(parents=True, exist_ok=True)
                summary_path = cfg_dir / "summary.json"
                eval_path = cfg_dir / "tossing_test_NN_policy_1.pkl"
                model_path = cfg_dir / "NN_policy.pt"

                if summary_path.exists() and eval_path.exists():
                    try:
                        old = json.load(open(summary_path))
                        payload = pkl.load(open(eval_path, "rb"))
                        if (
                            summary_matches_landing_source(old, landing_source)
                            and records_match_landing_source(payload, landing_source)
                            and old.get("n_eval", 0) >= n_eval
                            and len(payload.get("records", [])) >= n_eval
                        ):
                            print("SKIP completed: seed={}, Nh={}, Ntrain={}".format(seed, depth, budget))
                            continue
                    except Exception:
                        pass

                print("\n============================================================")
                print("TRAIN/EVAL seed={} Nh={} Ntrain={}".format(seed, depth, budget))
                print("============================================================")

                model, train_loss = train_policy(
                    X_all[:budget],
                    Y_all[:budget],
                    seed,
                    depth,
                    args.hidden_dim,
                    args.opt_steps,
                    args.batch_size,
                    device,
                )
                torch.save(model.state_dict(), model_path)

                # Every architecture and budget sees the same deterministic
                # delay realization sequence for this algorithm seed.
                reset_release_delay_stream(args.evaluation_seed + seed)
                stats = evaluate_policy(
                    model,
                    budget_eval_plan,
                    eval_path,
                    args.bullet_name,
                    args.target_radius,
                    landing_source,
                )
                summary = {
                    "seed": seed,
                    "num_hidden": depth,
                    "n_train": budget,
                    "train_loss": train_loss,
                    "evaluation_seed": args.evaluation_seed,
                    "bullet_name": args.bullet_name,
                    "target_radius": args.target_radius,
                    "landing_source": landing_source,
                    **stats,
                }
                atomic_json(summary, summary_path)
                write_global_summary(out_dir)

                print("Result: success={:.1f}% mean={:.4f} m median={:.4f} m".format(
                    100.0 * stats["success_rate"],
                    stats["mean_error"],
                    stats["median_error"],
                ))

    write_global_summary(out_dir)
    print("\nAll requested NN experiments completed.")
    print("Summary: {}".format(out_dir / "summary.csv"))


if __name__ == "__main__":
    main()
