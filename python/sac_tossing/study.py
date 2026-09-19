# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training, evaluation, and result serialization for SAC budget studies."""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from sac_tossing.config import save_config
from sac_tossing.environment import TossingSACEnv
from sac_tossing.metrics import normalized_auc, summarize_errors
from sac_tossing.targets import TargetDomain


def _write_rows(path: str, rows: Iterable[Mapping[str, Any]], append: bool = False) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows_list)


def _sac_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(config)
    net_arch = values.pop("net_arch", [256, 256])
    values["policy_kwargs"] = {"net_arch": list(net_arch)}
    return values


def evaluate_model(
    model: SAC,
    env: TossingSACEnv,
    targets: np.ndarray,
    repeats: int,
    budget: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        for repeat in range(int(repeats)):
            observation, _ = env.reset(options={"target": target})
            action, _ = model.predict(observation, deterministic=True)
            _, reward, terminated, truncated, info = env.step(action)
            if not terminated or truncated:
                raise RuntimeError("One-shot environment must terminate after one step")
            landing = np.asarray(info["landing_position"])
            rows.append(
                {
                    "seed": seed,
                    "budget": budget,
                    "target_index": target_index,
                    "repeat": repeat,
                    "target_x": target[0],
                    "target_y": target[1],
                    "target_z": target[2],
                    "landing_x": landing[0],
                    "landing_y": landing[1],
                    "landing_z": landing[2],
                    "landing_source": info["landing_source"],
                    "command_velocity": info["command_velocity"],
                    "command_anticipation_seconds": (
                        "" if info["command_anticipation_seconds"] is None
                        else info["command_anticipation_seconds"]
                    ),
                    "planar_error": info["planar_error"],
                    "success": int(info["is_success"]),
                    "reward": reward,
                    "delay_steps": "" if info["delay_steps"] is None else info["delay_steps"],
                    "throw_elapsed_seconds": info["throw_elapsed_seconds"],
                }
            )
    return rows


def run_seed(config: Dict[str, Any], seed: int, run_root: str) -> str:
    experiment_cfg = config["experiment"]
    environment_cfg = config["environment"]
    backend_name = str(experiment_cfg.get("backend", environment_cfg.get("backend", "ros")))
    seed_dir = os.path.join(run_root, "seed_{:03d}".format(seed))
    if os.path.isdir(seed_dir) and os.listdir(seed_dir):
        if bool(experiment_cfg.get("overwrite", False)):
            shutil.rmtree(seed_dir)
        else:
            raise FileExistsError(
                "Result directory already exists and is not empty: {}. "
                "Delete it or set experiment.overwrite=true.".format(seed_dir)
            )
    os.makedirs(seed_dir, exist_ok=True)
    save_config(config, os.path.join(seed_dir, "resolved_config.yaml"))

    train_env_raw = TossingSACEnv(
        environment_cfg,
        seed=seed,
        service_name=environment_cfg.get("service_train"),
        backend_name=backend_name,
        log_path=os.path.join(seed_dir, "training_episodes.csv"),
    )
    train_env = Monitor(train_env_raw, filename=os.path.join(seed_dir, "monitor.csv"))
    eval_env = TossingSACEnv(
        environment_cfg,
        seed=seed + 100000,
        service_name=environment_cfg.get("service_eval"),
        backend_name=backend_name,
        log_path=None,
    )
    eval_env.set_delay_config({
        "seed": int(experiment_cfg.get("evaluation_seed", 20260803)) + int(seed)
    })

    sac_kwargs = _sac_kwargs(config["sac"])
    model = SAC(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=int(experiment_cfg.get("verbose", 1)),
        tensorboard_log=os.path.join(seed_dir, "tensorboard"),
        **sac_kwargs,
    )

    budgets = sorted({int(value) for value in experiment_cfg["budgets"]})
    domain = TargetDomain.from_config(environment_cfg["target_domain"])
    # Match the legacy RandomState stream and interleaved radius/yaw draws used
    # by MC-PILOT, the NN protocol, and the analytical baseline. Therefore a
    # given algorithm seed receives exactly the same evaluation targets.
    eval_rng = np.random.RandomState(
        int(experiment_cfg.get("evaluation_seed", 20260803)) + int(seed)
    )
    n_eval_intermediate = int(experiment_cfg.get(
        "n_eval_targets_intermediate",
        experiment_cfg.get("n_eval_targets", 100),
    ))
    n_eval_final = int(experiment_cfg.get(
        "n_eval_targets_final",
        experiment_cfg.get("n_eval_targets", 100),
    ))
    n_eval_max = max(n_eval_intermediate, n_eval_final)
    targets = np.stack([
        np.asarray([
            radius * np.cos(gamma),
            radius * np.sin(gamma),
            domain.z,
        ], dtype=np.float64)
        for radius, gamma in [(
            eval_rng.uniform(domain.l_min, domain.l_max),
            eval_rng.uniform(-domain.gamma_max, domain.gamma_max),
        ) for _ in range(n_eval_max)]
    ], axis=0)
    np.save(os.path.join(seed_dir, "evaluation_targets.npy"), targets)

    all_eval_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    previous_budget = 0
    cumulative_train_seconds = 0.0
    for budget in budgets:
        increment = budget - previous_budget
        if increment <= 0:
            continue
        started = time.perf_counter()
        model.learn(
            total_timesteps=increment,
            reset_num_timesteps=False,
            progress_bar=bool(experiment_cfg.get("progress_bar", False)),
            tb_log_name="sac",
        )
        cumulative_train_seconds += time.perf_counter() - started
        model_path = os.path.join(seed_dir, "model_budget_{:04d}".format(budget))
        model.save(model_path)

        eval_started = time.perf_counter()
        # Pair the simulated delay sequence across interaction budgets.
        eval_env.reset_delay_stream()
        n_eval = n_eval_final if budget == budgets[-1] else n_eval_intermediate
        rows = evaluate_model(
            model=model,
            env=eval_env,
            targets=targets[:n_eval],
            repeats=int(experiment_cfg.get("eval_repeats", 1)),
            budget=budget,
            seed=seed,
        )
        evaluation_seconds = time.perf_counter() - eval_started
        all_eval_rows.extend(rows)
        errors = [float(row["planar_error"]) for row in rows]
        summary = summarize_errors(
            errors,
            hit_radius=float(environment_cfg["hit_radius"]),
            seed=seed + budget,
        )
        summary.update(
            {
                "seed": seed,
                "budget": budget,
                "training_seconds_cumulative": cumulative_train_seconds,
                "evaluation_seconds": evaluation_seconds,
                "training_interactions": budget,
                "evaluation_interactions": len(rows),
            }
        )
        summaries.append(summary)
        _write_rows(os.path.join(seed_dir, "summary.csv"), summaries, append=False)
        _write_rows(os.path.join(seed_dir, "evaluation.csv"), all_eval_rows, append=False)
        previous_budget = budget

    if summaries:
        auc_payload = {
            "success_auc_linear": normalized_auc(
                [row["budget"] for row in summaries],
                [row["success_rate"] for row in summaries],
                log_x=False,
            ),
            "success_auc_log_budget": normalized_auc(
                [row["budget"] for row in summaries],
                [row["success_rate"] for row in summaries],
                log_x=True,
            ),
        }
    else:
        auc_payload = {}
    import gymnasium
    import stable_baselines3
    import torch

    metadata = {
        "seed": seed,
        "backend": backend_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "budgets": budgets,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "sac_device": config["sac"].get("device", "auto"),
        **auc_payload,
    }
    with open(os.path.join(seed_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    train_env.close()
    eval_env.close()
    return seed_dir


def run_study(config: Dict[str, Any]) -> str:
    experiment_cfg = config["experiment"]
    output_root = os.path.abspath(os.path.expanduser(experiment_cfg.get("output_dir", "results/sac")))
    run_name = str(experiment_cfg.get("name", "sac_budget_study"))
    run_root = os.path.join(output_root, run_name)
    os.makedirs(run_root, exist_ok=True)
    save_config(config, os.path.join(run_root, "resolved_config.yaml"))
    for seed in [int(value) for value in experiment_cfg.get("seeds", [0])]:
        run_seed(config, seed, run_root)
    return run_root
