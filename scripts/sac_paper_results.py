#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Print and export paper-ready statistics from a SAC budget-study directory.

The script deliberately recomputes the success flag from ``planar_error`` and the
requested hit radius. This allows the same trained policies to be reported under
the exact simulation success criterion used by the manuscript.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_EVAL_COLUMNS = {
    "seed",
    "budget",
    "target_index",
    "target_x",
    "target_y",
    "target_z",
    "landing_x",
    "landing_y",
    "landing_z",
    "command_velocity",
    "planar_error",
}


def wilson_interval(successes, total, confidence=0.95):
    if total <= 0:
        return float("nan"), float("nan")
    # 1.959963984540054 for a two-sided 95% interval. The CLI currently keeps
    # confidence fixed at 0.95 to avoid a scipy dependency in ROS/Python 3.8.
    if abs(confidence - 0.95) > 1e-12:
        raise ValueError("Only confidence=0.95 is supported without scipy")
    z = 1.959963984540054
    p = float(successes) / float(total)
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_seed_mean(values, rng, n_bootstrap=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def read_evaluations(run_dir):
    frames = []
    for path in sorted(run_dir.glob("seed_*/evaluation.csv")):
        frame = pd.read_csv(path)
        missing = REQUIRED_EVAL_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
        frame = frame.copy()
        if "repeat" not in frame.columns:
            frame["repeat"] = 0
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No seed_*/evaluation.csv files found under {}".format(run_dir)
        )
    data = pd.concat(frames, ignore_index=True)
    for col in ["seed", "budget", "target_index", "repeat"]:
        data[col] = pd.to_numeric(data[col], errors="raise").astype(int)
    data["planar_error"] = pd.to_numeric(data["planar_error"], errors="raise")
    return data


def read_training_times(run_dir):
    rows = []
    for path in sorted(run_dir.glob("seed_*/summary.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns or "budget" not in frame.columns:
            continue
        keep = ["seed", "budget"]
        for optional in [
            "training_seconds_cumulative",
            "evaluation_seconds",
            "training_interactions",
            "evaluation_interactions",
        ]:
            if optional in frame.columns:
                keep.append(optional)
        rows.append(frame[keep])
    if not rows:
        return None
    result = pd.concat(rows, ignore_index=True)
    result["seed"] = result["seed"].astype(int)
    result["budget"] = result["budget"].astype(int)
    return result


def summarize_by_seed(data, hit_radius):
    data = data.copy()
    data["success_recomputed"] = data["planar_error"] <= hit_radius
    rows = []
    for (seed, budget), group in data.groupby(["seed", "budget"], sort=True):
        errors = group["planar_error"].to_numpy(dtype=float)
        successes = int(group["success_recomputed"].sum())
        rows.append(
            {
                "seed": int(seed),
                "budget": int(budget),
                "n_evaluations": int(len(group)),
                "successes": successes,
                "success_rate": successes / float(len(group)),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                "p90_error": float(np.quantile(errors, 0.90)),
                "max_error": float(np.max(errors)),
                "velocity_min": float(group["command_velocity"].min()),
                "velocity_max": float(group["command_velocity"].max()),
            }
        )
    return pd.DataFrame(rows), data


def summarize_by_budget(seed_summary, evaluations, training_times, rng):
    rows = []
    for budget, seed_group in seed_summary.groupby("budget", sort=True):
        eval_group = evaluations[evaluations["budget"] == budget]
        total = int(len(eval_group))
        successes = int(eval_group["success_recomputed"].sum())
        ci_low, ci_high = wilson_interval(successes, total)
        mean_ci_low, mean_ci_high = bootstrap_seed_mean(
            seed_group["mean_error"].to_numpy(), rng
        )
        median_ci_low, median_ci_high = bootstrap_seed_mean(
            seed_group["median_error"].to_numpy(), rng
        )
        row = {
            "budget": int(budget),
            "n_seeds": int(seed_group["seed"].nunique()),
            "n_evaluations": total,
            "successes": successes,
            "pooled_success_rate": successes / float(total),
            "success_ci_low": ci_low,
            "success_ci_high": ci_high,
            "seed_success_rate_mean": float(seed_group["success_rate"].mean()),
            "seed_success_rate_std": float(seed_group["success_rate"].std(ddof=1))
            if len(seed_group) > 1
            else 0.0,
            "success_rate_median": float(seed_group["success_rate"].median()),
            "success_rate_q25": float(seed_group["success_rate"].quantile(0.25)),
            "success_rate_q75": float(seed_group["success_rate"].quantile(0.75)),
            "mean_error_mean": float(seed_group["mean_error"].mean()),
            "mean_error_std": float(seed_group["mean_error"].std(ddof=1))
            if len(seed_group) > 1
            else 0.0,
            "mean_error_ci_low": mean_ci_low,
            "mean_error_ci_high": mean_ci_high,
            "median_error_mean": float(seed_group["median_error"].mean()),
            "median_error_ci_low": median_ci_low,
            "median_error_ci_high": median_ci_high,
            "p90_error_mean": float(seed_group["p90_error"].mean()),
            "max_error": float(seed_group["max_error"].max()),
        }
        if training_times is not None:
            time_group = training_times[training_times["budget"] == budget]
            if not time_group.empty:
                for col in [
                    "training_seconds_cumulative",
                    "evaluation_seconds",
                    "training_interactions",
                    "evaluation_interactions",
                ]:
                    if col in time_group.columns:
                        row[col + "_mean"] = float(time_group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_representative_seed(seed_summary, final_budget):
    group = seed_summary[seed_summary["budget"] == final_budget].copy()
    if group.empty:
        raise ValueError("No seed results at final budget {}".format(final_budget))
    median_error = float(group["mean_error"].median())
    group["distance_to_median"] = np.abs(group["mean_error"] - median_error)
    return int(group.sort_values(["distance_to_median", "seed"]).iloc[0]["seed"])


def write_text_table(summary, hit_radius):
    lines = []
    lines.append("SAC paper-ready results (hit radius = {:.3f} m)".format(hit_radius))
    lines.append(
        "budget  seeds  evaluations  success [95% Wilson CI]   mean error [95% seed-bootstrap CI]  median error"
    )
    for _, row in summary.iterrows():
        lines.append(
            "{budget:6d}  {seeds:5d}  {n:11d}  {rate:6.1f}% [{lo:5.1f}, {hi:5.1f}]%   "
            "{mean:.4f} m [{mlo:.4f}, {mhi:.4f}]   {median:.4f} m".format(
                budget=int(row["budget"]),
                seeds=int(row["n_seeds"]),
                n=int(row["n_evaluations"]),
                rate=100.0 * row["pooled_success_rate"],
                lo=100.0 * row["success_ci_low"],
                hi=100.0 * row["success_ci_high"],
                mean=row["mean_error_mean"],
                mlo=row["mean_error_ci_low"],
                mhi=row["mean_error_ci_high"],
                median=row["median_error_mean"],
            )
        )
    return "\n".join(lines) + "\n"


def write_latex_table(summary, hit_radius):
    lines = [
        "% Generated by scripts/sac_paper_results.py",
        "% Success is recomputed using hit radius {:.3f} m.".format(hit_radius),
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Target-conditioned Soft Actor-Critic results in simulation. Success is defined by a planar landing error not exceeding %.3f~m. Confidence intervals for success are pooled Wilson intervals; error intervals are bootstrap intervals over training seeds.}" % hit_radius,
        "\\label{tab:sac_simulation_results}",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "Interactions & Success & Mean error [m] & Median error [m] & Evaluations \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        success_text = "{:.1f}\\% [{:.1f}, {:.1f}]".format(
            100.0 * row["pooled_success_rate"],
            100.0 * row["success_ci_low"],
            100.0 * row["success_ci_high"],
        )
        mean_text = "{:.3f} [{:.3f}, {:.3f}]".format(
            row["mean_error_mean"],
            row["mean_error_ci_low"],
            row["mean_error_ci_high"],
        )
        lines.append(
            "{} & {} & {} & {:.3f} & {} \\\\".format(
                int(row["budget"]),
                success_text,
                mean_text,
                row["median_error_mean"],
                int(row["n_evaluations"]),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Print and export paper-ready SAC budget-study results."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--hit-radius",
        type=float,
        required=True,
        help="Success threshold in metres. Use the exact criterion of the compared simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: RUN_DIR/paper_export",
    )
    parser.add_argument(
        "--final-budget",
        type=int,
        default=None,
        help="Budget used for the representative target-distribution export.",
    )
    parser.add_argument(
        "--map-seed",
        type=int,
        default=None,
        help="Seed used for the target-distribution map. Default: seed closest to median final mean error.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "paper_export")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = read_evaluations(run_dir)
    seed_summary, evaluations = summarize_by_seed(evaluations, args.hit_radius)
    training_times = read_training_times(run_dir)
    if training_times is not None:
        seed_summary = seed_summary.merge(
            training_times,
            on=["seed", "budget"],
            how="left",
            validate="one_to_one",
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    budget_summary = summarize_by_budget(
        seed_summary, evaluations, training_times, rng
    )

    final_budget = args.final_budget or int(budget_summary["budget"].max())
    map_seed = args.map_seed
    if map_seed is None:
        map_seed = choose_representative_seed(seed_summary, final_budget)

    map_rows = evaluations[
        (evaluations["budget"] == final_budget)
        & (evaluations["seed"] == map_seed)
        & (evaluations["repeat"] == evaluations["repeat"].min())
    ].copy()
    if map_rows.empty:
        raise ValueError(
            "No evaluation rows for map seed {} at budget {}".format(
                map_seed, final_budget
            )
        )
    map_rows["success"] = map_rows["success_recomputed"].astype(int)

    curve_columns = [
        "budget",
        "n_seeds",
        "n_evaluations",
        "success_rate_median",
        "success_rate_q25",
        "success_rate_q75",
        "pooled_success_rate",
        "success_ci_low",
        "success_ci_high",
        "mean_error_mean",
        "mean_error_ci_low",
        "mean_error_ci_high",
        "median_error_mean",
    ]

    seed_summary.to_csv(output_dir / "sac_results_by_seed.csv", index=False)
    budget_summary.to_csv(output_dir / "sac_results_by_budget.csv", index=False)
    budget_summary[curve_columns].to_csv(output_dir / "sac_curve.csv", index=False)
    map_rows[
        [
            "seed",
            "budget",
            "target_index",
            "target_x",
            "target_y",
            "target_z",
            "landing_x",
            "landing_y",
            "landing_z",
            "command_velocity",
            "planar_error",
            "success",
        ]
    ].sort_values("target_index").to_csv(
        output_dir / "sac_final_targets.csv", index=False
    )

    text_table = write_text_table(budget_summary, args.hit_radius)
    latex_table = write_latex_table(budget_summary, args.hit_radius)
    (output_dir / "sac_results_table.txt").write_text(text_table)
    (output_dir / "sac_results_table.tex").write_text(latex_table)

    metadata = {
        "run_dir": str(run_dir),
        "hit_radius_m": args.hit_radius,
        "final_budget": final_budget,
        "representative_map_seed": map_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "n_seeds": int(seed_summary["seed"].nunique()),
        "budgets": [int(v) for v in sorted(seed_summary["budget"].unique())],
    }
    (output_dir / "paper_export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    sys.stdout.write(text_table)
    print("Representative target-map seed: {}".format(map_seed))
    print("Wrote paper exports to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
