#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
source "$(dirname "$0")/common.sh"; cd "$HERE"; require_ros; require_proxy
for seed in $SEEDS; do
  complete="$RESULTS_ROOT/main/mcpilot_complete/$seed"; mkdir -p "$complete"; set_residual_delay "$seed"
  "$PYTHON_BIN" -u test_mcpilcoros_tossing_with_panda_rbf_ker.py -seed "$seed" -device_name "$DEVICE" -num_threads 8 -num_particles 400 -model Speed -target_radius 0.05 -num_exp 5 -num_trials 1 -num_toss_test 40 -num_test_trials 1 -use_dropout True -flg_do_test True -evaluation_seed "$EVAL_SEED" -release_anticipation_seconds 0.10 -physical_delay_lower_seconds 0.12 -physical_delay_upper_seconds 0.13 -synthetic_ground_truth True -bullet_name red_ball_friction -flg_model_release_delay True -flg_apply_release_anticipation True -flg_load_t_delay_dist False -flg_use_cache False -flg_compensate_release_radial_velocity True -delay_loss trajectory_energy_score -delay_bo_a_lower_seconds 0.0 -delay_bo_a_upper_seconds 0.10 -delay_bo_b_lower_seconds 0.001 -delay_bo_b_upper_seconds 0.01 -delay_bo_particles 50 -delay_bo_init_points 10 -delay_bo_iterations 80 -delay_bo_seed "$seed" -policy_optimization_seed "$seed" -run_output_dir "$complete" 2>&1 | tee "$complete/run.log"
  for variant in fixed no_anticipation; do
    out="$RESULTS_ROOT/main/mcpilot_${variant}/$seed"; mkdir -p "$out"; copy_cache "$complete" "$out" 5
    if [[ "$variant" == fixed ]]; then ant=0.10; apply=True; else ant=0.0; apply=False; fi
    set_full_delay "$seed" "$ant"
    "$PYTHON_BIN" -u test_mcpilcoros_tossing_with_panda_rbf_ker.py -seed "$seed" -device_name "$DEVICE" -num_threads 8 -num_particles 400 -model Speed -target_radius 0.05 -num_exp 5 -num_trials 1 -num_toss_test 40 -num_test_trials 1 -use_dropout True -flg_do_test True -evaluation_seed "$EVAL_SEED" -release_anticipation_seconds "$ant" -physical_delay_lower_seconds 0.12 -physical_delay_upper_seconds 0.13 -synthetic_ground_truth True -bullet_name red_ball_friction -flg_model_release_delay False -flg_apply_release_anticipation "$apply" -flg_use_cache True -flg_compensate_release_radial_velocity True -policy_optimization_seed "$seed" -run_output_dir "$out" 2>&1 | tee "$out/run.log"
  done
done
