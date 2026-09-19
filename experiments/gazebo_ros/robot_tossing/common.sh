#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
PKG="$(rospack find mcpilot)"
HERE="$PKG/experiments/gazebo_ros/robot_tossing"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
RESULTS_ROOT="${RESULTS_ROOT:-$PKG/results}"
SEEDS="${SEEDS:-100 101 102 103 104}"
EVAL_SEED="${EVAL_SEED:-20260825}"
mkdir -p "$RESULTS_ROOT"
require_ros(){ rosservice info /tossing_experiment_testing >/dev/null 2>&1 || { echo 'Missing /tossing_experiment_testing. Start the ROS simulation and tossing_experiment_script.py.' >&2; exit 1; }; }
require_proxy(){ rostopic info /tossing_exp_proxy_in 2>/dev/null | grep -q MC_PILCO_experiment_proxy || { echo 'Start MC_PILCO_ros_tossing_proxy.py.' >&2; exit 1; }; }
set_full_delay(){ local seed="$1" ant="$2"; rosparam set /tossing/release_delay/anticipation_seconds "$ant"; rosparam set /tossing/release_delay/synthetic_ground_truth/enabled true; rosparam set /tossing/release_delay/synthetic_ground_truth/distribution uniform; rosparam set /tossing/release_delay/synthetic_ground_truth/min_ms 120.0; rosparam set /tossing/release_delay/synthetic_ground_truth/max_ms 130.0; rosparam set /tossing/release_delay/synthetic_ground_truth/compensate_command_anticipation true; rosparam set /tossing/release_delay/synthetic_ground_truth/seed "$seed"; local r; r=$(rosparam get /tossing/release_delay/synthetic_ground_truth/reset_id 2>/dev/null || echo 0); rosparam set /tossing/release_delay/synthetic_ground_truth/reset_id $((r+1)); }
set_residual_delay(){ local seed="$1"; rosparam set /tossing/release_delay/anticipation_seconds 0.10; rosparam set /tossing/release_delay/synthetic_ground_truth/enabled true; rosparam set /tossing/release_delay/synthetic_ground_truth/distribution uniform; rosparam set /tossing/release_delay/synthetic_ground_truth/min_ms 20.0; rosparam set /tossing/release_delay/synthetic_ground_truth/max_ms 30.0; rosparam set /tossing/release_delay/synthetic_ground_truth/compensate_command_anticipation false; rosparam set /tossing/release_delay/synthetic_ground_truth/seed "$seed"; local r; r=$(rosparam get /tossing/release_delay/synthetic_ground_truth/reset_id 2>/dev/null || echo 0); rosparam set /tossing/release_delay/synthetic_ground_truth/reset_id $((r+1)); }
source_delay(){ "$PYTHON_BIN" - "$1" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]+'/delay_compensation_config.json'))
print(j['effective_anticipation_seconds'], j['model_residual_distribution']['upper'])
PY
}
copy_cache(){ mkdir -p "$2/cache_tossing_trials"; for i in $(seq 0 $(($3-1))); do cp "$1/cache_tossing_trials/trial_${i}.pkl" "$2/cache_tossing_trials/"; done; }
