#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
source "$(dirname "$0")/common.sh"; cd "$HERE"; require_ros
for seed in $SEEDS; do set_full_delay "$seed" 0.10; mkdir -p "$RESULTS_ROOT/analytical/$seed"; "$PYTHON_BIN" -u testgravpolicy.py -seed "$seed" -target_radius .05 -bullet_name red_ball_friction -num_test_trials 1 -num_toss_test 40 -evaluation_seed "$EVAL_SEED" -output_dir "$RESULTS_ROOT/analytical/$seed" -release_anticipation_seconds .10 -physical_delay_min_ms 120 -physical_delay_max_ms 130; done
