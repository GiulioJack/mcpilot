#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
source "$(dirname "$0")/common.sh"; cd "$HERE"; require_ros; set_full_delay 0 0.10
"$PYTHON_BIN" -u run_nn_paper_protocol.py --seeds 100,101,102,103,104 --depths 3 --budgets 5,15,25,35,45,55,65,75 --n-train-max 75 --n-eval-intermediate 40 --n-eval-final 40 --evaluation-seed "$EVAL_SEED" --hidden-dim 200 --opt-steps 1000 --batch-size 25 --bullet-name red_ball_friction --target-radius .05 --device "$DEVICE" --output "$RESULTS_ROOT/nn_nh3"
