#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
source "$(dirname "$0")/common.sh"; require_ros
for cfg in learned_anticipation fixed_anticipation; do
  set_full_delay 0 0.0
  rosrun mcpilot sac_run_budget_study.py --config "$PKG/config/sac/${cfg}.yaml" --set "experiment.output_dir=$RESULTS_ROOT/sac" --set "experiment.name=$cfg" --set "sac.device=$DEVICE"
  rosrun mcpilot sac_paper_results.py "$RESULTS_ROOT/sac/$cfg" --hit-radius 0.05
done
