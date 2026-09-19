#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the target-conditioned SAC interaction-budget experiment."""
from __future__ import annotations

import argparse
import os
import sys

from sac_tossing.config import apply_overrides, load_config
from sac_tossing.study import run_study


def default_config_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "sac", "paper.yaml")
    )


def initialize_ros_if_needed(config):
    backend = str(config["experiment"].get("backend", config["environment"].get("backend", "ros")))
    if backend == "ros":
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("sac_budget_study", anonymous=True, disable_signals=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML key; may be repeated.",
    )
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    initialize_ros_if_needed(config)
    run_root = run_study(config)
    print("SAC study completed: {}".format(run_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
