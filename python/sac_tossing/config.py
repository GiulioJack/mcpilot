# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration loading and command-line overrides."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, Mapping

import yaml


def deep_update(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML value must be a mapping")
    data.setdefault("experiment", {})
    data.setdefault("environment", {})
    data.setdefault("sac", {})
    return data


def save_config(config: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError("Override must have KEY=VALUE form: {!r}".format(item))
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: Dict[str, Any] = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if child is None:
                child = {}
                cursor[part] = child
            if not isinstance(child, dict):
                raise ValueError("Cannot descend through non-mapping key {}".format(part))
            cursor = child
        cursor[parts[-1]] = value
    return result
