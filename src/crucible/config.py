"""Project-level defaults loaded from crucible.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BUILTIN_DEFAULTS = {
    "db": "results.db",
    "tests": "tests",
    "docs": None,
    "hardware": "m4-pro-24gb",
    "max_drop_pp": 5.0,
    "max_refusal_shift_pp": None,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load optional YAML defaults. Missing config is not an error."""
    cfg_path = Path(path or "crucible.yaml")
    if not cfg_path.exists():
        return {}
    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a YAML mapping")
    return data


def _flat_config(config: dict[str, Any]) -> dict[str, Any]:
    flat = dict(config)
    gate = config.get("gate")
    if isinstance(gate, dict):
        if "max_drop_pp" in gate:
            flat["max_drop_pp"] = gate["max_drop_pp"]
        if "max_refusal_shift_pp" in gate:
            flat["max_refusal_shift_pp"] = gate["max_refusal_shift_pp"]
    return flat


def apply_config_defaults(args, config: dict[str, Any]) -> None:
    """Apply config only where argparse still holds Crucible's built-in default."""
    flat = _flat_config(config)
    for attr, builtin in BUILTIN_DEFAULTS.items():
        if not hasattr(args, attr) or attr not in flat:
            continue
        if getattr(args, attr) == builtin:
            setattr(args, attr, flat[attr])
