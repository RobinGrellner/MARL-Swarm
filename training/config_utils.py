#!/usr/bin/env python3
"""Utilities for expanding experiment configs from matrix parameters."""

import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from itertools import product


def expand_matrix_parameters(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Expand matrix_parameters from config into individual experiment entries.

    Takes a config with matrix_parameters and generates all combinations.

    Args:
        config: The config dictionary with matrix_parameters

    Returns:
        Dictionary of expanded experiments
    """
    experiments = {}
    matrix_params = config.get("matrix_parameters", {})
    defaults = config.get("defaults", {})

    # Skip metadata keys
    skip_keys = {"description", "note"}
    param_dict = {k: v for k, v in matrix_params.items() if k not in skip_keys}

    if not param_dict:
        # No matrix parameters, return existing experiments
        return config.get("experiments", {})

    # Get parameter names and values
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]

    # Generate all combinations
    combinations = product(*param_values)

    for combo in combinations:
        # Build experiment name and config
        parts = []
        env_config_overrides = {}
        train_config_overrides = {}

        for param_name, param_value in zip(param_names, combo):
            # Add to name
            if isinstance(param_value, list):
                # For lists like [64, 64], use shorthand like "64"
                parts.append(f"{param_name}{param_value[0]}")
            else:
                parts.append(f"{param_name}{param_value}")

            # Route to correct config section
            env_keys = {
                "num_agents", "num_pursuers", "world_size", "obs_model",
                "comm_radius", "torus", "kinematics", "v_max", "omega_max",
                "evader_strategy", "capture_radius", "evader_speed",
                "max_pursuers", "max_agents"
            }

            if param_name in env_keys:
                env_config_overrides[param_name] = param_value
            else:
                train_config_overrides[param_name] = param_value

        exp_name = "_".join(parts)

        # Build experiment config
        exp_config = {
            "description": f"Auto-generated: {', '.join(f'{n}={v}' for n, v in zip(param_names, combo))}",
            "env_config": {**defaults.get("env_config", {}), **env_config_overrides},
            "train_config": {**defaults.get("train_config", {}), **train_config_overrides},
        }

        experiments[exp_name] = exp_config

    return experiments


def load_and_expand_config(config_path: str) -> Dict[str, Any]:
    """
    Load a config file and expand matrix_parameters into individual experiments.

    Args:
        config_path: Path to the config JSON file

    Returns:
        Config dict with expanded experiments
    """
    with open(config_path) as f:
        config = json.load(f)

    # Expand matrix parameters
    expanded_experiments = expand_matrix_parameters(config)
    config["experiments"] = expanded_experiments

    return config


def count_experiments(config_path: str) -> int:
    """Count total experiments in a config file."""
    config = load_and_expand_config(config_path)
    return len(config.get("experiments", {}))
