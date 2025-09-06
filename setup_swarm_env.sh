#!/usr/bin/env bash
# This script sets up a Python virtual environment and installs the
# dependencies required to run the deep_rl_for_swarms PettingZoo
# environments and training scripts.  It assumes that Python 3.12 is
# installed and available on your PATH as `python3.12`.  If your
# Python executable has a different name (e.g. `python` or
# `python3`), adjust the command accordingly.

set -e

VENV_DIR="../venvs/MARL"

echo "Creating Python 3.12 virtual environment in $VENV_DIR..."
python3.12 -m venv "$VENV_DIR"

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install required packages.  We specify version constraints that
# support Python 3.12.  Remove or adjust the versions if newer
# releases become available.
echo "Installing dependencies..."
python -m pip install \
  gymnasium>=0.29 \
  pettingzoo>=1.25 \
  stable-baselines3>=2.7 \
  sb3-contrib>=2.7 \
  supersuit>=3.9 \
  pygame>=2.5 

echo "All packages installed successfully.  To activate the virtual environment in the future run:"
echo "  source $VENV_DIR/bin/activate"