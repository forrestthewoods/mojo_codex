#!/usr/bin/env bash

set -euo pipefail

# This script prepares a fresh Ubuntu droplet for running the benchmark.
# It installs system packages required by MuJoCo's CPU renderer, Pixi, and Git.

sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    libosmesa6 \
    libosmesa6-dev \
    libgl1 \
    libglfw3 \
    libglew-dev \
    patchelf

# Install Pixi (cross-platform package manager used by this repo)
if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | bash
fi

export PATH="${HOME}/.pixi/bin:${PATH}"

# Ensure the workspace knows about Linux binaries (required on fresh installs)
pixi workspace platform add linux-64

# Set up the local Pixi environment (installs Python + dependencies)
pixi install

echo "Bootstrap complete. Remember to add '\$HOME/.pixi/bin' to PATH in future sessions."
