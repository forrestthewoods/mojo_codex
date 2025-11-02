#!/usr/bin/env bash

set -euo pipefail

# Usage: scripts/do/run_benchmark.sh [OUTPUT_DIR]
#
# Runs a CPU-based render benchmark suitable for headless droplets.
# The optional OUTPUT_DIR overrides the default outputs/<timestamp> directory.

export PATH="${HOME}/.pixi/bin:${PATH}"

OUTPUT_DIR="${1:-outputs/do_run}"

pixi run python -m src.benchmarks.render_bench \
    --backend cpu \
    --scene workbench \
    --duration 5 \
    --width 640 \
    --height 480 \
    --target-fps 60 \
    --output "${OUTPUT_DIR}" \
    --save-frames
