#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$TASK_ROOT/.deps:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/kaixin/miniconda3/envs/ros_jazzy}"
exec "${CALIBRATION_PYTHON:-/home/kaixin/miniconda3/envs/ros_jazzy/bin/python}" "$@"
