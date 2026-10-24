#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_DATASET_ROOT="${PROJECT_ROOT}/data/bread_joint_new"
DEFAULT_REPO_ID="yechen/b601-bi-quest3-demo"
DEFAULT_LEFT_PORT="/dev/serial/by-path/pci-0000:0b:00.3-usb-0:2.2:1.0"
DEFAULT_RIGHT_PORT="/dev/serial/by-path/pci-0000:0b:00.3-usb-0:2.1:1.0"

usage() {
    cat <<EOF
Usage:
  bash scripts/replay_data.sh <episode> [options...]
  bash scripts/replay_data.sh <dataset_root> <episode> [options...]

Examples:
  bash scripts/replay_data.sh 0 --dry-run
  bash scripts/replay_data.sh 0
  bash scripts/replay_data.sh data/bread_joint 0 --yes
  bash scripts/replay_data.sh data/bread 0 --yes

Replay uses measured bimanual joint observations, with smooth startup/return.
Missing Home motion is never interpolated: only segment 0 runs by default.
Use --segment N to select another segment; --dry-run only inspects the data.
Legacy joint-only actions are supported; raw Cartesian targets are not replayed.

Environment overrides:
  PYTHON_BIN=/path/to/python
  DATASET_ROOT=${DEFAULT_DATASET_ROOT}
  REPO_ID=${DEFAULT_REPO_ID}
  LEFT_PORT=${DEFAULT_LEFT_PORT}
  RIGHT_PORT=${DEFAULT_RIGHT_PORT}
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
REPO_ID="${REPO_ID:-$DEFAULT_REPO_ID}"
LEFT_PORT="${LEFT_PORT:-$DEFAULT_LEFT_PORT}"
RIGHT_PORT="${RIGHT_PORT:-$DEFAULT_RIGHT_PORT}"

first_arg="$1"
shift
if [[ "$first_arg" =~ ^[0-9]+$ ]]; then
    EPISODE="$first_arg"
else
    DATASET_ROOT="$first_arg"
    if [[ $# -lt 1 ]]; then
        usage
        exit 2
    fi
    EPISODE="$1"
    shift
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ "${CONDA_DEFAULT_ENV:-}" != "base" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/yechen/miniforge3/envs/rebot_lerobot_env/bin/python" ]]; then
    PYTHON="/home/yechen/miniforge3/envs/rebot_lerobot_env/bin/python"
else
    PYTHON="python"
fi

export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PYTHONPATH:-}"

exec "$PYTHON" "${PROJECT_ROOT}/scripts/replay_dataset.py" \
    "$DATASET_ROOT" "$EPISODE" \
    --repo-id "$REPO_ID" \
    --left-port "$LEFT_PORT" \
    --right-port "$RIGHT_PORT" \
    "$@"
