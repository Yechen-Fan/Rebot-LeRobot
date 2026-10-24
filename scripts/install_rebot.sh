#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lerobot" && pwd)"
PY="${PY:-python}"

if [[ ! -x "$(command -v "$PY")" ]]; then
  echo "Python not found: $PY" >&2
  exit 1
fi

"$PY" -m pip install \
  "pin==3.7.0" \
  "coal==3.0.1" \
  "cmeel-urdfdom==4.0.1" \
  "cmeel-tinyxml2==10.0.0" \
  "maturin==1.14.1" \
  "pybind11==3.0.4"

PY="$PY" bash "$ROOT/native/rebotarm_control_dm/build.sh" --wheel
"$PY" -m pip uninstall -y \
  rebotarm_control_rt \
  lerobot_robot_seeed_b601_rt \
  lerobot_robot_seeed_b601 \
  lerobot_robot_seeed_b601_rs \
  lerobot_teleoperator_quest3 \
  lerobot_teleoperator_switch || true
"$PY" - <<'PY'
import shutil
import sysconfig
from pathlib import Path

# Older RT installers copied native libraries outside pip's wheel manifest, so
# uninstalling the distribution can leave an importable namespace behind.
shutil.rmtree(Path(sysconfig.get_paths()["purelib"]) / "rebotarm_control_rt", ignore_errors=True)
PY
"$PY" -m pip install -e "$ROOT[rebot]"

"$PY" - <<'PY'
import inspect

import lerobot
import rebotarm_control_dm
from lerobot.robots import seeed_b601_dm
from lerobot.teleoperators import quest3, switch_joycon

print("LeRobot:", inspect.getfile(lerobot))
print("B601 DM:", inspect.getfile(seeed_b601_dm))
print("Quest3:", inspect.getfile(quest3))
print("Switch Joy-Con:", inspect.getfile(switch_joycon))
print("DM native backend:", inspect.getfile(rebotarm_control_dm))
PY

echo "reBot B601 DM stack installed."
