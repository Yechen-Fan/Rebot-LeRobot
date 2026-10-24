#!/usr/bin/env bash
# 干净地跑测试：剥掉 ROS 经 $PYTHONPATH 泄漏进来的 site-packages，并关闭第三方 pytest
# 插件自动加载（否则 ROS 的 launch_testing 等插件会被 pytest 自动载入、缺依赖而崩）。
#
# 用法：
#   bash ./run_tests.sh                 # 跑 tests/python
#   bash ./run_tests.sh tests/python/test_dynamics.py -v
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=("$HERE/tests/python")
exec env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest "${ARGS[@]}" -q
