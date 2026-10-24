#!/bin/bash
# rebotarm_control_dm 环境准备 + 安装（两阶段，不依赖 ROS）。
#
# 阶段 1：创建 conda/mamba 环境（Python 默认 3.12，可指定）+ Pinocchio C++3.x/Eigen/cmake/编译器
#   bash ./setup_env.sh --mamba [env_name] [py_version]     # Miniforge
#   bash ./setup_env.sh --conda [env_name] [py_version]     # Miniconda/Anaconda
#
# 阶段 2：在已激活的环境中安装本包（装 rust/maturin/pybind11 → build.sh --wheel）
#   mamba activate <env_name>      # 或 conda activate <env_name>
#   bash ./setup_env.sh --install
#
# 示例：
#   bash ./setup_env.sh --mamba rebot          # Python 3.12
#   bash ./setup_env.sh --mamba rebot 3.11      # 指定 Python 3.11
#   mamba activate rebot
#   bash ./setup_env.sh --install

set -uo pipefail

# 操作系统检查（仅提示，不阻断）
OS_NAME=$(uname -s)
if [[ "$OS_NAME" != "Linux" ]]; then
    echo "Unsupported operating system: $OS_NAME（本脚本仅支持 Linux）"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# conda-forge 原生依赖（内联，不用 yaml）。
# Pinocchio 钉死 3.x —— 4.0 重排了头文件布局，本项目按 3.x 编写。
CONDA_FORGE_DEPS=( "pinocchio>=3.2,<4" eigen cmake cxx-compiler )

# ── 创建环境（Python + Pinocchio C++ 等原生依赖） ────────────────────────────
create_environment() {
    local CONDA_CMD=$1
    local ENV_NAME=$2
    local PY_VER=$3

    conda deactivate 2>/dev/null || true

    if $CONDA_CMD env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Removing existing environment '$ENV_NAME'..."
        $CONDA_CMD env remove -n "$ENV_NAME" -y
    fi

    echo "Creating '$ENV_NAME' (Python $PY_VER) + 原生依赖 (Pinocchio C++ 3.x / Eigen / cmake / compiler)..."
    $CONDA_CMD create -y -n "$ENV_NAME" -c conda-forge "python=$PY_VER" "${CONDA_FORGE_DEPS[@]}" || {
        echo "[ERROR] 创建环境失败"; exit 1; }

    echo -e "\n[INFO] 已创建 $CONDA_CMD 环境 '$ENV_NAME'（Python $PY_VER）。\n"
    echo -e "\t1. 激活环境：       $CONDA_CMD activate $ENV_NAME"
    echo -e "\t2. 安装本包：       bash $SCRIPT_NAME --install"
    echo -e "\t3. 退出环境：       conda deactivate\n"
}

# ── 阶段 2：安装本包到当前已激活环境 ─────────────────────────────────────────
install_package() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
        echo "Error: 未激活目标环境。请先 conda/mamba activate <env_name> 再 --install。"
        exit 1
    fi
    echo "[INFO] 目标环境：$CONDA_DEFAULT_ENV  ($CONDA_PREFIX)"

    # 1) 串口权限（Damiao @ /dev/ttyACM0）
    if id -nG "$USER" | grep -qw dialout; then
        echo "[INFO] 用户已在 dialout 组。"
    else
        echo "[INFO] 将用户加入 dialout 组（串口访问）..."
        sudo usermod -aG dialout "$USER" || true
        echo "[WARN] 需重新登录/重启后生效。"
    fi

    # 2) Rust
    if command -v cargo >/dev/null 2>&1; then
        echo "[INFO] Rust 已安装：$(cargo --version)"
    else
        echo "[INFO] 安装 Rust（rustup）..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    command -v cargo >/dev/null 2>&1 || { echo "[ERROR] cargo 不可用"; exit 1; }

    # 3) Python 构建/运行依赖
    # 剥掉 PYTHONPATH：若 shell source 了 ROS，/opt/ros 的包会经 PYTHONPATH 泄漏进本环境，
    # 让 pip 打出无关的依赖冲突告警（如 generate-parameter-library-py）。清掉即干净。
    echo "[INFO] 安装 maturin / pybind11 / numpy / pyyaml ..."
    PYTHONPATH= python -m pip install -q --upgrade pip
    PYTHONPATH= python -m pip install -q maturin pybind11 numpy pyyaml || { echo "[ERROR] pip 安装失败"; exit 1; }

    # 4) 校验 Pinocchio C++（应来自当前 conda 环境）
    if [[ ! -f "$CONDA_PREFIX/include/pinocchio/spatial/se3-base.hpp" ]]; then
        echo "[ERROR] 当前环境未找到 Pinocchio C++ 3.x 头文件。"
        echo "        请用 --mamba/--conda 创建的环境，或手动：conda install -c conda-forge 'pinocchio>=3.2,<4'"
        exit 1
    fi

    # 5) 校验同级 motorbridge（Cargo path 依赖）
    if [[ ! -d "$SCRIPT_DIR/../motorbridge/motor_core" ]]; then
        echo "[ERROR] 未发现同级 ../motorbridge —— actuator 依赖它（Cargo path），请置于同级目录。"
        exit 1
    fi

    # 6) 构建并安装 wheel（CMake 编 _math + maturin 编 _native，自动探测 Pinocchio）
    echo "[INFO] 构建并安装（build.sh --wheel）..."
    PY="$(which python)" bash "$SCRIPT_DIR/build.sh" --wheel || { echo "[ERROR] build.sh 失败"; exit 1; }

    # 7) 自检
    echo "[INFO] 导入自检..."
    python - <<'PYEOF'
import rebotarm_control_dm as p
from rebotarm_control_dm.kinematics import load_robot_model
from rebotarm_control_dm.controllers import ArmEndPos  # noqa: F401
L = load_robot_model()
print("  package:", p.__version__, "| subpackages:", p.__all__)
print("  FK(neutral) shape:", L.fk(L.neutral())[2].shape)
print("[OK] rebotarm_control_dm 安装成功")
PYEOF
    echo -e "\n[INFO] 完成。跑测试：pip install pytest && bash ./run_tests.sh"
    echo "[INFO] 构建成功检查："
    echo "  conda activate rebot"
    echo "  python -c \"import rebotarm_control_dm._math, rebotarm_control_dm._native; print('ok')\""
}

# ── 解析参数 ─────────────────────────────────────────────────────────────────
MODE="${1:-}"
ENV_NAME="${2:-rebot}"
PY_VER="${3:-${PYVER:-3.12}}"

case "$MODE" in
  --conda|--mamba)
    if [[ "$MODE" == "--mamba" ]]; then
        BASES=("$HOME/miniforge3" "$HOME/mambaforge")
        CONDA_CMD="mamba"
    else
        BASES=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3")
        CONDA_CMD="conda"
    fi
    SOURCED=0
    for b in "${BASES[@]}"; do
        if [[ -f "$b/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            . "$b/etc/profile.d/conda.sh"
            [[ -f "$b/etc/profile.d/mamba.sh" ]] && . "$b/etc/profile.d/mamba.sh"
            SOURCED=1; break
        fi
    done
    if [[ "$SOURCED" -eq 0 ]]; then
        echo "未找到 conda/mamba。请安装 Miniforge3：https://github.com/conda-forge/miniforge"
        exit 1
    fi
    command -v "$CONDA_CMD" >/dev/null 2>&1 || CONDA_CMD="conda"
    create_environment "$CONDA_CMD" "$ENV_NAME" "$PY_VER"
    ;;
  --install)
    install_package
    ;;
  *)
    echo "用法："
    echo "  bash $SCRIPT_NAME --mamba [env_name] [py_version]   # 创建 mamba 环境（Miniforge）"
    echo "  bash $SCRIPT_NAME --conda [env_name] [py_version]   # 创建 conda 环境（Miniconda/Anaconda）"
    echo "  bash $SCRIPT_NAME --install                         # 在已激活环境中安装本包"
    echo ""
    echo "示例："
    echo "  bash $SCRIPT_NAME --mamba rebot        # Python 3.12（默认）"
    echo "  bash $SCRIPT_NAME --mamba rebot 3.11    # 指定 Python 3.11"
    echo "  mamba activate rebot && bash $SCRIPT_NAME --install"
    exit 1
    ;;
esac
