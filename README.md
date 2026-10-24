# Rebot_LeRobot

基于 LeRobot 的 reBot Arm B601 DM 单臂与双臂项目，支持 Quest 3、Nintendo Switch Joy-Con、FlexiTac 触觉、数据采集、ACT / Diffusion 训练、真机推理和数据回放。

此版本保留当前工作代码和配置模板，**不包含采集数据、训练权重、推理录像、运行报告或本机编译产物**。机器人控制必需的 URDF、网格资源、厂商通信依赖和许可证包含在源码中。`bread`、`egg` 是配置示例，不代表随项目附带对应数据或模型。

## 目录

```text
Rebot_LeRobot/
├── README.md
├── scripts/                          # 离线评估、回放和独立触觉测试
└── lerobot/
    ├── configs/                      # 遥操作、采集、训练、推理模板
    ├── install_rebot.sh              # 安装入口
    ├── native/                       # Rust / C++ 后端及机器人资源
    ├── pyproject.toml
    └── src/lerobot/                   # LeRobot 和机器人扩展
```

`data/`、`outputs/` 等目录在实际采集或训练时生成，并被 Git 忽略。配置中的相对数据、模型和输出路径以**项目根目录**为基准；运行命令前请切换到该目录。

## 安装

使用 Linux、Python 3.10 或更高版本，并准备 C++ 编译器、CMake 和 Rust/Cargo。训练和 CUDA 推理需要相应 GPU 环境。

先激活你准备用于本项目的 Python 环境，然后执行：

```bash
cd /home/yechen/Rebot_LeRobot
PY=python bash lerobot/install_rebot.sh
```

也可以把 `PY` 指定为所选环境的 Python 绝对路径。安装脚本会编译原生后端并安装 Python 包；本副本不包含预编译的 `_native`、`_math` 扩展和 wheel。

Python 包仍叫 `lerobot`，命令仍为 `lerobot-teleoperate`、`lerobot-record`、`lerobot-train`。目录复制不会自动切换现有环境中的可编辑安装；在所选环境安装后，确认导入位置属于本项目：

```bash
python -c "import lerobot; print(lerobot.__file__)"
python -c "import rebotarm_control_dm; print(rebotarm_control_dm.__file__)"
```

项目不附带本机机器人标定备份。设备路径、相机索引和触觉传感器映射保留为模板，应按实际硬件核对。

## 遥操作

以下命令都在项目根目录执行。

```bash
# 双臂 / 右臂 Quest 3
lerobot-teleoperate --config_path=lerobot/configs/tele_vr_bi.yaml
lerobot-teleoperate --config_path=lerobot/configs/tele_vr_right.yaml

# 双臂 / 右臂 Switch Joy-Con
lerobot-teleoperate --config_path=lerobot/configs/tele_switch_bi.yaml
lerobot-teleoperate --config_path=lerobot/configs/tele_switch_right.yaml
```

Quest 3 前扳机按住时启用控制，松开后保持；Switch 的按键与 IMU 配置见对应 YAML。当前控制代码中的释放保持、平移平滑和 IMU 处理均保留。

## 数据采集

```bash
# Switch 双臂，三路视觉
lerobot-record --config_path=lerobot/configs/collect_switch_bi.yaml

# Switch 双臂，三路视觉和四片 FlexiTac
lerobot-record --config_path=lerobot/configs/collect_switch_bi_flexitac.yaml

# Quest 3 双臂，三路视觉和 FlexiTac
lerobot-record --config_path=lerobot/configs/collect_vr_bi_flexitac.yaml
```

默认手动控制 episode：`c` 开始、`s` 停止并保存、退格键丢弃当前未保存的 episode、`Esc` 退出。修改 `dataset.single_task` 描述实际任务。

采集模板默认 `resume: false`，首次运行时目标目录应不存在。向已保存的完整数据集追加时加 `--resume=true`；仅有空目录而没有 `meta/info.json` 时不能恢复采集。

```bash
lerobot-record --config_path=lerobot/configs/collect_switch_bi_flexitac.yaml --resume=true
```

`dataset.repo_id` 保留为示例名称，本地采集不要求其已存在于 Hugging Face；采集配置默认关闭上传。双臂采集配置的 `dataset.fps: 30` 同时控制遥操作更新频率，底层 `rt_rate: 100.0` 是执行器控制频率。

## 训练

先采集或准备与配置输入、动作格式一致的本地数据集，再训练。以下模板保留原有训练参数：

| 配置 | 数据模板 | 用途 |
| --- | --- | --- |
| `lerobot/configs/train_act.yaml` | `data/bread` | ACT，视觉与机器人状态 |
| `lerobot/configs/train_act_flexitac.yaml` | `data/egg_auto_zero` | ACT，视觉、触觉与机器人状态 |
| `lerobot/configs/train_diffusion_flexitac.yaml` | `data/egg_auto_zero` | Diffusion，视觉、触觉与机器人状态 |

```bash
lerobot-train --config_path=lerobot/configs/train_act.yaml
lerobot-train --config_path=lerobot/configs/train_act_flexitac.yaml
lerobot-train --config_path=lerobot/configs/train_diffusion_flexitac.yaml
```

只训练到 20,000 step 时，可临时覆盖：

```bash
lerobot-train --config_path=lerobot/configs/train_act_flexitac.yaml \
  --steps=20000 --save_freq=10000
```

`dataset.root` 指向实际准备的数据，`output_dir` 指向新的训练输出目录。项目不附带这些数据和 checkpoint。

## 真机推理

先准备完整的 `pretrained_model` 目录（模型配置、权重及对应预处理和后处理参数），再把 `policy.pretrained_path` 指向它。模板中的 checkpoint 路径只是预期位置，并非项目附带的模型。

```bash
# 提供与模板一致的双臂末端位姿模型后运行
lerobot-record --config_path=lerobot/configs/eval_act_bread.yaml

# 提供与模板一致的视觉触觉模型后运行
lerobot-record --config_path=lerobot/configs/eval_act_egg_flexitac.yaml
```

所有 `eval_*.yaml` 默认 `recording_enabled: false`：不创建数据集或启动视频编码，仍可按配置预览。`dataset.fps` 控制推理循环频率，归一化参数从 checkpoint 加载。`c` 开始、`s` 停止；支持 Home 的策略模式下，`h` 返回本次连接的启动关节位置并等待再次按 `c`；`Esc` 退出。

需要保存推理过程时显式开启录制：

```bash
# 首次保存：dataset.root 应为尚不存在的新目录
lerobot-record --config_path=lerobot/configs/eval_act_bread.yaml \
  --recording_enabled=true --resume=false

# 向已有完整记录追加
lerobot-record --config_path=lerobot/configs/eval_act_bread.yaml \
  --recording_enabled=true --resume=true
```

`policy_motion` 保留关节参考速度、加速度和电机速度上限约束。Home 是返回启动关节位置的控制流程，不包含桌面避障规划。

### egg 专用交接配置

`eval_act_egg_flexitac.yaml` 保留左爪二值解码、到位后合爪，以及当前任务的单次交接释放配置。释放依据来自示范的成对双臂位姿、两爪闭合反馈和稳定时间；张开期间保持双臂位置及右爪闭合，确认左爪张开后恢复末端预测，本轮保持左爪张开。重新开始一轮时重置。

匹配使用原有进入阈值和 `match_exit_scale: 1.2` 的退出余量，稳定 1 秒的要求仍保留。该规则假定当前机器人安装位置和抓取偏移，不检测杯子是否存在。更换任务、安装方式或模型时需核对这些参数；不需要规则时移除 `policy_motion.handover_release`。位姿参数保留在模板内，实际示范数据和训练权重未附带。

## FlexiTac 触觉

每片传感器的原始数据经基线校正后生成接触热力图。采集与推理共用处理流程，显示、模型输入和新数据保存采用同一结果；新格式逐帧保存原始值及实际基线，供训练读取时重算核对。

自动校零包含释放后稳定校正，以及持续张开、假定无接触状态下的基线维护。持续张开维护不能区分长期稳定接触与漂移；需要保留张开状态下的真实持续压力时关闭 `flexitac_auto_zero`。采集和推理没有 B 键手动校零，B 仅用于独立测试：

```bash
python scripts/test_flexitac.py --sensor left_inner
python scripts/test_flexitac.py --sensor left_outer
python scripts/test_flexitac.py --sensor right_inner
python scripts/test_flexitac.py --sensor right_outer
```

旧数据缺少原始触觉或逐帧基线时，不能补算同等处理；改变传感器映射或预处理版本时应使用独立数据目录。

## 回放与离线评估

先检查自己的已有数据；项目没有可直接回放的示例数据：

```bash
bash scripts/replay_data.sh data/bread 0 --dry-run
bash scripts/replay_data.sh data/bread 0
```

回放优先使用双臂实际关节位置，保留静止帧并对 Home 等缺失轨迹分段，不跨段插值。默认请求 1 倍速度，仍受速度、加速度和起步衔接约束。支持 `--playback-speed`、`--segment` 等参数，设备设置沿用 `tele_switch_bi.yaml`。

ACT 离线检查工具位于 `scripts/evaluate_act_dataset.py`，用 `--help` 查看参数。没有数据或模型时不能完成模型评估。

## 版本管理与许可证

在项目根目录独立管理 Git，内层 `lerobot/` 是普通源码目录。数据、权重、报告、缓存和构建产物不进入 Git；机器人网格直接随源码保存，不要求 Git LFS。

保留上游和第三方组件的作者、许可证与来源信息。条款见 [LeRobot 许可证](lerobot/LICENSE)、[原生控制后端许可证](lerobot/native/rebotarm_control_dm/LICENSE) 和 [motorbridge 许可证](lerobot/native/motorbridge/LICENSE)。
