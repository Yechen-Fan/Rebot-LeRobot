//! rebotarm_control_dm._native —— 机械臂 actuator 层的 Rust 原生实现。
//!
//! 直接以 Cargo path 依赖 motorbridge 的 vendor crates（无 ctypes / C-ABI），
//! 经 PyO3 暴露与 `reBotArm_control_py.actuator` 一致的 Python 接口。

mod arm;
mod config;
mod gripper;
mod vendor;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::arm::RobotArm;
use crate::config::{parse_arm_config, parse_gripper_config, GripperCfg, JointCfg};
use crate::gripper::Gripper;

/// 解析机械臂配置，返回 {name, channel, rate, joints:[JointCfg]}（与 arm.py::load_cfg 对齐）。
#[pyfunction]
fn load_cfg<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyDict>> {
    let cfg = parse_arm_config(&path)?;
    let d = PyDict::new(py);
    d.set_item("name", cfg.name)?;
    d.set_item("channel", cfg.channel)?;
    d.set_item("rate", cfg.rate)?;
    let joints = PyList::new(py, cfg.joints.into_iter().map(|j| Py::new(py, j).unwrap()))?;
    d.set_item("joints", joints)?;
    Ok(d)
}

/// 解析夹爪配置，返回 {channel, gripper: GripperCfg}（与 gripper.py::load_cfg 对齐）。
#[pyfunction]
fn load_gripper_cfg<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyDict>> {
    let cfg = parse_gripper_config(&path)?;
    let d = PyDict::new(py);
    d.set_item("channel", cfg.channel)?;
    d.set_item("gripper", Py::new(py, cfg.gripper)?)?;
    Ok(d)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RobotArm>()?;
    m.add_class::<Gripper>()?;
    m.add_class::<JointCfg>()?;
    m.add_class::<GripperCfg>()?;
    m.add_function(wrap_pyfunction!(load_cfg, m)?)?;
    m.add_function(wrap_pyfunction!(load_gripper_cfg, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
