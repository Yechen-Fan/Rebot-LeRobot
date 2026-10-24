//! YAML 配置解析 + JointCfg / GripperCfg（暴露给 Python，与 reBotArm_control_py 的 dataclass 对齐）。
//!
//! 解析语义严格对照 `reBotArm_control_py/actuator/arm.py::load_cfg` 与
//! `gripper.py::load_cfg`（含十六进制 motor_id、嵌套 MIT / POS_VEL、各默认值）。

use pyo3::prelude::*;
use serde_yaml::Value;
use std::fs;
use std::path::Path;

/// 单个关节配置。字段集合与 arm.py 的 `JointCfg` dataclass 一致。
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct JointCfg {
    pub name: String,
    pub motor_id: u16,
    pub feedback_id: u16,
    pub model: String,
    pub vendor: String,
    pub kp: f64,
    pub kd: f64,
    pub vel_kp: f64,
    pub vel_ki: f64,
    pub pos_kp: f64,
    pub pos_ki: f64,
    pub vlim: f64,
}

#[pymethods]
impl JointCfg {
    fn __repr__(&self) -> String {
        format!(
            "JointCfg(name={:?}, motor_id={}, feedback_id={}, model={:?}, vendor={:?})",
            self.name, self.motor_id, self.feedback_id, self.model, self.vendor
        )
    }
}

/// 夹爪配置，字段集合与 gripper.py 的 `GripperCfg` dataclass 一致。
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct GripperCfg {
    pub name: String,
    pub motor_id: u16,
    pub feedback_id: u16,
    pub model: String,
    pub vendor: String,
    pub kp: f64,
    pub kd: f64,
    pub vel_kp: f64,
    pub vel_ki: f64,
    pub pos_kp: f64,
    pub pos_ki: f64,
    pub vlim: f64,
}

#[pymethods]
impl GripperCfg {
    fn __repr__(&self) -> String {
        format!(
            "GripperCfg(name={:?}, motor_id={}, model={:?}, vendor={:?})",
            self.name, self.motor_id, self.model, self.vendor
        )
    }
}

/// 解析后的机械臂配置。
pub struct ArmConfig {
    pub name: String,
    pub channel: String,
    pub rate: f64,
    pub joints: Vec<JointCfg>,
}

/// 解析后的夹爪配置。
pub struct GripperConfig {
    pub channel: String,
    pub gripper: GripperCfg,
}

// ------------------------------------------------------------------
// 取值辅助（兼容 int / float / "0x.." 十六进制字符串）
// ------------------------------------------------------------------

fn as_int(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => {
            let s = s.trim();
            if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                i64::from_str_radix(hex, 16).ok()
            } else {
                s.parse::<i64>().ok()
            }
        }
        _ => None,
    }
}

fn as_float(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64().or_else(|| n.as_i64().map(|i| i as f64)),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn as_str(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn get<'a>(map: &'a Value, key: &str) -> Option<&'a Value> {
    map.get(key)
}

fn read_yaml(path: &str) -> PyResult<Value> {
    let text = fs::read_to_string(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("无法读取 {path}: {e}")))?;
    serde_yaml::from_str(&text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("YAML 解析失败 {path}: {e}")))
}

// ------------------------------------------------------------------
// 机械臂配置
// ------------------------------------------------------------------

pub fn parse_arm_config(path: &str) -> PyResult<ArmConfig> {
    let data = read_yaml(path)?;

    let name = as_str(get(&data, "name")).unwrap_or_else(|| "reBotArm".to_string());
    let channel = as_str(get(&data, "channel")).unwrap_or_else(|| "/dev/ttyACM0".to_string());
    let rate = as_float(get(&data, "rate")).unwrap_or(500.0);

    let mut joints = Vec::new();
    if let Some(Value::Sequence(seq)) = data.get("joints") {
        for j in seq {
            let mit = get(j, "MIT");
            let pos_vel = get(j, "POS_VEL");
            let name = as_str(get(j, "name"))
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("joint 缺少 name"))?;
            let motor_id = as_int(get(j, "motor_id")).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("{name} 缺少/非法 motor_id"))
            })? as u16;
            let feedback_id = as_int(get(j, "feedback_id")).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("{name} 缺少/非法 feedback_id"))
            })? as u16;
            joints.push(JointCfg {
                name,
                motor_id,
                feedback_id,
                model: as_str(get(j, "model")).unwrap_or_else(|| "4340P".to_string()),
                vendor: as_str(get(j, "vendor"))
                    .unwrap_or_else(|| "damiao".to_string())
                    .to_lowercase(),
                kp: mit.and_then(|m| as_float(get(m, "kp"))).unwrap_or(0.0),
                kd: mit.and_then(|m| as_float(get(m, "kd"))).unwrap_or(0.0),
                vel_kp: pos_vel.and_then(|p| as_float(get(p, "vel_kp"))).unwrap_or(0.0),
                vel_ki: pos_vel.and_then(|p| as_float(get(p, "vel_ki"))).unwrap_or(0.0),
                pos_kp: pos_vel.and_then(|p| as_float(get(p, "pos_kp"))).unwrap_or(0.0),
                pos_ki: pos_vel.and_then(|p| as_float(get(p, "pos_ki"))).unwrap_or(0.0),
                vlim: pos_vel.and_then(|p| as_float(get(p, "vlim"))).unwrap_or(2.0),
            });
        }
    }

    Ok(ArmConfig {
        name,
        channel,
        rate,
        joints,
    })
}

// ------------------------------------------------------------------
// 夹爪配置
// ------------------------------------------------------------------

pub fn parse_gripper_config(path: &str) -> PyResult<GripperConfig> {
    let data = read_yaml(path)?;
    let channel = as_str(get(&data, "channel")).unwrap_or_else(|| "/dev/ttyACM0".to_string());

    // gripper.py: 取 data["gripper"][0]
    let gc = data
        .get("gripper")
        .and_then(|g| match g {
            Value::Sequence(seq) => seq.first(),
            other => Some(other),
        })
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("配置缺少 gripper"))?;

    let mit = get(gc, "MIT");
    let pos_vel = get(gc, "POS_VEL");
    let name = as_str(get(gc, "name"))
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("gripper 缺少 name"))?;
    let motor_id = as_int(get(gc, "motor_id")).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("{name} 缺少/非法 motor_id"))
    })? as u16;
    let feedback_id = as_int(get(gc, "feedback_id")).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("{name} 缺少/非法 feedback_id"))
    })? as u16;

    let gripper = GripperCfg {
        name,
        motor_id,
        feedback_id,
        model: as_str(get(gc, "model")).unwrap_or_else(|| "4310".to_string()),
        vendor: as_str(get(gc, "vendor"))
            .unwrap_or_else(|| "damiao".to_string())
            .to_lowercase(),
        kp: mit.and_then(|m| as_float(get(m, "kp"))).unwrap_or(18.0),
        kd: mit.and_then(|m| as_float(get(m, "kd"))).unwrap_or(2.0),
        vel_kp: pos_vel.and_then(|p| as_float(get(p, "vel_kp"))).unwrap_or(0.0008),
        vel_ki: pos_vel.and_then(|p| as_float(get(p, "vel_ki"))).unwrap_or(0.002),
        pos_kp: pos_vel.and_then(|p| as_float(get(p, "pos_kp"))).unwrap_or(50.0),
        pos_ki: pos_vel.and_then(|p| as_float(get(p, "pos_ki"))).unwrap_or(1.0),
        vlim: pos_vel.and_then(|p| as_float(get(p, "vlim"))).unwrap_or(3.0),
    };

    Ok(GripperConfig { channel, gripper })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_and_int_parsing() {
        // serde_yaml 对 0x01 的解析（YAML 1.1 风格）应给出整数 1，
        // 即便落到字符串分支，as_int 也能识别 "0x01"。
        let v: Value = serde_yaml::from_str("0x11").unwrap();
        assert_eq!(as_int(Some(&v)), Some(0x11));
        let v: Value = serde_yaml::from_str("7").unwrap();
        assert_eq!(as_int(Some(&v)), Some(7));
    }

    #[test]
    fn float_from_int_or_float() {
        let v: Value = serde_yaml::from_str("500").unwrap();
        assert_eq!(as_float(Some(&v)), Some(500.0));
        let v: Value = serde_yaml::from_str("0.0125").unwrap();
        assert_eq!(as_float(Some(&v)), Some(0.0125));
    }
}
