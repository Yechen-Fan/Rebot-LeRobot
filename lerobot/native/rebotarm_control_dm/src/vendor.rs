//! 厂商统一派发层：在 Rust 内直接调用 motorbridge 的 vendor crates，
//! 镜像 `motor_abi` 的 ControllerInner / MotorHandleInner 枚举派发与状态归一化。
//!
//! 与 motor_abi 的对应：
//!   - lifecycle  ← motor_lifecycle_ffi.rs
//!   - control    ← motor_control_ffi.rs
//!   - register   ← motor_register_ffi.rs
//!   - get_state  ← state_ffi.rs（含 deg→rad、robstride 故障位打包）

use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Duration;

use motor_vendor_damiao::{ControlMode as DamiaoMode, DamiaoController, DamiaoMotor};
use motor_vendor_hightorque::{HightorqueController, HightorqueMotor};
use motor_vendor_myactuator::{MyActuatorController, MyActuatorMotor};
use motor_vendor_robstride::{
    ControlMode as RsMode, ParameterValue, RobstrideController, RobstrideMotor,
};

/// 归一化后的电机状态（位置/速度单位为 rad、rad/s；力矩 N·m）。
#[derive(Clone, Copy, Debug, Default)]
pub struct NormState {
    pub status_code: u8,
    pub pos: f64,
    pub vel: f64,
    pub torq: f64,
}

// ------------------------------------------------------------------
// 模式映射（照搬 motor_abi/src/lib.rs）
// ------------------------------------------------------------------

fn to_damiao_mode(mode: u32) -> Result<DamiaoMode, String> {
    match mode {
        1 => Ok(DamiaoMode::Mit),
        2 => Ok(DamiaoMode::PosVel),
        3 => Ok(DamiaoMode::Vel),
        4 => Ok(DamiaoMode::ForcePos),
        _ => Err("Damiao mode 必须为 1(MIT)/2(POS_VEL)/3(VEL)/4(FORCE_POS)".to_string()),
    }
}

fn to_robstride_mode(mode: u32) -> Result<RsMode, String> {
    match mode {
        1 => Ok(RsMode::Mit),
        2 => Ok(RsMode::Position),
        3 => Ok(RsMode::Velocity),
        5 => Ok(RsMode::PositionCsp),
        _ => Err("RobStride mode 必须为 1(MIT)/2(POSITION)/3(VELOCITY)/5(POSITION-CSP)".to_string()),
    }
}

fn validate_myactuator_mode(mode: u32) -> Result<(), String> {
    match mode {
        1..=3 => Ok(()),
        _ => Err("MyActuator mode 必须为 1(CURRENT)/2(POSITION)/3(VELOCITY)".to_string()),
    }
}

// ------------------------------------------------------------------
// UniController
// ------------------------------------------------------------------

pub enum UniController {
    Damiao(DamiaoController),
    MyActuator(MyActuatorController),
    Robstride(RobstrideController),
    Hightorque(HightorqueController),
}

pub enum UniMotor {
    Damiao(Arc<DamiaoMotor>),
    MyActuator(Arc<MyActuatorMotor>),
    Robstride(Arc<RobstrideMotor>),
    Hightorque(Arc<HightorqueMotor>),
}

impl UniController {
    /// 按 channel 与 vendor 创建控制器。
    /// `/dev/tty*` ⇒ Damiao 串口桥（仅 Damiao 支持）；否则 socketcan。
    pub fn new(channel: &str, vendor: &str) -> Result<Self, String> {
        let is_serial = channel.starts_with("/dev/tty");
        match vendor {
            "damiao" => {
                let c = if is_serial {
                    DamiaoController::new_dm_serial(channel, 921600)
                } else {
                    DamiaoController::new_socketcan(channel)
                }
                .map_err(|e| e.to_string())?;
                Ok(UniController::Damiao(c))
            }
            other if is_serial => Err(format!(
                "串口桥 {channel} 仅支持 damiao；vendor={other} 需使用 CAN 通道"
            )),
            "myactuator" => MyActuatorController::new_socketcan(channel)
                .map(UniController::MyActuator)
                .map_err(|e| e.to_string()),
            "robstride" => RobstrideController::new_socketcan(channel)
                .map(UniController::Robstride)
                .map_err(|e| e.to_string()),
            "hightorque" => HightorqueController::new_socketcan(channel)
                .map(UniController::Hightorque)
                .map_err(|e| e.to_string()),
            other => Err(format!("不支持的 vendor: {other}")),
        }
    }

    pub fn add_motor(
        &self,
        motor_id: u16,
        feedback_id: u16,
        model: &str,
    ) -> Result<UniMotor, String> {
        match self {
            UniController::Damiao(c) => c
                .add_motor(motor_id, feedback_id, model)
                .map(UniMotor::Damiao)
                .map_err(|e| e.to_string()),
            UniController::MyActuator(c) => c
                .add_motor(motor_id, feedback_id, model)
                .map(UniMotor::MyActuator)
                .map_err(|e| e.to_string()),
            UniController::Robstride(c) => c
                .add_motor(motor_id, feedback_id, model)
                .map(UniMotor::Robstride)
                .map_err(|e| e.to_string()),
            UniController::Hightorque(c) => c
                .add_motor(motor_id, feedback_id, model)
                .map(UniMotor::Hightorque)
                .map_err(|e| e.to_string()),
        }
    }

    pub fn enable_all(&self) -> Result<(), String> {
        match self {
            UniController::Damiao(c) => c.enable_all().map_err(|e| e.to_string()),
            UniController::MyActuator(c) => c.enable_all().map_err(|e| e.to_string()),
            UniController::Robstride(c) => c.enable_all().map_err(|e| e.to_string()),
            UniController::Hightorque(c) => c.enable_all().map_err(|e| e.to_string()),
        }
    }

    pub fn disable_all(&self) -> Result<(), String> {
        match self {
            UniController::Damiao(c) => c.disable_all().map_err(|e| e.to_string()),
            UniController::MyActuator(c) => c.disable_all().map_err(|e| e.to_string()),
            UniController::Robstride(c) => c.disable_all().map_err(|e| e.to_string()),
            UniController::Hightorque(c) => c.disable_all().map_err(|e| e.to_string()),
        }
    }

    pub fn poll_feedback_once(&self) -> Result<(), String> {
        match self {
            UniController::Damiao(c) => c.poll_feedback_once().map_err(|e| e.to_string()),
            UniController::MyActuator(c) => c.poll_feedback_once().map_err(|e| e.to_string()),
            UniController::Robstride(c) => c.poll_feedback_once().map_err(|e| e.to_string()),
            UniController::Hightorque(c) => c.poll_feedback_once().map_err(|e| e.to_string()),
        }
    }

    pub fn shutdown(&self) -> Result<(), String> {
        match self {
            UniController::Damiao(c) => c.shutdown().map_err(|e| e.to_string()),
            UniController::MyActuator(c) => c.shutdown().map_err(|e| e.to_string()),
            UniController::Robstride(c) => c.shutdown().map_err(|e| e.to_string()),
            UniController::Hightorque(c) => c.shutdown().map_err(|e| e.to_string()),
        }
    }

    pub fn close_bus(&self) -> Result<(), String> {
        match self {
            UniController::Damiao(c) => c.close_bus().map_err(|e| e.to_string()),
            UniController::MyActuator(c) => c.close_bus().map_err(|e| e.to_string()),
            UniController::Robstride(c) => c.close_bus().map_err(|e| e.to_string()),
            UniController::Hightorque(c) => c.close_bus().map_err(|e| e.to_string()),
        }
    }
}

// ------------------------------------------------------------------
// UniMotor —— 每电机操作
// ------------------------------------------------------------------

impl UniMotor {
    // enable/disable per-motor 保留以对齐 vendor API（RobotArm 走控制器级 enable_all）。
    #[allow(dead_code)]
    pub fn enable(&self) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.enable().map_err(|e| e.to_string()),
            UniMotor::MyActuator(m) => m.release_brake().map_err(|e| e.to_string()),
            UniMotor::Robstride(m) => m.enable().map_err(|e| e.to_string()),
            UniMotor::Hightorque(m) => m.enable().map_err(|e| e.to_string()),
        }
    }

    #[allow(dead_code)]
    pub fn disable(&self) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.disable().map_err(|e| e.to_string()),
            UniMotor::MyActuator(m) => m.shutdown_motor().map_err(|e| e.to_string()),
            UniMotor::Robstride(m) => m.disable().map_err(|e| e.to_string()),
            UniMotor::Hightorque(m) => m.disable().map_err(|e| e.to_string()),
        }
    }

    pub fn set_zero(&self) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.set_zero_position().map_err(|e| e.to_string()),
            UniMotor::MyActuator(_) => {
                Err("MyActuator 不支持 set_zero_position".to_string())
            }
            UniMotor::Robstride(m) => m.set_zero_position().map_err(|e| e.to_string()),
            UniMotor::Hightorque(m) => m.set_zero_position().map_err(|e| e.to_string()),
        }
    }

    pub fn ensure_mode(&self, mode: u32, timeout_ms: u32) -> Result<(), String> {
        let timeout = Duration::from_millis(timeout_ms as u64);
        match self {
            UniMotor::Damiao(m) => {
                let dm = to_damiao_mode(mode)?;
                m.ensure_control_mode(dm, timeout).map_err(|e| e.to_string())
            }
            UniMotor::MyActuator(_) => validate_myactuator_mode(mode),
            UniMotor::Robstride(m) => {
                let rm = to_robstride_mode(mode)?;
                m.ensure_control_mode(rm, timeout).map_err(|e| e.to_string())
            }
            UniMotor::Hightorque(m) => {
                m.ensure_control_mode(mode, timeout).map_err(|e| e.to_string())
            }
        }
    }

    pub fn send_mit(&self, pos: f32, vel: f32, kp: f32, kd: f32, tau: f32) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m
                .send_cmd_mit(pos, vel, kp, kd, tau)
                .map_err(|e| e.to_string()),
            UniMotor::MyActuator(_) => {
                Err("MyActuator 不支持 send_mit；请用 pos_vel 或 vel".to_string())
            }
            UniMotor::Robstride(m) => m
                .send_cmd_mit(pos, vel, kp, kd, tau)
                .map_err(|e| e.to_string()),
            UniMotor::Hightorque(m) => m
                .send_cmd_mit(pos, vel, kp, kd, tau)
                .map_err(|e| e.to_string()),
        }
    }

    pub fn send_pos_vel(&self, pos: f32, vlim: f32) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.send_cmd_pos_vel(pos, vlim).map_err(|e| e.to_string()),
            UniMotor::MyActuator(m) => m
                .send_position_absolute_setpoint(pos * (180.0 / PI), vlim * (180.0 / PI))
                .map_err(|e| e.to_string()),
            UniMotor::Robstride(m) => {
                // 统一 POS_VEL → RobStride 原生 Position 模式
                m.set_mode(RsMode::Position).map_err(|e| e.to_string())?;
                let v = vlim.abs();
                if v.is_finite() && v > 0.0 {
                    m.write_parameter(0x7017, ParameterValue::F32(v))
                        .map_err(|e| e.to_string())?;
                }
                m.write_parameter(0x7016, ParameterValue::F32(pos))
                    .map_err(|e| e.to_string())
            }
            UniMotor::Hightorque(m) => m.send_cmd_pos_vel(pos, vlim).map_err(|e| e.to_string()),
        }
    }

    pub fn send_force_pos(&self, pos: f32, vlim: f32, torque_limit_ratio: f32) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m
                .send_cmd_force_pos(pos, vlim, torque_limit_ratio)
                .map_err(|e| e.to_string()),
            UniMotor::Hightorque(m) => m
                .send_cmd_force_pos(pos, vlim, torque_limit_ratio)
                .map_err(|e| e.to_string()),
            _ => Err("send_force_pos 仅 Damiao / HighTorque 支持".to_string()),
        }
    }

    pub fn send_vel(&self, vel: f32) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.send_cmd_vel(vel).map_err(|e| e.to_string()),
            UniMotor::MyActuator(m) => m
                .send_velocity_setpoint(vel * (180.0 / PI))
                .map_err(|e| e.to_string()),
            UniMotor::Robstride(m) => m.set_velocity_target(vel).map_err(|e| e.to_string()),
            UniMotor::Hightorque(m) => m.send_cmd_vel(vel).map_err(|e| e.to_string()),
        }
    }

    pub fn write_register_f32(&self, rid: u8, value: f32) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.write_register_f32(rid, value).map_err(|e| e.to_string()),
            _ => Err("寄存器写入仅 Damiao 支持".to_string()),
        }
    }

    pub fn get_register_f32(&self, rid: u8, timeout_ms: u64) -> Result<f32, String> {
        match self {
            UniMotor::Damiao(m) => m
                .get_register_f32(rid, Duration::from_millis(timeout_ms))
                .map_err(|e| e.to_string()),
            _ => Err("寄存器读取仅 Damiao 支持".to_string()),
        }
    }

    /// 设置电机侧 CAN 超时看门狗（ms）：超时未收到控制帧则电机自动停机。
    /// 镜像 motor_abi 的 set_can_timeout_ms：Damiao 写寄存器 9（值=ms*20），RobStride 写 0x7028。
    /// timeout_ms=0 表示禁用看门狗。
    pub fn set_can_timeout_ms(&self, timeout_ms: u32) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m
                .write_register_u32(9, timeout_ms.saturating_mul(20))
                .map_err(|e| e.to_string()),
            UniMotor::Robstride(m) => m
                .write_parameter(0x7028, ParameterValue::U32(timeout_ms))
                .map_err(|e| e.to_string()),
            _ => Err("set_can_timeout_ms 仅 Damiao / RobStride 支持".to_string()),
        }
    }

    pub fn request_feedback(&self) -> Result<(), String> {
        match self {
            UniMotor::Damiao(m) => m.request_motor_feedback().map_err(|e| e.to_string()),
            UniMotor::MyActuator(m) => m.request_status().map_err(|e| e.to_string()),
            // RobStride 无单帧状态请求命令；保持 no-op，避免阻塞控制循环。
            UniMotor::Robstride(_) => Ok(()),
            UniMotor::Hightorque(m) => m
                .request_motor_feedback(Duration::from_millis(500))
                .map(|_| ())
                .map_err(|e| e.to_string()),
        }
    }

    /// 读取最新反馈快照并归一化（无数据返回 None）。照搬 state_ffi.rs。
    pub fn get_state(&self) -> Option<NormState> {
        match self {
            UniMotor::Damiao(m) => m.latest_state().map(|s| NormState {
                status_code: s.status_code,
                pos: s.pos as f64,
                vel: s.vel as f64,
                torq: s.torq as f64,
            }),
            UniMotor::MyActuator(m) => m.latest_state().map(|s| NormState {
                status_code: s.command,
                pos: (s.shaft_angle_deg * (PI / 180.0)) as f64,
                vel: (s.speed_dps * (PI / 180.0)) as f64,
                torq: s.current_a as f64,
            }),
            UniMotor::Robstride(m) => m.latest_state().map(|s| {
                let mut status = 0u8;
                if s.uncalibrated {
                    status |= 1 << 5;
                }
                if s.stall {
                    status |= 1 << 4;
                }
                if s.magnetic_encoder_fault {
                    status |= 1 << 3;
                }
                if s.overtemperature {
                    status |= 1 << 2;
                }
                if s.overcurrent {
                    status |= 1 << 1;
                }
                if s.undervoltage {
                    status |= 1;
                }
                NormState {
                    status_code: status,
                    pos: s.position as f64,
                    vel: s.velocity as f64,
                    torq: s.torque as f64,
                }
            }),
            UniMotor::Hightorque(m) => m.latest_state().map(|s| NormState {
                status_code: s.status_code,
                pos: s.pos as f64,
                vel: s.vel as f64,
                torq: s.torq as f64,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_mapping() {
        assert!(matches!(to_damiao_mode(1), Ok(DamiaoMode::Mit)));
        assert!(matches!(to_damiao_mode(2), Ok(DamiaoMode::PosVel)));
        assert!(matches!(to_damiao_mode(3), Ok(DamiaoMode::Vel)));
        assert!(to_damiao_mode(9).is_err());
        assert!(validate_myactuator_mode(2).is_ok());
        assert!(validate_myactuator_mode(9).is_err());
        assert!(matches!(to_robstride_mode(5), Ok(RsMode::PositionCsp)));
    }
}
