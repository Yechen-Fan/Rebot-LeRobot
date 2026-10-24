use motor_core::error::{MotorError, Result};

pub const MIT_P_MIN: f32 = -12.57;
pub const MIT_P_MAX: f32 = 12.57;
pub const MIT_V_MIN: f32 = -33.0;
pub const MIT_V_MAX: f32 = 33.0;
pub const MIT_T_MIN: f32 = -14.0;
pub const MIT_T_MAX: f32 = 14.0;
pub const MIT_KP_MIN: f32 = 0.0;
pub const MIT_KP_MAX: f32 = 500.0;
pub const MIT_KD_MIN: f32 = 0.0;
pub const MIT_KD_MAX: f32 = 5.0;

pub const MODE_MIT: u8 = 0;
pub const MODE_POSITION: u8 = 1;
pub const MODE_VELOCITY: u8 = 2;

pub const PROTOCOL_PRIVATE: u8 = 0;
pub const PROTOCOL_CANOPEN: u8 = 1;
pub const PROTOCOL_MIT: u8 = 2;

pub const PARAM_READ_TYPE: u16 = 3;
pub const PARAM_WRITE_TYPE: u16 = 4;

#[derive(Debug, Clone, Copy)]
pub struct MitFeedback {
    pub motor_id: u8,
    pub position_rad: f32,
    pub velocity_rad_s: f32,
    pub torque_nm: f32,
    pub mode_state: u8,
    pub has_fault: bool,
    pub has_warning: bool,
    pub winding_temp_c: f32,
}

pub fn validate_node_id(node_id: u16) -> Result<()> {
    if node_id == 0 || node_id > 127 {
        return Err(MotorError::InvalidArgument(format!(
            "invalid RobStride MIT CAN id {node_id}, expected 1..127"
        )));
    }
    Ok(())
}

pub fn validate_host_id(host_id: u16) -> Result<()> {
    if host_id == 0 || host_id > 127 {
        return Err(MotorError::InvalidArgument(format!(
            "invalid RobStride MIT host id {host_id}, expected 1..127"
        )));
    }
    Ok(())
}

pub fn typed_id(mode_type: u16, motor_id: u16) -> u32 {
    (u32::from(mode_type & 0x7) << 8) | u32::from(motor_id & 0xFF)
}

pub fn position_cmd_id(motor_id: u16) -> u32 {
    typed_id(1, motor_id)
}

pub fn velocity_cmd_id(motor_id: u16) -> u32 {
    typed_id(2, motor_id)
}

pub fn param_read_id(motor_id: u16) -> u32 {
    typed_id(PARAM_READ_TYPE, motor_id)
}

pub fn param_write_id(motor_id: u16) -> u32 {
    typed_id(PARAM_WRITE_TYPE, motor_id)
}

pub fn float_to_uint(x: f32, x_min: f32, x_max: f32, bits: u8) -> u32 {
    let span = x_max - x_min;
    let clipped = x.clamp(x_min, x_max);
    ((clipped - x_min) * (((1u32 << bits) - 1) as f32) / span).round() as u32
}

pub fn uint_to_float(x: u32, x_min: f32, x_max: f32, bits: u8) -> f32 {
    let span = x_max - x_min;
    (x as f32) * span / (((1u32 << bits) - 1) as f32) + x_min
}

pub fn encode_enable() -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
}

pub fn encode_stop() -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
}

pub fn encode_zero() -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]
}

pub fn encode_clear_or_fault_query(cmd: u8) -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, cmd, 0xFB]
}

pub fn encode_set_mode(mode: u8) -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, mode, 0xFC]
}

pub fn encode_set_can_id(new_id: u8) -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, new_id, 0xFA]
}

pub fn encode_set_protocol(protocol: u8) -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, protocol, 0xFD]
}

pub fn encode_set_host_id(host_id: u8) -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, host_id, 0x01]
}

pub fn encode_save() -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF8]
}

pub fn encode_active_report(enable: bool) -> [u8; 8] {
    [
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        if enable { 1 } else { 0 },
        0xF9,
    ]
}

pub fn encode_mit_dynamic(pos: f32, vel: f32, kp: f32, kd: f32, tau: f32) -> [u8; 8] {
    let pos_u = float_to_uint(pos, MIT_P_MIN, MIT_P_MAX, 16);
    let vel_u = float_to_uint(vel, MIT_V_MIN, MIT_V_MAX, 12);
    let kp_u = float_to_uint(kp, MIT_KP_MIN, MIT_KP_MAX, 12);
    let kd_u = float_to_uint(kd, MIT_KD_MIN, MIT_KD_MAX, 12);
    let tau_u = float_to_uint(tau, MIT_T_MIN, MIT_T_MAX, 12);
    [
        ((pos_u >> 8) & 0xFF) as u8,
        (pos_u & 0xFF) as u8,
        ((vel_u >> 4) & 0xFF) as u8,
        (((vel_u & 0xF) << 4) | ((kp_u >> 8) & 0xF)) as u8,
        (kp_u & 0xFF) as u8,
        ((kd_u >> 4) & 0xFF) as u8,
        (((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)) as u8,
        (tau_u & 0xFF) as u8,
    ]
}

pub fn encode_position_control(pos_rad: f32, velocity_rad_s: f32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&pos_rad.to_le_bytes());
    out[4..8].copy_from_slice(&velocity_rad_s.to_le_bytes());
    out
}

pub fn encode_velocity_control(velocity_rad_s: f32, current_limit_a: f32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&velocity_rad_s.to_le_bytes());
    out[4..8].copy_from_slice(&current_limit_a.to_le_bytes());
    out
}

pub fn encode_param_read(index: u16) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&index.to_le_bytes());
    out
}

pub fn encode_param_write(index: u16, raw_value: [u8; 4]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&index.to_le_bytes());
    out[4..8].copy_from_slice(&raw_value);
    out
}

pub fn decode_feedback(data: [u8; 8]) -> MitFeedback {
    let pos_u = (u32::from(data[1]) << 8) | u32::from(data[2]);
    let vel_u = (u32::from(data[3]) << 4) | (u32::from(data[4]) >> 4);
    let tau_u = ((u32::from(data[4]) & 0x0F) << 8) | u32::from(data[5]);
    let temp_u = ((u16::from(data[6] & 0x0F)) << 8) | u16::from(data[7]);
    MitFeedback {
        motor_id: data[0],
        position_rad: uint_to_float(pos_u, MIT_P_MIN, MIT_P_MAX, 16),
        velocity_rad_s: uint_to_float(vel_u, MIT_V_MIN, MIT_V_MAX, 12),
        torque_nm: uint_to_float(tau_u, MIT_T_MIN, MIT_T_MAX, 12),
        mode_state: data[6] >> 6,
        has_fault: (data[6] & 0x20) != 0,
        has_warning: (data[6] & 0x10) != 0,
        winding_temp_c: f32::from(temp_u) / 10.0,
    }
}

pub fn decode_fault_value(data: [u8; 8]) -> u32 {
    u32::from_le_bytes([data[1], data[2], data[3], data[4]])
}

pub fn decode_param_reply(data: [u8; 8]) -> (u16, [u8; 4]) {
    (
        u16::from_le_bytes([data[0], data[1]]),
        [data[4], data[5], data[6], data[7]],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_mit_dynamic_pack() {
        let frame = encode_mit_dynamic(0.0, 0.0, 250.0, 2.5, 0.0);
        assert_eq!(frame, [0x80, 0x00, 0x80, 0x08, 0x00, 0x80, 0x08, 0x00]);
    }

    #[test]
    fn decodes_feedback_pack() {
        let data = [7, 0x80, 0x00, 0x80, 0x08, 0x00, 0x80, 0xFA];
        let fb = decode_feedback(data);
        assert_eq!(fb.motor_id, 7);
        assert_eq!(fb.mode_state, 2);
        assert!((fb.position_rad).abs() < 0.001);
        assert!((fb.velocity_rad_s).abs() < 0.02);
        assert!((fb.torque_nm).abs() < 0.02);
        assert!((fb.winding_temp_c - 25.0).abs() < 0.01);
    }
}
