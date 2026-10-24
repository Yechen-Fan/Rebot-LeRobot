use crate::model::{Target, Transport};
use motor_core::error::MotorError;
use motor_vendor_damiao::{display_models, match_models_by_limits, ControlMode, DamiaoMotor};
use serde_json::{json, Value};
use std::time::Duration;

use crate::commands::{
    as_u16, as_u64, build_scan_feedback_hints, build_scan_model_hints, parse_transport_in_msg,
};
use crate::vendors::transport_ws::open_damiao_controller;

pub(crate) fn ensure_control_mode_soft(
    motor: &DamiaoMotor,
    mode: ControlMode,
    timeout: Duration,
) -> Result<Option<String>, String> {
    match motor.ensure_control_mode(mode, timeout) {
        Ok(()) => Ok(None),
        Err(MotorError::Timeout(err)) => Ok(Some(format!(
            "Damiao CTRL_MODE register 10 verify timed out after requesting mode {}; command continued: {err}",
            mode as u32
        ))),
        Err(err) => Err(err.to_string()),
    }
}

fn dm_device_scan_channels(dm_device_type: &str) -> Vec<Option<String>> {
    let normalized = dm_device_type.to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "usb2canfd" => vec![Some("0".to_string())],
        "linkx4c" => vec![
            Some("0".to_string()),
            Some("1".to_string()),
            Some("2".to_string()),
            Some("3".to_string()),
        ],
        _ => vec![Some("0".to_string()), Some("1".to_string())],
    }
}

pub(crate) fn cmd_scan_damiao(v: &Value, base: &Target) -> Result<Value, String> {
    let transport = parse_transport_in_msg(v, base.transport)?;
    let mut target = base.clone();
    if let Some(channel) = v.get("channel").and_then(Value::as_str) {
        target.channel = channel.to_string();
    }
    if let Some(serial_port) = v.get("serial_port").and_then(Value::as_str) {
        target.serial_port = serial_port.to_string();
    }
    target.serial_baud = as_u64(v, "serial_baud", target.serial_baud as u64) as u32;
    if let Some(dm_device_type) = v
        .get("dm_device_type")
        .or_else(|| v.get("dm-device-type"))
        .and_then(Value::as_str)
    {
        target.dm_device_type = dm_device_type.to_string();
    }
    let requested_dm_channel = v.get("dm_channel").or_else(|| v.get("dm-channel"));
    if let Some(dm_channel) = requested_dm_channel.and_then(Value::as_str) {
        target.dm_channel = dm_channel.to_string();
    }
    let start_id = as_u16(v, "start_id", 1);
    let end_id = as_u16(v, "end_id", 16);
    let feedback_base = as_u16(v, "feedback_base", 16);
    let timeout_ms = as_u64(v, "timeout_ms", 100);
    if end_id < start_id {
        return Err("end_id must be >= start_id".to_string());
    }

    let scan_channels: Vec<Option<String>> =
        if transport == Transport::DmDevice && requested_dm_channel.is_none() {
            dm_device_scan_channels(&target.dm_device_type)
        } else {
            vec![None]
        };
    let model_hints = build_scan_model_hints(&target.model);
    let mut hits = Vec::new();
    let mut fallback_hits = 0usize;
    for scan_channel in scan_channels {
        let mut scan_target = target.clone();
        if let Some(dm_channel) = scan_channel {
            scan_target.dm_channel = dm_channel;
        }
        let controller = open_damiao_controller(&scan_target, transport)?;
        for mid in start_id..=end_id {
            enum ScanHit {
                Registers {
                    p: f32,
                    v: f32,
                    t: f32,
                    fid: u16,
                },
                Feedback {
                    fid: u16,
                    status: u8,
                    pos: f32,
                    vel: f32,
                    torq: f32,
                },
            }
            let mut found: Option<ScanHit> = None;
            let feedback_hints = build_scan_feedback_hints(feedback_base, mid);
            if transport != Transport::DmDevice {
                for fid in &feedback_hints {
                    for mh in &model_hints {
                        let Ok(candidate) = controller.add_motor(mid, *fid, mh) else {
                            continue;
                        };
                        let pmax =
                            candidate.get_register_f32(21, Duration::from_millis(timeout_ms));
                        let vmax =
                            candidate.get_register_f32(22, Duration::from_millis(timeout_ms));
                        let tmax =
                            candidate.get_register_f32(23, Duration::from_millis(timeout_ms));
                        if let (Ok(p), Ok(v), Ok(t)) = (pmax, vmax, tmax) {
                            found = Some(ScanHit::Registers { p, v, t, fid: *fid });
                            break;
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
            }
            if found.is_none() {
                for fid in &feedback_hints {
                    for mh in &model_hints {
                        let Ok(candidate) = controller.add_motor(mid, *fid, mh) else {
                            continue;
                        };
                        for _ in 0..20 {
                            let _ = candidate.request_motor_feedback();
                            let _ = controller.poll_feedback_once();
                            if let Some(s) = candidate.latest_state() {
                                found = Some(ScanHit::Feedback {
                                    fid: *fid,
                                    status: s.status_code,
                                    pos: s.pos,
                                    vel: s.vel,
                                    torq: s.torq,
                                });
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        if found.is_some() {
                            break;
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
            }
            if let Some(hit) = found {
                match hit {
                    ScanHit::Registers { p, v, t, fid } => {
                        let matched = match_models_by_limits(p, v, t, 0.2);
                        let model_guess = if matched.is_empty() {
                            "unknown".to_string()
                        } else {
                            display_models(&matched).join(",")
                        };
                        hits.push(json!({
                            "probe": mid,
                            "esc_id": mid,
                            "mst_id": fid,
                            "probe_feedback_id": fid,
                            "dm_channel": scan_target.dm_channel,
                            "model_guess": model_guess,
                            "pmax": p,
                            "vmax": v,
                            "tmax": t,
                            "detected_by": "registers"
                        }));
                    }
                    ScanHit::Feedback {
                        fid,
                        status,
                        pos,
                        vel,
                        torq,
                    } => {
                        hits.push(json!({
                            "probe": mid,
                            "esc_id": mid,
                            "mst_id": fid,
                            "probe_feedback_id": fid,
                            "dm_channel": scan_target.dm_channel,
                            "status": status,
                            "pos": pos,
                            "vel": vel,
                            "torq": torq,
                            "detected_by": "feedback"
                        }));
                        fallback_hits += 1;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = controller.close_bus();
    }

    Ok(json!({
        "vendor": "damiao",
        "transport": transport.as_str(),
        "dm_device_type": target.dm_device_type,
        "dm_channel": if transport == Transport::DmDevice && requested_dm_channel.is_none() { "all" } else { target.dm_channel.as_str() },
        "count": hits.len(),
        "start_id": start_id,
        "end_id": end_id,
        "fallback_hits": fallback_hits,
        "hits": hits,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use motor_core::bus::{CanBus, CanFrame};
    use motor_core::error::Result;
    use std::sync::Arc;

    struct SilentBus;

    impl CanBus for SilentBus {
        fn send(&self, _frame: CanFrame) -> Result<()> {
            Ok(())
        }

        fn recv(&self, _timeout: Duration) -> Result<Option<CanFrame>> {
            Ok(None)
        }

        fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn soft_mode_verify_returns_warning_on_ctrl_mode_timeout() {
        let motor =
            DamiaoMotor::new(0x01, 0x11, "4340P", Arc::new(SilentBus)).expect("create motor");

        let warning =
            ensure_control_mode_soft(&motor, ControlMode::PosVel, Duration::from_millis(1))
                .expect("timeout should be downgraded to warning")
                .expect("warning should be present");

        assert!(warning.contains("CTRL_MODE register 10 verify timed out"));
        assert!(warning.contains("command continued"));
    }
}
