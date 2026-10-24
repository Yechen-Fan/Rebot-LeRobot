use crate::objects::*;
use motor_core::bus::{CanBus, CanFrame};
use motor_core::device::MotorDevice;
use motor_core::error::{MotorError, Result};
use motor_core::model::{ModelCatalog, MotorModelSpec, PvTLimits, StaticModelCatalog};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ROBSTRIDE_CIA402_MODELS: &[MotorModelSpec] = &[
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-00",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 50.0,
        tmax: 17.0,
    },
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-01",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 44.0,
        tmax: 17.0,
    },
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-02",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 44.0,
        tmax: 17.0,
    },
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-03",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 50.0,
        tmax: 60.0,
    },
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-04",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 15.0,
        tmax: 120.0,
    },
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-05",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 33.0,
        tmax: 17.0,
    },
    MotorModelSpec {
        vendor: "robstride_cia402",
        model: "rs-06",
        pmax: 4.0 * std::f32::consts::PI,
        vmax: 20.0,
        tmax: 60.0,
    },
];

const ROBSTRIDE_CIA402_CATALOG: StaticModelCatalog = StaticModelCatalog {
    vendor_name: "robstride_cia402",
    models: ROBSTRIDE_CIA402_MODELS,
};

pub fn model_limits(model: &str) -> Option<(f32, f32, f32)> {
    ROBSTRIDE_CIA402_CATALOG
        .get(model)
        .map(|spec| (spec.pmax, spec.vmax, spec.tmax))
}

#[derive(Debug, Clone, Copy)]
pub struct RobstrideCia402Target {
    pub position_rad: f32,
    pub velocity_rad_s: f32,
    pub torque_nm: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RobstrideCia402Status {
    pub mode_display: i8,
    pub statusword: u16,
    pub error_code: u16,
    pub position_rad: f32,
    pub velocity_rad_s: f32,
    pub torque_nm: f32,
    pub current_ma: i16,
    pub dc_link_mv: i32,
    pub heartbeat_state: Option<u8>,
}

pub struct RobstrideCia402Motor {
    pub motor_id: u16,
    pub feedback_id: u16,
    pub model: String,
    bus: Arc<dyn CanBus>,
    limits: PvTLimits,
    sdo_reply_queue: Mutex<VecDeque<[u8; 8]>>,
    heartbeat_state: Mutex<Option<u8>>,
}

impl RobstrideCia402Motor {
    pub fn new(motor_id: u16, feedback_id: u16, model: &str, bus: Arc<dyn CanBus>) -> Result<Self> {
        if motor_id == 0 || motor_id > 127 {
            return Err(MotorError::InvalidArgument(format!(
                "invalid RobStride CiA402 node id {motor_id}, expected 1..127"
            )));
        }
        let spec = ROBSTRIDE_CIA402_CATALOG.get(model).ok_or_else(|| {
            MotorError::InvalidArgument(format!("unknown RobStride CiA402 model: {model}"))
        })?;
        Ok(Self {
            motor_id,
            feedback_id,
            model: model.to_string(),
            bus,
            limits: PvTLimits::from_spec(spec),
            sdo_reply_queue: Mutex::new(VecDeque::new()),
            heartbeat_state: Mutex::new(None),
        })
    }

    fn sdo_req_id(&self) -> u32 {
        sdo_req_id(self.motor_id)
    }

    fn sdo_rsp_id(&self) -> u32 {
        sdo_rsp_id(self.motor_id)
    }

    fn heartbeat_id(&self) -> u32 {
        heartbeat_id(self.motor_id)
    }

    fn send_std_frame(&self, arbitration_id: u32, payload: &[u8]) -> Result<()> {
        if payload.len() > 8 {
            return Err(MotorError::InvalidArgument(format!(
                "payload too long: {}, expected <=8",
                payload.len()
            )));
        }
        let mut data = [0u8; 8];
        data[..payload.len()].copy_from_slice(payload);
        self.bus.send(CanFrame {
            arbitration_id,
            data,
            dlc: payload.len() as u8,
            is_extended: false,
            is_rx: false,
        })
    }

    pub fn send_nmt(&self, command: u8, node_id: u8) -> Result<()> {
        self.send_std_frame(NMT_ID, &[command, node_id])
    }

    fn build_sdo_upload(index: u16, subindex: u8) -> [u8; 8] {
        [
            0x40,
            (index & 0xFF) as u8,
            ((index >> 8) & 0xFF) as u8,
            subindex,
            0,
            0,
            0,
            0,
        ]
    }

    fn build_sdo_download(index: u16, subindex: u8, payload4: [u8; 4], nbytes: usize) -> [u8; 8] {
        let cmd = match nbytes {
            1 => 0x2F,
            2 => 0x2B,
            4 => 0x23,
            _ => 0x23,
        };
        [
            cmd,
            (index & 0xFF) as u8,
            ((index >> 8) & 0xFF) as u8,
            subindex,
            payload4[0],
            payload4[1],
            payload4[2],
            payload4[3],
        ]
    }

    fn pop_matching_sdo_reply(
        &self,
        index: u16,
        subindex: u8,
        timeout: Duration,
    ) -> Result<[u8; 8]> {
        let deadline = Instant::now() + timeout;
        loop {
            let mut pending = self
                .sdo_reply_queue
                .lock()
                .map_err(|_| MotorError::Io("sdo queue lock poisoned".to_string()))?;
            let mut hold = VecDeque::new();
            let mut found = None;
            while let Some(msg) = pending.pop_front() {
                let idx = u16::from(msg[1]) | (u16::from(msg[2]) << 8);
                let sub = msg[3];
                if idx == index && sub == subindex {
                    found = Some(msg);
                    break;
                }
                hold.push_back(msg);
            }
            while let Some(m) = hold.pop_front() {
                pending.push_back(m);
            }
            drop(pending);
            if let Some(msg) = found {
                return Ok(msg);
            }
            if Instant::now() >= deadline {
                return Err(MotorError::Timeout(format!(
                    "sdo response timeout idx=0x{index:04X} sub=0x{subindex:02X}"
                )));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn sdo_read_raw(&self, index: u16, subindex: u8, timeout: Duration) -> Result<[u8; 4]> {
        let req = Self::build_sdo_upload(index, subindex);
        self.send_std_frame(self.sdo_req_id(), &req)?;
        let rsp = self.pop_matching_sdo_reply(index, subindex, timeout)?;
        let cmd = rsp[0];
        if cmd == 0x80 {
            let abort = u32::from_le_bytes([rsp[4], rsp[5], rsp[6], rsp[7]]);
            return Err(MotorError::Protocol(format!(
                "sdo abort idx=0x{index:04X} sub=0x{subindex:02X} code=0x{abort:08X}"
            )));
        }
        if !matches!(cmd, 0x43 | 0x4B | 0x4F | 0x47) {
            return Err(MotorError::Protocol(format!(
                "unexpected sdo read cmd=0x{cmd:02X} idx=0x{index:04X} sub=0x{subindex:02X}"
            )));
        }
        Ok([rsp[4], rsp[5], rsp[6], rsp[7]])
    }

    fn sdo_write_raw(
        &self,
        index: u16,
        subindex: u8,
        payload4: [u8; 4],
        nbytes: usize,
        timeout: Duration,
    ) -> Result<()> {
        let req = Self::build_sdo_download(index, subindex, payload4, nbytes);
        self.send_std_frame(self.sdo_req_id(), &req)?;
        let rsp = self.pop_matching_sdo_reply(index, subindex, timeout)?;
        let cmd = rsp[0];
        if cmd == 0x80 {
            let abort = u32::from_le_bytes([rsp[4], rsp[5], rsp[6], rsp[7]]);
            return Err(MotorError::Protocol(format!(
                "sdo abort idx=0x{index:04X} sub=0x{subindex:02X} code=0x{abort:08X}"
            )));
        }
        if cmd != 0x60 {
            return Err(MotorError::Protocol(format!(
                "unexpected sdo write ack cmd=0x{cmd:02X} idx=0x{index:04X} sub=0x{subindex:02X}"
            )));
        }
        Ok(())
    }

    pub fn sdo_read_i8(&self, index: u16, subindex: u8, timeout: Duration) -> Result<i8> {
        let raw = self.sdo_read_raw(index, subindex, timeout)?;
        Ok(raw[0] as i8)
    }

    pub fn sdo_read_u16(&self, index: u16, subindex: u8, timeout: Duration) -> Result<u16> {
        let raw = self.sdo_read_raw(index, subindex, timeout)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    pub fn sdo_read_i16(&self, index: u16, subindex: u8, timeout: Duration) -> Result<i16> {
        let raw = self.sdo_read_raw(index, subindex, timeout)?;
        Ok(i16::from_le_bytes([raw[0], raw[1]]))
    }

    pub fn sdo_read_i32(&self, index: u16, subindex: u8, timeout: Duration) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.sdo_read_raw(index, subindex, timeout)?,
        ))
    }

    pub fn sdo_write_i8(
        &self,
        index: u16,
        subindex: u8,
        value: i8,
        timeout: Duration,
    ) -> Result<()> {
        self.sdo_write_raw(index, subindex, [value as u8, 0, 0, 0], 1, timeout)
    }

    pub fn sdo_write_u16(
        &self,
        index: u16,
        subindex: u8,
        value: u16,
        timeout: Duration,
    ) -> Result<()> {
        let b = value.to_le_bytes();
        self.sdo_write_raw(index, subindex, [b[0], b[1], 0, 0], 2, timeout)
    }

    pub fn sdo_write_i16(
        &self,
        index: u16,
        subindex: u8,
        value: i16,
        timeout: Duration,
    ) -> Result<()> {
        let b = value.to_le_bytes();
        self.sdo_write_raw(index, subindex, [b[0], b[1], 0, 0], 2, timeout)
    }

    pub fn sdo_write_i32(
        &self,
        index: u16,
        subindex: u8,
        value: i32,
        timeout: Duration,
    ) -> Result<()> {
        self.sdo_write_raw(index, subindex, value.to_le_bytes(), 4, timeout)
    }

    pub fn start_node(&self) -> Result<()> {
        self.send_nmt(NMT_START_REMOTE_NODE, self.motor_id as u8)
    }

    pub fn set_mode(&self, mode: i8, timeout: Duration) -> Result<()> {
        self.sdo_write_i8(MODES_OF_OPERATION, 0x00, mode, timeout)
    }

    pub fn enable_drive(&self, timeout: Duration) -> Result<()> {
        self.start_node()?;
        std::thread::sleep(Duration::from_millis(10));
        for cw in [CW_SHUTDOWN, CW_SWITCH_ON, CW_ENABLE_OPERATION] {
            self.sdo_write_u16(CONTROLWORD, 0x00, cw, timeout)?;
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }

    pub fn disable_drive(&self, timeout: Duration) -> Result<()> {
        self.sdo_write_u16(CONTROLWORD, 0x00, CW_DISABLE_TO_SWITCH_ON_DISABLED, timeout)
    }

    pub fn quick_stop(&self, timeout: Duration) -> Result<()> {
        self.sdo_write_u16(CONTROLWORD, 0x00, CW_QUICK_STOP, timeout)
    }

    pub fn clear_fault(&self, timeout: Duration) -> Result<()> {
        self.sdo_write_u16(CONTROLWORD, 0x00, CW_FAULT_RESET, timeout)?;
        std::thread::sleep(Duration::from_millis(10));
        self.sdo_write_u16(CONTROLWORD, 0x00, 0x0000, timeout)
    }

    pub fn prepare_mode_disabled(&self, mode: i8, timeout: Duration) -> Result<()> {
        self.start_node()?;
        std::thread::sleep(Duration::from_millis(10));
        self.disable_drive(timeout).ok();
        std::thread::sleep(Duration::from_millis(10));
        self.set_mode(mode, timeout)
    }

    pub fn ensure_mode_enabled(&self, mode: i8, timeout: Duration) -> Result<()> {
        self.prepare_mode_disabled(mode, timeout)?;
        self.enable_drive(timeout)
    }

    pub fn set_can_watchdog_raw(&self, value: u32, timeout: Duration) -> Result<()> {
        let b = value.to_le_bytes();
        self.sdo_write_raw(CAN_WATCHDOG, 0x01, b, 4, timeout)
    }

    pub fn set_can_watchdog_seconds(&self, seconds: f32, timeout: Duration) -> Result<()> {
        self.set_can_watchdog_raw(watchdog_seconds_to_raw(seconds), timeout)
    }

    pub fn set_position_window_rad(&self, window_rad: f32, timeout: Duration) -> Result<()> {
        self.sdo_write_i32(
            POSITION_WINDOW,
            0x00,
            rad_to_pulses(window_rad.abs()),
            timeout,
        )
    }

    pub fn set_position_window_time_ms(&self, time_ms: u16, timeout: Duration) -> Result<()> {
        self.sdo_write_u16(POSITION_WINDOW_TIME, 0x00, time_ms, timeout)
    }

    pub fn set_current_position_zero(&self, timeout: Duration) -> Result<()> {
        self.prepare_mode_disabled(MODE_HOMING, timeout)?;
        self.sdo_write_u16(CONTROLWORD, 0x00, CW_ENABLE_OPERATION, timeout)
    }

    pub fn command_profile_position(
        &self,
        target_position_rad: f32,
        velocity_limit_rad_s: f32,
        acceleration_rad_s2: f32,
        timeout: Duration,
    ) -> Result<()> {
        self.command_profile_position_full(
            target_position_rad,
            velocity_limit_rad_s,
            acceleration_rad_s2,
            None,
            None,
            timeout,
        )
    }

    pub fn command_profile_position_full(
        &self,
        target_position_rad: f32,
        velocity_limit_rad_s: f32,
        acceleration_rad_s2: f32,
        position_window_rad: Option<f32>,
        position_window_time_ms: Option<u16>,
        timeout: Duration,
    ) -> Result<()> {
        self.prepare_mode_disabled(MODE_PROFILE_POSITION, timeout)?;
        self.sdo_write_i16(
            TARGET_TORQUE,
            0x00,
            torque_nm_to_robstride_raw(self.limits.t_max),
            timeout,
        )?;
        self.sdo_write_i32(
            PROFILE_VELOCITY,
            0x00,
            rad_s_to_0p1rpm(velocity_limit_rad_s.abs()),
            timeout,
        )?;
        self.sdo_write_i32(
            PROFILE_ACCELERATION,
            0x00,
            rad_s_to_0p1rpm(acceleration_rad_s2.abs()),
            timeout,
        )?;
        if let Some(window_rad) = position_window_rad {
            self.set_position_window_rad(window_rad, timeout)?;
        }
        if let Some(window_time_ms) = position_window_time_ms {
            self.set_position_window_time_ms(window_time_ms, timeout)?;
        }
        self.enable_drive(timeout)?;
        self.sdo_write_i32(
            TARGET_POSITION,
            0x00,
            rad_to_pulses(target_position_rad),
            timeout,
        )
    }

    pub fn command_csp(
        &self,
        target_position_rad: f32,
        velocity_limit_rad_s: f32,
        torque_limit_nm: f32,
        timeout: Duration,
    ) -> Result<()> {
        self.command_csp_full(
            target_position_rad,
            velocity_limit_rad_s,
            torque_limit_nm,
            None,
            None,
            timeout,
        )
    }

    pub fn command_csp_full(
        &self,
        target_position_rad: f32,
        velocity_limit_rad_s: f32,
        torque_limit_nm: f32,
        position_window_rad: Option<f32>,
        position_window_time_ms: Option<u16>,
        timeout: Duration,
    ) -> Result<()> {
        self.prepare_mode_disabled(MODE_CSP, timeout)?;
        self.sdo_write_i16(
            TARGET_TORQUE,
            0x00,
            torque_nm_to_robstride_raw(torque_limit_nm.abs()),
            timeout,
        )?;
        if velocity_limit_rad_s.is_finite() && velocity_limit_rad_s.abs() > 0.0 {
            self.sdo_write_i32(
                PROFILE_VELOCITY,
                0x00,
                rad_s_to_0p1rpm(velocity_limit_rad_s.abs()),
                timeout,
            )?;
        }
        if let Some(window_rad) = position_window_rad {
            self.set_position_window_rad(window_rad, timeout)?;
        }
        if let Some(window_time_ms) = position_window_time_ms {
            self.set_position_window_time_ms(window_time_ms, timeout)?;
        }
        self.enable_drive(timeout)?;
        self.sdo_write_i32(
            TARGET_POSITION,
            0x00,
            rad_to_pulses(target_position_rad),
            timeout,
        )
    }

    pub fn command_velocity(&self, target_velocity_rad_s: f32, timeout: Duration) -> Result<()> {
        self.prepare_mode_disabled(MODE_PROFILE_VELOCITY, timeout)?;
        self.sdo_write_i16(
            TARGET_TORQUE,
            0x00,
            torque_nm_to_robstride_raw(self.limits.t_max),
            timeout,
        )?;
        self.enable_drive(timeout)?;
        self.sdo_write_i32(
            TARGET_VELOCITY,
            0x00,
            rad_s_to_0p1rpm(target_velocity_rad_s),
            timeout,
        )
    }

    pub fn command_torque(&self, target_torque_nm: f32, timeout: Duration) -> Result<()> {
        self.prepare_mode_disabled(MODE_TORQUE, timeout)?;
        self.enable_drive(timeout)?;
        self.sdo_write_i16(
            TARGET_TORQUE,
            0x00,
            torque_nm_to_robstride_raw(target_torque_nm),
            timeout,
        )
    }

    pub fn query_status(&self, timeout: Duration) -> Result<RobstrideCia402Status> {
        let mode_display = self.sdo_read_i8(MODES_OF_OPERATION_DISPLAY, 0x00, timeout)?;
        let statusword = self.sdo_read_u16(STATUSWORD, 0x00, timeout)?;
        let error_code = self.sdo_read_u16(ERROR_CODE, 0x00, timeout).unwrap_or(0);
        let position = self
            .sdo_read_i32(POSITION_ACTUAL_VALUE, 0x00, timeout)
            .map(pulses_to_rad)
            .unwrap_or(0.0);
        let velocity = self
            .sdo_read_i32(VELOCITY_ACTUAL_VALUE, 0x00, timeout)
            .map(unit_0p1rpm_to_rad_s)
            .unwrap_or(0.0);
        let torque = self
            .sdo_read_i16(TORQUE_ACTUAL_VALUE, 0x00, timeout)
            .map(robstride_raw_to_torque_nm)
            .unwrap_or(0.0);
        let current_ma = self
            .sdo_read_i16(CURRENT_ACTUAL_VALUE, 0x00, timeout)
            .unwrap_or(0);
        let dc_link_mv = self
            .sdo_read_i32(DC_LINK_CIRCUIT_VOLTAGE, 0x00, timeout)
            .unwrap_or(0);
        let heartbeat_state = self
            .heartbeat_state
            .lock()
            .map_err(|_| MotorError::Io("heartbeat lock poisoned".to_string()))?
            .to_owned();
        Ok(RobstrideCia402Status {
            mode_display,
            statusword,
            error_code,
            position_rad: position,
            velocity_rad_s: velocity,
            torque_nm: torque,
            current_ma,
            dc_link_mv,
            heartbeat_state,
        })
    }
}

impl MotorDevice for RobstrideCia402Motor {
    fn vendor(&self) -> &'static str {
        "robstride_cia402"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn motor_id(&self) -> u16 {
        self.motor_id
    }

    fn feedback_id(&self) -> u16 {
        self.feedback_id
    }

    fn enable(&self) -> Result<()> {
        self.enable_drive(Duration::from_millis(250))
    }

    fn disable(&self) -> Result<()> {
        self.disable_drive(Duration::from_millis(250))
    }

    fn accepts_frame(&self, frame: &CanFrame) -> bool {
        !frame.is_extended
            && (frame.arbitration_id == self.sdo_rsp_id()
                || frame.arbitration_id == self.heartbeat_id())
    }

    fn process_feedback_frame(&self, frame: CanFrame) -> Result<()> {
        if frame.arbitration_id == self.sdo_rsp_id() && frame.dlc >= 8 {
            self.sdo_reply_queue
                .lock()
                .map_err(|_| MotorError::Io("sdo queue lock poisoned".to_string()))?
                .push_back(frame.data);
            return Ok(());
        }
        if frame.arbitration_id == self.heartbeat_id() && frame.dlc >= 1 {
            self.heartbeat_state
                .lock()
                .map_err(|_| MotorError::Io("heartbeat lock poisoned".to_string()))?
                .replace(frame.data[0]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motor_core::test_support::MockBus;

    fn ack(index: u16, sub: u8) -> CanFrame {
        CanFrame {
            arbitration_id: sdo_rsp_id(1),
            data: [
                0x60,
                (index & 0xFF) as u8,
                (index >> 8) as u8,
                sub,
                0,
                0,
                0,
                0,
            ],
            dlc: 8,
            is_extended: false,
            is_rx: true,
        }
    }

    #[test]
    fn command_velocity_uses_cia402_sdo_ids() {
        let bus = Arc::new(MockBus::new());
        let motor = RobstrideCia402Motor::new(1, 0, "rs-06", bus.clone()).expect("motor");
        for item in [
            ack(CONTROLWORD, 0),
            ack(MODES_OF_OPERATION, 0),
            ack(CONTROLWORD, 0),
            ack(CONTROLWORD, 0),
            ack(CONTROLWORD, 0),
            ack(TARGET_TORQUE, 0),
            ack(TARGET_VELOCITY, 0),
        ] {
            motor.process_feedback_frame(item).expect("queue sdo ack");
        }
        motor
            .command_velocity(TWO_PI, Duration::from_millis(10))
            .expect("velocity command");
        let sent = bus.sent.lock().expect("sent");
        assert_eq!(sent[0].arbitration_id, NMT_ID);
        assert_eq!(sent[1].arbitration_id, sdo_req_id(1));
        assert!(!sent[1].is_extended);
        let last = sent.last().expect("last");
        assert_eq!(last.arbitration_id, sdo_req_id(1));
        assert_eq!(last.data[1], (TARGET_VELOCITY & 0xFF) as u8);
        assert_eq!(last.data[2], (TARGET_VELOCITY >> 8) as u8);
        assert_eq!(
            i32::from_le_bytes([last.data[4], last.data[5], last.data[6], last.data[7]]),
            600
        );
    }
}
