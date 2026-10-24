use crate::objects::*;
use motor_core::bus::{CanBus, CanFrame};
use motor_core::device::MotorDevice;
use motor_core::error::{MotorError, Result};
use motor_core::model::{ModelCatalog, MotorModelSpec, PvTLimits, StaticModelCatalog};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ROBSTRIDE_MIT_MODELS: &[MotorModelSpec] = &[
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-00",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-01",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-02",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-03",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-04",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-05",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
    MotorModelSpec {
        vendor: "robstride_mit",
        model: "rs-06",
        pmax: MIT_P_MAX,
        vmax: MIT_V_MAX,
        tmax: MIT_T_MAX,
    },
];

const ROBSTRIDE_MIT_CATALOG: StaticModelCatalog = StaticModelCatalog {
    vendor_name: "robstride_mit",
    models: ROBSTRIDE_MIT_MODELS,
};

pub fn model_limits(model: &str) -> Option<(f32, f32, f32)> {
    ROBSTRIDE_MIT_CATALOG
        .get(model)
        .map(|spec| (spec.pmax, spec.vmax, spec.tmax))
}

#[derive(Debug, Clone, Copy)]
pub struct RobstrideMitStatus {
    pub feedback: Option<MitFeedback>,
    pub fault_code: Option<u32>,
}

pub struct RobstrideMitMotor {
    pub motor_id: u16,
    pub feedback_id: u16,
    pub model: String,
    bus: Arc<dyn CanBus>,
    limits: PvTLimits,
    response_queue: Mutex<VecDeque<CanFrame>>,
    latest_feedback: Mutex<Option<MitFeedback>>,
    latest_fault_code: Mutex<Option<u32>>,
}

impl RobstrideMitMotor {
    pub fn new(motor_id: u16, feedback_id: u16, model: &str, bus: Arc<dyn CanBus>) -> Result<Self> {
        validate_node_id(motor_id)?;
        validate_host_id(feedback_id)?;
        let spec = ROBSTRIDE_MIT_CATALOG.get(model).ok_or_else(|| {
            MotorError::InvalidArgument(format!("unknown RobStride MIT model: {model}"))
        })?;
        Ok(Self {
            motor_id,
            feedback_id,
            model: model.to_string(),
            bus,
            limits: PvTLimits::from_spec(spec),
            response_queue: Mutex::new(VecDeque::new()),
            latest_feedback: Mutex::new(None),
            latest_fault_code: Mutex::new(None),
        })
    }

    fn send_std_frame(&self, arbitration_id: u32, payload: [u8; 8]) -> Result<()> {
        self.bus.send(CanFrame {
            arbitration_id,
            data: payload,
            dlc: 8,
            is_extended: false,
            is_rx: false,
        })
    }

    fn send_to_motor(&self, payload: [u8; 8]) -> Result<()> {
        self.send_std_frame(u32::from(self.motor_id), payload)
    }

    fn pop_matching_response<F>(&self, timeout: Duration, mut predicate: F) -> Result<CanFrame>
    where
        F: FnMut(&CanFrame) -> bool,
    {
        let deadline = Instant::now() + timeout;
        loop {
            let mut pending = self
                .response_queue
                .lock()
                .map_err(|_| MotorError::Io("response queue lock poisoned".to_string()))?;
            let mut hold = VecDeque::new();
            let mut found = None;
            while let Some(frame) = pending.pop_front() {
                if predicate(&frame) {
                    found = Some(frame);
                    break;
                }
                hold.push_back(frame);
            }
            while let Some(frame) = hold.pop_front() {
                pending.push_back(frame);
            }
            drop(pending);
            if let Some(frame) = found {
                return Ok(frame);
            }
            if Instant::now() >= deadline {
                return Err(MotorError::Timeout(format!(
                    "robstride_mit response timeout id=0x{:X}",
                    self.motor_id
                )));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn wait_host_feedback(&self, timeout: Duration) -> Result<MitFeedback> {
        let frame = self.pop_matching_response(timeout, |frame| {
            !frame.is_extended
                && frame.arbitration_id == u32::from(self.feedback_id)
                && frame.dlc >= 8
                && u16::from(frame.data[0]) == self.motor_id
        })?;
        let feedback = decode_feedback(frame.data);
        self.latest_feedback
            .lock()
            .map_err(|_| MotorError::Io("latest feedback lock poisoned".to_string()))?
            .replace(feedback);
        Ok(feedback)
    }

    fn wait_unique_id_reply(&self, timeout: Duration) -> Result<[u8; 8]> {
        let frame = self.pop_matching_response(timeout, |frame| {
            !frame.is_extended && frame.arbitration_id == u32::from(self.motor_id) && frame.dlc >= 8
        })?;
        Ok(frame.data)
    }

    fn wait_param_reply(&self, mode_type: u16, index: u16, timeout: Duration) -> Result<[u8; 4]> {
        let expected_id = typed_id(mode_type, self.motor_id);
        let frame = self.pop_matching_response(timeout, |frame| {
            !frame.is_extended
                && frame.arbitration_id == expected_id
                && frame.dlc >= 8
                && u16::from_le_bytes([frame.data[0], frame.data[1]]) == index
        })?;
        let (_, value) = decode_param_reply(frame.data);
        Ok(value)
    }

    pub fn enable_drive(&self, timeout: Duration) -> Result<MitFeedback> {
        self.send_to_motor(encode_enable())?;
        self.wait_host_feedback(timeout)
    }

    pub fn disable_drive(&self, timeout: Duration) -> Result<MitFeedback> {
        self.send_to_motor(encode_stop())?;
        self.wait_host_feedback(timeout)
    }

    pub fn set_current_position_zero(&self, timeout: Duration) -> Result<MitFeedback> {
        self.send_to_motor(encode_zero())?;
        self.wait_host_feedback(timeout)
    }

    pub fn clear_fault(&self, timeout: Duration) -> Result<MitFeedback> {
        self.send_to_motor(encode_clear_or_fault_query(0xFF))?;
        self.wait_host_feedback(timeout)
    }

    pub fn query_fault(&self, timeout: Duration) -> Result<u32> {
        self.send_to_motor(encode_clear_or_fault_query(0x00))?;
        let frame = self.pop_matching_response(timeout, |frame| {
            !frame.is_extended
                && frame.arbitration_id == u32::from(self.feedback_id)
                && frame.dlc >= 8
                && u16::from(frame.data[0]) == self.motor_id
        })?;
        let fault = decode_fault_value(frame.data);
        self.latest_fault_code
            .lock()
            .map_err(|_| MotorError::Io("latest fault lock poisoned".to_string()))?
            .replace(fault);
        Ok(fault)
    }

    pub fn query_status(&self, timeout: Duration) -> Result<RobstrideMitStatus> {
        let fault_code = self.query_fault(timeout).ok();
        let feedback = self
            .latest_feedback
            .lock()
            .map_err(|_| MotorError::Io("latest feedback lock poisoned".to_string()))?
            .to_owned();
        Ok(RobstrideMitStatus {
            feedback,
            fault_code,
        })
    }

    pub fn set_mode(&self, mode: u8, timeout: Duration) -> Result<MitFeedback> {
        if !matches!(mode, MODE_MIT | MODE_POSITION | MODE_VELOCITY) {
            return Err(MotorError::InvalidArgument(format!(
                "invalid RobStride MIT mode {mode}, expected 0(MIT), 1(position), or 2(velocity)"
            )));
        }
        self.send_to_motor(encode_set_mode(mode))?;
        self.wait_host_feedback(timeout)
    }

    pub fn set_can_id(&self, new_id: u8, timeout: Duration) -> Result<[u8; 8]> {
        validate_node_id(u16::from(new_id))?;
        self.send_to_motor(encode_set_can_id(new_id))?;
        self.wait_unique_id_reply(timeout)
    }

    pub fn set_protocol(&self, protocol: u8, timeout: Duration) -> Result<[u8; 8]> {
        if !matches!(protocol, PROTOCOL_PRIVATE | PROTOCOL_CANOPEN | PROTOCOL_MIT) {
            return Err(MotorError::InvalidArgument(format!(
                "invalid RobStride protocol command {protocol}, expected 0(private), 1(canopen), or 2(mit)"
            )));
        }
        self.send_to_motor(encode_set_protocol(protocol))?;
        self.wait_unique_id_reply(timeout)
    }

    pub fn set_host_id(&self, host_id: u8, timeout: Duration) -> Result<[u8; 8]> {
        validate_host_id(u16::from(host_id))?;
        self.send_to_motor(encode_set_host_id(host_id))?;
        self.wait_unique_id_reply(timeout)
    }

    pub fn save(&self, timeout: Duration) -> Result<[u8; 8]> {
        self.send_to_motor(encode_save())?;
        self.wait_unique_id_reply(timeout)
    }

    pub fn set_active_report(&self, enable: bool, timeout: Duration) -> Result<MitFeedback> {
        self.send_to_motor(encode_active_report(enable))?;
        self.wait_host_feedback(timeout)
    }

    pub fn command_mit(
        &self,
        pos: f32,
        vel: f32,
        kp: f32,
        kd: f32,
        tau: f32,
        timeout: Duration,
    ) -> Result<MitFeedback> {
        let data = encode_mit_dynamic(
            pos.clamp(-self.limits.p_max, self.limits.p_max),
            vel.clamp(-self.limits.v_max, self.limits.v_max),
            kp,
            kd,
            tau.clamp(-self.limits.t_max, self.limits.t_max),
        );
        self.send_to_motor(data)?;
        self.wait_host_feedback(timeout)
    }

    pub fn command_position(
        &self,
        pos_rad: f32,
        velocity_rad_s: f32,
        timeout: Duration,
    ) -> Result<MitFeedback> {
        self.set_mode(MODE_POSITION, timeout)?;
        self.enable_drive(timeout)?;
        self.send_std_frame(
            position_cmd_id(self.motor_id),
            encode_position_control(pos_rad, velocity_rad_s.abs()),
        )?;
        self.wait_host_feedback(timeout)
    }

    pub fn command_velocity(
        &self,
        velocity_rad_s: f32,
        current_limit_a: f32,
        timeout: Duration,
    ) -> Result<MitFeedback> {
        self.set_mode(MODE_VELOCITY, timeout)?;
        self.enable_drive(timeout)?;
        self.send_std_frame(
            velocity_cmd_id(self.motor_id),
            encode_velocity_control(velocity_rad_s, current_limit_a.abs()),
        )?;
        self.wait_host_feedback(timeout)
    }

    pub fn read_parameter(&self, index: u16, timeout: Duration) -> Result<[u8; 4]> {
        self.send_std_frame(param_read_id(self.motor_id), encode_param_read(index))?;
        self.wait_param_reply(PARAM_READ_TYPE, index, timeout)
    }

    pub fn write_parameter(
        &self,
        index: u16,
        raw_value: [u8; 4],
        timeout: Duration,
    ) -> Result<[u8; 4]> {
        self.send_std_frame(
            param_write_id(self.motor_id),
            encode_param_write(index, raw_value),
        )?;
        self.wait_param_reply(PARAM_WRITE_TYPE, index, timeout)
    }
}

impl MotorDevice for RobstrideMitMotor {
    fn vendor(&self) -> &'static str {
        "robstride_mit"
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
        self.enable_drive(Duration::from_millis(250)).map(|_| ())
    }

    fn disable(&self) -> Result<()> {
        self.disable_drive(Duration::from_millis(250)).map(|_| ())
    }

    fn accepts_frame(&self, frame: &CanFrame) -> bool {
        !frame.is_extended
            && (frame.arbitration_id == u32::from(self.feedback_id)
                || frame.arbitration_id == u32::from(self.motor_id)
                || frame.arbitration_id == param_read_id(self.motor_id)
                || frame.arbitration_id == param_write_id(self.motor_id))
    }

    fn process_feedback_frame(&self, frame: CanFrame) -> Result<()> {
        self.response_queue
            .lock()
            .map_err(|_| MotorError::Io("response queue lock poisoned".to_string()))?
            .push_back(frame);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motor_core::test_support::MockBus;

    fn feedback_frame(host_id: u16, motor_id: u8) -> CanFrame {
        CanFrame {
            arbitration_id: u32::from(host_id),
            data: [motor_id, 0x80, 0x00, 0x80, 0x08, 0x00, 0x80, 0xFA],
            dlc: 8,
            is_extended: false,
            is_rx: true,
        }
    }

    #[test]
    fn mit_command_uses_standard_motor_id_and_packed_payload() {
        let bus = Arc::new(MockBus::new());
        let motor = RobstrideMitMotor::new(7, 0x7D, "rs-00", bus.clone()).expect("motor");
        motor
            .process_feedback_frame(feedback_frame(0x7D, 7))
            .expect("queue feedback");
        motor
            .command_mit(0.0, 0.0, 250.0, 2.5, 0.0, Duration::from_millis(10))
            .expect("mit command");
        let sent = bus.sent.lock().expect("sent");
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].arbitration_id, 7);
        assert!(!sent[0].is_extended);
        assert_eq!(
            sent[0].data,
            [0x80, 0x00, 0x80, 0x08, 0x00, 0x80, 0x08, 0x00]
        );
    }

    #[test]
    fn velocity_command_uses_typed_standard_id() {
        let bus = Arc::new(MockBus::new());
        let motor = RobstrideMitMotor::new(7, 0x7D, "rs-00", bus.clone()).expect("motor");
        for _ in 0..3 {
            motor
                .process_feedback_frame(feedback_frame(0x7D, 7))
                .expect("queue feedback");
        }
        motor
            .command_velocity(5.0, 2.0, Duration::from_millis(10))
            .expect("velocity command");
        let sent = bus.sent.lock().expect("sent");
        assert_eq!(sent[0].arbitration_id, 7);
        assert_eq!(sent[0].data, encode_set_mode(MODE_VELOCITY));
        assert_eq!(sent[1].arbitration_id, 7);
        assert_eq!(sent[1].data, encode_enable());
        assert_eq!(sent[2].arbitration_id, velocity_cmd_id(7));
        assert_eq!(
            f32::from_le_bytes(sent[2].data[0..4].try_into().unwrap()),
            5.0
        );
        assert_eq!(
            f32::from_le_bytes(sent[2].data[4..8].try_into().unwrap()),
            2.0
        );
    }
}
