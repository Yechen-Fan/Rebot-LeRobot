use crate::motor::RobstrideMitMotor;
use crate::objects::{
    decode_fault_value, encode_clear_or_fault_query, validate_host_id, validate_node_id,
};
use motor_core::bus::{open_can_bus, CanBus, CanFrame};
use motor_core::error::{MotorError, Result};
use motor_core::vendor_controller::VendorController;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct RobstrideMitScanHit {
    pub node_id: u16,
    pub host_id: u16,
    pub fault_code: Option<u32>,
}

pub struct RobstrideMitController {
    controller: VendorController<RobstrideMitMotor>,
}

impl RobstrideMitController {
    pub fn new(bus: Arc<dyn CanBus>) -> Self {
        Self {
            controller: VendorController::new(bus),
        }
    }

    pub fn new_socketcan(channel: &str) -> Result<Self> {
        Ok(Self::new(open_can_bus(channel)?))
    }

    pub fn add_motor(
        &self,
        motor_id: u16,
        feedback_id: u16,
        model: &str,
    ) -> Result<Arc<RobstrideMitMotor>> {
        self.controller.add_motor_with(motor_id, |bus| {
            RobstrideMitMotor::new(motor_id, feedback_id, model, bus)
        })
    }

    pub fn get_motor(&self, motor_id: u16) -> Result<Arc<RobstrideMitMotor>> {
        self.controller.get_motor(motor_id)
    }

    pub fn poll_feedback_once(&self) -> Result<()> {
        self.controller.poll_feedback_once()
    }

    pub fn enable_all(&self) -> Result<()> {
        self.controller.enable_all()
    }

    pub fn disable_all(&self) -> Result<()> {
        self.controller.disable_all()
    }

    pub fn shutdown(&self) -> Result<()> {
        self.controller.shutdown()
    }

    pub fn close_bus(&self) -> Result<()> {
        self.controller.close_bus()
    }

    fn bus(&self) -> Arc<dyn CanBus> {
        self.controller.bus()
    }

    fn send_std_frame(&self, arbitration_id: u32, payload: [u8; 8]) -> Result<()> {
        self.bus().send(CanFrame {
            arbitration_id,
            data: payload,
            dlc: 8,
            is_extended: false,
            is_rx: false,
        })
    }

    pub fn scan_ids(
        &self,
        start_id: u16,
        end_id: u16,
        host_id: u16,
        timeout: Duration,
    ) -> Result<Vec<RobstrideMitScanHit>> {
        validate_host_id(host_id)?;
        if start_id == 0 || end_id == 0 || start_id > 127 || end_id > 127 || start_id > end_id {
            return Err(MotorError::InvalidArgument(
                "invalid scan range, expected 1..127 and start<=end".to_string(),
            ));
        }
        let mut hits = Vec::new();
        for node in start_id..=end_id {
            validate_node_id(node)?;
            self.send_std_frame(u32::from(node), encode_clear_or_fault_query(0x00))?;
            let deadline = Instant::now() + timeout;
            let mut found = None;
            while Instant::now() < deadline {
                if let Some(frame) = self.bus().recv(Duration::from_millis(2))? {
                    if frame.is_extended
                        || frame.arbitration_id != u32::from(host_id)
                        || frame.dlc < 8
                        || u16::from(frame.data[0]) != node
                    {
                        continue;
                    }
                    found = Some(decode_fault_value(frame.data));
                    break;
                }
            }
            if found.is_some() {
                hits.push(RobstrideMitScanHit {
                    node_id: node,
                    host_id,
                    fault_code: found,
                });
            }
        }
        Ok(hits)
    }
}
