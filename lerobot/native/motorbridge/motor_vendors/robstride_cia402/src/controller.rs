use crate::motor::RobstrideCia402Motor;
use crate::objects::{
    ERROR_CODE, MODES_OF_OPERATION_DISPLAY, NMT_START_REMOTE_NODE, PROTOCOL_CANOPEN, PROTOCOL_MIT,
    PROTOCOL_PRIVATE, PROTOCOL_SWITCH_EXT_ID, STATUSWORD,
};
use motor_core::bus::{open_can_bus, CanBus, CanFrame};
use motor_core::error::{MotorError, Result};
use motor_core::vendor_controller::VendorController;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct RobstrideCia402ScanHit {
    pub node_id: u16,
    pub statusword: Option<u16>,
    pub mode_display: Option<i8>,
    pub error_code: Option<u16>,
}

pub struct RobstrideCia402Controller {
    controller: VendorController<RobstrideCia402Motor>,
}

impl RobstrideCia402Controller {
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
    ) -> Result<Arc<RobstrideCia402Motor>> {
        self.controller.add_motor_with(motor_id, |bus| {
            RobstrideCia402Motor::new(motor_id, feedback_id, model, bus)
        })
    }

    pub fn get_motor(&self, motor_id: u16) -> Result<Arc<RobstrideCia402Motor>> {
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

    fn send_std_frame(&self, arbitration_id: u32, payload: &[u8]) -> Result<()> {
        if payload.len() > 8 {
            return Err(MotorError::InvalidArgument(format!(
                "payload too long: {}, expected <=8",
                payload.len()
            )));
        }
        let mut data = [0u8; 8];
        data[..payload.len()].copy_from_slice(payload);
        self.bus().send(CanFrame {
            arbitration_id,
            data,
            dlc: payload.len() as u8,
            is_extended: false,
            is_rx: false,
        })
    }

    fn send_ext_frame(&self, arbitration_id: u32, payload: &[u8]) -> Result<()> {
        if payload.len() > 8 {
            return Err(MotorError::InvalidArgument(format!(
                "payload too long: {}, expected <=8",
                payload.len()
            )));
        }
        let mut data = [0u8; 8];
        data[..payload.len()].copy_from_slice(payload);
        self.bus().send(CanFrame {
            arbitration_id,
            data,
            dlc: payload.len() as u8,
            is_extended: true,
            is_rx: false,
        })
    }

    fn send_nmt(&self, command: u8, node_id: u8) -> Result<()> {
        self.send_std_frame(crate::objects::NMT_ID, &[command, node_id])
    }

    fn sdo_read_direct(
        &self,
        node_id: u16,
        index: u16,
        subindex: u8,
        timeout: Duration,
    ) -> Result<u32> {
        let req_id = crate::objects::sdo_req_id(node_id);
        let rsp_id = crate::objects::sdo_rsp_id(node_id);
        let req = [
            0x40,
            (index & 0xFF) as u8,
            ((index >> 8) & 0xFF) as u8,
            subindex,
            0,
            0,
            0,
            0,
        ];
        self.send_std_frame(req_id, &req)?;
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(frame) = self.bus().recv(Duration::from_millis(2))? {
                if frame.is_extended || frame.arbitration_id != rsp_id || frame.dlc < 8 {
                    continue;
                }
                if frame.data[1] != (index & 0xFF) as u8
                    || frame.data[2] != ((index >> 8) & 0xFF) as u8
                    || frame.data[3] != subindex
                {
                    continue;
                }
                if frame.data[0] == 0x80 {
                    let abort = u32::from_le_bytes([
                        frame.data[4],
                        frame.data[5],
                        frame.data[6],
                        frame.data[7],
                    ]);
                    return Err(MotorError::Protocol(format!(
                        "sdo abort node={node_id} idx=0x{index:04X} sub=0x{subindex:02X} code=0x{abort:08X}"
                    )));
                }
                if matches!(frame.data[0], 0x43 | 0x4B | 0x4F | 0x47) {
                    return Ok(u32::from_le_bytes([
                        frame.data[4],
                        frame.data[5],
                        frame.data[6],
                        frame.data[7],
                    ]));
                }
            }
        }
        Err(MotorError::Timeout(format!(
            "scan timeout node={node_id} idx=0x{index:04X} sub=0x{subindex:02X}"
        )))
    }

    pub fn scan_ids(
        &self,
        start_id: u16,
        end_id: u16,
        timeout: Duration,
    ) -> Result<Vec<RobstrideCia402ScanHit>> {
        if start_id == 0 || end_id == 0 || start_id > 127 || end_id > 127 || start_id > end_id {
            return Err(MotorError::InvalidArgument(
                "invalid scan range, expected 1..127 and start<=end".to_string(),
            ));
        }
        self.send_nmt(NMT_START_REMOTE_NODE, 0)?;
        let mut hits = Vec::new();
        for node in start_id..=end_id {
            let statusword = self
                .sdo_read_direct(node, STATUSWORD, 0x00, timeout)
                .ok()
                .map(|v| v as u16);
            let mode_display = self
                .sdo_read_direct(node, MODES_OF_OPERATION_DISPLAY, 0x00, timeout)
                .ok()
                .map(|v| v as u8 as i8);
            let error_code = self
                .sdo_read_direct(node, ERROR_CODE, 0x00, timeout)
                .ok()
                .map(|v| v as u16);
            if statusword.is_some() || mode_display.is_some() || error_code.is_some() {
                hits.push(RobstrideCia402ScanHit {
                    node_id: node,
                    statusword,
                    mode_display,
                    error_code,
                });
            }
        }
        Ok(hits)
    }

    pub fn switch_protocol(
        &self,
        protocol_cmd: u8,
        timeout: Duration,
    ) -> Result<Option<(u16, [u8; 8])>> {
        if !matches!(
            protocol_cmd,
            PROTOCOL_PRIVATE | PROTOCOL_CANOPEN | PROTOCOL_MIT
        ) {
            return Err(MotorError::InvalidArgument(format!(
                "invalid RobStride protocol command {protocol_cmd}, expected 0(private), 1(canopen), or 2(mit)"
            )));
        }
        self.send_ext_frame(
            PROTOCOL_SWITCH_EXT_ID,
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, protocol_cmd, 0x00],
        )?;
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(frame) = self.bus().recv(Duration::from_millis(2))? {
                if !frame.is_extended && frame.dlc >= 8 {
                    return Ok(Some((frame.arbitration_id as u16, frame.data)));
                }
            }
        }
        Ok(None)
    }
}
