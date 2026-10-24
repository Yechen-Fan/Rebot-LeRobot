use libloading::Library;
use motor_vendor_damiao::{ControlMode, DamiaoController, DamiaoMotor, MotorFeedbackState};
use std::env;
use std::error::Error;
use std::f64::consts::PI;
use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const ARM_DOF: usize = 6;
pub const ALL_DOF: usize = 7;
pub const DEFAULT_PORT: &str = "/dev/ttyACM0";
pub const DEFAULT_RATE_HZ: f64 = 150.0;
pub const DEFAULT_URDF_REL: &str =
    "urdf/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf";
pub const END_LINK_LOAD_SCALE_WITH_GRIPPER: f64 = 0.7;

pub const ARM_LIMITS_RAD: [(f64, f64); ARM_DOF] = [
    (-145.0_f64.to_radians(), 145.0_f64.to_radians()),
    (-170.0_f64.to_radians(), 1.0_f64.to_radians()),
    (-200.0_f64.to_radians(), 1.0_f64.to_radians()),
    (-80.0_f64.to_radians(), 90.0_f64.to_radians()),
    (-90.0_f64.to_radians(), 90.0_f64.to_radians()),
    (-90.0_f64.to_radians(), 90.0_f64.to_radians()),
];

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn handle_signal(_signal: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn install_signal_handler() {
    STOP_REQUESTED.store(false, Ordering::SeqCst);
    #[cfg(unix)]
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
    }
}

pub fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

#[derive(Clone, Copy)]
pub struct JointSpec {
    pub name: &'static str,
    pub motor_id: u16,
    pub feedback_id: u16,
    pub model: &'static str,
    pub mit_kp: f32,
    pub mit_kd: f32,
    pub vlim: f32,
}

pub const B601_JOINTS: [JointSpec; ALL_DOF] = [
    JointSpec {
        name: "shoulder_pan",
        motor_id: 0x01,
        feedback_id: 0x11,
        model: "4340P",
        mit_kp: 120.0,
        mit_kd: 8.0,
        vlim: 2.617_993_8,
    },
    JointSpec {
        name: "shoulder_lift",
        motor_id: 0x02,
        feedback_id: 0x12,
        model: "4340P",
        mit_kp: 120.0,
        mit_kd: 8.0,
        vlim: 2.617_993_8,
    },
    JointSpec {
        name: "elbow_flex",
        motor_id: 0x03,
        feedback_id: 0x13,
        model: "4340P",
        mit_kp: 120.0,
        mit_kd: 8.0,
        vlim: 2.617_993_8,
    },
    JointSpec {
        name: "wrist_flex",
        motor_id: 0x04,
        feedback_id: 0x14,
        model: "4310",
        mit_kp: 18.0,
        mit_kd: 2.0,
        vlim: 2.617_993_8,
    },
    JointSpec {
        name: "wrist_yaw",
        motor_id: 0x05,
        feedback_id: 0x15,
        model: "4310",
        mit_kp: 18.0,
        mit_kd: 2.0,
        vlim: 2.617_993_8,
    },
    JointSpec {
        name: "wrist_roll",
        motor_id: 0x06,
        feedback_id: 0x16,
        model: "4310",
        mit_kp: 18.0,
        mit_kd: 2.0,
        vlim: 2.617_993_8,
    },
    JointSpec {
        name: "gripper",
        motor_id: 0x07,
        feedback_id: 0x17,
        model: "4310",
        mit_kp: 8.0,
        mit_kd: 1.0,
        vlim: 5.235_987_7,
    },
];

pub fn arg_value(args: &[String], name: &str) -> Option<String> {
    let prefix = format!("{name}=");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
        if let Some(value) = arg.strip_prefix(&prefix) {
            return Some(value.to_string());
        }
    }
    None
}

pub fn arg_values(args: &[String], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let prefix = format!("{name}=");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            if let Some(value) = iter.next() {
                out.push(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix(&prefix) {
            out.push(value.to_string());
        }
    }
    out
}

pub fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

pub fn parse_port(args: &[String]) -> String {
    arg_value(args, "--port")
        .or_else(|| arg_value(args, "-p"))
        .unwrap_or_else(|| DEFAULT_PORT.to_string())
}

pub fn parse_rate(args: &[String], default_hz: f64) -> f64 {
    arg_value(args, "--rate")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .unwrap_or(default_hz)
}

pub fn parse_float_arg(args: &[String], name: &str, default: f64) -> f64 {
    arg_value(args, name)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

pub fn parse_bool_arg(args: &[String], name: &str, default: bool) -> bool {
    arg_value(args, name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "True" | "yes" | "on"))
        .unwrap_or(default)
}

pub fn parse_joint(value: &str) -> Result<usize, Box<dyn Error>> {
    if let Ok(index) = value.parse::<usize>() {
        if index < ALL_DOF {
            return Ok(index);
        }
        if (1..=ALL_DOF).contains(&index) {
            return Ok(index - 1);
        }
    }
    if let Some(rest) = value.strip_prefix("joint") {
        let one_based = rest.parse::<usize>()?;
        if (1..=ALL_DOF).contains(&one_based) {
            return Ok(one_based - 1);
        }
    }
    B601_JOINTS
        .iter()
        .position(|joint| joint.name == value)
        .ok_or_else(|| format!("unknown joint: {value}").into())
}

pub fn parse_floats(line: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    line.split_whitespace()
        .map(|part| Ok(part.parse::<f64>()?))
        .collect()
}

pub fn prompt(text: &str) -> io::Result<Option<String>> {
    print!("{text}");
    io::stdout().flush()?;
    let mut line = String::new();
    let n = io::stdin().read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim().to_string()))
}

pub fn deg_to_rad_f32(value: f64) -> f32 {
    value.to_radians() as f32
}

pub fn rad_to_deg_f32(value: f32) -> f32 {
    value.to_degrees()
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn default_urdf_path() -> PathBuf {
    repo_root().join(DEFAULT_URDF_REL)
}

pub fn parse_urdf_path(args: &[String]) -> PathBuf {
    arg_value(args, "--urdf")
        .map(PathBuf::from)
        .unwrap_or_else(default_urdf_path)
}

pub struct TemporaryUrdf {
    path: PathBuf,
}

impl TemporaryUrdf {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryUrdf {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn gravity_urdf_for_gripper(
    args: &[String],
    use_gripper: bool,
) -> Result<(PathBuf, Option<TemporaryUrdf>, f64), Box<dyn Error>> {
    let base_urdf = parse_urdf_path(args);
    let scale = if use_gripper {
        END_LINK_LOAD_SCALE_WITH_GRIPPER
    } else {
        0.0
    };
    if (scale - 1.0).abs() <= f64::EPSILON {
        return Ok((base_urdf, None, scale));
    }
    let temp = end_link_load_urdf(&base_urdf, scale)?;
    Ok((temp.path().to_path_buf(), Some(temp), scale))
}

fn end_link_load_urdf(urdf_path: &Path, scale: f64) -> Result<TemporaryUrdf, Box<dyn Error>> {
    if scale < 0.0 {
        return Err("end_link load scale must be >= 0".into());
    }
    let xml = fs::read_to_string(urdf_path)?;
    let modified = scale_end_link_inertial(&xml, scale)?;
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "rebotarm_control_dm_end_link_{}_{}.urdf",
        process::id(),
        now_ns
    ));
    fs::write(&path, modified)?;
    Ok(TemporaryUrdf { path })
}

fn scale_end_link_inertial(xml: &str, scale: f64) -> Result<String, Box<dyn Error>> {
    let end_link_pos = xml
        .find("name=\"end_link\"")
        .or_else(|| xml.find("name='end_link'"))
        .ok_or("URDF does not contain link name=\"end_link\"")?;
    let inertial_start = end_link_pos
        + xml[end_link_pos..]
            .find("<inertial")
            .ok_or("URDF end_link does not contain an inertial block")?;
    let inertial_open_end = inertial_start
        + xml[inertial_start..]
            .find('>')
            .ok_or("URDF end_link inertial block is malformed")?
        + 1;
    let inertial_end = inertial_open_end
        + xml[inertial_open_end..]
            .find("</inertial>")
            .ok_or("URDF end_link inertial block is not closed")?
        + "</inertial>".len();

    let mut out = String::with_capacity(xml.len());
    out.push_str(&xml[..inertial_start]);
    if scale > 0.0 {
        let block = &xml[inertial_start..inertial_end];
        out.push_str(&scale_inertial_block(block, scale)?);
    }
    out.push_str(&xml[inertial_end..]);
    Ok(out)
}

fn scale_inertial_block(block: &str, scale: f64) -> Result<String, Box<dyn Error>> {
    let mut out = scale_attr_once(block, "mass", "value", scale)?;
    for attr in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"] {
        out = scale_attr_once(&out, "inertia", attr, scale)?;
    }
    Ok(out)
}

fn scale_attr_once(
    source: &str,
    element: &str,
    attr: &str,
    scale: f64,
) -> Result<String, Box<dyn Error>> {
    let elem_start = find_xml_element(source, element)
        .ok_or_else(|| format!("URDF inertial block is missing <{element}>"))?;
    let elem_end = elem_start
        + source[elem_start..]
            .find('>')
            .ok_or_else(|| format!("URDF <{element}> tag is malformed"))?;
    let tag = &source[elem_start..elem_end];
    let (attr_offset, quote, pattern_len) = if let Some(offset) = tag.find(&format!("{attr}=\"")) {
        (offset, '"', attr.len() + 2)
    } else if let Some(offset) = tag.find(&format!("{attr}='")) {
        (offset, '\'', attr.len() + 2)
    } else {
        return Err(format!("URDF <{element}> tag is missing {attr}").into());
    };
    let value_start = elem_start + attr_offset + pattern_len;
    let value_end = value_start
        + source[value_start..]
            .find(quote)
            .ok_or_else(|| format!("URDF <{element}> {attr} quote is not closed"))?;
    let value = source[value_start..value_end].parse::<f64>()?;
    let replacement = format_float(value * scale);

    let mut out = String::with_capacity(source.len() + replacement.len());
    out.push_str(&source[..value_start]);
    out.push_str(&replacement);
    out.push_str(&source[value_end..]);
    Ok(out)
}

fn find_xml_element(source: &str, element: &str) -> Option<usize> {
    let needle = format!("<{element}");
    let mut offset = 0;
    while let Some(rel) = source[offset..].find(&needle) {
        let pos = offset + rel;
        let next = source[pos + needle.len()..].chars().next();
        if matches!(next, Some(' ' | '\t' | '\r' | '\n' | '/' | '>')) {
            return Some(pos);
        }
        offset = pos + needle.len();
    }
    None
}

fn format_float(value: f64) -> String {
    let mut text = format!("{value:.10}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    text
}

pub fn open_controller(port: &str) -> Result<DamiaoController, Box<dyn Error>> {
    let lower = port.to_ascii_lowercase();
    if port.starts_with("/dev/") || lower.starts_with("com") {
        Ok(DamiaoController::new_dm_serial(port, 921_600)?)
    } else {
        Ok(DamiaoController::new_socketcan(port)?)
    }
}

pub struct B601Arm {
    pub port: String,
    controller: DamiaoController,
    pub motors: Vec<Arc<DamiaoMotor>>,
}

impl B601Arm {
    pub fn open(port: &str) -> Result<Self, Box<dyn Error>> {
        let controller = open_controller(port)?;
        let mut motors = Vec::with_capacity(B601_JOINTS.len());
        for joint in B601_JOINTS {
            motors.push(controller.add_motor(joint.motor_id, joint.feedback_id, joint.model)?);
        }
        Ok(Self {
            port: port.to_string(),
            controller,
            motors,
        })
    }

    pub fn enable(&self) -> Result<(), Box<dyn Error>> {
        for motor in &self.motors {
            let _ = motor.clear_error();
        }
        self.controller.enable_all()?;
        Ok(())
    }

    pub fn disable(&self) -> Result<(), Box<dyn Error>> {
        self.controller.disable_all()?;
        Ok(())
    }

    pub fn close(&self) {
        let _ = self.controller.disable_all();
        thread::sleep(Duration::from_millis(20));
        let _ = self.controller.shutdown();
        let _ = self.controller.close_bus();
    }

    pub fn ensure_all_mode(&self, mode: ControlMode) {
        let timeout = Duration::from_millis(300);
        for (joint, motor) in B601_JOINTS.iter().zip(&self.motors) {
            if let Err(err) = motor.ensure_control_mode(mode, timeout) {
                eprintln!("warning: {} mode switch failed: {err}", joint.name);
            }
        }
    }

    pub fn ensure_arm_mode(&self, mode: ControlMode) {
        let timeout = Duration::from_millis(300);
        for (joint, motor) in B601_JOINTS.iter().zip(&self.motors).take(ARM_DOF) {
            if let Err(err) = motor.ensure_control_mode(mode, timeout) {
                eprintln!("warning: {} mode switch failed: {err}", joint.name);
            }
        }
    }

    pub fn request_feedback(&self) {
        for motor in &self.motors {
            let _ = motor.request_motor_feedback();
        }
        thread::sleep(Duration::from_millis(20));
    }

    pub fn states(&self) -> Vec<Option<MotorFeedbackState>> {
        self.request_feedback();
        self.motors
            .iter()
            .map(|motor| motor.latest_state())
            .collect()
    }

    pub fn positions_or_zero(&self) -> Vec<f32> {
        self.states()
            .into_iter()
            .map(|state| state.map(|s| s.pos).unwrap_or(0.0))
            .collect()
    }

    pub fn arm_positions_or_zero(&self) -> [f64; ARM_DOF] {
        let mut q = [0.0_f64; ARM_DOF];
        for (idx, value) in self
            .positions_or_zero()
            .into_iter()
            .take(ARM_DOF)
            .enumerate()
        {
            q[idx] = value as f64;
        }
        q
    }

    pub fn print_state(&self) {
        for (joint, state) in B601_JOINTS.iter().zip(self.states()) {
            match state {
                Some(s) => println!(
                    "{:<14} pos={:>8.2} deg vel={:>8.2} deg/s torque={:>8.3} status={}",
                    joint.name,
                    rad_to_deg_f32(s.pos),
                    rad_to_deg_f32(s.vel),
                    s.torq,
                    s.status_name
                ),
                None => println!("{:<14} no feedback", joint.name),
            }
        }
    }

    pub fn send_mit_all(
        &self,
        pos: &[f32],
        vel: &[f32],
        kp: &[f32],
        kd: &[f32],
        tau: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        for idx in 0..self.motors.len() {
            self.motors[idx].send_cmd_mit(
                pos.get(idx).copied().unwrap_or(0.0),
                vel.get(idx).copied().unwrap_or(0.0),
                kp.get(idx).copied().unwrap_or(B601_JOINTS[idx].mit_kp),
                kd.get(idx).copied().unwrap_or(B601_JOINTS[idx].mit_kd),
                tau.get(idx).copied().unwrap_or(0.0),
            )?;
        }
        Ok(())
    }

    pub fn send_pos_vel_all(&self, pos: &[f32], vlim: &[f32]) -> Result<(), Box<dyn Error>> {
        for idx in 0..self.motors.len() {
            self.motors[idx].send_cmd_pos_vel(
                pos.get(idx).copied().unwrap_or(0.0),
                vlim.get(idx).copied().unwrap_or(B601_JOINTS[idx].vlim),
            )?;
        }
        Ok(())
    }
}

pub fn sleep_to_rate(start: Instant, rate_hz: f64) {
    let period = Duration::from_secs_f64(1.0 / rate_hz.max(1.0));
    if let Some(remaining) = period.checked_sub(start.elapsed()) {
        thread::sleep(remaining);
    }
}

pub fn default_kp() -> Vec<f32> {
    B601_JOINTS.iter().map(|joint| joint.mit_kp).collect()
}

pub fn default_kd() -> Vec<f32> {
    B601_JOINTS.iter().map(|joint| joint.mit_kd).collect()
}

pub fn default_vlim() -> Vec<f32> {
    B601_JOINTS.iter().map(|joint| joint.vlim).collect()
}

pub fn move_pos_vel_path(
    arm: &B601Arm,
    start: &[f32],
    end: &[f32],
    duration_s: f64,
    rate_hz: f64,
) -> Result<(), Box<dyn Error>> {
    let steps = (duration_s.max(0.02) * rate_hz.max(1.0)).ceil() as usize;
    let vlim = default_vlim();
    for step in 1..=steps {
        let tick = Instant::now();
        let alpha = step as f32 / steps as f32;
        let mut target = vec![0.0_f32; ALL_DOF];
        for idx in 0..ALL_DOF {
            let s = start.get(idx).copied().unwrap_or(0.0);
            let e = end.get(idx).copied().unwrap_or(s);
            target[idx] = s + (e - s) * alpha;
        }
        arm.send_pos_vel_all(&target, &vlim)?;
        sleep_to_rate(tick, rate_hz);
    }
    Ok(())
}

pub fn run_single_motor_console() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!("Usage: cargo run --bin 0x01damiao_test -- --port /dev/ttyACM0 --joint 0");
        return Ok(());
    }

    let port = parse_port(&args);
    let joint_arg = arg_value(&args, "--joint")
        .or_else(|| arg_value(&args, "-j"))
        .unwrap_or_else(|| "0".to_string());
    let joint_idx = parse_joint(&joint_arg)?;
    let arm = B601Arm::open(&port)?;

    println!("connected: B601 on {}", arm.port);
    println!("joint: {} ({})", joint_idx, B601_JOINTS[joint_idx].name);
    println!(
        "commands: enable / disable / set_zero / mode / mit / posvel / vel / forcepos / state / q"
    );
    println!("examples: mit 10 0 20 2 0 | posvel 10 1.0 | vel 0.2 | forcepos -120 3.0 0.05");

    let mut target = arm.positions_or_zero();
    arm.enable()?;
    arm.ensure_all_mode(ControlMode::Mit);

    loop {
        let Some(line) = prompt("> ")? else {
            break;
        };
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if matches!(lower.as_str(), "q" | "quit" | "exit") {
            break;
        }
        if lower == "enable" {
            arm.enable()?;
            println!("enabled");
            continue;
        }
        if lower == "disable" {
            arm.disable()?;
            println!("disabled");
            continue;
        }
        if lower == "state" {
            arm.print_state();
            continue;
        }
        if lower == "set_zero" {
            println!("set zero requires disabled motor. Type YES to continue.");
            if prompt("confirm> ")?.as_deref() == Some("YES") {
                arm.motors[joint_idx].disable()?;
                arm.motors[joint_idx].set_zero_position()?;
                println!("zero set for {}", B601_JOINTS[joint_idx].name);
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.first().copied() {
            Some("mode") if parts.len() >= 2 => {
                match parts[1] {
                    "mit" => arm.motors[joint_idx]
                        .ensure_control_mode(ControlMode::Mit, Duration::from_millis(300))?,
                    "posvel" | "pos_vel" => arm.motors[joint_idx]
                        .ensure_control_mode(ControlMode::PosVel, Duration::from_millis(300))?,
                    "vel" => arm.motors[joint_idx]
                        .ensure_control_mode(ControlMode::Vel, Duration::from_millis(300))?,
                    "forcepos" | "force_pos" => arm.motors[joint_idx]
                        .ensure_control_mode(ControlMode::ForcePos, Duration::from_millis(300))?,
                    other => {
                        println!("unknown mode: {other}");
                        continue;
                    }
                }
                println!("mode set for {}", B601_JOINTS[joint_idx].name);
            }
            Some("mit") if parts.len() >= 2 => {
                let pos = deg_to_rad_f32(parts[1].parse()?);
                let vel = parts.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                let kp = parts
                    .get(3)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(B601_JOINTS[joint_idx].mit_kp);
                let kd = parts
                    .get(4)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(B601_JOINTS[joint_idx].mit_kd);
                let tau = parts.get(5).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                target[joint_idx] = pos;
                arm.motors[joint_idx].send_cmd_mit(pos, vel, kp, kd, tau)?;
                println!("sent MIT target {:.2} deg", parts[1].parse::<f64>()?);
            }
            Some("posvel") | Some("pos_vel") if parts.len() >= 2 => {
                let pos = deg_to_rad_f32(parts[1].parse()?);
                let vlim = parts
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(B601_JOINTS[joint_idx].vlim);
                target[joint_idx] = pos;
                arm.motors[joint_idx].send_cmd_pos_vel(pos, vlim)?;
                println!(
                    "sent POS_VEL target {:.2} deg, vlim={vlim:.3}",
                    parts[1].parse::<f64>()?
                );
            }
            Some("vel") if parts.len() >= 2 => {
                let vel = parts[1].parse::<f32>()?;
                arm.motors[joint_idx].send_cmd_vel(vel)?;
                println!("sent velocity {vel:.3} rad/s");
            }
            Some("forcepos") | Some("force_pos") if parts.len() >= 2 => {
                let pos = deg_to_rad_f32(parts[1].parse()?);
                let vlim = parts
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(B601_JOINTS[joint_idx].vlim);
                let ratio = parts.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.05);
                target[joint_idx] = pos;
                arm.motors[joint_idx].send_cmd_force_pos(pos, vlim, ratio)?;
                println!(
                    "sent FORCE_POS target {:.2} deg, vlim={vlim:.3}, ratio={ratio:.3}",
                    parts[1].parse::<f64>()?
                );
            }
            _ => println!("unknown command"),
        }
    }

    arm.close();
    Ok(())
}

pub type Mat4 = [[f64; 4]; 4];

fn eye() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn translation(x: f64, y: f64, z: f64) -> Mat4 {
    let mut t = eye();
    t[0][3] = x;
    t[1][3] = y;
    t[2][3] = z;
    t
}

fn rot_x(a: f64) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rot_y(a: f64) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rot_z(a: f64) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rpy(roll: f64, pitch: f64, yaw: f64) -> Mat4 {
    mat_mul(mat_mul(rot_z(yaw), rot_y(pitch)), rot_x(roll))
}

fn axis_angle(axis: [f64; 3], angle: f64) -> Mat4 {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm <= f64::EPSILON {
        return eye();
    }
    let x = axis[0] / norm;
    let y = axis[1] / norm;
    let z = axis[2] / norm;
    let (s, c) = angle.sin_cos();
    let v = 1.0 - c;
    [
        [x * x * v + c, x * y * v - z * s, x * z * v + y * s, 0.0],
        [y * x * v + z * s, y * y * v + c, y * z * v - x * s, 0.0],
        [z * x * v - y * s, z * y * v + x * s, z * z * v + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

const JOINT_ORIGINS: [([f64; 3], [f64; 3], [f64; 3]); ARM_DOF] = [
    (
        [-0.000_084_16, 0.0, 0.084_65],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ),
    (
        [0.020_084, 0.031_625, 0.055_55],
        [-1.5708, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ),
    ([-0.264, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    (
        [0.2426, -0.054, -0.001_625],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ),
    (
        [0.078_308, -0.0375, -0.03],
        [-1.5708, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ),
    ([0.028_008, 0.0, 0.04], [0.0, 1.5708, 0.0], [0.0, 0.0, 1.0]),
];

pub fn fk(q: &[f64; ARM_DOF]) -> Mat4 {
    let mut t = eye();
    for idx in 0..ARM_DOF {
        let (xyz, rpy0, axis) = JOINT_ORIGINS[idx];
        t = mat_mul(t, translation(xyz[0], xyz[1], xyz[2]));
        t = mat_mul(t, rpy(rpy0[0], rpy0[1], rpy0[2]));
        t = mat_mul(t, axis_angle(axis, q[idx]));
    }
    t = mat_mul(t, translation(0.0, 0.0, 0.155_39));
    t = mat_mul(t, rpy(0.0, -1.5708, PI));
    t
}

pub fn pose_xyz(t: &Mat4) -> [f64; 3] {
    [t[0][3], t[1][3], t[2][3]]
}

pub fn pose_rpy(t: &Mat4) -> [f64; 3] {
    let sy = (-t[2][0]).clamp(-1.0, 1.0);
    let pitch = sy.asin();
    let cp = pitch.cos();
    if cp.abs() > 1e-8 {
        [t[2][1].atan2(t[2][2]), pitch, t[1][0].atan2(t[0][0])]
    } else {
        [0.0, pitch, (-t[0][1]).atan2(t[1][1])]
    }
}

pub struct IkResult {
    pub q: [f64; ARM_DOF],
    pub error: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub fn solve_ik_position(target_xyz: [f64; 3], seed: [f64; ARM_DOF], max_iter: usize) -> IkResult {
    let mut q = seed;
    clamp_q(&mut q);
    let eps = 1e-4;
    let mut err_norm = f64::INFINITY;
    let mut iter_done = 0;

    for iter in 0..max_iter {
        iter_done = iter + 1;
        let current = pose_xyz(&fk(&q));
        let err = [
            target_xyz[0] - current[0],
            target_xyz[1] - current[1],
            target_xyz[2] - current[2],
        ];
        err_norm = (err[0] * err[0] + err[1] * err[1] + err[2] * err[2]).sqrt();
        if err_norm < 1e-4 {
            return IkResult {
                q,
                error: err_norm,
                iterations: iter_done,
                converged: true,
            };
        }

        let mut grad = [0.0_f64; ARM_DOF];
        for j in 0..ARM_DOF {
            let mut q2 = q;
            q2[j] += eps;
            let p2 = pose_xyz(&fk(&q2));
            let jac = [
                (p2[0] - current[0]) / eps,
                (p2[1] - current[1]) / eps,
                (p2[2] - current[2]) / eps,
            ];
            grad[j] = jac[0] * err[0] + jac[1] * err[1] + jac[2] * err[2];
        }

        for j in 0..ARM_DOF {
            let step = (0.8 * grad[j]).clamp(-0.08, 0.08);
            q[j] += step;
        }
        clamp_q(&mut q);
    }

    IkResult {
        q,
        error: err_norm,
        iterations: iter_done,
        converged: false,
    }
}

pub fn clamp_q(q: &mut [f64; ARM_DOF]) {
    for idx in 0..ARM_DOF {
        q[idx] = q[idx].clamp(ARM_LIMITS_RAD[idx].0, ARM_LIMITS_RAD[idx].1);
    }
}

pub fn q_deg(q: &[f64; ARM_DOF]) -> [f64; ARM_DOF] {
    let mut out = [0.0; ARM_DOF];
    for idx in 0..ARM_DOF {
        out[idx] = q[idx].to_degrees();
    }
    out
}

pub fn q_rad_from_deg(values: &[f64]) -> [f64; ARM_DOF] {
    let mut out = [0.0; ARM_DOF];
    for idx in 0..ARM_DOF.min(values.len()) {
        out[idx] = values[idx].to_radians();
    }
    out
}

pub fn print_pose(q: &[f64; ARM_DOF]) {
    let t = fk(q);
    let xyz = pose_xyz(&t);
    let rpy0 = pose_rpy(&t);
    println!(
        "  ee position: [{:+.4}, {:+.4}, {:+.4}] m",
        xyz[0], xyz[1], xyz[2]
    );
    println!(
        "  ee rpy:      [{:+.2}, {:+.2}, {:+.2}] deg",
        rpy0[0].to_degrees(),
        rpy0[1].to_degrees(),
        rpy0[2].to_degrees()
    );
}

pub fn approx_gravity_torque(q: &[f32], scale: f32) -> Vec<f32> {
    let mut tau = vec![0.0_f32; ALL_DOF];
    if q.len() >= 4 {
        tau[1] = -2.5 * scale * q[1].sin();
        tau[2] = -1.6 * scale * (q[1] + q[2]).sin();
        tau[3] = -0.4 * scale * (q[1] + q[2] + q[3]).sin();
    }
    tau
}

type ModelNewFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type ModelFreeFn = unsafe extern "C" fn(*mut c_void);
type ModelNqFn = unsafe extern "C" fn(*const c_void) -> c_int;
type EndFrameIdFn = unsafe extern "C" fn(*const c_void) -> c_int;
type NeutralFn = unsafe extern "C" fn(*const c_void, *mut c_double, c_int) -> c_int;
type FkFn = unsafe extern "C" fn(
    *const c_void,
    *const c_double,
    c_int,
    *const c_char,
    *mut c_double,
    *mut c_double,
    *mut c_double,
) -> c_int;
type IkFn = unsafe extern "C" fn(
    *const c_void,
    *const c_double,
    *const c_double,
    c_int,
    c_int,
    c_int,
    c_double,
    c_double,
    c_double,
    *mut c_double,
    *mut CIkResult,
) -> c_int;
type GravityFn =
    unsafe extern "C" fn(*const c_void, *const c_double, c_int, *mut c_double, c_int) -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CIkResult {
    success: c_int,
    error: c_double,
    iterations: c_int,
}

struct MathApi {
    _lib: Library,
    model_new: ModelNewFn,
    model_free: ModelFreeFn,
    model_nq: ModelNqFn,
    end_frame_id: EndFrameIdFn,
    neutral: NeutralFn,
    fk: FkFn,
    ik: IkFn,
    gravity: GravityFn,
    last_error: LastErrorFn,
}

impl MathApi {
    fn load() -> Result<Arc<Self>, Box<dyn Error>> {
        let lib_path = math_lib_path();
        let lib = unsafe { Library::new(&lib_path)? };
        unsafe {
            let model_new = *lib.get::<ModelNewFn>(b"rebotarm_math_model_new\0")?;
            let model_free = *lib.get::<ModelFreeFn>(b"rebotarm_math_model_free\0")?;
            let model_nq = *lib.get::<ModelNqFn>(b"rebotarm_math_model_nq\0")?;
            let end_frame_id = *lib.get::<EndFrameIdFn>(b"rebotarm_math_end_frame_id\0")?;
            let neutral = *lib.get::<NeutralFn>(b"rebotarm_math_neutral\0")?;
            let fk = *lib.get::<FkFn>(b"rebotarm_math_fk\0")?;
            let ik = *lib.get::<IkFn>(b"rebotarm_math_ik\0")?;
            let gravity = *lib.get::<GravityFn>(b"rebotarm_math_generalized_gravity\0")?;
            let last_error = *lib.get::<LastErrorFn>(b"rebotarm_math_last_error\0")?;
            Ok(Arc::new(Self {
                _lib: lib,
                model_new,
                model_free,
                model_nq,
                end_frame_id,
                neutral,
                fk,
                ik,
                gravity,
                last_error,
            }))
        }
    }

    fn last_error(&self) -> String {
        unsafe {
            let ptr = (self.last_error)();
            if ptr.is_null() {
                return "unknown C++ math error".to_string();
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub struct MathModel {
    api: Arc<MathApi>,
    handle: *mut c_void,
    pub nq: usize,
    pub end_frame_id: i32,
}

impl MathModel {
    pub fn load(urdf_path: &Path) -> Result<Self, Box<dyn Error>> {
        let api = MathApi::load()?;
        let urdf = CString::new(urdf_path.to_string_lossy().as_bytes())?;
        let handle = unsafe { (api.model_new)(urdf.as_ptr()) };
        if handle.is_null() {
            return Err(api.last_error().into());
        }
        let nq = unsafe { (api.model_nq)(handle) };
        if nq <= 0 {
            return Err("invalid nq returned by C++ math model".into());
        }
        let end_frame_id = unsafe { (api.end_frame_id)(handle) };
        if end_frame_id < 0 {
            return Err(api.last_error().into());
        }
        Ok(Self {
            api,
            handle,
            nq: nq as usize,
            end_frame_id,
        })
    }

    pub fn default() -> Result<Self, Box<dyn Error>> {
        Self::load(&default_urdf_path())
    }

    pub fn neutral(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        let mut q = vec![0.0_f64; self.nq];
        let rc = unsafe { (self.api.neutral)(self.handle, q.as_mut_ptr(), q.len() as c_int) };
        if rc != 0 {
            return Err(self.api.last_error().into());
        }
        Ok(q)
    }

    pub fn fk_cpp(&self, q: &[f64]) -> Result<Pose, Box<dyn Error>> {
        let frame = CString::new("")?;
        let mut xyz = [0.0_f64; 3];
        let mut rpy = [0.0_f64; 3];
        let mut raw_t = [0.0_f64; 16];
        let rc = unsafe {
            (self.api.fk)(
                self.handle,
                q.as_ptr(),
                q.len() as c_int,
                frame.as_ptr(),
                xyz.as_mut_ptr(),
                rpy.as_mut_ptr(),
                raw_t.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(self.api.last_error().into());
        }
        Ok(Pose {
            xyz,
            rpy,
            matrix: row_major_to_mat4(raw_t),
        })
    }

    pub fn ik_position_cpp(
        &self,
        target_xyz: [f64; 3],
        seed: &[f64],
        max_iter: usize,
    ) -> Result<IkResult, Box<dyn Error>> {
        let mut target = self.fk_cpp(seed)?.matrix;
        target[0][3] = target_xyz[0];
        target[1][3] = target_xyz[1];
        target[2][3] = target_xyz[2];
        self.ik_matrix_cpp(target, seed, max_iter)
    }

    pub fn ik_matrix_cpp(
        &self,
        target: Mat4,
        seed: &[f64],
        max_iter: usize,
    ) -> Result<IkResult, Box<dyn Error>> {
        let mut q = vec![0.0_f64; self.nq];
        let mut result = CIkResult::default();
        let raw_target = mat4_to_row_major(target);
        let rc = unsafe {
            (self.api.ik)(
                self.handle,
                raw_target.as_ptr(),
                seed.as_ptr(),
                seed.len() as c_int,
                self.end_frame_id as c_int,
                max_iter as c_int,
                1e-4,
                0.5,
                1e-6,
                q.as_mut_ptr(),
                &mut result,
            )
        };
        if rc != 0 {
            return Err(self.api.last_error().into());
        }
        Ok(IkResult {
            q: vec_to_q6(&q),
            error: result.error,
            iterations: result.iterations.max(0) as usize,
            converged: result.success != 0,
        })
    }

    pub fn generalized_gravity_cpp(&self, q: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
        let mut tau = vec![0.0_f64; self.nq];
        let rc = unsafe {
            (self.api.gravity)(
                self.handle,
                q.as_ptr(),
                q.len() as c_int,
                tau.as_mut_ptr(),
                tau.len() as c_int,
            )
        };
        if rc != 0 {
            return Err(self.api.last_error().into());
        }
        Ok(tau)
    }
}

impl Drop for MathModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.api.model_free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

pub struct Pose {
    pub xyz: [f64; 3],
    pub rpy: [f64; 3],
    pub matrix: Mat4,
}

fn math_lib_path() -> PathBuf {
    if let Ok(path) = env::var("REBOTARM_MATH_LIB") {
        return PathBuf::from(path);
    }
    let source_tree = repo_root().join("python/rebotarm_control_dm/librebotarm_math.so");
    if source_tree.exists() {
        return source_tree;
    }
    PathBuf::from("librebotarm_math.so")
}

fn row_major_to_mat4(raw: [f64; 16]) -> Mat4 {
    let mut out = [[0.0_f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = raw[r * 4 + c];
        }
    }
    out
}

fn mat4_to_row_major(matrix: Mat4) -> [f64; 16] {
    let mut out = [0.0_f64; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = matrix[r][c];
        }
    }
    out
}

fn vec_to_q6(values: &[f64]) -> [f64; ARM_DOF] {
    let mut out = [0.0_f64; ARM_DOF];
    for idx in 0..ARM_DOF.min(values.len()) {
        out[idx] = values[idx];
    }
    out
}

pub fn print_pose_with_model(model: &MathModel, q: &[f64]) -> Result<(), Box<dyn Error>> {
    let pose = model.fk_cpp(q)?;
    println!(
        "  ee position: [{:+.4}, {:+.4}, {:+.4}] m",
        pose.xyz[0], pose.xyz[1], pose.xyz[2]
    );
    println!(
        "  ee rpy:      [{:+.2}, {:+.2}, {:+.2}] deg",
        pose.rpy[0].to_degrees(),
        pose.rpy[1].to_degrees(),
        pose.rpy[2].to_degrees()
    );
    Ok(())
}
