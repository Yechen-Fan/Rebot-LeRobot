use motor_vendor_damiao::ControlMode;
use rebotarm_control_dm_rust_examples::common::{
    deg_to_rad_f32, has_flag, parse_port, prompt, B601Arm,
};
use std::env;
use std::error::Error;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!("Usage: cargo run --bin gripper_test -- --port /dev/ttyACM0");
        return Ok(());
    }

    let arm = B601Arm::open(&parse_port(&args))?;
    let gripper = &arm.motors[6];
    arm.enable()?;
    gripper.ensure_control_mode(ControlMode::ForcePos, Duration::from_millis(300))?;

    println!("Gripper commands: open / close / pos <deg> / forcepos <deg> [vlim] [ratio] / posvel <deg> [vlim] / state / q");
    loop {
        let Some(line) = prompt("> ")? else {
            break;
        };
        if matches!(line.as_str(), "q" | "quit" | "exit") {
            break;
        }
        if line == "state" {
            arm.print_state();
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.first().copied() {
            Some("open") => gripper.send_cmd_force_pos(0.0, 5.235_987_7, 0.05)?,
            Some("close") => {
                gripper.send_cmd_force_pos(deg_to_rad_f32(-270.0), 5.235_987_7, 0.05)?
            }
            Some("pos") if parts.len() >= 2 => {
                gripper.send_cmd_force_pos(deg_to_rad_f32(parts[1].parse()?), 5.235_987_7, 0.05)?
            }
            Some("forcepos") | Some("force_pos") if parts.len() >= 2 => {
                let pos = deg_to_rad_f32(parts[1].parse()?);
                let vlim = parts
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5.235_987_7);
                let ratio = parts.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.05);
                gripper.send_cmd_force_pos(pos, vlim, ratio)?;
            }
            Some("posvel") | Some("pos_vel") if parts.len() >= 2 => {
                gripper.ensure_control_mode(ControlMode::PosVel, Duration::from_millis(300))?;
                let pos = deg_to_rad_f32(parts[1].parse()?);
                let vlim = parts
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5.235_987_7);
                gripper.send_cmd_pos_vel(pos, vlim)?;
            }
            _ => println!("unknown command"),
        }
    }

    arm.close();
    Ok(())
}
