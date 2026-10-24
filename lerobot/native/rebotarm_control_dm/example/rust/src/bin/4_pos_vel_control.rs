use motor_vendor_damiao::ControlMode;
use rebotarm_control_dm_rust_examples::common::{
    default_vlim, has_flag, parse_floats, parse_port, parse_rate, prompt, sleep_to_rate, B601Arm,
    ALL_DOF, B601_JOINTS, DEFAULT_RATE_HZ,
};
use std::env;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!("Usage: cargo run --bin 4_pos_vel_control -- --port /dev/ttyACM0 --rate 150");
        return Ok(());
    }

    let rate = parse_rate(&args, DEFAULT_RATE_HZ);
    let arm = B601Arm::open(&parse_port(&args))?;
    arm.enable()?;
    arm.ensure_all_mode(ControlMode::PosVel);

    let target = Arc::new(Mutex::new(arm.positions_or_zero()));
    let vlim = Arc::new(Mutex::new(default_vlim()));
    let running = Arc::new(AtomicBool::new(true));
    let motors = arm.motors.clone();
    let target_rt = Arc::clone(&target);
    let vlim_rt = Arc::clone(&vlim);
    let running_rt = Arc::clone(&running);

    let handle = thread::spawn(move || {
        while running_rt.load(Ordering::Relaxed) {
            let tick = Instant::now();
            let target = target_rt.lock().map(|v| v.clone()).unwrap_or_default();
            let vlim = vlim_rt.lock().map(|v| v.clone()).unwrap_or_default();
            for idx in 0..motors.len() {
                let _ = motors[idx].send_cmd_pos_vel(
                    target.get(idx).copied().unwrap_or(0.0),
                    vlim.get(idx).copied().unwrap_or(B601_JOINTS[idx].vlim),
                );
            }
            sleep_to_rate(tick, rate);
        }
    });

    println!("POS_VEL loop started at {rate:.1} Hz.");
    println!("Input: q1 ... q6 [gripper] [vlim], state, or q. Angles are degrees.");
    loop {
        let Some(line) = prompt("> ")? else {
            break;
        };
        if line.is_empty() {
            continue;
        }
        if matches!(line.as_str(), "q" | "quit" | "exit") {
            break;
        }
        if line == "state" {
            arm.print_state();
            continue;
        }
        let values = parse_floats(&line)?;
        if values.len() < 6 {
            println!("need at least 6 joint angles");
            continue;
        }
        {
            let mut target = target.lock().expect("target lock");
            for idx in 0..ALL_DOF.min(values.len()) {
                target[idx] = values[idx].to_radians() as f32;
            }
        }
        if values.len() >= ALL_DOF + 1 {
            vlim.lock().expect("vlim lock").fill(values[ALL_DOF] as f32);
        }
    }

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    arm.close();
    Ok(())
}
