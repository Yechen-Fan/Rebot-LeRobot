use motor_vendor_damiao::ControlMode;
use rebotarm_control_dm_rust_examples::common::{
    gravity_urdf_for_gripper, has_flag, install_signal_handler, parse_bool_arg, parse_float_arg,
    parse_port, parse_rate, sleep_to_rate, stop_requested, B601Arm, MathModel,
};
use std::env;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!(
            "Usage: cargo run --bin 10_gravity_compensation_lock -- --port /dev/ttyACM0 --rate 200 --use_gripper=true [--lock-kp 8.0] [--lock-kd 1.0]"
        );
        return Ok(());
    }

    let rate = parse_rate(&args, 200.0);
    let use_gripper = parse_bool_arg(&args, "--use_gripper", true);
    let vel_threshold = parse_float_arg(&args, "--vel-threshold", 0.04) as f32;
    let lock_kp = parse_float_arg(&args, "--lock-kp", parse_float_arg(&args, "--kp", 8.0)) as f32;
    let lock_kd = parse_float_arg(&args, "--lock-kd", parse_float_arg(&args, "--kd", 1.0)) as f32;
    let (urdf_path, _temp_urdf, end_link_scale) = gravity_urdf_for_gripper(&args, use_gripper)?;
    let model = MathModel::load(&urdf_path)?;
    println!("Rust gravity-lock demo backend: C++/Pinocchio librebotarm_math.so.");
    println!(
        "use_gripper={use_gripper}; end_link inertial scale={end_link_scale:.3}; vel-threshold={vel_threshold:.3}"
    );
    println!("MIT lock gains: kp={lock_kp:.3}, kd={lock_kd:.3}");
    println!("Ctrl+C to stop and disconnect.");
    install_signal_handler();

    let arm = B601Arm::open(&parse_port(&args))?;
    arm.enable()?;
    if use_gripper {
        arm.ensure_all_mode(ControlMode::Mit);
    } else {
        arm.ensure_arm_mode(ControlMode::Mit);
    }
    let count = if use_gripper { 7 } else { 6 };
    let mut target = arm.positions_or_zero();

    while !stop_requested() {
        let tick = Instant::now();
        let states = arm.states();
        let mut current = target.clone();
        let mut max_vel = 0.0_f32;
        for idx in 0..count {
            if let Some(state) = states.get(idx).and_then(|s| *s) {
                current[idx] = state.pos;
                max_vel = max_vel.max(state.vel.abs());
            }
        }
        if max_vel > vel_threshold {
            target = current.clone();
        }
        let q_model: Vec<f64> = current
            .iter()
            .take(model.nq)
            .map(|v| f64::from(*v))
            .collect();
        let tau_model = model.generalized_gravity_cpp(&q_model)?;
        for idx in 0..count {
            let tau = tau_model.get(idx).copied().unwrap_or(0.0) as f32;
            arm.motors[idx].send_cmd_mit(target[idx], 0.0, lock_kp, lock_kd, tau)?;
        }
        sleep_to_rate(tick, rate);
    }

    arm.close();
    Ok(())
}
