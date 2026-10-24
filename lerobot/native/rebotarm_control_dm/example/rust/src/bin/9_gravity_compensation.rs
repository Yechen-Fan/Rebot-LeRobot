use motor_vendor_damiao::ControlMode;
use rebotarm_control_dm_rust_examples::common::{
    gravity_urdf_for_gripper, has_flag, install_signal_handler, parse_bool_arg, parse_float_arg,
    parse_port, parse_rate, sleep_to_rate, stop_requested, B601Arm, MathModel, DEFAULT_RATE_HZ,
};
use std::env;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!(
            "Usage: cargo run --bin 9_gravity_compensation -- --port /dev/ttyACM0 --rate 200 --use_gripper=true [--kp 0.0] [--kd 1.0]"
        );
        return Ok(());
    }

    let rate = parse_rate(&args, 200.0_f64.min(DEFAULT_RATE_HZ.max(200.0)));
    let use_gripper = parse_bool_arg(&args, "--use_gripper", true);
    let arm_kp = parse_float_arg(&args, "--kp", 0.0) as f32;
    let arm_kd = parse_float_arg(&args, "--kd", 1.0) as f32;
    let gripper_kp = parse_float_arg(&args, "--gripper-kp", 0.0) as f32;
    let gripper_kd = parse_float_arg(&args, "--gripper-kd", 1.0) as f32;
    let (urdf_path, _temp_urdf, end_link_scale) = gravity_urdf_for_gripper(&args, use_gripper)?;
    let model = MathModel::load(&urdf_path)?;
    println!("Rust gravity demo backend: C++/Pinocchio librebotarm_math.so.");
    println!("use_gripper={use_gripper}; end_link inertial scale={end_link_scale:.3}");
    println!(
        "MIT command gains: arm kp={arm_kp:.3}, kd={arm_kd:.3}; gripper kp={gripper_kp:.3}, kd={gripper_kd:.3}"
    );
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

    while !stop_requested() {
        let tick = Instant::now();
        let q = arm.positions_or_zero();
        let q_model: Vec<f64> = q.iter().take(model.nq).map(|v| f64::from(*v)).collect();
        let tau_model = model.generalized_gravity_cpp(&q_model)?;
        for idx in 0..count {
            let tau = tau_model.get(idx).copied().unwrap_or(0.0) as f32;
            let is_gripper = idx >= model.nq;
            let kp = if is_gripper { gripper_kp } else { arm_kp };
            let kd = if is_gripper { gripper_kd } else { arm_kd };
            arm.motors[idx].send_cmd_mit(q[idx], 0.0, kp, kd, tau)?;
        }
        sleep_to_rate(tick, rate);
    }

    arm.close();
    Ok(())
}
