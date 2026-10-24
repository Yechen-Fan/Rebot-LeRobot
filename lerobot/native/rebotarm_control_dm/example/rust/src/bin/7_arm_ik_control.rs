use motor_vendor_damiao::ControlMode;
use rebotarm_control_dm_rust_examples::common::{
    has_flag, move_pos_vel_path, parse_float_arg, parse_floats, parse_port, parse_rate, prompt,
    q_deg, B601Arm, MathModel, DEFAULT_RATE_HZ,
};
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!("Usage: cargo run --bin 7_arm_ik_control -- --port /dev/ttyACM0 --rate 150");
        return Ok(());
    }

    let rate = parse_rate(&args, DEFAULT_RATE_HZ);
    let duration = parse_float_arg(&args, "--duration", 2.0);
    let model = MathModel::load(&rebotarm_control_dm_rust_examples::common::parse_urdf_path(
        &args,
    ))?;
    let arm = B601Arm::open(&parse_port(&args))?;
    arm.enable()?;
    arm.ensure_all_mode(ControlMode::PosVel);
    println!("Connected. Input target x y z in meters, state, or q.");
    println!("Backend: C++/Pinocchio librebotarm_math.so.");

    loop {
        let Some(line) = prompt("target xyz > ")? else {
            break;
        };
        if matches!(line.as_str(), "q" | "quit" | "exit") {
            break;
        }
        if line == "state" {
            arm.print_state();
            continue;
        }
        let values = parse_floats(&line)?;
        if values.len() < 3 {
            println!("need x y z");
            continue;
        }

        let start = arm.positions_or_zero();
        let seed = arm.arm_positions_or_zero();
        let result = model.ik_position_cpp([values[0], values[1], values[2]], &seed, 2000)?;
        println!(
            "  [{}] iterations={} error={:.5}",
            if result.converged {
                "converged"
            } else {
                "not converged"
            },
            result.iterations,
            result.error
        );
        println!("  q(deg): {:?}", q_deg(&result.q));

        let mut end = start.clone();
        for idx in 0..6 {
            end[idx] = result.q[idx] as f32;
        }
        move_pos_vel_path(&arm, &start, &end, duration, rate)?;
    }

    arm.close();
    Ok(())
}
