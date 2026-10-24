use rebotarm_control_dm_rust_examples::common::{
    has_flag, parse_port, prompt, B601Arm, B601_JOINTS,
};
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        println!("Usage: cargo run --bin 2_zero_and_read -- --port /dev/ttyACM0 [--skip-zero]");
        return Ok(());
    }

    let arm = B601Arm::open(&parse_port(&args))?;
    println!("connected: {}", arm.port);

    if !has_flag(&args, "--skip-zero") {
        println!("This will set the current pose as zero for all B601 motors.");
        println!("Type YES to continue.");
        if prompt("confirm> ")?.as_deref() == Some("YES") {
            arm.disable()?;
            for (joint, motor) in B601_JOINTS.iter().zip(&arm.motors) {
                motor.set_zero_position()?;
                println!("zero set: {}", joint.name);
            }
        } else {
            println!("zero skipped");
        }
    }

    println!("Press Enter to read state again, q to quit.");
    loop {
        arm.print_state();
        let Some(line) = prompt("> ")? else {
            break;
        };
        if matches!(line.as_str(), "q" | "quit" | "exit") {
            break;
        }
    }

    arm.close();
    Ok(())
}
