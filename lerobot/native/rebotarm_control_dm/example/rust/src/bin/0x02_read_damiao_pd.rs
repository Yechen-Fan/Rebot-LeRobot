use rebotarm_control_dm_rust_examples::common::{
    arg_value, arg_values, has_flag, open_controller, B601_JOINTS,
};
use std::env;
use std::error::Error;
use std::time::Duration;

struct Target {
    label: String,
    port: String,
}

fn usage() {
    println!(
        "Read Damiao POS_VEL gain registers from B601 motors.\n\
Usage:\n\
  cargo run --bin 0x02_read_damiao_pd -- --default-bi [--timeout-ms 300]\n\
  cargo run --bin 0x02_read_damiao_pd -- --port /dev/ttyACM0 [--port /dev/ttyACM1]\n\
\n\
Registers: 25 vel_kp, 26 vel_ki, 27 pos_kp, 28 pos_ki"
    );
}

fn read_target(target: &Target, timeout: Duration) -> Result<(), Box<dyn Error>> {
    println!("\n[{}] {}", target.label, target.port);
    let controller = open_controller(&target.port)?;
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10}",
        "joint", "vel_kp", "vel_ki", "pos_kp", "pos_ki"
    );

    for joint in B601_JOINTS {
        let motor = controller.add_motor(joint.motor_id, joint.feedback_id, joint.model)?;
        let values = [25_u8, 26, 27, 28].map(|rid| {
            motor
                .get_register_f32(rid, timeout)
                .map(|v| format!("{v:>10.6}"))
                .unwrap_or_else(|err| format!("{:>10}", format!("ERR({err})")))
        });
        println!(
            "{:<16} {} {} {} {}",
            joint.name, values[0], values[1], values[2], values[3]
        );
    }

    let _ = controller.close_bus();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        usage();
        return Ok(());
    }

    let timeout_ms = arg_value(&args, "--timeout-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
    let mut targets = Vec::new();
    if has_flag(&args, "--default-bi") {
        targets.push(Target {
            label: "left".to_string(),
            port: arg_value(&args, "--left-port").unwrap_or_else(|| "/dev/ttyACM0".to_string()),
        });
        targets.push(Target {
            label: "right".to_string(),
            port: arg_value(&args, "--right-port").unwrap_or_else(|| "/dev/ttyACM1".to_string()),
        });
    }
    for (idx, port) in arg_values(&args, "--port").into_iter().enumerate() {
        targets.push(Target {
            label: format!("port{}", idx + 1),
            port,
        });
    }

    if targets.is_empty() {
        usage();
        return Err("no target specified; pass --default-bi or --port".into());
    }

    let timeout = Duration::from_millis(timeout_ms);
    for target in &targets {
        read_target(target, timeout)?;
    }
    Ok(())
}
