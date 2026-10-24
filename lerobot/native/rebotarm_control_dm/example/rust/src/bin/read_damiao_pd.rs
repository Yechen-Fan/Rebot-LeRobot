use motor_vendor_damiao::{DamiaoController, DamiaoMotor};
use std::env;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

const B601_JOINTS: &[Joint] = &[
    Joint {
        name: "shoulder_pan",
        motor_id: 0x01,
        feedback_id: 0x11,
        model: "4340P",
    },
    Joint {
        name: "shoulder_lift",
        motor_id: 0x02,
        feedback_id: 0x12,
        model: "4340P",
    },
    Joint {
        name: "elbow_flex",
        motor_id: 0x03,
        feedback_id: 0x13,
        model: "4340P",
    },
    Joint {
        name: "wrist_flex",
        motor_id: 0x04,
        feedback_id: 0x14,
        model: "4310",
    },
    Joint {
        name: "wrist_yaw",
        motor_id: 0x05,
        feedback_id: 0x15,
        model: "4310",
    },
    Joint {
        name: "wrist_roll",
        motor_id: 0x06,
        feedback_id: 0x16,
        model: "4310",
    },
    Joint {
        name: "gripper",
        motor_id: 0x07,
        feedback_id: 0x17,
        model: "4310",
    },
];

#[derive(Clone, Copy)]
struct Joint {
    name: &'static str,
    motor_id: u16,
    feedback_id: u16,
    model: &'static str,
}

struct Target {
    label: String,
    port: String,
}

struct Args {
    targets: Vec<Target>,
    timeout_ms: u64,
}

fn usage() {
    println!(
        "Read Damiao POS_VEL gain registers from B601 motors.\n\
\n\
Usage:\n\
  cargo run --bin read_damiao_pd -- --default-bi [--timeout-ms 300]\n\
  cargo run --bin read_damiao_pd -- --default-bi --left-port /dev/ttyACM0 --right-port /dev/ttyACM1\n\
  cargo run --bin read_damiao_pd -- --port /dev/ttyACM0 [--port /dev/ttyACM1]\n\
\n\
Registers:\n\
  25 vel_kp / KP_ASR\n\
  26 vel_ki / KI_ASR\n\
  27 pos_kp / KP_APR\n\
  28 pos_ki / KI_APR"
    );
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = env::args().skip(1);
    let mut timeout_ms = 300_u64;
    let mut default_bi = false;
    let mut left_port = "/dev/ttyACM0".to_string();
    let mut right_port = "/dev/ttyACM1".to_string();
    let mut explicit_ports = Vec::<String>::new();

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            "--timeout-ms" => {
                let value = raw.next().ok_or("--timeout-ms requires a value")?;
                timeout_ms = value.parse()?;
            }
            "--default-bi" => default_bi = true,
            "--left-port" => left_port = raw.next().ok_or("--left-port requires a value")?,
            "--right-port" => right_port = raw.next().ok_or("--right-port requires a value")?,
            "--port" => explicit_ports.push(raw.next().ok_or("--port requires a value")?),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let mut targets = Vec::new();
    if default_bi {
        targets.push(Target {
            label: "left".to_string(),
            port: left_port,
        });
        targets.push(Target {
            label: "right".to_string(),
            port: right_port,
        });
    }
    for (idx, port) in explicit_ports.into_iter().enumerate() {
        targets.push(Target {
            label: format!("port{}", idx + 1),
            port,
        });
    }

    if targets.is_empty() {
        usage();
        return Err("no target specified; pass --default-bi or --port".into());
    }

    Ok(Args {
        targets,
        timeout_ms,
    })
}

fn open_controller(port: &str) -> Result<DamiaoController, Box<dyn Error>> {
    let lower = port.to_ascii_lowercase();
    if port.starts_with("/dev/") || lower.starts_with("com") {
        Ok(DamiaoController::new_dm_serial(port, 921_600)?)
    } else {
        Ok(DamiaoController::new_socketcan(port)?)
    }
}

fn fmt_register(motor: &Arc<DamiaoMotor>, rid: u8, timeout: Duration) -> String {
    match motor.get_register_f32(rid, timeout) {
        Ok(value) => format!("{value:>10.6}"),
        Err(err) => format!("{:>10}", format!("ERR({err})")),
    }
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
        let vel_kp = fmt_register(&motor, 25, timeout);
        let vel_ki = fmt_register(&motor, 26, timeout);
        let pos_kp = fmt_register(&motor, 27, timeout);
        let pos_ki = fmt_register(&motor, 28, timeout);
        println!(
            "{:<16} {} {} {} {}",
            joint.name, vel_kp, vel_ki, pos_kp, pos_ki
        );
    }

    controller.close_bus()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let timeout = Duration::from_millis(args.timeout_ms);
    for target in &args.targets {
        read_target(target, timeout)?;
    }
    Ok(())
}
