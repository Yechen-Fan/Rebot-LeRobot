use rebotarm_control_dm_rust_examples::common::{
    parse_floats, parse_urdf_path, print_pose_with_model, prompt, q_deg, MathModel,
};
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let model = MathModel::load(&parse_urdf_path(&args))?;
    println!("Rust IK demo. Input target x y z in meters.");
    println!("Backend: C++/Pinocchio librebotarm_math.so");
    println!("examples: 0.2603 0.0 0.1917 | 0.20 0.10 0.20");
    let mut seed = model.neutral()?;
    loop {
        let Some(line) = prompt("target xyz > ")? else {
            break;
        };
        if matches!(line.as_str(), "q" | "quit" | "exit") {
            break;
        }
        let values = parse_floats(&line)?;
        if values.len() < 3 {
            println!("need x y z");
            continue;
        }
        let result = model.ik_position_cpp([values[0], values[1], values[2]], &seed, 2000)?;
        seed[..6].copy_from_slice(&result.q);
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
        print_pose_with_model(&model, &result.q)?;
    }
    Ok(())
}
