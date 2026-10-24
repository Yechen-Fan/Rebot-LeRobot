use rebotarm_control_dm_rust_examples::common::{
    parse_floats, parse_urdf_path, print_pose_with_model, prompt, q_rad_from_deg, MathModel,
};
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let model = MathModel::load(&parse_urdf_path(&args))?;
    println!("Rust FK demo. Input six joint angles in degrees.");
    println!("Backend: C++/Pinocchio librebotarm_math.so");
    println!("examples: 0 0 0 0 0 0 | 45 -30 15 -60 90 180");
    loop {
        let Some(line) = prompt("joint angles > ")? else {
            break;
        };
        if matches!(line.as_str(), "q" | "quit" | "exit") {
            break;
        }
        let values = parse_floats(&line)?;
        if values.len() != 6 {
            println!("need exactly 6 joint angles");
            continue;
        }
        let q = q_rad_from_deg(&values);
        print_pose_with_model(&model, &q)?;
    }
    Ok(())
}
