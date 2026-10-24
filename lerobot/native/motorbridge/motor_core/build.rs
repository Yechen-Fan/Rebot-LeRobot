use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(motorbridge_dm_device_supported)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/dm_device_shim.cpp");
    println!("cargo:rerun-if-changed=../third_party/dm_device/v1.1.0/dmcan.h");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap();
    let dm_root = repo_root.join("third_party/dm_device/v1.1.0");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let Some(dm_lib_rel) = platform_library_relative_path(&target_os, &target_arch, &target_env)
    else {
        println!("cargo:warning=dm-device transport disabled for unsupported target {target_os}/{target_arch}/{target_env}");
        return;
    };

    let dm_lib = dm_root.join(dm_lib_rel);
    println!("cargo:rerun-if-changed={}", dm_lib.display());
    if !dm_lib.exists() {
        println!(
            "cargo:warning=dm-device transport disabled; missing SDK runtime {}",
            dm_lib.display()
        );
        return;
    }

    println!("cargo:rustc-cfg=motorbridge_dm_device_supported");

    cc::Build::new()
        .cpp(true)
        .file("src/dm_device_shim.cpp")
        .include(dm_root)
        .flag_if_supported("-std=c++17")
        .compile("dm_device_shim");

    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}

fn platform_library_relative_path(os: &str, arch: &str, env: &str) -> Option<&'static str> {
    match (os, arch, env) {
        ("linux", "x86_64", _) => Some("linux/x86_64/libdm_device.so"),
        ("linux", "aarch64", _) => Some("linux/arm64/libdm_device.so"),
        ("macos", "aarch64", _) => Some("macos/arm64/libdm_device.dylib"),
        ("macos", "x86_64", _) => Some("macos/x86_64/libdm_device.dylib"),
        ("windows", "x86_64", "gnu") => Some("windows/mingw/libdm_device.dll"),
        ("windows", "x86_64", "msvc") => Some("windows/msvc/dm_device.dll"),
        _ => None,
    }
}
