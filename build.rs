use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CAMON_BUILD_VERSION");
    let version = env::var("CAMON_BUILD_VERSION")
        .ok()
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| {
            Command::new("git")
                .args(["describe", "--tags", "--always", "--dirty"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|output| output.trim().to_string())
                .unwrap_or_else(|| "dev".to_string())
        });

    println!("cargo:rustc-env=CAMON_VERSION={version}");
}
