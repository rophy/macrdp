use std::process::Command;

fn main() {
    // Rebuild when HEAD, refs or the index change, so describe stays current.
    for path in ["../../.git/HEAD", "../../.git/refs", "../../.git/index"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let describe = Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=MACRDP_GIT_DESCRIBE={describe}");
}
