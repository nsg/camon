use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

enum InitSystem {
    Systemd,
    OpenRc,
}

fn detect_init_system() -> Result<InitSystem, String> {
    if Path::new("/run/systemd/system").exists() {
        Ok(InitSystem::Systemd)
    } else if Path::new("/sbin/openrc-run").exists() {
        Ok(InitSystem::OpenRc)
    } else {
        Err("could not detect init system (neither systemd nor OpenRC found)".into())
    }
}

fn resolve_exe_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| PathBuf::from("/usr/local/bin/camon"))
}

fn resolve_working_dir() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("config.toml").exists() {
            return cwd;
        }
    }
    PathBuf::from("/etc/camon")
}

fn install_systemd(exe: &Path, working_dir: &Path) -> Result<(), String> {
    let unit_path = Path::new("/etc/systemd/system/camon.service");

    if unit_path.exists() {
        eprintln!("warning: overwriting existing {}", unit_path.display());
    }

    let unit = format!(
        "\
[Unit]
Description=Camon video surveillance
After=network.target

[Service]
Type=simple
ExecStart={exe}
WorkingDirectory={wd}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
",
        exe = exe.display(),
        wd = working_dir.display(),
    );

    fs::write(unit_path, unit).map_err(|e| permission_hint(e, unit_path))?;

    eprintln!("wrote {}", unit_path.display());

    run_command("systemctl", &["daemon-reload"])?;
    run_command("systemctl", &["enable", "camon.service"])?;

    eprintln!("service enabled — start with: systemctl start camon");
    Ok(())
}

fn install_openrc(exe: &Path, working_dir: &Path) -> Result<(), String> {
    let script_path = Path::new("/etc/init.d/camon");

    if script_path.exists() {
        eprintln!("warning: overwriting existing {}", script_path.display());
    }

    let script = format!(
        "\
#!/sbin/openrc-run

description=\"Camon video surveillance\"
command=\"{exe}\"
directory=\"{wd}\"
command_background=true
pidfile=\"/run/${{RC_SVCNAME}}.pid\"
output_log=\"/var/log/${{RC_SVCNAME}}.log\"
error_log=\"/var/log/${{RC_SVCNAME}}.err\"

depend() {{
    need net
}}
",
        exe = exe.display(),
        wd = working_dir.display(),
    );

    fs::write(script_path, script).map_err(|e| permission_hint(e, script_path))?;
    fs::set_permissions(script_path, fs::Permissions::from_mode(0o755))
        .map_err(|e| permission_hint(e, script_path))?;

    eprintln!("wrote {}", script_path.display());

    run_command("rc-update", &["add", "camon", "default"])?;

    eprintln!("service enabled — start with: rc-service camon start");
    Ok(())
}

fn run_command(program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .args(args)
        .status()
        .map_err(|e| format!("failed to run {program}: {e}"))?;

    if !status.success() {
        return Err(format!("{program} exited with {status}"));
    }
    Ok(())
}

fn permission_hint(err: std::io::Error, path: &Path) -> String {
    if err.kind() == std::io::ErrorKind::PermissionDenied {
        format!(
            "permission denied writing {}: try running with sudo",
            path.display()
        )
    } else {
        format!("failed to write {}: {err}", path.display())
    }
}

pub fn install_service() -> Result<(), String> {
    let init = detect_init_system()?;
    let exe = resolve_exe_path();
    let working_dir = resolve_working_dir();

    eprintln!("executable: {}", exe.display());
    eprintln!("working directory: {}", working_dir.display());

    match init {
        InitSystem::Systemd => install_systemd(&exe, &working_dir),
        InitSystem::OpenRc => install_openrc(&exe, &working_dir),
    }
}
