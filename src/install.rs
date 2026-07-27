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

/// How long a stop signal gives camon to drain before the service manager
/// kills it. Both defaults are wrong for an NVR: OpenRC's `supervise-daemon`
/// stops with `SIGTERM/5`, and systemd's `DefaultTimeoutStopSec` is usually
/// 90s, while shutdown flushes recordings that a remote warm-storage backend
/// uploads with a 300s per-request timeout. A SIGKILL here truncates exactly
/// the footage the graceful shutdown exists to save, and unlike the
/// update-initiated drain — which has [`crate::RESTART_DRAIN_DEADLINE`] as its
/// own backstop — nothing else bounds this one. Same budget as that deadline,
/// deliberately.
const STOP_TIMEOUT_SECS: u64 = crate::RESTART_DRAIN_DEADLINE.as_secs();

/// Camon exits cleanly of its own accord after installing an update, so the
/// service must be restarted on a *successful* exit, not only after a crash.
fn systemd_unit(exe: &Path, working_dir: &Path) -> String {
    format!(
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
TimeoutStopSec={stop}

[Install]
WantedBy=multi-user.target
",
        exe = exe.display(),
        wd = working_dir.display(),
        stop = STOP_TIMEOUT_SECS,
    )
}

fn install_systemd(exe: &Path, working_dir: &Path) -> Result<(), String> {
    let unit_path = Path::new("/etc/systemd/system/camon.service");

    if unit_path.exists() {
        eprintln!("warning: overwriting existing {}", unit_path.display());
    }

    let unit = systemd_unit(exe, working_dir);
    fs::write(unit_path, unit).map_err(|e| permission_hint(e, unit_path))?;

    eprintln!("wrote {}", unit_path.display());

    run_command("systemctl", &["daemon-reload"])?;
    run_command("systemctl", &["enable", "camon.service"])?;

    eprintln!("service enabled — start with: systemctl start camon");
    Ok(())
}

/// `supervise-daemon` rather than the default `start-stop-daemon`: camon exits
/// cleanly after installing an update, and start-stop-daemon has nothing
/// watching the process, so that exit would leave the service down (and a stale
/// pidfile behind) until someone started it by hand. camon never forks, which
/// is what supervise-daemon requires, and `command_background` is dropped
/// because supervise-daemon does the backgrounding itself and never reads it.
/// The respawn limits are spelled out instead of inherited: they differ between
/// supervise-daemon's own defaults and the OpenRC guide, and "give up after ten
/// crashes in five minutes" is a real crash loop rather than ten updates over
/// the service's lifetime. `pidfile` is the supervisor's pid here, not camon's.
/// Those limits are not what keeps a bad release from restarting camon forever
/// — they would answer an update loop by leaving the service down, and only
/// when its cycle is fast enough to fit ten restarts into the period, and
/// systemd's equivalent never trips at all with `RestartSec=5`. That is bounded
/// where it is caused, in [`crate::update`].
fn openrc_script(exe: &Path, working_dir: &Path) -> String {
    format!(
        "\
#!/sbin/openrc-run

# Needs OpenRC 0.33 or newer: older releases ignore `supervisor`, fall back to
# start-stop-daemon, and camon will not come back after a self-update.
# `pidfile` is the supervisor's pid, not camon's — delete a stale one left by a
# previously installed version of this script before the first start.

description=\"Camon video surveillance\"
command=\"{exe}\"
directory=\"{wd}\"
supervisor=\"supervise-daemon\"
respawn_delay=5
respawn_max=10
respawn_period=300
retry=\"SIGTERM/{stop}/SIGKILL/30\"
pidfile=\"/run/${{RC_SVCNAME}}.pid\"
output_log=\"/var/log/${{RC_SVCNAME}}.log\"
error_log=\"/var/log/${{RC_SVCNAME}}.err\"

depend() {{
    need net
}}
",
        exe = exe.display(),
        wd = working_dir.display(),
        stop = STOP_TIMEOUT_SECS,
    )
}

fn install_openrc(exe: &Path, working_dir: &Path) -> Result<(), String> {
    let script_path = Path::new("/etc/init.d/camon");

    if script_path.exists() {
        eprintln!("warning: overwriting existing {}", script_path.display());
    }

    let script = openrc_script(exe, working_dir);
    fs::write(script_path, script).map_err(|e| permission_hint(e, script_path))?;
    fs::set_permissions(script_path, fs::Permissions::from_mode(0o755))
        .map_err(|e| permission_hint(e, script_path))?;

    eprintln!("wrote {}", script_path.display());
    eprintln!("note: the service is supervised by supervise-daemon (OpenRC 0.33 or newer)");
    let stale_pidfile = Path::new("/run/camon.pid");
    if stale_pidfile.exists() {
        eprintln!(
            "warning: {} is left over from an earlier install and is now the supervisor's \
             pidfile — remove it before starting",
            stale_pidfile.display()
        );
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The self-updater relies on this: it drains, exits 0, and expects the
    /// service manager to bring the new binary up.
    #[test]
    fn generated_services_restart_after_a_clean_exit() {
        let unit = systemd_unit(Path::new("/usr/local/bin/camon"), Path::new("/etc/camon"));
        assert!(unit.contains("Restart=always"), "{unit}");

        let script = openrc_script(Path::new("/usr/local/bin/camon"), Path::new("/etc/camon"));
        assert!(
            script.contains("supervisor=\"supervise-daemon\""),
            "{script}"
        );
        // start-stop-daemon's backgrounding flag: ignored by supervise-daemon,
        // and camon must stay in the foreground for it to be supervised.
        assert!(!script.contains("command_background"), "{script}");
    }

    /// A signal shutdown is deliberately left without the update watchdog, so
    /// these are the only thing between a drain that is still writing footage
    /// and a SIGKILL.
    #[test]
    fn generated_services_let_the_drain_finish_before_killing_it() {
        let unit = systemd_unit(Path::new("/usr/local/bin/camon"), Path::new("/etc/camon"));
        assert!(
            unit.contains(&format!("TimeoutStopSec={STOP_TIMEOUT_SECS}")),
            "{unit}"
        );

        let script = openrc_script(Path::new("/usr/local/bin/camon"), Path::new("/etc/camon"));
        assert!(
            script.contains(&format!("retry=\"SIGTERM/{STOP_TIMEOUT_SECS}/SIGKILL/30\"")),
            "{script}"
        );
    }
}
