//! `camon version` is the one command the self-updater runs against a binary it
//! has just downloaded and knows nothing about, so what matters is the real
//! binary's behaviour rather than any function's: it has to answer whatever
//! else is wrong with the installation, and its first line has to stay in the
//! shape the updater parses. A unit test cannot see either — only a spawned
//! process can.

use std::process::Command;

const CAMON: &str = env!("CARGO_BIN_EXE_camon");

/// A config that camon refuses to start on, in a directory it will look in.
fn dir_with_a_broken_config() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.toml"), "[storage\nnot even toml").unwrap();
    dir
}

/// The dispatch has to happen before the config is read. A camon that could
/// only state its version when its configuration was already valid would be
/// useless to the updater: the probe runs the *new* binary in the old one's
/// working directory, and a new release may well be stricter about the config
/// than the release being replaced.
#[test]
fn version_is_answered_even_when_the_config_is_broken() {
    let dir = dir_with_a_broken_config();
    for argument in ["version", "--version", "-V"] {
        let out = Command::new(CAMON)
            .arg(argument)
            .current_dir(dir.path())
            .output()
            .expect("camon did not run");
        assert!(out.status.success(), "`camon {argument}` failed: {out:?}");

        let stdout = String::from_utf8(out.stdout).unwrap();
        let line = stdout.lines().next().expect("nothing was printed");
        let mut fields = line.split_whitespace();
        assert_eq!(fields.next(), Some("camon"), "in {line:?}");
        assert_eq!(
            fields.next(),
            Some(env!("CARGO_PKG_VERSION")),
            "the updater reads this field; it must be the version and nothing else, in {line:?}"
        );
    }
}

/// The other half of the probe's contract: an argument camon does not know is
/// an immediate failure, never a silent start-up. This is what makes running an
/// unknown binary to ask its version safe — a camon too old to have the command
/// exits instead of starting an NVR out of the updater's staging file.
#[test]
fn an_unknown_subcommand_exits_instead_of_starting() {
    let dir = dir_with_a_broken_config();
    let out = Command::new(CAMON)
        .arg("wat")
        .current_dir(dir.path())
        .output()
        .expect("camon did not run");

    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("unknown command: wat"), "{stderr:?}");
    assert!(stderr.contains("usage:"), "{stderr:?}");
}
