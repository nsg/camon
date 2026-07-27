//! The camon binary: argument dispatch, the self-updater, and a call into the
//! library that does everything else.

mod install;
mod update;

/// What `camon version` prints.
///
/// **Append-only.** The self-updater parses this out of a binary it has just
/// downloaded to find out what it really is, and the parser doing the reading
/// belongs to whichever camon is *already installed* — every version ever
/// released, indefinitely. Fields may be added after the existing ones; the
/// name must stay first and the version must stay second and stay a semantic
/// version, or every deployed updater refuses every future release. A test
/// pins this against the current parser
/// ([`crate::update::parse_version_output`]), which is the weaker half of the
/// guarantee: it passes for a change made to both halves at once, which is
/// exactly the change that would brick the installed base.
pub(crate) fn version_line() -> String {
    format!(
        "camon {} ({})",
        env!("CARGO_PKG_VERSION"),
        env!("CAMON_VERSION")
    )
}

fn dispatch_subcommand() -> bool {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Nothing to dispatch, or the first argument is a flag (e.g. `--config`) —
    // leave it to normal startup / `parse_cli_args`.
    match args.first().map(String::as_str) {
        None => return false,
        // Answered before the config is read, before logging is set up, and
        // before anything is opened — the tokio runtime `main` builds is all
        // that precedes it. The self-updater runs this against a binary it has
        // just downloaded, in the old binary's working directory, so it has to
        // answer whatever is wrong with that installation. Recognised as the
        // *first* argument only, which is how the updater always spells it;
        // anywhere else it is an unknown flag and ignored.
        Some("version" | "--version" | "-V") => {
            println!("{}", version_line());
            std::process::exit(0);
        }
        Some(first) if first.starts_with('-') => return false,
        _ => {}
    }

    match args[0].as_str() {
        "install" => {
            if args.get(1).map(|s| s.as_str()) == Some("service") {
                if let Err(e) = install::install_service() {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
                std::process::exit(0);
            }
            eprintln!("usage: camon install service");
            std::process::exit(1);
        }
        other => {
            eprintln!("unknown command: {other}");
            eprintln!(
                "usage: camon [--config <path>] [--set <dotted.path>=<value>]... \
                 [install service] [version | --version | -V]"
            );
            std::process::exit(1);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dispatch_subcommand();
    camon::app::run(update::check_and_update).await
}
