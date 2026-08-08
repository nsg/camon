//! The camon binary: argument dispatch, the self-updater, and a call into the
//! library that does everything else.

mod install;
mod update;

/// What `camon version` prints.
pub(crate) fn version_line() -> String {
    format!(
        "camon {} ({})",
        env!("CARGO_PKG_VERSION"),
        env!("CAMON_VERSION")
    )
}

fn dispatch_subcommand() -> bool {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => return false,
        // Must answer before config/logging/anything is opened: the self-updater runs this
        // against a freshly downloaded binary whatever the state of the installation. First
        // argument only, which is how the updater spells it.
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
