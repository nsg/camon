//! The self-updater: what camon does when a newer release of itself exists.

use std::cmp::Ordering;
use std::ffi::{OsStr, OsString};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::stream::{Stream, StreamExt as _};
use sha2::{Digest, Sha256};

const GITHUB_API_URL: &str = "https://api.github.com/repos/nsg/camon/releases/latest";
const CHECKSUMS_ASSET: &str = "sha256sums.txt";

/// How long a request may spend finding and reaching the server.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Idle budget: reqwest arms it flat until the response headers arrive, then per response frame
/// with a reset on each.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// How long each of the two small JSON/text requests gets, all in: connect,
/// TLS, whatever redirects it follows, and reading the body to its end.
const METADATA_TIMEOUT: Duration = Duration::from_secs(30);
/// And how much of them camon will hold. A release index runs to a few tens of
/// kilobytes and a checksum document to a few hundred bytes.
const METADATA_LIMIT: u64 = 4 * 1024 * 1024;

/// How long the release binary gets to arrive, from the first packet of the request to the last
/// of the body, redirects and all — GitHub answers an asset URL with one, so a per-body bound
/// would leave the hop before it unbounded.
const DOWNLOAD_DEADLINE: Duration = Duration::from_secs(600);
/// And how many bytes of it camon will allocate for.
const DOWNLOAD_LIMIT: u64 = 64 * 1024 * 1024;

/// How long the staged binary gets to say what version it is. It prints one
/// short line and exits, so this is a hang detector rather than a budget.
const VERSION_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the probe waits for a process group it has just killed to be
/// reaped.
const PROBE_REAP_TIMEOUT: Duration = Duration::from_secs(5);
/// Hard caps on what the probe reads from a binary it does not trust yet: a release that writes
/// a diagnostic dump to stdout must not be able to grow this process, which is recording while
/// the probe runs.
const PROBE_STDOUT_LIMIT: u64 = 4096;
const PROBE_STDERR_LIMIT: u64 = 4096;
/// How often, and how many times, the probe re-attempts an exec that failed for
/// a reason the machine can stop having — see [`spawn_failure_is_transient`].
/// Half a second total, far past any fork window.
const EXEC_RETRY_DELAY: Duration = Duration::from_millis(50);
const EXEC_RETRIES: u32 = 10;

/// How many times the same release version may be installed before camon stops installing it.
const MAX_INSTALL_ATTEMPTS: u32 = 3;

#[derive(serde::Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(serde::Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

pub async fn check_and_update(
    installed: camon::app::InstalledMarker,
) -> Result<bool, Box<dyn std::error::Error>> {
    let current_text = env!("CARGO_PKG_VERSION");
    tracing::info!(version = %current_text, "checking for updates");
    let current = Version::parse(current_text).ok_or_else(|| {
        format!("this build's own version ({current_text}) is not a semantic version")
    })?;

    let paths = UpdatePaths::for_exe(&std::env::current_exe()?);

    // Everything from here to the swap runs under one lock, so two camon
    // processes sharing an installation cannot both pass the attempt check,
    // both install, and both count the same attempt once.
    let lock = UpdateLock::acquire(&paths.lock).map_err(|e| {
        format!(
            "could not take the update lock at {}: {e} — camon cannot write beside its own binary, \
             so it could not install an update either",
            paths.lock.display()
        )
    })?;
    let _lock = match lock {
        Some(lock) => lock,
        None => {
            tracing::info!(
                lock = %paths.lock.display(),
                "another camon process is already updating this installation, skipping this check"
            );
            return Ok(false);
        }
    };
    // Under the lock, so nothing that looks abandoned is in fact another
    // updater's working file.
    sweep_stale_staging(&paths);
    let guard = read_guard(&paths.guard);

    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()?;
    let agent = format!("camon/{current_text}");
    let index = fetch_bounded(
        open_body(
            client
                .get(GITHUB_API_URL)
                .header("User-Agent", agent.as_str())
                .header("Accept", "application/vnd.github.v3+json"),
        ),
        "the release index",
        METADATA_LIMIT,
        METADATA_TIMEOUT,
    )
    .await?;
    let release: Release = serde_json::from_slice(&index)?;

    let latest = Version::parse(&release.tag_name).ok_or_else(|| {
        format!(
            "release tag {} is not a version camon can compare with its own — not updating",
            release.tag_name
        )
    })?;
    // Normalized, so `v0.6.0` and `0.6.0` are one guard key rather than two.
    let latest_text = latest.to_string();

    match latest.cmp(&current) {
        Ordering::Equal => {
            tracing::info!(version = %current_text, "already up to date");
            return Ok(false);
        }
        Ordering::Less => {
            tracing::info!(
                current = %current_text,
                latest = %latest_text,
                "current version is newer or equal"
            );
            return Ok(false);
        }
        Ordering::Greater => {}
    }

    // Asked before the download: a version camon has given up on must not cost
    // a multi-megabyte fetch every twelve hours to keep refusing it.
    if let Some(reason) = blocked_reason(guard.as_ref(), &latest_text, current_text, unix_now()) {
        tracing::error!(
            current = %current_text,
            latest = %latest_text,
            guard = %paths.guard.display(),
            "{reason}"
        );
        return Ok(false);
    }

    tracing::info!(
        current = %current_text,
        latest = %latest_text,
        "newer version available, updating"
    );

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == "camon-linux-glibc")
        .or_else(|| release.assets.iter().find(|a| a.name == "camon"))
        .ok_or("no 'camon-linux-glibc' binary asset found in release")?;

    let bytes = fetch_bounded(
        open_body(
            client
                .get(&asset.browser_download_url)
                .header("User-Agent", agent.as_str()),
        ),
        &asset.name,
        DOWNLOAD_LIMIT,
        DOWNLOAD_DEADLINE,
    )
    .await?;

    // Corruption protection (not a security boundary — see the trust model at the top of this
    // file).
    let checksums = match release.assets.iter().find(|a| a.name == CHECKSUMS_ASSET) {
        Some(sums_asset) => {
            let text = fetch_bounded(
                open_body(
                    client
                        .get(&sums_asset.browser_download_url)
                        .header("User-Agent", agent.as_str()),
                ),
                CHECKSUMS_ASSET,
                METADATA_LIMIT,
                METADATA_TIMEOUT,
            )
            .await?;
            Some(String::from_utf8_lossy(&text).into_owned())
        }
        None => None,
    };

    verify_download(&bytes, &asset.name, checksums.as_deref())?;

    // Every path this function *returns* by takes the staging file with it — it is a
    // complete, executable copy of a release binary sitting next to the real one.
    stage_binary(&paths.staging, &bytes)?;

    // The tag is a label a human typed; what the binary says about itself is what decides
    // whether the process started after the restart will download this same asset all over
    // again.
    let staged = match probe_version(&paths.staging, VERSION_PROBE_TIMEOUT).await {
        Ok(version) => version,
        Err(failure) => {
            let _ = std::fs::remove_file(&paths.staging);
            return Ok(after_a_failed_probe(
                &paths,
                guard.as_ref(),
                &latest_text,
                failure,
            ));
        }
    };
    if let Err(reason) = assess_staged(&staged, &latest, &current) {
        let _ = std::fs::remove_file(&paths.staging);
        return Ok(refuse(&paths, guard.as_ref(), &latest_text, reason));
    }

    // Written before the swap, not after: from here the process is on its way
    // out, and a count that only landed once the new binary was running would
    // never be written in the one case it exists for.
    let attempt = match record_attempt(&paths.guard, &latest_text, guard.as_ref()) {
        Ok(attempt) => attempt,
        Err(e) => {
            // Removed like every other refusal: a leftover staging file whose
            // inode is still being executed cannot be written over (ETXTBSY),
            // so it would wedge the next attempt too.
            let _ = std::fs::remove_file(&paths.staging);
            return Err(format!(
                "could not record the update attempt in {}: {e} — not installing, because without \
                 that record a release that fails to restart would reinstall itself forever",
                paths.guard.display()
            )
            .into());
        }
    };

    match publish_staged(&paths, &installed, sync_parent) {
        Err(e) => {
            // Nothing was installed, so the attempt must not stand — three
            // failed swaps would otherwise block a version that never ran.
            restore_guard(&paths.guard, guard.as_ref());
            let _ = std::fs::remove_file(&paths.staging);
            return Err(format!(
                "could not replace {} with the {latest_text} binary: {e}",
                paths.exe.display()
            )
            .into());
        }
        // Not an error to return: the binary *is* installed, and reporting failure here would
        // leave the process running the old one and downloading the same release again on the
        // next check, spending the attempts the guard exists to ration.
        Ok(Published::Unsynced(e)) => {
            tracing::warn!(
                path = %paths.exe.display(),
                error = %e,
                "installed {latest_text} but could not fsync the directory holding it: the new \
                 binary is in place and will start on the restart, but a power cut before the \
                 entry reaches the disk could resolve the name back to the old one"
            );
        }
        Ok(Published::Durable) => {}
    }

    tracing::info!(version = %latest_text, attempt, "update applied successfully");
    Ok(true)
}

/// How far publishing the staged binary got.
enum Published {
    /// Renamed, and the name is on the disk to be found after a power cut.
    Durable,
    /// Renamed, but the directory sync did not finish. The binary is installed
    /// either way — that is what makes this a warning rather than a failure.
    Unsynced(std::io::Error),
}

/// Put the staged binary in place, and say so before doing anything else.
fn publish_staged<S>(
    paths: &UpdatePaths,
    installed: &camon::app::InstalledMarker,
    sync: S,
) -> std::io::Result<Published>
where
    S: FnOnce(&Path) -> std::io::Result<()>,
{
    std::fs::rename(&paths.staging, &paths.exe)?;
    installed.record();
    match sync(&paths.exe) {
        Ok(()) => Ok(Published::Durable),
        Err(e) => Ok(Published::Unsynced(e)),
    }
}

/// Send a request and hand back the body without reading any of it: what the
/// server says its length is, and the frames it will arrive in.
async fn open_body(
    request: reqwest::RequestBuilder,
) -> Result<
    (
        Option<u64>,
        impl Stream<Item = reqwest::Result<bytes::Bytes>>,
    ),
    String,
> {
    let response = request
        .send()
        .await
        .map_err(|e| format!("the request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("the server answered with an error: {e}"))?;
    Ok((response.content_length(), response.bytes_stream()))
}

/// Fetch a whole response into memory under both bounds camon is able to state: how many bytes
/// it will allocate for, and how long the request has from its first packet to the end of its
/// body.
async fn fetch_bounded<F, S, B, E>(
    open: F,
    what: &str,
    limit: u64,
    deadline: Duration,
) -> Result<Vec<u8>, String>
where
    F: std::future::Future<Output = Result<(Option<u64>, S), String>>,
    S: Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let fetch = async {
        let (declared, body) = open.await.map_err(|e| format!("{what}: {e}"))?;
        if let Some(declared) = declared {
            if declared > limit {
                return Err(oversize(what, declared, limit, "says it is"));
            }
        }
        tokio::pin!(body);
        // At most the limit, so a server that declares a size it never sends
        // cannot make camon reserve more than it was already willing to hold.
        let mut collected: Vec<u8> = Vec::with_capacity(declared.unwrap_or(0).min(limit) as usize);
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|e| format!("{what} could not be read: {e}"))?;
            let chunk = chunk.as_ref();
            let would_hold = collected.len() as u64 + chunk.len() as u64;
            if would_hold > limit {
                return Err(oversize(what, would_hold, limit, "would have reached"));
            }
            collected.extend_from_slice(chunk);
        }
        Ok(collected)
    };
    match tokio::time::timeout(deadline, fetch).await {
        Ok(result) => result,
        Err(_) => Err(format!(
            "{what} had not finished downloading after {}s — abandoning it",
            deadline.as_secs()
        )),
    }
}

/// The one message for both ways a body can be too big, so an operator reading
/// it can tell a release that has honestly grown from a server feeding camon
/// something it should not: the number that broke the rule, and the rule.
fn oversize(what: &str, size: u64, limit: u64, verb: &str) -> String {
    format!(
        "{what} {verb} {size} bytes, past the {limit} bytes camon will allocate for it — \
         abandoning the download. A release that has legitimately grown past this limit needs it \
         raised in the updater"
    )
}

/// Write the downloaded binary to its staging path, executable and durable.
fn stage_binary(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let staged = camon::durable::write_synced(path, bytes).and_then(|()| {
        let file = std::fs::File::open(path)?;
        file.set_permissions(std::fs::Permissions::from_mode(0o755))?;
        file.sync_all()
    });
    if staged.is_err() {
        let _ = std::fs::remove_file(path);
    }
    staged
}

/// What a probe that produced no version costs the release.
#[derive(Debug)]
enum ProbeFailure {
    /// The binary ran, and what it did is an answer.
    Verdict(String),
    /// The probe could not be run or could not be heard, so there is no answer to record.
    Unrun(String),
    /// The same, except that nothing about waiting will change it: this installation cannot
    /// execute the file it just staged.
    Impeded(String),
}

/// Deal with a probe that produced no version.
fn after_a_failed_probe(
    paths: &UpdatePaths,
    previous: Option<&InstallGuard>,
    version: &str,
    failure: ProbeFailure,
) -> bool {
    match failure {
        ProbeFailure::Verdict(reason) => refuse(
            paths,
            previous,
            version,
            format!("the binary in release {version} could not say what version it is: {reason}"),
        ),
        ProbeFailure::Unrun(reason) => {
            tracing::warn!(
                version,
                "the binary in release {version} could not be probed on this machine: {reason} — \
                 this says nothing about the release, so nothing is recorded against it and the \
                 next check will try it again"
            );
            false
        }
        ProbeFailure::Impeded(reason) => {
            tracing::error!(
                version,
                staging = %paths.staging.display(),
                "camon cannot execute the update it staged beside its own binary: {reason} — this \
                 is not the release's doing and nothing is recorded against it, and nothing camon \
                 can do will change it: until this installation can run a binary it writes there, \
                 no release installs"
            );
            false
        }
    }
}

/// Refuse this release for good: say why, and write it down so the asset is not downloaded
/// again every twelve hours to reach the same verdict.
fn refuse(
    paths: &UpdatePaths,
    previous: Option<&InstallGuard>,
    version: &str,
    reason: String,
) -> bool {
    tracing::error!(
        version,
        guard = %paths.guard.display(),
        "{reason} — camon will not install this release; publish a fixed one, or delete the \
         guard file to try it again"
    );
    if let Err(e) = record_refusal(&paths.guard, version, &reason, previous) {
        tracing::warn!(
            path = %paths.guard.display(),
            error = %e,
            "could not record the refusal; this release will be downloaded again on the next check"
        );
    }
    false
}

/// May the staged binary replace the running one?
fn assess_staged(staged: &Version, tag: &Version, current: &Version) -> Result<(), String> {
    if staged <= current {
        return Err(format!(
            "release {tag} ships a binary that reports itself as {staged}, which is not newer than \
             the running {current}"
        ));
    }
    if staged != tag {
        return Err(format!(
            "release {tag} ships a binary that reports itself as {staged}: the tag does not match \
             the asset, so installing it would leave {tag} still looking newer and camon would \
             fetch it again after every restart"
        ));
    }
    Ok(())
}

/// Is this a spawn failure the machine can stop having?
fn spawn_failure_is_transient(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(
            libc::EAGAIN | libc::ENOMEM | libc::EMFILE | libc::ENFILE | libc::ETXTBSY | libc::EINTR
        )
    )
}

/// Start the probe, retrying the failures that are moments rather than states.
async fn spawn_probe<F>(mut attempt: F) -> Result<tokio::process::Child, ProbeFailure>
where
    F: FnMut() -> std::io::Result<tokio::process::Child>,
{
    let mut retries = EXEC_RETRIES;
    loop {
        match attempt() {
            Ok(child) => return Ok(child),
            Err(e) if spawn_failure_is_transient(&e) => {
                if retries == 0 {
                    return Err(ProbeFailure::Unrun(format!(
                        "it still could not be started {EXEC_RETRIES} retries later: {e}"
                    )));
                }
                retries -= 1;
                tokio::time::sleep(EXEC_RETRY_DELAY).await;
            }
            Err(e) if e.raw_os_error() == Some(libc::ENOEXEC) => {
                return Err(ProbeFailure::Verdict(format!(
                    "this kernel cannot run it at all: {e}"
                )))
            }
            // Anything left is about the staging file or the filesystem holding it — a
            // directory mounted `noexec`, a permission camon does not have, a file something
            // else removed — and the next release would fail here in exactly the same way.
            Err(e) => {
                return Err(ProbeFailure::Impeded(format!(
                    "it could not be started here: {e}"
                )))
            }
        }
    }
}

/// Ask a binary what version it is, by running it.
async fn probe_version(binary: &Path, timeout: Duration) -> Result<Version, ProbeFailure> {
    use tokio::io::AsyncReadExt as _;

    let mut child = spawn_probe(|| {
        tokio::process::Command::new(binary)
            .arg("version")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Its own group, so everything it spawns can be killed as one.
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
    })
    .await?;

    let group = match child.id() {
        Some(group) => group,
        None => {
            return Err(ProbeFailure::Unrun(
                "it was gone before it could be probed".to_string(),
            ))
        }
    };
    let mut stdout = child.stdout.take().expect("stdout is piped");
    let mut stderr = child.stderr.take().expect("stderr is piped");
    let mut out = Vec::new();
    let mut err = Vec::new();

    let mut capped_out = (&mut stdout).take(PROBE_STDOUT_LIMIT);
    let mut capped_err = (&mut stderr).take(PROBE_STDERR_LIMIT);
    let read = async {
        let _ = tokio::join!(
            capped_out.read_to_end(&mut out),
            capped_err.read_to_end(&mut err),
        );
    };
    let timed_out = tokio::time::timeout(timeout, read).await.is_err();

    // Sound only here, before the child is reaped: until then its pid — and with it the group
    // id, since it leads the group — cannot be handed to anything else.
    kill_group(group);
    let status = tokio::time::timeout(PROBE_REAP_TIMEOUT, child.wait()).await;

    // What the binary said is a verdict; what it did not manage to say in time is not.
    if out.len() as u64 >= PROBE_STDOUT_LIMIT || err.len() as u64 >= PROBE_STDERR_LIMIT {
        return Err(ProbeFailure::Verdict(format!(
            "it printed at least {PROBE_STDOUT_LIMIT} bytes instead of one version line"
        )));
    }
    if timed_out {
        return Err(ProbeFailure::Unrun(format!(
            "it did not answer within {}s",
            timeout.as_secs()
        )));
    }
    match status {
        Ok(Ok(status)) if status.success() => {}
        Ok(Ok(status)) => {
            return Err(ProbeFailure::Verdict(format!(
                "it exited with {status}: {}",
                String::from_utf8_lossy(&err).trim()
            )))
        }
        // Losing the child or failing to kill it describes this machine, not a bad release.
        Ok(Err(e)) => {
            return Err(ProbeFailure::Unrun(format!(
                "it could not be waited for: {e}"
            )))
        }
        Err(_) => {
            return Err(ProbeFailure::Unrun(
                "it could not be stopped afterwards".to_string(),
            ))
        }
    }

    let stdout = String::from_utf8_lossy(&out);
    parse_version_output(&stdout).ok_or_else(|| {
        ProbeFailure::Verdict(format!(
            "its output was not a version line: {:?}",
            stdout.trim()
        ))
    })
}

/// SIGKILL whatever is left of a probe. The usual outcome is `ESRCH` — the
/// group is already gone — so the result is deliberately ignored.
fn kill_group(group: u32) {
    unsafe { libc::killpg(group as libc::pid_t, libc::SIGKILL) };
}

/// Pull the version out of what `camon version` prints (`camon <version> …`).
fn parse_version_output(stdout: &str) -> Option<Version> {
    let mut fields = stdout.lines().next()?.split_whitespace();
    if fields.next()? != "camon" {
        return None;
    }
    Version::parse(fields.next()?)
}

/// A semantic version, as far as precedence is concerned.
#[derive(Debug, Clone)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    pre: Vec<PreRelease>,
    build: Option<String>,
}

/// A dot-separated pre-release identifier. The derived order is the specified
/// one: numeric identifiers rank below alphanumeric ones, numbers compare as
/// numbers, and text compares by ASCII.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PreRelease {
    Numeric(u64),
    Text(String),
}

impl Version {
    /// Strict on purpose: anything camon cannot compare exactly it refuses
    /// loudly rather than comparing approximately. A `v` prefix is accepted
    /// because release tags carry one.
    fn parse(text: &str) -> Option<Self> {
        let text = text.strip_prefix('v').unwrap_or(text);
        let (text, build) = match text.split_once('+') {
            Some((version, build)) => (version, Some(build)),
            None => (text, None),
        };
        if let Some(build) = build {
            if !build.split('.').all(is_identifier) {
                return None;
            }
        }
        let (core, pre) = match text.split_once('-') {
            Some((core, pre)) => (core, Some(pre)),
            None => (text, None),
        };

        let mut fields = core.split('.');
        let major = parse_numeric(fields.next()?)?;
        let minor = parse_numeric(fields.next()?)?;
        let patch = parse_numeric(fields.next()?)?;
        if fields.next().is_some() {
            return None;
        }

        let pre = match pre {
            None => Vec::new(),
            Some(pre) => pre
                .split('.')
                .map(|id| {
                    if !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()) {
                        parse_numeric(id).map(PreRelease::Numeric)
                    } else if is_identifier(id) {
                        Some(PreRelease::Text(id.to_string()))
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<_>>>()?,
        };

        Some(Self {
            major,
            minor,
            patch,
            pre,
            build: build.map(str::to_string),
        })
    }
}

/// A core or numeric pre-release field: digits only, no leading zero, and small
/// enough to be a number — an overflowing field is refused, not dropped.
fn parse_numeric(field: &str) -> Option<u64> {
    if field.is_empty() || !field.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if field.len() > 1 && field.starts_with('0') {
        return None;
    }
    field.parse().ok()
}

fn is_identifier(field: &str) -> bool {
    !field.is_empty()
        && field
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                // A release outranks any pre-release of itself.
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => self.pre.cmp(&other.pre),
            })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Version {}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        for (i, id) in self.pre.iter().enumerate() {
            let separator = if i == 0 { '-' } else { '.' };
            match id {
                PreRelease::Numeric(n) => write!(f, "{separator}{n}")?,
                PreRelease::Text(t) => write!(f, "{separator}{t}")?,
            }
        }
        match &self.build {
            Some(build) => write!(f, "+{build}"),
            None => Ok(()),
        }
    }
}

/// The files the updater keeps beside the binary it maintains.
struct UpdatePaths {
    exe: PathBuf,
    staging: PathBuf,
    guard: PathBuf,
    lock: PathBuf,
}

/// What every staging file's name is built from: `<exe>.update.<pid>.tmp`. Kept
/// as constants because the sweep recognises leftovers by exactly this shape,
/// and a name that drifted from it would leave them lying there for good.
const STAGING_PREFIX: &str = ".update.";
const STAGING_SUFFIX: &str = ".tmp";

impl UpdatePaths {
    fn for_exe(exe: &Path) -> Self {
        Self {
            exe: exe.to_path_buf(),
            // Process-unique: two updaters must not write one staging file, and
            // a leftover whose inode is still being executed cannot be written
            // over at all (ETXTBSY).
            staging: sibling(
                exe,
                &format!("{STAGING_PREFIX}{}{STAGING_SUFFIX}", std::process::id()),
            ),
            guard: sibling(exe, ".update-guard"),
            lock: sibling(exe, ".update-lock"),
        }
    }
}

/// Delete staging files left behind by attempts that never got to clean up after themselves.
fn sweep_stale_staging(paths: &UpdatePaths) {
    let Some(dir) = camon::durable::parent_dir(&paths.exe) else {
        return;
    };
    let Some(prefix) = paths
        .exe
        .file_name()
        .map(|name| format!("{}{STAGING_PREFIX}", name.to_string_lossy()))
    else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == paths.staging {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&prefix) || !name.ends_with(STAGING_SUFFIX) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or_default();
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::warn!(
                path = %path.display(),
                bytes = size,
                "removed a staging file from an update that was interrupted before it could clean \
                 up after itself"
            ),
            Err(e) => tracing::warn!(
                path = %path.display(),
                error = %e,
                "could not remove the staging file left by an interrupted update"
            ),
        }
    }
}

/// Append to the file name rather than `Path::set_extension`, which replaces an
/// existing extension: `camon.debug` and `camon.release` are two installations
/// and may not share one guard or one lock.
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = path
        .file_name()
        .unwrap_or_else(|| OsStr::new("camon"))
        .to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// Exclusive lock over one installation's update, held from before the guard is read until
/// after the binary is swapped.
struct UpdateLock {
    _file: std::fs::File,
}

impl UpdateLock {
    /// `Ok(None)` means another process holds it — not an error, just not this
    /// process's turn.
    fn acquire(path: &Path) -> std::io::Result<Option<Self>> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            return match error.kind() {
                std::io::ErrorKind::WouldBlock => Ok(None),
                _ => Err(error),
            };
        }
        Ok(Some(Self { _file: file }))
    }
}

/// What camon last did about a release, and why.
#[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct InstallGuard {
    /// The release version every other field is about. A record for one version
    /// says nothing about another, so a genuinely newer release is never held
    /// back by it.
    version: String,
    /// Installs *attempted* — written before the swap, since a count that only
    /// landed afterwards would never be written when it matters.
    attempts: u32,
    /// Why this version is refused outright, if it is: something about the
    /// artifact that only a new release can change.
    #[serde(default)]
    refused: Option<String>,
    /// Whether the refusal above was reached by a camon that can tell a binary which ran and
    /// failed from a probe that never ran ([`ProbeFailure`]).
    #[serde(default)]
    refusal_classified: bool,
    /// When this record was last written, so a repeated verdict can say how old
    /// it is rather than reading as news.
    last_attempt_unix: u64,
}

/// Read the guard, treating "not there" and "not readable" alike as no record.
fn read_guard(path: &Path) -> Option<InstallGuard> {
    let text = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&text) {
        Ok(guard) => Some(guard),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "update guard file is unreadable and is being ignored"
            );
            None
        }
    }
}

/// Why this version must not be fetched and installed again, if it must not.
fn blocked_reason(
    guard: Option<&InstallGuard>,
    version: &str,
    current: &str,
    now: u64,
) -> Option<String> {
    let guard = guard.filter(|guard| guard.version == version)?;
    let ago = describe_age(now.saturating_sub(guard.last_attempt_unix));
    match &guard.refused {
        Some(refused) if guard.refusal_classified => {
            return Some(format!(
                "release {version} was refused {ago} and nothing about it can have changed since: \
                 {refused}"
            ))
        }
        Some(refused) => tracing::warn!(
            version,
            "release {version} was refused {ago} by a camon that could not tell a probe which \
             failed to run from a binary which ran and failed ({refused}) — trying it again. A \
             verdict this time is recorded and final; a probe that again cannot run records \
             nothing, and this same line follows every check until one of the other two happens"
        ),
        None => {}
    }
    if guard.attempts >= MAX_INSTALL_ATTEMPTS {
        return Some(format!(
            "release {version} has been installed {} times, most recently {ago}, and this process \
             is still running {current}: the restart is not taking effect, so installing it again \
             would only repeat. Check that the service starts the binary the updater replaces",
            guard.attempts
        ));
    }
    None
}

/// Count this install of `version` and persist the count. Returns the attempt
/// number the caller is about to make.
fn record_attempt(
    path: &Path,
    version: &str,
    previous: Option<&InstallGuard>,
) -> std::io::Result<u32> {
    let attempts = match previous {
        Some(guard) if guard.version == version => guard.attempts + 1,
        _ => 1,
    };
    write_guard(
        path,
        &InstallGuard {
            version: version.to_string(),
            attempts,
            refused: None,
            refusal_classified: false,
            last_attempt_unix: unix_now(),
        },
    )?;
    Ok(attempts)
}

fn record_refusal(
    path: &Path,
    version: &str,
    reason: &str,
    previous: Option<&InstallGuard>,
) -> std::io::Result<()> {
    let attempts = match previous {
        Some(guard) if guard.version == version => guard.attempts,
        _ => 0,
    };
    write_guard(
        path,
        &InstallGuard {
            version: version.to_string(),
            attempts,
            refused: Some(reason.to_string()),
            // Written by this camon, which only refuses what a running binary
            // actually did.
            refusal_classified: true,
            last_attempt_unix: unix_now(),
        },
    )
}

/// Put back what the guard said before an attempt that turned out not to have
/// happened. Best effort by definition — it is already an error path — and a
/// failure here only over-counts, which costs retries rather than safety.
fn restore_guard(path: &Path, previous: Option<&InstallGuard>) {
    let restored = match previous {
        Some(guard) => write_guard(path, guard),
        None => std::fs::remove_file(path),
    };
    if let Err(e) = restored {
        tracing::warn!(
            path = %path.display(),
            error = %e,
            "could not undo the update attempt counted for an install that did not happen"
        );
    }
}

fn write_guard(path: &Path, guard: &InstallGuard) -> std::io::Result<()> {
    let text = serde_json::to_string(guard).map_err(std::io::Error::other)?;
    // Staged, flushed, renamed, and the directory flushed too, like every other file camon has
    // to find again after an unclean stop.
    let temp = sibling(path, &format!(".{}.tmp", std::process::id()));
    camon::durable::write_synced(&temp, text.as_bytes())?;
    std::fs::rename(&temp, path)?;
    sync_parent(path)
}

fn sync_parent(path: &Path) -> std::io::Result<()> {
    match camon::durable::parent_dir(path) {
        // A bare relative name resolves to `.`, which is a real directory to
        // sync rather than a reason to skip the sync.
        Some(dir) => camon::durable::sync_dir(dir),
        None => Ok(()),
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Rough but honest: what matters is whether the verdict being repeated is
/// minutes or months old.
fn describe_age(seconds: u64) -> String {
    match seconds {
        0..=90 => "just now".to_string(),
        s if s < 5400 => format!("{} minutes ago", s / 60),
        s if s < 172_800 => format!("{} hours ago", s / 3600),
        s => format!("{} days ago", s / 86400),
    }
}

/// Result of looking up an asset in a `sha256sums.txt` document.
#[derive(Debug, PartialEq, Eq)]
enum ChecksumLookup {
    /// A well-formed entry (64 lowercase hex chars) was found for the asset.
    Found(String),
    /// No line referenced the asset name.
    NotFound,
    /// A line referenced the asset but its hash field was malformed.
    Malformed,
}

/// Parse a standard `sha256sum` document and locate the entry for `asset_name`.
fn find_expected_hash(checksums: &str, asset_name: &str) -> ChecksumLookup {
    let mut malformed = false;
    for line in checksums.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((hash, rest)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let name = rest.trim_start().trim_start_matches('*').trim();
        if name != asset_name {
            continue;
        }
        if hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return ChecksumLookup::Found(hash.to_ascii_lowercase());
        }
        // Right name, unusable hash — remember and keep scanning in case a
        // later valid line exists, but never accept a bad one.
        malformed = true;
    }
    if malformed {
        ChecksumLookup::Malformed
    } else {
        ChecksumLookup::NotFound
    }
}

/// Cheap corruption tripwire: is `bytes` a non-empty, minimally sane ELF binary?
fn is_valid_elf(bytes: &[u8]) -> bool {
    // The 64-bit ELF header is 64 bytes; require at least that much.
    if bytes.len() < 64 {
        return false;
    }
    if bytes[0..4] != [0x7f, b'E', b'L', b'F'] {
        return false;
    }
    let class = bytes[4]; // 1 = 32-bit, 2 = 64-bit
    let data = bytes[5]; // 1 = little-endian, 2 = big-endian
    if class != 1 && class != 2 {
        return false;
    }
    if data != 1 && data != 2 {
        return false;
    }
    let e_type = if data == 1 {
        u16::from_le_bytes([bytes[16], bytes[17]])
    } else {
        u16::from_be_bytes([bytes[16], bytes[17]])
    };
    matches!(e_type, 2 | 3) // ET_EXEC | ET_DYN
}

/// Lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Validate downloaded bytes before they are written anywhere or run.
fn verify_download(bytes: &[u8], asset_name: &str, checksums: Option<&str>) -> Result<(), String> {
    if !is_valid_elf(bytes) {
        return Err(format!(
            "downloaded {asset_name} is not a valid ELF binary ({} bytes) — refusing to install",
            bytes.len()
        ));
    }

    let Some(checksums) = checksums else {
        return Err(format!(
            "release has no {CHECKSUMS_ASSET} to check {asset_name} against — aborting update"
        ));
    };

    match find_expected_hash(checksums, asset_name) {
        ChecksumLookup::Found(expected) => {
            let actual = sha256_hex(bytes);
            if actual == expected {
                tracing::info!(asset = %asset_name, "checksum verified");
                Ok(())
            } else {
                Err(format!(
                    "checksum mismatch for {asset_name}: expected {expected}, got {actual} — aborting update"
                ))
            }
        }
        ChecksumLookup::NotFound => Err(format!(
            "{CHECKSUMS_ASSET} does not list {asset_name} — aborting update"
        )),
        ChecksumLookup::Malformed => Err(format!(
            "{CHECKSUMS_ASSET} has a malformed entry for {asset_name} — aborting update"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(text: &str) -> Version {
        Version::parse(text).unwrap_or_else(|| panic!("{text} should parse"))
    }

    #[test]
    fn test_version_ordering() {
        assert!(version("1.0.1") > version("1.0.0"));
        assert!(version("1.1.0") > version("1.0.9"));
        assert!(version("2.0.0") > version("1.9.9"));
        assert!(version("1.0.10") > version("1.0.9"));
        assert_eq!(version("1.0.0"), version("1.0.0"));
        assert!(version("1.0.0") < version("1.0.1"));
        assert_eq!(version("v0.5.0"), version("0.5.0"));
    }

    #[test]
    fn test_version_ordering_of_prereleases_and_build_metadata() {
        assert!(version("1.0.0") > version("1.0.0-rc.1"));
        assert!(version("1.0.0-rc.2") > version("1.0.0-rc.1"));
        assert!(version("1.0.0-rc.11") > version("1.0.0-rc.2"));
        assert!(version("1.0.0-alpha") < version("1.0.0-beta"));
        assert!(version("1.0.0-alpha") < version("1.0.0-alpha.1"));
        assert!(version("1.0.0-alpha.1") < version("1.0.0-alpha.beta"));
        assert!(version("1.0.0-rc.1") > version("0.9.9"));

        assert_eq!(version("1.0.0+build"), version("1.0.0"));
        assert_eq!(version("1.0.0+build.999"), version("1.0.0+other"));
        assert!(version("1.2.3+build.999") < version("1.2.4"));
        assert_eq!(version("1.0.0+build.9").to_string(), "1.0.0+build.9");
        assert_eq!(version("v1.0.0-rc.1").to_string(), "1.0.0-rc.1");
    }

    #[test]
    fn test_version_parsing_refuses_what_it_cannot_compare() {
        for bad in [
            "",
            "1",
            "1.0",
            "1.0.0.0",
            "1.0.x",
            "one.0.0",
            "1.0.0-",
            "1.0.0+",
            "1.0.0-rc.01",              // leading zero in a numeric identifier
            "01.0.0",                   // leading zero in a core field
            "1.0.0-rc_1",               // underscore is not an identifier character
            "1.0.0+build_1",            // nor in build metadata
            "18446744073709551616.0.0", // overflows u64 instead of being dropped
            "dev",
        ] {
            assert!(Version::parse(bad).is_none(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn test_sha256_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_find_expected_hash_text_format() {
        let hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let doc = format!("{hash}  camon-linux-glibc\n");
        assert_eq!(
            find_expected_hash(&doc, "camon-linux-glibc"),
            ChecksumLookup::Found(hash.to_string())
        );
    }

    #[test]
    fn test_find_expected_hash_binary_format_and_uppercase() {
        let hash = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
        let doc = format!(
            "0000000000000000000000000000000000000000000000000000000000000000  other-file\n\
             {hash} *camon-linux-glibc\n"
        );
        assert_eq!(
            find_expected_hash(&doc, "camon-linux-glibc"),
            ChecksumLookup::Found(hash.to_ascii_lowercase())
        );
    }

    #[test]
    fn test_find_expected_hash_skips_blank_and_comment_lines() {
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let doc = format!("# a comment\n\n{hash}  camon-linux-glibc\n");
        assert_eq!(
            find_expected_hash(&doc, "camon-linux-glibc"),
            ChecksumLookup::Found(hash.to_string())
        );
    }

    #[test]
    fn test_find_expected_hash_not_found() {
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let doc = format!("{hash}  some-other-binary\n");
        assert_eq!(
            find_expected_hash(&doc, "camon-linux-glibc"),
            ChecksumLookup::NotFound
        );
    }

    #[test]
    fn test_find_expected_hash_rejects_malformed() {
        let doc = "deadbeef  camon-linux-glibc\n";
        assert_eq!(
            find_expected_hash(doc, "camon-linux-glibc"),
            ChecksumLookup::Malformed
        );
        let doc =
            "zzzz816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f200ZZ  camon-linux-glibc\n";
        assert_eq!(
            find_expected_hash(doc, "camon-linux-glibc"),
            ChecksumLookup::Malformed
        );
    }

    fn fake_elf() -> Vec<u8> {
        let mut b = vec![0u8; 64];
        b[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        b[4] = 2; // 64-bit
        b[5] = 1; // little-endian
        b[16] = 3; // ET_DYN
        b[17] = 0;
        b
    }

    #[test]
    fn test_is_valid_elf_accepts_sane_header() {
        assert!(is_valid_elf(&fake_elf()));
    }

    #[test]
    fn test_is_valid_elf_rejects_bad_input() {
        assert!(!is_valid_elf(b""));
        assert!(!is_valid_elf(
            b"not an elf file at all, just some text bytes here to pad it out!!"
        ));
        let mut too_short = fake_elf();
        too_short.truncate(32);
        assert!(!is_valid_elf(&too_short));
        let mut bad_magic = fake_elf();
        bad_magic[1] = b'X';
        assert!(!is_valid_elf(&bad_magic));
        let mut bad_type = fake_elf();
        bad_type[16] = 0; // ET_NONE
        assert!(!is_valid_elf(&bad_type));
    }

    #[test]
    fn test_verify_download_matching_checksum() {
        let bytes = fake_elf();
        let doc = format!("{}  camon-linux-glibc\n", sha256_hex(&bytes));
        assert!(verify_download(&bytes, "camon-linux-glibc", Some(&doc)).is_ok());
    }

    #[test]
    fn test_verify_download_mismatch_aborts() {
        let bytes = fake_elf();
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        let doc = format!("{wrong}  camon-linux-glibc\n");
        let err = verify_download(&bytes, "camon-linux-glibc", Some(&doc)).unwrap_err();
        assert!(err.contains("checksum mismatch"), "got: {err}");
    }

    #[test]
    fn test_verify_download_without_checksums_aborts() {
        let bytes = fake_elf();
        let err = verify_download(&bytes, "camon-linux-glibc", None).unwrap_err();
        assert!(err.contains("has no sha256sums.txt"), "got: {err}");
    }

    #[test]
    fn test_verify_download_present_but_missing_entry_aborts() {
        let bytes = fake_elf();
        let doc = format!("{}  some-other-file\n", sha256_hex(&bytes));
        let err = verify_download(&bytes, "camon-linux-glibc", Some(&doc)).unwrap_err();
        assert!(err.contains("does not list"), "got: {err}");
    }

    #[test]
    fn test_verify_download_non_elf_aborts_even_with_checksums() {
        let bytes = b"this is not a binary".to_vec();
        let doc = format!("{}  camon-linux-glibc\n", sha256_hex(&bytes));
        let err = verify_download(&bytes, "camon-linux-glibc", Some(&doc)).unwrap_err();
        assert!(err.contains("not a valid ELF"), "got: {err}");
    }

    #[test]
    fn a_staged_binary_is_complete_and_executable() {
        let dir = tempfile::tempdir().unwrap();
        let path = UpdatePaths::for_exe(&dir.path().join("camon")).staging;

        stage_binary(&path, &fake_elf()).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), fake_elf());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755, "the staged binary is not executable");

        stage_binary(&path, b"short").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"short");
    }

    #[test]
    fn a_staged_binary_that_cannot_be_written_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("camon.update.tmp");
        std::fs::create_dir(&path).unwrap();

        assert!(stage_binary(&path, &fake_elf()).is_err());
        assert!(!path.is_file());
    }

    #[test]
    fn the_version_command_output_is_what_the_probe_parses() {
        assert_eq!(
            parse_version_output(&crate::version_line()),
            Version::parse(env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn test_parse_version_output() {
        assert_eq!(
            parse_version_output("camon 0.6.0 (v0.6.0-2-gdeadbee)\n"),
            Version::parse("0.6.0")
        );
        assert_eq!(
            parse_version_output("camon 0.7.0-rc.1 (v0.7.0-rc.1)\n"),
            Version::parse("0.7.0-rc.1")
        );
        for bad in [
            "camon dev",
            "camon",
            "",
            "ffmpeg 6.1.1",
            "unknown command: version",
        ] {
            assert!(parse_version_output(bad).is_none(), "{bad:?}");
        }
    }

    fn fake_binary(dir: &Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn verdict(failure: ProbeFailure) -> String {
        match failure {
            ProbeFailure::Verdict(reason) => reason,
            other => panic!("a binary that answered was written off as unheard: {other:?}"),
        }
    }

    fn unrun(failure: ProbeFailure) -> String {
        match failure {
            ProbeFailure::Unrun(reason) => reason,
            other => panic!("a probe with no answer was classified as {other:?}"),
        }
    }

    fn impeded(failure: ProbeFailure) -> String {
        match failure {
            ProbeFailure::Impeded(reason) => reason,
            other => panic!("an installation camon cannot execute in was classified as {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_probe_version_reads_the_binarys_own_version() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(
            dir.path(),
            "camon",
            "[ \"$1\" = version ] || { echo \"unknown command: $1\" >&2; exit 1; }\n\
             echo 'camon 9.9.9 (v9.9.9)'",
        );
        assert_eq!(
            probe_version(&bin, VERSION_PROBE_TIMEOUT)
                .await
                .expect("a healthy binary was not probed"),
            version("9.9.9")
        );
    }

    #[tokio::test]
    async fn test_probe_version_refuses_a_binary_that_cannot_identify_itself() {
        let dir = tempfile::tempdir().unwrap();

        let old = fake_binary(
            dir.path(),
            "old",
            "echo 'unknown command: version' >&2\nexit 1",
        );
        let err = verdict(
            probe_version(&old, VERSION_PROBE_TIMEOUT)
                .await
                .unwrap_err(),
        );
        assert!(err.contains("exited with"), "got: {err}");
        assert!(err.contains("unknown command: version"), "got: {err}");

        let mute = fake_binary(dir.path(), "mute", "echo 'not a version at all'");
        let err = verdict(
            probe_version(&mute, VERSION_PROBE_TIMEOUT)
                .await
                .unwrap_err(),
        );
        assert!(err.contains("not a version line"), "got: {err}");
    }

    #[tokio::test]
    async fn a_binary_that_could_not_be_started_is_not_judged_by_the_probe() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("absent");
        let err = impeded(
            probe_version(&absent, VERSION_PROBE_TIMEOUT)
                .await
                .unwrap_err(),
        );
        assert!(err.contains("could not be started here"), "got: {err}");
    }

    #[tokio::test]
    async fn a_probe_that_ran_out_of_time_is_not_held_against_the_release() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), "hangs", "sleep 30");
        let err = unrun(
            probe_version(&bin, Duration::from_millis(300))
                .await
                .unwrap_err(),
        );
        assert!(err.contains("did not answer within"), "got: {err}");
    }

    #[tokio::test]
    async fn test_probe_version_refuses_a_binary_that_floods_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let bin = fake_binary(dir.path(), "loud", "head -c 1000000 /dev/zero | tr '\\0' x");
        let err = verdict(
            probe_version(&bin, Duration::from_millis(500))
                .await
                .unwrap_err(),
        );
        assert!(err.contains("printed at least"), "got: {err}");
    }

    #[tokio::test]
    async fn test_probe_version_kills_what_the_binary_forked() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("alive");
        let bin = fake_binary(
            dir.path(),
            "forks",
            &format!(
                "sh -c 'while :; do echo x >> {}; sleep 0.05; done' &\nsleep 30",
                marker.display()
            ),
        );
        let err = unrun(
            probe_version(&bin, Duration::from_millis(300))
                .await
                .unwrap_err(),
        );
        assert!(err.contains("did not answer within"), "got: {err}");

        tokio::time::sleep(Duration::from_millis(200)).await;
        let after_kill = std::fs::metadata(&marker).map(|m| m.len()).unwrap_or(0);
        tokio::time::sleep(Duration::from_millis(300)).await;
        let later = std::fs::metadata(&marker).map(|m| m.len()).unwrap_or(0);
        assert_eq!(after_kill, later, "a forked descendant outlived the probe");
    }

    #[tokio::test(start_paused = true)]
    async fn a_fork_the_kernel_refused_is_retried_and_then_deferred() {
        let attempts = std::cell::Cell::new(0u32);
        let started = tokio::time::Instant::now();

        let failure = spawn_probe(|| {
            attempts.set(attempts.get() + 1);
            Err(std::io::Error::from_raw_os_error(libc::EAGAIN))
        })
        .await
        .expect_err("a fork the kernel refused looked like a running binary");

        let reason = unrun(failure);
        assert!(reason.contains("still could not be started"), "{reason}");
        assert_eq!(
            attempts.get(),
            EXEC_RETRIES + 1,
            "the transient class did not get the retry policy"
        );
        assert_eq!(
            started.elapsed(),
            EXEC_RETRY_DELAY * EXEC_RETRIES,
            "the retries did not wait between attempts"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_fork_window_that_closes_costs_the_probe_nothing_but_the_wait() {
        let attempts = std::cell::Cell::new(0u32);
        let started = tokio::time::Instant::now();

        let child = spawn_probe(|| {
            attempts.set(attempts.get() + 1);
            if attempts.get() <= 2 {
                return Err(std::io::Error::from_raw_os_error(libc::ETXTBSY));
            }
            tokio::process::Command::new("/bin/sh")
                .arg("-c")
                .arg("exit 0")
                .spawn()
        })
        .await;

        let mut child = child.expect("a window that had already closed still refused the probe");
        assert_eq!(attempts.get(), 3);
        assert_eq!(
            started.elapsed(),
            EXEC_RETRY_DELAY * 2,
            "the probe waited for more windows than it hit"
        );
        child.wait().await.expect("the probe's child never exited");
    }

    #[tokio::test(start_paused = true)]
    async fn a_binary_this_kernel_cannot_run_is_the_only_start_failure_held_against_it() {
        for (errno, expected) in [(libc::ENOEXEC, true), (libc::EACCES, false)] {
            let attempts = std::cell::Cell::new(0u32);
            let failure = spawn_probe(|| {
                attempts.set(attempts.get() + 1);
                Err(std::io::Error::from_raw_os_error(errno))
            })
            .await
            .expect_err("a spawn that failed looked like a running binary");

            assert_eq!(attempts.get(), 1, "errno {errno} was retried");
            assert_eq!(
                matches!(failure, ProbeFailure::Verdict(_)),
                expected,
                "errno {errno} was classified the wrong way: {failure:?}"
            );
        }
    }

    #[test]
    fn every_way_the_machine_can_refuse_a_fork_is_transient() {
        for errno in [
            libc::EAGAIN,
            libc::ENOMEM,
            libc::EMFILE,
            libc::ENFILE,
            libc::ETXTBSY,
            libc::EINTR,
        ] {
            assert!(
                spawn_failure_is_transient(&std::io::Error::from_raw_os_error(errno)),
                "errno {errno} is not treated as a moment"
            );
        }
        for errno in [libc::ENOEXEC, libc::EACCES, libc::ENOENT] {
            assert!(
                !spawn_failure_is_transient(&std::io::Error::from_raw_os_error(errno)),
                "errno {errno} is treated as a moment"
            );
        }
    }

    #[test]
    fn only_a_binary_that_ran_is_written_down_against_the_release() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));

        for failure in [
            ProbeFailure::Unrun("no memory for a fork".to_string()),
            ProbeFailure::Impeded("the directory is mounted noexec".to_string()),
        ] {
            assert!(!after_a_failed_probe(&paths, None, "0.6.0", failure));
            assert_eq!(
                read_guard(&paths.guard),
                None,
                "a probe that produced no answer was recorded as a property of the release"
            );
            assert!(
                blocked_reason(None, "0.6.0", "0.5.0", unix_now()).is_none(),
                "the next check would not even try again"
            );
        }

        assert!(!after_a_failed_probe(
            &paths,
            None,
            "0.6.0",
            ProbeFailure::Verdict("it exited with signal 11".to_string()),
        ));
        let guard = read_guard(&paths.guard).expect("a crashing binary was not written down");
        assert!(guard.refused.unwrap().contains("signal 11"));
        assert!(
            guard.refusal_classified,
            "this camon's own verdict was recorded as one it could not classify"
        );
        let reason = blocked_reason(
            read_guard(&paths.guard).as_ref(),
            "0.6.0",
            "0.5.0",
            unix_now(),
        )
        .expect("a crashing binary would be downloaded again");
        assert!(reason.contains("signal 11"), "{reason}");

        let written = std::fs::read_to_string(&paths.guard).unwrap();
        for field in [
            "\"version\"",
            "\"attempts\"",
            "\"refused\"",
            "\"last_attempt_unix\"",
        ] {
            assert!(written.contains(field), "{field} is gone from {written}");
        }
    }

    #[test]
    fn a_refusal_from_a_camon_that_could_not_classify_it_is_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        std::fs::write(
            &paths.guard,
            "{\"version\":\"0.6.0\",\"attempts\":0,\"refused\":\"it could not be run: \
             Resource temporarily unavailable (os error 11)\",\"last_attempt_unix\":1753600000}",
        )
        .unwrap();

        let guard = read_guard(&paths.guard).expect("an older guard was discarded");
        assert!(guard.refused.is_some());
        assert!(!guard.refusal_classified);
        assert!(
            blocked_reason(Some(&guard), "0.6.0", "0.5.0", unix_now()).is_none(),
            "a box that upgraded mid-quarantine stays bricked"
        );

        after_a_failed_probe(
            &paths,
            Some(&guard),
            "0.6.0",
            ProbeFailure::Verdict("it exited with signal 11".to_string()),
        );
        let guard = read_guard(&paths.guard).expect("the re-test recorded nothing");
        assert!(guard.refusal_classified);
        assert!(blocked_reason(Some(&guard), "0.6.0", "0.5.0", unix_now()).is_some());

        std::fs::write(
            &paths.guard,
            "{\"version\":\"0.7.0\",\"attempts\":3,\"last_attempt_unix\":1753600000}",
        )
        .unwrap();
        let reason = blocked_reason(
            read_guard(&paths.guard).as_ref(),
            "0.7.0",
            "0.5.0",
            unix_now(),
        )
        .expect("an older camon's attempt count stopped counting");
        assert!(reason.contains("restart is not taking effect"), "{reason}");
    }

    #[test]
    fn a_re_test_that_cannot_run_leaves_the_old_record_where_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        let record = "{\"version\":\"0.6.0\",\"attempts\":0,\"refused\":\"it could not be run: \
                      Resource temporarily unavailable (os error 11)\",\
                      \"last_attempt_unix\":1753600000}";
        std::fs::write(&paths.guard, record).unwrap();

        let guard = read_guard(&paths.guard).expect("an older guard was discarded");
        assert!(blocked_reason(Some(&guard), "0.6.0", "0.5.0", unix_now()).is_none());

        after_a_failed_probe(
            &paths,
            Some(&guard),
            "0.6.0",
            ProbeFailure::Unrun("no memory for a fork".to_string()),
        );

        assert_eq!(
            std::fs::read_to_string(&paths.guard).unwrap(),
            record,
            "a re-test that produced no answer rewrote the record anyway"
        );
        assert!(
            blocked_reason(
                read_guard(&paths.guard).as_ref(),
                "0.6.0",
                "0.5.0",
                unix_now()
            )
            .is_none(),
            "the next check would not try again, so nothing could ever settle this"
        );
    }

    #[test]
    fn an_installed_binary_is_recorded_before_the_sync_it_could_wedge_in() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        let exe = paths.exe.clone();
        std::fs::write(&exe, b"the old binary").unwrap();
        std::fs::write(&paths.staging, b"the new binary").unwrap();

        let marker = camon::app::InstalledMarker::new();
        let (reached_sync, in_sync) = std::sync::mpsc::channel();
        let (release, held) = std::sync::mpsc::channel();
        let publisher = {
            let marker = marker.clone();
            std::thread::spawn(move || {
                publish_staged(&paths, &marker, |_| {
                    reached_sync.send(()).unwrap();
                    held.recv().unwrap();
                    Ok(())
                })
            })
        };

        in_sync
            .recv_timeout(Duration::from_secs(30))
            .expect("the sync was never reached");
        assert!(
            marker.recorded(),
            "a wedge in the sync would leave the installed update unrecorded"
        );
        assert_eq!(
            std::fs::read(&exe).unwrap(),
            b"the new binary",
            "the swap had not happened when the install was recorded"
        );

        release.send(()).unwrap();
        assert!(matches!(
            publisher.join().expect("the publisher panicked"),
            Ok(Published::Durable)
        ));
    }

    #[test]
    fn a_swap_that_never_happened_records_no_install() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        let marker = camon::app::InstalledMarker::new();

        let published = publish_staged(&paths, &marker, |_| Ok(()));

        assert!(published.is_err());
        assert!(
            !marker.recorded(),
            "a failed swap armed the restart enforcement"
        );
    }

    #[test]
    fn staging_files_from_interrupted_updates_are_collected_by_the_next_check() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        let stale = dir.path().join("camon.update.424242.tmp");
        std::fs::write(&stale, b"a whole release binary").unwrap();
        std::fs::write(&paths.staging, b"this process's own, not yet written").unwrap();

        let keep = [
            paths.guard.clone(),
            paths.lock.clone(),
            sibling(&paths.guard, ".424242.tmp"),
            dir.path().join("camon"),
            dir.path().join("camon-other.update.424242.tmp"),
        ];
        for path in &keep {
            std::fs::write(path, b"x").unwrap();
        }

        sweep_stale_staging(&paths);

        assert!(!stale.exists(), "a leftover staging file was left to rot");
        assert!(
            paths.staging.exists(),
            "the sweep took the staging file of the process running it"
        );
        for path in &keep {
            assert!(path.exists(), "the sweep took {}", path.display());
        }
    }

    fn declared<I>(length: Option<u64>, chunks: I) -> std::future::Ready<BodyResult<I::IntoIter>>
    where
        I: IntoIterator<Item = Result<Vec<u8>, String>>,
    {
        std::future::ready(Ok((length, futures_util::stream::iter(chunks))))
    }

    type BodyResult<S> = Result<(Option<u64>, futures_util::stream::Iter<S>), String>;

    #[tokio::test]
    async fn a_body_larger_than_the_limit_is_abandoned_by_name() {
        let body = std::future::ready(Ok((
            None,
            futures_util::stream::repeat(Ok::<Vec<u8>, String>(vec![0u8; 4096])),
        )));
        let err = fetch_bounded(
            body,
            "camon-linux-glibc",
            64 * 1024,
            Duration::from_secs(600),
        )
        .await
        .expect_err("an endless body was read to its end");

        assert!(err.contains("camon-linux-glibc"), "{err}");
        assert!(err.contains("would have reached 69632 bytes"), "{err}");
        assert!(err.contains("past the 65536 bytes"), "{err}");
        assert!(err.contains("raised in the updater"), "{err}");
    }

    #[tokio::test]
    async fn a_body_that_declares_itself_oversize_is_refused_before_it_is_read() {
        let body = declared(Some(70_000), [Ok(vec![0u8; 8])]);
        let err = fetch_bounded(
            body,
            "camon-linux-glibc",
            64 * 1024,
            Duration::from_secs(600),
        )
        .await
        .expect_err("a body that announced it was too big was downloaded anyway");

        assert!(err.contains("says it is 70000 bytes"), "{err}");
        assert!(err.contains("past the 65536 bytes"), "{err}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_request_that_never_finishes_is_abandoned_at_the_deadline() {
        let deadline = Duration::from_secs(600);

        let started = tokio::time::Instant::now();
        let body = std::future::ready(Ok((
            None,
            futures_util::stream::once(async { Ok::<Vec<u8>, String>(vec![0u8; 8]) })
                .chain(futures_util::stream::pending()),
        )));
        let err = fetch_bounded(body, "camon-linux-glibc", DOWNLOAD_LIMIT, deadline)
            .await
            .expect_err("a body that never ended was waited for forever");
        assert!(err.contains("had not finished downloading"), "{err}");
        assert!(err.contains("600s"), "{err}");
        assert_eq!(
            started.elapsed(),
            deadline,
            "the deadline is not what ended the download"
        );

        let started = tokio::time::Instant::now();
        let slow_to_open = async {
            tokio::time::sleep(deadline * 2).await;
            Ok((
                None,
                futures_util::stream::iter(Vec::<Result<Vec<u8>, String>>::new()),
            ))
        };
        let err = fetch_bounded(slow_to_open, "camon-linux-glibc", DOWNLOAD_LIMIT, deadline)
            .await
            .expect_err("a request that had not reached a body yet was waited out");
        assert!(err.contains("had not finished downloading"), "{err}");
        assert_eq!(
            started.elapsed(),
            deadline,
            "the deadline does not cover getting to the body"
        );
    }

    #[tokio::test]
    async fn a_body_within_the_limit_arrives_whole() {
        for length in [None, Some(8)] {
            let body = declared(length, [Ok(vec![1u8; 3]), Ok(vec![2u8; 5])]);
            let collected = fetch_bounded(body, "sha256sums.txt", 8, Duration::from_secs(600))
                .await
                .expect("a body inside every bound was refused");
            assert_eq!(collected, [1, 1, 1, 2, 2, 2, 2, 2], "declared {length:?}");
        }
    }

    #[test]
    fn a_release_whose_asset_is_not_its_tag_is_refused() {
        let reason = assess_staged(&version("0.6.0"), &version("0.7.0"), &version("0.5.0"))
            .expect_err("a mis-tagged release was accepted");
        assert!(reason.contains("does not match"), "got: {reason}");

        assert!(assess_staged(&version("0.8.0"), &version("0.7.0"), &version("0.5.0")).is_err());
    }

    #[test]
    fn a_release_that_is_not_actually_newer_is_refused() {
        let reason = assess_staged(&version("0.5.0"), &version("0.5.0"), &version("0.5.0"))
            .expect_err("a same-version install was accepted");
        assert!(reason.contains("not newer"), "got: {reason}");
        assert!(assess_staged(&version("0.4.0"), &version("0.4.0"), &version("0.5.0")).is_err());
    }

    #[test]
    fn an_honest_release_installs() {
        assert_eq!(
            assess_staged(&version("0.6.0"), &version("0.6.0"), &version("0.5.0")),
            Ok(())
        );
        assert_eq!(
            assess_staged(
                &version("0.6.0-rc.1"),
                &version("v0.6.0-rc.1"),
                &version("0.5.0")
            ),
            Ok(())
        );
    }

    #[test]
    fn the_same_version_is_not_installed_indefinitely() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        let mut installs = 0;
        for _ in 0..10 {
            let guard = read_guard(&guard_path);
            if blocked_reason(guard.as_ref(), "0.6.0", "0.5.0", unix_now()).is_some() {
                break;
            }
            record_attempt(&guard_path, "0.6.0", guard.as_ref()).unwrap();
            installs += 1;
        }

        assert_eq!(installs, MAX_INSTALL_ATTEMPTS, "the loop was not bounded");
        let reason = blocked_reason(
            read_guard(&guard_path).as_ref(),
            "0.6.0",
            "0.5.0",
            unix_now(),
        )
        .expect("the loop was not stopped");
        assert!(reason.contains("restart is not taking effect"), "{reason}");
        assert!(
            reason.contains("just now"),
            "the age is not reported: {reason}"
        );
    }

    #[test]
    fn a_refused_release_is_not_downloaded_again() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        record_refusal(
            &guard_path,
            "0.7.0",
            "the tag does not match the asset",
            None,
        )
        .unwrap();
        let reason = blocked_reason(
            read_guard(&guard_path).as_ref(),
            "0.7.0",
            "0.5.0",
            unix_now() + 86_400 * 3,
        )
        .expect("a refused release was fetched again");
        assert!(
            reason.contains("the tag does not match the asset"),
            "{reason}"
        );
        assert!(
            reason.contains("3 days ago"),
            "the age is not reported: {reason}"
        );
        assert!(blocked_reason(
            read_guard(&guard_path).as_ref(),
            "0.7.1",
            "0.5.0",
            unix_now()
        )
        .is_none());
    }

    #[test]
    fn a_newer_version_still_installs_after_one_was_given_up_on() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;
        for _ in 0..MAX_INSTALL_ATTEMPTS {
            let guard = read_guard(&guard_path);
            record_attempt(&guard_path, "0.6.0", guard.as_ref()).unwrap();
        }
        let guard = read_guard(&guard_path);
        assert!(blocked_reason(guard.as_ref(), "0.6.0", "0.5.0", unix_now()).is_some());

        assert!(blocked_reason(guard.as_ref(), "0.6.1", "0.5.0", unix_now()).is_none());
        assert_eq!(
            record_attempt(&guard_path, "0.6.1", guard.as_ref()).unwrap(),
            1
        );
        assert!(blocked_reason(
            read_guard(&guard_path).as_ref(),
            "0.6.1",
            "0.5.0",
            unix_now()
        )
        .is_none());
    }

    #[test]
    fn the_attempt_count_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        assert_eq!(record_attempt(&guard_path, "0.6.0", None).unwrap(), 1);
        let reloaded = read_guard(&guard_path).expect("guard did not survive");
        assert_eq!(reloaded.version, "0.6.0");
        assert_eq!(reloaded.attempts, 1);
        assert!(reloaded.last_attempt_unix > 0);
        assert_eq!(
            record_attempt(&guard_path, "0.6.0", Some(&reloaded)).unwrap(),
            2
        );
        assert_eq!(read_guard(&guard_path).unwrap().attempts, 2);
    }

    #[test]
    fn a_swap_that_fails_gives_the_attempt_back() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        record_attempt(&guard_path, "0.6.0", None).unwrap();
        restore_guard(&guard_path, None);
        assert_eq!(read_guard(&guard_path), None);

        record_attempt(&guard_path, "0.6.0", None).unwrap();
        let first = read_guard(&guard_path).unwrap();
        record_attempt(&guard_path, "0.6.0", Some(&first)).unwrap();
        restore_guard(&guard_path, Some(&first));
        assert_eq!(read_guard(&guard_path).unwrap().attempts, 1);
    }

    #[test]
    fn a_missing_or_corrupt_guard_does_not_brick_updates() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        assert_eq!(read_guard(&guard_path), None);
        assert!(blocked_reason(None, "0.6.0", "0.5.0", unix_now()).is_none());

        for corrupt in ["", "{", "not json", "{\"version\":\"0.6.0\"}"] {
            std::fs::write(&guard_path, corrupt).unwrap();
            let guard = read_guard(&guard_path);
            assert_eq!(guard, None, "for {corrupt:?}");
            assert!(
                blocked_reason(guard.as_ref(), "0.6.0", "0.5.0", unix_now()).is_none(),
                "for {corrupt:?}"
            );
        }

        assert_eq!(record_attempt(&guard_path, "0.6.0", None).unwrap(), 1);
        assert_eq!(read_guard(&guard_path).unwrap().attempts, 1);
    }

    #[test]
    fn a_guard_from_an_older_camon_still_reads() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;
        std::fs::write(
            &guard_path,
            "{\"version\":\"0.6.0\",\"attempts\":3,\"last_attempt_unix\":1753600000}",
        )
        .unwrap();
        let guard = read_guard(&guard_path).expect("an older guard was discarded");
        assert_eq!(guard.attempts, 3);
        assert_eq!(guard.refused, None);
        assert!(blocked_reason(Some(&guard), "0.6.0", "0.5.0", unix_now()).is_some());
    }

    #[test]
    fn each_installation_has_its_own_guard() {
        let debug = UpdatePaths::for_exe(Path::new("/opt/camon.debug"));
        let release = UpdatePaths::for_exe(Path::new("/opt/camon.release"));
        assert_eq!(debug.guard, PathBuf::from("/opt/camon.debug.update-guard"));
        assert_eq!(
            release.guard,
            PathBuf::from("/opt/camon.release.update-guard")
        );
        assert_ne!(debug.lock, release.lock);
        assert_ne!(debug.staging, release.staging);

        let plain = UpdatePaths::for_exe(Path::new("/usr/local/bin/camon"));
        assert_eq!(
            plain.guard,
            PathBuf::from("/usr/local/bin/camon.update-guard")
        );
        assert!(plain
            .staging
            .to_string_lossy()
            .contains(&std::process::id().to_string()));
    }

    #[test]
    fn only_one_updater_at_a_time_touches_an_installation() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));

        let held = UpdateLock::acquire(&paths.lock)
            .unwrap()
            .expect("first updater did not get the lock");
        assert!(
            UpdateLock::acquire(&paths.lock).unwrap().is_none(),
            "a second updater ran concurrently with the first"
        );

        drop(held);
        assert!(UpdateLock::acquire(&paths.lock).unwrap().is_some());
    }

    #[test]
    fn test_describe_age() {
        assert_eq!(describe_age(0), "just now");
        assert_eq!(describe_age(90), "just now");
        assert_eq!(describe_age(600), "10 minutes ago");
        assert_eq!(describe_age(7200), "2 hours ago");
        assert_eq!(describe_age(86_400 * 5), "5 days ago");
    }
}
