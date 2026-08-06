//! The self-updater: what camon does when a newer release of itself exists.
//!
//! Once at startup and every twelve hours after that, camon asks GitHub for the
//! latest release and — if its tag names a version newer than this build —
//! downloads the binary asset, checks it, runs it to ask what it really is, and
//! renames it over the binary this process is executing. The process then asks
//! for the same graceful drain a SIGTERM does, and the service manager starts
//! the replacement.
//!
//! None of that sits on the path that starts the cameras. The check is a
//! supervised task ([`camon::app`]) spawned during startup and left to run
//! beside the recording, so a release download that is slow, stalled or
//! deliberately trickled costs no footage — it used to be awaited inline,
//! between the HTTP bind and the first camera, where a tarpit could hold an NVR
//! off the air indefinitely. Off that path it can still do two things to a
//! recording process: grow it, and never finish. So every request it makes is
//! bounded by an allocation limit and by one deadline covering the send and the
//! body together ([`fetch_bounded`]).
//!
//! # What it cannot fix by itself
//!
//! Some conditions here never heal, and camon has nowhere to report them but
//! the log — there is no MQTT entity, metric or API field carrying update
//! state, so the level a line is written at is the whole of who notices.
//!
//! One of them is certain enough to shout about: a staged binary this
//! installation may not execute — a `noexec` mount, a permission camon does not
//! have. Nothing about that is the release's doing and nothing about waiting
//! changes it, so it is retried every twelve hours for as long as the box runs
//! and every attempt says so at `error`, because an operator has to change
//! something before any release will ever install here.
//!
//! The rest are indistinguishable from bad luck and stay at `warn`, which is
//! where the periodic check logs a failure ([`camon::app`]). A box that cannot
//! reach GitHub at all and a box whose uplink was down for the minute the check
//! ran produce exactly the same error, and a release that hangs when asked its
//! version looks like a release probed on a machine that was too squeezed to
//! hear the answer — deliberately, since that ambiguity is the whole of
//! [`ProbeFailure`]'s asymmetry. Raising those to `error` would put a routine
//! network blip in the same class as a broken installation twice a day, and the
//! class would stop meaning anything. What an operator has instead is the
//! repetition: the same warning at every check, twice a day, is the signal.
//!
//! # What the checksum does, and what it does not
//!
//! Every asset is checked against the `sha256sums.txt` published beside it in
//! the same release. That is an integrity check and only an integrity check: it
//! establishes that the bytes which arrived are the bytes the release names, so
//! a truncated, corrupted, or mirror-mangled download is caught before anything
//! is executed. It establishes nothing about *authenticity*, because the
//! checksum document travels the same path from the same release — anything
//! able to publish or edit a camon release publishes the hash of whatever
//! binary it likes, and camon verifies it, installs it, and the service manager
//! starts it as root. The trust root is therefore GitHub's release permissions,
//! not any cryptography camon itself can check; closing that gap needs a
//! signature over the release made with a key the binary ships (minisign), and
//! that is a deliberate, deferred decision rather than an oversight. Until it
//! lands, `update.enabled` defaults to false and turning it on is a statement
//! that whoever can publish a camon release is trusted with root on this box.
//!
//! # Which installations run this at all
//!
//! Only a native install whose config says `update.enabled = true`. The default
//! is false ([`camon::config`]), and the Home Assistant add-on forces
//! `--set update.enabled=false` at launch because its container filesystem is
//! ephemeral and its updates arrive through the add-on store and the Supervisor
//! instead. So this file is opt-in on bare-metal and systemd installs, and dead
//! code everywhere else — which is why its failures have to be loud on the few
//! boxes that do run it.

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
/// Idle budget: reqwest arms it flat until the response headers arrive, then
/// per response frame with a reset on each. It bounds a connection that stops
/// producing, which is not the same as bounding one that never stops — a body
/// dripped a byte at a time resets this forever, and is bounded by the
/// deadlines below instead.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// How long each of the two small JSON/text requests gets, all in: connect,
/// TLS, whatever redirects it follows, and reading the body to its end. Both
/// are documents a human could read; a server that cannot finish one in half a
/// minute is not one to take a binary from.
const METADATA_TIMEOUT: Duration = Duration::from_secs(30);
/// And how much of them camon will hold. A release index runs to a few tens of
/// kilobytes and a checksum document to a few hundred bytes, so this is three
/// orders of magnitude of headroom and still not enough to matter to the box.
const METADATA_LIMIT: u64 = 4 * 1024 * 1024;

/// How long the release binary gets to arrive, from the first packet of the
/// request to the last of the body, redirects and all — GitHub answers an asset
/// URL with one, so a per-body bound would leave the hop before it unbounded.
///
/// Derived from the other end: the download has to fit in it at a rate an NVR
/// necessarily already has. A release asset is around 26 MB today, so ten
/// minutes asks for about 43 kB/s sustained — a fraction of what one camera's
/// RTSP stream costs, on a box whose entire job is carrying several of those.
/// A link too slow for this is a link too slow for camon to be recording over.
const DOWNLOAD_DEADLINE: Duration = Duration::from_secs(600);
/// And how many bytes of it camon will allocate for.
///
/// The download is buffered whole — it has to be hashed before any of it is
/// written where it could be executed — so this is memory taken from a process
/// that is recording, on boxes with a history of being OOM-killed overnight.
/// That is what sets the number, rather than any guess about how large a
/// release may one day be: the measured asset is around 26 MB, and two and a
/// half times that is generous without being an amount a small box would
/// notice losing. Growth past it is not a silent failure — the update aborts
/// naming this limit, which is a one-line change away from being raised.
const DOWNLOAD_LIMIT: u64 = 64 * 1024 * 1024;

/// How long the staged binary gets to say what version it is. It prints one
/// short line and exits, so this is a hang detector rather than a budget.
const VERSION_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the probe waits for a process group it has just killed to be
/// reaped.
const PROBE_REAP_TIMEOUT: Duration = Duration::from_secs(5);
/// Hard caps on what the probe reads from a binary it does not trust yet. The
/// version line is about forty bytes; a release that writes a diagnostic dump
/// to stdout instead must not be able to grow this process, which is recording
/// while the probe runs and may be on a box with a history of OOM kills.
///
/// **A limit every deployed camon enforces on every future release.** Reaching
/// it is a verdict, and the camon applying it is whichever one is *already
/// installed* — so a future camon that prints four kilobytes at `version` would
/// be refused by every older deployment there is, permanently, and no change
/// made in that future camon could undo it. Like [`crate::version_line`], this
/// is a contract with the installed base rather than a local choice: what
/// `version` prints has to stay one short line.
const PROBE_STDOUT_LIMIT: u64 = 4096;
const PROBE_STDERR_LIMIT: u64 = 4096;
/// How often, and how many times, the probe re-attempts an exec that failed for
/// a reason the machine can stop having — see [`spawn_failure_is_transient`].
/// Half a second total, far past any fork window.
const EXEC_RETRY_DELAY: Duration = Duration::from_millis(50);
const EXEC_RETRIES: u32 = 10;

/// How many times the same release version may be installed before camon stops
/// installing it.
///
/// An install that works needs exactly one attempt: the restarted process finds
/// its own version equal to the release and never reaches this code again. A
/// second attempt therefore means the restart did not take effect — the service
/// manager starts a different binary than the one that was replaced, or
/// something puts the old one back — and that is a restart loop, not an update.
/// Two spare attempts are allowed for the case where the new process died
/// before it could check anything.
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

    // Corruption protection (not a security boundary — see the trust model at
    // the top of this file). A failure here is never recorded as a refusal: a
    // mismatch is what a truncated or corrupted download looks like, and the
    // next attempt may well succeed.
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

    // Every path this function *returns* by takes the staging file with it — it
    // is a complete, executable copy of a release binary sitting next to the
    // real one. The paths that do not return leave it: a process killed here,
    // or this future dropped by a supervised restart or a shutdown, both of
    // which now happen while the box is doing everything else it does, since
    // the check no longer runs alone in front of startup. That is what
    // `sweep_stale_staging` above is for — one leftover per abandoned attempt,
    // collected by the next check.
    stage_binary(&paths.staging, &bytes)?;

    // The tag is a label a human typed; what the binary says about itself is
    // what decides whether the process started after the restart will download
    // this same asset all over again. Asked while the download is still staged,
    // so an asset that is not what its tag claims is refused rather than
    // installed and then discovered a restart later.
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
        // Not an error to return: the binary *is* installed, and reporting
        // failure here would leave the process running the old one and
        // downloading the same release again on the next check, spending the
        // attempts the guard exists to ration.
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
///
/// Three steps whose *order* is the whole point. The rename is an ordinary
/// inode swap, running executable or not: nothing opens the live binary for
/// writing (which is what would earn `ETXTBSY`), only the directory entry
/// changes, and this process goes on executing the inode it started from until
/// it exits, which is when that inode is finally freed. The moment it returns,
/// a replacement binary exists under the name the service manager will start,
/// and the process is committed to restarting — so that is where the marker is
/// flipped, before the sync and before anything else on this path.
///
/// The sync is what makes the rename survive a power cut, and it is also the
/// one call here that can wait forever: it opens the parent directory and
/// `fsync`s it, synchronously, on the filesystem that may be exactly what is
/// failing. Wedged there with the marker already up, the enforcement thread
/// still ends the process at its deadline and the replacement still starts.
/// Wedged there with the marker set afterwards — which is what this used to do,
/// by way of the updater's return — nothing anywhere knows an update happened,
/// and the box runs the old binary until someone notices. It is a parameter so
/// a test can park in it.
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

/// Fetch a whole response into memory under both bounds camon is able to state:
/// how many bytes it will allocate for, and how long the request has from its
/// first packet to the end of its body.
///
/// The client's connect and read timeouts bound reaching the server and any
/// stall between frames. Neither bounds a server that never stalls and never
/// finishes — a body dripped one byte per second resets the read timeout
/// forever, and an asset URL that redirects can spend a fresh connect budget on
/// every hop — and neither bounds how much it drips. Both are the same hazard
/// here: this reads inside the process that is recording, from a release camon
/// has not yet decided to trust, so an unbounded body is an unbounded
/// allocation and an unbounded wait in an NVR. That is why the deadline wraps
/// the send as well as the read, and why the limit is a limit on the
/// *allocation*: a body that declares itself oversize is refused before a
/// single byte of it is fetched, one that lies about its length is abandoned at
/// the frame that crosses the limit rather than read to its end, and an honest
/// one is allocated for exactly once instead of being doubled into by a growing
/// `Vec` — which on a 26 MB release is 26 MB of peak rather than 52.
///
/// Both failures name the limit *and* what was attempted against it: a release
/// that has legitimately outgrown the limit has to be recognisable as that from
/// one log line.
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
///
/// The fsync is the point: the rename that publishes this file is atomic but
/// says nothing about its contents, so without it a power cut moments after an
/// update can leave a truncated or empty `camon` under the live name — an
/// installation that no longer starts at all, which is a worse outcome than any
/// the update was meant to fix. The mode belongs to the same guarantee: a binary
/// that comes back without its `x` bit is just as unbootable, so it is set
/// before the fsync rather than after one that could not have covered it.
///
/// A failed write takes the staging file with it. It is a multi-megabyte file
/// named after this process's pid, so nothing — not even the next update from
/// this same installation — would ever clean it up.
///
/// The mode is *all* this carries over, and that is a documented limitation
/// rather than an oversight. The file is created fresh and published by rename,
/// so anything else the old binary's inode carried — file capabilities, other
/// extended attributes, a non-default SELinux label — is not on the new one.
/// Nothing camon installs puts them there: [`crate::install`] writes a systemd
/// unit and an OpenRC script that both run camon as root with no capability
/// set of their own, the Home Assistant
/// add-on runs as root in a container and disables the updater outright, and no
/// shipped path calls `setcap`. So this is only ever felt by an operator who
/// hardened the install by hand — dropped camon to an unprivileged user and
/// granted, say, `cap_net_bind_service` so it could serve on port 80 — for whom
/// the first successful self-update silently produces a binary that no longer
/// starts. Copying the attributes across would mean either an xattr crate or
/// raw `libc` calls, which is out of proportion to a case camon never creates;
/// the workaround is to re-apply the hardening after an update, or to pin
/// `update.enabled = false` and update through whatever put the capabilities
/// there in the first place.
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
///
/// The kinds are not the same fact and must not be written down as if they
/// were. A binary that *ran* and then exited nonzero, printed nonsense, or
/// flooded its output has told camon something about itself that no amount of
/// retrying will change: that is a verdict, and it is worth the guard file it
/// costs. A probe that never ran — a fork refused for want of memory or process
/// slots, an exec that lost a race with a camera pipeline's fork — has told
/// camon nothing about the release at all. It is a fact about the machine at
/// that moment, on machines whose nightly OOM kill is the reason it happens,
/// and recording it as a property of the artifact is how a transient shortage
/// used to blacklist a version until someone deleted a file by hand.
///
/// The asymmetry decides every case that could be argued either way, and it is
/// the same one [`blocked_reason`] is built on: refusing a release that would
/// have worked costs an operator with a shell, while re-testing one that will
/// never work costs a download every twelve hours. So the answer timeout is
/// *not* a verdict — a squeezed box faulting in a fresh 26 MB binary off a slow
/// card, or a starved runtime that misses the answer it was given, both look
/// exactly like a hang, and the retry loop above deliberately waits into the
/// worst instant of that squeeze before even starting the probe.
#[derive(Debug)]
enum ProbeFailure {
    /// The binary ran, and what it did is an answer.
    Verdict(String),
    /// The probe could not be run or could not be heard, so there is no answer
    /// to record.
    ///
    /// Usually the machine's doing and usually over by the next check. But the
    /// bucket is defined by what camon can *tell*, not by what is true, so a
    /// release that genuinely hangs when asked its version — or leaves a child
    /// holding the pipe the probe is reading — lands here too, and is fetched
    /// and probed again every twelve hours for as long as it is the latest
    /// release, without ever being written down. That is the side of the
    /// asymmetry this file chose: a download twice a day on a box that is
    /// otherwise fine, rather than a permanent refusal on a box that was
    /// briefly short of memory.
    Unrun(String),
    /// The same, except that nothing about waiting will change it: this
    /// installation cannot execute the file it just staged. Recorded against
    /// the release no more than an [`Self::Unrun`] is — the next release would
    /// meet the same wall — but said louder, because it needs a person.
    Impeded(String),
}

/// Deal with a probe that produced no version.
///
/// Returns `false` every way — nothing was installed — but only the verdict
/// leaves anything behind on disk. The other two differ in volume only: an
/// impediment is at `error` because it will still be there in twelve hours and
/// the log is the only place camon can say so.
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

/// Refuse this release for good: say why, and write it down so the asset is not
/// downloaded again every twelve hours to reach the same verdict. Everything
/// recorded here is a property of the artifact, which only a new release can
/// change — unlike a checksum mismatch, which is what a corrupt download looks
/// like and is always retried, or a probe that could not run, which is a
/// property of the machine ([`ProbeFailure`]).
///
/// Returns `false` for the caller to hand on: nothing was installed.
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
///
/// Both conditions are about the *next* process rather than this one: it will
/// compare the release tag against whatever version the installed binary
/// reports, so an asset that is not the version its tag claims leaves the tag
/// looking newer than what is installed — the restart loop, one cycle later.
/// Requiring equality rather than merely "not older than the tag" is the honest
/// expectation, since a release is built from its tag.
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
///
/// Every one of these is about the state of the box at the instant of the fork
/// or exec, and none of them is about the file being started:
///
/// * `EAGAIN`/`ENOMEM` — the kernel would not give this process another task or
///   the memory to hold it. Routine on a small box under pressure, which is
///   exactly the box this runs on.
/// * `EMFILE`/`ENFILE` — no descriptor for the pipes, here or system-wide.
/// * `ETXTBSY` — the file is open for writing somewhere. The download finished
///   and closed its own fd, but any thread that forked while it was open — and
///   the camera pipelines fork ffmpeg all day — handed a copy to a child, and
///   the file counts as written-to until that child execs or exits. A moment,
///   not a state.
/// * `EINTR` — a signal landed in the middle of the call.
///
/// `ENOEXEC` is deliberately not here: "this kernel has nothing that can run
/// this file" is a statement about the artifact — a release built for another
/// architecture, say — and it will be just as true after every retry, on this
/// box, forever. That one is a verdict, and what makes it a safe one is that
/// the guard is keyed by version: a corrected release published tomorrow is a
/// different key and is fetched and tried as if nothing had happened, so the
/// worst this can do is stop camon re-downloading the one build that cannot
/// run here.
fn spawn_failure_is_transient(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(
            libc::EAGAIN | libc::ENOMEM | libc::EMFILE | libc::ENFILE | libc::ETXTBSY | libc::EINTR
        )
    )
}

/// Start the probe, retrying the failures that are moments rather than states.
///
/// Retrying comes first because most of these last less than one sleep: the
/// fork window that produces `ETXTBSY` closes as soon as the child that
/// inherited the descriptor execs, and a memory or task shortage on a box that
/// is otherwise working is usually over just as quickly. Only one that outlives
/// half a second of retrying is reported, and even then as something about this
/// machine rather than about the release.
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
            // Anything left is about the staging file or the filesystem holding
            // it — a directory mounted `noexec`, a permission camon does not
            // have, a file something else removed — and the next release would
            // fail here in exactly the same way. Not the artifact's doing, and
            // not recorded against it; but not something that passes on its own
            // either, which is the difference `Impeded` carries.
            Err(e) => {
                return Err(ProbeFailure::Impeded(format!(
                    "it could not be started here: {e}"
                )))
            }
        }
    }
}

/// Ask a binary what version it is, by running it.
///
/// The bare `version` subcommand rather than `--version`: a camon from before
/// either existed rejects an unknown subcommand and exits immediately, while it
/// ignores an unknown flag and would start a second NVR out of the staging
/// file. A binary that runs and cannot state its version is refused, not
/// retried — it cannot be checked against the tag, and only a new release can
/// change that. A binary that could not be started is a different answer
/// entirely; [`ProbeFailure`] is where the two are kept apart.
///
/// What this bounds: how long the binary runs, how much of its output is kept,
/// and its process group, which is killed once it has had its say so anything
/// it forked cannot outlive it. A descendant that leaves the group on purpose
/// (`setsid`) is beyond that, as it would be for any supervisor — this bounds
/// accidents, not a hostile binary. What runs here has passed the release
/// checksum, and if it installs, the service manager starts it as root moments
/// later anyway.
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

    // Sound only here, before the child is reaped: until then its pid — and
    // with it the group id, since it leads the group — cannot be handed to
    // anything else. Reaching this point without timing out means both pipes
    // hit EOF, which for an ordinary process happens inside exit, past the
    // point where its status is already decided, so this cannot turn a healthy
    // exit into a killed one.
    kill_group(group);
    let status = tokio::time::timeout(PROBE_REAP_TIMEOUT, child.wait()).await;

    // What the binary said is a verdict; what it did not manage to say in time
    // is not. The probe reads a process that was started moments after the
    // exec-retry loop gave the machine every chance to find the resources for
    // it: on the box this classification exists for, faulting in and linking a
    // fresh 26 MB binary off a squeezed card can outlast this budget, and a
    // starved runtime can miss an answer that did arrive. Both would be
    // recorded as a permanent property of a release that is fine.
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
        // Neither of these two is an answer about the binary: the first is this
        // process losing track of a child it started, the second a process that
        // will not die even for `SIGKILL`, which on Linux means it is stuck in
        // the kernel — a machine in trouble, not a release to give up on.
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
///
/// The trailing field is the git describe string, which is not a version and is
/// never compared; only the second field is read, through the same parser as
/// every other version camon handles.
fn parse_version_output(stdout: &str) -> Option<Version> {
    let mut fields = stdout.lines().next()?.split_whitespace();
    if fields.next()? != "camon" {
        return None;
    }
    Version::parse(fields.next()?)
}

/// A semantic version, as far as precedence is concerned.
///
/// Hand-written because comparing versions wrongly is how an updater loops or
/// stalls, and the rules are short: the numeric core fields compare as numbers,
/// a pre-release is *older* than the release it precedes, and build metadata
/// does not count at all — `1.0.0+a` and `1.0.0` are the same version, which is
/// why it is kept for display only and left out of the comparison.
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

/// Delete staging files left behind by attempts that never got to clean up
/// after themselves.
///
/// Every path that *returns* from [`check_and_update`] removes its own staging
/// file. The paths that do not return cannot: the process is killed, or the
/// future is dropped where it was awaiting — by a supervised restart, by the
/// drain, by the deadline on a download. Each of those leaves a multi-megabyte
/// file named after a pid that will not come round again, and nothing else in
/// camon has any reason to look at it. Left alone they accumulate, one per
/// abandoned attempt, in the directory holding the binary.
///
/// Safe under the update lock and only there: the lock is exclusive per
/// installation, so no other updater has a staging file open, and every
/// `<exe>.update.*.tmp` beside it is therefore finished with. This process's
/// own name is skipped anyway — it has not written it yet at this point, and
/// skipping it costs nothing and removes the one way this could delete a file
/// still in use. The guard's temporary (`<exe>.update-guard.<pid>.tmp`) does
/// not match the prefix, which ends in the dot that separates it.
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

/// Exclusive lock over one installation's update, held from before the guard is
/// read until after the binary is swapped.
///
/// `flock` rather than a pid file: the kernel drops it when the holder dies,
/// however it dies, so a crashed updater cannot leave updates wedged forever.
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
///
/// Kept on disk because the failure it bounds is a *restart* loop: every count
/// held in memory is thrown away by the very restart that would repeat the
/// install. It lives beside the binary rather than in the data dir — the
/// updater already needs write access there to replace the binary, and it has
/// no config to learn a data dir from — and it is deliberately plain enough to
/// read and delete by hand, which is how an operator retries a version camon
/// has given up on.
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
    /// Whether the refusal above was reached by a camon that can tell a binary
    /// which ran and failed from a probe that never ran ([`ProbeFailure`]).
    ///
    /// Absent — and so false — in every record written before that distinction
    /// existed, which is what makes this the migration. Those refusals may have
    /// been nothing but a fork the kernel refused during a memory squeeze, and
    /// a box that upgraded mid-quarantine would otherwise stay bricked on one
    /// forever, so they are not believed: see [`blocked_reason`] for the three
    /// ways that plays out.
    #[serde(default)]
    refusal_classified: bool,
    /// When this record was last written, so a repeated verdict can say how old
    /// it is rather than reading as news.
    last_attempt_unix: u64,
}

/// Read the guard, treating "not there" and "not readable" alike as no record.
///
/// Fail-safe by design: a guard camon cannot read must not be able to stop
/// updates, since that is a state a truncated write or a stray edit can produce
/// and there would be nothing to restart into to fix it. The very next write
/// replaces it with a well-formed one, so a corrupt file costs at most the
/// attempts it had counted.
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
///
/// An unclassified refusal — one written by a camon from before a probe that
/// could not run was told apart from a binary that ran and failed — is the one
/// record here that is not believed. It is re-tested, and there are three ways
/// that ends:
///
/// * the re-test reaches a verdict, which is written down classified, and the
///   release is refused from then on — one extra download, once, ever;
/// * the re-test succeeds, and a release that was never broken installs — which
///   is the whole reason for not believing the record;
/// * the re-test *also* fails to run, which records nothing, so the
///   unclassified record is still there and grants another re-test on the next
///   check. On a box that is permanently short of memory that repeats every
///   twelve hours indefinitely.
///
/// The third is the accepted cost, and it is the same steady state a release
/// with no guard at all would have: a download every twelve hours, bounded by
/// the cadence, on a box that is already failing at something more important.
/// It is bought with the asymmetry this whole file is built on — a mistaken
/// refusal costs an operator with a shell, and there may not be one.
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
    // Staged, flushed, renamed, and the directory flushed too, like every other
    // file camon has to find again after an unclean stop.
    // Not the shared `{name}.tmp` staging name: two camon processes can race
    // for the same guard, and each needs its own staging file.
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
///
/// Handles both text (`<hash>  name`) and binary (`<hash> *name`) line formats,
/// skips blank and comment lines, and validates the hash is 64 hex digits.
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
///
/// Checks the `\x7fELF` magic, a valid class/data-encoding, and an object type
/// of executable (`ET_EXEC`) or shared object (`ET_DYN`, used by PIE binaries).
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
///
/// The ELF tripwire and the release's own `sha256sums.txt` are both required. A
/// release that publishes no checksums is refused rather than installed
/// unverified: every release since that document was introduced has one, and
/// camon only ever installs a version newer than the one running, so a newer
/// release without it is a broken publish — and an unverified install is a
/// worse answer to that than refusing to update until it is fixed.
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

    /// The two rules a plain dotted-number compare gets backwards, and the
    /// reason the parser is worth writing: a release is newer than its own
    /// pre-releases, and build metadata is not a version difference at all.
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
        // Kept for display even though it takes no part in the comparison.
        assert_eq!(version("1.0.0+build.9").to_string(), "1.0.0+build.9");
        assert_eq!(version("v1.0.0-rc.1").to_string(), "1.0.0-rc.1");
    }

    /// Anything camon cannot compare exactly it refuses rather than
    /// approximates: the old comparison silently dropped fields it could not
    /// read, which made nonsense of every comparison involving them.
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

    // Known SHA-256 test vectors (FIPS 180-2 / NIST).
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
        // Binary mode uses `*` before the name; hash may be uppercase.
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
        // Right name, but hash is too short / not hex.
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

    /// Minimal well-formed 64-bit little-endian ELF header (ET_DYN), padded to
    /// the required 64 bytes.
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

    /// Nothing is installed — let alone executed — without the release's own
    /// checksum to check it against.
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

    /// Whether the fsync reached the platter is not observable from a test; the
    /// two properties the swap depends on are — the staged file holds the whole
    /// download, and it is executable before anything renames it into place.
    #[test]
    fn a_staged_binary_is_complete_and_executable() {
        let dir = tempfile::tempdir().unwrap();
        let path = UpdatePaths::for_exe(&dir.path().join("camon")).staging;

        stage_binary(&path, &fake_elf()).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), fake_elf());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755, "the staged binary is not executable");

        // Truncating, so a shorter release cannot inherit the tail of a longer
        // one left by an earlier attempt of the same process.
        stage_binary(&path, b"short").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"short");
    }

    /// A staging file is named after this process's pid, so one left behind by a
    /// failed write is never written over and never removed by anything else.
    #[test]
    fn a_staged_binary_that_cannot_be_written_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("camon.update.tmp");
        std::fs::create_dir(&path).unwrap();

        assert!(stage_binary(&path, &fake_elf()).is_err());
        assert!(!path.is_file());
    }

    /// The updater and `camon version` are one contract: what main prints has
    /// to be what the probe reads back, or every update is refused.
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

    /// Stand-in for a downloaded binary: a shell script camon can run.
    fn fake_binary(dir: &Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// What a probe failure was, when the test expects the binary to have run
    /// and answered badly. Panics on the other kind rather than letting it pass
    /// as a matching string: the whole point of the distinction is that these
    /// two are not interchangeable.
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

    /// The never-healing kind: not recorded against the release either, but
    /// nothing about waiting fixes it.
    fn impeded(failure: ProbeFailure) -> String {
        match failure {
            ProbeFailure::Impeded(reason) => reason,
            other => panic!("an installation camon cannot execute in was classified as {other:?}"),
        }
    }

    /// Also pins the argument: the script answers only to the bare `version`
    /// subcommand, which is the spelling an older camon rejects instead of
    /// starting a second NVR out of the staging file.
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

    /// Every way a staged binary that *ran* can fail to identify itself is a
    /// verdict on the release: it cannot be checked against the tag, and no
    /// number of retries will change what it printed.
    #[tokio::test]
    async fn test_probe_version_refuses_a_binary_that_cannot_identify_itself() {
        let dir = tempfile::tempdir().unwrap();

        // A camon from before the `version` subcommand existed. It reports on
        // stderr, as the real one does, so what camon logs is its own words.
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

    /// A binary camon could not start is not a binary camon has judged. The
    /// staging file being gone is a fault of this installation — a `noexec`
    /// mount, a permission, a stray cleanup — and would meet the next release
    /// in exactly the same way, so it is never held against this one; it is the
    /// kind that does not pass on its own, which is what earns the louder log.
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

    /// The probe's own budget is a fact about this box, not about the release.
    /// Ten seconds is long enough for a binary to print one line and short
    /// enough to be missed by a box that is being squeezed — faulting in and
    /// linking a fresh 26 MB binary off a slow card, or a runtime too starved
    /// to notice the answer — and that squeeze is precisely the state the
    /// exec-retry loop just finished waiting through.
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

    /// An unbounded read here is a memory bomb inside a process that is
    /// recording: the probe stops reading long before that, and says so. What
    /// it says is a verdict — this is output the binary produced, not an
    /// answer camon failed to hear.
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

    /// A binary that forks leaves nothing behind: the probe kills its process
    /// group, not just the process it started.
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

    /// The failure this classification exists for. A fork the kernel refuses
    /// for want of memory or task slots is a fact about the box at that instant
    /// — routine on one that gets OOM-killed overnight, which is the kind of
    /// box that runs this — and used to be written down as a permanent property
    /// of the release, blacklisting a version until an operator deleted a file
    /// by hand. It is retried on the spot instead, and if it is still failing
    /// afterwards it defers.
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

    /// And what the retry is *for*: the window closes and the probe goes ahead
    /// as if nothing had happened. This is M3's ETXTBSY case — a camera
    /// pipeline's fork holding a copy of the descriptor the download wrote
    /// through — which lasts until that child execs and is gone long before the
    /// retries are.
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
        // Reaped, so this test leaves no child behind for the next one to find.
        child.wait().await.expect("the probe's child never exited");
    }

    /// One attempt each, no retry, for the two that retrying cannot help — and
    /// they land on opposite sides of the line: a file this kernel has no way
    /// to execute is a property of the artifact, while one this installation
    /// may not execute is a property of the installation and would meet every
    /// future release the same way.
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

    /// The transient set, named one by one, because the cost of getting a
    /// member of it wrong is an installation that stops updating until someone
    /// with a shell notices.
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
        // And the ones that are not about the machine's state.
        for errno in [libc::ENOEXEC, libc::EACCES, libc::ENOENT] {
            assert!(
                !spawn_failure_is_transient(&std::io::Error::from_raw_os_error(errno)),
                "errno {errno} is treated as a moment"
            );
        }
    }

    /// The two halves at the seam where they cost something: only one of them
    /// leaves a guard file behind, and only that one stops the next check.
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

        // The other direction of the same upgrade: the record an older camon
        // knows how to read is still all there, with the new field added
        // beside it rather than in place of anything. A downgrade — or a
        // rollback to the binary that wrote the guard before this one — reads
        // its own fields and ignores what it does not know.
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

    /// The upgrade story, in the two endings that settle. A guard written by a
    /// camon that could not tell the two apart may be recording nothing worse
    /// than a fork that failed once during a memory squeeze, so it is not
    /// believed — it is tried again, and a verdict *that* attempt reaches is
    /// recorded in the new form and is believed forever after. A box that
    /// upgraded mid-quarantine therefore neither stays bricked on a transient
    /// failure nor re-downloads a genuinely broken release more than once. What
    /// the old record still decides unchanged is the attempt count: that half
    /// never confused the two. The third ending — a re-test that cannot run
    /// either — is in
    /// [`a_re_test_that_cannot_run_leaves_the_old_record_where_it_was`].
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

        // Whatever this attempt finds is recorded in the new form — and if it
        // is a verdict, that is the end of it.
        after_a_failed_probe(
            &paths,
            Some(&guard),
            "0.6.0",
            ProbeFailure::Verdict("it exited with signal 11".to_string()),
        );
        let guard = read_guard(&paths.guard).expect("the re-test recorded nothing");
        assert!(guard.refusal_classified);
        assert!(blocked_reason(Some(&guard), "0.6.0", "0.5.0", unix_now()).is_some());

        // The other half of an old record is untouched: a restart loop counted
        // by an older camon still stops this one.
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

    /// The third way the migration ends, and the one the cost model has to own:
    /// the re-test cannot run either, so there is nothing to record, so the
    /// unclassified record is still there and grants another re-test on the
    /// next check — every twelve hours, indefinitely, on a box that never stops
    /// being short of memory. That is the accepted price. It is the same steady
    /// state as a release with no guard at all, it is bounded by the check
    /// cadence, and the alternative — believing a verdict that may have been
    /// nothing but a squeezed fork — needs an operator with a shell to undo.
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

    /// The instant a replacement binary exists under the running name, the
    /// thread that guarantees the restart has to know — not when the updater
    /// gets round to returning, because between those two moments is a
    /// directory `fsync` on the filesystem that may be exactly what is failing.
    /// Parked in that sync, this process has the update installed, the
    /// enforcement running, and — if the flag came later — nothing connecting
    /// the two, which is the old binary running until someone notices.
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

    /// And nothing is recorded when nothing was installed: a rename that fails
    /// leaves the old binary in place, so arming the enforcement would end a
    /// perfectly healthy process for no reason.
    #[test]
    fn a_swap_that_never_happened_records_no_install() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        let marker = camon::app::InstalledMarker::new();

        // No staging file to rename.
        let published = publish_staged(&paths, &marker, |_| Ok(()));

        assert!(published.is_err());
        assert!(
            !marker.recorded(),
            "a failed swap armed the restart enforcement"
        );
    }

    /// Interrupted attempts leave a complete release binary beside the real
    /// one, and only the next check can collect it: the paths that leak are the
    /// ones that never return — a killed process, a future dropped by a
    /// supervised restart or by the drain — and those are ordinary events now
    /// that the check runs beside everything else camon does.
    #[test]
    fn staging_files_from_interrupted_updates_are_collected_by_the_next_check() {
        let dir = tempfile::tempdir().unwrap();
        let paths = UpdatePaths::for_exe(&dir.path().join("camon"));
        let stale = dir.path().join("camon.update.424242.tmp");
        std::fs::write(&stale, b"a whole release binary").unwrap();
        std::fs::write(&paths.staging, b"this process's own, not yet written").unwrap();

        // None of these is a staging file, and every one of them is something
        // an update needs to survive the sweep.
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

    /// A body already open, with whatever length the server declared for it.
    fn declared<I>(length: Option<u64>, chunks: I) -> std::future::Ready<BodyResult<I::IntoIter>>
    where
        I: IntoIterator<Item = Result<Vec<u8>, String>>,
    {
        std::future::ready(Ok((length, futures_util::stream::iter(chunks))))
    }

    type BodyResult<S> = Result<(Option<u64>, futures_util::stream::Iter<S>), String>;

    /// A body camon has not decided to trust is read into the memory of a
    /// process that is recording, on boxes that get OOM-killed as it is. A
    /// server that says nothing about the length is held to the limit frame by
    /// frame: this one stops at the limit rather than at the end of the body —
    /// the stream never ends — and says both numbers, because a release that
    /// has honestly grown past the limit has to be told apart from a server
    /// feeding camon something else from one log line.
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

    /// And a server that *does* say how long the body is is taken at its word
    /// before a byte of it is fetched — the cheapest possible refusal, and the
    /// one that keeps the allocation honest: what is reserved up front is what
    /// the body claims, never more than the limit.
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

    /// The bound neither the read timeout nor a per-body one can provide: a
    /// server that never stalls and never finishes. The deadline covers the
    /// request from its first packet, because an asset URL answers with a
    /// redirect and the hop after it would otherwise start a fresh budget of
    /// its own. Driven by the paused clock rather than by waiting, so what it
    /// measures is the deadline and not the machine it runs on.
    #[tokio::test(start_paused = true)]
    async fn a_request_that_never_finishes_is_abandoned_at_the_deadline() {
        let deadline = Duration::from_secs(600);

        // Stalled in the body, one chunk in: the connection is live and every
        // per-frame timeout is being reset.
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

        // And stalled before that, in the send: connecting, handshaking, or
        // following redirects. This one would eventually answer — twice the
        // deadline later — which is how the assertion tells a bound that covers
        // getting to the body from one that only starts once it is open.
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

    /// The ordinary case, and the boundary: a body exactly the size of the
    /// limit is not over it, declared or not.
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

    /// The case that makes the probe worth having, and the one an "is it newer
    /// than me" check misses: installing 0.6.0 under the tag 0.7.0 passes that
    /// check — it *is* newer — and the next process finds 0.7.0 newer all over
    /// again, which is the loop.
    #[test]
    fn a_release_whose_asset_is_not_its_tag_is_refused() {
        let reason = assess_staged(&version("0.6.0"), &version("0.7.0"), &version("0.5.0"))
            .expect_err("a mis-tagged release was accepted");
        assert!(reason.contains("does not match"), "got: {reason}");

        // Equally wrong the other way: an asset built from a later commit than
        // the tag it is published under.
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
        // Including a pre-release, which the project has to stay able to ship.
        assert_eq!(
            assess_staged(
                &version("0.6.0-rc.1"),
                &version("v0.6.0-rc.1"),
                &version("0.5.0")
            ),
            Ok(())
        );
    }

    /// The loop this whole guard exists for: a release that installs, restarts,
    /// and comes back to the same decision. Each pass reads the guard from disk
    /// exactly as a freshly started process does — nothing carries over in
    /// memory, because in the real failure nothing can.
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

    /// A refused release is not fetched again to be refused again — and the
    /// reason survives, so the operator reads the same explanation every time.
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
        // And it says nothing about any other version.
        assert!(blocked_reason(
            read_guard(&guard_path).as_ref(),
            "0.7.1",
            "0.5.0",
            unix_now()
        )
        .is_none());
    }

    /// The guard may never stand in the way of a real update: a version camon
    /// has given up on says nothing about the next one.
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
        // And the count starts over rather than inheriting the dead version's.
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

    /// Only the file matters — a process that starts, counts an attempt and
    /// dies must leave the count behind for the next one.
    #[test]
    fn the_attempt_count_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        assert_eq!(record_attempt(&guard_path, "0.6.0", None).unwrap(), 1);
        // A new process: everything it knows it reads back off the disk.
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

    /// An install that was counted and then did not happen gives the attempt
    /// back, so three failed swaps cannot block a version that never ran.
    #[test]
    fn a_swap_that_fails_gives_the_attempt_back() {
        let dir = tempfile::tempdir().unwrap();
        let guard_path = UpdatePaths::for_exe(&dir.path().join("camon")).guard;

        // Nothing was recorded before: the guard goes away entirely.
        record_attempt(&guard_path, "0.6.0", None).unwrap();
        restore_guard(&guard_path, None);
        assert_eq!(read_guard(&guard_path), None);

        record_attempt(&guard_path, "0.6.0", None).unwrap();
        let first = read_guard(&guard_path).unwrap();
        record_attempt(&guard_path, "0.6.0", Some(&first)).unwrap();
        restore_guard(&guard_path, Some(&first));
        assert_eq!(read_guard(&guard_path).unwrap().attempts, 1);
    }

    /// Fail safe, in both directions: no guard and an unreadable guard leave
    /// updates working, and the next install replaces the unreadable one.
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

    /// A guard written before `refused` existed has to keep working, or an
    /// update to this very code would read as a corrupt guard.
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

    /// The updater's files hang off the binary's *name*, not its extension:
    /// `camon.debug` and `camon.release` are separate installations and must
    /// not share an attempt budget, which `set_extension` would have made them
    /// do by rewriting both to `camon.update-guard`.
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
        // Staging is process-unique, so two updaters cannot write one file.
        assert!(plain
            .staging
            .to_string_lossy()
            .contains(&std::process::id().to_string()));
    }

    /// Two updaters on one installation would otherwise both read attempts = 2,
    /// both pass the check, and both write 3 — two installs on a budget of one.
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

        // Released with the holder, however it goes away.
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
