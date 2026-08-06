//! What the API asks of a request, decided once at startup.
//!
//! Camon's default bind is `0.0.0.0`, because both shipped deployments need it:
//! a systemd install is reached from browsers on the LAN, and the Home
//! Assistant add-on is reached by an ingress proxy over the container network.
//! That default used to come with no authentication at all, which made every
//! recording readable and — far worse — made the one write route reachable:
//! anybody on the LAN could PUT an all-blocking ignore mask and turn motion
//! recording off for a camera without a single line appearing in the log. Lost
//! footage, not just disclosure, and nothing about it looks like an attack
//! afterwards.
//!
//! So a deployment that would otherwise be open now gets a token of its own,
//! generated on first start and kept in a file beside the config, and that
//! token is required by everything that can change state. Reads are
//! deliberately left open in that case: an existing install upgrades in place
//! through the self-updater, unattended, and a binary that came back demanding
//! a secret nobody has yet for the *live view* would be an outage that arrives
//! at 3am. Setting `[http] token` closes reads too and is the documented way to
//! do it; see [`ApiAuth`] for the whole table.
//!
//! # DNS rebinding, and why there is no `Host` check
//!
//! A page on a domain the attacker controls can lure the operator's own browser
//! into reaching camon: the name resolves to the attacker's server first, is
//! re-resolved to camon's LAN address a moment later, and the page's scripts go
//! on talking to what is now the API. What that page can do is worth being
//! exact about, because the usual reassurance — "the browser will treat it as
//! cross-origin and block it" — is simply false here. An origin is scheme, host
//! and port; which address the host resolved to is no part of it. After the
//! rebind the page and the API share an origin as far as the browser is
//! concerned, so nothing is blocked, no preflight is sent, and the page is free
//! to issue a `PUT` carrying any `Authorization` header it likes.
//!
//! What stops that write is the token being secret, and nothing else. The
//! operator's copy of it lives in local storage under the origin they actually
//! browse camon at — its LAN address or name — which the attacker's origin
//! cannot read, and the token itself is 192 bits of kernel randomness, so it is
//! not arrived at by guessing either.
//!
//! Reading is the other answer, and it is not a good one: reads are open under
//! [`ApiAuth::Writes`], the rebound page is same-origin, and so it can read
//! them — footage included. The read residual is therefore not only "a
//! neighbour on the LAN"; it is also "the operator visited a page". `[http]
//! token` closes both, and is the only thing that does.
//!
//! A `Host` allowlist is the standard defence against this and is deliberately
//! not here. Default-on it breaks both shipped deployments: the operator who
//! reaches camon at `http://nvr.lan:8080` sends a name camon cannot know, and
//! the add-on is proxied with the Home Assistant frontend's own `Host`.
//! Default-off it protects nobody, while being a second and weaker lever for a
//! threat `[http] token` already closes completely.

use std::io::{Read, Write};
use std::net::IpAddr;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::sync::Arc;

use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// What the API asks of a request. One value, decided at startup by
/// [`ApiAuth::resolve`], so there is exactly one place that answers "is this
/// deployment protected, and from what".
///
/// | `[http]` settings | bind | policy |
/// |---|---|---|
/// | `token` set | any | [`Everything`](ApiAuth::Everything) — every `/api` route needs it |
/// | no token | loopback | [`Open`](ApiAuth::Open) — only this machine can reach the port |
/// | no token, `allow_open = true` | any | [`Open`](ApiAuth::Open) — something in front authenticates |
/// | no token | off-box | [`Writes`](ApiAuth::Writes) — a generated token guards state changes |
pub enum ApiAuth {
    /// No token is asked for. The port itself is the boundary.
    Open,
    /// Reads are served to anyone who can reach the port; anything that changes
    /// state must present the token. This is what a default install lands in.
    Writes(String),
    /// Every `/api` request must present the token, reads included.
    Everything(String),
}

/// Redacted on purpose: this is the type that holds the secret, and a `Debug`
/// that printed it would put it in every future error message and test failure.
impl std::fmt::Debug for ApiAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ApiAuth::Open => "Open",
            ApiAuth::Writes(_) => "Writes(<token>)",
            ApiAuth::Everything(_) => "Everything(<token>)",
        };
        f.write_str(name)
    }
}

impl ApiAuth {
    /// Work out the policy and make it true: read or generate the token the
    /// chosen policy needs, and say on the log what the operator is left with.
    ///
    /// `token_file` is where a generated token is kept — beside the config file
    /// that did not name one — or `None` when camon does not know where its
    /// config came from, in which case the generated token lives only as long
    /// as the process.
    ///
    /// The only failure is not being able to get randomness for a token camon
    /// has decided it needs. Continuing past that would mean serving the writes
    /// unguarded after having just logged that they are guarded, so it ends
    /// startup instead.
    pub fn resolve(
        bind: IpAddr,
        configured: Option<&str>,
        allow_open: bool,
        token_file: Option<&Path>,
    ) -> std::io::Result<Self> {
        if let Some(token) = configured {
            return Ok(ApiAuth::Everything(token.to_string()));
        }
        // An outer layer is the boundary (Home Assistant ingress, a reverse
        // proxy that authenticates). Generating a token here would gate writes
        // behind a secret that layer has no way to present.
        if allow_open {
            return Ok(ApiAuth::Open);
        }
        // Nothing off this machine can open the socket, so a token would only
        // be a lock on the operator's own front door.
        if bind.is_loopback() {
            return Ok(ApiAuth::Open);
        }

        let (token, origin) = ensure_token(token_file)?;
        // Where the operator will find it — `None` when the token exists only
        // in this process and there is nothing to point them at.
        let stored_at = match (&origin, token_file) {
            (TokenOrigin::Loaded, path) => path,
            (TokenOrigin::Written, path) => {
                if let Some(path) = path {
                    tracing::warn!(
                        token_file = %shown(path),
                        "no [http] token is set and the API is reachable from the network: \
                         generated one and wrote it to this file (mode 0600)"
                    );
                }
                path
            }
            (TokenOrigin::Unwritable(error), path) => {
                tracing::warn!(
                    token_file = %path.map(shown).unwrap_or_default(),
                    error = error.as_ref().map(|e| e.to_string()).unwrap_or_default(),
                    "the generated API token could not be stored in this file: writes are still \
                     guarded, but this run's token is held in memory only and a restart replaces \
                     it. Set [http] token in config.toml to a value of your own."
                );
                None
            }
        };
        // Said at every start, not only the first: the operator who has to find
        // the token is rarely the one who read the log the day it was made. The
        // token itself is never logged — the file is 0600 and one `cat` away,
        // and a secret written to the journal is a secret in every log archive
        // that journal is ever shipped to.
        tracing::warn!(
            %bind,
            token_file = %stored_at.map(shown).unwrap_or_default(),
            "the API is reachable from the network: anyone who can reach this address can watch \
             all footage, and changing anything (motion settings, ignore masks) needs the token \
             camon generated — the web UI asks for it the first time you save a setting. Set \
             [http] token in config.toml to require a token for reading too, or [http] bind = \
             \"127.0.0.1\" to keep camon on this machine."
        );
        Ok(ApiAuth::Writes(token))
    }

    /// The middleware state this policy needs, or `None` when it asks nothing
    /// and no layer should be installed at all.
    pub(super) fn layer(&self) -> Option<TokenAuth> {
        match self {
            ApiAuth::Open => None,
            ApiAuth::Writes(token) => Some(TokenAuth::new(token, true)),
            ApiAuth::Everything(token) => Some(TokenAuth::new(token, false)),
        }
    }
}

/// The token file's path as the operator should be told it: absolute, because
/// the service unit sets a working directory and the config path camon was
/// given is usually the bare `config.toml` relative to it — a log line reading
/// `api-token` names a file the reader cannot find.
fn shown(path: &Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

/// Where the token camon is using came from, which is all the difference
/// between the three things worth saying about it on the log.
enum TokenOrigin {
    /// Read back from the token file a previous start wrote.
    Loaded,
    /// Generated now and persisted, so it survives the next restart.
    Written,
    /// Generated now and nowhere else: no path to write to, or the write
    /// failed. It changes at every restart.
    Unwritable(Option<std::io::Error>),
}

fn ensure_token(path: Option<&Path>) -> std::io::Result<(String, TokenOrigin)> {
    let Some(path) = path else {
        return Ok((generate_token()?, TokenOrigin::Unwritable(None)));
    };
    match read_token(path) {
        Ok(Some(existing)) => Ok((existing, TokenOrigin::Loaded)),
        // No file yet, or one holding nothing usable — a blank left by an
        // interrupted first start, or an operator who emptied it. Treated as
        // absent rather than believed: believing it would mean matching every
        // request against an empty string, which is the config path's reason
        // for refusing an empty `[http] token` too.
        Ok(None) => {
            let token = generate_token()?;
            match publish_token(path, &token) {
                Ok(()) => Ok((token, TokenOrigin::Written)),
                Err(e) => Ok((token, TokenOrigin::Unwritable(Some(e)))),
            }
        }
        // Something is at the path that camon will not write through — a
        // symlink, a directory — or it cannot be read at all. Refuse it and run
        // on an in-memory token: the writes stay guarded either way, and the
        // one outcome that must never follow from a surprise here is the open
        // API this whole module exists to close.
        Err(e) => Ok((generate_token()?, TokenOrigin::Unwritable(Some(e)))),
    }
}

/// Read the stored token, refusing to follow a symlink or to read anything that
/// is not a regular file.
///
/// `Ok(None)` means "no usable token is stored, write one"; `Err` means "do not
/// write here". The distinction is the whole point of the check: camon's
/// installed service runs as root (no `User=` in the unit it writes), so a
/// symlink planted at `api-token` by anyone who can write the config directory
/// would otherwise be opened, truncated and chmodded 0600 as root — an
/// arbitrary file replaced with a token. It is a weak escalation story on its
/// own (write access to `/etc/camon` is most of the way to root already, and
/// the config file itself is read from there), which is why it is a cheap
/// guard rather than a redesign: `O_NOFOLLOW` plus a file-type check costs two
/// lines and removes the primitive entirely.
fn read_token(path: &Path) -> std::io::Result<Option<String>> {
    let mut file = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if !file.metadata()?.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let token = content.trim().to_string();
    Ok((!token.is_empty()).then_some(token))
}

/// Put the token in place atomically: a fresh 0600 temp file in the same
/// directory, fsynced, renamed over the target, and the directory fsynced after.
///
/// Every step earns its place. The temp file is `create_new`, so it can never
/// be an existing symlink; the mode is set explicitly as well as passed to
/// `open`, restoring any bit a restrictive umask stripped (a umask can only
/// narrow, never widen). The rename is what makes the
/// token appear whole — a reader is never handed a half-written secret, and it
/// also closes the gap between the read that found nothing and the write that
/// puts something there, since the rename replaces whatever the path names
/// without ever opening it. The directory fsync is what makes the rename
/// durable: without it a power cut can take the new name away while the file's
/// contents survive, and camon would come back generating a *different* token
/// while the operator's browser still holds the one from before.
fn publish_token(path: &Path, token: &str) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut temp_name = std::ffi::OsString::from(".");
    temp_name.push(path.file_name().unwrap_or_else(|| "api-token".as_ref()));
    temp_name.push(format!(".{}.tmp", std::process::id()));
    let temp = dir.join(temp_name);

    // A leftover from a previous process that died between creating the temp
    // and renaming it. Unlinked, not opened — this removes a symlink itself
    // rather than following it.
    let _ = std::fs::remove_file(&temp);

    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        file.write_all(token.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        std::fs::File::open(dir)?.sync_all()
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// 24 bytes of kernel randomness, base64url-encoded: 32 characters that survive
/// a URL, a shell and a double-click unescaped and unmangled.
///
/// `/dev/urandom` rather than a crate — camon is Linux-only (it installs
/// systemd/OpenRC units and ships as a Home Assistant add-on), no random
/// number generator is otherwise in the dependency graph, and this is the whole
/// requirement.
fn generate_token() -> std::io::Result<String> {
    let mut bytes = [0u8; 24];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// SHA-256 of the token in force. The presented token is hashed the same way
/// before comparison: `==` on `[u8; 32]` is not guaranteed to be constant-time,
/// but it runs over two fixed-width digests, so how far it gets says nothing
/// usable about the secret's length or content.
#[derive(Clone)]
pub(super) struct TokenAuth {
    digest: Arc<[u8; 32]>,
    /// True when reads are served to anyone and only state changes are gated —
    /// [`ApiAuth::Writes`].
    writes_only: bool,
}

impl TokenAuth {
    fn new(token: &str, writes_only: bool) -> Self {
        Self {
            digest: Arc::new(token_digest(token)),
            writes_only,
        }
    }
}

fn token_digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// The `?token=` fallback, for requests that cannot carry headers: `<img>`
/// sources (thumbnails, filmstrips, debug maps) and native video elements.
/// Those are reads, so the fallback is confined to GET and HEAD — anything
/// that changes state must present the header.
///
/// It leaks, and knowingly. A URL routinely ends up in proxy logs and browser
/// caches in a way a header does not, so a token that has been through this
/// fallback should be assumed to exist in those places. The web UI appends it
/// to media URLs whenever it holds a token at all — including the *generated*
/// write token, from the first save onwards — so the exposure is not confined
/// to deployments that set `[http] token`, and it is one more reason the token
/// file is the operator's to rotate (delete it and restart) if a proxy log ever
/// goes somewhere it should not.
#[derive(Deserialize)]
struct TokenQuery {
    token: Option<String>,
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

/// What counts as a read, and therefore as something a [`ApiAuth::Writes`]
/// deployment serves without a token.
///
/// Decided by method, not by route, and deliberately so: a route added later is
/// gated the moment it is spelled `post`/`put`/`delete`, with nobody having to
/// remember a list. GET handlers are held to that promise — the detection-debug
/// poll is the one with a side effect at all, and what it changes is how long
/// the detector keeps its own frames, not anything the operator stored.
fn is_read(method: &Method) -> bool {
    matches!(*method, Method::GET | Method::HEAD)
}

/// `401`, not `403`: the caller has presented no usable credential and can fix
/// that by presenting one, which is exactly what `WWW-Authenticate` invites and
/// what the web UI's own handler does — a 401 from any request raises the token
/// prompt, stores what is typed, and reloads. A 403 would say "authenticated,
/// still not allowed", which is never true here, and would leave the UI with no
/// way to ask.
pub(super) async fn require_token(
    State(auth): State<TokenAuth>,
    request: Request,
    next: Next,
) -> Response {
    let read = is_read(request.method());
    if auth.writes_only && read {
        return next.run(request).await;
    }

    let presented = match bearer_token(request.headers()) {
        Some(token) => Some(token.to_string()),
        None if read => Query::<TokenQuery>::try_from_uri(request.uri())
            .ok()
            .and_then(|q| q.0.token),
        None => None,
    };

    match presented {
        Some(token) if token_digest(&token) == *auth.digest => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "unauthorized",
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const OPEN_BIND: IpAddr = IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED);
    const LOOPBACK: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);

    fn token_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("api-token")
    }

    /// The one case the whole change exists for: the shipped default — bind
    /// `0.0.0.0`, no token, no opt-out — must not resolve to an open API.
    #[test]
    fn the_default_bind_with_no_token_guards_the_writes() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ApiAuth::resolve(OPEN_BIND, None, false, Some(&token_path(&dir))).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");
    }

    /// A token the operator set is a lock on the whole API, reads included, and
    /// stays one wherever camon is bound — that is the documented way to close
    /// the read exposure a generated token deliberately leaves open.
    #[test]
    fn a_token_in_the_config_is_required_for_reading_too() {
        let dir = tempfile::tempdir().unwrap();
        for bind in [OPEN_BIND, LOOPBACK] {
            let auth =
                ApiAuth::resolve(bind, Some("mine"), false, Some(&token_path(&dir))).unwrap();
            assert!(
                matches!(&auth, ApiAuth::Everything(t) if t == "mine"),
                "{auth:?}"
            );
        }
        assert!(
            !token_path(&dir).exists(),
            "a configured token still had one generated behind it"
        );
    }

    /// Loopback is the boundary already: nothing off the machine can open the
    /// socket, so camon does not invent a secret for the operator's own box —
    /// and does not leave a token file lying in their working directory.
    #[test]
    fn a_loopback_bind_asks_for_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ApiAuth::resolve(LOOPBACK, None, false, Some(&token_path(&dir))).unwrap();
        assert!(matches!(auth, ApiAuth::Open), "{auth:?}");
        assert!(!token_path(&dir).exists());
    }

    /// How the Home Assistant add-on stays working: `run.sh` forces
    /// `http.allow_open = true`, because ingress authenticates the user before
    /// proxying and reaches camon from the container network — never loopback.
    /// A generated token would be a secret the proxy has no way to present, and
    /// the add-on's UI would lose its settings editor with no way to fix it.
    #[test]
    fn allow_open_is_the_add_on_saying_ingress_is_the_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ApiAuth::resolve(OPEN_BIND, None, true, Some(&token_path(&dir))).unwrap();
        assert!(matches!(auth, ApiAuth::Open), "{auth:?}");
        assert!(
            !token_path(&dir).exists(),
            "the add-on's config folder got a token file it never asked for"
        );
    }

    /// The token has to be the same one after a restart — a token that rotated
    /// every start would send the operator back to the file on every update,
    /// and the self-updater restarts unattended.
    #[test]
    fn the_generated_token_is_kept_and_reused() {
        let dir = tempfile::tempdir().unwrap();
        let path = token_path(&dir);

        let ApiAuth::Writes(first) = ApiAuth::resolve(OPEN_BIND, None, false, Some(&path)).unwrap()
        else {
            panic!("expected a generated token");
        };
        let ApiAuth::Writes(second) =
            ApiAuth::resolve(OPEN_BIND, None, false, Some(&path)).unwrap()
        else {
            panic!("expected a generated token");
        };

        assert_eq!(first, second);
        assert_eq!(std::fs::read_to_string(&path).unwrap().trim(), first);
        assert!(first.len() >= 32, "a short token: {}", first.len());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "the token file was readable by others");
    }

    /// Two starts must not produce the same secret. Cheap to check and the one
    /// property a home-rolled generator can get catastrophically wrong.
    #[test]
    fn two_generated_tokens_are_not_the_same_token() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let ApiAuth::Writes(first) =
            ApiAuth::resolve(OPEN_BIND, None, false, Some(&token_path(&a))).unwrap()
        else {
            panic!("expected a generated token");
        };
        let ApiAuth::Writes(second) =
            ApiAuth::resolve(OPEN_BIND, None, false, Some(&token_path(&b))).unwrap()
        else {
            panic!("expected a generated token");
        };
        assert_ne!(first, second);
    }

    /// A blank token file — an interrupted first start, an operator emptying it
    /// — is replaced, not believed. Believing it would hand every request an
    /// empty string to match and open the writes back up without a word.
    #[test]
    fn a_blank_token_file_is_replaced_rather_than_believed() {
        let dir = tempfile::tempdir().unwrap();
        let path = token_path(&dir);
        std::fs::write(&path, "   \n").unwrap();

        let ApiAuth::Writes(token) = ApiAuth::resolve(OPEN_BIND, None, false, Some(&path)).unwrap()
        else {
            panic!("expected a generated token");
        };
        assert!(!token.trim().is_empty());
        assert_eq!(std::fs::read_to_string(&path).unwrap().trim(), token);
    }

    /// A token file that cannot be written — a read-only `/etc`, a config
    /// directory camon's user does not own — still leaves the writes guarded.
    /// The token is ephemeral and the log says so; what it must never do is
    /// fall back to serving them open.
    #[test]
    fn a_token_that_cannot_be_persisted_still_guards_the_writes() {
        let dir = tempfile::tempdir().unwrap();
        // A directory where the file belongs: every open of it fails, whatever
        // user the test runs as.
        let path = token_path(&dir);
        std::fs::create_dir(&path).unwrap();

        let auth = ApiAuth::resolve(OPEN_BIND, None, false, Some(&path)).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");

        // And with no path at all to keep it in.
        let auth = ApiAuth::resolve(OPEN_BIND, None, false, None).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");
    }

    /// The installed service runs as root, so a symlink sitting where the token
    /// file belongs must not be followed: opening it would truncate whatever it
    /// points at, write a secret into it and chmod it 0600 — as root, and
    /// silently. Camon refuses the path instead and runs on an in-memory token,
    /// which is the one outcome that keeps both the writes guarded and the
    /// operator's file intact.
    #[test]
    #[cfg(unix)]
    fn a_symlink_where_the_token_belongs_is_refused_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        std::fs::write(&victim, "important\n").unwrap();
        let path = token_path(&dir);
        std::os::unix::fs::symlink(&victim, &path).unwrap();

        let auth = ApiAuth::resolve(OPEN_BIND, None, false, Some(&path)).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");

        assert_eq!(
            std::fs::read_to_string(&victim).unwrap(),
            "important\n",
            "the symlink was followed and its target overwritten"
        );
        assert!(
            std::fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the planted symlink was replaced instead of refused"
        );
        // Nor was the target's content taken as the token.
        let ApiAuth::Writes(token) = auth else {
            unreachable!()
        };
        assert_ne!(token, "important");
    }

    /// A token published while the machine loses power must be all there or not
    /// there — never a truncated secret the operator pastes in and cannot use.
    /// The rename is what guarantees it, and it also means no reader ever sees
    /// the file mid-write.
    #[test]
    fn publishing_a_token_leaves_no_partial_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = token_path(&dir);
        publish_token(&path, "a-token").unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a-token\n");
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|name| name != "api-token")
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );
    }

    #[test]
    fn only_get_and_head_count_as_reads() {
        assert!(is_read(&Method::GET));
        assert!(is_read(&Method::HEAD));
        for method in [Method::PUT, Method::POST, Method::DELETE, Method::PATCH] {
            assert!(!is_read(&method), "{method} was treated as a read");
        }
    }
}
