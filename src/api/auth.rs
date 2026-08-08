//! What the API asks of a request, decided once at startup.

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

/// What the API asks of a request. One value, decided at startup by [`ApiAuth::resolve`], so
/// there is exactly one place that answers "is this deployment protected, and from what".
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
    /// Work out the policy and make it true: read or generate the token it needs, and log what
    /// the operator is left with.
    pub fn resolve(
        bind: IpAddr,
        configured: Option<&str>,
        allow_open: bool,
        token_file: Option<&Path>,
    ) -> std::io::Result<Self> {
        if let Some(token) = configured {
            return Ok(ApiAuth::Everything(token.to_string()));
        }
        // An outer layer (ingress, authenticating proxy) is the boundary; a
        // generated token would be a secret it has no way to present.
        if allow_open {
            return Ok(ApiAuth::Open);
        }
        // Loopback: the port itself is the boundary.
        if bind.is_loopback() {
            return Ok(ApiAuth::Open);
        }

        let (token, origin) = ensure_token(token_file)?;
        // Where the operator will find it; `None` when it is in-memory only.
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
        // Said at every start: the operator who needs the token rarely read
        // the log the day it was made. The token itself is never logged — a
        // secret in the journal is a secret in every archive it ships to.
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
/// a log line naming the bare relative `api-token` is a file the reader
/// cannot find.
fn shown(path: &Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

/// Where the token in use came from — what decides which warning to log.
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
        // No file, or a blank one: replaced, not believed — believing it would
        // match every request against an empty string.
        Ok(None) => {
            let token = generate_token()?;
            match publish_token(path, &token) {
                Ok(()) => Ok((token, TokenOrigin::Written)),
                Err(e) => Ok((token, TokenOrigin::Unwritable(Some(e)))),
            }
        }
        // Something unwritable-through at the path (symlink, directory) or
        // unreadable: refuse it and run on an in-memory token — a surprise
        // here must never produce an open API.
        Err(e) => Ok((generate_token()?, TokenOrigin::Unwritable(Some(e)))),
    }
}

/// Read the stored token.
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

/// Put the token in place atomically: a fresh 0600 `create_new` temp file (never an existing
/// symlink; mode set explicitly too, since a umask can strip bits), fsynced, renamed over the
/// target, directory fsynced after.
fn publish_token(path: &Path, token: &str) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut temp_name = std::ffi::OsString::from(".");
    temp_name.push(path.file_name().unwrap_or_else(|| "api-token".as_ref()));
    temp_name.push(format!(".{}.tmp", std::process::id()));
    let temp = dir.join(temp_name);

    // A leftover temp from a process that died mid-publish. Unlinked, not
    // opened — this removes a symlink rather than following it.
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

/// 24 bytes of kernel randomness, base64url-encoded: 32 characters that
/// survive a URL, a shell and a double-click unmangled. `/dev/urandom` rather
/// than a crate: camon is Linux-only and nothing else needs an RNG.
fn generate_token() -> std::io::Result<String> {
    let mut bytes = [0u8; 24];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// SHA-256 of the token in force.
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

/// The `?token=` fallback, for requests that cannot carry headers (`<img>`, native video);
/// confined to GET and HEAD.
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

/// What a [`ApiAuth::Writes`] deployment serves without a token.
fn is_read(method: &Method) -> bool {
    matches!(*method, Method::GET | Method::HEAD)
}

/// `401`, not `403`: `WWW-Authenticate` invites a credential, and the web
/// UI's 401 handler raises its token prompt. A 403 would say "authenticated,
/// still not allowed" — never true here — and leave the UI no way to ask.
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

    #[test]
    fn the_default_bind_with_no_token_guards_the_writes() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ApiAuth::resolve(OPEN_BIND, None, false, Some(&token_path(&dir))).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");
    }

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

    #[test]
    fn a_loopback_bind_asks_for_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ApiAuth::resolve(LOOPBACK, None, false, Some(&token_path(&dir))).unwrap();
        assert!(matches!(auth, ApiAuth::Open), "{auth:?}");
        assert!(!token_path(&dir).exists());
    }

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

    #[test]
    fn a_token_that_cannot_be_persisted_still_guards_the_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = token_path(&dir);
        std::fs::create_dir(&path).unwrap();

        let auth = ApiAuth::resolve(OPEN_BIND, None, false, Some(&path)).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");

        let auth = ApiAuth::resolve(OPEN_BIND, None, false, None).unwrap();
        assert!(matches!(auth, ApiAuth::Writes(_)), "{auth:?}");
    }

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
        let ApiAuth::Writes(token) = auth else {
            unreachable!()
        };
        assert_ne!(token, "important");
    }

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
