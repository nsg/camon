use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::time::Duration;

use sha2::{Digest, Sha256};

const GITHUB_API_URL: &str = "https://api.github.com/repos/nsg/camon/releases/latest";
const CHECKSUMS_ASSET: &str = "sha256sums.txt";

/// This runs before the cameras start, so a stalled request keeps the whole NVR
/// offline until it gives up.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Idle budget: reqwest arms it flat until the response headers arrive, then
/// per response frame with a reset on each. These requests carry no body, so
/// the flat phase is only connect-and-wait, and the multi-MB binary download is
/// bounded by stalling rather than by how long it legitimately takes.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Total ceiling for the two small JSON/text requests. The binary download
/// deliberately has none: a slow link may need minutes for it.
const METADATA_TIMEOUT: Duration = Duration::from_secs(30);

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

pub async fn check_and_update() -> Result<bool, Box<dyn std::error::Error>> {
    let current_version = env!("CARGO_PKG_VERSION");
    tracing::info!(version = %current_version, "checking for updates");

    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()?;
    let release: Release = client
        .get(GITHUB_API_URL)
        .header("User-Agent", format!("camon/{current_version}"))
        .header("Accept", "application/vnd.github.v3+json")
        .timeout(METADATA_TIMEOUT)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let latest_version = release
        .tag_name
        .strip_prefix('v')
        .unwrap_or(&release.tag_name);

    if latest_version == current_version {
        tracing::info!(version = %current_version, "already up to date");
        return Ok(false);
    }

    if !is_newer(latest_version, current_version) {
        tracing::info!(
            current = %current_version,
            latest = %latest_version,
            "current version is newer or equal"
        );
        return Ok(false);
    }

    tracing::info!(
        current = %current_version,
        latest = %latest_version,
        "newer version available, updating"
    );

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == "camon-linux-glibc")
        .or_else(|| release.assets.iter().find(|a| a.name == "camon"))
        .ok_or("no 'camon-linux-glibc' binary asset found in release")?;

    let bytes = client
        .get(&asset.browser_download_url)
        .header("User-Agent", format!("camon/{current_version}"))
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;

    // Corruption protection (not a security boundary — an attacker able to swap
    // the binary asset can swap the checksum too). Releases published before
    // sha256sums.txt existed have no such asset: we warn and proceed unverified
    // so self-update does not brick, but once the asset is present it must
    // verify. Presence/absence of the CHECKSUMS_ASSET asset is the distinction.
    let checksums = match release.assets.iter().find(|a| a.name == CHECKSUMS_ASSET) {
        Some(sums_asset) => {
            let text = client
                .get(&sums_asset.browser_download_url)
                .header("User-Agent", format!("camon/{current_version}"))
                .timeout(METADATA_TIMEOUT)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?;
            Some(text)
        }
        None => None,
    };

    verify_download(&bytes, &asset.name, checksums.as_deref())?;

    let current_exe = std::env::current_exe()?;
    let temp_path = temp_path_for(&current_exe);

    std::fs::write(&temp_path, &bytes)?;
    std::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(0o755))?;
    std::fs::rename(&temp_path, &current_exe)?;

    tracing::info!(version = %latest_version, "update applied successfully");
    Ok(true)
}

fn temp_path_for(exe: &std::path::Path) -> PathBuf {
    let mut temp = exe.to_path_buf();
    temp.set_extension("update.tmp");
    temp
}

fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    let l = parse(latest);
    let c = parse(current);
    l > c
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

/// Validate downloaded bytes before they replace the running binary.
///
/// Always enforces the ELF tripwire. When a `sha256sums.txt` document is
/// available it must list this asset with a matching hash; a missing document
/// (legacy release) is allowed through with a warning.
fn verify_download(bytes: &[u8], asset_name: &str, checksums: Option<&str>) -> Result<(), String> {
    if !is_valid_elf(bytes) {
        return Err(format!(
            "downloaded {asset_name} is not a valid ELF binary ({} bytes) — refusing to install",
            bytes.len()
        ));
    }

    let Some(checksums) = checksums else {
        tracing::warn!(
            asset = %asset_name,
            "release has no {CHECKSUMS_ASSET}; proceeding with update unverified"
        );
        return Ok(());
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

    #[test]
    fn test_is_newer() {
        assert!(is_newer("1.0.1", "1.0.0"));
        assert!(is_newer("1.1.0", "1.0.9"));
        assert!(is_newer("2.0.0", "1.9.9"));
        assert!(!is_newer("1.0.0", "1.0.0"));
        assert!(!is_newer("1.0.0", "1.0.1"));
        assert!(!is_newer("0.1.0", "0.1.0"));
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

    #[test]
    fn test_verify_download_legacy_no_checksums_proceeds() {
        let bytes = fake_elf();
        assert!(verify_download(&bytes, "camon-linux-glibc", None).is_ok());
    }

    #[test]
    fn test_verify_download_present_but_missing_entry_aborts() {
        let bytes = fake_elf();
        let doc = format!("{}  some-other-file\n", sha256_hex(&bytes));
        let err = verify_download(&bytes, "camon-linux-glibc", Some(&doc)).unwrap_err();
        assert!(err.contains("does not list"), "got: {err}");
    }

    #[test]
    fn test_verify_download_non_elf_aborts_even_without_checksums() {
        let bytes = b"this is not a binary".to_vec();
        let err = verify_download(&bytes, "camon-linux-glibc", None).unwrap_err();
        assert!(err.contains("not a valid ELF"), "got: {err}");
    }
}
