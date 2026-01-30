use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

const GITHUB_API_URL: &str = "https://api.github.com/repos/nsg/camon/releases/latest";

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

    let client = reqwest::Client::new();
    let release: Release = client
        .get(GITHUB_API_URL)
        .header("User-Agent", format!("camon/{current_version}"))
        .header("Accept", "application/vnd.github.v3+json")
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
        .find(|a| a.name == "camon")
        .ok_or("no 'camon' binary asset found in release")?;

    let bytes = client
        .get(&asset.browser_download_url)
        .header("User-Agent", format!("camon/{current_version}"))
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;

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
}
