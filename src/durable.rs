//! The staged-write convention every file camon must find again after an unclean stop is
//! written with: stage as `{name}.tmp`, fsync the staging file, rename it into place, fsync the
//! directory holding it.

use std::path::{Path, PathBuf};

/// Staging path for an atomic write: `{file_name}.tmp` next to the final path.
/// Startup orphan recovery keys off this exact convention.
pub fn tmp_path(final_path: &Path) -> PathBuf {
    let mut name = final_path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    final_path.with_file_name(name)
}

/// Write `data` to `path` and fsync it. Says nothing about the *name*: a
/// freshly created file still needs [`sync_dir`] on its directory.
pub fn write_synced(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    file.write_all(data)?;
    file.sync_all()
}

/// Async twin of [`write_synced`], taking the contents as chunks written back
/// to back, so segmented events need not be concatenated into one buffer.
pub async fn write_all_synced_async(path: &Path, chunks: &[&[u8]]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::File::create(path).await?;
    for chunk in chunks {
        file.write_all(chunk).await?;
    }
    file.sync_all().await?;
    Ok(())
}

/// Replace `final_path` with `data` through a staging file, so a reader (or a crash) never sees
/// a half-written file under the live name.
pub fn replace_atomic(final_path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = tmp_path(final_path);
    let result = std::fs::write(&tmp, data).and_then(|()| std::fs::rename(&tmp, final_path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Async twin of [`replace_atomic`].
pub async fn replace_atomic_async(final_path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = tmp_path(final_path);
    let result = match tokio::fs::write(&tmp, data).await {
        Ok(()) => tokio::fs::rename(&tmp, final_path).await,
        Err(e) => Err(e),
    };
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

/// The directory holding `path`'s entry, as something that can be opened:
/// `Path::parent` answers the empty path for a bare relative name (permitted
/// as `data_dir`), and opening `""` is ENOENT — the entry lives in `.`.
pub fn parent_dir(path: &Path) -> Option<&Path> {
    match path.parent() {
        Some(p) if p.as_os_str().is_empty() => Some(Path::new(".")),
        other => other,
    }
}

/// fsync a directory, making every pending entry operation inside it (create,
/// rename, remove) durable in one call.
pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

/// Async twin of [`sync_dir`].
pub async fn sync_dir_async(dir: &Path) -> std::io::Result<()> {
    tokio::fs::File::open(dir).await?.sync_all().await
}

/// `create_dir_all`, then sync the parent of every directory it had to create
/// — the whole way up, since a directory only exists for certain once the
/// parent holding its entry is synced. Free once the tree exists.
pub fn create_dir_all_synced(dir: &Path) -> std::io::Result<()> {
    let mut created = Vec::new();
    let mut missing = Some(dir);
    while let Some(d) = missing.filter(|d| !d.exists()) {
        created.push(d);
        missing = parent_dir(d);
    }
    std::fs::create_dir_all(dir)?;
    // Top down: an entry is only durable once the directory holding it is.
    for d in created.iter().rev() {
        if let Some(parent) = parent_dir(d) {
            sync_dir(parent)?;
        }
    }
    Ok(())
}

/// Async twin of [`create_dir_all_synced`].
pub async fn create_dir_all_synced_async(dir: &Path) -> std::io::Result<()> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || create_dir_all_synced(&dir))
        .await
        .map_err(std::io::Error::other)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmp_path_appends_to_the_whole_file_name() {
        assert_eq!(
            tmp_path(Path::new("/a/b/1000_5000.ts")),
            PathBuf::from("/a/b/1000_5000.ts.tmp")
        );
        assert_eq!(
            tmp_path(Path::new("/a/b/settings.json")),
            PathBuf::from("/a/b/settings.json.tmp")
        );
    }

    #[test]
    fn replace_atomic_leaves_no_staging_file_and_keeps_the_old_content_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.json");
        replace_atomic(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        assert!(!tmp_path(&path).exists(), "staging file left behind");

        std::fs::create_dir(tmp_path(&path)).unwrap();
        assert!(replace_atomic(&path, b"second").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
    }

    #[tokio::test]
    async fn replace_atomic_async_matches_its_sync_twin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.json");
        replace_atomic_async(&path, b"first").await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        assert!(!tmp_path(&path).exists());

        std::fs::create_dir(tmp_path(&path)).unwrap();
        assert!(replace_atomic_async(&path, b"second").await.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
    }

    #[tokio::test]
    async fn write_all_synced_async_lands_the_chunks_back_to_back() {
        let dir = tempfile::tempdir().unwrap();
        let chunked = dir.path().join("chunked");
        let pieces: [&[u8]; 3] = [b"one", b"two", b"three"];

        write_all_synced_async(&chunked, &pieces).await.unwrap();
        assert_eq!(std::fs::read(&chunked).unwrap(), b"onetwothree");

        let whole = dir.path().join("whole");
        write_all_synced_async(&whole, &[b"onetwothree"])
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(&whole).unwrap(),
            std::fs::read(&chunked).unwrap()
        );

        let empty = dir.path().join("empty");
        write_all_synced_async(&empty, &[]).await.unwrap();
        assert_eq!(std::fs::read(&empty).unwrap(), b"");
    }

    #[test]
    fn sync_dir_reports_a_directory_it_cannot_sync() {
        let dir = tempfile::tempdir().unwrap();
        sync_dir(dir.path()).unwrap();

        let missing = dir.path().join("gone");
        assert!(sync_dir(&missing).is_err(), "missing directory synced");
    }

    #[tokio::test]
    async fn sync_dir_async_reports_a_directory_it_cannot_sync() {
        let dir = tempfile::tempdir().unwrap();
        sync_dir_async(dir.path()).await.unwrap();
        assert!(sync_dir_async(&dir.path().join("gone")).await.is_err());
    }

    #[test]
    fn parent_dir_reads_a_bare_relative_name_as_the_working_directory() {
        assert!(
            sync_dir(Path::new("")).is_err(),
            "the empty path is openable"
        );
        assert_eq!(parent_dir(Path::new("storage")), Some(Path::new(".")));
        assert_eq!(parent_dir(Path::new("/a/b")), Some(Path::new("/a")));
        assert_eq!(parent_dir(Path::new("/")), None);
    }

    #[test]
    fn create_dir_all_synced_handles_a_bare_relative_path() {
        let root = PathBuf::from(format!("durable-relative-{}", std::process::id()));
        let leaf = root.join("cam1").join("movements");
        let result = create_dir_all_synced(&leaf);
        let created = leaf.is_dir();
        let _ = std::fs::remove_dir_all(&root);
        result.expect("a relative data_dir must not fail on the empty parent path");
        assert!(created);
    }

    #[test]
    fn create_dir_all_synced_creates_the_whole_tree_and_repeats_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let leaf = dir.path().join("data").join("cam1").join("movements");
        create_dir_all_synced(&leaf).unwrap();
        assert!(leaf.is_dir());
        create_dir_all_synced(&leaf).unwrap();
        assert!(leaf.is_dir());
    }

    #[tokio::test]
    async fn create_dir_all_synced_async_reports_a_tree_it_cannot_create() {
        let dir = tempfile::tempdir().unwrap();
        let blocked = dir.path().join("cam1");
        std::fs::write(&blocked, b"not a directory").unwrap();
        assert!(create_dir_all_synced_async(&blocked.join("movements"))
            .await
            .is_err());

        let leaf = dir.path().join("cam2").join("objects");
        create_dir_all_synced_async(&leaf).await.unwrap();
        assert!(leaf.is_dir());
    }
}
