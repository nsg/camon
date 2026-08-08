//! Startup recovery of orphaned temp files left by a crash or power cut.

use std::path::Path;

use super::event_index::{EventType, MAX_FILMSTRIP_FRAMES};
use crate::mpegts;

/// How a salvage reads the bytes of an orphaned `.ts.tmp`.
type TmpScan = dyn Fn(&Path) -> std::io::Result<mpegts::TsScan>;

/// Scan every event directory of every configured camera for `*.tmp` orphans
/// and recover or clean them. Synchronous: runs once at startup, before the
/// warm index scan.
pub fn recover_orphans(data_dir: &Path, camera_ids: &[String]) {
    recover_orphans_with(data_dir, camera_ids, &mpegts::scan_ts_file);
}

fn recover_orphans_with(data_dir: &Path, camera_ids: &[String], scan: &TmpScan) {
    for camera_id in camera_ids {
        for event_type in [
            EventType::Movement,
            EventType::Object,
            EventType::Continuous,
        ] {
            let dir = data_dir.join(camera_id).join(event_type.dir_name());
            recover_dir(&dir, camera_id, scan);
        }
    }
}

fn recover_dir(dir: &Path, camera_id: &str, scan: &TmpScan) {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return, // directory does not exist yet — nothing to recover
    };
    // Collect first: recovery itself creates and renames files in this
    // directory, and mutating while iterating read_dir could surface our own
    // fresh `.json.tmp` staging file as a bogus orphan.
    let tmp_paths: Vec<std::path::PathBuf> = read_dir
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".tmp"))
        })
        .collect();

    for path in tmp_paths {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".ts.tmp") {
            recover_video_tmp(&path, camera_id, scan);
        } else {
            // An interrupted sidecar/thumbnail staging file. Cannot be
            // reconstructed and carries no video — safe to delete.
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    tracing::info!(camera = %camera_id, path = %path.display(),
                        "removed orphaned metadata temp file")
                }
                Err(e) => tracing::warn!(camera = %camera_id, path = %path.display(),
                        error = %e, "failed to remove orphaned metadata temp file"),
            }
        }
    }
}

/// Salvage a single orphaned `{stem}.ts.tmp`.
fn recover_video_tmp(path: &Path, camera_id: &str, scan: &TmpScan) {
    let scanned = match scan(path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(camera = %camera_id, path = %path.display(), error = %e,
                "failed to read orphaned video temp file, leaving it in place");
            return;
        }
    };
    let file_len = file_len(path);
    let valid_len = scanned.valid_len;

    // A recoverable file needs at least one PES packet with a PTS: the writer
    // starts every file with PAT/PMT + a keyframe, so real footage always
    // carries PES timestamps. PSI-only or empty content has nothing decodable.
    let (first_pts_90k, last_pts_90k) = match (scanned.first_pts, scanned.last_pts) {
        (Some(f), Some(l)) => (f, l),
        _ => {
            tracing::warn!(camera = %camera_id, path = %path.display(), bytes = file_len,
                "deleting orphaned video temp file with no decodable content");
            let _ = std::fs::remove_file(path);
            return;
        }
    };

    // Real duration of what survived, from the file content (PES PTS delta,
    // 90 kHz, wrap-aware). The stem in the tmp filename may claim a longer
    // duration than the truncated file actually holds.
    let duration_ms = mpegts::pts_delta_ms(first_pts_90k, last_pts_90k);

    // Start timestamp: keep the epoch-ns value from the tmp filename stem.
    let dir = path.parent().unwrap_or(Path::new("."));
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let orig_stem = file_name.strip_suffix(".ts.tmp").unwrap_or(file_name);
    let first_pts_ns: u64 = orig_stem
        .split_once('_')
        .and_then(|(s, _)| s.parse().ok())
        .unwrap_or_else(|| file_mtime_ns(path));

    let new_stem = format!("{first_pts_ns}_{duration_ms}");
    let final_path = dir.join(format!("{new_stem}.ts"));
    if final_path.exists() {
        // A committed event already owns this stem; the tmp is a stale
        // duplicate of it (or an unrecoverable collision). The committed file
        // wins — recovering over it would clobber good data.
        tracing::warn!(camera = %camera_id, path = %path.display(),
            "orphaned video temp file collides with a committed event, deleting");
        let _ = std::fs::remove_file(path);
        return;
    }

    // Trim in place (no copy), make it durable, then commit via rename.
    if let Err(e) = truncate_synced(path, valid_len) {
        tracing::warn!(camera = %camera_id, path = %path.display(), error = %e,
            "failed to trim orphaned video temp file, leaving it in place");
        return;
    }
    if let Err(e) = std::fs::rename(path, &final_path) {
        tracing::warn!(camera = %camera_id, path = %path.display(), error = %e,
            "failed to finalize recovered video file, leaving it in place");
        return;
    }

    write_recovered_sidecar(dir, orig_stem, &new_stem, camera_id);
    if orig_stem != new_stem {
        adopt_thumbnails(dir, orig_stem, &new_stem);
    }
    // One fsync per salvaged file, covering every entry this recovery just renamed in the
    // directory.
    if let Err(e) = crate::durable::sync_dir(dir) {
        tracing::warn!(camera = %camera_id, path = %dir.display(), error = %e,
            "failed to fsync directory after recovering an event file");
    }

    tracing::warn!(
        camera = %camera_id,
        path = %final_path.display(),
        duration_ms,
        trimmed_bytes = file_len.saturating_sub(valid_len),
        "recovered orphaned event file after unclean shutdown"
    );
}

fn truncate_synced(path: &Path, len: u64) -> std::io::Result<()> {
    let f = std::fs::OpenOptions::new().write(true).open(path)?;
    f.set_len(len)?;
    f.sync_all()?;
    Ok(())
}

/// Write the recovered event's sidecar, adopting any metadata the writer
/// finished before the crash (sidecar and thumbnails are written before the
/// video's commit rename, so they may already exist under the original stem).
fn write_recovered_sidecar(dir: &Path, orig_stem: &str, new_stem: &str, camera_id: &str) {
    let orig_sidecar = dir.join(format!("{orig_stem}.json"));
    let mut meta = std::fs::read_to_string(&orig_sidecar)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_else(|| {
            let mut m = serde_json::Map::new();
            m.insert("detections".to_string(), serde_json::json!([]));
            m
        });
    meta.insert("recovered".to_string(), serde_json::json!(true));

    // Atomic like every other sidecar write; an interruption here just leaves
    // a .json.tmp for the next startup to clean up.
    let final_sidecar = dir.join(format!("{new_stem}.json"));
    let json = serde_json::to_string(&meta).unwrap();
    if let Err(e) = crate::durable::replace_atomic(&final_sidecar, json.as_bytes()) {
        tracing::warn!(camera = %camera_id, path = %final_sidecar.display(), error = %e,
            "failed to write recovered-event sidecar");
        return;
    }
    if orig_stem != new_stem {
        let _ = std::fs::remove_file(&orig_sidecar);
    }
}

/// Move any finished filmstrip thumbnails over to the recovered stem.
fn adopt_thumbnails(dir: &Path, orig_stem: &str, new_stem: &str) {
    for i in 0..MAX_FILMSTRIP_FRAMES {
        let from = dir.join(format!("{orig_stem}_thumb_{i}.jpg"));
        if from.exists() {
            let _ = std::fs::rename(&from, dir.join(format!("{new_stem}_thumb_{i}.jpg")));
        }
    }
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn file_mtime_ns(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpegts::testutil::{null_packet, pes_packet};
    use crate::storage::{EventRef, WarmEventIndex};

    const CAM: &str = "cam";

    fn movements_dir(root: &Path) -> std::path::PathBuf {
        root.join(CAM).join("movements")
    }

    fn video_bytes(first_pts_90k: u64, last_pts_90k: u64) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&pes_packet(0x100, first_pts_90k));
        data.extend_from_slice(&null_packet());
        data.extend_from_slice(&pes_packet(0x100, last_pts_90k));
        data
    }

    fn recover(root: &Path) {
        recover_orphans(root, &[CAM.to_string()]);
    }

    #[test]
    fn a_large_interrupted_recording_is_salvaged_without_being_held_in_memory() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();

        let mut data = video_bytes(0, 2 * 90_000);
        while data.len() < 8 * 1024 * 1024 {
            data.extend_from_slice(&null_packet());
        }
        let valid_len = data.len();
        data.extend_from_slice(&[0x47, 0x11]);
        std::fs::write(dir.join("4000_9000.ts.tmp"), &data).unwrap();

        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = std::sync::Arc::clone(&peak);
        recover_orphans_with(tmp_dir.path(), &[CAM.to_string()], &move |path| {
            let file = std::fs::File::open(path)?;
            crate::mpegts::scan_ts_stream(WatchedReader {
                inner: file,
                peak: std::sync::Arc::clone(&observed),
            })
        });

        let held = peak.load(std::sync::atomic::Ordering::Relaxed);
        assert!(held > 0, "the salvage never read through the scan at all");
        assert!(
            held <= crate::mpegts::SCAN_BUFFER_BYTES,
            "held {held} bytes of an {} byte file at once",
            data.len()
        );

        let final_path = dir.join("4000_2000.ts");
        assert!(final_path.exists());
        assert_eq!(
            std::fs::metadata(&final_path).unwrap().len(),
            valid_len as u64
        );
    }

    struct WatchedReader {
        inner: std::fs::File,
        peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl std::io::Read for WatchedReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.peak
                .fetch_max(buf.len(), std::sync::atomic::Ordering::Relaxed);
            std::io::Read::read(&mut self.inner, buf)
        }
    }

    #[test]
    fn truncated_tmp_round_trips_to_indexed_recovered_event() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();

        let mut data = video_bytes(90_000, 90_000 + 5 * 90_000);
        data.extend_from_slice(&[0x47, 0xDE, 0xAD]); // torn final packet
        let full_len = data.len();
        std::fs::write(dir.join("7777000000_12000.ts.tmp"), &data).unwrap();

        recover(tmp_dir.path());

        let final_path = dir.join("7777000000_5000.ts");
        assert!(final_path.exists());
        assert!(!dir.join("7777000000_12000.ts.tmp").exists());
        assert_eq!(
            std::fs::metadata(&final_path).unwrap().len(),
            (full_len - 3) as u64
        );

        let sidecar = std::fs::read_to_string(dir.join("7777000000_5000.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&sidecar).unwrap();
        assert_eq!(parsed["recovered"], serde_json::json!(true));

        let index = WarmEventIndex::new(&[CAM.to_string()], tmp_dir.path().to_path_buf());
        index.scan();
        let entry = index
            .find_event(CAM, EventRef::new(7_777_000_000, 5000, EventType::Movement))
            .unwrap();
        assert_eq!(entry.duration_ms, 5000);
        assert!(entry.recovered);
        assert!(!entry.continues);
    }

    #[test]
    fn tmp_with_zero_valid_packets_is_deleted() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("1000_5000.ts.tmp"), [0xAB; 300]).unwrap();
        std::fs::write(dir.join("2000_5000.ts.tmp"), []).unwrap();
        std::fs::write(dir.join("3000_5000.ts.tmp"), null_packet()).unwrap();

        recover(tmp_dir.path());

        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
        assert!(left.is_empty(), "leftovers: {left:?}");
    }

    #[test]
    fn orphaned_metadata_tmps_are_deleted() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("1000_5000.json.tmp"), b"{").unwrap();
        std::fs::write(dir.join("1000_5000_thumb_0.jpg.tmp"), b"x").unwrap();
        std::fs::write(dir.join("1000_5000.ts"), b"tsdata").unwrap();
        std::fs::write(dir.join("1000_5000.json"), b"{}").unwrap();

        recover(tmp_dir.path());

        assert!(!dir.join("1000_5000.json.tmp").exists());
        assert!(!dir.join("1000_5000_thumb_0.jpg.tmp").exists());
        assert!(dir.join("1000_5000.ts").exists());
        assert!(dir.join("1000_5000.json").exists());
    }

    #[test]
    fn recovery_adopts_finished_sidecar_and_thumbnails() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();

        let data = video_bytes(0, 3 * 90_000);
        std::fs::write(dir.join("5000_3000.ts.tmp"), &data).unwrap();
        std::fs::write(
            dir.join("5000_3000.json"),
            r#"{"backend":"ollama","model":"m","detections":[{"class":"person","confidence":0.9}],"continues":true}"#,
        )
        .unwrap();
        std::fs::write(dir.join("5000_3000_thumb_0.jpg"), b"jpeg").unwrap();

        recover(tmp_dir.path());

        assert!(dir.join("5000_3000.ts").exists());
        let sidecar = std::fs::read_to_string(dir.join("5000_3000.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&sidecar).unwrap();
        assert_eq!(parsed["recovered"], serde_json::json!(true));
        assert_eq!(parsed["continues"], serde_json::json!(true));
        assert_eq!(
            parsed["detections"][0]["class"],
            serde_json::json!("person")
        );
        assert!(dir.join("5000_3000_thumb_0.jpg").exists());

        let index = WarmEventIndex::new(&[CAM.to_string()], tmp_dir.path().to_path_buf());
        index.scan();
        let entry = index
            .find_event(CAM, EventRef::new(5000, 3000, EventType::Movement))
            .unwrap();
        assert!(entry.recovered);
        assert!(entry.continues);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
    }

    #[test]
    fn stem_change_moves_sidecar_and_thumbnails_to_new_stem() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();

        let data = video_bytes(0, 2 * 90_000);
        std::fs::write(dir.join("6000_9000.ts.tmp"), &data).unwrap();
        std::fs::write(dir.join("6000_9000.json"), r#"{"detections":[]}"#).unwrap();
        std::fs::write(dir.join("6000_9000_thumb_0.jpg"), b"jpeg").unwrap();

        recover(tmp_dir.path());

        assert!(dir.join("6000_2000.ts").exists());
        assert!(dir.join("6000_2000.json").exists());
        assert!(dir.join("6000_2000_thumb_0.jpg").exists());
        assert!(!dir.join("6000_9000.json").exists());
        assert!(!dir.join("6000_9000_thumb_0.jpg").exists());
    }

    #[test]
    fn colliding_tmp_never_clobbers_committed_event() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = movements_dir(tmp_dir.path());
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(dir.join("8000_1000.ts"), b"committed").unwrap();
        let data = video_bytes(0, 90_000); // 1s → stem 8000_1000 collision
        std::fs::write(dir.join("8000_1000.ts.tmp"), &data).unwrap();

        recover(tmp_dir.path());

        assert!(!dir.join("8000_1000.ts.tmp").exists());
        assert_eq!(
            std::fs::read(dir.join("8000_1000.ts")).unwrap(),
            b"committed"
        );
    }
}
