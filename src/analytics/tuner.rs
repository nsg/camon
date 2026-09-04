//! Safe per-cell adaptation for sustained stationary motion.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::analytics::motion_settings::{
    TunerMode, CELL_CONTOUR_AREA_CEILING, MASK_CELLS, MASK_COLS, MASK_ROWS,
};
use crate::config::MotionConfig;
use crate::durable::{create_dir_all_synced, sync_dir, tmp_path, write_synced};
use crate::locks::LockExt;

const BUCKET_SECS: u64 = 60;
/// Half a minute of the roughly one-segment-per-second analyzer stream.
const MIN_BUCKET_SEGMENTS: u32 = 30;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TunerParams {
    pub window_secs: u64,
    pub global_event_cell_fraction: f64,
    pub tighten_bar: f64,
    pub tighten_step: f64,
    pub cell_ceiling: f64,
    pub relax_bar: f64,
    pub relax_dwell_secs: u64,
    pub relax_step: f64,
    pub min_step_interval_secs: u64,
}

impl Default for TunerParams {
    fn default() -> Self {
        Self {
            window_secs: 1_200,
            global_event_cell_fraction: 0.5,
            tighten_bar: 0.60,
            tighten_step: 150.0,
            cell_ceiling: CELL_CONTOUR_AREA_CEILING,
            relax_bar: 0.10,
            relax_dwell_secs: 2_400,
            relax_step: 100.0,
            min_step_interval_secs: 1_200,
        }
    }
}

impl From<&MotionConfig> for TunerParams {
    fn from(config: &MotionConfig) -> Self {
        Self {
            window_secs: config.tuner_window_secs,
            global_event_cell_fraction: config.tuner_global_event_cell_fraction,
            tighten_bar: config.tuner_tighten_bar,
            tighten_step: config.tuner_tighten_step,
            relax_bar: config.tuner_relax_bar,
            relax_dwell_secs: config.tuner_relax_dwell_secs,
            relax_step: config.tuner_relax_step,
            min_step_interval_secs: config.tuner_window_secs,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedCellChange {
    pub wall_unix_ms: u64,
    pub delta: f64,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct CellChange {
    pub cell: usize,
    pub old: f64,
    pub new: f64,
    pub wall: SystemTime,
    pub delta: f64,
    pub reason: String,
}

impl CellChange {
    fn persisted(&self) -> PersistedCellChange {
        PersistedCellChange {
            wall_unix_ms: self
                .wall
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            delta: self.delta,
            reason: self.reason.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TunerSnapshot {
    pub mode: TunerMode,
    pub cols: usize,
    pub rows: usize,
    pub window_secs: u64,
    pub window_full: bool,
    pub global_events_in_window: u64,
    pub base: Vec<f64>,
    pub learned: Vec<f64>,
    pub proposed: Vec<f64>,
    pub effective: Vec<f64>,
    pub trigger_fraction: Vec<f64>,
    pub last_change: Vec<Option<PersistedCellChange>>,
    pub params: TunerParams,
}

impl TunerSnapshot {
    pub fn empty(mode: TunerMode) -> Self {
        Self::empty_with_params(mode, TunerParams::default())
    }

    pub fn empty_with_params(mode: TunerMode, params: TunerParams) -> Self {
        Self {
            mode,
            cols: MASK_COLS,
            rows: MASK_ROWS,
            window_secs: params.window_secs,
            window_full: false,
            global_events_in_window: 0,
            base: vec![0.0; MASK_CELLS],
            learned: vec![0.0; MASK_CELLS],
            proposed: vec![0.0; MASK_CELLS],
            effective: vec![0.0; MASK_CELLS],
            trigger_fraction: vec![0.0; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
            params,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TunerState {
    pub version: u32,
    pub learned: Vec<f64>,
    pub last_change: Vec<Option<PersistedCellChange>>,
}

#[derive(Clone)]
struct Bucket {
    minute: u64,
    segments: u32,
    global_events: u32,
    cell_hits: [u32; MASK_CELLS],
}

impl Bucket {
    fn new(minute: u64) -> Self {
        Self {
            minute,
            segments: 0,
            global_events: 0,
            cell_hits: [0; MASK_CELLS],
        }
    }
}

pub struct MotionTuner {
    params: TunerParams,
    mode: TunerMode,
    learned: [f64; MASK_CELLS],
    proposed: [f64; MASK_CELLS],
    last_step: [Option<Instant>; MASK_CELLS],
    quiet_since: [Option<Instant>; MASK_CELLS],
    last_change: Vec<Option<PersistedCellChange>>,
    started: Option<Instant>,
    buckets: VecDeque<Bucket>,
}

impl MotionTuner {
    pub fn new(params: TunerParams) -> Self {
        Self {
            params,
            mode: TunerMode::Off,
            learned: [0.0; MASK_CELLS],
            proposed: [0.0; MASK_CELLS],
            last_step: [None; MASK_CELLS],
            quiet_since: [None; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
            started: None,
            buckets: VecDeque::new(),
        }
    }

    pub fn set_mode(&mut self, mode: TunerMode) {
        self.mode = mode;
    }

    pub fn mode(&self) -> TunerMode {
        self.mode
    }

    pub fn observe_segment(
        &mut self,
        triggered: bool,
        motion_cells: &[bool; MASK_CELLS],
        now: Instant,
    ) {
        if self.started.is_none() {
            self.started = Some(now);
        }
        self.rotate(now);
        let minute = self.minute_at(now);
        let Some(oldest) = self.buckets.front().map(|bucket| bucket.minute) else {
            return;
        };
        if minute < oldest {
            return;
        }
        let Some(bucket) = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.minute == minute)
        else {
            return;
        };
        bucket.segments = bucket.segments.saturating_add(1);
        if triggered {
            let marked_cells = motion_cells.iter().filter(|&&present| present).count();
            if marked_cells as f64 > self.params.global_event_cell_fraction * MASK_CELLS as f64 {
                bucket.global_events = bucket.global_events.saturating_add(1);
                return;
            }
            for (hit, present) in bucket.cell_hits.iter_mut().zip(motion_cells) {
                if *present {
                    *hit = hit.saturating_add(1);
                }
            }
        }
    }

    pub fn evaluate(&mut self, now: Instant, wall: SystemTime) -> Vec<CellChange> {
        self.rotate(now);
        if self.mode == TunerMode::Off || self.started.is_none() {
            return Vec::new();
        }

        let fractions = self.trigger_fractions();
        let tighten_ready = self.tighten_ready(now);
        let relax_ready = self.elapsed_window_ready(now);
        let step_interval = Duration::from_secs(self.params.min_step_interval_secs);
        let relax_dwell = Duration::from_secs(self.params.relax_dwell_secs);
        let window_minutes = self.params.window_secs / BUCKET_SECS;
        let relax_minutes = self.params.relax_dwell_secs / BUCKET_SECS;
        let mut changes = Vec::new();

        for (cell, &fraction) in fractions.iter().enumerate() {
            let current = match self.mode {
                TunerMode::Auto => self.learned[cell],
                TunerMode::Shadow => self.proposed[cell],
                TunerMode::Off => unreachable!(),
            };
            let step_ready = self.last_step[cell]
                .is_none_or(|last| now.saturating_duration_since(last) >= step_interval);
            let change = if fraction >= self.params.tighten_bar && tighten_ready && step_ready {
                self.quiet_since[cell] = None;
                let target = (current + self.params.tighten_step).min(self.params.cell_ceiling);
                (target > current).then(|| {
                    let percent = (fraction * 100.0).round() as u64;
                    (
                        target,
                        format!(
                            "sustained motion: {percent}% of segments over {window_minutes} min"
                        ),
                    )
                })
            } else if fraction < self.params.relax_bar {
                let quiet_since = *self.quiet_since[cell].get_or_insert(now);
                let quiet_long_enough = now.saturating_duration_since(quiet_since) >= relax_dwell;
                if relax_ready && quiet_long_enough && current > 0.0 && step_ready {
                    let target = (current - self.params.relax_step).max(0.0);
                    let percent = (fraction * 100.0).round() as u64;
                    Some((
                        target,
                        format!("quiet: {percent}% of segments for {relax_minutes} min"),
                    ))
                } else {
                    None
                }
            } else {
                self.quiet_since[cell] = None;
                None
            };

            if let Some((target, reason)) = change {
                match self.mode {
                    TunerMode::Auto => self.learned[cell] = target,
                    TunerMode::Shadow => self.proposed[cell] = target,
                    TunerMode::Off => unreachable!(),
                }
                self.last_step[cell] = Some(now);
                if target < current {
                    self.quiet_since[cell] = Some(now);
                }
                let change = CellChange {
                    cell,
                    old: current,
                    new: target,
                    wall,
                    delta: target - current,
                    reason,
                };
                self.last_change[cell] = Some(change.persisted());
                changes.push(change);
            }
        }
        changes
    }

    pub fn effective_grid(&self, base: &[f64]) -> Vec<f64> {
        if self.mode != TunerMode::Auto {
            return base.to_vec();
        }
        (0..MASK_CELLS)
            .map(|cell| {
                base.get(cell)
                    .copied()
                    .unwrap_or(0.0)
                    .max(self.learned[cell])
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.learned = [0.0; MASK_CELLS];
        self.proposed = [0.0; MASK_CELLS];
        self.last_step = [None; MASK_CELLS];
        self.quiet_since = [None; MASK_CELLS];
        self.last_change.fill(None);
        self.started = None;
        self.buckets.clear();
    }

    pub fn snapshot(&mut self, base: &[f64], now: Instant) -> TunerSnapshot {
        self.rotate(now);
        let base = normalized_grid(base);
        TunerSnapshot {
            mode: self.mode,
            cols: MASK_COLS,
            rows: MASK_ROWS,
            window_secs: self.params.window_secs,
            window_full: self.tighten_ready(now),
            global_events_in_window: self
                .buckets
                .iter()
                .map(|bucket| u64::from(bucket.global_events))
                .sum(),
            base: base.clone(),
            learned: self.learned.to_vec(),
            proposed: self.proposed.to_vec(),
            effective: self.effective_grid(&base),
            trigger_fraction: self.trigger_fractions().to_vec(),
            last_change: self.last_change.clone(),
            params: self.params.clone(),
        }
    }

    pub fn load_state(&mut self, state: &TunerState) {
        if state.version != 2 {
            return;
        }
        for (cell, value) in self.learned.iter_mut().enumerate() {
            let loaded = state.learned.get(cell).copied().unwrap_or(0.0);
            *value = if loaded.is_finite() {
                loaded.clamp(0.0, self.params.cell_ceiling)
            } else {
                0.0
            };
        }
        self.last_change = (0..MASK_CELLS)
            .map(|cell| state.last_change.get(cell).cloned().unwrap_or(None))
            .collect();
    }

    pub fn state(&self) -> TunerState {
        TunerState {
            version: 2,
            learned: self.learned.to_vec(),
            last_change: self.last_change.clone(),
        }
    }

    fn rotate(&mut self, now: Instant) {
        if self.started.is_none() {
            return;
        }

        let minute = self.minute_at(now);
        let window_minutes = self.window_minutes();
        let newest = self.buckets.back().map(|bucket| bucket.minute);
        if newest.is_none_or(|newest| minute > newest) {
            let oldest_live = minute.saturating_sub(window_minutes);
            let first_new = match newest {
                Some(newest) if newest >= oldest_live => newest.saturating_add(1),
                _ => {
                    self.buckets.clear();
                    oldest_live
                }
            };
            for next in first_new..=minute {
                self.buckets.push_back(Bucket::new(next));
            }
        }

        let current_minute = self
            .buckets
            .back()
            .map_or(minute, |bucket| bucket.minute.max(minute));
        let oldest_live = current_minute.saturating_sub(window_minutes);
        while self
            .buckets
            .front()
            .is_some_and(|bucket| bucket.minute < oldest_live)
        {
            self.buckets.pop_front();
        }
    }

    fn minute_at(&self, now: Instant) -> u64 {
        self.started
            .map(|started| now.saturating_duration_since(started).as_secs() / BUCKET_SECS)
            .unwrap_or(0)
    }

    fn window_minutes(&self) -> u64 {
        (self.params.window_secs / BUCKET_SECS).max(1)
    }

    fn elapsed_window_ready(&self, now: Instant) -> bool {
        self.started.is_some_and(|started| {
            now.saturating_duration_since(started).as_secs() >= self.params.window_secs
        })
    }

    fn tighten_ready(&self, now: Instant) -> bool {
        if !self.elapsed_window_ready(now) {
            return false;
        }

        let current_minute = self.minute_at(now);
        let window_minutes = self.window_minutes();
        if current_minute < window_minutes {
            return false;
        }
        let first_complete = current_minute - window_minutes;
        let complete_buckets_observed = (first_complete..current_minute).all(|minute| {
            self.buckets
                .iter()
                .find(|bucket| bucket.minute == minute)
                .is_some_and(|bucket| bucket.segments >= MIN_BUCKET_SEGMENTS)
        });
        if !complete_buckets_observed {
            return false;
        }

        let segments: u64 = self
            .buckets
            .iter()
            .map(|bucket| u64::from(bucket.segments))
            .sum();
        segments >= self.params.window_secs / 2
    }

    fn trigger_fractions(&self) -> [f64; MASK_CELLS] {
        let segments: u64 = self
            .buckets
            .iter()
            .map(|bucket| u64::from(bucket.segments))
            .sum();
        if segments == 0 {
            return [0.0; MASK_CELLS];
        }
        std::array::from_fn(|cell| {
            let hits: u64 = self
                .buckets
                .iter()
                .map(|bucket| u64::from(bucket.cell_hits[cell]))
                .sum();
            hits as f64 / segments as f64
        })
    }
}

fn normalized_grid(base: &[f64]) -> Vec<f64> {
    (0..MASK_CELLS)
        .map(|cell| base.get(cell).copied().unwrap_or(0.0))
        .collect()
}

pub fn tuner_state_path(data_dir: &Path, camera_id: &str) -> std::path::PathBuf {
    data_dir.join(camera_id).join("motion_tuner.json")
}

pub fn load_tuner_state(path: &Path) -> std::io::Result<Option<TunerState>> {
    let data = match std::fs::read_to_string(path) {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let state = serde_json::from_str::<TunerState>(&data).map_err(std::io::Error::other)?;
    if state.version != 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported tuner state version {}", state.version),
        ));
    }
    Ok(Some(state))
}

pub fn save_tuner_state(path: &Path, state: &TunerState) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    create_dir_all_synced(dir)?;
    let json = serde_json::to_string_pretty(state).map_err(std::io::Error::other)?;
    let tmp = tmp_path(path);
    if let Err(error) =
        write_synced(&tmp, json.as_bytes()).and_then(|()| std::fs::rename(&tmp, path))
    {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    sync_dir(dir)
}

struct TunerSlot {
    snapshot: Option<TunerSnapshot>,
    reset_requested: bool,
}

#[derive(Clone)]
pub struct TunerStore {
    cameras: Arc<HashMap<String, RwLock<TunerSlot>>>,
    params: TunerParams,
}

impl TunerStore {
    pub fn new(camera_ids: &[String]) -> Self {
        Self::with_params(camera_ids, TunerParams::default())
    }

    pub fn with_params(camera_ids: &[String], params: TunerParams) -> Self {
        Self {
            cameras: Arc::new(
                camera_ids
                    .iter()
                    .map(|id| {
                        (
                            id.clone(),
                            RwLock::new(TunerSlot {
                                snapshot: None,
                                reset_requested: false,
                            }),
                        )
                    })
                    .collect(),
            ),
            params,
        }
    }

    pub fn params(&self) -> TunerParams {
        self.params.clone()
    }

    pub fn publish(&self, camera: &str, snapshot: TunerSnapshot) {
        if let Some(slot) = self.cameras.get(camera) {
            slot.write_recover().snapshot = Some(snapshot);
        }
    }

    pub fn get(&self, camera: &str) -> Option<TunerSnapshot> {
        self.cameras
            .get(camera)
            .and_then(|slot| slot.read_recover().snapshot.clone())
    }

    pub fn request_reset(&self, camera: &str) -> bool {
        let Some(slot) = self.cameras.get(camera) else {
            return false;
        };
        slot.write_recover().reset_requested = true;
        true
    }

    pub fn take_reset(&self, camera: &str) -> bool {
        let Some(slot) = self.cameras.get(camera) else {
            return false;
        };
        let mut slot = slot.write_recover();
        std::mem::take(&mut slot.reset_requested)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn params() -> TunerParams {
        TunerParams {
            window_secs: 120,
            global_event_cell_fraction: 0.5,
            tighten_bar: 0.6,
            tighten_step: 150.0,
            cell_ceiling: 300.0,
            relax_bar: 0.1,
            relax_dwell_secs: 120,
            relax_step: 100.0,
            min_step_interval_secs: 120,
        }
    }

    #[test]
    fn global_event_counts_as_a_segment_without_cell_hits() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        let mut cells = [false; MASK_CELLS];
        cells[..120].fill(true);

        tuner.observe_segment(true, &cells, start);

        assert_eq!(tuner.buckets.front().unwrap().segments, 1);
        assert!(tuner
            .buckets
            .front()
            .unwrap()
            .cell_hits
            .iter()
            .all(|&hits| hits == 0));
        let snapshot = tuner.snapshot(&[], start);
        assert_eq!(snapshot.global_events_in_window, 1);
        assert!(snapshot
            .trigger_fraction
            .iter()
            .all(|&fraction| fraction == 0.0));
    }

    #[test]
    fn local_event_credits_its_marked_cells() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        let mut cells = [false; MASK_CELLS];
        cells[..40].fill(true);

        tuner.observe_segment(true, &cells, start);

        let snapshot = tuner.snapshot(&[], start);
        assert_eq!(snapshot.global_events_in_window, 0);
        assert!(snapshot.trigger_fraction[..40]
            .iter()
            .all(|&fraction| fraction == 1.0));
        assert!(snapshot.trigger_fraction[40..]
            .iter()
            .all(|&fraction| fraction == 0.0));
    }

    #[test]
    fn fraction_one_never_excludes_an_event() {
        let start = Instant::now();
        let mut configured = params();
        configured.global_event_cell_fraction = 1.0;
        let mut tuner = MotionTuner::new(configured);

        tuner.observe_segment(true, &[true; MASK_CELLS], start);

        let snapshot = tuner.snapshot(&[], start);
        assert_eq!(snapshot.global_events_in_window, 0);
        assert!(snapshot
            .trigger_fraction
            .iter()
            .all(|&fraction| fraction == 1.0));
    }

    fn observe(
        tuner: &mut MotionTuner,
        start: Instant,
        seconds: std::ops::RangeInclusive<u64>,
        triggered: bool,
        cell: usize,
    ) {
        for second in seconds {
            let mut cells = [false; MASK_CELLS];
            cells[cell] = true;
            tuner.observe_segment(triggered, &cells, start + Duration::from_secs(second));
        }
    }

    #[test]
    fn sustained_motion_tightens_with_rate_limit_and_ceiling() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=120, true, 3);

        assert_eq!(
            tuner
                .evaluate(start + Duration::from_secs(120), SystemTime::now())
                .len(),
            1
        );
        assert_eq!(tuner.state().learned[3], 150.0);
        assert!(tuner
            .evaluate(start + Duration::from_secs(121), SystemTime::now())
            .is_empty());
        observe(&mut tuner, start, 121..=240, true, 3);
        assert_eq!(
            tuner
                .evaluate(start + Duration::from_secs(240), SystemTime::now())
                .len(),
            1
        );
        assert_eq!(tuner.state().learned[3], 300.0);
        assert!(tuner
            .evaluate(start + Duration::from_secs(360), SystemTime::now())
            .is_empty());
        assert_eq!(tuner.state().learned[3], 300.0);
    }

    #[test]
    fn shadow_only_changes_proposed_and_never_effective() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Shadow);
        observe(&mut tuner, start, 0..=120, true, 2);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());
        let base = vec![500.0; MASK_CELLS];
        let snapshot = tuner.snapshot(&base, start + Duration::from_secs(120));
        assert_eq!(snapshot.proposed[2], 150.0);
        assert_eq!(snapshot.learned[2], 0.0);
        assert_eq!(snapshot.effective, base);
    }

    #[test]
    fn transient_burst_cannot_tighten() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(TunerParams::default());
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=29, true, 1);
        observe(&mut tuner, start, 30..=1_200, false, 1);
        assert!(tuner
            .evaluate(start + Duration::from_secs(1_200), SystemTime::now())
            .is_empty());
        assert_eq!(tuner.state().learned[1], 0.0);
    }

    #[test]
    fn gap_after_full_window_blocks_tightening() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=119, false, 5);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());
        assert!(
            tuner
                .snapshot(&[], start + Duration::from_secs(120))
                .window_full
        );

        observe(&mut tuner, start, 240..=242, true, 5);
        assert!(tuner
            .evaluate(start + Duration::from_secs(242), SystemTime::now())
            .is_empty());
        let snapshot = tuner.snapshot(&[], start + Duration::from_secs(242));
        assert_eq!(snapshot.trigger_fraction[5], 1.0);
        assert!(!snapshot.window_full);
        assert_eq!(snapshot.learned[5], 0.0);
    }

    #[test]
    fn sparse_window_blocks_tightening() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        tuner.observe_segment(true, &[true; MASK_CELLS], start);
        observe(&mut tuner, start, 120..=128, true, 6);

        assert!(tuner
            .evaluate(start + Duration::from_secs(128), SystemTime::now())
            .is_empty());
        assert!(
            !tuner
                .snapshot(&[], start + Duration::from_secs(128))
                .window_full
        );
        assert_eq!(tuner.state().learned[6], 0.0);
    }

    #[test]
    fn tightening_resumes_after_fresh_gap_free_window() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=119, false, 7);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());

        observe(&mut tuner, start, 240..=359, true, 7);
        let changes = tuner.evaluate(start + Duration::from_secs(360), SystemTime::now());

        assert!(changes.iter().any(|change| change.cell == 7));
        assert!(
            tuner
                .snapshot(&[], start + Duration::from_secs(360))
                .window_full
        );
        assert_eq!(tuner.state().learned[7], 150.0);
    }

    #[test]
    fn boundary_does_not_evict_before_evaluate() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=59, false, 8);
        observe(&mut tuner, start, 60..=119, true, 8);

        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());
        let snapshot = tuner.snapshot(&[], start + Duration::from_secs(120));
        assert_eq!(snapshot.trigger_fraction[8], 0.5);
        assert!(snapshot.window_full);
    }

    #[test]
    fn backdated_observation_lands_in_its_own_minute() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        let cells = [false; MASK_CELLS];
        tuner.observe_segment(false, &cells, start);
        tuner.observe_segment(false, &cells, start + Duration::from_secs(130));
        tuner.observe_segment(false, &cells, start + Duration::from_secs(70));

        assert_eq!(
            tuner
                .buckets
                .iter()
                .find(|bucket| bucket.minute == 1)
                .unwrap()
                .segments,
            1
        );
        assert_eq!(
            tuner
                .buckets
                .iter()
                .find(|bucket| bucket.minute == 2)
                .unwrap()
                .segments,
            1
        );

        tuner.evaluate(start + Duration::from_secs(190), SystemTime::now());
        let before: u32 = tuner.buckets.iter().map(|bucket| bucket.segments).sum();
        tuner.observe_segment(false, &cells, start + Duration::from_secs(30));
        let after: u32 = tuner.buckets.iter().map(|bucket| bucket.segments).sum();
        assert_eq!(before, after);
        assert!(tuner.buckets.iter().all(|bucket| bucket.minute >= 1));
    }

    #[test]
    fn relax_is_gap_tolerant() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        let mut learned = vec![0.0; MASK_CELLS];
        learned[9] = 100.0;
        tuner.load_state(&TunerState {
            version: 2,
            learned,
            last_change: vec![None; MASK_CELLS],
        });
        tuner.observe_segment(false, &[false; MASK_CELLS], start);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());

        let changes = tuner.evaluate(start + Duration::from_secs(240), SystemTime::now());
        assert!(changes.iter().any(|change| change.cell == 9));
        assert_eq!(tuner.state().learned[9], 0.0);
        assert!(
            !tuner
                .snapshot(&[], start + Duration::from_secs(240))
                .window_full
        );
    }

    #[test]
    fn quiet_relaxes_to_zero_and_activity_resets_dwell() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        tuner.load_state(&TunerState {
            version: 2,
            learned: vec![200.0; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
        });
        observe(&mut tuner, start, 0..=120, false, 0);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());
        assert!(tuner
            .evaluate(start + Duration::from_secs(239), SystemTime::now())
            .is_empty());

        observe(&mut tuner, start, 121..=150, true, 0);
        observe(&mut tuner, start, 151..=240, false, 0);
        tuner.evaluate(start + Duration::from_secs(240), SystemTime::now());
        observe(&mut tuner, start, 241..=360, false, 0);
        tuner.evaluate(start + Duration::from_secs(360), SystemTime::now());
        let changes = tuner.evaluate(start + Duration::from_secs(480), SystemTime::now());
        assert!(changes.iter().any(|change| change.cell == 0));
        assert_eq!(tuner.state().learned[0], 100.0);
        tuner.evaluate(start + Duration::from_secs(600), SystemTime::now());
        assert_eq!(tuner.state().learned[0], 0.0);
    }

    #[test]
    fn off_collects_stats_without_changes() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        observe(&mut tuner, start, 0..=120, true, 4);
        assert!(tuner
            .evaluate(start + Duration::from_secs(120), SystemTime::now())
            .is_empty());
        let snapshot = tuner.snapshot(&[], start + Duration::from_secs(120));
        assert!(snapshot.trigger_fraction[4] > 0.9);
        assert!(snapshot.learned.iter().all(|&value| value == 0.0));
        assert!(snapshot.proposed.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn evaluations_before_first_observation_do_not_fill_the_window() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        assert!(tuner
            .evaluate(start + Duration::from_secs(10_000), SystemTime::now())
            .is_empty());
        assert!(
            !tuner
                .snapshot(&[], start + Duration::from_secs(10_000))
                .window_full
        );
    }

    #[test]
    fn effective_grid_uses_max_only_in_auto() {
        let mut tuner = MotionTuner::new(params());
        tuner.load_state(&TunerState {
            version: 2,
            learned: {
                let mut grid = vec![0.0; MASK_CELLS];
                grid[0] = 300.0;
                grid[1] = 300.0;
                grid
            },
            last_change: vec![None; MASK_CELLS],
        });
        let mut base = vec![0.0; MASK_CELLS];
        base[1] = 500.0;
        for mode in [TunerMode::Off, TunerMode::Shadow] {
            tuner.set_mode(mode);
            assert_eq!(tuner.effective_grid(&base), base);
        }
        tuner.set_mode(TunerMode::Auto);
        let effective = tuner.effective_grid(&base);
        assert_eq!(&effective[..3], &[300.0, 500.0, 0.0]);
    }

    #[test]
    fn reset_clears_values_and_stats() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Shadow);
        observe(&mut tuner, start, 0..=120, true, 0);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());
        tuner.reset();
        let snapshot = tuner.snapshot(&[], start + Duration::from_secs(120));
        assert!(!snapshot.window_full);
        assert!(snapshot.learned.iter().all(|&value| value == 0.0));
        assert!(snapshot.proposed.iter().all(|&value| value == 0.0));
        assert!(snapshot.trigger_fraction.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn state_json_roundtrip_preserves_learned_and_changes() {
        let state = TunerState {
            version: 2,
            learned: vec![123.0; MASK_CELLS],
            last_change: vec![
                Some(PersistedCellChange {
                    wall_unix_ms: 42,
                    delta: 123.0,
                    reason: "test".to_string(),
                });
                MASK_CELLS
            ],
        };
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(serde_json::from_str::<TunerState>(&json).unwrap(), state);
    }

    #[test]
    fn state_file_uses_the_v2_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = tuner_state_path(dir.path(), "camera");
        let state = TunerState {
            version: 2,
            learned: vec![321.0; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
        };

        save_tuner_state(&path, &state).unwrap();
        assert_eq!(load_tuner_state(&path).unwrap(), Some(state));
        assert!(!tmp_path(&path).exists());
    }
}
