//! Safe per-cell adaptation for sustained stationary motion.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::analytics::motion_settings::{
    MotionSettings, TunerMode, CELL_CONTOUR_AREA_CEILING, MASK_CELLS, MASK_COLS, MASK_ROWS,
};
use crate::config::MotionConfig;
use crate::durable::{create_dir_all_synced, sync_dir, tmp_path, write_synced};
use crate::locks::LockExt;

const BUCKET_SECS: u64 = 60;
const MIN_COVERAGE_FRACTION: f64 = 0.9;
const MAX_MISSING_CADENCES: u32 = 3;

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
            tighten_bar: 0.05,
            tighten_step: 150.0,
            cell_ceiling: CELL_CONTOUR_AREA_CEILING,
            relax_bar: 0.01,
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
            min_step_interval_secs: config.tuner_min_step_interval_secs,
            ..Self::default()
        }
    }
}

impl TunerParams {
    pub fn from_settings(settings: &MotionSettings, global_event_cell_fraction: f64) -> Self {
        Self {
            window_secs: settings.tuner_window_secs,
            global_event_cell_fraction,
            tighten_bar: settings.tuner_tighten_bar,
            tighten_step: settings.tuner_tighten_step,
            cell_ceiling: CELL_CONTOUR_AREA_CEILING,
            relax_bar: settings.tuner_relax_bar,
            relax_dwell_secs: settings.tuner_relax_dwell_secs,
            relax_step: settings.tuner_relax_step,
            min_step_interval_secs: settings.tuner_min_step_interval_secs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CellAdaptationStatus {
    Off,
    InsufficientCoverage,
    BelowThreshold,
    Cooldown,
    Ceiling,
    Ready,
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
    pub adaptation_status: Vec<CellAdaptationStatus>,
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
            adaptation_status: vec![CellAdaptationStatus::InsufficientCoverage; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
            params,
        }
    }

    pub fn empty_for_settings(settings: &MotionSettings, global_event_cell_fraction: f64) -> Self {
        let params = TunerParams::from_settings(settings, global_event_cell_fraction);
        let mut snapshot = Self::empty_with_params(settings.tuner_mode, params);
        snapshot.base = (0..MASK_CELLS)
            .map(|cell| {
                manual_baseline(
                    settings.min_contour_area,
                    &settings.min_contour_area_grid,
                    cell,
                )
            })
            .collect();
        snapshot.effective.clone_from(&snapshot.base);
        if settings.tuner_mode == TunerMode::Off {
            snapshot.adaptation_status.fill(CellAdaptationStatus::Off);
        }
        snapshot
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

#[derive(Clone, Copy)]
struct Observation {
    at: Instant,
    expected_duration: Duration,
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
    observations: VecDeque<Observation>,
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
            observations: VecDeque::new(),
        }
    }

    pub fn set_mode(&mut self, mode: TunerMode) {
        if self.mode != mode {
            self.last_step = [None; MASK_CELLS];
            self.quiet_since = [None; MASK_CELLS];
        }
        self.mode = mode;
    }

    pub fn mode(&self) -> TunerMode {
        self.mode
    }

    pub fn set_params(&mut self, params: TunerParams) {
        if self.params != params {
            self.last_step = [None; MASK_CELLS];
            self.quiet_since = [None; MASK_CELLS];
            self.started = None;
            self.buckets.clear();
            self.observations.clear();
        }
        self.params = params;
    }

    pub fn observe_segment(
        &mut self,
        triggered: bool,
        motion_cells: &[bool; MASK_CELLS],
        now: Instant,
    ) {
        self.observe_segment_with_duration(triggered, motion_cells, Duration::from_secs(1), now);
    }

    pub fn observe_segment_with_duration(
        &mut self,
        triggered: bool,
        motion_cells: &[bool; MASK_CELLS],
        expected_duration: Duration,
        now: Instant,
    ) {
        if self.started.is_none() {
            self.started = Some(now);
        }
        self.rotate(now);
        self.record_observation(now, expected_duration);
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
        self.evaluate_with_baseline(0.0, &[], now, wall)
    }

    pub fn evaluate_with_baseline(
        &mut self,
        min_contour_area: f64,
        manual_grid: &[f64],
        now: Instant,
        wall: SystemTime,
    ) -> Vec<CellChange> {
        self.rotate(now);
        if self.mode == TunerMode::Off || self.started.is_none() {
            return Vec::new();
        }

        let fractions = self.trigger_fractions();
        let coverage_ready = self.coverage_ready(now);
        let step_interval = Duration::from_secs(self.params.min_step_interval_secs);
        let relax_dwell = Duration::from_secs(self.params.relax_dwell_secs);
        let window_minutes = self.params.window_secs / BUCKET_SECS;
        let relax_minutes = self.params.relax_dwell_secs / BUCKET_SECS;
        let mut changes = Vec::new();

        for (cell, &fraction) in fractions.iter().enumerate() {
            let baseline = manual_baseline(min_contour_area, manual_grid, cell);
            let stored = match self.mode {
                TunerMode::Auto => self.learned[cell],
                TunerMode::Shadow => self.proposed[cell],
                TunerMode::Off => unreachable!(),
            };
            let current = stored.min(self.params.cell_ceiling).max(baseline);
            let step_ready = self.last_step[cell]
                .is_none_or(|last| now.saturating_duration_since(last) >= step_interval);
            let change = if fraction >= self.params.tighten_bar && coverage_ready && step_ready {
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
                if !coverage_ready {
                    self.quiet_since[cell] = None;
                    continue;
                }
                let quiet_since = *self.quiet_since[cell].get_or_insert(now);
                let quiet_long_enough = now.saturating_duration_since(quiet_since) >= relax_dwell;
                if quiet_long_enough && current > baseline && step_ready {
                    let target = (current - self.params.relax_step).max(baseline);
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
                let stored_target = if target <= baseline { 0.0 } else { target };
                match self.mode {
                    TunerMode::Auto => self.learned[cell] = stored_target,
                    TunerMode::Shadow => self.proposed[cell] = stored_target,
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

    pub fn effective_grid(&self, manual_grid: &[f64]) -> Vec<f64> {
        self.effective_grid_from_baseline(0.0, manual_grid)
    }

    pub fn effective_grid_from_baseline(
        &self,
        min_contour_area: f64,
        manual_grid: &[f64],
    ) -> Vec<f64> {
        if self.mode != TunerMode::Auto {
            return manual_grid.to_vec();
        }
        (0..MASK_CELLS)
            .map(|cell| {
                manual_baseline(min_contour_area, manual_grid, cell)
                    .max(self.learned[cell].min(self.params.cell_ceiling))
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
        self.observations.clear();
    }

    pub fn snapshot(&mut self, manual_grid: &[f64], now: Instant) -> TunerSnapshot {
        self.snapshot_with_baseline(0.0, manual_grid, now)
    }

    pub fn snapshot_with_baseline(
        &mut self,
        min_contour_area: f64,
        manual_grid: &[f64],
        now: Instant,
    ) -> TunerSnapshot {
        self.rotate(now);
        let base: Vec<f64> = (0..MASK_CELLS)
            .map(|cell| manual_baseline(min_contour_area, manual_grid, cell))
            .collect();
        let effective = if self.mode == TunerMode::Auto {
            self.effective_grid_from_baseline(min_contour_area, manual_grid)
        } else {
            base.clone()
        };
        TunerSnapshot {
            mode: self.mode,
            cols: MASK_COLS,
            rows: MASK_ROWS,
            window_secs: self.params.window_secs,
            window_full: self.coverage_ready(now),
            global_events_in_window: self
                .buckets
                .iter()
                .map(|bucket| u64::from(bucket.global_events))
                .sum(),
            base: base.clone(),
            learned: self.learned.to_vec(),
            proposed: self.proposed.to_vec(),
            effective,
            trigger_fraction: self.trigger_fractions().to_vec(),
            adaptation_status: self.adaptation_statuses(&base, now),
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
        let oldest_observation = now
            .checked_sub(Duration::from_secs(self.params.window_secs))
            .unwrap_or(now);
        while self
            .observations
            .front()
            .is_some_and(|sample| sample.at < oldest_observation)
        {
            self.observations.pop_front();
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

    fn coverage_ready(&self, now: Instant) -> bool {
        if !self.elapsed_window_ready(now) {
            return false;
        }

        let window_start = now
            .checked_sub(Duration::from_secs(self.params.window_secs))
            .unwrap_or(now);
        let samples: Vec<_> = self
            .observations
            .iter()
            .filter(|sample| sample.at >= window_start && sample.at < now)
            .copied()
            .collect();
        let covered_secs: f64 = samples
            .iter()
            .map(|sample| sample.expected_duration.as_secs_f64())
            .sum();
        if covered_secs < self.params.window_secs as f64 * MIN_COVERAGE_FRACTION {
            return false;
        }

        let mut previous_at = window_start;
        let mut previous_cadence = samples
            .first()
            .map_or(Duration::ZERO, |sample| sample.expected_duration);
        for sample in &samples {
            let cadence = previous_cadence.max(sample.expected_duration);
            if sample.at.saturating_duration_since(previous_at)
                > cadence.saturating_mul(MAX_MISSING_CADENCES)
            {
                return false;
            }
            previous_at = sample.at;
            previous_cadence = sample.expected_duration;
        }
        now.saturating_duration_since(previous_at)
            <= previous_cadence.saturating_mul(MAX_MISSING_CADENCES)
    }

    fn record_observation(&mut self, at: Instant, expected_duration: Duration) {
        let expected_duration = expected_duration.max(Duration::from_millis(1));
        let sample = Observation {
            at,
            expected_duration,
        };
        let insert_at = self
            .observations
            .iter()
            .position(|existing| existing.at > at)
            .unwrap_or(self.observations.len());
        self.observations.insert(insert_at, sample);
    }

    fn adaptation_statuses(&self, base: &[f64], now: Instant) -> Vec<CellAdaptationStatus> {
        let coverage_ready = self.coverage_ready(now);
        let fractions = self.trigger_fractions();
        let step_interval = Duration::from_secs(self.params.min_step_interval_secs);
        let relax_dwell = Duration::from_secs(self.params.relax_dwell_secs);
        (0..MASK_CELLS)
            .map(|cell| {
                if self.mode == TunerMode::Off {
                    return CellAdaptationStatus::Off;
                }
                if !coverage_ready {
                    return CellAdaptationStatus::InsufficientCoverage;
                }
                let stored = match self.mode {
                    TunerMode::Shadow => self.proposed[cell],
                    TunerMode::Auto => self.learned[cell],
                    TunerMode::Off => unreachable!(),
                };
                let current = stored.min(self.params.cell_ceiling).max(base[cell]);
                let step_ready = self.last_step[cell]
                    .is_none_or(|last| now.saturating_duration_since(last) >= step_interval);
                if fractions[cell] >= self.params.tighten_bar {
                    if current >= self.params.cell_ceiling {
                        return CellAdaptationStatus::Ceiling;
                    }
                    if !step_ready {
                        return CellAdaptationStatus::Cooldown;
                    }
                    return CellAdaptationStatus::Ready;
                }
                if fractions[cell] < self.params.relax_bar && current > base[cell] {
                    let dwell_ready = self.quiet_since[cell].is_some_and(|quiet_since| {
                        now.saturating_duration_since(quiet_since) >= relax_dwell
                    });
                    return if step_ready && dwell_ready {
                        CellAdaptationStatus::Ready
                    } else {
                        CellAdaptationStatus::Cooldown
                    };
                }
                CellAdaptationStatus::BelowThreshold
            })
            .collect()
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

fn manual_baseline(min_contour_area: f64, manual_grid: &[f64], cell: usize) -> f64 {
    manual_grid
        .get(cell)
        .copied()
        .filter(|value| *value > 0.0)
        .unwrap_or(min_contour_area)
        .max(min_contour_area)
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
    fn first_step_starts_above_camera_and_manual_cell_baselines() {
        let start = Instant::now();
        let mut configured = params();
        configured.cell_ceiling = 1_000.0;
        let mut tuner = MotionTuner::new(configured);
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=120, true, 3);

        let first = tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(120),
            SystemTime::now(),
        );
        assert_eq!((first[0].old, first[0].new), (200.0, 350.0));

        tuner.reset();
        tuner.set_mode(TunerMode::Auto);
        let mut manual = vec![0.0; MASK_CELLS];
        manual[3] = 500.0;
        observe(&mut tuner, start, 0..=120, true, 3);
        let first = tuner.evaluate_with_baseline(
            200.0,
            &manual,
            start + Duration::from_secs(120),
            SystemTime::now(),
        );
        assert_eq!((first[0].old, first[0].new), (500.0, 650.0));
    }

    #[test]
    fn automatic_ceiling_never_lowers_a_manual_baseline() {
        let mut configured = params();
        configured.cell_ceiling = 300.0;
        let mut tuner = MotionTuner::new(configured);
        tuner.set_mode(TunerMode::Auto);
        tuner.load_state(&TunerState {
            version: 2,
            learned: vec![300.0; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
        });
        let mut manual = vec![0.0; MASK_CELLS];
        manual[4] = 500.0;

        let effective = tuner.effective_grid_from_baseline(200.0, &manual);
        assert_eq!(effective[4], 500.0);
        assert_eq!(effective[5], 300.0);
    }

    #[test]
    fn cadence_coverage_accepts_missing_samples_and_non_one_second_segments() {
        let start = Instant::now();
        for cadence_secs in [1, 2] {
            let mut tuner = MotionTuner::new(params());
            tuner.set_mode(TunerMode::Auto);
            for second in (0..120).step_by(cadence_secs as usize) {
                if cadence_secs == 1 && second % 20 == 10 {
                    continue;
                }
                let mut cells = [false; MASK_CELLS];
                cells[6] = true;
                tuner.observe_segment_with_duration(
                    true,
                    &cells,
                    Duration::from_secs(cadence_secs),
                    start + Duration::from_secs(second),
                );
            }
            assert!(
                tuner
                    .snapshot(&[], start + Duration::from_secs(120))
                    .window_full,
                "{cadence_secs}s cadence"
            );
        }
    }

    #[test]
    fn cadence_coverage_rejects_a_long_gap_even_with_enough_total_samples() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        for second in 0..120 {
            if (55..=65).contains(&second) {
                continue;
            }
            tuner.observe_segment(
                false,
                &[false; MASK_CELLS],
                start + Duration::from_secs(second),
            );
        }

        let snapshot = tuner.snapshot(&[], start + Duration::from_secs(120));
        assert!(!snapshot.window_full);
        assert_eq!(
            snapshot.adaptation_status[0],
            CellAdaptationStatus::InsufficientCoverage
        );
    }

    #[test]
    fn shadow_only_changes_proposed_and_never_effective() {
        let start = Instant::now();
        let mut configured = params();
        configured.cell_ceiling = 1_000.0;
        let mut tuner = MotionTuner::new(configured);
        tuner.set_mode(TunerMode::Shadow);
        observe(&mut tuner, start, 0..=120, true, 2);
        tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(120),
            SystemTime::now(),
        );
        let snapshot = tuner.snapshot_with_baseline(200.0, &[], start + Duration::from_secs(120));
        assert_eq!(snapshot.proposed[2], 350.0);
        assert_eq!(snapshot.learned[2], 0.0);
        assert!(snapshot.effective.iter().all(|&value| value == 200.0));
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
    fn a_gap_cannot_be_mistaken_for_quiet_relaxation() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        let mut learned = vec![0.0; MASK_CELLS];
        learned[9] = 400.0;
        tuner.load_state(&TunerState {
            version: 2,
            learned,
            last_change: vec![None; MASK_CELLS],
        });
        tuner.observe_segment(false, &[false; MASK_CELLS], start);
        tuner.evaluate(start + Duration::from_secs(120), SystemTime::now());

        let changes = tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(240),
            SystemTime::now(),
        );
        assert!(changes.is_empty());
        assert_eq!(tuner.state().learned[9], 300.0);
        assert!(
            !tuner
                .snapshot(&[], start + Duration::from_secs(240))
                .window_full
        );
    }

    #[test]
    fn quiet_relaxes_to_the_manual_baseline_and_activity_resets_dwell() {
        let start = Instant::now();
        let mut configured = params();
        configured.cell_ceiling = 600.0;
        let mut tuner = MotionTuner::new(configured);
        tuner.set_mode(TunerMode::Auto);
        tuner.load_state(&TunerState {
            version: 2,
            learned: vec![400.0; MASK_CELLS],
            last_change: vec![None; MASK_CELLS],
        });
        observe(&mut tuner, start, 0..=120, false, 0);
        tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(120),
            SystemTime::now(),
        );
        assert!(tuner
            .evaluate_with_baseline(
                200.0,
                &[],
                start + Duration::from_secs(239),
                SystemTime::now(),
            )
            .is_empty());

        observe(&mut tuner, start, 121..=150, true, 0);
        observe(&mut tuner, start, 151..=240, false, 0);
        tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(240),
            SystemTime::now(),
        );
        observe(&mut tuner, start, 241..=359, false, 0);
        tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(360),
            SystemTime::now(),
        );
        observe(&mut tuner, start, 360..=479, false, 0);
        let changes = tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(480),
            SystemTime::now(),
        );
        assert!(changes.iter().any(|change| change.cell == 0));
        assert_eq!(tuner.state().learned[0], 300.0);
        observe(&mut tuner, start, 480..=599, false, 0);
        tuner.evaluate_with_baseline(
            200.0,
            &[],
            start + Duration::from_secs(600),
            SystemTime::now(),
        );
        assert_eq!(tuner.state().learned[0], 0.0);
        assert_eq!(
            tuner.effective_grid_from_baseline(100.0, &[])[0],
            100.0,
            "a later manual decrease must not be held up by a fully relaxed value"
        );
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
        assert_eq!(snapshot.adaptation_status[4], CellAdaptationStatus::Off);
    }

    #[test]
    fn snapshot_explains_below_threshold_cooldown_and_ceiling() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Auto);
        observe(&mut tuner, start, 0..=119, false, 0);
        let quiet = tuner.snapshot(&[], start + Duration::from_secs(120));
        assert_eq!(
            quiet.adaptation_status[0],
            CellAdaptationStatus::BelowThreshold
        );

        observe(&mut tuner, start, 120..=239, true, 1);
        tuner.evaluate(start + Duration::from_secs(240), SystemTime::now());
        let changed = tuner.snapshot(&[], start + Duration::from_secs(240));
        assert!(changed.trigger_fraction[1] > 0.9);
        assert_eq!(changed.adaptation_status[1], CellAdaptationStatus::Cooldown);

        observe(&mut tuner, start, 240..=359, true, 1);
        tuner.evaluate(start + Duration::from_secs(360), SystemTime::now());
        let capped = tuner.snapshot(&[], start + Duration::from_secs(360));
        assert_eq!(capped.adaptation_status[1], CellAdaptationStatus::Ceiling);
    }

    #[test]
    fn changing_mode_or_parameters_discards_timing_history() {
        let start = Instant::now();
        let mut tuner = MotionTuner::new(params());
        tuner.set_mode(TunerMode::Shadow);
        observe(&mut tuner, start, 0..=119, true, 0);
        assert!(
            tuner
                .snapshot(&[], start + Duration::from_secs(120))
                .window_full
        );

        tuner.set_mode(TunerMode::Auto);
        assert!(tuner.last_step.iter().all(Option::is_none));
        assert!(tuner.quiet_since.iter().all(Option::is_none));

        let mut changed = params();
        changed.window_secs = 180;
        tuner.set_params(changed);
        assert!(tuner.buckets.is_empty());
        assert!(tuner.observations.is_empty());
        assert!(
            !tuner
                .snapshot(&[], start + Duration::from_secs(180))
                .window_full
        );
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
