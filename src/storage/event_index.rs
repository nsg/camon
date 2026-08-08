//! The in-RAM warm event index, and the retention skeletons built on it.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::locks::LockExt;

const NANOS_PER_MS: u64 = 1_000_000;

/// The most filmstrip frames one event ever has. Taken from the analyzer that writes them, not
/// agreed by convention: every probe, count and delete stops here, so a producer raised past a
/// stale copy would leak frames nothing counts or deletes.
pub(crate) const MAX_FILMSTRIP_FRAMES: usize = crate::analytics::pipeline::FILMSTRIP_FRAMES;

/// Frames to index for an event whose frames answer `present`: one past the highest that
/// exists, capped at [`MAX_FILMSTRIP_FRAMES`].
pub(crate) fn filmstrip_frame_count(mut present: impl FnMut(usize) -> bool) -> usize {
    (0..MAX_FILMSTRIP_FRAMES)
        .rfind(|&i| present(i))
        .map_or(0, |highest| highest + 1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Movement,
    Object,
    /// A chunk of gapless continuous recording (analytics disabled — "dumb NVR"
    /// mode). Not motion-gated; every segment reaches disk.
    Continuous,
}

impl EventType {
    pub(crate) fn dir_name(self) -> &'static str {
        match self {
            EventType::Movement => "movements",
            EventType::Object => "objects",
            EventType::Continuous => "continuous",
        }
    }

    /// Wire name used by the remote (stathost) backend's sidecar, which carries
    /// the event type in JSON rather than in a directory name.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            EventType::Movement => "movement",
            EventType::Object => "object",
            EventType::Continuous => "continuous",
        }
    }

    /// Parse a wire name back into an [`EventType`]; unknown strings yield `None`.
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "movement" => Some(EventType::Movement),
            "object" => Some(EventType::Object),
            "continuous" => Some(EventType::Continuous),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionDetail {
    pub class: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct WarmEventEntry {
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    pub event_type: EventType,
    /// Size of the video object alone — what playback ranges are resolved
    /// against; folding in the sidecar would serve ranges past the video's end.
    pub file_size: u64,
    /// Size of the event's sidecar, or zero where there is none.
    pub sidecar_bytes: u64,
    /// Size of the filmstrip frames that are really there — which on a filmstrip with a hole
    /// is fewer than `filmstrip_frames` names: the count is a high-water mark
    /// ([`filmstrip_frame_count`]), the bytes only what the listing accounted for.
    pub thumbnail_bytes: u64,
    pub object_classes: Vec<String>,
    pub backend: Option<String>,
    pub model: Option<String>,
    pub detections: Vec<DetectionDetail>,
    /// Number of filmstrip thumbnail frames stored for this event
    /// (`{stem}_thumb_{0..n-1}.jpg`). A motion run is subsampled to at most 4
    /// frames, and short runs yield fewer — the UI renders exactly this many.
    pub filmstrip_frames: usize,
    /// True when this event is a follow-on chunk of a longer motion run split
    /// at the duration cap (from the sidecar `"continues"` flag).
    pub continues: bool,
    /// True when this event was salvaged from an orphaned `.ts.tmp` at startup
    /// after a crash or power cut (from the sidecar `"recovered"` flag). The
    /// tail may be truncated at the last intact packet.
    pub recovered: bool,
    /// Set once a deletion of this event has failed. In-RAM only (a restart
    /// clears it). A flag, not an exclusion: [`EvictionPolicy`] skips or
    /// demotes flagged entries, and the hourly sweep ignores it and retries.
    pub delete_failed: bool,
}

impl WarmEventEntry {
    /// Every byte this event costs the store — video, sidecar and filmstrip
    /// together, since the latter two are only ever reclaimed with the event.
    /// Saturating, because a corrupt listing must not wrap a budget to "empty".
    pub fn stored_bytes(&self) -> u64 {
        self.file_size
            .saturating_add(self.sidecar_bytes)
            .saturating_add(self.thumbnail_bytes)
    }

    /// End of this event in wall-clock nanoseconds. Saturating, because a
    /// corrupt or hostile duration must not wrap the window arithmetic that
    /// decides what is served.
    fn end_pts_ns(&self) -> u64 {
        self.start_pts_ns
            .saturating_add((self.duration_ms as u64) * NANOS_PER_MS)
    }
}

/// The best confidence seen per class. A sidecar records one line per class,
/// and the analytics pipeline can report the same class several times in one
/// event.
pub(crate) fn deduplicate_detections(details: &[DetectionDetail]) -> Vec<(String, f32)> {
    let mut best: HashMap<String, f32> = HashMap::new();
    for d in details {
        let entry = best.entry(d.class.clone()).or_insert(0.0);
        if d.confidence > *entry {
            *entry = d.confidence;
        }
    }
    best.into_iter().collect()
}

/// What identifies one indexed event, and with it exactly one set of stored objects. Start PTS
/// alone does not — nothing enforces its uniqueness, so unindexing on it could drop a
/// surviving entry on another entry's delete.
pub(crate) trait EventIdentity: Copy + Eq {
    /// The identity of an entry already in hand.
    fn of(entry: &WarmEventEntry) -> Self;
    /// This key's place in [`page_key`]'s order — what every list is sorted by — relative
    /// to an entry. `Equal` exactly where `of` would yield this key: each identity spells
    /// enough of the full key to be unique in its own index.
    fn cmp_entry(self, entry: &WarmEventEntry) -> std::cmp::Ordering;
}

/// The whole identity of one event, as an API request spells it: the composite path segment
/// `{start_pts_ns}_{duration_ms}_{event_type}`, e.g.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventRef {
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    pub event_type: EventType,
}

impl EventRef {
    pub fn new(start_pts_ns: u64, duration_ms: u32, event_type: EventType) -> Self {
        Self {
            start_pts_ns,
            duration_ms,
            event_type,
        }
    }

    /// The key an indexed entry answers to.
    pub fn of(entry: &WarmEventEntry) -> Self {
        Self::new(entry.start_pts_ns, entry.duration_ms, entry.event_type)
    }

    /// Parse one path segment. Every part is required and exact; anything else
    /// is `None` (a 400). Left-splitting is unambiguous — no type name contains
    /// an underscore, so extra parts fail rather than truncate silently.
    pub fn parse(segment: &str) -> Option<Self> {
        let (start, rest) = segment.split_once('_')?;
        let (duration, event_type) = rest.split_once('_')?;
        Some(Self::new(
            start.parse().ok()?,
            duration.parse().ok()?,
            EventType::from_str(event_type)?,
        ))
    }
}

impl std::fmt::Display for EventRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}_{}_{}",
            self.start_pts_ns,
            self.duration_ms,
            self.event_type.as_str()
        )
    }
}

/// Local disk: the event type is a directory, and so part of the path.
impl EventIdentity for (u64, EventType, u32) {
    fn of(entry: &WarmEventEntry) -> Self {
        (entry.start_pts_ns, entry.event_type, entry.duration_ms)
    }

    fn cmp_entry(self, entry: &WarmEventEntry) -> std::cmp::Ordering {
        (self.0, self.2, self.1.as_str()).cmp(&page_key(entry))
    }
}

/// The remote store: the key is the object stem `{start_pts_ns}_{duration_ms}`,
/// a prefix of [`page_key`] no two entries here share.
impl EventIdentity for (u64, u32) {
    fn of(entry: &WarmEventEntry) -> Self {
        (entry.start_pts_ns, entry.duration_ms)
    }

    fn cmp_entry(self, entry: &WarmEventEntry) -> std::cmp::Ordering {
        self.cmp(&(entry.start_pts_ns, entry.duration_ms))
    }
}

/// Outcome of deleting one indexed event's stored objects; only
/// [`Removal::Deleted`] reclaimed any space.
pub(crate) enum Removal {
    /// The video is gone; its bytes are back.
    Deleted,
    /// The video was already absent. Nothing was reclaimed, but the stale
    /// index entry has to go too.
    Missing,
    /// Shutdown arrived before or during the deletion. Ends the pass, nothing flagged or
    /// counted — distinct from [`Failed`](Self::Failed) because eviction demotes flagged
    /// entries, and a shutdown is not the store refusing.
    Abandoned,
    /// The store refused or could not be reached. The entry stays indexed so a
    /// later pass retries it instead of leaking the objects — a visible
    /// retention violation beats a file that is never retried.
    Failed,
}

/// What one deletion pass achieved, split by outcome — the three counts mean
/// different things to an operator.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EmergencyOutcome {
    /// Events deleted; the only count that reflects reclaimed bytes.
    pub deleted: u64,
    /// Events whose deletion failed. They are still stored and still indexed.
    pub failed: u64,
    /// Events whose objects had already vanished. Nothing was reclaimed here,
    /// but the stale index entries were dropped.
    pub missing: u64,
}

/// Position of the entry with this key: a binary search in [`page_key`]'s
/// order, which every list is kept sorted in and each identity is unique
/// under.
fn position<K: EventIdentity>(entries: &[WarmEventEntry], key: K) -> Option<usize> {
    entries
        .binary_search_by(|e| key.cmp_entry(e).reverse())
        .ok()
}

/// The total order the lists are sorted in and a listing pages on: start PTS, then duration,
/// then the type's wire name.
fn page_key(entry: &WarmEventEntry) -> (u64, u32, &'static str) {
    (
        entry.start_pts_ns,
        entry.duration_ms,
        entry.event_type.as_str(),
    )
}

/// Where a page resumes: an exclusive upper bound in [`page_key`]'s order.
/// Separate from `to_ns`, which matches by overlap — walking `to_ns` down
/// would re-serve long events; a bound that only descends cannot repeat.
#[derive(Debug, Clone, Copy)]
pub enum EventCursor {
    /// Everything that starts before this moment. Enough while starts are
    /// distinct, but cannot resume inside a run of equal starts.
    Start(u64),
    /// Everything ordered below this event — a position even inside a run of
    /// equal starts: the key of a page's oldest event resumes beneath it.
    Event(EventRef),
}

impl EventCursor {
    /// Whether an entry with this key is below the cursor. Monotone in
    /// [`page_key`]'s order, so a sorted list binary-searches on it.
    fn admits(self, key: (u64, u32, &'static str)) -> bool {
        match self {
            EventCursor::Start(start_pts_ns) => key.0 < start_pts_ns,
            EventCursor::Event(event) => {
                key < (
                    event.start_pts_ns,
                    event.duration_ms,
                    event.event_type.as_str(),
                )
            }
        }
    }
}

/// What one listing request may read: window, resume point, and a hard ceiling on how much of
/// the archive it may copy.
#[derive(Debug, Clone, Copy)]
pub struct EventPage {
    /// Events are matched by overlap with `[from_ns, to_ns]`, not by start.
    pub from_ns: u64,
    pub to_ns: u64,
    /// Where the previous page ended, if this is not the first.
    pub before: Option<EventCursor>,
    /// The most entries this page may copy. Never exceeded.
    pub limit: usize,
}

impl EventPage {
    /// The newest `limit` events overlapping `[from_ns, to_ns]`.
    pub fn new(from_ns: u64, to_ns: u64, limit: usize) -> Self {
        Self {
            from_ns,
            to_ns,
            before: None,
            limit,
        }
    }

    /// Resume beneath this cursor — for a walker, the key of the oldest event
    /// in the page it just received.
    pub fn before(self, cursor: EventCursor) -> Self {
        Self {
            before: Some(cursor),
            ..self
        }
    }

    /// The whole window, however deep the archive is. Tests only.
    #[cfg(test)]
    pub(crate) fn unbounded(from_ns: u64, to_ns: u64) -> Self {
        Self::new(from_ns, to_ns, usize::MAX)
    }
}

/// The greatest `limit` entries of `newest_first` that reach into the window, in ascending
/// [`page_key`] order.
fn page_of<'a>(
    newest_first: impl Iterator<Item = &'a WarmEventEntry>,
    from_ns: u64,
    limit: usize,
) -> Vec<WarmEventEntry> {
    let mut page: Vec<WarmEventEntry> = newest_first
        .filter(|e| e.end_pts_ns() >= from_ns)
        .take(limit)
        .cloned()
        .collect();
    page.reverse();
    page
}

/// The per-camera event lists both backends index into, keyed by `K`.
pub(crate) struct EventIndex<K> {
    cameras: HashMap<String, RwLock<Vec<WarmEventEntry>>>,
    used_bytes: AtomicU64,
    /// Entries shifted to keep the lists sorted, so a test can tell a
    /// one-at-a-time build from a bulk one — see `replace_camera`.
    #[cfg(test)]
    shifted_entries: AtomicU64,
    _key: PhantomData<fn() -> K>,
}

impl<K: EventIdentity> EventIndex<K> {
    pub(crate) fn new(camera_ids: &[String]) -> Self {
        Self {
            cameras: camera_ids
                .iter()
                .map(|id| (id.clone(), RwLock::new(Vec::new())))
                .collect(),
            used_bytes: AtomicU64::new(0),
            #[cfg(test)]
            shifted_entries: AtomicU64::new(0),
            _key: PhantomData,
        }
    }

    /// Entries shifted so far to keep the lists sorted; see the field.
    #[cfg(test)]
    pub(crate) fn shifted_entries(&self) -> u64 {
        self.shifted_entries.load(Ordering::Relaxed)
    }

    pub(crate) fn camera_ids(&self) -> impl Iterator<Item = &str> {
        self.cameras.keys().map(String::as_str)
    }

    pub(crate) fn owns_camera(&self, camera_id: &str) -> bool {
        self.cameras.contains_key(camera_id)
    }

    /// Sum of what every indexed event costs the store
    /// ([`WarmEventEntry::stored_bytes`]), across every camera.
    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Move tracked usage by the difference, in one atomic operation: an add then a subtract
    /// would let the concurrent budget guard see an inflated total and evict against it.
    fn charge(&self, added: u64, removed: u64) {
        if removed > added {
            self.used_bytes
                .fetch_sub(removed - added, Ordering::Relaxed);
        } else {
            self.used_bytes
                .fetch_add(added - removed, Ordering::Relaxed);
        }
    }

    /// Replace one camera's whole list, as a startup scan does: unsorted input, one O(n log n)
    /// sort — per-event insertion would be O(n²) on the one pass that covers the whole
    /// archive.
    pub(crate) fn replace_camera(&self, camera_id: &str, mut entries: Vec<WarmEventEntry>) {
        let Some(lock) = self.cameras.get(camera_id) else {
            return;
        };
        entries.sort_unstable_by_key(page_key);
        let added: u64 = entries.iter().map(WarmEventEntry::stored_bytes).sum();
        let removed: u64 = {
            let mut slot = lock.write_recover();
            let removed = slot.iter().map(WarmEventEntry::stored_bytes).sum();
            *slot = entries;
            removed
        };
        self.charge(added, removed);
    }

    /// Index one event, replacing whatever entry already held its identity and returning it.
    pub(crate) fn insert(&self, camera_id: &str, entry: WarmEventEntry) -> Option<WarmEventEntry> {
        let lock = self.cameras.get(camera_id)?;
        let added = entry.stored_bytes();
        let replaced = self.insert_locked(&mut lock.write_recover(), entry);
        self.charge(
            added,
            replaced.as_ref().map_or(0, WarmEventEntry::stored_bytes),
        );
        replaced
    }

    /// Index one event only if nothing holds its identity yet, reporting whether it landed.
    pub(crate) fn insert_absent(&self, camera_id: &str, entry: WarmEventEntry) -> bool {
        let Some(lock) = self.cameras.get(camera_id) else {
            return false;
        };
        let added = entry.stored_bytes();
        {
            let mut entries = lock.write_recover();
            if position(&entries, K::of(&entry)).is_some() {
                return false;
            }
            self.insert_locked(&mut entries, entry);
        }
        self.charge(added, 0);
        true
    }

    /// [`insert`](Self::insert)'s list surgery, without the byte accounting, so
    /// [`reidentify`](Self::reidentify) can re-place an entry under the write lock it already
    /// holds.
    fn insert_locked(
        &self,
        entries: &mut Vec<WarmEventEntry>,
        entry: WarmEventEntry,
    ) -> Option<WarmEventEntry> {
        let key = K::of(&entry);
        match entries.binary_search_by(|e| key.cmp_entry(e).reverse()) {
            Ok(i) => Some(std::mem::replace(&mut entries[i], entry)),
            Err(pos) => {
                #[cfg(test)]
                self.shifted_entries
                    .fetch_add((entries.len() - pos) as u64, Ordering::Relaxed);
                entries.insert(pos, entry);
                None
            }
        }
    }

    /// Drop one event from the index and refund its bytes, returning it.
    pub(crate) fn remove(&self, camera_id: &str, key: K) -> Option<WarmEventEntry> {
        let lock = self.cameras.get(camera_id)?;
        let removed = {
            let mut entries = lock.write_recover();
            let idx = position(&entries, key)?;
            entries.remove(idx)
        };
        self.charge(0, removed.stored_bytes());
        Some(removed)
    }

    /// Mutate the entry with this key in place.
    pub(crate) fn update<R>(
        &self,
        camera_id: &str,
        key: K,
        f: impl FnOnce(&mut WarmEventEntry) -> R,
    ) -> Option<R> {
        let lock = self.cameras.get(camera_id)?;
        let mut entries = lock.write_recover();
        let idx = position(&entries, key)?;
        Some(f(&mut entries[idx]))
    }

    /// Mutate the entry with this key when the mutation changes the key — the
    /// movement→object upgrade. `false` when no such event is indexed.
    pub(crate) fn reidentify(
        &self,
        camera_id: &str,
        key: K,
        f: impl FnOnce(&mut WarmEventEntry),
    ) -> bool {
        self.reidentify_if(camera_id, key, |entry| {
            f(entry);
            true
        })
    }

    /// [`reidentify`](Self::reidentify) for a mutation that decides, entry in hand, whether to
    /// happen at all — `false` leaves the index as it was.
    pub(crate) fn reidentify_if(
        &self,
        camera_id: &str,
        key: K,
        f: impl FnOnce(&mut WarmEventEntry) -> bool,
    ) -> bool {
        let Some(lock) = self.cameras.get(camera_id) else {
            return false;
        };
        let (old_size, new_size, displaced) = {
            let mut entries = lock.write_recover();
            let Some(i) = position(&entries, key) else {
                return false;
            };
            let mut entry = entries[i].clone();
            let old_size = entry.stored_bytes();
            if !f(&mut entry) {
                return false;
            }
            let new_size = entry.stored_bytes();
            entries.remove(i);
            (old_size, new_size, self.insert_locked(&mut entries, entry))
        };
        // New size charged; the former size and anything displaced refunded.
        self.charge(
            new_size,
            old_size.saturating_add(displaced.map_or(0, |e| e.stored_bytes())),
        );
        true
    }

    pub(crate) fn contains(&self, camera_id: &str, key: K) -> bool {
        self.cameras
            .get(camera_id)
            .is_some_and(|lock| position(&lock.read_recover(), key).is_some())
    }

    /// Remember that this event resisted deletion. Keyed on the full identity:
    /// an entry that has been upgraded since is a different event and must not
    /// inherit the flag.
    pub(crate) fn flag_delete_failed(&self, camera_id: &str, key: K) {
        self.update(camera_id, key, |entry| entry.delete_failed = true);
    }

    /// One page of the events overlapping [`EventPage`]'s window, in ascending [`page_key`]
    /// order. An inverted range is empty.
    pub(crate) fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry> {
        if page.from_ns > page.to_ns {
            return Vec::new();
        }
        let Some(lock) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        let entries = lock.read_recover();
        let mut end = entries.partition_point(|e| e.start_pts_ns <= page.to_ns);
        if let Some(cursor) = page.before {
            end = end.min(entries.partition_point(|e| cursor.admits(page_key(e))));
        }
        page_of(entries[..end].iter().rev(), page.from_ns, page.limit)
    }

    /// The entry with this key, for the API read path. Keyed in full: a search
    /// on the start alone would offer an arbitrary member of a same-start run
    /// for playback.
    pub(crate) fn find(&self, camera_id: &str, key: K) -> Option<WarmEventEntry> {
        let entries = self.cameras.get(camera_id)?.read_recover();
        position(&entries, key).map(|i| entries[i].clone())
    }

    /// End of the newest indexed event, in wall-clock nanoseconds. The sort
    /// key leads with the start, so the last entry is the newest — and of a
    /// run of equal starts, the longest.
    pub(crate) fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        let entries = self.cameras.get(camera_id)?.read_recover();
        entries.last().map(WarmEventEntry::end_pts_ns)
    }

    /// This camera's events past their retention, oldest first and already capped to this
    /// sweep's share (see [`cap_sweep_deletions`]).
    pub(crate) fn expired_for_sweep(
        &self,
        camera_id: &str,
        now_ns: u64,
        max_age: impl Fn(&WarmEventEntry) -> u64,
    ) -> Vec<WarmEventEntry> {
        let Some(lock) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        let (indexed, expired) = {
            let entries = lock.read_recover();
            let expired: Vec<WarmEventEntry> = entries
                .iter()
                .filter(|e| now_ns.saturating_sub(e.start_pts_ns) > max_age(e))
                .cloned()
                .collect();
            (entries.len(), expired)
        };
        if expired.is_empty() {
            return Vec::new();
        }
        cap_sweep_deletions(camera_id, indexed, expired)
    }

    /// Every indexed event `accept` says yes to, across all cameras, each
    /// paired with the camera it belongs to.
    fn candidates(
        &self,
        accept: impl Fn(&str, &WarmEventEntry) -> bool,
    ) -> Vec<(String, WarmEventEntry)> {
        let mut candidates = Vec::new();
        for (camera_id, lock) in self.cameras.iter() {
            candidates.extend(
                lock.read_recover()
                    .iter()
                    .filter(|e| accept(camera_id, e))
                    .cloned()
                    .map(|e| (camera_id.clone(), e)),
            );
        }
        candidates
    }
}

/// Delete one camera's expired events and unindex what actually went — the failure handling
/// both backends share.
pub(crate) async fn sweep_expired<K, F, Fut>(
    index: &EventIndex<K>,
    camera_id: &str,
    expired: Vec<WarmEventEntry>,
    mut cancel: impl FnMut() -> bool,
    mut remove: F,
) -> EmergencyOutcome
where
    K: EventIdentity,
    F: FnMut(WarmEventEntry) -> Fut,
    Fut: std::future::Future<Output = Removal>,
{
    let mut outcome = EmergencyOutcome::default();
    for entry in expired {
        if cancel() {
            break;
        }
        let key = K::of(&entry);
        match remove(entry).await {
            Removal::Deleted => {
                index.remove(camera_id, key);
                outcome.deleted += 1;
            }
            Removal::Missing => {
                index.remove(camera_id, key);
                outcome.missing += 1;
            }
            Removal::Failed => {
                outcome.failed += 1;
                index.flag_delete_failed(camera_id, key);
            }
            // Shutdown: the entry stays, unflagged and uncounted.
            Removal::Abandoned => break,
        }
    }
    outcome
}

/// Eviction order, cheapest footage to lose first.
const EVICTION_TIERS: [EventType; 3] = [
    EventType::Continuous,
    EventType::Movement,
    EventType::Object,
];

/// How a space-pressure pass treats an event that has already refused to be
/// deleted, and what it says when one goes — the two backends fail differently
/// enough that these cannot be defaults.
pub(crate) struct EvictionPolicy {
    /// Never offer a flagged event again (local disk) rather than only demoting it to the back
    /// of its tier (the remote store).
    pub skip_failed: bool,
    /// One refused delete ends the whole pass (the remote store) rather than being stepped over
    /// (local disk).
    pub stop_on_failure: bool,
    /// What to log when an event is evicted — disk pressure and a full
    /// client-side budget are different situations to be told about.
    pub reason: &'static str,
}

/// Delete the oldest events, cheapest tier first, until `satisfied` reports the pressure is
/// gone or the candidates are exhausted.
pub(crate) async fn evict_tiers<K, F, Fut>(
    index: &EventIndex<K>,
    policy: EvictionPolicy,
    tier_of: impl Fn(&str, &WarmEventEntry) -> EventType,
    mut satisfied: impl FnMut() -> bool,
    mut cancel: impl FnMut() -> bool,
    mut remove: F,
) -> EmergencyOutcome
where
    K: EventIdentity,
    F: FnMut(String, WarmEventEntry) -> Fut,
    Fut: std::future::Future<Output = Removal>,
{
    let mut outcome = EmergencyOutcome::default();
    'tiers: for tier in EVICTION_TIERS {
        let mut candidates = index.candidates(|camera_id, entry| {
            (!policy.skip_failed || !entry.delete_failed) && tier_of(camera_id, entry) == tier
        });
        // Oldest first, with already-failed entries demoted behind the rest:
        // retries are reached only once the untried candidates are exhausted.
        candidates.sort_by_key(|(_, e)| (e.delete_failed, e.start_pts_ns));

        for (camera_id, entry) in candidates {
            if satisfied() || cancel() {
                break 'tiers;
            }
            let key = K::of(&entry);
            let (start_pts_ns, event_type) = (entry.start_pts_ns, entry.event_type);
            match remove(camera_id.clone(), entry).await {
                Removal::Failed => {
                    outcome.failed += 1;
                    index.flag_delete_failed(&camera_id, key);
                    if policy.stop_on_failure {
                        break 'tiers;
                    }
                }
                // Shutdown: unflagged, uncounted, end of the pass.
                Removal::Abandoned => break 'tiers,
                Removal::Missing => {
                    index.remove(&camera_id, key);
                    outcome.missing += 1;
                }
                Removal::Deleted => {
                    index.remove(&camera_id, key);
                    outcome.deleted += 1;
                    tracing::warn!(
                        camera = %camera_id,
                        start_pts_ns,
                        ?event_type,
                        "{}",
                        policy.reason
                    );
                }
            }
        }
    }
    outcome
}

/// Share of a camera's archive one sweep may delete — a quarter.
const SWEEP_DELETE_SHARE: usize = 4;

/// Floor under the share, so an archive of a handful of events still expires
/// in one sweep instead of dribbling out an hour at a time.
const SWEEP_DELETE_FLOOR: usize = 4;

fn sweep_delete_limit(indexed: usize) -> usize {
    indexed.div_ceil(SWEEP_DELETE_SHARE).max(SWEEP_DELETE_FLOOR)
}

/// Hold back the tail of an over-large expiry, so no single sweep can empty an archive.
fn cap_sweep_deletions(
    camera_id: &str,
    indexed: usize,
    expired: Vec<WarmEventEntry>,
) -> Vec<WarmEventEntry> {
    let expired_count = expired.len();
    let limit = sweep_delete_limit(indexed);
    let mut budget = limit;
    let mut held_back = 0usize;
    // Filtering in place keeps oldest-first order, so a sweep cut short by
    // shutdown has still deleted the events nearest their retention.
    let deleting: Vec<WarmEventEntry> = expired
        .into_iter()
        .filter(|entry| {
            if entry.delete_failed {
                return true;
            }
            if budget > 0 {
                budget -= 1;
                return true;
            }
            held_back += 1;
            false
        })
        .collect();

    let deleting_count = deleting.len();
    if held_back > 0 {
        tracing::warn!(
            camera = %camera_id,
            indexed,
            expired = expired_count,
            deleting = deleting_count,
            held_back,
            "more warm events expired at once than one sweep may delete — a forward clock \
             jump, a shortened retention and a long outage all look like this. Deleting the \
             oldest {deleting_count} of {expired_count} expired; the {held_back} held back \
             follow on later sweeps"
        );
    }
    deleting
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(start_pts_ns: u64, event_type: EventType, duration_ms: u32) -> WarmEventEntry {
        WarmEventEntry {
            start_pts_ns,
            duration_ms,
            event_type,
            file_size: 0,
            sidecar_bytes: 0,
            thumbnail_bytes: 0,
            object_classes: Vec::new(),
            backend: None,
            model: None,
            detections: Vec::new(),
            filmstrip_frames: 0,
            continues: false,
            recovered: false,
            delete_failed: false,
        }
    }

    #[test]
    fn the_frame_count_stops_where_the_analyzer_stops_writing() {
        assert_eq!(
            MAX_FILMSTRIP_FRAMES,
            crate::analytics::pipeline::FILMSTRIP_FRAMES
        );
        assert_eq!(
            filmstrip_frame_count(|_| true),
            crate::analytics::pipeline::FILMSTRIP_FRAMES,
            "a full filmstrip has a frame the store would never account for"
        );
    }

    #[test]
    fn a_filmstrip_count_names_every_frame_that_exists() {
        let count = |present: &[usize]| {
            let present = present.to_vec();
            filmstrip_frame_count(move |i| present.contains(&i))
        };
        assert_eq!(count(&[]), 0);
        assert_eq!(count(&[0]), 1);
        assert_eq!(count(&[0, 1, 2, 3]), 4);
        assert_eq!(count(&[1, 2, 3]), 4);
        assert_eq!(count(&[0, 2]), 3);
        assert_eq!(count(&[3]), 4);
        let mut asked = Vec::new();
        for present in [true, false] {
            asked.clear();
            filmstrip_frame_count(|i| {
                asked.push(i);
                present
            });
            assert!(asked.len() <= MAX_FILMSTRIP_FRAMES, "{asked:?}");
            assert!(asked.iter().all(|&i| i < MAX_FILMSTRIP_FRAMES), "{asked:?}");
        }
    }

    #[test]
    fn position_finds_the_named_entry_within_a_run_of_equal_starts() {
        let entries = [
            entry(1000, EventType::Movement, 1000),
            entry(2000, EventType::Movement, 1000),
            entry(2000, EventType::Object, 1000),
            entry(2000, EventType::Movement, 2000),
            entry(3000, EventType::Movement, 1000),
        ];
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(position(&entries, <(u64, EventType, u32)>::of(e)), Some(i));
        }
        assert_eq!(
            position(&entries, (2000, EventType::Continuous, 1000)),
            None
        );
        assert_eq!(position(&entries, (2000, EventType::Object, 2000)), None);
        assert_eq!(position(&entries, (2500, EventType::Movement, 1000)), None);
        assert_eq!(
            position::<(u64, EventType, u32)>(&[], (1000, EventType::Movement, 1000)),
            None
        );
    }

    #[test]
    fn position_separates_a_run_by_the_remote_identity() {
        let entries = [
            entry(2000, EventType::Movement, 1000),
            entry(2000, EventType::Object, 2000),
        ];
        assert_eq!(position(&entries, (2000u64, 1000u32)), Some(0));
        assert_eq!(position(&entries, (2000u64, 2000u32)), Some(1));
        assert_eq!(position(&entries, (2000u64, 3000u32)), None);
    }

    #[test]
    fn find_returns_the_named_member_of_a_run_of_equal_starts() {
        let local: EventIndex<(u64, EventType, u32)> = EventIndex::new(&["cam".to_string()]);
        local.insert("cam", sized(entry(2000, EventType::Movement, 1000), 10));
        local.insert("cam", sized(entry(2000, EventType::Continuous, 1000), 20));
        assert_eq!(
            local
                .find("cam", (2000, EventType::Movement, 1000))
                .unwrap()
                .file_size,
            10
        );
        assert_eq!(
            local
                .find("cam", (2000, EventType::Continuous, 1000))
                .unwrap()
                .file_size,
            20
        );
        assert!(local
            .find("cam", (2500, EventType::Movement, 1000))
            .is_none());
        assert!(local.find("cam", (2000, EventType::Object, 1000)).is_none());
        assert!(local
            .find("other", (2000, EventType::Movement, 1000))
            .is_none());

        let remote: EventIndex<(u64, u32)> = EventIndex::new(&["cam".to_string()]);
        remote.insert("cam", sized(entry(2000, EventType::Movement, 1000), 30));
        remote.insert("cam", sized(entry(2000, EventType::Object, 2000), 40));
        assert_eq!(remote.find("cam", (2000, 1000)).unwrap().file_size, 30);
        assert_eq!(remote.find("cam", (2000, 2000)).unwrap().file_size, 40);
        assert!(remote.find("cam", (2000, 3000)).is_none());
    }

    #[test]
    fn an_event_ref_round_trips_through_its_path_segment() {
        for event_type in [
            EventType::Movement,
            EventType::Object,
            EventType::Continuous,
        ] {
            let key = EventRef::new(81_234_000_000, 5200, event_type);
            assert_eq!(
                key.to_string(),
                format!("81234000000_5200_{}", event_type.as_str())
            );
            assert_eq!(EventRef::parse(&key.to_string()), Some(key));
        }
        assert_eq!(
            EventRef::parse("81234000000_5200_movement"),
            Some(EventRef::new(81_234_000_000, 5200, EventType::Movement))
        );

        for bad in [
            "81234000000",                // the old bare start PTS
            "81234000000_5200",           // stem only, no type
            "81234000000_5200_",          // empty type
            "_5200_movement",             // empty start
            "81234000000__movement",      // empty duration
            "81234000000_5200_movements", // the directory name, not the type
            "81234000000_5200_movement_", // trailing junk
            "81234000000_5200_movement_extra",
            "81234000000_5200_MOVEMENT",
            "-1_5200_movement",
            "81234000000_-5_movement",
            "81234000000_99999999999_movement",   // past u32
            "99999999999999999999_5200_movement", // past u64
            " 81234000000_5200_movement",
            "",
        ] {
            assert_eq!(EventRef::parse(bad), None, "{bad:?} parsed");
        }
    }

    #[test]
    fn charge_moves_used_bytes_by_the_difference() {
        let index: EventIndex<(u64, EventType, u32)> = EventIndex::new(&[]);
        index.charge(100, 0);
        assert_eq!(index.used_bytes(), 100);
        index.charge(30, 100);
        assert_eq!(index.used_bytes(), 30);
        index.charge(80, 30);
        assert_eq!(index.used_bytes(), 80);
        index.charge(80, 80);
        assert_eq!(index.used_bytes(), 80);
        index.charge(0, 80);
        assert_eq!(index.used_bytes(), 0);
    }

    fn sized(mut entry: WarmEventEntry, file_size: u64) -> WarmEventEntry {
        entry.file_size = file_size;
        entry
    }

    fn index_with_cam() -> EventIndex<(u64, EventType, u32)> {
        EventIndex::new(&["cam".to_string()])
    }

    struct Counted<I> {
        inner: I,
        visited: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl<I: Iterator> Iterator for Counted<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<I::Item> {
            let next = self.inner.next();
            if next.is_some() {
                self.visited.set(self.visited.get() + 1);
            }
            next
        }
    }

    fn walk_pages(
        index: &EventIndex<(u64, EventType, u32)>,
        limit: usize,
        pages: usize,
    ) -> (Vec<WarmEventEntry>, Vec<usize>) {
        let mut seen: Vec<WarmEventEntry> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        let mut cursor: Option<EventCursor> = None;
        for _ in 0..pages {
            let mut page = EventPage::new(0, u64::MAX, limit);
            if let Some(c) = cursor {
                page = page.before(c);
            }
            let events = index.query("cam", page);
            if events.is_empty() {
                break;
            }
            sizes.push(events.len());
            cursor = Some(EventCursor::Event(EventRef::of(&events[0])));
            seen.splice(0..0, events);
        }
        (seen, sizes)
    }

    #[test]
    fn a_page_walks_only_as_far_as_its_limit() {
        let archive: Vec<WarmEventEntry> = (1..=100_000)
            .map(|i| entry(i * 1000, EventType::Movement, 1))
            .collect();

        let visited = std::rc::Rc::new(std::cell::Cell::new(0));
        let page = page_of(
            Counted {
                inner: archive.iter().rev(),
                visited: std::rc::Rc::clone(&visited),
            },
            0,
            50,
        );

        assert_eq!(page.len(), 50);
        assert_eq!(visited.get(), 50);
        assert_eq!(page[0].start_pts_ns, 99_951 * 1000);
        assert_eq!(page[49].start_pts_ns, 100_000 * 1000);
    }

    #[test]
    fn a_page_ends_inside_a_run_of_equal_starts_rather_than_finishing_it() {
        let index = index_with_cam();
        for (start, event_type) in [
            (1000, EventType::Movement),
            (2000, EventType::Movement),
            (2000, EventType::Object),
            (2000, EventType::Continuous),
            (3000, EventType::Movement),
        ] {
            index.insert("cam", entry(start, event_type, 1));
        }

        let page = index.query("cam", EventPage::new(0, u64::MAX, 2));

        assert_eq!(page.len(), 2);
        assert_eq!(page_key(&page[0]), (2000, 1, "object"));
        assert_eq!(page_key(&page[1]), (3000, 1, "movement"));

        assert!(index
            .query("cam", EventPage::new(0, u64::MAX, 0))
            .is_empty());

        let next = index.query(
            "cam",
            EventPage::new(0, u64::MAX, 2).before(EventCursor::Event(EventRef::of(&page[0]))),
        );
        assert_eq!(page_key(&next[0]), (2000, 1, "continuous"));
        assert_eq!(page_key(&next[1]), (2000, 1, "movement"));
    }

    #[test]
    fn a_single_start_archive_pages_at_exactly_its_limit() {
        let index = index_with_cam();
        let mut durations: Vec<u32> = (1..=3000).collect();
        durations.rotate_left(1499);
        for duration_ms in durations {
            index.insert("cam", entry(0, EventType::Movement, duration_ms));
        }

        let (seen, sizes) = walk_pages(&index, 1000, 6);

        assert_eq!(
            sizes,
            vec![1000, 1000, 1000],
            "pages are the size asked for"
        );
        let keys: Vec<(u64, u32, &str)> = seen.iter().map(page_key).collect();
        let every: Vec<(u64, u32, &str)> =
            (1..=3000).map(|d| (0u64, d as u32, "movement")).collect();
        assert_eq!(keys, every, "every event once, in page order, no holes");
    }

    #[test]
    fn ties_page_across_a_boundary_in_full_key_order() {
        let index = index_with_cam();
        for (event_type, duration_ms) in [
            (EventType::Object, 1),
            (EventType::Movement, 1),
            (EventType::Continuous, 2),
            (EventType::Movement, 2),
            (EventType::Continuous, 1),
        ] {
            index.insert("cam", entry(5000, event_type, duration_ms));
        }
        index.insert("cam", entry(4000, EventType::Movement, 1));
        index.insert("cam", entry(6000, EventType::Movement, 1));

        let (seen, sizes) = walk_pages(&index, 2, 5);

        assert_eq!(sizes, vec![2, 2, 2, 1]);
        let keys: Vec<(u64, u32, &str)> = seen.iter().map(page_key).collect();
        assert_eq!(
            keys,
            vec![
                (4000, 1, "movement"),
                (5000, 1, "continuous"),
                (5000, 1, "movement"),
                (5000, 1, "object"),
                (5000, 2, "continuous"),
                (5000, 2, "movement"),
                (6000, 1, "movement"),
            ]
        );
    }

    #[test]
    fn a_page_walk_reaches_the_oldest_event_without_repeating_any() {
        let index = index_with_cam();
        for i in 1..=25u64 {
            index.insert("cam", entry(i * 1000, EventType::Movement, 1));
        }

        let (seen, sizes) = walk_pages(&index, 7, 5);

        assert_eq!(sizes, vec![7, 7, 7, 4]);
        let starts: Vec<u64> = seen.iter().map(|e| e.start_pts_ns).collect();
        assert_eq!(starts, (1..=25u64).map(|i| i * 1000).collect::<Vec<_>>());
    }

    #[test]
    fn a_bare_start_cursor_resumes_a_walk_of_distinct_starts() {
        let index = index_with_cam();
        for i in 1..=10u64 {
            index.insert("cam", entry(i * 1000, EventType::Movement, 1));
        }

        let first = index.query("cam", EventPage::new(0, u64::MAX, 4));
        let cursor = EventCursor::Start(first[0].start_pts_ns);
        let second = index.query("cam", EventPage::new(0, u64::MAX, 4).before(cursor));

        assert_eq!(
            first.iter().map(|e| e.start_pts_ns).collect::<Vec<_>>(),
            vec![7000, 8000, 9000, 10_000]
        );
        assert_eq!(
            second.iter().map(|e| e.start_pts_ns).collect::<Vec<_>>(),
            vec![3000, 4000, 5000, 6000]
        );

        let third = index.query(
            "cam",
            EventPage::new(0, u64::MAX, 10).before(EventCursor::Start(3000)),
        );
        assert_eq!(
            third.iter().map(|e| e.start_pts_ns).collect::<Vec<_>>(),
            vec![1000, 2000]
        );
    }

    #[test]
    fn a_page_walk_survives_the_archive_changing_under_it() {
        let index = index_with_cam();
        for i in 1..=20u64 {
            index.insert("cam", entry(i * 1000, EventType::Movement, 1));
        }

        let first = index.query("cam", EventPage::new(0, u64::MAX, 10));
        assert_eq!(first.len(), 10);
        assert_eq!(first[0].start_pts_ns, 11_000);
        let cursor = EventCursor::Event(EventRef::of(&first[0]));

        index.insert("cam", entry(21_000, EventType::Movement, 1));
        index.remove("cam", (1000, EventType::Movement, 1));
        index.remove("cam", (11_000, EventType::Movement, 1));

        let second = index.query("cam", EventPage::new(0, u64::MAX, 10).before(cursor));
        let starts: Vec<u64> = second.iter().map(|e| e.start_pts_ns).collect();

        assert!(!starts.contains(&21_000));
        assert_eq!(starts, (2..=10u64).map(|i| i * 1000).collect::<Vec<_>>());
    }

    #[test]
    fn a_mid_run_cursor_survives_the_run_changing_under_it() {
        let index = index_with_cam();
        for duration_ms in 1..=10u32 {
            index.insert("cam", entry(0, EventType::Movement, duration_ms));
        }

        let first = index.query("cam", EventPage::new(0, u64::MAX, 4));
        let durations: Vec<u32> = first.iter().map(|e| e.duration_ms).collect();
        assert_eq!(durations, vec![7, 8, 9, 10]);
        let cursor = EventCursor::Event(EventRef::of(&first[0]));

        index.remove("cam", (0, EventType::Movement, 7));
        index.remove("cam", (0, EventType::Movement, 5));
        index.insert("cam", entry(0, EventType::Object, 3));

        let second = index.query("cam", EventPage::new(0, u64::MAX, 4).before(cursor));
        let keys: Vec<(u64, u32, &str)> = second.iter().map(page_key).collect();

        assert_eq!(
            keys,
            vec![
                (0, 3, "movement"),
                (0, 3, "object"),
                (0, 4, "movement"),
                (0, 6, "movement"),
            ]
        );
    }

    #[test]
    fn reidentify_displaces_the_entry_holding_the_new_identity() {
        let index = index_with_cam();
        index.insert("cam", sized(entry(1000, EventType::Movement, 500), 300));
        index.insert("cam", sized(entry(1000, EventType::Object, 500), 70));
        assert_eq!(index.used_bytes(), 370);

        assert!(
            index.reidentify("cam", (1000, EventType::Movement, 500), |e| {
                e.event_type = EventType::Object;
            })
        );

        let entries = index.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_size, 300);
        assert_eq!(entries[0].event_type, EventType::Object);
        assert_eq!(index.used_bytes(), 300);
    }

    #[test]
    fn insert_absent_yields_to_the_entry_already_holding_the_identity() {
        let index = index_with_cam();
        assert!(index.insert_absent("cam", sized(entry(1000, EventType::Movement, 500), 300)));
        assert_eq!(index.used_bytes(), 300);

        assert!(!index.insert_absent("cam", sized(entry(1000, EventType::Movement, 500), 70)));
        let entries = index.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_size, 300, "overwrote the indexed entry");
        assert_eq!(
            index.used_bytes(),
            300,
            "charged an insert that did not happen"
        );

        assert!(index.insert_absent("cam", sized(entry(1000, EventType::Object, 500), 70)));
        assert_eq!(
            index.query("cam", EventPage::unbounded(0, u64::MAX)).len(),
            2
        );
        assert_eq!(index.used_bytes(), 370);
        assert!(!index.insert_absent("other", sized(entry(1000, EventType::Movement, 500), 5)));
    }

    #[test]
    fn reidentify_with_a_no_op_closure_leaves_the_index_alone() {
        let index = index_with_cam();
        index.insert("cam", sized(entry(1000, EventType::Movement, 500), 300));

        assert!(index.reidentify("cam", (1000, EventType::Movement, 500), |_| {}));

        assert_eq!(
            index.query("cam", EventPage::unbounded(0, u64::MAX)).len(),
            1
        );
        assert_eq!(index.used_bytes(), 300);
        assert!(!index.reidentify("cam", (2000, EventType::Movement, 500), |_| {}));
        assert!(!index.reidentify("other", (1000, EventType::Movement, 500), |_| {}));
        assert_eq!(index.used_bytes(), 300);
    }

    #[test]
    fn reidentify_follows_a_resize_by_the_closure() {
        let index = index_with_cam();
        index.insert("cam", sized(entry(1000, EventType::Movement, 500), 300));
        index.insert("cam", sized(entry(1000, EventType::Object, 500), 70));

        assert!(
            index.reidentify("cam", (1000, EventType::Movement, 500), |e| {
                e.event_type = EventType::Object;
                e.file_size = 500;
            })
        );

        let entries = index.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_size, 500);
        assert_eq!(index.used_bytes(), 500);
    }

    #[tokio::test]
    async fn a_sweep_stops_between_events_when_it_is_cancelled_part_way() {
        let index = index_with_cam();
        let expired: Vec<WarmEventEntry> = (1..=4)
            .map(|i| sized(entry(i * 1000, EventType::Movement, 500), 10))
            .collect();
        for e in expired.iter() {
            index.insert("cam", e.clone());
        }

        let mut asked = 0;
        let outcome = sweep_expired(
            &index,
            "cam",
            expired,
            || {
                asked += 1;
                asked > 1
            },
            |_| async { Removal::Deleted },
        )
        .await;

        assert_eq!(outcome.deleted, 1);
        assert_eq!(
            index.query("cam", EventPage::unbounded(0, u64::MAX)).len(),
            3
        );
    }

    #[tokio::test]
    async fn an_eviction_stops_between_events_when_it_is_cancelled_part_way() {
        let index = index_with_cam();
        for i in 1..=4 {
            index.insert("cam", sized(entry(i * 1000, EventType::Movement, 500), 10));
        }

        let mut asked = 0;
        let outcome = evict_tiers(
            &index,
            EvictionPolicy {
                skip_failed: false,
                stop_on_failure: true,
                reason: "test",
            },
            |_, e| e.event_type,
            || false,
            || {
                asked += 1;
                asked > 1
            },
            |_, _| async { Removal::Deleted },
        )
        .await;

        assert_eq!(outcome.deleted, 1);
        assert_eq!(
            index.query("cam", EventPage::unbounded(0, u64::MAX)).len(),
            3
        );
    }

    #[test]
    fn reidentify_leaves_the_index_intact_when_the_closure_panics() {
        let index = index_with_cam();
        index.insert("cam", sized(entry(1000, EventType::Movement, 500), 300));

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            index.reidentify("cam", (1000, EventType::Movement, 500), |e| {
                e.event_type = EventType::Object;
                panic!("closure gives up half way");
            })
        }));
        assert!(panicked.is_err());

        let entries = index.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].event_type, EventType::Movement);
        assert_eq!(entries[0].file_size, 300);
        assert_eq!(index.used_bytes(), 300);
    }
}
