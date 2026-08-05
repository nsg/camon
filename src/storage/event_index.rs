//! The in-RAM warm event index, and the retention skeletons built on it.
//!
//! Both storage backends — [`LocalDiskBackend`](crate::storage::LocalDiskBackend)
//! and [`StathostBackend`](crate::storage::StathostBackend) — keep the same
//! thing in memory: per camera, a list of [`WarmEventEntry`] sorted by start
//! PTS, answering the API's `query`/`find_event` and driving retention. Only
//! the *objects* differ (files in a directory tree versus keys on an HTTP
//! store), so only the object I/O is the backends' own; everything above it
//! lives here and is written once.
//!
//! What is deliberately *not* unified is the identity of an entry. Local disk
//! carries the event type in the path (the directory), the remote store carries
//! it inside the sidecar, and each layout admits a different set of distinct
//! events — so the index is generic over an [`EventIdentity`] rather than
//! picking one spelling and making the other backend live with it. See that
//! trait for the argument in full.
//!
//! Every lookup — the read path's `find_event` included — names an entry by a
//! whole key and walks the run of equal starts ([`position`]). There is no
//! by-start-PTS variant to reach for: nothing makes a start unique, so a binary
//! search on one returns an arbitrary member of its run, which on the read path
//! meant serving the wrong recording of a same-start pair. What an API request
//! names is an [`EventRef`], and each backend narrows that to the identity its
//! own layout has.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::locks::LockExt;

const NANOS_PER_MS: u64 = 1_000_000;

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
    /// Size of the event's video object. What playback ranges are resolved
    /// against, and so the *video's* size and nothing else's.
    pub file_size: u64,
    /// Size of the event's sidecar, or zero where there is none.
    /// Kept apart from `file_size` because the two answer different questions —
    /// one is how many bytes a player may seek within, the other is how many
    /// bytes retention will reclaim — and a backend that folded them together
    /// would serve ranges past the end of the video.
    ///
    /// Kept apart from [`thumbnail_bytes`](Self::thumbnail_bytes) because the
    /// two change at different moments: an object upgrade rewrites the sidecar
    /// and touches no frame, and a shorter rewrite of a stem drops frames and
    /// leaves the sidecar. A single lumped figure could follow neither without
    /// knowing what the other half of it had been.
    ///
    /// Filled by whichever backend's accounting depends on it. Local disk leaves
    /// it zero and says why in [`contract`](crate::storage::contract): there the
    /// filesystem counts every byte natively and `statvfs` is the authority, so
    /// a figure maintained beside it could only ever be a second opinion that
    /// drifts.
    pub sidecar_bytes: u64,
    /// Size of the event's filmstrip frames, all `filmstrip_frames` of them.
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
    /// Set once a deletion of this event has failed. Purely in-RAM (a restart
    /// clears it, and the scan retries everything). What it buys differs by
    /// backend, which is why it is a flag and not an exclusion:
    /// [`EvictionPolicy`] either skips flagged entries outright or only demotes
    /// them to the back of their tier. The hourly sweep ignores it and keeps
    /// retrying — that is where a transient failure gets its second chance.
    pub delete_failed: bool,
}

impl WarmEventEntry {
    /// Every byte this event costs the store — the figure a client-side budget
    /// is measured against.
    ///
    /// Counting the video alone is what let a store sit permanently over a cap
    /// it believed it was under: an event's sidecar and its four filmstrip
    /// frames are small next to its video and are not small next to nothing, and
    /// they are never reclaimed on their own — they go when the event goes. So
    /// they are charged when the event is charged. Saturating, because a corrupt
    /// listing must not wrap a budget into "empty".
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

/// What identifies one indexed event, and with it exactly one set of stored
/// objects.
///
/// The start PTS alone does not. Nothing enforces its uniqueness — a scan
/// happily indexes two events sharing a start — so unindexing on it would drop
/// a surviving entry on the strength of some other entry's delete, which is the
/// leak the whole retention path exists to prevent.
///
/// Beyond that the two backends genuinely disagree, and neither spelling can be
/// imposed on the other:
///
/// * **Local disk** keys on `(start, event_type, duration)`. The type *is* a
///   path component there — `{camera}/{event_type}/{start}_{duration}.ts` — so
///   the same start and duration under two types are two files, and a key that
///   dropped the type would have one entry's delete unindex the other's file.
/// * **The remote store** keys on `(start, duration)`. The type lives inside
///   the sidecar and an upgrade rewrites it without moving a single object, so
///   two entries differing only in type would name *the same* objects — and the
///   upgrade would leave an entry answering to a key nothing on the host has.
///
/// Both say the same thing — "the one stored event these bytes belong to" —
/// spelled in the layout each backend actually has, so the index takes the
/// spelling as a type parameter.
pub(crate) trait EventIdentity: Copy + Eq {
    /// The identity of an entry already in hand.
    fn of(entry: &WarmEventEntry) -> Self;
    /// The start PTS this key sorts and searches under. Every identity has one:
    /// it is the index's sort key.
    fn start_pts_ns(self) -> u64;
}

/// The whole identity of one event, as an API request spells it: the composite
/// path segment `{start_pts_ns}_{duration_ms}_{event_type}`, e.g.
/// `81234000000_5200_movement`.
///
/// The read path used to name an event by its start PTS alone, which is not an
/// identity — a movement event and a continuous chunk can begin on the same
/// keyframe, and the lookup then served whichever of them a binary search
/// landed on. So the URL carries everything either backend's identity is made
/// of, and each [`WarmStorageBackend`](crate::storage::WarmStorageBackend)
/// narrows it to its own: local disk uses all three fields, the remote store
/// only the stem, deliberately (see its `find_event`).
///
/// The `{start}_{duration}` prefix is the remote store's object stem, and the
/// type is spelled as the event listing spells it ([`EventType::as_str`]), so a
/// key in a URL is readable against both the listing and the bucket.
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

    /// Parse one path segment. Every part must be there and be exactly what it
    /// claims: two integers that fit, and a type spelled the way the listing
    /// spells it. Anything else is `None` and answers `400` — a key that
    /// resolved partially would be the start-PTS-only lookup back again.
    ///
    /// Splitting from the left is what makes this unambiguous: no type name
    /// contains an underscore, so a segment with extra parts fails on the type
    /// rather than being silently truncated.
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

    fn start_pts_ns(self) -> u64 {
        self.0
    }
}

/// The remote store: every object of an event is built from one stem,
/// `{start_pts_ns}_{duration_ms}`, and the type is inside the sidecar.
impl EventIdentity for (u64, u32) {
    fn of(entry: &WarmEventEntry) -> Self {
        (entry.start_pts_ns, entry.duration_ms)
    }

    fn start_pts_ns(self) -> u64 {
        self.0
    }
}

/// Outcome of deleting one indexed event's stored objects. The three cases mean
/// different things to the index and to the caller's counters, and only
/// [`Removal::Deleted`] reclaimed any space.
pub(crate) enum Removal {
    /// The video is gone; its bytes are back.
    Deleted,
    /// The video was already absent. Nothing was reclaimed, but the index entry
    /// has to go too — it describes something that does not exist.
    Missing,
    /// Shutdown arrived before the deletion could be started, or between two of
    /// the requests it takes. Nothing is flagged and nothing is counted — the
    /// pass simply ends, because whatever was not deleted is not a fault of the
    /// store's and the next sweep (or the next start) finds it exactly as it
    /// was.
    ///
    /// Distinct from [`Failed`](Self::Failed) because that flag is read by
    /// eviction, which demotes what carries it: a store that was working
    /// perfectly must not come back from a restart-free shutdown with its
    /// oldest events marked as having resisted deletion.
    Abandoned,
    /// The store refused or could not be reached. The entry stays indexed so a
    /// later pass retries it instead of leaking the objects.
    ///
    /// The cost is deliberate: such an event stays listed, and stays offered
    /// for playback, indefinitely past its configured retention — for as long
    /// as the deletion keeps failing. A visible retention violation an operator
    /// can see and act on beats a file that is gone from the index, still
    /// eating space, and never retried by anything.
    Failed,
}

/// What one deletion pass achieved. The three counts are distinct outcomes with
/// distinct operator meanings — "nothing to delete", "deletions are failing",
/// and "someone else already reclaimed it" all produce zero deleted events and
/// call for different reactions.
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

/// Position of the entry with this key. Entries are sorted by start PTS, which
/// repeats across types and durations, so the run of equal starts is walked
/// rather than trusting a single binary-search hit.
fn position<K: EventIdentity>(entries: &[WarmEventEntry], key: K) -> Option<usize> {
    let start = key.start_pts_ns();
    let from = entries.partition_point(|e| e.start_pts_ns < start);
    entries[from..]
        .iter()
        .take_while(|e| e.start_pts_ns == start)
        .position(|e| K::of(e) == key)
        .map(|i| from + i)
}

/// The per-camera event lists both backends index into, keyed by `K`.
///
/// Every list is sorted by `start_pts_ns` and every entry is unique under `K`;
/// [`insert`](Self::insert) maintains both. `used_bytes` is maintained as the
/// sum of indexed `file_size` — the remote backend measures its storage budget
/// against it, and keeping it here rather than beside the budget is what stops
/// the two drifting apart.
pub(crate) struct EventIndex<K> {
    cameras: HashMap<String, RwLock<Vec<WarmEventEntry>>>,
    used_bytes: AtomicU64,
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
            _key: PhantomData,
        }
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

    /// Move tracked usage by the difference between what was indexed and what
    /// it replaced, in one atomic operation: an add followed by a subtract
    /// would let a concurrent reader — the budget guard runs on every camera's
    /// writer task — see a total inflated by the old entry's whole size and
    /// evict against it.
    ///
    /// Callers owe it the truth: `removed` must be bytes this index actually
    /// has charged. Nothing here checks that, and an over-large `removed` wraps
    /// the counter to near `u64::MAX` rather than clamping at zero — which the
    /// remote backend's budget would then read as a full store.
    fn charge(&self, added: u64, removed: u64) {
        if removed > added {
            self.used_bytes
                .fetch_sub(removed - added, Ordering::Relaxed);
        } else {
            self.used_bytes
                .fetch_add(added - removed, Ordering::Relaxed);
        }
    }

    /// Replace one camera's whole list, as a startup scan does. `entries` need
    /// not be sorted; the index's ordering invariant is restored here.
    pub(crate) fn replace_camera(&self, camera_id: &str, mut entries: Vec<WarmEventEntry>) {
        let Some(lock) = self.cameras.get(camera_id) else {
            return;
        };
        entries.sort_by_key(|e| e.start_pts_ns);
        let added: u64 = entries.iter().map(WarmEventEntry::stored_bytes).sum();
        let removed: u64 = {
            let mut slot = lock.write_recover();
            let removed = slot.iter().map(WarmEventEntry::stored_bytes).sum();
            *slot = entries;
            removed
        };
        self.charge(added, removed);
    }

    /// Index one event, replacing whatever entry already held its identity and
    /// returning it.
    ///
    /// A second entry under an identity that already has one would describe
    /// objects that do not exist: on the remote store a `PUT` is an upload *or*
    /// an update, and on local disk a re-written stem overwrites the file it
    /// names. Either way the storage holds one event, so the index holds one
    /// entry — and the byte total moves by the difference rather than counting
    /// the same bytes twice.
    pub(crate) fn insert(&self, camera_id: &str, entry: WarmEventEntry) -> Option<WarmEventEntry> {
        let lock = self.cameras.get(camera_id)?;
        let added = entry.stored_bytes();
        let replaced = Self::insert_locked(&mut lock.write_recover(), entry);
        self.charge(
            added,
            replaced.as_ref().map_or(0, WarmEventEntry::stored_bytes),
        );
        replaced
    }

    /// Index one event only if nothing holds its identity yet, reporting
    /// whether it landed.
    ///
    /// For a rebuild that runs while the write path is live: there, what is
    /// already indexed was put there by this process from what it just uploaded
    /// and is newer than anything a listing taken seconds ago can say, so the
    /// rebuild must yield to it rather than overwrite it. The test and the
    /// insertion are one locked step because the two racing writers are exactly
    /// what this is for — a `contains` followed by an `insert` can be overtaken
    /// between them, and the entry that loses is the fresh one.
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
            Self::insert_locked(&mut entries, entry);
        }
        self.charge(added, 0);
        true
    }

    /// [`insert`](Self::insert)'s list surgery, without the byte accounting, so
    /// [`reidentify`](Self::reidentify) can re-place an entry under the write
    /// lock it is already holding.
    fn insert_locked(
        entries: &mut Vec<WarmEventEntry>,
        entry: WarmEventEntry,
    ) -> Option<WarmEventEntry> {
        match position(entries, K::of(&entry)) {
            Some(i) => Some(std::mem::replace(&mut entries[i], entry)),
            None => {
                let pos = entries.partition_point(|e| e.start_pts_ns < entry.start_pts_ns);
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

    /// Mutate the entry with this key in place. `None` when no such event is
    /// indexed. The sort key never changes, and neither may `file_size` — a
    /// resize is a different set of bytes and goes through
    /// [`insert`](Self::insert), which is what keeps `used_bytes` honest.
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
    /// movement→object upgrade, which rewrites the very field local disk's
    /// identity is partly made of. `false` when no such event is indexed.
    ///
    /// The entry is taken out and re-placed rather than mutated where it lies,
    /// so it answers to the identity it now has and the index keeps one entry
    /// per identity. Anything already holding the new identity is displaced,
    /// which is what the storage did too: an upgrade that renames its stem over
    /// an existing object leaves one stored event, so the index keeps one entry
    /// and refunds the bytes the overwrite cost.
    ///
    /// Addressing this by start PTS alone — which nothing enforces the
    /// uniqueness of — is the bug this replaced: a binary search on the start
    /// returns an arbitrary member of the run, so an upgrade could reclassify a
    /// sibling event and leave the one it named untouched. See
    /// [`EventIdentity`].
    ///
    /// Unlike [`update`](Self::update), `f` *may* change what the entry weighs
    /// ([`WarmEventEntry::stored_bytes`]): the entry is re-placed rather than
    /// edited where it lies, so the accounting below can follow the resize.
    /// What it must not do is lie — what it leaves behind is what `used_bytes`
    /// will count, so it has to be the size of the objects the store now holds
    /// under the new identity.
    ///
    /// `f` runs on a copy, and the list is touched only once it has returned.
    /// A panic inside it therefore leaves the index exactly as it was, bytes
    /// and entries both — the single-step property poison recovery rests on
    /// (see [`crate::locks::LockExt`]).
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

    /// [`reidentify`](Self::reidentify) for a mutation that decides, once it
    /// has the entry in front of it, whether to happen at all — returning
    /// `false` to leave the index exactly as it was.
    ///
    /// The decision has to be made *inside* the write lock, which is what this
    /// exists for: the callers are repairs applied to an entry a concurrent
    /// live write may have already made the repair unnecessary for, and a look
    /// followed by a separate mutation can be overtaken between the two.
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
            (old_size, new_size, Self::insert_locked(&mut entries, entry))
        };
        // What the entry now weighs, against what left the index: its own
        // former size — it was re-placed, not added to — plus anything it
        // displaced. With the size unchanged this is exactly the displaced
        // refund.
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

    /// Every event overlapping `[from_ns, to_ns]`. An inverted range is empty.
    ///
    /// Entries are ordered by start PTS only, so the upper bound binary-searches
    /// but the lower one cannot: a long event (a continuous chunk) can start
    /// far before the window and still reach into it, and "ends after `from_ns`"
    /// is not monotone in start order. The candidate prefix is filtered instead.
    pub(crate) fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        if from_ns > to_ns {
            return Vec::new();
        }
        let Some(lock) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        let entries = lock.read_recover();
        let end = entries.partition_point(|e| e.start_pts_ns <= to_ns);
        entries[..end]
            .iter()
            .filter(|e| e.end_pts_ns() >= from_ns)
            .cloned()
            .collect()
    }

    /// The entry with this key, for the API read path.
    ///
    /// Keyed like every other lookup here, and for the same reason: two events
    /// can share a start PTS, so a search on the start alone would offer an
    /// arbitrary one of them for playback — the wrong recording, its own
    /// duration and thumbnails included.
    pub(crate) fn find(&self, camera_id: &str, key: K) -> Option<WarmEventEntry> {
        let entries = self.cameras.get(camera_id)?.read_recover();
        position(&entries, key).map(|i| entries[i].clone())
    }

    /// End of the newest indexed event, in wall-clock nanoseconds. Entries are
    /// kept sorted by start, so the last one is the newest.
    pub(crate) fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        let entries = self.cameras.get(camera_id)?.read_recover();
        entries.last().map(WarmEventEntry::end_pts_ns)
    }

    /// This camera's events past their retention, oldest first and already
    /// capped to this sweep's share (see [`cap_sweep_deletions`]).
    ///
    /// `max_age` is asked per entry rather than per type: the remote backend
    /// measures an event whose type it could not read against the longest
    /// configured retention instead of its placeholder's.
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

/// Delete one camera's expired events and unindex what actually went.
///
/// This is the shared half of a retention sweep: the failure handling, which is
/// the same on both backends and has to stay that way. A refused delete keeps
/// its entry — the objects are still stored, so the next sweep must see them
/// again — and is flagged so the per-sweep cap stops being spent on it. A
/// delete that found nothing there unindexes like a successful one but
/// reclaimed nothing, and is counted separately. Neither ends the pass: the
/// events behind a poisoned one are the space this exists to reclaim.
///
/// `cancel` is polled between events. It is deliberately *not* the only place a
/// shutdown is noticed: one event can be several remote requests, and a flag
/// read only here would leave all of them to run after it went up. A backend
/// whose deletion is remote reads the flag between its own requests too and
/// reports [`Removal::Abandoned`], which ends the pass from the inside — the
/// entry stays, nothing is flagged, and nothing is counted, because a shutdown
/// is not the store refusing.
///
/// What makes stopping mid-event safe is each backend's deletion *order*, not
/// this poll: both are arranged so that whatever survives an interruption is
/// something a later pass or a later start can finish, and never a video that
/// has lost the record of what it is. See each backend's `remove`/`delete` for
/// which way round it goes and why.
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
            // The entry stays, unflagged and uncounted; the loop's own `cancel`
            // would end the pass on the next turn anyway, and ending it here
            // saves that turn.
            Removal::Abandoned => break,
        }
    }
    outcome
}

/// Eviction order, cheapest footage to lose first. Both space-pressure paths —
/// local disk's low-space guard and the remote store's byte budget — delete in
/// this order, and both are deliberately outside [`cap_sweep_deletions`]:
/// neither *trigger* is clock-derived, and running out of room stops recording
/// altogether. So a pass here can delete the very footage a held-back sweep is
/// holding.
///
/// The choice of victim within a tier is clock-derived even though the trigger
/// is not: it is oldest [`start_pts`](crate::buffer::GopSegment) first, and
/// that stamp is only as ordered as the clock that wrote it. On a box whose
/// clock reads 0 until NTP lands, everything recorded before it lands sorts
/// ahead of an archive that is genuinely years older, so sustained space
/// pressure eats the newest footage first while age expiry — which needs the
/// same clock — is inert. That is a known cost of recording through a wrong
/// clock rather than not recording at all; it is not repaired here, because
/// the alternative to evicting something under space pressure is stopping.
const EVICTION_TIERS: [EventType; 3] = [
    EventType::Continuous,
    EventType::Movement,
    EventType::Object,
];

/// How a space-pressure pass treats an event that has already refused to be
/// deleted, and what it says when one goes. The two backends fail differently
/// enough that these cannot be defaults.
pub(crate) struct EvictionPolicy {
    /// Never offer a flagged event again (local disk) rather than only demoting
    /// it to the back of its tier (the remote store).
    ///
    /// Local disk can afford to exclude because it steps over failures: a
    /// flagged file never blocks the rest, and re-attempting one the filesystem
    /// has refused costs a syscall to learn nothing on a path that runs ahead of
    /// every write. Excluding *and* stopping would starve the remote pass
    /// outright — an outage flags one candidate per pass, and the hourly sweep
    /// only ever retries events that are already age-expired, so once the store
    /// came back nothing under retention would be reclaimable and the budget
    /// would sit over its limit permanently.
    pub skip_failed: bool,
    /// One refused delete ends the whole pass (the remote store) rather than
    /// being stepped over (local disk).
    ///
    /// A refusal from a network store is an answer about the store, not about
    /// one poisoned object, so the next candidate is overwhelmingly likely to
    /// fail the same way — and each attempt can sit for a request timeout inline
    /// in a warm writer with a camera's recording waiting behind it. A local
    /// unlink that fails says something about one file, and stopping there would
    /// let the oldest few undeletable events starve every newer deletable one
    /// and stop recording outright.
    pub stop_on_failure: bool,
    /// What to log when an event is evicted. Space pressure on a disk and a
    /// full client-side budget are different situations to be told about.
    pub reason: &'static str,
}

/// Delete the oldest events, cheapest tier first, until `satisfied` reports the
/// pressure is gone or the candidates are exhausted.
///
/// A pass never ends on a failure *count*, only on [`EvictionPolicy`]'s
/// explicit `stop_on_failure`: counting failures would let a handful of
/// undeletable events at the head of the queue starve every deletion behind
/// them for good.
///
/// `tier_of` is asked rather than read off the entry because the remote backend
/// evicts an event whose type it could not establish with the objects — the
/// tier kept longest — where the entry's own `event_type` is only a placeholder.
///
/// `cancel` is the shutdown flag, polled between events exactly as
/// [`sweep_expired`] polls it and for a stronger reason: this pass runs *ahead
/// of a write*, on a camera's own writer task, which the drain is waiting on.
/// A pass that kept evicting after the flag went up would spend the drain's
/// budget deleting real stored footage to make room for an event the same
/// shutdown is about to abandon unwritten. Local disk passes a predicate that
/// never fires, which is the honest answer for a backend whose eviction is a
/// handful of unlinks — see [`crate::storage::contract`]'s third guarantee.
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
        // Oldest first, but everything that has already refused to be deleted
        // after everything that has not: with `skip_failed` there is nothing
        // flagged left to order, and without it the retries are reached only
        // once this tier's untried candidates are exhausted and the pressure is
        // still there — the point at which the pass would otherwise give up.
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
                // Shutdown, part-way through this event: unflagged, uncounted,
                // and the end of the pass.
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

/// Floor under the share, so an archive of a handful of events still expires in
/// one sweep. Losing four events is not the loss the cap exists to prevent, and
/// dribbling them out an hour at a time would only make retention look broken
/// on small installs.
const SWEEP_DELETE_FLOOR: usize = 4;

fn sweep_delete_limit(indexed: usize) -> usize {
    indexed.div_ceil(SWEEP_DELETE_SHARE).max(SWEEP_DELETE_FLOOR)
}

/// Hold back the tail of an over-large expiry, so no single sweep can empty an
/// archive.
///
/// An event's start time is wall clock at keyframe time and its age is
/// `now - start`, so a forward clock correction of J ages every stored event by
/// J at once. A box with no battery-backed RTC does exactly that on an ordinary
/// boot: systemd-timesyncd (and fake-hwclock) restore the clock they saved at
/// shutdown, so the box comes up as far behind as it was switched off and jumps
/// forward when NTP lands — J is the off-time, and a box switched off over a
/// weekend jumps a weekend. Once J reaches the retention window, every event
/// ever recorded is expired in the same sweep. That sweep used to delete the
/// lot, and report it at `info!`.
///
/// So one flat cap covers every expiry, on both backends: at most
/// [`sweep_delete_limit`] events per camera per sweep, oldest first, with a
/// `warn!` naming the counts whenever anything is held back. The cap is
/// deliberately blind to *why* an event expired. A clock jump, a shortened
/// retention and a long outage are indistinguishable from inside a sweep, and
/// every test that tries to tell them apart is a hole at the jump sizes it
/// guesses wrong about — "how far past due is it" leaves J up to 1.25 retention
/// windows uncapped, and "does anything recent survive" disengages the moment
/// the first post-jump event is recorded, which is within seconds.
///
/// Ordinary retention never comes near the cap: an hourly sweep of an R-day
/// retention expires 1/(24R) of an archive, under 4% at the one-day minimum
/// camon accepts. What does reach it is a real mass expiry — retention cut from
/// 30 days to 2, a camera whose whole archive aged out while it was offline —
/// and that still drains completely, a quarter of what is left per sweep (never
/// fewer than [`SWEEP_DELETE_FLOOR`]) until only what the retention keeps
/// remains. It takes a working day instead of one pass, and says so at `warn!`
/// the whole way down.
///
/// The space-pressure paths deliberately bypass it; see [`EVICTION_TIERS`].
///
/// Events whose deletion already failed do not count against the cap: they were
/// let through an earlier sweep's cap and are only being retried, and charging
/// them again would let a few undeletable events at the head of the queue
/// starve every deletion behind them for good. That relies on failures being
/// flagged ([`WarmEventEntry::delete_failed`]), which [`sweep_expired`] does for
/// both backends.
fn cap_sweep_deletions(
    camera_id: &str,
    indexed: usize,
    expired: Vec<WarmEventEntry>,
) -> Vec<WarmEventEntry> {
    let expired_count = expired.len();
    let limit = sweep_delete_limit(indexed);
    let mut budget = limit;
    let mut held_back = 0usize;
    // Filtering in place keeps the index's oldest-first order, so the oldest
    // footage goes first and a sweep cut short by shutdown has still deleted
    // the events nearest their retention.
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

    /// Entries are ordered by start PTS alone, and nothing makes that unique —
    /// so a lookup has to walk the whole run of equal starts and compare full
    /// keys. A binary search would land somewhere in the run and stop there,
    /// which is how the local upgrade used to reclassify a sibling event.
    #[test]
    fn position_finds_the_named_entry_within_a_run_of_equal_starts() {
        let entries = [
            entry(1000, EventType::Movement, 1000),
            entry(2000, EventType::Movement, 1000),
            entry(2000, EventType::Object, 1000),
            entry(2000, EventType::Movement, 2000),
            entry(3000, EventType::Movement, 1000),
        ];
        // Every member of the run is reachable, at either end and in the middle.
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(position(&entries, <(u64, EventType, u32)>::of(e)), Some(i));
        }
        // Absent: the right start with the wrong rest of the key, and a start
        // nothing has.
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

    /// The same walk under the remote store's identity, where the type is not
    /// part of the key and the duration is all that separates the run.
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

    /// The read path's lookup, under both identities: each member of a run of
    /// equal starts comes back as itself. The old `find` binary-searched the
    /// start alone and returned whichever member it landed on, so one of the
    /// two lookups below was always the wrong event.
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
        // A start nothing has, and a start something has under another key.
        assert!(local
            .find("cam", (2500, EventType::Movement, 1000))
            .is_none());
        assert!(local.find("cam", (2000, EventType::Object, 1000)).is_none());
        assert!(local
            .find("other", (2000, EventType::Movement, 1000))
            .is_none());

        // The remote store's identity: the duration separates the run.
        let remote: EventIndex<(u64, u32)> = EventIndex::new(&["cam".to_string()]);
        remote.insert("cam", sized(entry(2000, EventType::Movement, 1000), 30));
        remote.insert("cam", sized(entry(2000, EventType::Object, 2000), 40));
        assert_eq!(remote.find("cam", (2000, 1000)).unwrap().file_size, 30);
        assert_eq!(remote.find("cam", (2000, 2000)).unwrap().file_size, 40);
        assert!(remote.find("cam", (2000, 3000)).is_none());
    }

    /// The wire spelling of a key, both ways. A URL that resolved partially —
    /// a missing part, a type this build does not know — would be the start-only
    /// lookup back again, so nothing here is optional.
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

    /// `used_bytes` moves by the difference between what was indexed and what
    /// it replaced, in either direction.
    #[test]
    fn charge_moves_used_bytes_by_the_difference() {
        let index: EventIndex<(u64, EventType, u32)> = EventIndex::new(&[]);
        index.charge(100, 0);
        assert_eq!(index.used_bytes(), 100);
        // Replaced by something smaller, then something larger.
        index.charge(30, 100);
        assert_eq!(index.used_bytes(), 30);
        index.charge(80, 30);
        assert_eq!(index.used_bytes(), 80);
        // Replaced like for like.
        index.charge(80, 80);
        assert_eq!(index.used_bytes(), 80);
        // Removed.
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

    /// A `reidentify` onto an identity another entry already holds. The store
    /// wrote one event over the other, so the index keeps one entry — the
    /// re-placed one — and the displaced entry's bytes come back.
    #[test]
    fn reidentify_displaces_the_entry_holding_the_new_identity() {
        let index = index_with_cam();
        index.insert("cam", sized(entry(1000, EventType::Movement, 500), 300));
        index.insert("cam", sized(entry(1000, EventType::Object, 500), 70));
        assert_eq!(index.used_bytes(), 370);

        // The movement is upgraded onto the object sibling's identity.
        assert!(
            index.reidentify("cam", (1000, EventType::Movement, 500), |e| {
                e.event_type = EventType::Object;
            })
        );

        let entries = index.query("cam", 0, u64::MAX);
        assert_eq!(entries.len(), 1);
        // The survivor is the entry that moved, not the one it landed on.
        assert_eq!(entries[0].file_size, 300);
        assert_eq!(entries[0].event_type, EventType::Object);
        assert_eq!(index.used_bytes(), 300);
    }

    /// `insert_absent` yields to whatever holds the identity — bytes included,
    /// since an insert that did not happen must not be charged — and behaves
    /// like `insert` when nothing does.
    #[test]
    fn insert_absent_yields_to_the_entry_already_holding_the_identity() {
        let index = index_with_cam();
        assert!(index.insert_absent("cam", sized(entry(1000, EventType::Movement, 500), 300)));
        assert_eq!(index.used_bytes(), 300);

        assert!(!index.insert_absent("cam", sized(entry(1000, EventType::Movement, 500), 70)));
        let entries = index.query("cam", 0, u64::MAX);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_size, 300, "overwrote the indexed entry");
        assert_eq!(
            index.used_bytes(),
            300,
            "charged an insert that did not happen"
        );

        // A different identity is not the same event, whatever it shares.
        assert!(index.insert_absent("cam", sized(entry(1000, EventType::Object, 500), 70)));
        assert_eq!(index.query("cam", 0, u64::MAX).len(), 2);
        assert_eq!(index.used_bytes(), 370);
        assert!(!index.insert_absent("other", sized(entry(1000, EventType::Movement, 500), 5)));
    }

    /// Nothing to displace, and a closure that changes nothing: the entry and
    /// the byte total both stay put.
    #[test]
    fn reidentify_with_a_no_op_closure_leaves_the_index_alone() {
        let index = index_with_cam();
        index.insert("cam", sized(entry(1000, EventType::Movement, 500), 300));

        assert!(index.reidentify("cam", (1000, EventType::Movement, 500), |_| {}));

        assert_eq!(index.query("cam", 0, u64::MAX).len(), 1);
        assert_eq!(index.used_bytes(), 300);
        assert!(!index.reidentify("cam", (2000, EventType::Movement, 500), |_| {}));
        assert!(!index.reidentify("other", (1000, EventType::Movement, 500), |_| {}));
        assert_eq!(index.used_bytes(), 300);
    }

    /// The closure may resize the entry it re-places, and `used_bytes` counts
    /// what it left behind — the old size *and* any displaced entry refunded,
    /// the new size charged.
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

        let entries = index.query("cam", 0, u64::MAX);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_size, 500);
        assert_eq!(index.used_bytes(), 500);
    }

    /// The shared sweep polls its cancel between events, which is what stops a
    /// pass that was already under way when the flag went up. Both backends'
    /// `prune` also check before the camera loop, so a sweep that starts
    /// stopped never reaches this — which is exactly why it is pinned here,
    /// against the skeleton, rather than only through a backend.
    #[tokio::test]
    async fn a_sweep_stops_between_events_when_it_is_cancelled_part_way() {
        let index = index_with_cam();
        let expired: Vec<WarmEventEntry> = (1..=4)
            .map(|i| sized(entry(i * 1000, EventType::Movement, 500), 10))
            .collect();
        for e in expired.iter() {
            index.insert("cam", e.clone());
        }

        // Runs on the second ask: one event is deleted, the rest are not.
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
        assert_eq!(index.query("cam", 0, u64::MAX).len(), 3);
    }

    /// The same for the eviction pass, which runs ahead of a write on a
    /// camera's own writer task — so a flag that goes up mid-pass has a drain
    /// waiting behind it.
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
        assert_eq!(index.query("cam", 0, u64::MAX).len(), 3);
    }

    /// A panic in the closure leaves the index exactly as it was: the entry is
    /// still there and still charged. `reidentify` builds the new value before
    /// touching the list, which is the property `LockExt`'s poison recovery is
    /// justified by.
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

        let entries = index.query("cam", 0, u64::MAX);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].event_type, EventType::Movement);
        assert_eq!(entries[0].file_size, 300);
        assert_eq!(index.used_bytes(), 300);
    }
}
