// Warm event data, shared by the live view (history panel), the event
// browser, and playback (prev/next navigation).

// Warm events (shared between views 2 & 3). `eventChains` maps an event's
// key (see eventKey) to its place in a chain of `continues` chunks; see
// buildEventChains.
let warmEvents = [];
let eventChains = new Map();
let warmEventPoller = null;

// The key that identifies one stored event, everywhere: in the playback
// route, in the chain map, and in every event URL the server answers.
//
// A start PTS alone identifies nothing — two recordings can begin on the
// same keyframe (a motion event and the continuous chunk covering it), and
// asking for one of them by start used to serve whichever the server found
// first. So the key carries the duration and the type as well, in the
// spelling the API uses: `{start_pts_ns}_{duration_ms}_{event_type}`.
//
// Built by string concatenation, never by arithmetic: `start_pts_ns` arrives
// as a JSON string because nanoseconds since the epoch are past the range
// Number holds exactly, and a key that had been through Number would name an
// event the server does not have.
function eventKey(ev) {
    return `${ev.start_pts_ns}_${ev.duration_ms}_${ev.event_type}`;
}

// A listing answers with one bounded page of the newest events (the server
// used to answer with everything at once, which it did by cloning the whole
// index under the lock the camera's warm writer needs — on a deep archive
// one poll of this view stalled recording). So the archive is walked
// backwards: each request carries `before`, set to the key of the oldest
// event the previous page held, which resumes exactly beneath it even when
// the page ended inside a run of events sharing one start PTS.
//
// No page size is sent and none is assumed. The walk ends on an empty page,
// which costs one extra request per poll and — unlike reading a short page
// as the end — cannot be misread as the end of the archive if the server's
// cap is ever lowered below what this asked for. The ten-page horizon still
// scales with that cap either way: the panel reaches at most ten pages of
// whatever size the server serves. The first paint of the history panel
// waits on this sequential walk, one round trip per page.
//
// The walk stops after MAX_EVENT_PAGES either way, so a deep archive costs
// this view a bounded number of requests every poll and a bounded number of
// rows to render. What is dropped past that is the oldest end of a very deep
// archive: with continuous recording kept for a fortnight or more, the
// oldest days are simply absent from the history panel, silently. The event
// list is due a rework that will have to face this properly.
const MAX_EVENT_PAGES = 10;

function fetchWarmEvents(cameraId) {
    if (warmEventPoller) warmEventPoller.stop();
    warmEventPoller = startPoller('warm events', 15000, async (signal) => {
        // Pages arrive newest-first but each is oldest-first within itself,
        // so an older page goes in front of what is already collected.
        let raw = [];
        let cursor = null;
        for (let page = 0; page < MAX_EVENT_PAGES; page++) {
            const before = cursor === null ? '' : `?before=${encodeURIComponent(cursor)}`;
            const url = `api/cameras/${encodeURIComponent(cameraId)}/events${before}`;
            const response = await apiFetch(url, { signal });
            if (currentDetailCameraId !== cameraId || !response.ok) return;
            const events = await response.json();
            if (events.length === 0) break;
            raw = events.concat(raw);
            cursor = eventKey(events[0]);
        }
        warmEvents = raw.map(ev => ({
            ...ev,
            key: eventKey(ev),
            start_ms: Number(BigInt(ev.start_pts_ns) / 1_000_000n),
        }));
        eventChains = buildEventChains(warmEvents);
        renderHistoryPanel();
        // Re-render event list if visible
        if (!eventsView.hidden) renderEventList();
        // Update nav if in playback
        if (!playbackView.hidden) updatePlaybackNav();
    });
    // Callers that cannot render until the events are here await this.
    return warmEventPoller.first;
}

// The event types the API can send, exhaustively (see WarmEventResponse in
// src/api/server.rs). Calling everything that isn't an object "Movement"
// mislabelled continuous-recording chunks as something that was detected.
// An unknown type means a newer server talking to an older UI: show the
// wire name rather than quietly filing it under the wrong one.
const EVENT_TYPE_LABELS = {
    object: 'Object detected',
    movement: 'Movement',
    continuous: 'Continuous recording',
};

function eventTypeLabel(ev) {
    return EVENT_TYPE_LABELS[ev.event_type] || ev.event_type || 'Event';
}

function eventTypeClass(ev) {
    return EVENT_TYPE_LABELS[ev.event_type] ? ev.event_type : 'unknown';
}

// `continues` marks a chunk that carries on the one before it: a motion run
// split at max_event_duration_secs, or the fixed-length chunks continuous
// recording rolls. Walking the events oldest-first turns those flags into
// runs, so each chunk can say where it sits in one recording instead of
// standing there as an unrelated event.
//
// The flag alone does not identify the predecessor, and outliving it is the
// normal case rather than a corner: a movement chunk expires after
// movement_retention_days while the object chunk that continues it keeps
// the flag and the far longer object retention, and continuous chunks
// expire daily while the flag says "not the first chunk since startup"
// forever. So a chunk joins the run only if it also starts where the
// previous one ended — both producers emit strictly contiguous chunks (the
// follow-on begins at the segment after the last one written, with no
// pre-padding), which makes adjacency an exact test rather than a
// heuristic. A second of slack absorbs millisecond rounding.
const CHUNK_GAP_TOLERANCE_MS = 2000;

function buildEventChains(events) {
    // event key -> { part, headKnown }
    const chains = new Map();
    const ascending = [...events].sort((a, b) => a.start_ms - b.start_ms);
    let run = [];
    const flush = () => {
        // A run whose first chunk is itself flagged `continues` reaches back
        // past what the archive still holds: its head has been pruned, so
        // nothing here can say which part of the recording these are — only
        // that they carry one on. A run with its head intact can be counted
        // forward, but never totalled: the recording may still be growing.
        const headKnown = run.length > 0 && !run[0].continues;
        run.forEach((ev, i) => chains.set(ev.key, {
            part: i + 1,
            length: run.length,
            headKnown,
        }));
        run = [];
    };
    ascending.forEach(ev => {
        const prev = run[run.length - 1];
        const followsPrev = prev &&
            Math.abs(ev.start_ms - (prev.start_ms + prev.duration_ms)) < CHUNK_GAP_TOLERANCE_MS;
        if (!ev.continues || !followsPrev) flush();
        run.push(ev);
    });
    flush();
    return chains;
}

// "part 3" counts from a head we can actually see. "continued" is all that
// can honestly be said when we cannot: a number counted from the oldest
// surviving chunk would claim a position in the recording that it does not
// have, and would change on every retention sweep.
function chainPartLabel(ev) {
    const chain = eventChains.get(ev.key);
    if (!chain) return '';
    if (!chain.headKnown) return ' · continued';
    return chain.length > 1 ? ` · part ${chain.part}` : '';
}

// Whether this event is a chunk of a longer recording — either because more
// of that recording is on screen, or because it continues one that has been
// pruned away.
function isChunkOfRun(ev) {
    const chain = eventChains.get(ev.key);
    return !!chain && (chain.length > 1 || !chain.headKnown);
}
