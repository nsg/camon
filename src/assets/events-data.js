// Warm event data shared by history, browsing, and playback.

let warmEvents = [];
let eventChains = new Map();
let warmEventPoller = null;

// Duration and type disambiguate events sharing a start PTS. Keep start_pts_ns as a string:
// epoch nanoseconds exceed JavaScript's exact integer range.
function eventKey(ev) {
    return `${ev.start_pts_ns}_${ev.duration_ms}_${ev.event_type}`;
}

// Stop only on an empty page: the client must not assume the server's page size.
// The cap bounds polling and rendering cost; deeper archive history is omitted.
const MAX_EVENT_PAGES = 10;

function fetchWarmEvents(cameraId) {
    if (warmEventPoller) warmEventPoller.stop();
    warmEventPoller = startPoller('warm events', 15000, async (signal) => {
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
        if (!eventsView.hidden) renderEventList();
        if (!playbackView.hidden) updatePlaybackNav();
    });
    return warmEventPoller.first;
}

// Show unknown wire names verbatim so a newer event type is not mislabeled.
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

// `continues` may outlive its predecessor, so adjacency must also match before chunks join.
// Tolerance absorbs millisecond rounding of otherwise contiguous boundaries.
const CHUNK_GAP_TOLERANCE_MS = 2000;

function buildEventChains(events) {
    const chains = new Map();
    const ascending = [...events].sort((a, b) => a.start_ms - b.start_ms);
    let run = [];
    const flush = () => {
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

// Do not number a run whose head has already expired.
function chainPartLabel(ev) {
    const chain = eventChains.get(ev.key);
    if (!chain) return '';
    if (!chain.headKnown) return ' · continued';
    return chain.length > 1 ? ` · part ${chain.part}` : '';
}

function isChunkOfRun(ev) {
    const chain = eventChains.get(ev.key);
    return !!chain && (chain.length > 1 || !chain.headKnown);
}
