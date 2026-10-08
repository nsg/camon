// Warm event data shared by history, browsing, and playback.

let warmEvents = [];
let eventChains = new Map();
let warmEventPoller = null;
let warmEventsSignature = null;
let warmEventsMayHaveMore = false;

// Duration and type disambiguate events sharing a start PTS. Keep start_pts_ns as a string:
// epoch nanoseconds exceed JavaScript's exact integer range.
function eventKey(ev) {
    return `${ev.start_pts_ns}_${ev.duration_ms}_${ev.event_type}`;
}

const MAX_EVENT_PAGES = 10;
const INLINE_EVENT_PAGE_SIZE = 64;
const INLINE_EVENT_MAX_PAGES = 4;
const INLINE_EVENT_CARD_TARGET = 9;

// Slots in an event card's film: the most frames the server keeps per event.
const EVENT_FILM_SLOTS = 4;

const FRAME_PRELOAD_CONCURRENCY = 4;
// One FIFO bounds the live view's overlay image loads. Event card frames are
// lazy <img>s the browser paces itself and do not go through it.
const framePreloadQueue = [];
const framePreloadsInFlight = new Set();

function createImagePreloadOwner() {
    return { cancelled: false, loads: new Set() };
}

function finishImagePreload(entry, loaded) {
    if (entry.state !== 'loading') return;
    const loader = entry.loader;
    loader.onload = null;
    loader.onerror = null;
    entry.state = 'done';
    entry.loader = null;
    entry.owner.loads.delete(entry);
    framePreloadsInFlight.delete(entry);
    pumpImagePreloads();
    if (entry.owner.cancelled) return;
    (loaded ? entry.onLoad : entry.onError)(loader);
}

function pumpImagePreloads() {
    while (framePreloadsInFlight.size < FRAME_PRELOAD_CONCURRENCY && framePreloadQueue.length > 0) {
        const entry = framePreloadQueue.shift();
        if (entry.owner.cancelled) continue;
        entry.state = 'loading';
        entry.loader = new Image();
        framePreloadsInFlight.add(entry);
        entry.loader.onload = () => finishImagePreload(entry, true);
        entry.loader.onerror = () => finishImagePreload(entry, false);
        entry.loader.src = entry.url;
    }
}

function enqueueImagePreload(owner, url, onLoad, onError) {
    if (owner.cancelled) return;
    const entry = { owner, url, onLoad, onError, loader: null, state: 'queued' };
    owner.loads.add(entry);
    framePreloadQueue.push(entry);
    pumpImagePreloads();
}

function cancelImagePreloads(owner) {
    if (owner.cancelled) return;
    owner.cancelled = true;
    owner.loads.forEach(entry => {
        if (entry.state === 'queued') {
            const index = framePreloadQueue.indexOf(entry);
            if (index >= 0) framePreloadQueue.splice(index, 1);
        } else if (entry.state === 'loading') {
            entry.loader.onload = null;
            entry.loader.onerror = null;
            entry.loader.src = '';
            framePreloadsInFlight.delete(entry);
            entry.loader = null;
        }
        entry.state = 'cancelled';
    });
    owner.loads.clear();
    pumpImagePreloads();
}

function mapWarmEvents(events) {
    return events.map(ev => ({
        ...ev,
        key: eventKey(ev),
        start_ms: Number(BigInt(ev.start_pts_ns) / 1_000_000n),
    }));
}

function fetchWarmEvents(cameraId, { depth = 'inline' } = {}) {
    if (warmEventPoller && warmEventPoller.cameraId === cameraId &&
        warmEventPoller.depth === depth) {
        return warmEventPoller.first;
    }
    if (warmEventPoller) warmEventPoller.stop();
    warmEventsSignature = null;
    warmEventPoller = startPoller('warm events', 15000, async (signal) => {
        let raw = [];
        let cursor = null;
        let mayHaveMore = false;
        const pageLimit = depth === 'inline' ? INLINE_EVENT_PAGE_SIZE : null;
        const pageCap = depth === 'inline' ? INLINE_EVENT_MAX_PAGES : MAX_EVENT_PAGES;
        for (let page = 0; page < pageCap; page++) {
            const params = new URLSearchParams();
            if (cursor !== null) params.set('before', cursor);
            if (pageLimit !== null) params.set('limit', pageLimit);
            const query = params.toString();
            const url = `api/cameras/${encodeURIComponent(cameraId)}/events${query ? `?${query}` : ''}`;
            const response = await apiFetch(url, { signal });
            if (currentDetailCameraId !== cameraId || !response.ok) return;
            const events = await response.json();
            if (events.length === 0) {
                mayHaveMore = false;
                break;
            }
            raw = events.concat(raw);
            cursor = eventKey(events[0]);
            const shortPage = pageLimit !== null && events.length < pageLimit;
            if (shortPage) {
                mayHaveMore = false;
                break;
            }
            mayHaveMore = page + 1 === pageCap;
            if (depth === 'inline') {
                const partial = mapWarmEvents(raw);
                const partialChains = buildEventChains(partial);
                // A ninth collapsed card proves the eighth card's chain ended;
                // a short page or four-page cap also bounds the walk.
                if (collapseEventChains(partial, partialChains).length >= INLINE_EVENT_CARD_TARGET) {
                    mayHaveMore = true;
                    break;
                }
            }
        }
        const mapped = mapWarmEvents(raw);
        // Re-rendering identical data every poll would rebuild every card and
        // reload its frames, so unchanged results are dropped here.
        const signature = `${mayHaveMore}\n${mapped
            .map(ev => `${ev.key}:${ev.filmstrip_frames}:${ev.recovered}`)
            .join('\n')}`;
        if (signature === warmEventsSignature) return;
        warmEventsSignature = signature;
        warmEventsMayHaveMore = mayHaveMore;
        warmEvents = mapped;
        eventChains = buildEventChains(warmEvents);
        renderHistoryPanel();
        if (!eventsView.hidden) renderEventList();
        if (!playbackView.hidden) updatePlaybackNav();
    });
    warmEventPoller.cameraId = cameraId;
    warmEventPoller.depth = depth;
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
        if (run.length === 0) return;
        const members = run;
        const headKnown = !members[0].continues;
        const totalDurationMs = members.reduce((total, ev) => total + ev.duration_ms, 0);
        run.forEach((ev, i) => chains.set(ev.key, {
            part: i + 1,
            length: members.length,
            headKnown,
            members,
            totalDurationMs,
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

function collapseEventChains(events, chains = eventChains) {
    const collapsed = [];
    const seen = new Set();
    events.forEach(ev => {
        const chain = chains.get(ev.key);
        const members = chain ? chain.members : [ev];
        if (seen.has(members)) return;
        seen.add(members);
        // A detection may fire on any chunk of a capped run, and recovery marks
        // the chunk that was salvaged — so type, classes, and the warning must
        // aggregate over members, not mirror the head.
        const isObject = members.some(member => member.event_type === 'object');
        const objectClasses = [];
        members.forEach(member => (member.object_classes || []).forEach(name => {
            if (!objectClasses.includes(name)) objectClasses.push(name);
        }));
        collapsed.push({
            event: members[0],
            members,
            headKnown: chain ? chain.headKnown : !ev.continues,
            totalDurationMs: chain
                ? chain.totalDurationMs
                : members.reduce((total, member) => total + member.duration_ms, 0),
            eventType: isObject ? 'object' : members[0].event_type,
            objectClasses,
            recovered: members.some(member => member.recovered),
        });
    });
    return collapsed;
}

// Do not number a run whose head has already expired.
function chainPartLabel(ev) {
    const chain = eventChains.get(ev.key);
    if (!chain) return '';
    if (!chain.headKnown) return ' · continued';
    return chain.length > 1 ? ` · part ${chain.part}` : '';
}

function formatEventClock(startMs, includeSeconds) {
    const date = new Date(startMs);
    const parts = [date.getHours(), date.getMinutes()];
    if (includeSeconds) parts.push(date.getSeconds());
    return parts.map(part => String(part).padStart(2, '0')).join(':');
}

function formatEventDuration(durationMs) {
    const seconds = Math.round(durationMs / 1000);
    if (seconds < 100) return `${seconds} s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 100) return `${minutes} m ${String(seconds % 60).padStart(2, '0')} s`;
    return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} m`;
}

function eventObjectIcon(classes) {
    const iconClass = classes.find(name => {
        const normalized = String(name).toLowerCase();
        return normalized === 'person' || normalized === 'car' || normalized === 'vehicle';
    });
    if (!iconClass) return '';
    if (String(iconClass).toLowerCase() === 'person') {
        return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.5 5.5c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zM9.8 8.9L7 23h2.1l1.8-8 2.1 2v6h2v-7.5l-2.1-2 .6-3C14.8 12 16.8 13 19 13v-2c-1.9 0-3.5-1-4.3-2.4l-1-1.6c-.4-.6-1-1-1.7-1-.3 0-.5.1-.8.1L6 8.3V13h2V9.6l1.8-.7"/></svg>`;
    }
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/></svg>`;
}

function buildEventCard(collapsed, frameSize) {
    const ev = collapsed.event;
    const card = document.createElement('div');
    card.className = 'event-card';
    card.setAttribute('role', 'link');
    card.setAttribute('tabindex', '0');

    const cameraId = encodeURIComponent(currentDetailCameraId);
    const key = encodeURIComponent(ev.key);
    const frameCount = Math.min(EVENT_FILM_SLOTS, Math.max(0, Number(ev.filmstrip_frames) || 0));
    // An event with no filmstrip shows its thumbnail in the first slot.
    const frameUrls = frameCount > 0
        ? Array.from({ length: frameCount }, (_, i) =>
            authUrl(`api/cameras/${cameraId}/events/${key}/filmstrip/${i}?${frameSize}`))
        : [authUrl(`api/cameras/${cameraId}/events/${key}/thumbnail?${frameSize}`)];
    const timeStr = formatEventClock(ev.start_ms, true);
    const durationStr = formatEventDuration(collapsed.totalDurationMs) +
        (collapsed.members.length > 1 ? ' total' : '');
    const typeClass = eventTypeClass({ event_type: collapsed.eventType });
    const objectClasses = collapsed.objectClasses;
    const objectLabel = objectClasses.length > 0
        ? objectClasses.map(name => {
            const value = String(name);
            return value.charAt(0).toUpperCase() + value.slice(1);
        }).join(', ')
        : 'Object';
    const typeLabel = collapsed.eventType === 'object'
        ? objectLabel
        : collapsed.eventType === 'continuous' ? 'Continuous' : eventTypeLabel(ev);
    const typeIcon = collapsed.eventType === 'object' ? eventObjectIcon(objectClasses) : '';

    const recoveredBadge = collapsed.recovered
        ? ` <span class="event-recovered-badge" title="Recovered after an interruption — footage may be truncated">⚠</span>`
        : '';
    // "+" marks a run whose earliest chunks have already expired from storage.
    const clipCount = collapsed.members.length;
    const clipLabel = collapsed.headKnown
        ? `${clipCount} clips`
        : clipCount > 1 ? `${clipCount}+ clips` : 'continued';
    const chainBadge = !collapsed.headKnown || clipCount > 1
        ? `<div class="event-card-clips">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
            ${clipLabel}
        </div>`
        : '';
    // Every card has the same slots, so a short filmstrip leaves the last ones empty.
    const slots = Array.from({ length: EVENT_FILM_SLOTS }, (_, i) => i < frameUrls.length
        ? '<div class="event-card-frame"><img loading="lazy" alt="" draggable="false"></div>'
        : '<div class="event-card-frame empty"></div>').join('');

    card.innerHTML = `
        <div class="event-card-head">
            <span class="event-card-time">${timeStr}</span>
            <div class="event-card-badge ${esc(typeClass)}">${typeIcon}<span class="event-card-badge-label">${esc(typeLabel)}${recoveredBadge}</span></div>
            ${chainBadge}
            <span class="event-card-duration">${durationStr}</span>
        </div>
        <div class="event-card-film">${slots}</div>
    `;

    card.querySelectorAll('.event-card-frame img').forEach((image, i) => {
        // A frame that cannot be loaded empties its slot instead of showing a broken image.
        image.addEventListener('error', () => {
            image.parentElement.classList.add('empty');
            image.remove();
        });
        image.src = frameUrls[i];
    });

    const openEvent = () => {
        window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${ev.key}`;
    };
    card.addEventListener('click', openEvent);
    card.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        openEvent();
    });

    return card;
}

function formatQuietGap(durationMs) {
    if (durationMs >= 120 * 60 * 1000) {
        return `${Math.round(durationMs / (60 * 60 * 1000))} h quiet`;
    }
    return `${Math.round(durationMs / (60 * 1000))} min quiet`;
}

function appendEventCards(container, collapsedEvents, includeQuietGaps) {
    // A film's slots share the container's width, and a hidden container has none yet.
    const frameSize = frameSizeQuery(
        (container.clientWidth || window.innerWidth) / EVENT_FILM_SLOTS);
    collapsedEvents.forEach((collapsed, index) => {
        if (includeQuietGaps && index > 0) {
            const newer = collapsedEvents[index - 1];
            const quietMs = newer.event.start_ms -
                (collapsed.event.start_ms + collapsed.totalDurationMs);
            if (quietMs >= 10 * 60 * 1000) {
                const gap = document.createElement('div');
                gap.className = 'event-quiet-gap';
                gap.textContent = formatQuietGap(quietMs);
                container.appendChild(gap);
            }
        }
        container.appendChild(buildEventCard(collapsed, frameSize));
    });
}
