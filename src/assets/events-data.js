// Warm event data shared by history, browsing, and playback.

let warmEvents = [];
let eventChains = new Map();
let warmEventPoller = null;
let warmEventsSignature = null;

// Duration and type disambiguate events sharing a start PTS. Keep start_pts_ns as a string:
// epoch nanoseconds exceed JavaScript's exact integer range.
function eventKey(ev) {
    return `${ev.start_pts_ns}_${ev.duration_ms}_${ev.event_type}`;
}

// Stop only on an empty page: the client must not assume the server's page size.
// The cap bounds polling and rendering cost; deeper archive history is omitted.
const MAX_EVENT_PAGES = 10;

const FRAME_CYCLE_MS = 1000;
const eventCardCycleSubscribers = new Set();
const eventCardControllers = new WeakMap();
const prefersReducedMotion = typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let eventCardCycleTimer = null;

function stopEventCardCycle() {
    if (eventCardCycleTimer === null) return;
    clearInterval(eventCardCycleTimer);
    eventCardCycleTimer = null;
}

function startEventCardCycle() {
    if (eventCardCycleTimer !== null || eventCardCycleSubscribers.size === 0 || document.hidden) {
        return;
    }
    eventCardCycleTimer = setInterval(() => {
        eventCardCycleSubscribers.forEach(controller => {
            if (controller.card.isConnected) return;
            eventCardCycleSubscribers.delete(controller);
            eventCardIntersectionObserver.unobserve(controller.card);
        });
        if (eventCardCycleSubscribers.size === 0) {
            stopEventCardCycle();
            return;
        }
        eventCardCycleSubscribers.forEach(controller => controller.advanceFrame());
    }, FRAME_CYCLE_MS);
}

function subscribeEventCard(controller) {
    eventCardCycleSubscribers.add(controller);
    startEventCardCycle();
}

function unsubscribeEventCard(controller) {
    eventCardCycleSubscribers.delete(controller);
    if (eventCardCycleSubscribers.size === 0) stopEventCardCycle();
}

const eventCardIntersectionObserver = !prefersReducedMotion &&
    typeof IntersectionObserver !== 'undefined'
    ? new IntersectionObserver(entries => {
        entries.forEach(entry => {
            const controller = eventCardControllers.get(entry.target);
            if (!controller) return;
            if (!controller.card.isConnected) {
                unsubscribeEventCard(controller);
                eventCardIntersectionObserver.unobserve(controller.card);
            } else if (entry.isIntersecting) {
                controller.preloadFrames();
                subscribeEventCard(controller);
            } else {
                unsubscribeEventCard(controller);
            }
        });
    }, { threshold: 0.25 })
    : null;

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopEventCardCycle();
    } else {
        startEventCardCycle();
    }
});

function fetchWarmEvents(cameraId) {
    if (warmEventPoller) warmEventPoller.stop();
    warmEventsSignature = null;
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
        const mapped = raw.map(ev => ({
            ...ev,
            key: eventKey(ev),
            start_ms: Number(BigInt(ev.start_pts_ns) / 1_000_000n),
        }));
        // Re-rendering identical data every poll would wipe in-progress
        // filmstrip scrubbing, so unchanged results are dropped here.
        const signature = mapped
            .map(ev => `${ev.key}:${ev.filmstrip_frames}:${ev.recovered}`)
            .join('\n');
        if (signature === warmEventsSignature) return;
        warmEventsSignature = signature;
        warmEvents = mapped;
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

function collapseEventChains(events) {
    const collapsed = [];
    const seen = new Set();
    events.forEach(ev => {
        const chain = eventChains.get(ev.key);
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

function buildEventCard(collapsed) {
    const ev = collapsed.event;
    const card = document.createElement('div');
    const isHero = collapsed.eventType === 'object';
    card.className = `event-card ${isHero ? 'event-card-hero' : 'event-card-tile'}`;
    card.setAttribute('role', 'link');
    card.setAttribute('tabindex', '0');

    const cameraId = encodeURIComponent(currentDetailCameraId);
    const key = encodeURIComponent(ev.key);
    const thumbnailUrl = authUrl(`api/cameras/${cameraId}/events/${key}/thumbnail`);
    const frameCount = Math.max(0, Number(ev.filmstrip_frames) || 0);
    const middleFrame = frameCount > 0 ? Math.floor(frameCount / 2) : -1;
    const frameUrls = Array.from({ length: frameCount }, (_, i) =>
        authUrl(`api/cameras/${cameraId}/events/${key}/filmstrip/${i}`));
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
    if (chainBadge) card.classList.add('has-chain-badge');
    const frameDots = frameCount >= 2
        ? `<div class="event-card-frames">${Array.from({ length: frameCount }, (_, i) =>
            `<i${i === middleFrame ? ' class="active"' : ''}></i>`).join('')}</div>`
        : '';

    card.innerHTML = `
        <img class="event-card-image" loading="lazy" alt="" draggable="false">
        <div class="event-card-fade"></div>
        <div class="event-card-badge ${esc(typeClass)}">${typeIcon}<span class="event-card-badge-label">${esc(typeLabel)}${recoveredBadge}</span></div>
        ${chainBadge}
        ${frameDots}
        <div class="event-card-meta">
            <span class="event-card-time">${timeStr}</span>
            <span class="event-card-duration">${durationStr}</span>
        </div>
    `;

    const image = card.querySelector('.event-card-image');
    const dots = [...card.querySelectorAll('.event-card-frames i')];
    const frameStatus = Array(frameCount).fill(0);
    let shownFrame = middleFrame;
    let desiredFrame = middleFrame;
    let lastGoodFrame = -1;
    let lastGoodSrc = '';
    let preloadingStarted = false;
    let touchStart = null;
    let suppressClick = false;
    let interacting = false;

    function updateDots(index) {
        dots.forEach((dot, i) => dot.classList.toggle('active', i === index));
    }

    function showFrame(index) {
        if (frameStatus[index] !== 1) return;
        shownFrame = index;
        image.hidden = false;
        image.src = frameUrls[index];
        updateDots(index);
    }

    function preloadFrames() {
        if (preloadingStarted) return;
        preloadingStarted = true;
        frameUrls.forEach((url, index) => {
            if (frameStatus[index] !== 0) return;
            const loader = new Image();
            loader.addEventListener('load', () => {
                frameStatus[index] = 1;
                if (desiredFrame === index) showFrame(index);
            });
            loader.addEventListener('error', () => { frameStatus[index] = -1; });
            loader.src = url;
        });
    }

    function scrubTo(clientX) {
        const rect = card.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(0.999999, (clientX - rect.left) / rect.width));
        desiredFrame = Math.floor(fraction * frameCount);
        showFrame(desiredFrame);
    }

    function advanceFrame() {
        if (interacting || touchStart !== null) return;
        const currentFrame = shownFrame >= 0 ? shownFrame : middleFrame;
        for (let offset = 1; offset < frameCount; offset++) {
            const index = (currentFrame + offset) % frameCount;
            if (frameStatus[index] !== 1) continue;
            showFrame(index);
            desiredFrame = shownFrame;
            return;
        }
        desiredFrame = shownFrame;
    }

    // Both handlers derive the frame index from the src that actually fired,
    // so a stale error from a superseded request cannot poison another frame.
    image.addEventListener('load', () => {
        const src = image.getAttribute('src');
        const index = frameUrls.indexOf(src);
        if (index >= 0) frameStatus[index] = 1;
        lastGoodFrame = index;
        lastGoodSrc = src;
    });
    image.addEventListener('error', () => {
        const src = image.getAttribute('src');
        const index = frameUrls.indexOf(src);
        if (index >= 0) frameStatus[index] = -1;
        if (src === lastGoodSrc) lastGoodSrc = '';
        if (!lastGoodSrc && src !== thumbnailUrl) {
            shownFrame = -1;
            updateDots(-1);
            image.src = thumbnailUrl;
        } else if (lastGoodSrc && src !== lastGoodSrc) {
            shownFrame = lastGoodFrame;
            updateDots(lastGoodFrame);
            image.src = lastGoodSrc;
        } else {
            image.hidden = true;
        }
    });
    image.src = middleFrame >= 0 ? frameUrls[middleFrame] : thumbnailUrl;

    if (frameCount >= 2) {
        card.addEventListener('pointerenter', () => {
            interacting = true;
            preloadFrames();
        });
        card.addEventListener('pointerdown', event => {
            preloadFrames();
            if (event.pointerType === 'touch') {
                interacting = true;
                touchStart = { x: event.clientX, y: event.clientY };
            }
        });
        card.addEventListener('pointermove', event => {
            scrubTo(event.clientX);
        });
        card.addEventListener('pointerleave', () => {
            interacting = false;
            desiredFrame = shownFrame;
        });
        card.addEventListener('pointerup', event => {
            if (!touchStart || event.pointerType !== 'touch') return;
            const dx = Math.abs(event.clientX - touchStart.x);
            const dy = Math.abs(event.clientY - touchStart.y);
            touchStart = null;
            interacting = false;
            desiredFrame = shownFrame;
            if (dx >= 12 && dx > dy) {
                suppressClick = true;
                setTimeout(() => { suppressClick = false; }, 400);
            }
        });
        card.addEventListener('pointercancel', () => {
            touchStart = null;
            interacting = false;
            desiredFrame = shownFrame;
        });

        if (eventCardIntersectionObserver) {
            const controller = { card, preloadFrames, advanceFrame };
            eventCardControllers.set(card, controller);
            eventCardIntersectionObserver.observe(card);
        }
    }

    const openEvent = () => {
        window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${ev.key}`;
    };
    card.addEventListener('click', () => {
        if (suppressClick) {
            suppressClick = false;
            return;
        }
        openEvent();
    });
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
    let grid = null;
    collapsedEvents.forEach((collapsed, index) => {
        if (includeQuietGaps && index > 0) {
            const newer = collapsedEvents[index - 1];
            const quietMs = newer.event.start_ms -
                (collapsed.event.start_ms + collapsed.totalDurationMs);
            if (quietMs >= 10 * 60 * 1000) {
                grid = null;
                const gap = document.createElement('div');
                gap.className = 'event-quiet-gap';
                gap.textContent = formatQuietGap(quietMs);
                container.appendChild(gap);
            }
        }

        const card = buildEventCard(collapsed);
        if (collapsed.eventType === 'object') {
            grid = null;
            container.appendChild(card);
            return;
        }
        if (!grid) {
            grid = document.createElement('div');
            grid.className = 'event-card-grid';
            container.appendChild(grid);
        }
        grid.appendChild(card);
    });
}
