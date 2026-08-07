// View 2: Event Browser.

const eventsView = document.getElementById('events-view');
const eventsBackBtn = document.getElementById('events-back-btn');
const eventsCameraName = document.getElementById('events-camera-name');
const eventList = document.getElementById('event-list');
const filterBtns = document.querySelectorAll('.filter-btn');

let eventFilter = 'all';
let eventsScrollDay = null;

function wireEventsView() {
    eventsBackBtn.addEventListener('click', () => {
        if (currentDetailCameraId) {
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}`;
        }
    });

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            eventFilter = btn.dataset.filter;
            renderEventList();
        });
    });
}

function showEventsView(cameraId) {
    cleanupPlaybackView();
    cleanupDebugView();

    // Ensure we have warm events loaded
    if (currentDetailCameraId !== cameraId) {
        currentDetailCameraId = cameraId;
        fetchWarmEvents(cameraId);
    }

    hideAllViews();
    eventsView.hidden = false;
    eventsCameraName.textContent = cameraId;

    // Reset filter
    eventFilter = 'all';
    filterBtns.forEach(b => b.classList.toggle('active', b.dataset.filter === 'all'));

    renderEventList();
}

// === View 2: Event List Rendering ===

function renderEventList() {
    eventList.innerHTML = '';

    // One filter per event type, matched on the wire name, so no type can
    // fall through every filter the way continuous chunks used to.
    const filtered = eventFilter === 'all'
        ? warmEvents
        : warmEvents.filter(e => e.event_type === eventFilter);

    if (filtered.length === 0) {
        eventList.innerHTML = '<div class="event-list-empty">No events found</div>';
        return;
    }

    // Sort by time descending
    const sorted = [...filtered].sort((a, b) => b.start_ms - a.start_ms);

    // Group by date
    const groups = new Map();
    sorted.forEach(ev => {
        const label = formatDateLabel(new Date(ev.start_ms));
        if (!groups.has(label)) groups.set(label, []);
        groups.get(label).push(ev);
    });

    groups.forEach((events, label) => {
        const groupEl = document.createElement('div');
        groupEl.className = 'event-day-group';
        groupEl.dataset.day = label;

        const dayLabel = document.createElement('div');
        dayLabel.className = 'event-day-label';
        dayLabel.textContent = label;
        groupEl.appendChild(dayLabel);

        events.forEach(ev => {
            const item = document.createElement('div');
            item.className = 'event-list-item';
            // Chunks of one long recording get a rail down their left edge
            // tying them together; the list runs newest-first, so a chunk
            // continues the one below it.
            if (isChunkOfRun(ev)) item.classList.add('chain-part');

            const thumbSrc = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${ev.key}/thumbnail`);
            const evDate = new Date(ev.start_ms);
            const timeStr = evDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const durSec = (ev.duration_ms / 1000).toFixed(1);
            const typeLabel = eventTypeLabel(ev) + chainPartLabel(ev);
            const typeClass = eventTypeClass(ev);

            // Salvaged from an interrupted write at startup: flag it so the
            // viewer knows the tail may be cut short.
            const recoveredBadge = ev.recovered
                ? ` <span class="event-recovered-badge" title="Recovered after an interruption — footage may be truncated">⚠</span>`
                : '';

            // A motion run is subsampled to at most 4 filmstrip thumbs, and
            // short runs yield fewer — render exactly the frames that exist so
            // we never request a missing index (which 404s to a broken glyph).
            // The onerror handler hides any frame that still fails to load
            // (e.g. pre-count events or a partial write).
            let thumbHtml;
            if (ev.filmstrip_frames > 0) {
                const cid = encodeURIComponent(currentDetailCameraId);
                thumbHtml = `<div class="event-filmstrip">` +
                    Array.from({ length: ev.filmstrip_frames }, (_, i) => `<img class="filmstrip-frame" src="${esc(authUrl(`api/cameras/${cid}/events/${ev.key}/filmstrip/${i}`))}" loading="lazy" alt="" onerror="this.style.display='none'">`).join('') +
                    `</div>`;
            } else {
                thumbHtml = `<img class="event-list-thumb" src="${esc(thumbSrc)}" loading="lazy" alt="">`;
            }

            // Nothing about the chain goes here: the type line already says
            // "part 3" or "continued", and in continuous recording every
            // row is a continuation, so repeating it distinguishes nothing.
            const detailText = ev.event_type === 'object' && ev.object_classes ? ev.object_classes.join(', ') : '';

            item.innerHTML = `
                ${thumbHtml}
                <div class="event-list-info">
                    <div class="event-list-type ${esc(typeClass)}">${esc(typeLabel)}${recoveredBadge}</div>
                    <div class="event-list-detail">${esc(detailText)}</div>
                </div>
                <div class="event-list-meta">
                    <div class="event-list-time">${timeStr}</div>
                    <div class="event-list-duration">${durSec}s</div>
                </div>
                <div class="event-list-chevron">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/></svg>
                </div>
            `;

            item.addEventListener('click', () => {
                window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${ev.key}`;
            });

            groupEl.appendChild(item);
        });

        eventList.appendChild(groupEl);
    });

    // Arriving from a history day row: jump to that day's section.
    if (eventsScrollDay) {
        const target = eventList.querySelector(`.event-day-group[data-day="${CSS.escape(eventsScrollDay)}"]`);
        eventsScrollDay = null;
        if (target) target.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
}
