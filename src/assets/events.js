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

    if (currentDetailCameraId !== cameraId) {
        currentDetailCameraId = cameraId;
        fetchWarmEvents(cameraId);
    }

    hideAllViews();
    eventsView.hidden = false;
    eventsCameraName.textContent = cameraId;

    eventFilter = 'all';
    filterBtns.forEach(b => b.classList.toggle('active', b.dataset.filter === 'all'));

    renderEventList();
}

function renderEventList() {
    eventList.innerHTML = '';

    const collapsed = collapseEventChains(warmEvents);
    // A capped run can change type mid-way, so a card matches a filter when
    // ANY of its chunks does — filtering on the head would hide a detection
    // that fired on a later chunk.
    const filtered = eventFilter === 'all'
        ? collapsed
        : collapsed.filter(run => run.members.some(m => m.event_type === eventFilter));

    if (filtered.length === 0) {
        eventList.innerHTML = '<div class="event-list-empty">No events found</div>';
        return;
    }

    const sorted = [...filtered].sort((a, b) => b.event.start_ms - a.event.start_ms);

    const groups = new Map();
    sorted.forEach(run => {
        const label = formatDateLabel(new Date(run.event.start_ms));
        if (!groups.has(label)) groups.set(label, []);
        groups.get(label).push(run);
    });

    groups.forEach((events, label) => {
        const groupEl = document.createElement('div');
        groupEl.className = 'event-day-group';
        groupEl.dataset.day = label;

        const dayHeader = document.createElement('div');
        dayHeader.className = 'event-day-header';

        const dayLabel = document.createElement('div');
        dayLabel.className = 'event-day-label';
        dayLabel.textContent = label;
        dayHeader.appendChild(dayLabel);

        const oldest = events[events.length - 1].event.start_ms;
        const newest = events[0].event.start_ms;
        const summary = document.createElement('div');
        summary.className = 'event-day-summary';
        summary.textContent = `${events.length} events · ${formatEventClock(oldest, false)} – ${formatEventClock(newest, false)}`;
        dayHeader.appendChild(summary);
        groupEl.appendChild(dayHeader);

        appendEventCards(groupEl, events, true);

        eventList.appendChild(groupEl);
    });

    if (eventsScrollDay) {
        const target = eventList.querySelector(`.event-day-group[data-day="${CSS.escape(eventsScrollDay)}"]`);
        eventsScrollDay = null;
        if (target) target.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
}
