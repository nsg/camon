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

    const filtered = eventFilter === 'all'
        ? warmEvents
        : warmEvents.filter(e => e.event_type === eventFilter);

    if (filtered.length === 0) {
        eventList.innerHTML = '<div class="event-list-empty">No events found</div>';
        return;
    }

    const sorted = [...filtered].sort((a, b) => b.start_ms - a.start_ms);

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
            groupEl.appendChild(buildEventListItem(ev));
        });

        eventList.appendChild(groupEl);
    });

    if (eventsScrollDay) {
        const target = eventList.querySelector(`.event-day-group[data-day="${CSS.escape(eventsScrollDay)}"]`);
        eventsScrollDay = null;
        if (target) target.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
}
