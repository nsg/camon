const playbackView = document.getElementById('playback-view');
const playbackBackBtn = document.getElementById('playback-back-btn');
const playbackEventInfo = document.getElementById('playback-event-info');
const playbackVideo = document.getElementById('playback-video');
const playbackLoading = document.getElementById('playback-loading');
const playbackScrubber = document.getElementById('playback-scrubber');
const playbackProgressFill = document.getElementById('playback-progress-fill');
const playbackCurrentTime = document.getElementById('playback-current-time');
const playbackDuration = document.getElementById('playback-duration');
const playbackMuteBtn = document.getElementById('playback-mute-btn');
const prevEventBtn = document.getElementById('prev-event-btn');
const nextEventBtn = document.getElementById('next-event-btn');
const prevEventThumb = document.getElementById('prev-event-thumb');
const nextEventThumb = document.getElementById('next-event-thumb');
const prevEventText = document.getElementById('prev-event-text');
const nextEventText = document.getElementById('next-event-text');

let playbackHls = null;
let currentPlaybackKey = null;
let playbackAnimationId = null;
let isScrubbing = false;

function wirePlaybackView() {
    playbackBackBtn.addEventListener('click', () => {
        if (currentDetailCameraId) {
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events`;
        }
    });

    playbackScrubber.addEventListener('input', () => {
        isScrubbing = true;
        const duration = playbackVideo.duration;
        if (duration && isFinite(duration)) {
            const progress = playbackScrubber.value / 1000;
            playbackProgressFill.style.width = (progress * 100) + '%';
            playbackCurrentTime.textContent = formatTimeShort(progress * duration);
        }
    });

    playbackScrubber.addEventListener('change', () => {
        const duration = playbackVideo.duration;
        if (duration && isFinite(duration)) {
            playbackVideo.currentTime = (playbackScrubber.value / 1000) * duration;
        }
        isScrubbing = false;
    });

    playbackMuteBtn.addEventListener('click', () => {
        playbackVideo.muted = !playbackVideo.muted;
        updateMuteIcon(playbackMuteBtn, playbackVideo);
    });

    prevEventBtn.addEventListener('click', () => {
        const nav = getAdjacentEvents(currentPlaybackKey);
        if (nav.prev) {
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${nav.prev.key}`;
        }
    });

    nextEventBtn.addEventListener('click', () => {
        const nav = getAdjacentEvents(currentPlaybackKey);
        if (nav.next) {
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${nav.next.key}`;
        }
    });
}

function showPlaybackView(cameraId, key) {
    cleanupDebugView();

    if (currentDetailCameraId !== cameraId) {
        currentDetailCameraId = cameraId;
        fetchWarmEvents(cameraId).then(() => {
            updatePlaybackNav();
        });
    }

    hideAllViews();
    playbackView.hidden = false;
    currentPlaybackKey = key;

    const ev = warmEvents.find(e => e.key === key);
    if (ev) {
        const evDate = new Date(ev.start_ms);
        const timeStr = evDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        playbackEventInfo.textContent = `${eventTypeLabel(ev)}${chainPartLabel(ev)} \u00b7 ${timeStr}`;
    } else {
        playbackEventInfo.textContent = 'Event';
    }

    loadPlaybackVideo(cameraId, key);
    updatePlaybackNav();
}

function cleanupPlaybackView() {
    if (playbackAnimationId) {
        cancelAnimationFrame(playbackAnimationId);
        playbackAnimationId = null;
    }
    if (playbackHls) { playbackHls.destroy(); playbackHls = null; }
    playbackVideo.src = '';
    currentPlaybackKey = null;
    isScrubbing = false;
    playbackScrubber.value = 0;
    playbackProgressFill.style.width = '0%';
}

function loadPlaybackVideo(cameraId, key) {
    const src = `api/cameras/${encodeURIComponent(cameraId)}/events/${key}/playlist.m3u8`;

    if (playbackHls) { playbackHls.destroy(); playbackHls = null; }
    playbackLoading.hidden = false;

    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        playbackHls = new Hls({ enableWorker: false, ...hlsAuthConfig() });
        playbackHls.loadSource(src);
        playbackHls.attachMedia(playbackVideo);

        playbackHls.on(Hls.Events.MANIFEST_PARSED, () => {
            playbackLoading.hidden = true;
            playbackVideo.play().catch(e => console.error('Playback failed:', e));
            startPlaybackUpdate();
        });

        playbackHls.on(Hls.Events.ERROR, (event, data) => {
            console.error('Playback HLS error:', data.type, data.details);
            if (data.fatal) {
                playbackLoading.querySelector('p').textContent = 'Playback error';
                playbackLoading.hidden = false;
            }
        });
    } else if (playbackVideo.canPlayType('application/vnd.apple.mpegurl')) {
        playbackVideo.src = authUrl(src);
        playbackVideo.addEventListener('loadedmetadata', () => {
            playbackLoading.hidden = true;
            playbackVideo.play().catch(e => console.error('Playback failed:', e));
            startPlaybackUpdate();
        }, { once: true });
    }
}

function startPlaybackUpdate() {
    if (playbackAnimationId) cancelAnimationFrame(playbackAnimationId);

    function update() {
        const duration = playbackVideo.duration;
        if (duration && isFinite(duration)) {
            if (!isScrubbing) {
                const progress = playbackVideo.currentTime / duration;
                playbackScrubber.value = Math.round(progress * 1000);
                playbackProgressFill.style.width = (progress * 100) + '%';
                playbackCurrentTime.textContent = formatTimeShort(playbackVideo.currentTime);
            }
            playbackDuration.textContent = formatTimeShort(duration);
        }
        playbackAnimationId = requestAnimationFrame(update);
    }
    update();
}

function getAdjacentEvents(key) {
    const sorted = [...warmEvents].sort((a, b) => b.start_ms - a.start_ms);
    const idx = sorted.findIndex(e => e.key === key);
    return {
        prev: idx > 0 ? sorted[idx - 1] : null,
        next: idx >= 0 && idx < sorted.length - 1 ? sorted[idx + 1] : null,
    };
}

function updatePlaybackNav() {
    const nav = getAdjacentEvents(currentPlaybackKey);

    if (nav.prev) {
        prevEventBtn.hidden = false;
        const prevDate = new Date(nav.prev.start_ms);
        prevEventText.textContent = prevDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        prevEventThumb.src = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${nav.prev.key}/thumbnail`);
    } else {
        prevEventBtn.hidden = true;
    }

    if (nav.next) {
        nextEventBtn.hidden = false;
        const nextDate = new Date(nav.next.start_ms);
        nextEventText.textContent = nextDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        nextEventThumb.src = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${nav.next.key}/thumbnail`);
    } else {
        nextEventBtn.hidden = true;
    }
}
