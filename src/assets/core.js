// Shared UI state and utilities; the assets are ordered classic scripts.

const gridView = document.getElementById('grid-view');
const grid = document.getElementById('camera-grid');
const noCameras = document.getElementById('no-cameras');

const tokenPrompt = document.getElementById('token-prompt');
const tokenInput = document.getElementById('token-input');
const tokenSubmit = document.getElementById('token-submit');

let cameras = [];
const gridHlsInstances = new Map();
const GRID_PLAYER_REUSE_MS = 10_000;
let gridObserver = null;
let currentView = null;
let isFirstLoad = true;
let currentDetailCameraId = null;

function withViewTransition(callback, isBack = false) {
    if (!isFirstLoad && document.startViewTransition) {
        document.documentElement.classList.toggle('swipe-back', isBack);
        const transition = document.startViewTransition(callback);
        transition.finished.then(() => {
            document.documentElement.classList.remove('swipe-back');
        });
    } else {
        callback();
        isFirstLoad = false;
    }
}

// Every request carries the token because the UI cannot know the server's auth mode.
// Native media elements cannot set headers and use ?token= instead.
const TOKEN_STORAGE_KEY = 'camon.token';
let apiToken = localStorage.getItem(TOKEN_STORAGE_KEY) || '';

function authHeaders(extra) {
    const headers = Object.assign({}, extra);
    if (apiToken) headers['Authorization'] = `Bearer ${apiToken}`;
    return headers;
}

function authUrl(url) {
    if (!apiToken) return url;
    return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(apiToken)}`;
}

async function apiFetch(url, options = {}) {
    const response = await fetch(url, Object.assign({}, options, {
        headers: authHeaders(options.headers),
    }));
    if (response.status === 401) showTokenPrompt();
    return response;
}

// Passes never overlap, preventing stale responses from replacing newer state.
// Each poller owns its abort controller so stopping one view cannot affect another.
function startPoller(name, intervalMs, pass) {
    const controller = new AbortController();
    let timer = null;
    let stopped = false;

    async function tick() {
        timer = null;
        try {
            await pass(controller.signal);
        } catch (err) {
            if (err && err.name === 'AbortError') return;
            console.error(`Failed to fetch ${name}:`, err);
        }
        if (!stopped) timer = setTimeout(tick, intervalMs);
    }

    return {
        first: tick(),
        stop() {
            stopped = true;
            if (timer !== null) { clearTimeout(timer); timer = null; }
            controller.abort();
        },
    };
}

function hlsAuthConfig() {
    return {
        xhrSetup: (xhr) => {
            if (apiToken) xhr.setRequestHeader('Authorization', `Bearer ${apiToken}`);
        },
    };
}

// Move the playhead toward the live edge without leaving buffered media.
// Returns true when it seeked.
function seekTowardLive(video, syncPosition) {
    if (!Number.isFinite(syncPosition) || syncPosition - video.currentTime <= 10) return false;
    try {
        const buffered = video.buffered;
        for (let i = 0; i < buffered.length; i++) {
            if (syncPosition >= buffered.start(i) - 0.25 &&
                syncPosition <= buffered.end(i) + 0.25) {
                video.currentTime = syncPosition;
                return true;
            }
        }
        if (buffered.length > 0) {
            const bufferedTarget = buffered.end(buffered.length - 1) - 0.5;
            if (bufferedTarget - video.currentTime > 1) {
                video.currentTime = bufferedTarget;
                return true;
            }
        }
    } catch (_) {
        // Detached media elements can reject buffered access.
    }
    return false;
}

async function fetchStreamStatus(cameraId, stream, signal) {
    const suffix = stream === 'sub' ? '?stream=sub' : '';
    try {
        const response = await apiFetch(
            `api/stream/${encodeURIComponent(cameraId)}/status${suffix}`, { signal, cache: 'no-store' });
        if (response.status === 401) return { status: 'unavailable', message: 'Authentication required' };
        if (response.status === 404) return { status: 'unavailable', message: 'Camera not found' };
        if (!response.ok) return { status: 'unavailable', message: 'Cannot check stream; retrying' };
        let result;
        try { result = await response.json(); }
        catch (_) { return { status: 'unavailable', message: 'Cannot check stream; retrying' }; }
        if (!['starting', 'receiving', 'no_video', 'interrupted'].includes(result.status)) {
            return { status: 'unavailable', message: 'Cannot check stream; retrying' };
        }
        return result;
    } catch (error) {
        if (error.name === 'AbortError') throw error;
        return { status: 'unavailable', message: 'Cannot reach camon; retrying' };
    }
}

function streamHasFailed(status) {
    return status === 'no_video' || status === 'interrupted';
}

function showStreamFailure(loading, status) {
    const title = status === 'interrupted' ? 'Stream interrupted' : 'No video received';
    const explanation = status === 'interrupted' ?
        'Video is no longer reaching Camon. Check camera power and network. Retrying automatically.' :
        'Check camera power, network and RTSP stream. Retrying automatically.';
    const heading = document.createElement('strong');
    heading.textContent = title;
    const detail = document.createElement('small');
    detail.textContent = explanation;
    loading.querySelector('p').replaceChildren(heading, detail);
    loading.hidden = false;
}

function showTokenPrompt() {
    if (!tokenPrompt.hidden) return;
    tokenPrompt.hidden = false;
    tokenInput.focus();
}

function submitToken() {
    const value = tokenInput.value.trim();
    if (!value) return;
    localStorage.setItem(TOKEN_STORAGE_KEY, value);
    location.reload();
}

function wireTokenPrompt() {
    tokenSubmit.addEventListener('click', submitToken);
    tokenInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') submitToken();
    });
}

async function loadCameras() {
    try {
        const response = await apiFetch('api/cameras');
        cameras = response.ok ? await response.json() : [];

        if (cameras.length === 0) {
            noCameras.hidden = false;
        } else {
            cameras.forEach(cameraId => {
                const cell = createCameraCell(cameraId);
                grid.appendChild(cell);
            });
        }
    } catch (err) {
        console.error('Failed to fetch cameras:', err);
        noCameras.querySelector('p').textContent = 'Failed to load cameras';
        noCameras.hidden = false;
    }
}

function router() {
    const hash = window.location.hash || '#/';

    // Accept newer event-type spellings and let the server validate the key.
    const playbackMatch = hash.match(/^#\/camera\/(.+)\/events\/(\d+_\d+_[a-z0-9-]+)$/);
    if (playbackMatch) {
        const cameraId = decodeURIComponent(playbackMatch[1]);
        const key = playbackMatch[2];
        if (cameras.includes(cameraId)) {
            const targetView = `playback:${cameraId}:${key}`;
            if (currentView !== targetView) {
                const isBack = currentView && currentView.startsWith('playback:');
                withViewTransition(() => showPlaybackView(cameraId, key), isBack);
                currentView = targetView;
            }
            return;
        }
    }

    const debugMatch = hash.match(/^#\/camera\/(.+)\/debug$/);
    if (debugMatch) {
        const cameraId = decodeURIComponent(debugMatch[1]);
        if (cameras.includes(cameraId)) {
            const targetView = `debug:${cameraId}`;
            if (currentView !== targetView) {
                const isBack = false;
                withViewTransition(() => showDebugView(cameraId), isBack);
                currentView = targetView;
            }
            return;
        }
    }

    const eventsMatch = hash.match(/^#\/camera\/(.+)\/events$/);
    if (eventsMatch) {
        const cameraId = decodeURIComponent(eventsMatch[1]);
        if (cameras.includes(cameraId)) {
            const targetView = `events:${cameraId}`;
            if (currentView !== targetView) {
                const isBack = currentView && currentView.startsWith('playback:');
                withViewTransition(() => showEventsView(cameraId), isBack);
                currentView = targetView;
            }
            return;
        }
    }

    const cameraMatch = hash.match(/^#\/camera\/([^/]+)$/);
    if (cameraMatch) {
        const cameraId = decodeURIComponent(cameraMatch[1]);
        if (cameras.includes(cameraId)) {
            const targetView = `live:${cameraId}`;
            if (currentView !== targetView) {
                const isBack = currentView !== null && !currentView.startsWith('live:') ||
                               (currentView && currentView.startsWith('events:'));
                withViewTransition(() => showLiveView(cameraId), isBack);
                currentView = targetView;
            }
            return;
        }
    }

    if (currentView !== 'grid') {
        const isBack = currentView !== null;
        withViewTransition(() => showGridView(), isBack);
        currentView = 'grid';
    }
}

const volumeOnPath = 'M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z';
const volumeOffPath = 'M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z';

function updateMuteIcon(btn, video) {
    btn.querySelector('path').setAttribute('d', video.muted ? volumeOffPath : volumeOnPath);
    btn.classList.toggle('muted', video.muted);
}

function hideAllViews() {
    stopGridObservation();
    gridView.hidden = true;
    liveView.hidden = true;
    eventsView.hidden = true;
    playbackView.hidden = true;
    debugView.hidden = true;
    if (typeof syncDetailCameraVisibility === 'function') syncDetailCameraVisibility();
}

function showGridView() {
    cleanupLiveView();
    cleanupPlaybackView();
    cleanupDebugView();
    hideAllViews();
    gridView.hidden = false;
    startGridObservation();
}

function stopGridObservation() {
    if (gridObserver) { gridObserver.disconnect(); gridObserver = null; }
    gridHlsInstances.forEach((entry) => setGridCameraActive(entry, false));
}

function startGridObservation() {
    if (gridObserver || document.hidden || gridView.hidden) return;
    if (typeof IntersectionObserver !== 'undefined') {
        const observer = new IntersectionObserver((entries) => {
            if (gridObserver !== observer || gridView.hidden || document.hidden) return;
            entries.forEach(({ target, isIntersecting }) => {
                const cameraId = target.dataset.cameraId;
                if (isIntersecting) {
                    setGridCameraActive(getGridCamera(cameraId, target.querySelector('video')), true);
                } else {
                    const entry = gridHlsInstances.get(cameraId);
                    if (entry) setGridCameraActive(entry, false);
                }
            });
        });
        gridObserver = observer;
        Array.from(grid.children).forEach(cell => gridObserver.observe(cell));
    } else {
        // Older WebViews still load only tiles on screen.
        updateGridVisibility();
    }
}

function syncCameraVisibility() {
    if (document.hidden) stopGridObservation();
    else if (!gridView.hidden) startGridObservation();
    if (typeof syncDetailCameraVisibility === 'function') syncDetailCameraVisibility();
}

function updateGridVisibility() {
    if (typeof IntersectionObserver !== 'undefined' || gridView.hidden || document.hidden) return;
    Array.from(grid.children).forEach(cell => {
        const rect = cell.getBoundingClientRect();
        const visible = rect.bottom > 0 && rect.top < window.innerHeight &&
            rect.right > 0 && rect.left < window.innerWidth;
        const cameraId = cell.dataset.cameraId;
        if (visible) {
            setGridCameraActive(getGridCamera(cameraId, cell.querySelector('video')), true);
        } else {
            const entry = gridHlsInstances.get(cameraId);
            if (entry) setGridCameraActive(entry, false);
        }
    });
}

function createCameraCell(cameraId) {
    const cell = document.createElement('div');
    cell.className = 'camera-cell';
    cell.dataset.cameraId = cameraId;
    cell.innerHTML = `
        <span class="camera-label">${esc(cameraId)}</span>
        <video playsinline muted></video>
        <div class="loading"><p>Loading...</p></div>
    `;
    cell.addEventListener('click', () => {
        window.location.hash = `/camera/${encodeURIComponent(cameraId)}`;
    });
    return cell;
}

function getGridCamera(cameraId, video) {
    const existing = gridHlsInstances.get(cameraId);
    if (existing) return existing;
    const src = `api/stream/${encodeURIComponent(cameraId)}/playlist.m3u8?live=true&stream=sub`;
    const loading = video.parentElement.querySelector('.loading');
    const entry = { cameraId, video, loading, src, hls: null, active: false,
        initialized: false, nativeHls: false, generation: 0, statusPoller: null,
        streamStatus: null, statusMessage: null, browserError: null, hasPlayed: false,
        deactivatedAt: null };
    gridHlsInstances.set(cameraId, entry);

    video.addEventListener('playing', () => {
        if (entry.active) {
            entry.hasPlayed = true;
            entry.browserError = null;
        }
        renderGridCameraLoading(entry);
    });
    video.addEventListener('waiting', () => {
        if (entry.active) {
            entry.hasPlayed = false;
            renderGridCameraLoading(entry);
        }
    });
    video.addEventListener('error', () => {
        if (entry.active) {
            entry.browserError = 'Cannot play stream; retrying';
            renderGridCameraLoading(entry);
        }
    });

    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        entry.createHls = function createGridHls() {
            const hls = new Hls({
                enableWorker: true,
                autoStartLoad: false,
                // Grid tiles have no history controls; keep only a short buffer on phones.
                backBufferLength: 15,
                liveBackBufferLength: 15,
                maxBufferLength: 15,
                maxMaxBufferLength: 15,
                ...hlsAuthConfig(),
            });
            entry.hls = hls;

            hls.on(Hls.Events.MANIFEST_PARSED, () => {
                if (entry.active) {
                    hls.startLoad(-1);
                    playGridCamera(entry);
                }
            });

            hls.on(Hls.Events.ERROR, (event, data) => {
                console.error(`HLS error for ${cameraId}:`, data.type, data.details);
                if (entry.active && data.fatal) {
                    switch (data.type) {
                        case Hls.ErrorTypes.NETWORK_ERROR:
                            entry.browserError = 'Cannot load stream; retrying';
                            hls.startLoad();
                            break;
                        case Hls.ErrorTypes.MEDIA_ERROR:
                            entry.browserError = 'Cannot play stream; retrying';
                            hls.recoverMediaError();
                            break;
                        default: entry.browserError = 'Stream error';
                    }
                    renderGridCameraLoading(entry);
                }
            });
        };
        entry.createHls();
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        entry.nativeHls = true;
        video.addEventListener('loadedmetadata', () => {
            if (entry.active) playGridCamera(entry);
        });
    } else {
        entry.browserError = 'HLS not supported';
        renderGridCameraLoading(entry);
    }
    return entry;
}

function showGridCameraError(entry, message) {
    entry.loading.querySelector('p').textContent = message;
    entry.loading.hidden = false;
}

function renderGridCameraLoading(entry) {
    if (!entry.active) return;
    if (entry.streamStatus === 'unavailable') {
        showGridCameraError(entry, entry.statusMessage);
    } else if (streamHasFailed(entry.streamStatus)) {
        showStreamFailure(entry.loading, entry.streamStatus);
    } else if (entry.browserError) {
        const message = entry.browserError;
        showGridCameraError(entry, message);
    } else if (entry.hasPlayed) {
        entry.loading.hidden = true;
    } else {
        showGridCameraError(entry, 'Loading...');
    }
}

function startGridStatusPoller(entry) {
    if (entry.statusPoller) return;
    const generation = entry.generation;
    entry.statusPoller = startPoller(`stream status for ${entry.cameraId}`, 5000,
        async (signal) => {
            const result = await fetchStreamStatus(entry.cameraId, 'sub', signal);
            if (!entry.active || entry.generation !== generation) return;
            if (streamHasFailed(result.status) && !streamHasFailed(entry.streamStatus)) {
                entry.hasPlayed = false;
            }
            entry.streamStatus = result.status;
            entry.statusMessage = result.message || null;
            renderGridCameraLoading(entry);
        });
}

function playGridCamera(entry) {
    const { video, cameraId } = entry;
    const generation = entry.generation;
    seekGridCameraToLive(entry);
    video.play().catch(e => {
        if (entry.active && entry.generation === generation) {
            entry.browserError = 'Playback blocked';
            renderGridCameraLoading(entry);
        }
        console.error(`Play failed for ${cameraId}:`, e);
    });
}

function seekGridCameraToLive(entry) {
    // A retained MSE buffer can lag behind the live edge after visiting detail.
    if (entry.hls) seekTowardLive(entry.video, entry.hls.liveSyncPosition);
}

function setGridCameraActive(entry, active) {
    if (entry.active === active) return;
    entry.active = active;
    entry.generation++;
    if (!active) {
        entry.deactivatedAt = performance.now();
        if (entry.statusPoller) { entry.statusPoller.stop(); entry.statusPoller = null; }
        entry.video.pause();
        if (entry.hls) entry.hls.stopLoad();
        else if (entry.nativeHls) {
            entry.video.removeAttribute('src');
            entry.video.load();
        }
        return;
    }

    const reuseExpired = entry.deactivatedAt !== null &&
        performance.now() - entry.deactivatedAt > GRID_PLAYER_REUSE_MS;
    entry.deactivatedAt = null;
    entry.hasPlayed = false;
    if (entry.browserError !== 'HLS not supported') entry.browserError = null;
    renderGridCameraLoading(entry);
    if (entry.hls || entry.nativeHls) startGridStatusPoller(entry);
    if (entry.hls) {
        if (!entry.initialized) {
            entry.initialized = true;
            entry.hls.loadSource(entry.src);
            entry.hls.attachMedia(entry.video);
        } else if (reuseExpired) {
            entry.hls.destroy();
            entry.createHls();
            entry.hls.loadSource(entry.src);
            entry.hls.attachMedia(entry.video);
        } else {
            entry.hls.startLoad(-1);
            playGridCamera(entry);
        }
    } else if (entry.nativeHls) {
        entry.video.src = authUrl(entry.src);
        entry.video.load();
    } else {
        entry.browserError = 'HLS not supported';
        renderGridCameraLoading(entry);
    }
}

function formatDateLabel(date) {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
    const eventDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());

    if (eventDay.getTime() === today.getTime()) return 'Today';
    if (eventDay.getTime() === yesterday.getTime()) return 'Yesterday';
    return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
}

// Escape every value interpolated into an innerHTML template.
function esc(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatTimeShort(seconds) {
    if (!isFinite(seconds)) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}
