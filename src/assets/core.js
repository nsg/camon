// Shared UI state and utilities; the assets are ordered classic scripts.

const gridView = document.getElementById('grid-view');
const grid = document.getElementById('camera-grid');
const noCameras = document.getElementById('no-cameras');

const tokenPrompt = document.getElementById('token-prompt');
const tokenInput = document.getElementById('token-input');
const tokenSubmit = document.getElementById('token-submit');

let cameras = [];
const gridHlsInstances = new Map();
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
    gridView.hidden = true;
    liveView.hidden = true;
    eventsView.hidden = true;
    playbackView.hidden = true;
    debugView.hidden = true;
}

function showGridView() {
    cleanupLiveView();
    cleanupPlaybackView();
    cleanupDebugView();
    hideAllViews();
    gridView.hidden = false;

    cameras.forEach(cameraId => {
        if (!gridHlsInstances.has(cameraId)) {
            // Avoid interpolating camera ids into selectors.
            const cell = Array.from(grid.children).find(c => c.dataset.cameraId === cameraId);
            if (cell) {
                loadGridCamera(cameraId, cell.querySelector('video'));
            }
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

function loadGridCamera(cameraId, video) {
    const src = `api/stream/${encodeURIComponent(cameraId)}/playlist.m3u8?live=true&stream=sub`;
    const loading = video.parentElement.querySelector('.loading');

    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        const hls = new Hls({ enableWorker: false, ...hlsAuthConfig() });
        gridHlsInstances.set(cameraId, hls);
        hls.loadSource(src);
        hls.attachMedia(video);

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
            loading.hidden = true;
            video.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
        });

        hls.on(Hls.Events.ERROR, (event, data) => {
            console.error(`HLS error for ${cameraId}:`, data.type, data.details);
            if (data.fatal) {
                switch (data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR: hls.startLoad(); break;
                    case Hls.ErrorTypes.MEDIA_ERROR: hls.recoverMediaError(); break;
                    default:
                        loading.querySelector('p').textContent = 'Stream error';
                        loading.hidden = false;
                }
            }
        });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = authUrl(src);
        video.addEventListener('loadedmetadata', () => {
            loading.hidden = true;
            video.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
        });
    } else {
        loading.querySelector('p').textContent = 'HLS not supported';
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
