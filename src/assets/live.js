// Live monitor, timeline, overlays, and warm-history maps.

const liveView = document.getElementById('live-view');
const detailVideo = document.getElementById('detail-video');
const detailLoading = document.getElementById('detail-loading');
const detailCameraName = document.getElementById('detail-camera-name');
const backBtn = document.getElementById('back-btn');
const muteToggleBtn = document.getElementById('mute-toggle-btn');
const maskToggleBtn = document.getElementById('mask-toggle-btn');
const bgToggleBtn = document.getElementById('bg-toggle-btn');
const tlTrack = document.getElementById('tl-track');
const tlCanvas = document.getElementById('tl-canvas');
const tlCtx = tlCanvas.getContext('2d');
const tlMarkers = document.getElementById('tl-markers');
const tlPlayhead = document.getElementById('tl-playhead');
const tlWindowLabel = document.getElementById('tl-window-label');
const tlOffset = document.getElementById('tl-offset');
const tlTicks = document.getElementById('tl-ticks');
const historyPanel = document.getElementById('history-panel');
const historyDays = document.getElementById('history-days');
const stabilityOverlay = document.getElementById('stability-overlay');
const stabilityCtx = stabilityOverlay.getContext('2d');
const bgOverlay = document.getElementById('bg-overlay');
const bgCtx = bgOverlay.getContext('2d');
const liveBtn = document.getElementById('live-btn');

// Timeline timestamps are wall-clock anchored because hot-buffer offsets slide on eviction.
let currentDetections = []; // {id, t, object_class, confidence}
let motionSegs = [];        // {tStart, tEnd, intensity}
let lastDetIds = null;
let openMarker = null;
let tlPointerId = null;
let lastTimelineDraw = 0;
let lastTickKey = null;
const hoverCapable = window.matchMedia('(hover: hover)').matches;
let bufferDuration = 0;
let motionPoller = null;
let detectionPoller = null;
let stabilityPoller = null;
let stabilityOverlayEnabled = false;
let stabilityDrawPending = false;
let stabilityImage = null;
let rawMog2Image = null;
let morphImage = null;
let bgOverlayEnabled = false;
let bgImage = null;
let overlayAnimationId = null;
let isLiveScrubbing = false;
let isAtLiveEdge = true;
let detailHls = null;
let detailPlayingHandler = null;

function trackFraction(e) {
    const rect = tlTrack.getBoundingClientRect();
    if (rect.width === 0) return 1;
    return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
}

function scrubPreview(e) {
    const frac = trackFraction(e);
    tlPlayhead.style.left = (frac * 100) + '%';
    const r = timelineRange();
    if (r) {
        const behind = (1 - frac) * r.range;
        tlOffset.textContent = behind < 3 ? '' : '-' + formatTimeShort(behind);
    }
}

function setLiveEdge(atEdge) {
    isAtLiveEdge = atEdge;
    liveBtn.classList.toggle('active', atEdge);
}

function wireLiveView() {
    backBtn.addEventListener('click', () => {
        window.location.hash = '/';
    });

    muteToggleBtn.addEventListener('click', () => {
        detailVideo.muted = !detailVideo.muted;
        updateMuteIcon(muteToggleBtn, detailVideo);
    });

    maskToggleBtn.addEventListener('click', () => {
        stabilityOverlayEnabled = !stabilityOverlayEnabled;
        maskToggleBtn.classList.toggle('active', stabilityOverlayEnabled);
        stabilityOverlay.hidden = !stabilityOverlayEnabled;
        if (!stabilityOverlayEnabled) {
            stabilityCtx.clearRect(0, 0, stabilityOverlay.width, stabilityOverlay.height);
            stabilityImage = null;
            rawMog2Image = null;
            morphImage = null;
        } else {
            fetchStabilityMap();
        }
    });

    bgToggleBtn.addEventListener('click', () => {
        bgOverlayEnabled = !bgOverlayEnabled;
        bgToggleBtn.classList.toggle('active', bgOverlayEnabled);
        bgOverlay.hidden = !bgOverlayEnabled;
        if (!bgOverlayEnabled) {
            bgCtx.clearRect(0, 0, bgOverlay.width, bgOverlay.height);
            bgImage = null;
        } else {
            fetchBackgroundMap();
        }
    });

    tlTrack.addEventListener('pointerdown', (e) => {
        if (e.target.closest('.tl-marker')) return;
        closeMarkerCard();
        tlPointerId = e.pointerId;
        isLiveScrubbing = true;
        tlTrack.setPointerCapture(e.pointerId);
        scrubPreview(e);
        e.preventDefault();
    });
    tlTrack.addEventListener('pointermove', (e) => {
        if (tlPointerId !== e.pointerId || !isLiveScrubbing) return;
        scrubPreview(e);
    });
    tlTrack.addEventListener('pointerup', (e) => {
        if (tlPointerId !== e.pointerId || !isLiveScrubbing) return;
        tlPointerId = null;
        isLiveScrubbing = false;
        const frac = trackFraction(e);
        const r = timelineRange();
        if (r) {
            detailVideo.currentTime = r.start + frac * r.range;
            setLiveEdge(frac > 0.98);
        }
    });
    tlTrack.addEventListener('pointercancel', (e) => {
        if (tlPointerId === e.pointerId) {
            tlPointerId = null;
            isLiveScrubbing = false;
        }
    });

    document.addEventListener('click', (e) => {
        if (openMarker && !e.target.closest('.tl-marker')) closeMarkerCard();
    });

    liveBtn.addEventListener('click', () => {
        if (detailHls) {
            const sync = detailHls.liveSyncPosition;
            if (typeof sync === 'number') {
                detailVideo.currentTime = sync;
            } else if (detailVideo.buffered.length > 0) {
                detailVideo.currentTime = detailVideo.buffered.end(detailVideo.buffered.length - 1) - 0.5;
            }
        }
        setLiveEdge(true);
    });

    // ResizeObserver covers window changes, rotation, and initial stream sizing.
    new ResizeObserver(() => {
        drawBackground();
        scheduleStabilityDraw();
        drawMask();
    }).observe(detailVideo);
}

function showLiveView(cameraId) {
    cleanupPlaybackView();
    cleanupDebugView();

    if (currentDetailCameraId !== cameraId) {
        cleanupLiveView();
    }

    gridHlsInstances.forEach((hls) => hls.destroy());
    gridHlsInstances.clear();

    hideAllViews();
    liveView.hidden = false;
    detailCameraName.textContent = cameraId;
    currentDetailCameraId = cameraId;

    if (!detailHls) {
        detailLoading.querySelector('p').textContent = 'Loading...';
        detailLoading.hidden = false;
        stabilityOverlay.hidden = !stabilityOverlayEnabled;
        bgOverlay.hidden = !bgOverlayEnabled;

        if (stabilityOverlayEnabled) fetchStabilityMap();
        if (bgOverlayEnabled) fetchBackgroundMap();
        fetchMotionSettings(cameraId);

        loadDetailCamera(cameraId);
        fetchWarmEvents(cameraId);
    } else {
        renderHistoryPanel();
    }
}

function cleanupLiveView() {
    if (overlayAnimationId) {
        cancelAnimationFrame(overlayAnimationId);
        overlayAnimationId = null;
    }
    if (motionPoller) { motionPoller.stop(); motionPoller = null; }
    if (detectionPoller) { detectionPoller.stop(); detectionPoller = null; }
    if (warmEventPoller) { warmEventPoller.stop(); warmEventPoller = null; }
    if (stabilityPoller) { stabilityPoller.stop(); stabilityPoller = null; }
    if (detailPlayingHandler) {
        detailVideo.removeEventListener('playing', detailPlayingHandler);
        detailPlayingHandler = null;
    }
    if (detailHls) { detailHls.destroy(); detailHls = null; }
    detailVideo.src = '';
    currentDetections = [];
    motionSegs = [];
    lastDetIds = null;
    currentDetailCameraId = null;
    bufferDuration = 0;
    warmEvents = [];
    eventChains = new Map();

    stabilityImage = null;
    rawMog2Image = null;
    morphImage = null;
    stabilityOverlay.hidden = true;
    stabilityOverlayEnabled = false;
    maskToggleBtn.classList.remove('active');
    stabilityCtx.clearRect(0, 0, stabilityOverlay.width, stabilityOverlay.height);

    bgImage = null;
    bgOverlay.hidden = true;
    bgOverlayEnabled = false;
    bgToggleBtn.classList.remove('active');
    bgCtx.clearRect(0, 0, bgOverlay.width, bgOverlay.height);

    setMaskEditEnabled(false);
    setActiveMaskLayer('movement');
    maskCtx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
    motionSettings = null;
    maskCells = [];
    detectionCells = [];
    settingsPanel.hidden = true;
    settingsBtn.classList.remove('active');
    clearSettingsError();

    isLiveScrubbing = false;
    tlPointerId = null;
    setLiveEdge(true);
    closeMarkerCard();
    tlMarkers.innerHTML = '';
    tlCtx.clearRect(0, 0, tlCanvas.width, tlCanvas.height);
    tlPlayhead.style.left = '100%';
    tlOffset.textContent = '';
    tlWindowLabel.textContent = '';
    tlTicks.innerHTML = '';
    lastTickKey = null;
    historyPanel.hidden = true;
    historyDays.innerHTML = '';
}

function loadDetailCamera(cameraId) {
    const src = `api/stream/${encodeURIComponent(cameraId)}/playlist.m3u8`;
    if (detailPlayingHandler) {
        detailVideo.removeEventListener('playing', detailPlayingHandler);
    }
    detailPlayingHandler = () => {
        detailLoading.hidden = true;
        detailPlayingHandler = null;
    };
    detailVideo.addEventListener('playing', detailPlayingHandler, { once: true });

    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        detailHls = new Hls({
            enableWorker: false,
            liveBackBufferLength: 600,
            backBufferLength: 600,
            liveDurationInfinity: false,
            ...hlsAuthConfig(),
        });
        detailHls.loadSource(src);
        detailHls.attachMedia(detailVideo);

        detailHls.on(Hls.Events.MANIFEST_PARSED, () => {
            detailVideo.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
            startOverlayUpdates();
            fetchMotionSegments(cameraId);
            fetchDetections(cameraId);
        });

        detailHls.on(Hls.Events.ERROR, (event, data) => {
            console.error(`HLS error for ${cameraId}:`, data.type, data.details);
            if (data.fatal) {
                switch (data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR: detailHls.startLoad(); break;
                    case Hls.ErrorTypes.MEDIA_ERROR: detailHls.recoverMediaError(); break;
                    default:
                        detailLoading.querySelector('p').textContent = 'Stream error';
                        detailLoading.hidden = false;
                }
            }
        });
    } else if (detailVideo.canPlayType('application/vnd.apple.mpegurl')) {
        detailVideo.src = authUrl(src);
        detailVideo.addEventListener('loadedmetadata', () => {
            detailVideo.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
            startOverlayUpdates();
            fetchMotionSegments(cameraId);
            fetchDetections(cameraId);
        }, { once: true });
    } else {
        detailLoading.querySelector('p').textContent = 'HLS not supported';
    }
}

// Only playback-following elements redraw per frame; polled overlays redraw on arrival or resize.
function startOverlayUpdates() {
    if (overlayAnimationId) cancelAnimationFrame(overlayAnimationId);
    function update() {
        updateTimeline();
        overlayAnimationId = requestAnimationFrame(update);
    }
    update();
}

function timelineRange() {
    const seekable = detailVideo.seekable;
    if (seekable.length === 0) return null;
    const start = seekable.start(0);
    const end = seekable.end(seekable.length - 1);
    if (end - start <= 0) return null;
    // hls.js can retain media older than the server window, where no overlay data exists.
    let range = end - start;
    if (bufferDuration > 0 && bufferDuration < range) range = bufferDuration;
    return { start: end - range, end, range };
}

function timeToFrac(t, r, nowS) {
    return 1 - (nowS - t) / r.range;
}

function updateTimeline() {
    const r = timelineRange();
    if (!r) return;

    tlWindowLabel.textContent = 'Buffer · ' + formatTimeShort(r.range);

    if (!isLiveScrubbing) {
        const current = detailVideo.currentTime;
        const frac = Math.max(0, Math.min(1, (current - r.start) / r.range));
        tlPlayhead.style.left = (frac * 100) + '%';
        const timeToLive = r.end - current;
        if (isAtLiveEdge) {
            tlOffset.textContent = '';
            if (detailHls) {
                const sync = detailHls.liveSyncPosition;
                if (typeof sync === 'number' && sync - current > 10) {
                    // seekable.end is the playlist edge, not buffered media; seeking there stalls
                    // playback until the next segment arrives.
                    detailVideo.currentTime = sync;
                }
            }
        } else if (timeToLive < 3) {
            tlOffset.textContent = '';
            setLiveEdge(true);
        } else {
            tlOffset.textContent = '-' + formatTimeShort(timeToLive);
        }
    }

    const nowS = Date.now() / 1000;
    if (nowS - lastTimelineDraw >= 1) {
        lastTimelineDraw = nowS;
        drawTimelineBars(r, nowS);
        positionMarkers(r, nowS);
        renderTicks(r);
    }
}

function drawTimelineBars(r, nowS) {
    const w = tlTrack.clientWidth;
    const h = tlTrack.clientHeight;
    if (w === 0 || h === 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (tlCanvas.width !== Math.round(w * dpr) || tlCanvas.height !== Math.round(h * dpr)) {
        tlCanvas.width = Math.round(w * dpr);
        tlCanvas.height = Math.round(h * dpr);
    }
    tlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    tlCtx.clearRect(0, 0, w, h);
    if (motionSegs.length === 0) return;

    // Motion scores have no fixed scale; normalize within the visible window.
    let maxScore = 0;
    motionSegs.forEach(s => { maxScore = Math.max(maxScore, s.intensity); });
    if (maxScore <= 0) maxScore = 1;

    tlCtx.fillStyle = '#d8d78f';
    motionSegs.forEach(s => {
        const f0 = timeToFrac(s.tStart, r, nowS);
        const f1 = timeToFrac(s.tEnd, r, nowS);
        if (f1 <= 0 || f0 >= 1) return;
        const x0 = Math.max(0, f0) * w;
        const x1 = Math.min(1, f1) * w;
        const norm = Math.min(1, s.intensity / maxScore);
        const barH = Math.max(3, (0.15 + 0.85 * norm) * (h - 4));
        tlCtx.globalAlpha = 0.45 + 0.55 * norm;
        tlCtx.fillRect(x0, h - barH, Math.max(2, x1 - x0 - 1), barH);
    });
    tlCtx.globalAlpha = 1;
}

function renderTicks(r) {
    const key = Math.round(r.range / 10);
    if (key === lastTickKey) return;
    lastTickKey = key;
    const parts = [];
    for (let i = 0; i <= 4; i++) {
        const behind = r.range * (1 - i / 4);
        parts.push(`<span>${behind < 3 ? 'now' : '-' + formatTimeShort(behind)}</span>`);
    }
    tlTicks.innerHTML = parts.join('');
}

function seekToTime(t) {
    const seekable = detailVideo.seekable;
    if (seekable.length === 0) return;
    const end = seekable.end(seekable.length - 1);
    const behind = Date.now() / 1000 - t;
    detailVideo.currentTime = Math.max(seekable.start(0), end - behind);
    setLiveEdge(false);
}

// <img> cannot set auth headers, so overlay URLs carry the token. Bound stalled image loads so
// one request cannot stop the poller indefinitely.
const OVERLAY_IMAGE_TIMEOUT_MS = 15000;

function loadOverlayImage(url) {
    return new Promise(resolve => {
        const img = new Image();
        const done = (value) => { clearTimeout(timer); resolve(value); };
        const timer = setTimeout(() => done(null), OVERLAY_IMAGE_TIMEOUT_MS);
        img.onload = () => done(img);
        img.onerror = () => done(null);
        img.src = authUrl(url);
    });
}

// Swap all stages together so layers from different detector frames are never mixed.
async function fetchStabilityMap() {
    if (!stabilityOverlayEnabled || !currentDetailCameraId) return;
    const cameraId = currentDetailCameraId;
    const cam = encodeURIComponent(cameraId);
    const t = Date.now();

    const [raw, morph, filtered] = await Promise.all([
        loadOverlayImage(`api/cameras/${cam}/motion/maps/raw?t=${t}`),
        loadOverlayImage(`api/cameras/${cam}/motion/maps/morph?t=${t}`),
        loadOverlayImage(`api/cameras/${cam}/motion/maps/stability?t=${t}`),
    ]);
    if (!stabilityOverlayEnabled || currentDetailCameraId !== cameraId) return;

    rawMog2Image = raw;
    morphImage = morph;
    stabilityImage = filtered;
    scheduleStabilityDraw();
}

// Coalesce triggers because recolouring four video-sized layers is expensive.
function scheduleStabilityDraw() {
    if (stabilityDrawPending) return;
    stabilityDrawPending = true;
    requestAnimationFrame(() => {
        stabilityDrawPending = false;
        drawStability();
    });
}

function drawStability() {
    const w = detailVideo.clientWidth;
    const h = detailVideo.clientHeight;
    if (w === 0 || h === 0) return;
    if (stabilityOverlay.width !== w || stabilityOverlay.height !== h) {
        stabilityOverlay.width = w;
        stabilityOverlay.height = h;
    }
    stabilityCtx.clearRect(0, 0, w, h);

    // Paint the progressively smaller masks bottom-to-top.
    if (rawMog2Image) {
        recolorMask(rawMog2Image, w, h, 180, 60, 60, 150, 50);
    }
    if (morphImage) {
        recolorMask(morphImage, w, h, 240, 240, 0, 170, 128);
    }
    if (stabilityImage) {
        recolorMask(stabilityImage, w, h, 0, 255, 0, 180, 128);
    }
}

function recolorMask(img, w, h, r, g, b, a, threshold) {
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    const ctx = tmp.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const px = imageData.data;
    for (let i = 0; i < px.length; i += 4) {
        if (px[i] > threshold) {
            px[i]     = r;
            px[i + 1] = g;
            px[i + 2] = b;
            px[i + 3] = a;
        } else {
            px[i + 3] = 0;
        }
    }
    ctx.putImageData(imageData, 0, 0);
    stabilityCtx.drawImage(tmp, 0, 0);
}

// Failed loads clear the overlay; stale detector output would look like a quiet scene.
async function fetchBackgroundMap() {
    if (!bgOverlayEnabled || !currentDetailCameraId) return;
    const cameraId = currentDetailCameraId;
    const img = await loadOverlayImage(
        `api/cameras/${encodeURIComponent(cameraId)}/motion/maps/background?t=${Date.now()}`);
    if (!bgOverlayEnabled || currentDetailCameraId !== cameraId) return;
    bgImage = img;
    drawBackground();
}

function drawBackground() {
    const w = detailVideo.clientWidth;
    const h = detailVideo.clientHeight;
    if (w === 0 || h === 0) return;
    if (bgOverlay.width !== w || bgOverlay.height !== h) {
        bgOverlay.width = w;
        bgOverlay.height = h;
    }
    bgCtx.clearRect(0, 0, w, h);
    if (bgImage) bgCtx.drawImage(bgImage, 0, 0, w, h);
}

function fetchMotionSegments(cameraId) {
    if (motionPoller) motionPoller.stop();
    motionPoller = startPoller('motion data', 5000, async (signal) => {
        const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/motion`, { signal });
        if (currentDetailCameraId !== cameraId || !response.ok) return;
        const data = await response.json();
        if (data.total_duration > 0) {
            bufferDuration = data.total_duration;
        }
        const nowS = Date.now() / 1000;
        motionSegs = (data.segments || []).map(s => ({
            tStart: nowS - (data.total_duration - s.start),
            tEnd: nowS - (data.total_duration - s.end),
            intensity: s.intensity,
        }));
        lastTimelineDraw = 0;
    });

    if (stabilityPoller) stabilityPoller.stop();
    stabilityPoller = startPoller('motion overlays', 5000, () =>
        Promise.all([fetchStabilityMap(), fetchBackgroundMap()]));
}

function fetchDetections(cameraId) {
    if (detectionPoller) detectionPoller.stop();
    detectionPoller = startPoller('detection data', 5000, async (signal) => {
        const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/detections`, { signal });
        if (currentDetailCameraId !== cameraId || !response.ok) return;
        const data = await response.json();
        if (data.total_duration > 0) {
            bufferDuration = data.total_duration;
        }
        const nowS = Date.now() / 1000;
        currentDetections = (data.detections || []).map(d => ({
            id: d.id,
            t: nowS - (data.total_duration - d.timestamp),
            object_class: d.object_class,
            confidence: d.confidence,
        }));
        const ids = currentDetections.map(d => d.id).join(',');
        // Rebuilding closes the open card.
        if (ids !== lastDetIds && !openMarker) {
            rebuildMarkers();
        }
    });
}

// Detections closer than this fraction of the track collapse into one
// marker with a count badge.
const CLUSTER_FRAC = 0.03;

function rebuildMarkers() {
    const r = timelineRange();
    if (!r) return;
    lastDetIds = currentDetections.map(d => d.id).join(',');
    closeMarkerCard();
    tlMarkers.innerHTML = '';
    const nowS = Date.now() / 1000;

    const sorted = [...currentDetections].sort((a, b) => b.t - a.t);
    const clusters = [];
    sorted.forEach(det => {
        const frac = timeToFrac(det.t, r, nowS);
        const near = clusters.find(c => Math.abs(c.frac - frac) < CLUSTER_FRAC);
        if (near) {
            near.dets.push(det);
        } else {
            clusters.push({ frac, dets: [det] });
        }
    });

    clusters.forEach(cluster => {
        const marker = document.createElement('div');
        marker.className = 'tl-marker';
        marker._cluster = cluster;

        const rows = cluster.dets.slice(0, 3).map(det => {
            const conf = Math.round(det.confidence * 100);
            const src = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/detections/${det.id}/frame`);
            return `<div class="tl-card-row" data-t="${det.t}">
                <img src="${esc(src)}" loading="lazy" alt="${esc(det.object_class)}">
                <div class="tl-card-text">
                    <span class="tl-card-class">${esc(det.object_class)}</span>
                    <span class="tl-card-conf">${conf}%</span>
                    <span class="tl-card-ago"></span>
                </div>
            </div>`;
        }).join('');
        const more = cluster.dets.length > 3
            ? `<div class="tl-card-more">+${cluster.dets.length - 3} more</div>`
            : '';

        marker.innerHTML = `
            <div class="tl-card">${rows}${more}</div>
            <div class="tl-dot">${cluster.dets.length > 1 ? cluster.dets.length : ''}</div>
        `;

        marker.querySelectorAll('.tl-card-row').forEach(row => {
            row.addEventListener('click', (e) => {
                e.stopPropagation();
                seekToTime(Number(row.dataset.t));
                closeMarkerCard();
            });
        });

        // Touch needs one tap to open the hover-equivalent card before seeking.
        marker.querySelector('.tl-dot').addEventListener('click', (e) => {
            e.stopPropagation();
            if (hoverCapable || openMarker === marker) {
                seekToTime(cluster.dets[0].t);
                closeMarkerCard();
            } else {
                closeMarkerCard();
                openMarker = marker;
                marker.classList.add('open');
            }
        });

        tlMarkers.appendChild(marker);
    });

    positionMarkers(r, nowS);
}

function positionMarkers(r, nowS) {
    tlMarkers.querySelectorAll('.tl-marker').forEach(marker => {
        const cluster = marker._cluster;
        const frac = timeToFrac(cluster.dets[0].t, r, nowS);
        if (frac < 0) {
            if (openMarker === marker) closeMarkerCard();
            marker.remove();
            return;
        }
        marker.style.left = (Math.min(1, frac) * 100) + '%';
        marker.classList.toggle('edge-l', frac < 0.12);
        marker.classList.toggle('edge-r', frac > 0.88);
        marker.querySelectorAll('.tl-card-ago').forEach((el, i) => {
            const det = cluster.dets[i];
            if (det) el.textContent = formatAgo(nowS - det.t);
        });
    });
}

function closeMarkerCard() {
    if (openMarker) {
        openMarker.classList.remove('open');
        openMarker = null;
    }
}

function formatAgo(secs) {
    if (secs < 60) return `${Math.round(secs)}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    return `${Math.round(secs / 3600)}h ago`;
}

function renderHistoryPanel() {
    if (warmEvents.length === 0) {
        historyPanel.hidden = true;
        return;
    }
    historyPanel.hidden = false;

    const groups = new Map();
    const sorted = [...warmEvents].sort((a, b) => b.start_ms - a.start_ms);
    sorted.forEach(ev => {
        const d = new Date(ev.start_ms);
        const key = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
        if (!groups.has(key)) groups.set(key, { label: formatDateLabel(d), events: [] });
        groups.get(key).events.push(ev);
    });

    const now = new Date();
    const todayFrac = (now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds()) / 86400;

    historyDays.innerHTML = '';
    groups.forEach(group => {
        const objects = group.events.filter(e => e.event_type === 'object').length;
        const continuous = group.events.filter(e => e.event_type === 'continuous').length;
        const row = document.createElement('div');
        row.className = 'history-day';

        const ticks = group.events.map(ev => {
            const d = new Date(ev.start_ms);
            const frac = (d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds()) / 86400;
            // Continuous chunks render as coverage, not hundreds of incidents.
            const cls = ev.event_type === 'object' ? ' obj'
                : ev.event_type === 'continuous' ? ' cont' : '';
            return `<span class="history-tick${cls}" style="left:${(frac * 100).toFixed(2)}%"></span>`;
        }).join('');
        const future = group.label === 'Today'
            ? `<span class="history-future" style="left:${(todayFrac * 100).toFixed(2)}%"></span>`
            : '';

        let counts = `${group.events.length} event${group.events.length !== 1 ? 's' : ''}`;
        if (objects > 0) {
            counts += ` \u00b7 ${objects} object${objects !== 1 ? 's' : ''}`;
        }
        if (continuous > 0) {
            counts += ` \u00b7 ${continuous} continuous chunk${continuous !== 1 ? 's' : ''}`;
        }

        row.innerHTML = `
            <div class="history-day-head">
                <span class="history-day-label">${esc(group.label)}</span>
                <span class="history-day-count">${counts}
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/></svg>
                </span>
            </div>
            <div class="history-map">${ticks}${future}</div>
        `;
        row.addEventListener('click', () => {
            eventsScrollDay = group.label;
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events`;
        });
        historyDays.appendChild(row);
    });
}
