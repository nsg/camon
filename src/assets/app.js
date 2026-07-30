document.addEventListener('DOMContentLoaded', async () => {
    // === DOM Elements ===

    // Grid view
    const gridView = document.getElementById('grid-view');
    const grid = document.getElementById('camera-grid');
    const noCameras = document.getElementById('no-cameras');

    // Token prompt (shown only when the API answers 401)
    const tokenPrompt = document.getElementById('token-prompt');
    const tokenInput = document.getElementById('token-input');
    const tokenSubmit = document.getElementById('token-submit');

    // View 1: Live Monitor
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
    const maskOverlay = document.getElementById('mask-overlay');
    const maskCtx = maskOverlay.getContext('2d');
    // Motion settings panel
    const settingsBtn = document.getElementById('settings-btn');
    const settingsPanel = document.getElementById('motion-settings-panel');
    const sensitivitySlider = document.getElementById('sensitivity-slider');
    const sensitivityValue = document.getElementById('sensitivity-value');
    const minsizeSlider = document.getElementById('minsize-slider');
    const minsizeValue = document.getElementById('minsize-value');
    const maskEditBtn = document.getElementById('mask-edit-btn');
    const maskLayerRow = document.getElementById('mask-layer-row');
    const maskLayerHint = document.getElementById('mask-layer-hint');
    const layerMovementBtn = document.getElementById('layer-movement-btn');
    const layerDetectionBtn = document.getElementById('layer-detection-btn');
    const settingsError = document.getElementById('settings-error');
    const settingsErrorText = document.getElementById('settings-error-text');
    const settingsErrorDismiss = document.getElementById('settings-error-dismiss');
    const liveBtn = document.getElementById('live-btn');

    // View 2: Event Browser
    const eventsView = document.getElementById('events-view');
    const eventsBackBtn = document.getElementById('events-back-btn');
    const eventsCameraName = document.getElementById('events-camera-name');
    const eventList = document.getElementById('event-list');
    const filterBtns = document.querySelectorAll('.filter-btn');

    // View 4: Detection Debug
    const debugView = document.getElementById('debug-view');
    const debugBackBtn = document.getElementById('debug-back-btn');
    const debugCameraName = document.getElementById('debug-camera-name');
    const debugList = document.getElementById('debug-list');
    const debugEmpty = document.getElementById('debug-empty');
    const debugLinkBtn = document.getElementById('debug-link-btn');
    let debugPoller = null;

    // View 3: Event Playback
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

    // === State ===
    let cameras = [];
    const gridHlsInstances = new Map();
    let detailHls = null;
    let playbackHls = null;
    let currentView = null;
    let isFirstLoad = true;
    let currentDetailCameraId = null;

    // Live monitor state. Timeline items carry absolute unix-seconds
    // timestamps (`t`, `tStart`, `tEnd`), derived from each fetch's
    // total_duration: the API's buffer offsets slide as segments evict, but
    // "seconds before the live edge at fetch time" anchored to the wall clock
    // does not.
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
    let warmEventPoller = null;
    let stabilityPoller = null;
    let stabilityOverlayEnabled = false;
    let stabilityDrawPending = false;
    let stabilityImage = null;
    let rawMog2Image = null;
    let noShadowImage = null;
    let morphImage = null;
    let bgOverlayEnabled = false;
    let bgImage = null;
    // Motion settings + mask editor. Two painted layers share the same 16x12
    // grid: the movement mask (suppresses motion detection) and the detection
    // mask (blacks pixels out of frames sent to the vision model). Painting
    // targets the active layer; both render at once in distinct colors.
    let motionSettings = null;
    let maskEditEnabled = false;
    let maskCells = [];
    let detectionCells = [];
    let activeMaskLayer = 'movement'; // 'movement' | 'detection'
    let maskCols = 16;
    let maskRows = 12;
    let maskPainting = false;
    let maskPaintValue = true;
    let overlayAnimationId = null;
    let isLiveScrubbing = false;
    let isAtLiveEdge = true;

    // Warm events (shared between views 2 & 3). `eventChains` maps an event's
    // key (see eventKey) to its place in a chain of `continues` chunks; see
    // buildEventChains.
    let warmEvents = [];
    let eventChains = new Map();
    let eventFilter = 'all';
    let eventsScrollDay = null;

    // Playback state
    let currentPlaybackKey = null;
    let playbackAnimationId = null;
    let isScrubbing = false;

    // === View Transition Helper ===
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

    // === API Auth ===
    // Optional server-side ([http] token in config.toml). When the server
    // requires one, every fetch and hls.js request carries it as a bearer
    // header; <img> and native <video> sources, which cannot set headers, fall
    // back to ?token=. A 401 means what we have is missing or stale, so we ask
    // for a new one and reload — every request then starts out authenticated.
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

    // === Polling ===

    // `setInterval` with an async callback lets a slow request still be in
    // flight when the next tick fires, and lets an older answer land after a
    // newer one and overwrite it. A poller instead runs one pass at a time and
    // arms the next timer only once the current pass has settled; `stop()`
    // aborts whatever that pass still has in flight and drops the timer, so a
    // poller that has been stopped issues nothing further. Requests started by
    // a *different* poller are not touched — each view stops its own.
    //
    // There is deliberately no watchdog: a pass that never settles holds its
    // poller until the view is torn down. Every fetch here is abortable and
    // subject to the browser's own network timeouts; the one wait with no
    // timeout of its own is an <img>, which is why loadOverlayImage caps it.
    function startPoller(name, intervalMs, pass) {
        const controller = new AbortController();
        let timer = null;
        let stopped = false;

        async function tick() {
            timer = null;
            try {
                await pass(controller.signal);
            } catch (err) {
                // Aborts are how stopping works, not a failure to report.
                if (err && err.name === 'AbortError') return;
                console.error(`Failed to fetch ${name}:`, err);
            }
            if (!stopped) timer = setTimeout(tick, intervalMs);
        }

        return {
            // Settles when the first pass is done, for callers that cannot
            // render until the data is there.
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

    tokenSubmit.addEventListener('click', submitToken);
    tokenInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') submitToken();
    });

    // === Initialize ===
    try {
        const response = await apiFetch('api/cameras');
        // A 401 already raised the token prompt, which covers the view.
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

    // === Router ===
    window.addEventListener('hashchange', router);
    router();

    function router() {
        const hash = window.location.hash || '#/';

        // #/camera/{id}/events/{key}, where {key} is an event key (see eventKey).
        // The type part is matched loosely, the way the event list already
        // tolerates a type it has no label for: a newer server's spelling should
        // still route to playback and let the server answer.
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

        // #/camera/{id}/debug
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

        // #/camera/{id}/events
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

        // #/camera/{id}
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

        // Default: grid
        if (currentView !== 'grid') {
            const isBack = currentView !== null;
            withViewTransition(() => showGridView(), isBack);
            currentView = 'grid';
        }
    }

    // === Event Listeners ===

    // View 1: Live Monitor
    backBtn.addEventListener('click', () => {
        window.location.hash = '/';
    });

    const volumeOnPath = 'M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z';
    const volumeOffPath = 'M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z';

    function updateMuteIcon(btn, video) {
        btn.querySelector('path').setAttribute('d', video.muted ? volumeOffPath : volumeOnPath);
        btn.classList.toggle('muted', video.muted);
    }

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
            noShadowImage = null;
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

    // === Motion settings panel ===

    settingsBtn.addEventListener('click', () => {
        const show = settingsPanel.hidden;
        settingsPanel.hidden = !show;
        settingsBtn.classList.toggle('active', show);
        if (show) {
            // Whatever failed last time the panel was open has been read or
            // ignored by now; reopening should not replay it.
            clearSettingsError();
        } else if (maskEditEnabled) {
            // Collapsing the panel exits mask-edit mode.
            setMaskEditEnabled(false);
        }
    });

    settingsErrorDismiss.addEventListener('click', clearSettingsError);

    sensitivitySlider.addEventListener('input', () => {
        sensitivityValue.textContent = sensitivitySlider.value;
    });
    sensitivitySlider.addEventListener('change', () => {
        putMotionSettings({ var_threshold: Number(sensitivitySlider.value) });
    });

    minsizeSlider.addEventListener('input', () => {
        minsizeValue.textContent = minsizeSlider.value;
    });
    minsizeSlider.addEventListener('change', () => {
        putMotionSettings({ min_contour_area: Number(minsizeSlider.value) });
    });

    maskEditBtn.addEventListener('click', () => {
        setMaskEditEnabled(!maskEditEnabled);
    });

    layerMovementBtn.addEventListener('click', () => setActiveMaskLayer('movement'));
    layerDetectionBtn.addEventListener('click', () => setActiveMaskLayer('detection'));

    // The cells array for the layer currently being painted.
    function activeCells() {
        return activeMaskLayer === 'detection' ? detectionCells : maskCells;
    }

    maskOverlay.addEventListener('pointerdown', (e) => {
        if (!maskEditEnabled) return;
        const idx = maskCellFromEvent(e);
        if (idx < 0) return;
        const cells = activeCells();
        maskPainting = true;
        maskPaintValue = !cells[idx];
        cells[idx] = maskPaintValue;
        drawMask();
        maskOverlay.setPointerCapture(e.pointerId);
        e.preventDefault();
    });
    maskOverlay.addEventListener('pointermove', (e) => {
        if (!maskEditEnabled || !maskPainting) return;
        const idx = maskCellFromEvent(e);
        if (idx < 0) return;
        const cells = activeCells();
        if (cells[idx] !== maskPaintValue) {
            cells[idx] = maskPaintValue;
            drawMask();
        }
    });
    function endMaskPaint() {
        if (!maskPainting) return;
        maskPainting = false;
        // Persist only the layer that was painted, as a partial update.
        if (activeMaskLayer === 'detection') {
            putMotionSettings({ detection_mask: detectionCells.slice() });
        } else {
            putMotionSettings({ mask: maskCells.slice() });
        }
    }
    maskOverlay.addEventListener('pointerup', endMaskPaint);
    maskOverlay.addEventListener('pointercancel', endMaskPaint);

    // Timeline scrubbing: drag anywhere on the track. Marker taps never reach
    // here (the early return), so seeking and marker cards don't fight.
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

    // Tapping outside an open marker card dismisses it (mobile two-step tap).
    document.addEventListener('click', (e) => {
        if (openMarker && !e.target.closest('.tl-marker')) closeMarkerCard();
    });

    liveBtn.addEventListener('click', () => {
        if (detailHls) {
            const seekable = detailVideo.seekable;
            if (seekable.length > 0) {
                detailVideo.currentTime = seekable.end(seekable.length - 1) - 0.5;
            }
            setLiveEdge(true);
        }
    });

    function setLiveEdge(atEdge) {
        isAtLiveEdge = atEdge;
        liveBtn.classList.toggle('active', atEdge);
    }

    // View 2: Event Browser
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

    // View 3: Event Playback
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

    // The overlay canvases are sized from the video box and hold images that
    // only change when a poll delivers a new one, so they are repainted when
    // one arrives and when that box changes — which is window resizes,
    // rotation, and the element taking its real size once the stream's
    // dimensions are known. Watching the element covers all three at once.
    new ResizeObserver(() => {
        drawBackground();
        scheduleStabilityDraw();
        drawMask();
    }).observe(detailVideo);

    // === View Functions ===

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
                // Matched on the property rather than through a selector, so a
                // camera id never has to survive being spliced into one.
                const cell = Array.from(grid.children).find(c => c.dataset.cameraId === cameraId);
                if (cell) {
                    loadGridCamera(cameraId, cell.querySelector('video'));
                }
            }
        });
    }

    function showLiveView(cameraId) {
        cleanupPlaybackView();
        cleanupDebugView();

        // Only reload if switching cameras
        if (currentDetailCameraId !== cameraId) {
            cleanupLiveView();
        }

        // Cleanup grid
        gridHlsInstances.forEach((hls) => hls.destroy());
        gridHlsInstances.clear();

        hideAllViews();
        liveView.hidden = false;
        detailCameraName.textContent = cameraId;
        currentDetailCameraId = cameraId;

        // Only start stream if not already running
        if (!detailHls) {
            detailLoading.hidden = false;
            stabilityOverlay.hidden = !stabilityOverlayEnabled;
            bgOverlay.hidden = !bgOverlayEnabled;

            if (stabilityOverlayEnabled) fetchStabilityMap();
            if (bgOverlayEnabled) fetchBackgroundMap();
            fetchMotionSettings(cameraId);

            loadDetailCamera(cameraId);
            fetchWarmEvents(cameraId);
        } else {
            // Returning from events/playback — just show the view
            renderHistoryPanel();
        }
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

    function showPlaybackView(cameraId, key) {
        cleanupDebugView();

        // Ensure warm events are available
        if (currentDetailCameraId !== cameraId) {
            currentDetailCameraId = cameraId;
            // Need to fetch warm events before we can show prev/next
            fetchWarmEvents(cameraId).then(() => {
                updatePlaybackNav();
            });
        }

        hideAllViews();
        playbackView.hidden = false;
        currentPlaybackKey = key;

        // Find the event
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

    // === Cleanup ===

    function cleanupLiveView() {
        if (overlayAnimationId) {
            cancelAnimationFrame(overlayAnimationId);
            overlayAnimationId = null;
        }
        if (motionPoller) { motionPoller.stop(); motionPoller = null; }
        if (detectionPoller) { detectionPoller.stop(); detectionPoller = null; }
        if (warmEventPoller) { warmEventPoller.stop(); warmEventPoller = null; }
        if (stabilityPoller) { stabilityPoller.stop(); stabilityPoller = null; }
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
        noShadowImage = null;
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

    // === Camera Cell ===

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

    // === HLS Loading ===

    function loadGridCamera(cameraId, video) {
        const src = `api/stream/${encodeURIComponent(cameraId)}/playlist.m3u8?live=true`;
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

    function loadDetailCamera(cameraId) {
        const src = `api/stream/${encodeURIComponent(cameraId)}/playlist.m3u8`;

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
                detailLoading.hidden = true;
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
                detailLoading.hidden = true;
                detailVideo.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
                startOverlayUpdates();
                fetchMotionSegments(cameraId);
                fetchDetections(cameraId);
            }, { once: true });
        } else {
            detailLoading.querySelector('p').textContent = 'HLS not supported';
        }
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

    // === Live Monitor: Overlay Updates ===

    // Only the playhead and the timeline follow playback closely enough to need
    // an animation frame. The overlays are painted from images that arrive on a
    // 5-second poll, and repainting them per frame meant recolouring up to four
    // video-sized layers pixel by pixel ~60 times a second to draw the same
    // thing; they redraw on arrival and on resize instead.
    function startOverlayUpdates() {
        if (overlayAnimationId) cancelAnimationFrame(overlayAnimationId);
        function update() {
            updateTimeline();
            overlayAnimationId = requestAnimationFrame(update);
        }
        update();
    }

    // === Live Monitor: Hot Timeline ===

    function timelineRange() {
        const seekable = detailVideo.seekable;
        if (seekable.length === 0) return null;
        const start = seekable.start(0);
        const end = seekable.end(seekable.length - 1);
        if (end - start <= 0) return null;
        // hls.js retains client-side back-buffer beyond the server's playlist
        // window, so `seekable` can outgrow the hot buffer. The timeline
        // covers only the server window (right-aligned at live); anything
        // older has no motion/detection data to show.
        let range = end - start;
        if (bufferDuration > 0 && bufferDuration < range) range = bufferDuration;
        return { start: end - range, end, range };
    }

    // Track fraction for an absolute unix-seconds timestamp: 1 = live edge,
    // 0 = the oldest scrubbable moment. Drifts left as time passes.
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
                if (timeToLive > 10) {
                    // Only explicit seeks leave live mode; this gap is drift
                    // from a buffer stall (or a background stay) — snap back.
                    detailVideo.currentTime = r.end - 0.5;
                }
            } else if (timeToLive < 3) {
                tlOffset.textContent = '';
                setLiveEdge(true);
            } else {
                tlOffset.textContent = '-' + formatTimeShort(timeToLive);
            }
        }

        // The histogram and marker positions only shift perceptibly by the
        // second; no need to repaint every frame.
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

        // motion_score has no fixed scale; normalize against the strongest
        // segment currently in view so the histogram always uses full height.
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
        // Re-render only when the window length changes noticeably.
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

    // <img> cannot set an Authorization header, so these carry the token in the
    // query string. A layer that fails to load simply doesn't paint.
    //
    // A stalled image fires neither load nor error until the browser gives up
    // on it, which can be minutes; since the overlay poller waits for these,
    // that would stall the overlays entirely rather than just skipping a round.
    // Giving up after three poll intervals returns them to a fresh attempt.
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

    // All four layers are swapped together once they have all arrived: they are
    // stages of one frame, so showing them mixed across two polls would draw a
    // mask that never existed — and it costs one redraw per poll instead of
    // four.
    async function fetchStabilityMap() {
        if (!stabilityOverlayEnabled || !currentDetailCameraId) return;
        const cameraId = currentDetailCameraId;
        const cam = encodeURIComponent(cameraId);
        const t = Date.now();

        const [raw, noShadow, morph, filtered] = await Promise.all([
            loadOverlayImage(`api/cameras/${cam}/motion/maps/raw?t=${t}`),
            loadOverlayImage(`api/cameras/${cam}/motion/maps/no-shadow?t=${t}`),
            loadOverlayImage(`api/cameras/${cam}/motion/maps/morph?t=${t}`),
            loadOverlayImage(`api/cameras/${cam}/motion/maps/stability?t=${t}`),
        ]);
        // The overlay may have been switched off, or the view moved to another
        // camera, while these were loading.
        if (!stabilityOverlayEnabled || currentDetailCameraId !== cameraId) return;

        rawMog2Image = raw;
        noShadowImage = noShadow;
        morphImage = morph;
        stabilityImage = filtered;
        scheduleStabilityDraw();
    }

    // Recolouring four video-sized layers is the most expensive thing this
    // page does, so several triggers landing in one frame get one redraw.
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

        // Paint layers bottom-to-top: each successive stage is a subset,
        // so later layers overwrite earlier ones where they overlap.
        // Layer 1: Raw MOG2 foreground mask (red) — largest area
        if (rawMog2Image) {
            recolorMask(rawMog2Image, w, h, 180, 60, 60, 150, 50);
        }
        // Layer 2: "no-shadow" stage (orange) — now an alias of the raw mask
        // (the pure-Rust detector has no shadow class), kept for continuity
        if (noShadowImage) {
            recolorMask(noShadowImage, w, h, 220, 140, 0, 160, 128);
        }
        // Layer 3: After morphological opening (yellow)
        if (morphImage) {
            recolorMask(morphImage, w, h, 240, 240, 0, 170, 128);
        }
        // Layer 4: Final filtered (green) — smallest area, always on top
        if (stabilityImage) {
            recolorMask(stabilityImage, w, h, 0, 255, 0, 180, 128);
        }
    }

    function recolorMask(img, w, h, r, g, b, a, threshold) {
        // Use a temporary canvas to avoid corrupting the main overlay between layers.
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

    // Same rule as the stability layers: what fails to load is shown as absent,
    // not as the last thing that worked. These overlays exist to say what the
    // detector is seeing right now, and a frame that has quietly stopped
    // updating reads exactly like a scene that has stopped changing.
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

    // === Live Monitor: Motion Settings + Ignore Mask ===

    function fetchMotionSettings(cameraId) {
        apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/motion/settings`)
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (!data || currentDetailCameraId !== cameraId) return;
                applyMotionSettings(data);
            })
            .catch(() => {});
    }

    function applyMotionSettings(data) {
        motionSettings = data;
        maskCols = data.mask_cols;
        maskRows = data.mask_rows;
        maskCells = Array.isArray(data.mask) ? data.mask.slice() : [];
        detectionCells = Array.isArray(data.detection_mask) ? data.detection_mask.slice() : [];

        sensitivitySlider.min = data.var_threshold_min;
        sensitivitySlider.max = data.var_threshold_max;
        sensitivitySlider.value = data.var_threshold;
        sensitivityValue.textContent = String(Math.round(data.var_threshold));

        minsizeSlider.min = data.min_contour_area_min;
        minsizeSlider.max = data.min_contour_area_max;
        minsizeSlider.value = data.min_contour_area;
        minsizeValue.textContent = String(Math.round(data.min_contour_area));

        if (maskEditEnabled) drawMask();
    }

    function showSettingsError(message) {
        settingsErrorText.textContent = message;
        settingsError.hidden = false;
    }

    function clearSettingsError() {
        settingsError.hidden = true;
        settingsErrorText.textContent = '';
    }

    // The server keeps a change it could not persist applied to the running
    // detector, so a failure here deliberately leaves the sliders and the
    // painted mask showing the new value — only the message says it will not
    // survive a restart. A network failure is a different answer and says so:
    // the change may or may not have reached the server at all.
    function putMotionSettings(partial) {
        if (!currentDetailCameraId) return;
        clearSettingsError();
        apiFetch(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/motion/settings`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(partial),
        })
            .then(r => {
                if (r.ok) return r.json().then(applyMotionSettings);
                // A 401 raises the token prompt, which is the whole message.
                if (r.status === 401) return;
                return r.text().then(body => showSettingsError(
                    body.trim() || `the server refused the change (HTTP ${r.status})`));
            })
            .catch(err => {
                console.error('Failed to update motion settings:', err);
                showSettingsError('could not reach camon — the change may or may not have been saved, reload to see what stuck');
            });
    }

    function setMaskEditEnabled(enabled) {
        maskEditEnabled = enabled;
        maskOverlay.hidden = !enabled;
        maskOverlay.classList.toggle('editing', enabled);
        maskEditBtn.classList.toggle('active', enabled);
        maskEditBtn.textContent = enabled ? 'Done editing masks' : 'Edit masks';
        maskLayerRow.hidden = !enabled;
        if (enabled) {
            // The edit button sits below the video, so activating from there can
            // leave the paintable canvas scrolled partly above the viewport —
            // bring the whole video wrapper into view so every mask row is reachable.
            detailVideo.parentElement.scrollIntoView({ block: 'start', behavior: 'smooth' });
            drawMask();
        } else {
            maskPainting = false;
            maskCtx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
        }
    }

    function setActiveMaskLayer(layer) {
        activeMaskLayer = layer;
        const detection = layer === 'detection';
        layerMovementBtn.classList.toggle('active', !detection);
        layerDetectionBtn.classList.toggle('active', detection);
        maskLayerHint.textContent = detection
            ? 'Detection mask: the vision model never sees these pixels (classification only).'
            : 'Movement mask: nothing ever moves here (ignored by motion detection).';
    }

    function maskCellFromEvent(e) {
        const rect = maskOverlay.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return -1;
        const col = Math.floor((e.clientX - rect.left) / rect.width * maskCols);
        const row = Math.floor((e.clientY - rect.top) / rect.height * maskRows);
        if (col < 0 || col >= maskCols || row < 0 || row >= maskRows) return -1;
        return row * maskCols + col;
    }

    function drawMask() {
        if (!maskEditEnabled) return;
        const w = detailVideo.clientWidth;
        const h = detailVideo.clientHeight;
        if (w === 0 || h === 0) return;
        if (maskOverlay.width !== w || maskOverlay.height !== h) {
            maskOverlay.width = w;
            maskOverlay.height = h;
        }
        maskCtx.clearRect(0, 0, w, h);

        const cellW = w / maskCols;
        const cellH = h / maskRows;

        // Both layers render at once in distinct colors so overlaps are visible:
        // movement mask (ignored motion) in red, detection mask (blacked out of
        // the vision model) in orange. Overlapping cells simply blend.
        const fillLayer = (cells, color) => {
            maskCtx.fillStyle = color;
            for (let i = 0; i < cells.length; i++) {
                if (!cells[i]) continue;
                const col = i % maskCols;
                const row = Math.floor(i / maskCols);
                maskCtx.fillRect(col * cellW, row * cellH, cellW, cellH);
            }
        };
        fillLayer(maskCells, 'rgba(220, 50, 50, 0.4)');
        fillLayer(detectionCells, 'rgba(255, 140, 0, 0.45)');

        // Grid lines.
        maskCtx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
        maskCtx.lineWidth = 1;
        for (let c = 1; c < maskCols; c++) {
            const x = Math.round(c * cellW) + 0.5;
            maskCtx.beginPath();
            maskCtx.moveTo(x, 0);
            maskCtx.lineTo(x, h);
            maskCtx.stroke();
        }
        for (let r = 1; r < maskRows; r++) {
            const y = Math.round(r * cellH) + 0.5;
            maskCtx.beginPath();
            maskCtx.moveTo(0, y);
            maskCtx.lineTo(w, y);
            maskCtx.stroke();
        }
    }

    // === Live Monitor: Data Fetching ===

    function fetchMotionSegments(cameraId) {
        if (motionPoller) motionPoller.stop();
        motionPoller = startPoller('motion data', 5000, async (signal) => {
            const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/motion`, { signal });
            if (currentDetailCameraId !== cameraId || !response.ok) return;
            const data = await response.json();
            if (data.total_duration > 0) {
                bufferDuration = data.total_duration;
            }
            // Segment offsets are relative to the (sliding) buffer
            // start; anchor them to the wall clock via the live edge.
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
            // Rebuilding closes any open card, so skip when nothing
            // changed or while the user is reading one.
            if (ids !== lastDetIds && !openMarker) {
                rebuildMarkers();
            }
        });
    }

    // The key that identifies one stored event, everywhere: in the playback
    // route, in the chain map, and in every event URL the server answers.
    //
    // A start PTS alone identifies nothing — two recordings can begin on the
    // same keyframe (a motion event and the continuous chunk covering it), and
    // asking for one of them by start used to serve whichever the server found
    // first. So the key carries the duration and the type as well, in the
    // spelling the API uses: `{start_pts_ns}_{duration_ms}_{event_type}`.
    //
    // Built by string concatenation, never by arithmetic: `start_pts_ns` arrives
    // as a JSON string because nanoseconds since the epoch are past the range
    // Number holds exactly, and a key that had been through Number would name an
    // event the server does not have.
    function eventKey(ev) {
        return `${ev.start_pts_ns}_${ev.duration_ms}_${ev.event_type}`;
    }

    function fetchWarmEvents(cameraId) {
        if (warmEventPoller) warmEventPoller.stop();
        warmEventPoller = startPoller('warm events', 15000, async (signal) => {
            const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/events`, { signal });
            if (currentDetailCameraId !== cameraId || !response.ok) return;
            const raw = await response.json();
            warmEvents = raw.map(ev => ({
                ...ev,
                key: eventKey(ev),
                start_ms: Number(BigInt(ev.start_pts_ns) / 1_000_000n),
            }));
            eventChains = buildEventChains(warmEvents);
            renderHistoryPanel();
            // Re-render event list if visible
            if (!eventsView.hidden) renderEventList();
            // Update nav if in playback
            if (!playbackView.hidden) updatePlaybackNav();
        });
        // Callers that cannot render until the events are here await this.
        return warmEventPoller.first;
    }

    // === Live Monitor: Detection Markers ===

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

        // Newest first, so a cluster's dot and top card row show the latest
        // sighting at that spot.
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

            // Desktop (hover shows the card): click seeks. Touch: first tap
            // opens the card, tapping the dot again seeks.
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
                // Slid out of the buffer.
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

    // === Live Monitor: Warm History Day Maps ===

    function renderHistoryPanel() {
        if (warmEvents.length === 0) {
            historyPanel.hidden = true;
            return;
        }
        historyPanel.hidden = false;

        // One row per local calendar day that has events, newest day first.
        // Days shown follow the data, so server retention config (which can
        // differ per event type) needs no mirroring here.
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
                // Continuous chunks are a recording, not an incident: they get
                // a dim full-height tick so a day of them reads as coverage
                // rather than as hundreds of things that happened.
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

    // === View 2: Event List Rendering ===

    // The event types the API can send, exhaustively (see WarmEventResponse in
    // src/api/server.rs). Calling everything that isn't an object "Movement"
    // mislabelled continuous-recording chunks as something that was detected.
    // An unknown type means a newer server talking to an older UI: show the
    // wire name rather than quietly filing it under the wrong one.
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

    // `continues` marks a chunk that carries on the one before it: a motion run
    // split at max_event_duration_secs, or the fixed-length chunks continuous
    // recording rolls. Walking the events oldest-first turns those flags into
    // runs, so each chunk can say where it sits in one recording instead of
    // standing there as an unrelated event.
    //
    // The flag alone does not identify the predecessor, and outliving it is the
    // normal case rather than a corner: a movement chunk expires after
    // movement_retention_days while the object chunk that continues it keeps
    // the flag and the far longer object retention, and continuous chunks
    // expire daily while the flag says "not the first chunk since startup"
    // forever. So a chunk joins the run only if it also starts where the
    // previous one ended — both producers emit strictly contiguous chunks (the
    // follow-on begins at the segment after the last one written, with no
    // pre-padding), which makes adjacency an exact test rather than a
    // heuristic. A second of slack absorbs millisecond rounding.
    const CHUNK_GAP_TOLERANCE_MS = 2000;

    function buildEventChains(events) {
        // event key -> { part, headKnown }
        const chains = new Map();
        const ascending = [...events].sort((a, b) => a.start_ms - b.start_ms);
        let run = [];
        const flush = () => {
            // A run whose first chunk is itself flagged `continues` reaches back
            // past what the archive still holds: its head has been pruned, so
            // nothing here can say which part of the recording these are — only
            // that they carry one on. A run with its head intact can be counted
            // forward, but never totalled: the recording may still be growing.
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

    // "part 3" counts from a head we can actually see. "continued" is all that
    // can honestly be said when we cannot: a number counted from the oldest
    // surviving chunk would claim a position in the recording that it does not
    // have, and would change on every retention sweep.
    function chainPartLabel(ev) {
        const chain = eventChains.get(ev.key);
        if (!chain) return '';
        if (!chain.headKnown) return ' · continued';
        return chain.length > 1 ? ` · part ${chain.part}` : '';
    }

    // Whether this event is a chunk of a longer recording — either because more
    // of that recording is on screen, or because it continues one that has been
    // pruned away.
    function isChunkOfRun(ev) {
        const chain = eventChains.get(ev.key);
        return !!chain && (chain.length > 1 || !chain.headKnown);
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

    // === View 3: Playback Controls ===

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
        // Sort events by time descending (newest first) to match event list order
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

    // === Detection Debug ===

    debugBackBtn.addEventListener('click', () => {
        window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}`;
    });

    debugLinkBtn.addEventListener('click', () => {
        window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/debug`;
    });

    function showDebugView(cameraId) {
        cleanupLiveView();
        cleanupPlaybackView();
        cleanupDebugView();
        hideAllViews();
        debugView.hidden = false;
        debugCameraName.textContent = `${cameraId} — Detection Debug`;
        currentDetailCameraId = cameraId;
        debugPoller = startPoller('debug entries', 5000, async (signal) => {
            const res = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/detection-debug`, { signal });
            if (!res.ok || currentDetailCameraId !== cameraId) return;
            renderDebugList(cameraId, await res.json());
        });
    }

    // Called by every other view as well: leaving the debug view is otherwise
    // the one way to walk away from a poller that keeps running.
    function cleanupDebugView() {
        if (debugPoller) {
            debugPoller.stop();
            debugPoller = null;
        }
    }

    function renderDebugList(cameraId, entries) {
        if (entries.length === 0) {
            debugList.innerHTML = '';
            debugEmpty.hidden = false;
            return;
        }
        debugEmpty.hidden = true;

        // Show newest first
        const reversed = [...entries].reverse();
        const encodedId = encodeURIComponent(cameraId);

        debugList.innerHTML = reversed.map(entry => {
            const date = new Date(entry.timestamp * 1000);
            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const detectionBadge = entry.detection_count > 0
                ? `<span class="debug-badge positive">${entry.detection_count} detection${entry.detection_count !== 1 ? 's' : ''}</span>`
                : '<span class="debug-badge none">no detections</span>';

            const fullFrameHtml = entry.has_full_frame
                ? `<div class="debug-full-frame-container">
                    <div class="debug-full-frame-wrap">
                        <img class="debug-full-frame-image" data-entry-id="${esc(entry.id)}" src="${esc(authUrl(`api/cameras/${encodedId}/detection-debug/${entry.id}/full-frame`))}" alt="Full frame" loading="lazy">
                        <canvas class="debug-overlay-canvas" data-entry-id="${esc(entry.id)}"></canvas>
                    </div>
                    <div class="debug-overlay-legend">
                        <span class="debug-legend-item"><span class="debug-legend-color" style="border-color:#0f0"></span>Motion</span>
                        <span class="debug-legend-item"><span class="debug-legend-color" style="border-color:#ff0"></span>Crop</span>
                        <span class="debug-legend-item"><span class="debug-legend-color" style="border-color:#f44"></span>Ollama</span>
                    </div>
                </div>`
                : '';

            const framesHtml = Array.from({length: entry.frame_count}, (_, i) => {
                const response = esc(entry.raw_responses[i] || '(no response)');
                return `<div class="debug-frame-pair">
                    <img class="debug-frame-image" src="${esc(authUrl(`api/cameras/${encodedId}/detection-debug/${entry.id}/frame/${i}`))}" alt="Frame ${i + 1}" loading="lazy">
                    <pre class="debug-raw-response">${response}</pre>
                </div>`;
            }).join('');

            return `<div class="debug-entry">
                <div class="debug-entry-header">
                    <span class="debug-time">${timeStr}</span>
                    <span class="debug-model">${esc(entry.model)}</span>
                    ${detectionBadge}
                </div>
                ${fullFrameHtml}
                <div class="debug-frames">${framesHtml}</div>
            </div>`;
        }).join('');

        // Draw overlays once full-frame images load
        debugList.querySelectorAll('.debug-full-frame-image').forEach(img => {
            const entryId = img.dataset.entryId;
            const entry = entries.find(e => String(e.id) === entryId);
            if (!entry) return;
            const draw = () => drawDebugOverlay(img, entry);
            if (img.complete && img.naturalWidth) draw();
            else img.addEventListener('load', draw);
        });
    }

    function drawDebugOverlay(img, entry) {
        const canvas = img.parentElement.querySelector('.debug-overlay-canvas');
        if (!canvas) return;
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);

        // Motion rects — green
        ctx.strokeStyle = '#0f0';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        for (const [rx, ry, rw, rh] of entry.motion_rects) {
            ctx.strokeRect(rx * w, ry * h, rw * w, rh * h);
        }

        // Crop rect — yellow
        ctx.setLineDash([]);
        if (entry.crop_rect) {
            const [cx, cy, cw, ch] = entry.crop_rect;
            ctx.strokeStyle = '#ff0';
            ctx.lineWidth = 2;
            ctx.strokeRect(cx * w, cy * h, cw * w, ch * h);
            ctx.fillStyle = 'rgba(255,255,0,0.06)';
            ctx.fillRect(cx * w, cy * h, cw * w, ch * h);
        }

        // Ollama rects — red with class label
        ctx.setLineDash([]);
        ctx.lineWidth = 2;
        ctx.font = `bold ${Math.max(12, Math.round(h * 0.025))}px monospace`;
        ctx.textBaseline = 'bottom';
        for (const [className, ox, oy, ow, oh] of entry.ollama_rects) {
            ctx.strokeStyle = '#f44';
            ctx.strokeRect(ox * w, oy * h, ow * w, oh * h);
            const label = className;
            const tx = ox * w + 2;
            const ty = oy * h - 3;
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            const tm = ctx.measureText(label);
            ctx.fillRect(tx - 1, ty - tm.actualBoundingBoxAscent - 2, tm.width + 4, tm.actualBoundingBoxAscent + 4);
            ctx.fillStyle = '#f44';
            ctx.fillText(label, tx, ty);
        }
    }

    // === Utility ===

    // Every value interpolated into an innerHTML template goes through here.
    // Camera ids, object classes and model names come from operator config and
    // from metadata camon has already filtered, not straight off the network —
    // but "trusted enough" is not something the markup can check, and one
    // escaped hole beside four raw ones is the bug regardless of who fills it.
    function esc(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatTime(seconds) {
        if (!isFinite(seconds)) return '00:00:00';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    function formatTimeShort(seconds) {
        if (!isFinite(seconds)) return '0:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
});
