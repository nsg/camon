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
    let debugPollInterval = null;

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
    let motionPollInterval = null;
    let detectionPollInterval = null;
    let warmEventPollInterval = null;
    let stabilityPollInterval = null;
    let stabilityOverlayEnabled = false;
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

    // Warm events (shared between views 2 & 3)
    let warmEvents = [];
    let eventFilter = 'all';
    let eventsScrollDay = null;

    // Playback state
    let currentPlaybackPts = null;
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

        // #/camera/{id}/events/{pts}
        const playbackMatch = hash.match(/^#\/camera\/(.+)\/events\/(\d+)$/);
        if (playbackMatch) {
            const cameraId = decodeURIComponent(playbackMatch[1]);
            const pts = playbackMatch[2];
            if (cameras.includes(cameraId)) {
                const targetView = `playback:${cameraId}:${pts}`;
                if (currentView !== targetView) {
                    const isBack = currentView && currentView.startsWith('playback:');
                    withViewTransition(() => showPlaybackView(cameraId, pts), isBack);
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
        const nav = getAdjacentEvents(currentPlaybackPts);
        if (nav.prev) {
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${nav.prev.start_pts_ns}`;
        }
    });

    nextEventBtn.addEventListener('click', () => {
        const nav = getAdjacentEvents(currentPlaybackPts);
        if (nav.next) {
            window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${nav.next.start_pts_ns}`;
        }
    });

    // Resize handler for overlays
    window.addEventListener('resize', () => {
        drawBackground();
        drawStability();
        drawMask();
    });

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
        hideAllViews();
        gridView.hidden = false;

        cameras.forEach(cameraId => {
            if (!gridHlsInstances.has(cameraId)) {
                const cell = grid.querySelector(`[data-camera-id="${cameraId}"]`);
                if (cell) {
                    loadGridCamera(cameraId, cell.querySelector('video'));
                }
            }
        });
    }

    function showLiveView(cameraId) {
        cleanupPlaybackView();

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

    function showPlaybackView(cameraId, pts) {
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
        currentPlaybackPts = pts;

        // Find the event
        const ev = warmEvents.find(e => e.start_pts_ns === pts);
        if (ev) {
            const evDate = new Date(ev.start_ms);
            const typeLabel = ev.event_type === 'object' ? 'Object' : 'Motion';
            const timeStr = evDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            playbackEventInfo.textContent = `${typeLabel} \u00b7 ${timeStr}`;
        } else {
            playbackEventInfo.textContent = 'Event';
        }

        loadPlaybackVideo(cameraId, pts);
        updatePlaybackNav();
    }

    // === Cleanup ===

    function cleanupLiveView() {
        if (overlayAnimationId) {
            cancelAnimationFrame(overlayAnimationId);
            overlayAnimationId = null;
        }
        if (motionPollInterval) { clearInterval(motionPollInterval); motionPollInterval = null; }
        if (detectionPollInterval) { clearInterval(detectionPollInterval); detectionPollInterval = null; }
        if (warmEventPollInterval) { clearInterval(warmEventPollInterval); warmEventPollInterval = null; }
        if (stabilityPollInterval) { clearInterval(stabilityPollInterval); stabilityPollInterval = null; }
        if (detailHls) { detailHls.destroy(); detailHls = null; }
        detailVideo.src = '';
        currentDetections = [];
        motionSegs = [];
        lastDetIds = null;
        currentDetailCameraId = null;
        bufferDuration = 0;
        warmEvents = [];

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
        currentPlaybackPts = null;
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
            <span class="camera-label">${cameraId}</span>
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
        const src = `api/stream/${cameraId}/playlist.m3u8?live=true`;
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
        const src = `api/stream/${cameraId}/playlist.m3u8`;

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

    function loadPlaybackVideo(cameraId, pts) {
        const src = `api/cameras/${encodeURIComponent(cameraId)}/events/${pts}/playlist.m3u8`;

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

    function startOverlayUpdates() {
        function update() {
            drawBackground();
            drawStability();
            drawMask();
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

    function fetchStabilityMap() {
        if (!stabilityOverlayEnabled || !currentDetailCameraId) return;
        const cam = encodeURIComponent(currentDetailCameraId);
        const t = Date.now();

        const img = new Image();
        img.onload = () => { stabilityImage = img; drawStability(); };
        img.onerror = () => {};
        img.src = authUrl(`api/cameras/${cam}/motion/stability?t=${t}`);

        const raw = new Image();
        raw.onload = () => { rawMog2Image = raw; drawStability(); };
        raw.onerror = () => {};
        raw.src = authUrl(`api/cameras/${cam}/motion/stability/raw?t=${t}`);

        const noShadow = new Image();
        noShadow.onload = () => { noShadowImage = noShadow; drawStability(); };
        noShadow.onerror = () => {};
        noShadow.src = authUrl(`api/cameras/${cam}/motion/stability/no-shadow?t=${t}`);

        const morph = new Image();
        morph.onload = () => { morphImage = morph; drawStability(); };
        morph.onerror = () => {};
        morph.src = authUrl(`api/cameras/${cam}/motion/stability/morph?t=${t}`);
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

    function fetchBackgroundMap() {
        if (!bgOverlayEnabled || !currentDetailCameraId) return;
        const img = new Image();
        img.onload = () => { bgImage = img; drawBackground(); };
        img.onerror = () => {};
        img.src = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/motion/background?t=${Date.now()}`);
    }

    function drawBackground() {
        if (!bgImage) return;
        const w = detailVideo.clientWidth;
        const h = detailVideo.clientHeight;
        if (w === 0 || h === 0) return;
        if (bgOverlay.width !== w || bgOverlay.height !== h) {
            bgOverlay.width = w;
            bgOverlay.height = h;
        }
        bgCtx.clearRect(0, 0, w, h);
        bgCtx.drawImage(bgImage, 0, 0, w, h);
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

    async function fetchMotionSegments(cameraId) {
        if (motionPollInterval) clearInterval(motionPollInterval);

        async function poll() {
            try {
                const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/motion`);
                if (currentDetailCameraId !== cameraId) return;
                if (response.ok) {
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
                }
            } catch (err) {
                console.error('Failed to fetch motion data:', err);
            }
        }

        await poll();
        motionPollInterval = setInterval(poll, 5000);

        if (stabilityPollInterval) clearInterval(stabilityPollInterval);
        stabilityPollInterval = setInterval(() => {
            fetchStabilityMap();
            fetchBackgroundMap();
        }, 5000);
    }

    async function fetchDetections(cameraId) {
        if (detectionPollInterval) clearInterval(detectionPollInterval);

        async function poll() {
            try {
                const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/detections`);
                if (currentDetailCameraId !== cameraId) return;
                if (response.ok) {
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
                }
            } catch (err) {
                console.error('Failed to fetch detection data:', err);
            }
        }

        await poll();
        detectionPollInterval = setInterval(poll, 5000);
    }

    async function fetchWarmEvents(cameraId) {
        if (warmEventPollInterval) clearInterval(warmEventPollInterval);

        async function poll() {
            try {
                const response = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/events`);
                if (currentDetailCameraId !== cameraId) return;
                if (response.ok) {
                    const raw = await response.json();
                    warmEvents = raw.map(ev => ({
                        ...ev,
                        start_ms: Number(BigInt(ev.start_pts_ns) / 1_000_000n),
                    }));
                    renderHistoryPanel();
                    // Re-render event list if visible
                    if (!eventsView.hidden) renderEventList();
                    // Update nav if in playback
                    if (!playbackView.hidden) updatePlaybackNav();
                }
            } catch (err) {
                console.error('Failed to fetch warm events:', err);
            }
        }

        await poll();
        warmEventPollInterval = setInterval(poll, 15000);
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
                    <img src="${src}" loading="lazy" alt="${det.object_class}">
                    <div class="tl-card-text">
                        <span class="tl-card-class">${det.object_class}</span>
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
            const row = document.createElement('div');
            row.className = 'history-day';

            const ticks = group.events.map(ev => {
                const d = new Date(ev.start_ms);
                const frac = (d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds()) / 86400;
                const cls = ev.event_type === 'object' ? ' obj' : '';
                return `<span class="history-tick${cls}" style="left:${(frac * 100).toFixed(2)}%"></span>`;
            }).join('');
            const future = group.label === 'Today'
                ? `<span class="history-future" style="left:${(todayFrac * 100).toFixed(2)}%"></span>`
                : '';

            let counts = `${group.events.length} event${group.events.length !== 1 ? 's' : ''}`;
            if (objects > 0) {
                counts += ` \u00b7 ${objects} object${objects !== 1 ? 's' : ''}`;
            }

            row.innerHTML = `
                <div class="history-day-head">
                    <span class="history-day-label">${group.label}</span>
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

        let filtered = warmEvents;
        if (eventFilter === 'object') {
            filtered = warmEvents.filter(e => e.event_type === 'object');
        } else if (eventFilter === 'movement') {
            filtered = warmEvents.filter(e => e.event_type === 'movement');
        }

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

                const thumbSrc = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${ev.start_pts_ns}/thumbnail`);
                const evDate = new Date(ev.start_ms);
                const timeStr = evDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const durSec = (ev.duration_ms / 1000).toFixed(1);
                const typeLabel = ev.event_type === 'object' ? 'Object detected' : 'Movement';
                const typeClass = ev.event_type === 'object' ? 'object' : 'movement';

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
                        Array.from({ length: ev.filmstrip_frames }, (_, i) => `<img class="filmstrip-frame" src="${authUrl(`api/cameras/${cid}/events/${ev.start_pts_ns}/filmstrip/${i}`)}" loading="lazy" alt="" onerror="this.style.display='none'">`).join('') +
                        `</div>`;
                } else {
                    thumbHtml = `<img class="event-list-thumb" src="${thumbSrc}" loading="lazy" alt="">`;
                }

                const detailText = ev.event_type === 'object' && ev.object_classes ? ev.object_classes.join(', ') : '';

                item.innerHTML = `
                    ${thumbHtml}
                    <div class="event-list-info">
                        <div class="event-list-type ${typeClass}">${typeLabel}${recoveredBadge}</div>
                        <div class="event-list-detail">${detailText}</div>
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
                    window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/events/${ev.start_pts_ns}`;
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

    function getAdjacentEvents(pts) {
        // Sort events by time descending (newest first) to match event list order
        const sorted = [...warmEvents].sort((a, b) => b.start_ms - a.start_ms);
        const idx = sorted.findIndex(e => e.start_pts_ns === pts);
        return {
            prev: idx > 0 ? sorted[idx - 1] : null,
            next: idx >= 0 && idx < sorted.length - 1 ? sorted[idx + 1] : null,
        };
    }

    function updatePlaybackNav() {
        const nav = getAdjacentEvents(currentPlaybackPts);

        if (nav.prev) {
            prevEventBtn.hidden = false;
            const prevDate = new Date(nav.prev.start_ms);
            prevEventText.textContent = prevDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            prevEventThumb.src = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${nav.prev.start_pts_ns}/thumbnail`);
        } else {
            prevEventBtn.hidden = true;
        }

        if (nav.next) {
            nextEventBtn.hidden = false;
            const nextDate = new Date(nav.next.start_ms);
            nextEventText.textContent = nextDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            nextEventThumb.src = authUrl(`api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${nav.next.start_pts_ns}/thumbnail`);
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
        fetchDebugEntries(cameraId);
        debugPollInterval = setInterval(() => fetchDebugEntries(cameraId), 5000);
    }

    function cleanupDebugView() {
        if (debugPollInterval) {
            clearInterval(debugPollInterval);
            debugPollInterval = null;
        }
    }

    async function fetchDebugEntries(cameraId) {
        try {
            const res = await apiFetch(`api/cameras/${encodeURIComponent(cameraId)}/detection-debug`);
            if (!res.ok) return;
            const entries = await res.json();
            renderDebugList(cameraId, entries);
        } catch (e) {
            console.error('Failed to fetch debug entries:', e);
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
                        <img class="debug-full-frame-image" data-entry-id="${entry.id}" src="${authUrl(`api/cameras/${encodedId}/detection-debug/${entry.id}/full-frame`)}" alt="Full frame" loading="lazy">
                        <canvas class="debug-overlay-canvas" data-entry-id="${entry.id}"></canvas>
                    </div>
                    <div class="debug-overlay-legend">
                        <span class="debug-legend-item"><span class="debug-legend-color" style="border-color:#0f0"></span>Motion</span>
                        <span class="debug-legend-item"><span class="debug-legend-color" style="border-color:#ff0"></span>Crop</span>
                        <span class="debug-legend-item"><span class="debug-legend-color" style="border-color:#f44"></span>Ollama</span>
                    </div>
                </div>`
                : '';

            const framesHtml = Array.from({length: entry.frame_count}, (_, i) => {
                const response = (entry.raw_responses[i] || '(no response)')
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
                return `<div class="debug-frame-pair">
                    <img class="debug-frame-image" src="${authUrl(`api/cameras/${encodedId}/detection-debug/${entry.id}/frame/${i}`)}" alt="Frame ${i + 1}" loading="lazy">
                    <pre class="debug-raw-response">${response}</pre>
                </div>`;
            }).join('');

            return `<div class="debug-entry">
                <div class="debug-entry-header">
                    <span class="debug-time">${timeStr}</span>
                    <span class="debug-model">${entry.model}</span>
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
