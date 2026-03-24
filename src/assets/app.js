document.addEventListener('DOMContentLoaded', async () => {
    // DOM elements
    const gridView = document.getElementById('grid-view');
    const grid = document.getElementById('camera-grid');
    const noCameras = document.getElementById('no-cameras');
    const detailView = document.getElementById('detail-view');
    const detailVideo = document.getElementById('detail-video');
    const detailLoading = document.getElementById('detail-loading');
    const detailCameraName = document.getElementById('detail-camera-name');
    const backBtn = document.getElementById('back-btn');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    const currentTimeDisplay = document.getElementById('current-time');
    const durationDisplay = document.getElementById('duration');
    const liveBtn = document.getElementById('live-btn');
    const timelineCanvas = document.getElementById('timeline-canvas');
    const timelineCtx = timelineCanvas.getContext('2d');
    const detectionTooltip = document.getElementById('detection-tooltip');
    const tooltipImage = document.getElementById('tooltip-image');
    const tooltipLabel = document.getElementById('tooltip-label');
    const stabilityOverlay = document.getElementById('stability-overlay');
    const stabilityCtx = stabilityOverlay.getContext('2d');
    const maskToggleBtn = document.getElementById('mask-toggle-btn');
    const muteToggleBtn = document.getElementById('mute-toggle-btn');
    const detectionGallery = document.getElementById('detection-gallery');
    const hoverTime = document.getElementById('hover-time');
    const timelineWrapper = document.querySelector('.timeline-wrapper');
    const zoomButtons = document.querySelectorAll('.zoom-btn');
    const recordingsSection = document.getElementById('recordings-section');
    const recordingsGroups = document.getElementById('recordings-groups');
    const RECORDINGS_PER_PAGE = 12;
    const collapsedGroups = new Map();
    const groupPageLimits = new Map();

    // State
    let cameras = [];
    const gridHlsInstances = new Map();
    let detailHls = null;
    let timelineAnimationId = null;
    let isSeeking = false;
    let currentView = null;
    let isFirstLoad = true;
    let currentMotionSegments = [];
    let currentDetections = [];
    let motionPollInterval = null;
    let detectionPollInterval = null;
    let warmEventPollInterval = null;
    let currentDetailCameraId = null;
    let bufferDuration = 0;
    // Overlay: 'off' -> 'stability' -> 'off'
    let overlayMode = 'off';
    let stabilityOverlayEnabled = false;
    let stabilityImage = null;
    let stabilityPollInterval = null;

    // Warm event state
    let warmEvents = [];
    let eventStripZoomHours = 24;
    let isPlayingWarmEvent = false;
    let currentWarmEventPts = null;

    // Timeline drag state
    let isDraggingTimeline = false;
    let dragTarget = null; // 'buffer' or 'warm'

    // View transition helper
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

    // Initialize
    try {
        const response = await fetch('/api/cameras');
        cameras = await response.json();

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

    // Start router
    window.addEventListener('hashchange', router);
    router();

    // Event listeners
    backBtn.addEventListener('click', () => {
        window.location.hash = '/';
    });

    timelineScrubber.addEventListener('input', () => {
        isSeeking = true;
        const duration = isPlayingWarmEvent ? detailVideo.duration : (bufferDuration || detailVideo.duration);
        if (!isPlayingWarmEvent) updateLiveState();
    });

    timelineScrubber.addEventListener('change', () => {
        const duration = isPlayingWarmEvent ? detailVideo.duration : (bufferDuration || detailVideo.duration);
        const time = (timelineScrubber.value / 100) * duration;
        detailVideo.currentTime = time;
        isSeeking = false;
    });

    liveBtn.addEventListener('click', () => {
        if (isPlayingWarmEvent) {
            returnToLive();
            return;
        }
        const duration = bufferDuration || detailVideo.duration;
        if (duration && isFinite(duration)) {
            detailVideo.currentTime = duration;
            updateLiveState();
        }
    });

    const volumeOnPath = 'M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z';
    const volumeOffPath = 'M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z';

    function updateMuteIcon() {
        muteToggleBtn.querySelector('path').setAttribute('d', detailVideo.muted ? volumeOffPath : volumeOnPath);
        muteToggleBtn.classList.toggle('muted', detailVideo.muted);
    }

    muteToggleBtn.addEventListener('click', () => {
        detailVideo.muted = !detailVideo.muted;
        updateMuteIcon();
    });

    maskToggleBtn.addEventListener('click', () => {
        overlayMode = overlayMode === 'off' ? 'stability' : 'off';
        stabilityOverlayEnabled = overlayMode === 'stability';
        maskToggleBtn.classList.toggle('active', stabilityOverlayEnabled);
        stabilityOverlay.hidden = !stabilityOverlayEnabled;
        if (!stabilityOverlayEnabled) {
            stabilityCtx.clearRect(0, 0, stabilityOverlay.width, stabilityOverlay.height);
            stabilityImage = null;
        } else {
            fetchStabilityMap();
        }
        maskToggleBtn.title = `Overlay: ${overlayMode}`;
    });

    // Zoom button listeners
    zoomButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            zoomButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            eventStripZoomHours = parseInt(btn.dataset.hours, 10);
            renderTimeline();
        });
    });

    // Timeline seek helpers
    function getTimelineRatio(clientX) {
        const rect = timelineWrapper.getBoundingClientRect();
        return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    }

    function getBufferBounds() {
        const windowMs = eventStripZoomHours * 3600_000;
        const bufferMs = (bufferDuration || 0) * 1000;
        const bufferRatio = bufferMs / windowMs;
        return { startRatio: 1.0 - bufferRatio, bufferRatio };
    }

    function seekBufferAtRatio(ratio) {
        const { startRatio, bufferRatio } = getBufferBounds();
        if (bufferRatio <= 0) return;
        const clamped = Math.max(startRatio, Math.min(1, ratio));
        const seekTime = ((clamped - startRatio) / bufferRatio) * bufferDuration;
        detailVideo.currentTime = seekTime;
        updateLiveState();
    }

    function seekWarmEventAtRatio(ratio) {
        if (!currentWarmEventPts) return;
        const ev = warmEvents.find(e => e.start_pts_ns === currentWarmEventPts);
        if (!ev) return;
        const windowMs = eventStripZoomHours * 3600_000;
        const windowStart = Date.now() - windowMs;
        const evStartRatio = (ev.start_ms - windowStart) / windowMs;
        const evEndRatio = (ev.start_ms + ev.duration_ms - windowStart) / windowMs;
        const evSpan = evEndRatio - evStartRatio;
        if (evSpan <= 0) return;
        const clamped = Math.max(evStartRatio, Math.min(evEndRatio, ratio));
        const progress = (clamped - evStartRatio) / evSpan;
        const duration = detailVideo.duration;
        if (duration && isFinite(duration)) {
            detailVideo.currentTime = progress * duration;
        }
    }

    // Timeline drag handlers
    function timelineDragStart(clientX) {
        if (!currentDetailCameraId) return;
        const ratio = getTimelineRatio(clientX);
        const { startRatio } = getBufferBounds();

        if (isPlayingWarmEvent) {
            dragTarget = 'warm';
            isDraggingTimeline = true;
            isSeeking = true;
            timelineWrapper.classList.add('dragging');
            seekWarmEventAtRatio(ratio);
        } else if (ratio >= startRatio && bufferDuration > 0) {
            dragTarget = 'buffer';
            isDraggingTimeline = true;
            isSeeking = true;
            timelineWrapper.classList.add('dragging');
            seekBufferAtRatio(ratio);
        }
    }

    function timelineDragMove(clientX) {
        if (!isDraggingTimeline) return;
        const ratio = getTimelineRatio(clientX);
        if (dragTarget === 'buffer') {
            seekBufferAtRatio(ratio);
        } else if (dragTarget === 'warm') {
            seekWarmEventAtRatio(ratio);
        }
    }

    function timelineDragEnd() {
        if (!isDraggingTimeline) return;
        isDraggingTimeline = false;
        isSeeking = false;
        dragTarget = null;
        timelineWrapper.classList.remove('dragging');
    }

    // Mouse drag
    timelineWrapper.addEventListener('mousedown', (e) => {
        if (e.target === timelineScrubber) return;
        e.preventDefault();
        timelineDragStart(e.clientX);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDraggingTimeline) {
            e.preventDefault();
            timelineDragMove(e.clientX);
        }
    });

    document.addEventListener('mouseup', () => {
        timelineDragEnd();
    });

    // Touch drag
    timelineWrapper.addEventListener('touchstart', (e) => {
        if (e.target === timelineScrubber) return;
        timelineDragStart(e.touches[0].clientX);
    }, { passive: true });

    document.addEventListener('touchmove', (e) => {
        if (isDraggingTimeline) {
            e.preventDefault();
            timelineDragMove(e.touches[0].clientX);
        }
    }, { passive: false });

    document.addEventListener('touchend', () => {
        timelineDragEnd();
    });

    // Unified timeline click handler (for warm event selection)
    timelineWrapper.addEventListener('click', (e) => {
        if (!currentDetailCameraId) return;
        if (e.target === timelineScrubber) return;

        const ratio = getTimelineRatio(e.clientX);
        const { startRatio } = getBufferBounds();

        // Clicks in the buffer region are handled by drag handlers
        if (ratio >= startRatio && bufferDuration > 0) return;
        // Warm event scrubbing is handled by drag handlers
        if (isPlayingWarmEvent) return;

        // Check if click is on a warm event
        if (warmEvents.length === 0) return;

        const now = Date.now();
        const windowMs = eventStripZoomHours * 3600_000;
        const windowStart = now - windowMs;
        const clickedMs = windowStart + ratio * windowMs;

        let closest = null;
        let closestDist = Infinity;
        for (const ev of warmEvents) {
            const evEnd = ev.start_ms + ev.duration_ms;
            if (clickedMs >= ev.start_ms && clickedMs <= evEnd) {
                closest = ev;
                break;
            }
            const dist = Math.min(
                Math.abs(clickedMs - ev.start_ms),
                Math.abs(clickedMs - evEnd)
            );
            if (dist < closestDist) {
                closestDist = dist;
                closest = ev;
            }
        }

        if (closest) {
            const evEnd = closest.start_ms + closest.duration_ms;
            const threshold = windowMs * 0.02;
            if (clickedMs >= closest.start_ms - threshold && clickedMs <= evEnd + threshold) {
                loadWarmEvent(currentDetailCameraId, closest.start_pts_ns);
            }
        }
    });

    // Unified timeline hover handler
    timelineWrapper.addEventListener('mousemove', (e) => {
        const rect = timelineWrapper.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const ratio = x / rect.width;

        const now = Date.now();
        const windowMs = eventStripZoomHours * 3600_000;
        const windowStart = now - windowMs;
        const hoveredMs = windowStart + ratio * windowMs;
        const hoveredDate = new Date(hoveredMs);
        hoverTime.textContent = hoveredDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        hoverTime.style.left = (x - hoverTime.offsetWidth / 2) + 'px';
        hoverTime.classList.add('visible');

        // Tooltip for detections in buffer region
        if (bufferDuration > 0 && !isPlayingWarmEvent) {
            const bufferMs = bufferDuration * 1000;
            const bufferRatio = bufferMs / windowMs;
            const bufferStartX = 1.0 - bufferRatio;
            if (ratio >= bufferStartX) {
                const bufferClickRatio = (ratio - bufferStartX) / bufferRatio;
                const time = bufferClickRatio * bufferDuration;
                const detection = findDetectionNear(time, 1.0);
                if (detection && currentDetailCameraId) {
                    showTooltip(e.clientX, e.clientY, detection);
                    return;
                }
            }
        }
        hideTooltip();
    });

    timelineWrapper.addEventListener('mouseleave', () => {
        hoverTime.classList.remove('visible');
        hideTooltip();
    });

    // Router
    function router() {
        const hash = window.location.hash || '#/';
        const cameraMatch = hash.match(/^#\/camera\/(.+)$/);

        if (cameraMatch) {
            const cameraId = decodeURIComponent(cameraMatch[1]);
            if (cameras.includes(cameraId)) {
                const targetView = `detail:${cameraId}`;
                if (currentView !== targetView) {
                    const isBack = currentView && currentView.startsWith('detail:');
                    withViewTransition(() => showDetailView(cameraId), isBack);
                    currentView = targetView;
                }
            } else {
                window.location.hash = '/';
            }
        } else {
            if (currentView !== 'grid') {
                const isBack = currentView !== null;
                withViewTransition(() => showGridView(), isBack);
                currentView = 'grid';
            }
        }
    }

    // View functions
    function showGridView() {
        // Cleanup detail view
        cleanupDetailView();

        // Show grid view
        detailView.hidden = true;
        gridView.hidden = false;

        // Load grid cameras if not already loaded
        cameras.forEach(cameraId => {
            if (!gridHlsInstances.has(cameraId)) {
                const cell = grid.querySelector(`[data-camera-id="${cameraId}"]`);
                if (cell) {
                    loadGridCamera(cameraId, cell.querySelector('video'));
                }
            }
        });
    }

    function showDetailView(cameraId) {
        // Cleanup grid HLS instances to save resources
        gridHlsInstances.forEach((hls, id) => {
            hls.destroy();
        });
        gridHlsInstances.clear();

        // Update UI
        gridView.hidden = true;
        detailView.hidden = false;
        detailCameraName.textContent = cameraId;
        detailLoading.hidden = false;
        currentDetailCameraId = cameraId;

        // Reset timeline
        timelineScrubber.value = 100;
        currentTimeDisplay.textContent = '00:00:00';
        durationDisplay.textContent = '00:00:00';
        liveBtn.classList.add('is-live');
        liveBtn.classList.remove('is-warm');
        liveBtn.querySelector('span:last-child') || updateLiveBtnText('Live');
        stabilityOverlay.hidden = !stabilityOverlayEnabled;
        if (stabilityOverlayEnabled) {
            fetchStabilityMap();
        }

        // Reset warm state
        isPlayingWarmEvent = false;
        currentWarmEventPts = null;
        warmEvents = [];
        collapsedGroups.clear();
        groupPageLimits.clear();

        // Load camera stream
        loadDetailCamera(cameraId);

        // Fetch warm events
        fetchWarmEvents(cameraId);
    }

    function updateLiveBtnText(text) {
        // The button has: <span class="live-indicator"></span> + text node
        const indicator = liveBtn.querySelector('.live-indicator');
        liveBtn.textContent = '';
        liveBtn.appendChild(indicator);
        liveBtn.appendChild(document.createTextNode(' ' + text));
    }

    function cleanupDetailView() {
        if (timelineAnimationId) {
            cancelAnimationFrame(timelineAnimationId);
            timelineAnimationId = null;
        }
        if (motionPollInterval) {
            clearInterval(motionPollInterval);
            motionPollInterval = null;
        }
        if (detectionPollInterval) {
            clearInterval(detectionPollInterval);
            detectionPollInterval = null;
        }
        if (warmEventPollInterval) {
            clearInterval(warmEventPollInterval);
            warmEventPollInterval = null;
        }
        if (detailHls) {
            detailHls.destroy();
            detailHls = null;
        }
        detailVideo.src = '';
        currentMotionSegments = [];
        currentDetections = [];
        currentDetailCameraId = null;
        bufferDuration = 0;
        stabilityImage = null;
        stabilityOverlay.hidden = true;
        overlayMode = 'off';
        stabilityOverlayEnabled = false;
        maskToggleBtn.classList.remove('active');
        stabilityCtx.clearRect(0, 0, stabilityOverlay.width, stabilityOverlay.height);
        if (stabilityPollInterval) {
            clearInterval(stabilityPollInterval);
            stabilityPollInterval = null;
        }
        hideTooltip();
        detectionGallery.innerHTML = '';
        recordingsGroups.innerHTML = '';
        recordingsSection.hidden = true;
        const rect = timelineCanvas.getBoundingClientRect();
        timelineCtx.clearRect(0, 0, rect.width, rect.height);
        warmEvents = [];
        isPlayingWarmEvent = false;
        currentWarmEventPts = null;
    }

    // Camera cell creation
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

    // Grid camera loading
    function loadGridCamera(cameraId, video) {
        const src = `/api/stream/${cameraId}/playlist.m3u8`;
        const loading = video.parentElement.querySelector('.loading');

        if (typeof Hls !== 'undefined' && Hls.isSupported()) {
            const hls = new Hls({
                enableWorker: false,
            });
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
                        case Hls.ErrorTypes.NETWORK_ERROR:
                            hls.startLoad();
                            break;
                        case Hls.ErrorTypes.MEDIA_ERROR:
                            hls.recoverMediaError();
                            break;
                        default:
                            loading.querySelector('p').textContent = 'Stream error';
                            loading.hidden = false;
                            break;
                    }
                }
            });
        } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
            video.src = src;
            video.addEventListener('loadedmetadata', () => {
                loading.hidden = true;
                video.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
            });
        } else {
            loading.querySelector('p').textContent = 'HLS not supported';
        }
    }

    // Detail camera loading (live stream)
    function loadDetailCamera(cameraId) {
        const src = `/api/stream/${cameraId}/playlist.m3u8`;

        if (typeof Hls !== 'undefined' && Hls.isSupported()) {
            detailHls = new Hls({
                enableWorker: false,
            });

            detailHls.loadSource(src);
            detailHls.attachMedia(detailVideo);

            detailHls.on(Hls.Events.MANIFEST_PARSED, () => {
                detailLoading.hidden = true;
                detailVideo.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
                startTimelineUpdate();
                fetchMotionSegments(cameraId);
                fetchDetections(cameraId);
            });

            detailHls.on(Hls.Events.ERROR, (event, data) => {
                console.error(`HLS error for ${cameraId}:`, data.type, data.details);
                if (data.fatal) {
                    switch (data.type) {
                        case Hls.ErrorTypes.NETWORK_ERROR:
                            detailHls.startLoad();
                            break;
                        case Hls.ErrorTypes.MEDIA_ERROR:
                            detailHls.recoverMediaError();
                            break;
                        default:
                            detailLoading.querySelector('p').textContent = 'Stream error';
                            detailLoading.hidden = false;
                            break;
                    }
                }
            });
        } else if (detailVideo.canPlayType('application/vnd.apple.mpegurl')) {
            detailVideo.src = src;
            detailVideo.addEventListener('loadedmetadata', () => {
                detailLoading.hidden = true;
                detailVideo.play().catch(e => console.error(`Play failed for ${cameraId}:`, e));
                startTimelineUpdate();
                fetchMotionSegments(cameraId);
                fetchDetections(cameraId);
            }, { once: true });
        } else {
            detailLoading.querySelector('p').textContent = 'HLS not supported';
        }
    }

    // Warm event playback
    function loadWarmEvent(cameraId, startPtsNs) {
        const src = `/api/cameras/${encodeURIComponent(cameraId)}/events/${startPtsNs}/playlist.m3u8`;

        // Destroy current HLS instance
        if (detailHls) {
            detailHls.destroy();
            detailHls = null;
        }

        isPlayingWarmEvent = true;
        currentWarmEventPts = startPtsNs;

        // Update UI state
        liveBtn.classList.remove('is-live');
        liveBtn.classList.add('is-warm');
        timelineScrubber.classList.add('active');
        updateLiveBtnText('Return to Live');

        detailLoading.hidden = false;

        if (typeof Hls !== 'undefined' && Hls.isSupported()) {
            detailHls = new Hls({
                enableWorker: false,
            });

            detailHls.loadSource(src);
            detailHls.attachMedia(detailVideo);

            detailHls.on(Hls.Events.MANIFEST_PARSED, () => {
                detailLoading.hidden = true;
                detailVideo.play().catch(e => console.error(`Warm play failed:`, e));
            });

            detailHls.on(Hls.Events.ERROR, (event, data) => {
                console.error(`Warm HLS error:`, data.type, data.details);
                if (data.fatal) {
                    detailLoading.querySelector('p').textContent = 'Playback error';
                    detailLoading.hidden = false;
                }
            });
        } else if (detailVideo.canPlayType('application/vnd.apple.mpegurl')) {
            detailVideo.src = src;
            detailVideo.addEventListener('loadedmetadata', () => {
                detailLoading.hidden = true;
                detailVideo.play().catch(e => console.error(`Warm play failed:`, e));
            }, { once: true });
        }

        renderTimeline();
        renderRecordingsGallery();
    }

    function returnToLive() {
        if (!currentDetailCameraId) return;

        isPlayingWarmEvent = false;
        currentWarmEventPts = null;

        liveBtn.classList.remove('is-warm');
        timelineScrubber.classList.remove('active');
        updateLiveBtnText('Live');

        // Reload live stream
        if (detailHls) {
            detailHls.destroy();
            detailHls = null;
        }

        loadDetailCamera(currentDetailCameraId);
        renderTimeline();
        renderRecordingsGallery();
    }

    // Timeline functions
    function formatWindowTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function startTimelineUpdate() {
        function update() {
            if (isPlayingWarmEvent) {
                const duration = detailVideo.duration;
                if (!isSeeking && duration && isFinite(duration)) {
                    const progress = (detailVideo.currentTime / duration) * 100;
                    timelineScrubber.value = progress;
                }
            } else {
                const duration = bufferDuration || detailVideo.duration;
                if (!isSeeking && duration && isFinite(duration)) {
                    const progress = (detailVideo.currentTime / duration) * 100;
                    timelineScrubber.value = progress;
                    updateLiveState();
                    drawStability();
                }
            }
            // Update window time labels
            const now = Date.now();
            const windowMs = eventStripZoomHours * 3600_000;
            if (isPlayingWarmEvent && currentWarmEventPts) {
                const evStartMs = Number(BigInt(currentWarmEventPts) / 1_000_000n);
                const ev = warmEvents.find(e => e.start_pts_ns === currentWarmEventPts);
                const evDurationMs = ev ? ev.duration_ms : (detailVideo.duration * 1000);
                currentTimeDisplay.textContent = formatWindowTime(new Date(evStartMs));
                durationDisplay.textContent = formatWindowTime(new Date(evStartMs + evDurationMs));
            } else {
                currentTimeDisplay.textContent = formatWindowTime(new Date(now - windowMs));
                durationDisplay.textContent = 'Now';
            }

            renderTimeline();
            timelineAnimationId = requestAnimationFrame(update);
        }
        update();
    }

    function fetchStabilityMap() {
        if (!stabilityOverlayEnabled || !currentDetailCameraId || isPlayingWarmEvent) return;
        const img = new Image();
        img.onload = () => {
            stabilityImage = img;
            drawStability();
        };
        img.onerror = () => {};
        img.src = `/api/cameras/${encodeURIComponent(currentDetailCameraId)}/motion/stability?t=${Date.now()}`;
    }

    function drawStability() {
        if (!stabilityImage) return;
        const w = detailVideo.clientWidth;
        const h = detailVideo.clientHeight;
        if (w === 0 || h === 0) return;
        if (stabilityOverlay.width !== w || stabilityOverlay.height !== h) {
            stabilityOverlay.width = w;
            stabilityOverlay.height = h;
        }
        stabilityCtx.clearRect(0, 0, w, h);
        stabilityCtx.drawImage(stabilityImage, 0, 0, w, h);
        // Convert grayscale JPEG to blue/purple-tinted alpha mask:
        // bright (volatile) -> blue at 50% opacity
        // dark (stable) -> fully transparent
        const imageData = stabilityCtx.getImageData(0, 0, w, h);
        const px = imageData.data;
        for (let i = 0; i < px.length; i += 4) {
            const brightness = px[i];
            px[i]     = 100;
            px[i + 1] = 60;
            px[i + 2] = 255;
            px[i + 3] = (brightness / 255) * 128; // 0.5 * 255 = 128
        }
        stabilityCtx.putImageData(imageData, 0, 0);
    }

    function updateLiveState() {
        const duration = bufferDuration || detailVideo.duration;
        if (duration && isFinite(duration)) {
            const isAtLive = (duration - detailVideo.currentTime) < 3;
            liveBtn.classList.toggle('is-live', isAtLive);
        }
    }

    function formatTime(seconds) {
        if (!isFinite(seconds)) return '00:00:00';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    // Motion segment data fetching
    async function fetchMotionSegments(cameraId) {
        if (motionPollInterval) {
            clearInterval(motionPollInterval);
        }

        async function poll() {
            try {
                const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion`);
                if (response.ok) {
                    const data = await response.json();
                    currentMotionSegments = data.segments || [];
                    if (data.total_duration > 0) {
                        bufferDuration = data.total_duration;
                    }
                    renderTimeline();
                }
            } catch (err) {
                console.error('Failed to fetch motion data:', err);
            }
        }

        await poll();
        motionPollInterval = setInterval(poll, 5000);
        // Poll stability map every 5 seconds when overlay is enabled
        if (stabilityPollInterval) clearInterval(stabilityPollInterval);
        stabilityPollInterval = setInterval(fetchStabilityMap, 5000);
    }

    // Detection data fetching
    async function fetchDetections(cameraId) {
        if (detectionPollInterval) {
            clearInterval(detectionPollInterval);
        }

        async function poll() {
            try {
                const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/detections`);
                if (response.ok) {
                    const data = await response.json();
                    currentDetections = data.detections || [];
                    if (data.total_duration > 0) {
                        bufferDuration = data.total_duration;
                    }
                    renderTimeline();
                    renderDetectionGallery();
                }
            } catch (err) {
                console.error('Failed to fetch detection data:', err);
            }
        }

        await poll();
        detectionPollInterval = setInterval(poll, 5000);
    }

    // Warm event fetching
    async function fetchWarmEvents(cameraId) {
        if (warmEventPollInterval) {
            clearInterval(warmEventPollInterval);
        }

        async function poll() {
            try {
                const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/events`);
                if (currentDetailCameraId !== cameraId) return;
                if (response.ok) {
                    const raw = await response.json();
                    warmEvents = raw.map(ev => ({
                        ...ev,
                        start_ms: Number(BigInt(ev.start_pts_ns) / 1_000_000n),
                    }));
                    renderTimeline();
                    renderRecordingsGallery();
                }
            } catch (err) {
                console.error('Failed to fetch warm events:', err);
            }
        }

        await poll();
        warmEventPollInterval = setInterval(poll, 15000);
    }

    // Unified timeline rendering
    function renderTimeline() {
        const rect = timelineWrapper.getBoundingClientRect();
        if (rect.width === 0) return;

        const dpr = window.devicePixelRatio || 1;
        timelineCanvas.width = rect.width * dpr;
        timelineCanvas.height = rect.height * dpr;
        timelineCtx.scale(dpr, dpr);
        timelineCtx.clearRect(0, 0, rect.width, rect.height);

        const w = rect.width;
        const h = rect.height;

        const now = Date.now();
        const windowMs = eventStripZoomHours * 3600_000;
        const windowStart = now - windowMs;
        const windowEnd = now;

        // 1. Time axis ticks
        timelineCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        timelineCtx.lineWidth = 1;
        const tickIntervalHours = eventStripZoomHours <= 1 ? 0.25 :
                                   eventStripZoomHours <= 6 ? 1 :
                                   eventStripZoomHours <= 24 ? 4 : 8;
        const tickIntervalMs = tickIntervalHours * 3600_000;
        const firstTick = Math.ceil(windowStart / tickIntervalMs) * tickIntervalMs;
        for (let t = firstTick; t < windowEnd; t += tickIntervalMs) {
            const x = ((t - windowStart) / windowMs) * w;
            timelineCtx.beginPath();
            timelineCtx.moveTo(x, 0);
            timelineCtx.lineTo(x, h);
            timelineCtx.stroke();
        }

        // 2. Warm event blocks
        warmEvents.forEach(ev => {
            const evStart = ev.start_ms;
            const evEnd = evStart + ev.duration_ms;

            if (evEnd < windowStart || evStart > windowEnd) return;

            const startX = Math.max(0, ((evStart - windowStart) / windowMs) * w);
            const endX = Math.min(w, ((evEnd - windowStart) / windowMs) * w);
            const evW = Math.max(2, endX - startX);

            const isPlaying = isPlayingWarmEvent && currentWarmEventPts === ev.start_pts_ns;
            if (ev.event_type === 'object') {
                timelineCtx.fillStyle = isPlaying ? 'rgba(220, 50, 50, 1)' : 'rgba(220, 50, 50, 0.8)';
            } else {
                timelineCtx.fillStyle = isPlaying ? 'rgba(255, 200, 50, 1)' : 'rgba(255, 200, 50, 0.7)';
            }

            timelineCtx.beginPath();
            timelineCtx.roundRect(startX, 2, evW, h - 4, 2);
            timelineCtx.fill();

            if (isPlaying) {
                timelineCtx.strokeStyle = '#fff';
                timelineCtx.lineWidth = 2;
                timelineCtx.beginPath();
                timelineCtx.roundRect(startX, 2, evW, h - 4, 2);
                timelineCtx.stroke();

                // Playhead within event: show progress through event
                const duration = detailVideo.duration;
                if (duration && isFinite(duration) && duration > 0) {
                    const progress = detailVideo.currentTime / duration;
                    const playheadX = startX + progress * evW;
                    timelineCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                    timelineCtx.lineWidth = 2;
                    timelineCtx.beginPath();
                    timelineCtx.moveTo(playheadX, 0);
                    timelineCtx.lineTo(playheadX, h);
                    timelineCtx.stroke();
                }
            }
        });

        // 3. Live buffer region (right edge)
        const bufferMs = (bufferDuration || 0) * 1000;
        if (bufferMs > 0) {
            const bufferRatio = Math.min(1, bufferMs / windowMs);
            const bufferStartX = w * (1.0 - bufferRatio);
            const bufferW = w * bufferRatio;

            // Subtle background tint for buffer region
            timelineCtx.fillStyle = 'rgba(255, 255, 255, 0.08)';
            timelineCtx.fillRect(bufferStartX, 0, bufferW, h);

            // 4. Motion segments within buffer region
            const detectionTimes = currentDetections.map(d => d.timestamp);
            currentMotionSegments.forEach(segment => {
                const segStartX = bufferStartX + (segment.start / bufferDuration) * bufferW;
                const segEndX = bufferStartX + (segment.end / bufferDuration) * bufferW;
                const segW = segEndX - segStartX;

                const hasDetection = detectionTimes.some(t => t >= segment.start && t <= segment.end);
                if (hasDetection) return;

                const alpha = 0.5 + segment.intensity * 0.5;
                timelineCtx.fillStyle = `rgba(255, 200, 50, ${alpha})`;
                timelineCtx.beginPath();
                timelineCtx.roundRect(segStartX, 2, segW, h - 4, 2);
                timelineCtx.fill();
            });

            // 5. Detection markers within buffer region
            currentDetections.forEach(det => {
                const x = bufferStartX + (det.timestamp / bufferDuration) * bufferW;
                const alpha = 0.6 + det.confidence * 0.4;
                timelineCtx.fillStyle = `rgba(220, 50, 50, ${alpha})`;
                timelineCtx.fillRect(x - 2, 2, 4, h - 4);
            });

            // 6. Playhead in live mode
            if (!isPlayingWarmEvent) {
                const currentTime = detailVideo.currentTime;
                const duration = bufferDuration || detailVideo.duration;
                if (duration && isFinite(duration) && duration > 0) {
                    const playheadX = bufferStartX + (currentTime / duration) * bufferW;
                    timelineCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                    timelineCtx.lineWidth = 2;
                    timelineCtx.beginPath();
                    timelineCtx.moveTo(playheadX, 0);
                    timelineCtx.lineTo(playheadX, h);
                    timelineCtx.stroke();
                }
            }
        }
    }

    function findDetectionNear(time, threshold) {
        let closest = null;
        let minDist = threshold;

        for (const det of currentDetections) {
            const dist = Math.abs(det.timestamp - time);
            if (dist < minDist) {
                minDist = dist;
                closest = det;
            }
        }

        return closest;
    }

    function showTooltip(x, y, detection) {
        tooltipImage.src = `/api/cameras/${encodeURIComponent(currentDetailCameraId)}/detections/${detection.id}/frame`;
        tooltipLabel.textContent = `${detection.object_class} (${Math.round(detection.confidence * 100)}%)`;
        detectionTooltip.style.left = `${x + 10}px`;
        detectionTooltip.style.top = `${y - 170}px`;
        detectionTooltip.hidden = false;
    }

    function hideTooltip() {
        detectionTooltip.hidden = true;
    }

    function renderDetectionGallery() {
        detectionGallery.innerHTML = '';
        currentDetections.forEach(det => {
            const card = document.createElement('div');
            card.className = 'detection-card';
            const imgSrc = `/api/cameras/${encodeURIComponent(currentDetailCameraId)}/detections/${det.id}/frame`;
            card.innerHTML = `
                <img src="${imgSrc}" loading="lazy" alt="${det.object_class}">
                <div class="det-label">${det.object_class} (${Math.round(det.confidence * 100)}%)</div>
                <div class="det-time">${formatTime(det.timestamp)}</div>
            `;
            card.addEventListener('click', () => {
                detailVideo.currentTime = det.timestamp;
            });
            detectionGallery.appendChild(card);
        });
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

    function renderRecordingsGallery() {
        recordingsGroups.innerHTML = '';
        if (warmEvents.length === 0) {
            recordingsSection.hidden = true;
            return;
        }
        recordingsSection.hidden = false;

        // Group all events by date
        const groups = new Map();
        warmEvents.forEach(ev => {
            const label = formatDateLabel(new Date(ev.start_ms));
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push(ev);
        });

        // Sort groups by most recent event (descending)
        const sortedGroups = [...groups.entries()].sort((a, b) => {
            const aMax = Math.max(...a[1].map(e => e.start_ms));
            const bMax = Math.max(...b[1].map(e => e.start_ms));
            return bMax - aMax;
        });

        let isFirst = true;
        sortedGroups.forEach(([label, events]) => {
            // Sort within group: objects first, then by recency
            events.sort((a, b) => {
                if (a.event_type !== b.event_type) return a.event_type === 'object' ? -1 : 1;
                return b.start_pts_ns - a.start_pts_ns;
            });

            const objectCount = events.filter(e => e.event_type === 'object').length;
            const defaultCollapsed = !isFirst;
            const collapsed = collapsedGroups.has(label) ? collapsedGroups.get(label) : defaultCollapsed;

            const group = document.createElement('div');
            group.className = 'rec-group';
            if (collapsed) group.classList.add('collapsed');

            const heading = document.createElement('button');
            heading.className = 'rec-group-label';
            const countParts = [];
            if (objectCount > 0) countParts.push(`${objectCount} object${objectCount !== 1 ? 's' : ''}`);
            const motionCount = events.length - objectCount;
            if (motionCount > 0) countParts.push(`${motionCount} motion`);
            heading.innerHTML = `<span class="rec-group-arrow"></span>${label} <span class="rec-group-count">${countParts.join(', ')}</span>`;

            heading.addEventListener('click', () => {
                const nowCollapsed = !group.classList.contains('collapsed');
                group.classList.toggle('collapsed', nowCollapsed);
                collapsedGroups.set(label, nowCollapsed);
            });

            group.appendChild(heading);

            const grid = document.createElement('div');
            grid.className = 'rec-group-grid';

            const limit = groupPageLimits.get(label) || RECORDINGS_PER_PAGE;
            const visible = events.slice(0, limit);

            visible.forEach(ev => {
                const card = document.createElement('div');
                card.className = 'recording-card';
                if (isPlayingWarmEvent && currentWarmEventPts === ev.start_pts_ns) {
                    card.classList.add('active');
                }

                const thumbSrc = `/api/cameras/${encodeURIComponent(currentDetailCameraId)}/events/${ev.start_pts_ns}/thumbnail`;
                const evDate = new Date(ev.start_ms);
                const timeStr = evDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const durSec = (ev.duration_ms / 1000).toFixed(0);
                const typeLabel = ev.event_type === 'object' ? 'Object' : 'Motion';
                const typeClass = ev.event_type === 'object' ? 'object' : 'movement';

                card.innerHTML = `
                    <img class="rec-thumb" src="${thumbSrc}" loading="lazy" alt="Recording">
                    <div class="rec-type ${typeClass}">${typeLabel}</div>
                    <div class="rec-time">${timeStr}</div>
                    <div class="rec-duration">${durSec}s</div>
                `;

                card.addEventListener('click', () => {
                    loadWarmEvent(currentDetailCameraId, ev.start_pts_ns);
                    renderRecordingsGallery();
                });

                grid.appendChild(card);
            });

            group.appendChild(grid);

            if (events.length > limit) {
                const moreBtn = document.createElement('button');
                moreBtn.className = 'recordings-more-btn';
                moreBtn.textContent = `Show more (${events.length - limit} remaining)`;
                moreBtn.addEventListener('click', () => {
                    groupPageLimits.set(label, limit + RECORDINGS_PER_PAGE);
                    renderRecordingsGallery();
                });
                group.appendChild(moreBtn);
            }

            recordingsGroups.appendChild(group);
            isFirst = false;
        });
    }

    // Handle canvas resize
    window.addEventListener('resize', () => {
        renderTimeline();
        drawStability();
    });
});
