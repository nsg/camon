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
    const motionCanvas = document.getElementById('motion-canvas');
    const motionCtx = motionCanvas.getContext('2d');
    const detectionTooltip = document.getElementById('detection-tooltip');
    const tooltipImage = document.getElementById('tooltip-image');
    const tooltipLabel = document.getElementById('tooltip-label');
    const maskOverlay = document.getElementById('mask-overlay');
    const maskCtx = maskOverlay.getContext('2d');
    const maskToggleBtn = document.getElementById('mask-toggle-btn');

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
    let currentDetailCameraId = null;
    let bufferDuration = 0;
    let maskOverlayEnabled = false;
    let currentMaskSeq = null;
    let maskImage = null;
    const failedMaskSeqs = new Set();

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
        const duration = bufferDuration || detailVideo.duration;
        const time = (timelineScrubber.value / 100) * duration;
        currentTimeDisplay.textContent = formatTime(time);
        updateLiveState();
    });

    timelineScrubber.addEventListener('change', () => {
        const duration = bufferDuration || detailVideo.duration;
        const time = (timelineScrubber.value / 100) * duration;
        detailVideo.currentTime = time;
        isSeeking = false;
    });

    liveBtn.addEventListener('click', () => {
        const duration = bufferDuration || detailVideo.duration;
        if (duration && isFinite(duration)) {
            detailVideo.currentTime = duration;
            updateLiveState();
        }
    });

    maskToggleBtn.addEventListener('click', () => {
        maskOverlayEnabled = !maskOverlayEnabled;
        maskToggleBtn.classList.toggle('active', maskOverlayEnabled);
        maskOverlay.hidden = !maskOverlayEnabled;
        if (!maskOverlayEnabled) {
            maskCtx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
            currentMaskSeq = null;
            maskImage = null;
            failedMaskSeqs.clear();
        }
    });

    // Tooltip event listeners (on wrapper since canvas has pointer-events: none)
    const timelineWrapper = document.querySelector('.timeline-wrapper');
    timelineWrapper.addEventListener('mousemove', (e) => {
        if (!bufferDuration) return;

        const rect = timelineWrapper.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const time = (x / rect.width) * bufferDuration;

        const detection = findDetectionNear(time, 1.0);
        if (detection && currentDetailCameraId) {
            showTooltip(e.clientX, e.clientY, detection);
        } else {
            hideTooltip();
        }
    });

    timelineWrapper.addEventListener('mouseleave', () => {
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
        maskOverlay.hidden = !maskOverlayEnabled;

        // Load camera stream
        loadDetailCamera(cameraId);
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
        if (detailHls) {
            detailHls.destroy();
            detailHls = null;
        }
        detailVideo.src = '';
        currentMotionSegments = [];
        currentDetections = [];
        currentDetailCameraId = null;
        bufferDuration = 0;
        currentMaskSeq = null;
        maskImage = null;
        failedMaskSeqs.clear();
        maskOverlay.hidden = true;
        maskOverlayEnabled = false;
        maskToggleBtn.classList.remove('active');
        maskCtx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
        hideTooltip();
        const rect = motionCanvas.getBoundingClientRect();
        motionCtx.clearRect(0, 0, rect.width, rect.height);
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

    // Detail camera loading
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

    // Timeline functions
    function startTimelineUpdate() {
        function update() {
            const duration = bufferDuration || detailVideo.duration;
            if (!isSeeking && duration && isFinite(duration)) {
                const progress = (detailVideo.currentTime / duration) * 100;
                timelineScrubber.value = progress;
                currentTimeDisplay.textContent = formatTime(detailVideo.currentTime);
                durationDisplay.textContent = formatTime(duration);
                updateLiveState();
                updateMaskOverlay();
            }
            timelineAnimationId = requestAnimationFrame(update);
        }
        update();
    }

    function updateMaskOverlay() {
        if (!maskOverlayEnabled || !currentDetailCameraId) return;

        const time = detailVideo.currentTime;
        const seg = currentMotionSegments.find(s => time >= s.start && time <= s.end);

        if (!seg) {
            if (currentMaskSeq !== null) {
                maskCtx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
                currentMaskSeq = null;
                maskImage = null;
            }
            return;
        }

        if (seg.sequence === currentMaskSeq || failedMaskSeqs.has(seg.sequence)) {
            return;
        }

        currentMaskSeq = seg.sequence;
        const seq = seg.sequence;
        const img = new Image();
        img.onload = () => {
            if (currentMaskSeq === seq) {
                maskImage = img;
                drawMask();
            }
        };
        img.onerror = () => {
            failedMaskSeqs.add(seq);
        };
        img.src = `/api/cameras/${encodeURIComponent(currentDetailCameraId)}/motion/${seq}/mask`;
    }

    function drawMask() {
        if (!maskImage) return;
        const w = detailVideo.clientWidth;
        const h = detailVideo.clientHeight;
        if (w === 0 || h === 0) return;
        if (maskOverlay.width !== w || maskOverlay.height !== h) {
            maskOverlay.width = w;
            maskOverlay.height = h;
        }
        maskCtx.clearRect(0, 0, w, h);
        maskCtx.drawImage(maskImage, 0, 0, w, h);
        // Convert grayscale JPEG to green-tinted alpha mask:
        // white (foreground) -> green at 60% opacity
        // black (background) -> fully transparent
        const imageData = maskCtx.getImageData(0, 0, w, h);
        const px = imageData.data;
        for (let i = 0; i < px.length; i += 4) {
            const brightness = px[i];
            px[i]     = 0;
            px[i + 1] = 255;
            px[i + 2] = 80;
            px[i + 3] = (brightness / 255) * 153; // 0.6 * 255 = 153
        }
        maskCtx.putImageData(imageData, 0, 0);
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
                        renderTimeline(bufferDuration);
                    }
                }
            } catch (err) {
                console.error('Failed to fetch motion data:', err);
            }
        }

        await poll();
        motionPollInterval = setInterval(poll, 5000);
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
                        renderTimeline(bufferDuration);
                    }
                }
            } catch (err) {
                console.error('Failed to fetch detection data:', err);
            }
        }

        await poll();
        detectionPollInterval = setInterval(poll, 5000);
    }

    function renderTimeline(duration) {
        if (!duration || !isFinite(duration)) return;

        const rect = motionCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        motionCanvas.width = rect.width * dpr;
        motionCanvas.height = rect.height * dpr;
        motionCtx.scale(dpr, dpr);

        motionCtx.clearRect(0, 0, rect.width, rect.height);

        // Build set of detection timestamps for overlap checking
        const detectionTimes = currentDetections.map(d => d.timestamp);

        // Draw motion segments (yellow), skipping areas with object detections
        currentMotionSegments.forEach(segment => {
            const startX = (segment.start / duration) * rect.width;
            const endX = (segment.end / duration) * rect.width;
            const width = endX - startX;

            // Check if any detection falls within this segment
            const hasDetection = detectionTimes.some(t => t >= segment.start && t <= segment.end);
            if (hasDetection) return;

            const alpha = 0.5 + segment.intensity * 0.5;
            motionCtx.fillStyle = `rgba(255, 200, 50, ${alpha})`;

            const radius = 4;
            motionCtx.beginPath();
            motionCtx.roundRect(startX, 0, width, rect.height, radius);
            motionCtx.fill();
        });

        // Draw detection markers (red)
        currentDetections.forEach(det => {
            const x = (det.timestamp / duration) * rect.width;
            const alpha = 0.6 + det.confidence * 0.4;
            motionCtx.fillStyle = `rgba(220, 50, 50, ${alpha})`;
            motionCtx.fillRect(x - 2, 0, 4, rect.height);
        });
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

    // Handle canvas resize
    window.addEventListener('resize', () => {
        if (bufferDuration > 0) {
            renderTimeline(bufferDuration);
        }
        drawMask();
    });
});
