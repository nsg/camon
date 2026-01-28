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

    // State
    let cameras = [];
    const gridHlsInstances = new Map();
    let detailHls = null;
    let timelineAnimationId = null;
    let isSeeking = false;
    let currentView = null;
    let isFirstLoad = true;

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
        const time = (timelineScrubber.value / 100) * detailVideo.duration;
        currentTimeDisplay.textContent = formatTime(time);
        updateLiveState();
    });

    timelineScrubber.addEventListener('change', () => {
        const time = (timelineScrubber.value / 100) * detailVideo.duration;
        detailVideo.currentTime = time;
        isSeeking = false;
    });

    liveBtn.addEventListener('click', () => {
        if (detailVideo.duration && isFinite(detailVideo.duration)) {
            detailVideo.currentTime = detailVideo.duration;
            updateLiveState();
        }
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

        // Reset timeline
        timelineScrubber.value = 100;
        currentTimeDisplay.textContent = '00:00:00';
        durationDisplay.textContent = '00:00:00';
        liveBtn.classList.add('is-live');

        // Load camera stream
        loadDetailCamera(cameraId);
    }

    function cleanupDetailView() {
        if (timelineAnimationId) {
            cancelAnimationFrame(timelineAnimationId);
            timelineAnimationId = null;
        }
        if (detailHls) {
            detailHls.destroy();
            detailHls = null;
        }
        detailVideo.src = '';
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
            }, { once: true });
        } else {
            detailLoading.querySelector('p').textContent = 'HLS not supported';
        }
    }

    // Timeline functions
    function startTimelineUpdate() {
        function update() {
            if (!isSeeking && detailVideo.duration && isFinite(detailVideo.duration)) {
                const progress = (detailVideo.currentTime / detailVideo.duration) * 100;
                timelineScrubber.value = progress;
                currentTimeDisplay.textContent = formatTime(detailVideo.currentTime);
                durationDisplay.textContent = formatTime(detailVideo.duration);
                updateLiveState();
            }
            timelineAnimationId = requestAnimationFrame(update);
        }
        update();
    }

    function updateLiveState() {
        if (detailVideo.duration && isFinite(detailVideo.duration)) {
            const isAtLive = (detailVideo.duration - detailVideo.currentTime) < 3;
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
});
