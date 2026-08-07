// View 4: Detection Debug.

const debugView = document.getElementById('debug-view');
const debugBackBtn = document.getElementById('debug-back-btn');
const debugCameraName = document.getElementById('debug-camera-name');
const debugList = document.getElementById('debug-list');
const debugEmpty = document.getElementById('debug-empty');
const debugLinkBtn = document.getElementById('debug-link-btn');
let debugPoller = null;

// === Detection Debug ===

function wireDebugView() {
    debugBackBtn.addEventListener('click', () => {
        window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}`;
    });

    debugLinkBtn.addEventListener('click', () => {
        window.location.hash = `/camera/${encodeURIComponent(currentDetailCameraId)}/debug`;
    });
}

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
