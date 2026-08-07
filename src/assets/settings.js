// Motion settings panel and the mask editor (movement + detection layers),
// hosted inside the live view.

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

// The cells array for the layer currently being painted.
function activeCells() {
    return activeMaskLayer === 'detection' ? detectionCells : maskCells;
}

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

// === Motion settings panel ===

function wireSettingsPanel() {
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
    maskOverlay.addEventListener('pointerup', endMaskPaint);
    maskOverlay.addEventListener('pointercancel', endMaskPaint);
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
