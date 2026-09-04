// Motion settings and the live view's mask/base-grid editor.

const maskOverlay = document.getElementById('mask-overlay');
const maskCtx = maskOverlay.getContext('2d');
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
const layerSizeBtn = document.getElementById('layer-size-btn');
const sizeBrush = document.getElementById('size-brush');
const tunerMode = document.getElementById('tuner-mode');
const tunerResetBtn = document.getElementById('tuner-reset-btn');
const settingsError = document.getElementById('settings-error');
const settingsErrorText = document.getElementById('settings-error-text');
const settingsErrorDismiss = document.getElementById('settings-error-dismiss');

// Movement suppresses motion; detection blacks pixels out before model input.
let motionSettings = null;
let maskEditEnabled = false;
let maskCells = [];
let detectionCells = [];
let sizeCells = [];
let activeMaskLayer = 'movement'; // 'movement' | 'detection' | 'size'
let maskCols = 16;
let maskRows = 12;
let maskPainting = false;
let maskPaintValue = true;
let currentMinContourArea = 0;
let currentCellContourAreaCeiling = 2000;

function activeCells() {
    if (activeMaskLayer === 'detection') return detectionCells;
    if (activeMaskLayer === 'size') return sizeCells;
    return maskCells;
}

function endMaskPaint() {
    if (!maskPainting) return;
    maskPainting = false;
    if (activeMaskLayer === 'detection') {
        putMotionSettings({ detection_mask: detectionCells.slice() });
    } else if (activeMaskLayer === 'size') {
        putMotionSettings({ min_contour_area_grid: sizeCells.slice() });
    } else {
        putMotionSettings({ mask: maskCells.slice() });
    }
}

function wireSettingsPanel() {
    settingsBtn.addEventListener('click', () => {
        const show = settingsPanel.hidden;
        settingsPanel.hidden = !show;
        settingsBtn.classList.toggle('active', show);
        if (show) {
            clearSettingsError();
        } else if (maskEditEnabled) {
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

    tunerMode.addEventListener('change', () => {
        putMotionSettings({ tuner_mode: tunerMode.value });
    });

    tunerResetBtn.addEventListener('click', async () => {
        if (!currentDetailCameraId) return;
        const cameraId = currentDetailCameraId;
        clearSettingsError();
        try {
            const response = await apiFetch(
                `api/cameras/${encodeURIComponent(cameraId)}/motion/tuner/reset`,
                { method: 'POST' });
            if (!response.ok) {
                if (response.status === 401) return;
                const body = await response.text();
                showSettingsError(body.trim() ||
                    `the server refused the reset (HTTP ${response.status})`);
                return;
            }
            if (currentDetailCameraId === cameraId) await fetchTunerSnapshot();
        } catch (err) {
            console.error('Failed to reset motion tuner:', err);
            showSettingsError('could not reach camon — tuning was not reset');
        }
    });

    maskEditBtn.addEventListener('click', () => {
        setMaskEditEnabled(!maskEditEnabled);
    });

    layerMovementBtn.addEventListener('click', () => setActiveMaskLayer('movement'));
    layerDetectionBtn.addEventListener('click', () => setActiveMaskLayer('detection'));
    layerSizeBtn.addEventListener('click', () => setActiveMaskLayer('size'));

    maskOverlay.addEventListener('pointerdown', (e) => {
        if (!maskEditEnabled) return;
        const idx = maskCellFromEvent(e);
        if (idx < 0) return;
        const cells = activeCells();
        maskPainting = true;
        maskPaintValue = activeMaskLayer === 'size'
            ? Number(sizeBrush.value)
            : !cells[idx];
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
    const cellCount = maskCols * maskRows;
    sizeCells = new Array(cellCount).fill(0);
    if (Array.isArray(data.min_contour_area_grid)) {
        data.min_contour_area_grid.slice(0, cellCount).forEach((value, i) => {
            sizeCells[i] = Number(value) || 0;
        });
    }
    currentMinContourArea = Number(data.min_contour_area) || 0;
    currentCellContourAreaCeiling = Number(data.cell_contour_area_ceiling) || 2000;

    sensitivitySlider.min = data.var_threshold_min;
    sensitivitySlider.max = data.var_threshold_max;
    sensitivitySlider.value = data.var_threshold;
    sensitivityValue.textContent = String(Math.round(data.var_threshold));

    minsizeSlider.min = data.min_contour_area_min;
    minsizeSlider.max = data.min_contour_area_max;
    minsizeSlider.value = data.min_contour_area;
    minsizeValue.textContent = String(Math.round(data.min_contour_area));
    tunerMode.value = data.tuner_mode || 'off';

    if (maskEditEnabled) drawMask();
    if (tunerEnabled) fetchTunerSnapshot();
}

function showSettingsError(message) {
    settingsErrorText.textContent = message;
    settingsError.hidden = false;
}

function clearSettingsError() {
    settingsError.hidden = true;
    settingsErrorText.textContent = '';
}

// A persistence failure leaves the live value visible but warns it will not survive restart.
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
    updateTunerPointerEvents();
    maskEditBtn.classList.toggle('active', enabled);
    maskEditBtn.textContent = enabled ? 'Done editing masks' : 'Edit masks';
    maskLayerRow.hidden = !enabled;
    if (enabled) {
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
    const size = layer === 'size';
    layerMovementBtn.classList.toggle('active', !detection && !size);
    layerDetectionBtn.classList.toggle('active', detection);
    layerSizeBtn.classList.toggle('active', size);
    sizeBrush.hidden = !size;
    maskLayerHint.textContent = size
        ? 'Paint a minimum blob size per cell; larger = less sensitive there.'
        : detection
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

    const fillLayer = (cells, color) => {
        maskCtx.fillStyle = color;
        for (let i = 0; i < cells.length; i++) {
            if (!cells[i]) continue;
            const col = i % maskCols;
            const row = Math.floor(i / maskCols);
            maskCtx.fillRect(col * cellW, row * cellH, cellW, cellH);
        }
    };

    for (let i = 0; i < sizeCells.length; i++) {
        const value = Number(sizeCells[i]) || 0;
        if (value <= 0) continue;
        const col = i % maskCols;
        const row = Math.floor(i / maskCols);
        const t = Math.max(0, Math.min(1, value / currentCellContourAreaCeiling));
        const alpha = 0.15 + t * 0.3;
        maskCtx.fillStyle = `rgba(45, 140, 255, ${alpha})`;
        maskCtx.fillRect(col * cellW, row * cellH, cellW, cellH);
    }
    fillLayer(maskCells, 'rgba(220, 50, 50, 0.4)');
    fillLayer(detectionCells, 'rgba(255, 140, 0, 0.45)');

    const fontSize = Math.min(cellH * 0.35, cellW * 0.3, 14);
    if (fontSize >= 8) {
        maskCtx.font = `${fontSize}px monospace`;
        maskCtx.textAlign = 'center';
        maskCtx.textBaseline = 'middle';
        for (let i = 0; i < sizeCells.length; i++) {
            const value = Number(sizeCells[i]) || 0;
            if (value <= 0) continue;
            const col = i % maskCols;
            const row = Math.floor(i / maskCols);
            const cx = col * cellW + cellW / 2;
            const cy = row * cellH + cellH / 2;
            const label = String(Math.round(value));
            const textW = maskCtx.measureText(label).width;
            maskCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            maskCtx.fillRect(cx - textW / 2 - 2, cy - fontSize / 2 - 1, textW + 4, fontSize + 2);
            maskCtx.fillStyle = '#fff';
            maskCtx.fillText(label, cx, cy);
        }
    }

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
