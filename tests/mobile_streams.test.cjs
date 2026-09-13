const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const assets = path.join(__dirname, '..', 'src', 'assets');

function harness({ native = false, observer = true } = {}) {
    const elements = new Map();
    function element(id) {
        if (!elements.has(id)) {
            elements.set(id, { hidden: true, getContext: () => ({}) });
        }
        return elements.get(id);
    }
    const grid = element('camera-grid');
    grid.children = [];
    const video = {
        paused: true, currentTime: 0, readyState: 2, src: '',
        listeners: new Map(),
        addEventListener(name, callback) { this.listeners.set(name, callback); },
        removeEventListener(name) { this.listeners.delete(name); },
        dispatch(name) { this.listeners.get(name)?.(); },
        play() { this.paused = false; this.dispatch('playing'); return Promise.resolve(); },
        pause() { this.paused = true; },
        load() { this.loads = (this.loads || 0) + 1; },
        removeAttribute(name) { if (name === 'src') this.src = ''; },
        canPlayType() { return native ? 'probably' : ''; },
    };
    const loading = { hidden: false, message: { textContent: 'Loading...' },
        querySelector: () => loading.message };
    video.parentElement = { querySelector: () => loading };
    elements.set('detail-video', video);
    elements.set('detail-loading', loading);
    const cell = { dataset: { cameraId: 'driveway' }, querySelector: () => video,
        getBoundingClientRect: () => ({ top: 0, bottom: 100, left: 0, right: 100 }) };
    grid.children.push(cell);

    class Hls {
        static instances = [];
        static Events = { MANIFEST_PARSED: 'manifest', LEVEL_UPDATED: 'level', ERROR: 'error' };
        static ErrorTypes = { NETWORK_ERROR: 'network', MEDIA_ERROR: 'media' };
        static isSupported() { return !native; }
        constructor(config) { this.config = config; this.handlers = new Map(); Hls.instances.push(this); }
        on(name, callback) { this.handlers.set(name, callback); }
        emit(name, data) { this.handlers.get(name)?.(name, data); }
        loadSource(src) { this.source = src; }
        attachMedia(media) { this.media = media; }
        startLoad() { this.starts = (this.starts || 0) + 1; }
        stopLoad() { this.stops = (this.stops || 0) + 1; }
        destroy() { this.destroyed = true; }
    }
    class IntersectionObserver {
        static instances = [];
        constructor(callback) { this.callback = callback; this.targets = []; IntersectionObserver.instances.push(this); }
        observe(target) { this.targets.push(target); }
        disconnect() { this.disconnected = true; }
        send(isIntersecting) { this.callback([{ target: cell, isIntersecting }]); }
    }
    const document = { hidden: false, getElementById: element };
    const window = { innerHeight: 500, innerWidth: 500, matchMedia: () => ({ matches: false }) };
    const context = vm.createContext({ document, window, Hls, localStorage: { getItem: () => '' },
        console, IntersectionObserver: observer ? IntersectionObserver : undefined,
        liveView: element('live-view'), eventsView: element('events-view'),
        playbackView: element('playback-view'), debugView: element('debug-view') });
    vm.runInContext(fs.readFileSync(path.join(assets, 'core.js'), 'utf8'), context);
    const run = (code) => vm.runInContext(code, context);
    return { run, context, document, video, loading, cell, grid, Hls, IntersectionObserver, element };
}

test('grid starts only observed tiles and retains a stopped HLS player on return', () => {
    const h = harness();
    h.element('grid-view').hidden = false;
    h.run('startGridObservation()');
    const first = h.IntersectionObserver.instances[0];
    assert.equal(h.Hls.instances.length, 0);
    first.send(true);
    const player = h.Hls.instances[0];
    assert.equal(player.config.enableWorker, true);
    assert.equal(player.config.autoStartLoad, false);
    assert.equal(player.config.backBufferLength, 15);
    assert.equal(player.config.maxMaxBufferLength, 15);
    assert.equal(player.starts || 0, 0);
    assert.equal(h.loading.hidden, false);
    player.emit(h.Hls.Events.MANIFEST_PARSED);
    assert.equal(player.starts, 1);
    assert.equal(h.loading.hidden, true);

    first.send(false);
    assert.equal(player.stops, 1);
    assert.equal(h.video.paused, true);
    h.run('hideAllViews()');
    first.send(true); // A queued callback from a disconnected observer cannot restart a stream.
    assert.equal(player.starts, 1);
    h.element('grid-view').hidden = false;
    h.run('startGridObservation()');
    h.IntersectionObserver.instances[1].send(true);
    assert.equal(h.Hls.instances.length, 1);
    assert.equal(player.starts, 2);
    assert.equal(player.destroyed, undefined);
});

test('page visibility stops the grid and reobserves on return', () => {
    const h = harness();
    h.element('grid-view').hidden = false;
    h.run('startGridObservation()');
    h.IntersectionObserver.instances[0].send(true);
    const player = h.Hls.instances[0];
    h.document.hidden = true;
    h.run('syncCameraVisibility()');
    assert.equal(player.stops, 1);
    player.emit(h.Hls.Events.MANIFEST_PARSED);
    assert.equal(player.starts || 0, 0);
    h.document.hidden = false;
    h.run('syncCameraVisibility()');
    h.IntersectionObserver.instances[1].send(true);
    assert.equal(player.starts, 1);
});

test('native HLS drops its offscreen source and restores it when visible', () => {
    const h = harness({ native: true, observer: false });
    h.element('grid-view').hidden = false;
    h.run('startGridObservation()');
    assert.match(h.video.src, /playlist\.m3u8/);
    assert.equal(h.Hls.instances.length, 0);
    h.video.dispatch('loadedmetadata');
    assert.equal(h.loading.hidden, true);
    h.cell.getBoundingClientRect = () => ({ top: 600, bottom: 700, left: 0, right: 100 });
    h.run('updateGridVisibility()');
    assert.equal(h.video.src, '');
    assert.equal(h.video.paused, true);
    h.cell.getBoundingClientRect = () => ({ top: 0, bottom: 100, left: 0, right: 100 });
    h.run('updateGridVisibility()');
    assert.match(h.video.src, /playlist\.m3u8/);
});

test('detail HLS pauses and resumes with page visibility', () => {
    const h = harness();
    vm.runInContext(fs.readFileSync(path.join(assets, 'live.js'), 'utf8'), h.context);
    h.run('startOverlayUpdates = () => {}; fetchMotionSegments = () => {}; fetchDetections = () => {}');
    h.element('live-view').hidden = false;
    h.run('loadDetailCamera("driveway")');
    const player = h.Hls.instances[0];
    assert.equal(player.config.enableWorker, true);
    assert.equal(player.config.autoStartLoad, false);
    assert.equal(player.starts || 0, 0);
    h.document.hidden = true;
    h.run('syncDetailCameraVisibility()');
    assert.equal(player.stops, 1);
    player.emit(h.Hls.Events.MANIFEST_PARSED);
    assert.equal(player.starts || 0, 0);
    h.document.hidden = false;
    h.run('syncDetailCameraVisibility()');
    assert.equal(player.starts, 1);
    assert.equal(h.video.paused, false);
});

test('native detail waits for visibility before playing', () => {
    const h = harness({ native: true });
    vm.runInContext(fs.readFileSync(path.join(assets, 'live.js'), 'utf8'), h.context);
    h.run('startOverlayUpdates = () => {}; fetchMotionSegments = () => {}; fetchDetections = () => {}');
    h.element('live-view').hidden = false;
    h.run('currentDetailCameraId = "driveway"; loadDetailCamera("driveway")');
    h.document.hidden = true;
    h.run('syncDetailCameraVisibility()');
    h.video.dispatch('loadedmetadata');
    assert.equal(h.video.paused, true);
    h.document.hidden = false;
    h.run('syncDetailCameraVisibility()');
    assert.equal(h.video.paused, false);
});
