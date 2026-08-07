// Startup for the Camon UI. The other assets/*.js files (loaded before this
// one, see index.html) only declare functions and state into the shared
// global scope; this file runs the original startup sequence.
document.addEventListener('DOMContentLoaded', async () => {
    wireTokenPrompt();
    await loadCameras();

    // === Router ===
    window.addEventListener('hashchange', router);
    router();

    // View listeners register after the first route renders, as they always
    // have: nothing dispatches an event during that synchronous render.
    wireLiveView();
    wireSettingsPanel();
    wireEventsView();
    wirePlaybackView();
    wireDebugView();
});
