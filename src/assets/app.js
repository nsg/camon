// UI startup; the preceding assets only declare shared functions and state.
document.addEventListener('DOMContentLoaded', async () => {
    wireTokenPrompt();
    await loadCameras();

    window.addEventListener('hashchange', router);
    router();

    wireLiveView();
    wireSettingsPanel();
    wireEventsView();
    wirePlaybackView();
    wireDebugView();
});
