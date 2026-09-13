// UI startup; the preceding assets only declare shared functions and state.
document.addEventListener('DOMContentLoaded', async () => {
    wireTokenPrompt();
    await loadCameras();

    window.addEventListener('hashchange', router);
    document.addEventListener('visibilitychange', syncCameraVisibility);
    window.addEventListener('scroll', updateGridVisibility, { passive: true, capture: true });
    window.addEventListener('resize', updateGridVisibility);
    router();

    wireLiveView();
    wireSettingsPanel();
    wireEventsView();
    wirePlaybackView();
    wireDebugView();
});
