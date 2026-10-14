// What kind of device the page runs on (src/main.rs sizes GPU memory use by it), asked on the
// page's main thread: workers lack navigator.maxTouchPoints.
addToLibrary({
  skyrt_handheld__proxy: 'sync',
  // 1 on phones and tablets (iOS and iPadOS, including iPads that report a Mac, Android) and on
  // devices with at most 4 GB of memory; else 0.
  skyrt_handheld: () => {
    if (typeof navigator === 'undefined') return 0;
    const ua = navigator.userAgent || '';
    const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
    const lowMemory = navigator.deviceMemory !== undefined && navigator.deviceMemory <= 4;
    return ios || /Android/.test(ua) || lowMemory ? 1 : 0;
  },
});
