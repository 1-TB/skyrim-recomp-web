// The title's on-screen keyboard (XamShowKeyboardUI, see src/text_input.rs): a browser prompt on
// the page's main thread. The calling worker blocks until it is answered.
addToLibrary({
  skyrt_prompt__proxy: 'sync',
  skyrt_prompt__deps: ['$UTF8ToString', '$stringToNewUTF8'],
  skyrt_prompt: (title, description, defaultText, maxChars) => {
    const message = [UTF8ToString(title), UTF8ToString(description)].filter(Boolean).join('\n\n');
    // The mouse is captured while playing; the prompt needs it.
    if (document.pointerLockElement) document.exitPointerLock();
    const text = prompt(message || 'Enter text', UTF8ToString(defaultText));
    if (text === null) return 0;
    return stringToNewUTF8(Array.from(text).slice(0, maxChars).join(''));
  },
});
