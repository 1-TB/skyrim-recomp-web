// Boot and load status, driven by the log lines the page already receives (boot.js `log`): a
// panel over the game's frame until the title draws (compiling the runtime, opening or
// streaming the game files, starting the title), then a small chip with the game's phase (title
// screen, main menu, loading a save, in game) that fades once there is nothing to wait for.
// Most of these lines are the runtime's Info level: with --log warn only the page's own
// "Startup:" lines arrive, so the panel steps aside once the runtime has started.
'use strict';

// Draws per frame above which the title is in the world rather than a menu or loading screen.
const IN_GAME_DRAWS = 1000;
// How long the chip shows a phase that needs no waiting (title screen, main menu, in game).
const CHIP_LINGER_MS = 4000;

const RULES = [
  [/^Booting with:/, (s) => s.step('runtime', 'Preparing')],
  [/^Startup: memory .*compiling/, (s) => s.step('runtime', 'Downloading and compiling the runtime', 'About 27 MB to download the first time; compiled code is cached by the browser')],
  [/^Startup: compiled in ([\d.]+) s; instantiating/, (s, m) => s.step('runtime', 'Starting the runtime', `Compiled in ${m[1]} s`)],
  [/^Startup: instantiated/, (s) => s.instantiated()],
  [/^Startup failed after [\d.]+ s: (.*)/, (s, m) => s.fail(m[1])],
  [/streaming: fetching the game file list from (\S+)/, (s, m) => s.step('files', 'Fetching the file list', m[1])],
  [/streaming the game from .*: (\d+) files listed; reading default\.xex \(([\d.]+) MiB\)/,
    (s, m) => { s.streaming = true; s.step('files', 'Streaming default.xex', `${m[1]} files on the server; default.xex is ${m[2]} MiB`); }],
  [/\btitle default\.xex\b/, (s) => s.step('files', 'Opening the game files')],
  [/streaming: (\d+) requests, ([\d.]+) MiB in the last (\d+) s/, (s, m) => s.streamed(Number(m[2]), Number(m[3]))],
  [/streaming: (\d+) bytes of (.+) took ([\d.]+) s \(([\d.]+) MiB\/s\)/,
    (s, m) => s.note(`Slow connection: ${(Number(m[1]) / 2 ** 20).toFixed(1)} MiB of ${m[2]} took ${m[3]} s`)],
  [/open "[^"]*\\([^"\\]+\.es[mp])" -> .*\(Opened\)/i, (s, m) => s.step('title', 'Starting the title', `Loading ${m[1]}`)],
  [/open "([^"]*\\)?([^"\\]+\.bsa)" -> .*\(Opened\)/i, (s, m) => s.fileOpened(m[2])],
  [/first call: xam\.xex!XamContentCreateEnumerator/, (s) => s.phase('title', 'Title screen', 'Press Start')],
  [/XamContentCreateEnumerator: type 0x1,/, (s) => s.phase('menu', 'Main menu')],
  // Logged once per load, on the first read past a save's header (the save menu reads only that).
  [/content: loading save (.+?)\s*$/, (s, m) => s.loadingSave(m[1])],
  [/perf: ([\d.]+) fps, (\d+) draws\/frame/, (s, m) => s.perf(Number(m[1]), Number(m[2]))],
];

const PHASE_ORDER = ['boot', 'title', 'menu', 'loading', 'ingame'];

// Uses the panel and chip elements of index.html; `onPhase(text, state)` mirrors the current
// phase elsewhere (state: 'busy', 'ok' or 'err').
export function createStatus(onPhase = () => {}) {
  const $ = (id) => document.getElementById(id);
  const panel = $('boot-status');
  const chip = $('phase-chip');
  const steps = new Map([...panel.querySelectorAll('[data-step]')].map((li) => [li.dataset.step, li]));
  let started = 0;
  let stepStarted = 0;
  let currentStep = null;
  let timer = 0;
  let chipTimer = 0;
  let panelOpen = false;
  let failed = false;
  let lowLog = false;
  let mibStreamed = 0;

  const secs = (ms) => `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;

  function tick() {
    const now = performance.now();
    $('bs-time').textContent = secs(now - started);
    const li = currentStep && steps.get(currentStep);
    if (li) li.querySelector('.num').textContent = secs(now - stepStarted);
  }

  function hidePanel() {
    if (!panelOpen) return;
    panelOpen = false;
    clearInterval(timer);
    for (const li of steps.values()) {
      if (li.classList.contains('active')) li.classList.replace('active', 'done');
    }
    panel.classList.add('fade');
    setTimeout(() => { if (!panelOpen) panel.hidden = true; }, 500);
  }

  function showChip(text, linger) {
    clearTimeout(chipTimer);
    $('phase-text').textContent = text;
    chip.hidden = false;
    chip.classList.remove('fade');
    chip.querySelector('.dot').className = linger ? 'dot ok' : 'dot busy';
    if (linger) chipTimer = setTimeout(() => chip.classList.add('fade'), CHIP_LINGER_MS);
  }

  const state = {
    phaseName: 'boot',
    streaming: false,

    step(id, text, detail = '') {
      if (failed) return;
      if (id !== currentStep) {
        const order = [...steps.keys()];
        const at = order.indexOf(id);
        // Steps only move forward (late lines from an earlier step don't move them back).
        if (currentStep && at < order.indexOf(currentStep)) return;
        order.forEach((k, i) => {
          const li = steps.get(k);
          li.classList.toggle('done', i < at);
          li.classList.toggle('active', i === at);
          if (i > at) li.querySelector('.num').textContent = '';
        });
        if (currentStep) {
          steps.get(currentStep).querySelector('.num').textContent = secs(performance.now() - stepStarted);
        }
        currentStep = id;
        stepStarted = performance.now();
      }
      $('bs-phase').textContent = text;
      $('bs-detail').textContent = detail;
      onPhase(text, 'busy');
    },

    instantiated() {
      if (lowLog) {
        // No Info lines will follow: nothing more to show here.
        hidePanel();
        onPhase('Running (status needs log level Info)', 'ok');
        return;
      }
      this.step('files', this.streamingWanted ? 'Connecting to the server' : 'Opening the game files');
    },

    fileOpened(name) {
      if (currentStep === 'files') $('bs-detail').textContent = `Opened ${name}`;
    },

    // `name`: the save's package (e.g. "Save 3 - Prisoner  Skyrim  00.07.53.exs").
    loadingSave(name) {
      state.saveName = name.replace(/\.exs$/i, '').replace(/\s+/g, ' ');
      // A load from inside the game goes back to loading.
      if (state.phaseName === 'ingame' || state.phaseName === 'loading') state.phaseName = 'menu';
      this.phase('loading', 'Loading save', state.saveName);
    },

    streamed(mib, seconds) {
      mibStreamed += mib;
      const rate = `${(mib / seconds).toFixed(1)} MiB/s`;
      const text = `${mibStreamed.toFixed(0)} MiB streamed from the server (${rate} lately)`;
      if (panelOpen) $('bs-note').textContent = text;
      else if (state.phaseName === 'loading') $('phase-text').textContent = `Loading save · ${state.saveName} · ${mibStreamed.toFixed(0)} MiB streamed`;
    },

    note(text) {
      if (panelOpen) $('bs-note').textContent = text;
    },

    phase(name, text, detail) {
      if (failed) return;
      // Phases move forward, except that the main menu can be reached again from the game.
      const back = PHASE_ORDER.indexOf(name) < PHASE_ORDER.indexOf(state.phaseName);
      if (back && !(name === 'menu' && state.phaseName === 'ingame')) return;
      if (name === state.phaseName) return;
      state.phaseName = name;
      hidePanel();
      const label = detail ? `${text} · ${detail}` : text;
      const waiting = name === 'loading';
      showChip(label, !waiting);
      onPhase(text, waiting ? 'busy' : 'ok');
    },

    perf(fps, draws) {
      if (draws >= 1 && panelOpen) {
        // The title draws: its own frames (logos, video) show progress from here.
        hidePanel();
        showChip('Starting the title', true);
        onPhase('Starting the title', 'busy');
      }
      if (draws >= IN_GAME_DRAWS) this.phase('ingame', 'In game', `${fps.toFixed(0)} fps`);
    },

    fail(message) {
      failed = true;
      clearInterval(timer);
      if (panelOpen) {
        panel.classList.add('error');
        $('bs-phase').textContent = 'Stopped with an error';
        $('bs-detail').textContent = message;
      }
      chip.hidden = true;
      onPhase('Stopped with an error', 'err');
    },
  };

  return {
    // Shows the panel for a boot with `options`: { logLevel, streaming }.
    start(options) {
      started = stepStarted = performance.now();
      lowLog = options.logLevel === 'warn';
      state.streamingWanted = options.streaming;
      panelOpen = true;
      panel.hidden = false;
      panel.classList.remove('fade', 'error');
      $('bs-note').textContent = lowLog ? 'Log level is Warnings: only the startup is shown here.' : '';
      timer = setInterval(tick, 250);
      state.step('runtime', 'Preparing');
    },
    // One log line.
    line(text) {
      for (const [re, action] of RULES) {
        const m = re.exec(text);
        if (m) {
          action(state, m);
          return;
        }
      }
    },
    fail: (message) => state.fail(message),
  };
}
