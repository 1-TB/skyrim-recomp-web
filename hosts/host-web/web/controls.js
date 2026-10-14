// The controls help: keyboard and mouse (with rebinding), gamepad and touch mappings. Rendered
// into every `[data-controls]` container (the page's Controls tab and the in-game menu); the
// keyboard bindings come from input.js and are stored in the page's settings as overrides.
'use strict';

import { KEY_ACTIONS, RESERVED_KEYS, keyBindings } from './input.js';

// What each controller input does in Skyrim (Xbox 360 layout).
const IN_GAME = {
  moveUp: 'Move', moveDown: 'Move', moveLeft: 'Move', moveRight: 'Move',
  lookUp: 'Look', lookDown: 'Look', lookLeft: 'Look', lookRight: 'Look',
  a: 'Activate, confirm', b: 'Character menu, back', x: 'Ready or sheathe weapon', y: 'Jump',
  lb: 'Sprint', rb: 'Shout or power', lt: 'Left hand', rt: 'Right hand',
  ls: 'Sneak', rs: 'Switch first/third person', start: 'Journal, system menu', back: 'Wait',
  up: 'Favorites', down: 'Favorites', left: 'Favorite hotkeys', right: 'Favorite hotkeys',
};

// Standard-mapping gamepad: [button on the pad, controller input id].
const PAD = [
  ['Left stick', 'moveUp'], ['Right stick', 'lookUp'], ['A / Cross', 'a'], ['B / Circle', 'b'],
  ['X / Square', 'x'], ['Y / Triangle', 'y'], ['LB / L1', 'lb'], ['RB / R1', 'rb'], ['LT / L2', 'lt'],
  ['RT / R2', 'rt'], ['Left stick click', 'ls'], ['Right stick click', 'rs'], ['Start / Options', 'start'],
  ['Back / Share', 'back'], ['D-pad', 'up'],
];

// The on-screen controller (touch.js).
const TOUCH = [
  ['Drag on the left half', 'Left stick', 'moveUp'],
  ['Drag on the right half', 'Right stick (look)', 'lookUp'],
  ['A, B, X, Y (bottom right)', 'Face buttons', null],
  ['LT, LB / RT, RB (top corners)', 'Triggers and bumpers', null],
  ['LS, RS (beside the bumpers)', 'Stick clicks', null],
  ['Back, Start (top middle)', 'Back and Start', null],
  ['Arrows (left)', 'D-pad', 'up'],
];

// Readable names for KeyboardEvent.code values.
function keyName(code) {
  if (code.startsWith('Key')) return code.slice(3);
  if (code.startsWith('Digit')) return code.slice(5);
  if (code.startsWith('Numpad')) return `Num ${code.slice(6)}`;
  const names = {
    Space: 'Space', Escape: 'Esc', ArrowUp: '↑', ArrowDown: '↓', ArrowLeft: '←', ArrowRight: '→',
    Backspace: 'Backspace', Enter: 'Enter', Tab: 'Tab', ShiftLeft: 'Left Shift', ShiftRight: 'Right Shift',
    ControlLeft: 'Left Ctrl', ControlRight: 'Right Ctrl', AltLeft: 'Left Alt', AltRight: 'Right Alt',
    Backquote: '`', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Semicolon: ';', Quote: "'",
    Comma: ',', Period: '.', Slash: '/', Backslash: '\\', CapsLock: 'Caps Lock',
  };
  return names[code] ?? code;
}

const el = (tag, props = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  e.append(...children.filter((c) => c != null));
  return e;
};
const kbd = (text) => el('kbd', {}, text);

function table(head, rows) {
  return el('div', { class: 'table-wrap' },
    el('table', {},
      el('thead', {}, el('tr', {}, ...head.map((h, i) => el('th', i === 2 && head.length === 4 ? { class: 'wide-only' } : {}, h)))),
      el('tbody', {}, ...rows)));
}

// `options`: { overrides(): the stored overrides, save(overrides): stores them (null resets) }.
export function createControls(options) {
  const containers = [...document.querySelectorAll('[data-controls]')];
  // The action being rebound: { id, container } or null.
  let capturing = null;

  // Grabs the next key for the action being rebound, before input.js (or the menu) sees it.
  addEventListener('keydown', (e) => {
    if (!capturing) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (e.code === 'Escape' || RESERVED_KEYS.has(e.code)) {
      capturing = null;
      render();
      return;
    }
    const overrides = { ...(options.overrides() ?? {}) };
    // A key does one thing: take it from any other action first.
    for (const [id, codes] of Object.entries(keyBindings(overrides))) {
      if (id !== capturing.id && codes.includes(e.code)) overrides[id] = codes.filter((c) => c !== e.code);
    }
    overrides[capturing.id] = [e.code];
    capturing = null;
    options.save(overrides);
    render();
  }, true);

  // A controller input cell, with what it does in Skyrim beneath it on narrow screens.
  const inputCell = (input, what) => el('td', {}, input, what ? el('div', { class: 'save-sub phone-only' }, what) : null);
  const whatCell = (what) => el('td', { class: 'muted wide-only' }, what ?? '');

  function keyboardSection() {
    const overrides = options.overrides();
    const bindings = keyBindings(overrides);
    const customized = overrides && Object.keys(overrides).length > 0;
    const row = ([id, input, defaults]) => {
      const codes = bindings[id];
      const isCapturing = capturing?.id === id;
      const changed = codes.join() !== defaults.join();
      const keys = isCapturing
        ? el('span', { class: 'muted' }, 'Press a key… (Esc cancels)')
        : codes.length ? el('span', {}, ...codes.map((c) => kbd(keyName(c)))) : el('span', { class: 'muted' }, 'unbound');
      return el('tr', { class: isCapturing ? 'capturing' : '' },
        el('td', { class: 'keys' }, keys),
        inputCell(input, IN_GAME[id]),
        whatCell(IN_GAME[id]),
        el('td', { class: 'act' },
          el('button', {
            class: 'small ghost', type: 'button', 'aria-label': `Change the key for ${input}`,
            title: changed ? 'Changed from the default' : 'Change the key',
            onclick: () => { capturing = isCapturing ? null : { id }; render(); },
          }, isCapturing ? 'Cancel' : changed ? 'Change •' : 'Change')));
    };
    const sticks = KEY_ACTIONS.filter(([id]) => id.startsWith('move') || id.startsWith('look'));
    const buttons = KEY_ACTIONS.filter((a) => !sticks.includes(a));
    const mouseRows = [
      ['Move the mouse', 'Right stick', 'Look'],
      ['Left button', 'RT', IN_GAME.rt],
      ['Right button', 'LT', IN_GAME.lt],
    ].map(([k, input, what]) => el('tr', {}, el('td', { class: 'keys' }, k), inputCell(input, what), whatCell(what), el('td', {})));
    const head = ['Key', 'Controller', 'In Skyrim', ''];
        return el('div', { class: 'controls-section' },
      el('div', { class: 'row' },
        el('h3', {}, 'Keyboard and mouse'),
        el('button', {
          class: 'small', type: 'button', ...(customized ? {} : { disabled: '' }),
          onclick: () => { capturing = null; options.save(null); render(); },
        }, 'Reset to defaults')),
      el('div', { class: 'controls-grid' },
        el('div', {}, table(head, [...sticks.map(row), ...mouseRows])),
        el('div', {}, table(head, buttons.map(row)))),
      el('ul', { class: 'notes' },
        el('li', {}, 'Click the game to capture the mouse; ', kbd('Esc'), ' releases it. Click ', el('em', {}, 'Change'), ', then press the new key.'),
        el('li', {}, kbd('F1'), ' opens the menu. A connected gamepad takes over from the keyboard.')));
  }

  function padSection() {
    const rows = PAD.map(([button, id]) => el('tr', {}, el('td', {}, button), el('td', { class: 'muted' }, IN_GAME[id])));
    return el('div', { class: 'controls-section' },
      el('div', { class: 'row' }, el('h3', {}, 'Gamepad')),
      table(['Button', 'In Skyrim'], rows),
      el('ul', { class: 'notes' }, el('li', {}, 'The first connected gamepad with the standard mapping; press a button once so the browser reports it.')));
  }

  function touchSection() {
    const rows = TOUCH.map(([where, input, id]) => el('tr', {}, el('td', {}, where), el('td', {}, input),
      el('td', { class: 'muted' }, id ? IN_GAME[id] : 'As on a pad')));
    return el('div', { class: 'controls-section' },
      el('div', { class: 'row' }, el('h3', {}, 'Touch')),
      table(['Where', 'Controller', 'In Skyrim'], rows),
      el('ul', { class: 'notes' }, el('li', {}, 'Shown on touch screens (Settings → Touch controls). Fullscreen turns phones to landscape.')));
  }

  function render() {
    for (const c of containers) {
      c.replaceChildren(keyboardSection(), el('div', { class: 'controls-grid', style: 'margin-top: 18px' }, padSection(), touchSection()));
    }
  }
  render();
  return { render };
}
