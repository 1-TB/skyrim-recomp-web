// Feeds controller state to the runtime every animation frame: the first
// browser gamepad (standard mapping) if one is connected, else the keyboard and
// mouse. Clicking the game canvas captures the mouse: moving it turns the camera
// (right stick), left/right buttons are the right/left triggers; Esc releases it.
// The on-screen touch controller (touch.js), when shown, adds to whichever is in use.
'use strict';

const XINPUT = {
  DPAD_UP: 0x0001, DPAD_DOWN: 0x0002, DPAD_LEFT: 0x0004, DPAD_RIGHT: 0x0008,
  START: 0x0010, BACK: 0x0020, LEFT_THUMB: 0x0040, RIGHT_THUMB: 0x0080,
  LEFT_SHOULDER: 0x0100, RIGHT_SHOULDER: 0x0200, A: 0x1000, B: 0x2000, X: 0x4000, Y: 0x8000,
};

// Standard-mapping button index -> XInput bit (6 and 7 are the triggers).
const PAD_BUTTONS = [
  XINPUT.A, XINPUT.B, XINPUT.X, XINPUT.Y, XINPUT.LEFT_SHOULDER, XINPUT.RIGHT_SHOULDER, 0, 0,
  XINPUT.BACK, XINPUT.START, XINPUT.LEFT_THUMB, XINPUT.RIGHT_THUMB,
  XINPUT.DPAD_UP, XINPUT.DPAD_DOWN, XINPUT.DPAD_LEFT, XINPUT.DPAD_RIGHT,
];

// Keyboard actions: [id, controller input, default KeyboardEvent.codes]. The page lets the
// user rebind them (`options.keys()` returns the overrides, action id -> codes); the mouse
// (look, triggers) is fixed.
export const KEY_ACTIONS = [
  ['moveUp', 'Left stick up', ['KeyW']],
  ['moveDown', 'Left stick down', ['KeyS']],
  ['moveLeft', 'Left stick left', ['KeyA']],
  ['moveRight', 'Left stick right', ['KeyD']],
  ['lookUp', 'Right stick up', ['ArrowUp']],
  ['lookDown', 'Right stick down', ['ArrowDown']],
  ['lookLeft', 'Right stick left', ['ArrowLeft']],
  ['lookRight', 'Right stick right', ['ArrowRight']],
  ['a', 'A', ['Space']],
  ['b', 'B', ['Tab', 'Escape']],
  ['x', 'X', ['KeyR']],
  ['y', 'Y', ['KeyF']],
  ['lb', 'LB', ['KeyQ']],
  ['rb', 'RB', ['KeyE']],
  ['lt', 'LT', ['KeyZ']],
  ['rt', 'RT', ['KeyC']],
  ['ls', 'LS (click)', ['KeyX']],
  ['rs', 'RS (click)', ['KeyV']],
  ['start', 'Start', ['Enter']],
  ['back', 'Back', ['Backspace']],
  ['up', 'D-pad up', ['Digit1']],
  ['down', 'D-pad down', ['Digit2']],
  ['left', 'D-pad left', ['Digit3']],
  ['right', 'D-pad right', ['Digit4']],
];

// Keys the page keeps for itself (the menu): never bound.
export const RESERVED_KEYS = new Set(['F1']);

const ACTION_BITS = {
  a: XINPUT.A, b: XINPUT.B, x: XINPUT.X, y: XINPUT.Y, lb: XINPUT.LEFT_SHOULDER, rb: XINPUT.RIGHT_SHOULDER,
  ls: XINPUT.LEFT_THUMB, rs: XINPUT.RIGHT_THUMB, start: XINPUT.START, back: XINPUT.BACK,
  up: XINPUT.DPAD_UP, down: XINPUT.DPAD_DOWN, left: XINPUT.DPAD_LEFT, right: XINPUT.DPAD_RIGHT,
};

// Action id -> codes: the defaults with `overrides` applied.
export function keyBindings(overrides = {}) {
  const out = {};
  for (const [id, , codes] of KEY_ACTIONS) out[id] = Array.isArray(overrides?.[id]) ? overrides[id] : codes;
  return out;
}

// KeyboardEvent.code -> action id for `overrides`, rebuilt only when the overrides object changes.
let mapSource;
let keyMap = new Map();
function codeMap(overrides) {
  if (overrides !== mapSource || keyMap.size === 0) {
    mapSource = overrides;
    keyMap = new Map();
    for (const [id, codes] of Object.entries(keyBindings(overrides))) {
      for (const code of codes) if (!RESERVED_KEYS.has(code)) keyMap.set(code, id);
    }
  }
  return keyMap;
}

const held = new Set();
const STICK = 32767;
// Mouse movement (pixels) accumulated since the last frame, and the buttons held.
const mouse = { dx: 0, dy: 0, buttons: 0, locked: false };
// Pixels of movement per frame that deflect the right stick fully.
const MOUSE_FULL_DEFLECTION = 12;

function axis(v) {
  return Math.round(Math.max(-1, Math.min(1, v)) * STICK);
}

function fromGamepad(pad) {
  let buttons = 0;
  PAD_BUTTONS.forEach((bit, i) => {
    if (bit && pad.buttons[i]?.pressed) buttons |= bit;
  });
  const trigger = (i) => Math.round((pad.buttons[i]?.value ?? 0) * 255);
  // Browser Y axes point down; XInput's point up.
  return [buttons, trigger(6), trigger(7), axis(pad.axes[0] ?? 0), axis(-(pad.axes[1] ?? 0)),
          axis(pad.axes[2] ?? 0), axis(-(pad.axes[3] ?? 0))];
}

// Right-stick deflection for `dx`, `dy` pixels of look movement this frame.
function look(dx, dy, sensitivity) {
  return [axis(dx * sensitivity / MOUSE_FULL_DEFLECTION), axis(-dy * sensitivity / MOUSE_FULL_DEFLECTION)];
}

function fromKeyboard(sensitivity, overrides) {
  const map = codeMap(overrides);
  const on = new Set();
  for (const code of held) {
    const id = map.get(code);
    if (id) on.add(id);
  }
  let buttons = 0;
  for (const id of on) buttons |= ACTION_BITS[id] ?? 0;
  const dir = (neg, pos) => (on.has(pos) ? STICK : 0) - (on.has(neg) ? STICK : 0);
  let rx = dir('lookLeft', 'lookRight');
  let ry = dir('lookDown', 'lookUp');
  if (mouse.locked && (mouse.dx !== 0 || mouse.dy !== 0)) {
    [rx, ry] = look(mouse.dx, mouse.dy, sensitivity);
  }
  mouse.dx = 0;
  mouse.dy = 0;
  const lt = on.has('lt') || (mouse.buttons & 2) ? 255 : 0;
  const rt = on.has('rt') || (mouse.buttons & 1) ? 255 : 0;
  return [buttons, lt, rt, dir('moveLeft', 'moveRight'), dir('moveDown', 'moveUp'), rx, ry];
}

function captureMouse(canvas) {
  if (!canvas) return;
  canvas.addEventListener('click', () => canvas.requestPointerLock?.());
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  document.addEventListener('pointerlockchange', () => {
    mouse.locked = document.pointerLockElement === canvas;
    if (!mouse.locked) mouse.buttons = 0;
  });
  addEventListener('mousemove', (e) => {
    if (!mouse.locked) return;
    mouse.dx += e.movementX;
    mouse.dy += e.movementY;
  });
  addEventListener('mousedown', (e) => {
    if (mouse.locked) mouse.buttons |= e.button === 2 ? 2 : e.button === 0 ? 1 : 0;
  });
  addEventListener('mouseup', (e) => {
    mouse.buttons &= ~(e.button === 2 ? 2 : e.button === 0 ? 1 : 0);
  });
}

// Adds the touch controller's state to `state` ([buttons, lt, rt, lx, ly, rx, ry]).
function withTouch(state, touch, sensitivity) {
  const t = touch.take();
  state[0] |= t.buttons;
  state[1] = Math.max(state[1], t.lt);
  state[2] = Math.max(state[2], t.rt);
  if (t.lx !== 0 || t.ly !== 0) {
    state[3] = axis(t.lx);
    state[4] = axis(t.ly);
  }
  if (t.lookDx !== 0 || t.lookDy !== 0) {
    // A finger moves further than a mouse for the same turn: half the deflection per pixel.
    [state[5], state[6]] = look(t.lookDx / 2, t.lookDy / 2, sensitivity);
  }
  return state;
}

// `options.sensitivity()`: mouse/touch look multiplier (1 = default); `options.keys()`: key
// binding overrides (see `keyBindings`); `options.suspended()`: true while the page's menu is
// open (the game then sees a released controller); `options.touch`: the on-screen controller.
export function startInput(module, options = {}) {
  const sensitivity = options.sensitivity ?? (() => 1);
  const keys = options.keys ?? (() => undefined);
  const suspended = options.suspended ?? (() => false);
  const set = module._skyrt_set_pad;
  if (typeof set !== 'function') return;
  addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (suspended() || e.defaultPrevented) return;
    if (codeMap(keys()).has(e.code)) {
      held.add(e.code);
      e.preventDefault();
    }
  });
  addEventListener('keyup', (e) => held.delete(e.code));
  addEventListener('blur', () => held.clear());
  captureMouse(module.canvas);
  let last = '';
  const tick = () => {
    let state;
    if (suspended()) {
      held.clear();
      mouse.dx = mouse.dy = mouse.buttons = 0;
      options.touch?.take();
      state = [0, 0, 0, 0, 0, 0, 0];
    } else {
      const pad = navigator.getGamepads?.().find((p) => p && p.mapping === 'standard');
      state = pad ? fromGamepad(pad) : fromKeyboard(sensitivity(), keys());
      if (options.touch?.visible) state = withTouch(state, options.touch, sensitivity());
    }
    const key = state.join(',');
    if (key !== last) {
      last = key;
      set(0, ...state);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
