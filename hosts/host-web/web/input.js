// Feeds controller state to the runtime every animation frame: the first
// browser gamepad (standard mapping) if one is connected, else the keyboard and
// mouse. Clicking the game canvas captures the mouse: moving it turns the camera
// (right stick), left/right buttons are the right/left triggers; Esc releases it.
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

// Keyboard fallback (KeyboardEvent.code). Sticks: WASD (left), arrows (right).
const KEY_BUTTONS = {
  Space: XINPUT.A, Escape: XINPUT.B, Tab: XINPUT.B, KeyR: XINPUT.X, KeyF: XINPUT.Y,
  KeyQ: XINPUT.LEFT_SHOULDER, KeyE: XINPUT.RIGHT_SHOULDER,
  Enter: XINPUT.START, Backspace: XINPUT.BACK, KeyX: XINPUT.LEFT_THUMB, KeyV: XINPUT.RIGHT_THUMB,
  Digit1: XINPUT.DPAD_UP, Digit2: XINPUT.DPAD_DOWN, Digit3: XINPUT.DPAD_LEFT, Digit4: XINPUT.DPAD_RIGHT,
};
const KEY_LEFT_TRIGGER = 'KeyZ';
const KEY_RIGHT_TRIGGER = 'KeyC';
const KEY_AXES = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
                          KEY_LEFT_TRIGGER, KEY_RIGHT_TRIGGER]);

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

function fromKeyboard() {
  let buttons = 0;
  for (const code of held) buttons |= KEY_BUTTONS[code] ?? 0;
  const dir = (neg, pos) => (held.has(pos) ? STICK : 0) - (held.has(neg) ? STICK : 0);
  let rx = dir('ArrowLeft', 'ArrowRight');
  let ry = dir('ArrowDown', 'ArrowUp');
  if (mouse.locked && (mouse.dx !== 0 || mouse.dy !== 0)) {
    rx = axis(mouse.dx / MOUSE_FULL_DEFLECTION);
    ry = axis(-mouse.dy / MOUSE_FULL_DEFLECTION);
  }
  mouse.dx = 0;
  mouse.dy = 0;
  const lt = held.has(KEY_LEFT_TRIGGER) || (mouse.buttons & 2) ? 255 : 0;
  const rt = held.has(KEY_RIGHT_TRIGGER) || (mouse.buttons & 1) ? 255 : 0;
  return [buttons, lt, rt, dir('KeyA', 'KeyD'), dir('KeyS', 'KeyW'), rx, ry];
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

export function startInput(module) {
  const set = module._skyrt_set_pad;
  if (typeof set !== 'function') return;
  addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.code in KEY_BUTTONS || KEY_AXES.has(e.code)) {
      held.add(e.code);
      e.preventDefault();
    }
  });
  addEventListener('keyup', (e) => held.delete(e.code));
  addEventListener('blur', () => held.clear());
  captureMouse(module.canvas);
  let last = '';
  const tick = () => {
    const pad = navigator.getGamepads?.().find((p) => p && p.mapping === 'standard');
    const state = pad ? fromGamepad(pad) : fromKeyboard();
    const key = state.join(',');
    if (key !== last) {
      last = key;
      set(0, ...state);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
