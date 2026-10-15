// On-screen controller for touch screens, drawn over the game: floating left and right sticks
// (touch anywhere on that side, away from the buttons), and the face, shoulder, trigger,
// stick-click, D-pad, Start and Back buttons. input.js merges its state with the other inputs.
'use strict';

const BITS = {
  UP: 0x0001, DOWN: 0x0002, LEFT: 0x0004, RIGHT: 0x0008, START: 0x0010, BACK: 0x0020,
  LS: 0x0040, RS: 0x0080, LB: 0x0100, RB: 0x0200, A: 0x1000, B: 0x2000, X: 0x4000, Y: 0x8000,
};

// [label, bit or trigger name, CSS position]. Positions are relative to the game's frame.
const BUTTONS = [
  ['A', 'A', 'right: 8.5%; bottom: 5%'],
  ['B', 'B', 'right: 3%; bottom: 15.5%'],
  ['X', 'X', 'right: 14%; bottom: 15.5%'],
  ['Y', 'Y', 'right: 8.5%; bottom: 26%'],
  ['LB', 'LB', 'left: 3%; top: 14%'],
  ['LT', 'lt', 'left: 3%; top: 3%'],
  ['RB', 'RB', 'right: 3%; top: 14%'],
  ['RT', 'rt', 'right: 3%; top: 3%'],
  ['LS', 'LS', 'left: 10%; top: 14%'],
  ['RS', 'RS', 'right: 10%; top: 14%'],
  ['Back', 'BACK', 'left: 41%; top: 3%'],
  ['Start', 'START', 'right: 41%; top: 3%'],
  ['▲', 'UP', 'left: 7.5%; top: 33%'],
  ['▼', 'DOWN', 'left: 7.5%; top: 51%'],
  ['◀', 'LEFT', 'left: 3%; top: 42%'],
  ['▶', 'RIGHT', 'left: 12%; top: 42%'],
];

const CSS = `
.touchpad { position: absolute; inset: 0; touch-action: none; user-select: none; -webkit-user-select: none;
  z-index: 2; }
.touchpad .tbtn { position: absolute; min-width: 7%; aspect-ratio: 1; display: flex; align-items: center;
  justify-content: center; border-radius: 50%; border: 2px solid rgba(255,255,255,.45);
  background: rgba(0,0,0,.25); color: rgba(255,255,255,.85); font: 600 clamp(9px, 1.8vw, 16px) system-ui;
  box-sizing: border-box; }
.touchpad .tbtn.small { min-width: 5%; border-radius: 30%; font-size: clamp(8px, 1.5vw, 14px); }
.touchpad .tbtn.on { background: rgba(255,255,255,.35); }
.touchpad .stick { position: absolute; width: 16%; aspect-ratio: 1; border-radius: 50%;
  border: 2px solid rgba(255,255,255,.35); transform: translate(-50%, -50%); pointer-events: none;
  display: none; }
.touchpad .knob { position: absolute; left: 50%; top: 50%; width: 42%; aspect-ratio: 1; border-radius: 50%;
  background: rgba(255,255,255,.4); transform: translate(-50%, -50%); }
`;

// Fraction of the frame's height a stick deflects fully over.
const STICK_RADIUS = 0.12;

export function createTouch(stage) {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  const pad = document.createElement('div');
  pad.className = 'touchpad';
  pad.hidden = true;
  stage.appendChild(pad);

  const state = { buttons: 0, lt: 0, rt: 0, lx: 0, ly: 0, rx: 0, ry: 0 };
  // Pointer id -> what it is doing: { kind: 'button', name } | { kind: 'stick', side, x, y }.
  const pointers = new Map();
  const pressed = new Map(); // button name -> count of pointers holding it

  // The two floating sticks: each appears where its side is touched.
  const sticks = {};
  for (const side of ['left', 'right']) {
    const ring = document.createElement('div');
    ring.className = 'stick';
    const knob = document.createElement('div');
    knob.className = 'knob';
    ring.appendChild(knob);
    pad.appendChild(ring);
    sticks[side] = { ring, knob, pointer: null };
  }
  function setStick(side, x, y) {
    if (side === 'left') { state.lx = x; state.ly = y; } else { state.rx = x; state.ry = y; }
  }

  const elements = new Map();
  for (const [label, name, pos] of BUTTONS) {
    const b = document.createElement('div');
    b.className = 'tbtn' + (['A', 'B', 'X', 'Y'].includes(name) ? '' : ' small');
    b.textContent = label;
    b.style.cssText = pos;
    b.dataset.name = name;
    pad.appendChild(b);
    elements.set(name, b);
  }

  function recompute() {
    state.buttons = 0;
    state.lt = 0;
    state.rt = 0;
    for (const [name, n] of pressed) {
      if (n <= 0) continue;
      if (name === 'lt') state.lt = 255;
      else if (name === 'rt') state.rt = 255;
      else state.buttons |= BITS[name];
    }
    for (const [name, el] of elements) el.classList.toggle('on', (pressed.get(name) ?? 0) > 0);
  }

  function press(name, delta) {
    pressed.set(name, Math.max(0, (pressed.get(name) ?? 0) + delta));
    recompute();
  }

  const rect = () => pad.getBoundingClientRect();

  pad.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    pad.setPointerCapture?.(e.pointerId);
    const button = e.target.closest?.('.tbtn');
    if (button) {
      pointers.set(e.pointerId, { kind: 'button', name: button.dataset.name });
      press(button.dataset.name, 1);
      navigator.vibrate?.(8);
      return;
    }
    const r = rect();
    const side = e.clientX - r.left < r.width / 2 ? 'left' : 'right';
    const s = sticks[side];
    // One finger per stick: a second touch on the same side moves the stick there.
    if (s.pointer !== null) pointers.delete(s.pointer);
    s.pointer = e.pointerId;
    pointers.set(e.pointerId, { kind: 'stick', side, x: e.clientX, y: e.clientY });
    s.ring.style.left = `${e.clientX - r.left}px`;
    s.ring.style.top = `${e.clientY - r.top}px`;
    s.knob.style.left = s.knob.style.top = '50%';
    s.ring.style.display = 'block';
    setStick(side, 0, 0);
  });

  pad.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    e.preventDefault();
    if (p.kind === 'stick') {
      const radius = rect().height * STICK_RADIUS;
      let dx = (e.clientX - p.x) / radius;
      let dy = (e.clientY - p.y) / radius;
      const len = Math.hypot(dx, dy);
      if (len > 1) { dx /= len; dy /= len; }
      setStick(p.side, dx, -dy);
      const { knob } = sticks[p.side];
      knob.style.left = `${50 + dx * 50}%`;
      knob.style.top = `${50 + dy * 50}%`;
    } else if (p.kind === 'button') {
      // Sliding off a button releases it; sliding onto another presses that one.
      const el = document.elementFromPoint(e.clientX, e.clientY)?.closest?.('.tbtn');
      const name = el?.dataset.name;
      if (name !== p.name) {
        if (p.name) press(p.name, -1);
        p.name = name;
        if (name) press(name, 1);
      }
    }
  });

  function release(e) {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    if (p.kind === 'button' && p.name) press(p.name, -1);
    if (p.kind === 'stick') {
      const s = sticks[p.side];
      s.pointer = null;
      s.ring.style.display = 'none';
      setStick(p.side, 0, 0);
    }
  }
  pad.addEventListener('pointerup', release);
  pad.addEventListener('pointercancel', release);
  pad.addEventListener('contextmenu', (e) => e.preventDefault());

  return {
    get visible() { return !pad.hidden; },
    set visible(v) {
      pad.hidden = !v;
      if (!v) {
        pointers.clear();
        pressed.clear();
        recompute();
        state.lx = state.ly = state.rx = state.ry = 0;
        for (const s of Object.values(sticks)) {
          s.pointer = null;
          s.ring.style.display = 'none';
        }
      }
    },
    // Current state (stick deflections from -1 to 1, y up).
    take() {
      return { ...state };
    },
  };
}
