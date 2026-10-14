//! Checks the device layout ([`crate::device::BLOCKS`]) against the command processor.
//!
//! At each draw the hook copies the device's register shadow into a [`Snapshot`] and puts a
//! marker packet with the snapshot's sequence number into the command buffer; when the command
//! processor reaches the draw after the marker, [`Verifier::compare`] compares its registers with
//! the snapshot. The shadow is copied before the draw flushes it, so for every register the draw
//! writes, the values must agree. Registers that disagree are written some other way (by packets
//! the engine or D3D emits outside the shadow), which a D3D-level renderer must account for.
//!
//! The first draws also carry a copy of the whole device, used to look for registers outside the
//! table whose values always equal some device field ([`Verifier::discoveries`]).

use std::collections::BTreeMap;

use crate::device::{self, BLOCKS};

/// Draws whose whole device is kept for discovering further shadowed registers.
pub const DISCOVERY_DRAWS: u64 = 4000;
/// Registers searched for outside the table: the context registers the draws can depend on.
const DISCOVERY_RANGE: std::ops::Range<u32> = 0x2000..0x2400;

/// The device's register shadow at a draw.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub seq: u32,
    /// Index of the hook (in [`crate::HOOKS`]) that took it.
    pub hook: u32,
    /// Values of [`device::shadowed_registers`], in order.
    pub registers: Vec<u32>,
    /// The whole device as dwords, for the first [`DISCOVERY_DRAWS`] draws.
    pub device: Option<Vec<u32>>,
}

impl Snapshot {
    /// Captures the shadow of the device at `device`, reading guest dwords with `read`.
    pub fn capture(seq: u32, hook: u32, device: u32, whole: bool, read: impl Fn(u32) -> u32) -> Self {
        let registers = device::shadowed_registers().map(|(_, offset)| read(device + offset)).collect();
        let device = whole.then(|| (0..device::SIZE / 4).map(|i| read(device + 4 * i)).collect());
        Snapshot { seq, hook, registers, device }
    }
}

#[derive(Clone, Debug, Default)]
struct RegisterStats {
    checked: u64,
    mismatched: u64,
    /// Mismatches by hook.
    by_hook: BTreeMap<u32, u64>,
    /// Mismatches where the draw's own packets wrote the register (after the marker), and the
    /// last such value of the first few.
    written: u64,
    written_samples: Vec<(u32, u32)>,
    /// Draws whose packets wrote the register, and how many of those wrote a value other than
    /// the shadow's.
    /// By hook: (writes, of them different).
    flushed: BTreeMap<u32, (u64, u64)>,
    /// Mismatches by the kind of packet that last wrote the register (see [`Verifier::compare`]),
    /// with the address of the last such packet.
    writers: BTreeMap<u32, (u64, u32)>,
    /// First mismatches: (shadow, command processor, hook).
    samples: Vec<(u32, u32, u32)>,
}

/// A register outside the table and the device dword offsets that matched it at every draw.
#[derive(Clone, Debug)]
struct Candidate {
    offsets: Vec<u32>,
    values_seen: std::collections::BTreeSet<u32>,
}

/// A register that disagreed with the command processor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub register: u32,
    /// Draws compared, and of them those where it differed.
    pub checked: u64,
    pub mismatched: u64,
    /// First differences: (shadow, command processor, hook).
    pub samples: Vec<(u32, u32, u32)>,
}

/// Comparison results over a run.
#[derive(Default)]
pub struct Verifier {
    stats: BTreeMap<u32, RegisterStats>,
    /// Draws compared, per hook.
    draws: BTreeMap<u32, u64>,
    /// Draws whose snapshot had a mismatch, per hook.
    draws_mismatched: BTreeMap<u32, u64>,
    discovery: BTreeMap<u32, Candidate>,
    discovery_draws: u64,
    /// Registers outside the table written by draws' packets: (hook, register) -> draws.
    unshadowed_writes: BTreeMap<(u32, u32), u64>,
}

impl Verifier {
    /// Compares `snap` with the command processor's registers (`cp(register)`); `written` are
    /// the register writes the command processor executed between the marker and the draw, and
    /// `last_writer(register)` the kind (type-3 opcode, or 0x100/0x101 for type 0/1) and address
    /// of the packet that last wrote a register.
    pub fn compare(
        &mut self,
        snap: &Snapshot,
        written: &[(u32, u32)],
        cp: impl Fn(u32) -> u32,
        last_writer: impl Fn(u32) -> (u32, u32),
    ) {
        *self.draws.entry(snap.hook).or_default() += 1;
        let mut last_written: BTreeMap<u32, u32> = BTreeMap::new();
        for &(r, v) in written {
            last_written.insert(r, v);
        }
        for &r in last_written.keys() {
            if device::shadow_offset(r).is_none() {
                *self.unshadowed_writes.entry((snap.hook, r)).or_default() += 1;
            }
        }
        let mut any = false;
        for ((register, _), &shadow) in device::shadowed_registers().zip(&snap.registers) {
            let actual = cp(register);
            let s = self.stats.entry(register).or_default();
            s.checked += 1;
            if let Some(&w) = last_written.get(&register) {
                let f = s.flushed.entry(snap.hook).or_default();
                f.0 += 1;
                if w != shadow {
                    f.1 += 1;
                }
            }
            if actual != shadow {
                any = true;
                s.mismatched += 1;
                *s.by_hook.entry(snap.hook).or_default() += 1;
                let (kind, address) = last_writer(register);
                let w = s.writers.entry(kind).or_default();
                w.0 += 1;
                w.1 = address;
                if let Some(&(_, value)) = written.iter().rev().find(|(r, _)| *r == register) {
                    s.written += 1;
                    if s.written_samples.len() < 4 {
                        s.written_samples.push((shadow, value));
                    }
                }
                if s.samples.len() < 4 {
                    s.samples.push((shadow, actual, snap.hook));
                }
            }
        }
        if any {
            *self.draws_mismatched.entry(snap.hook).or_default() += 1;
        }
        if let Some(dev) = &snap.device {
            self.discover(dev, &cp);
        }
    }

    fn discover(&mut self, dev: &[u32], cp: &impl Fn(u32) -> u32) {
        let first = self.discovery_draws == 0;
        self.discovery_draws += 1;
        for register in DISCOVERY_RANGE {
            if device::shadow_offset(register).is_some() {
                continue;
            }
            let value = cp(register);
            if first {
                let offsets: Vec<u32> =
                    dev.iter().enumerate().filter(|&(_, &v)| v == value).map(|(i, _)| 4 * i as u32).collect();
                self.discovery.insert(register, Candidate { offsets, values_seen: [value].into() });
            } else if let Some(c) = self.discovery.get_mut(&register) {
                c.offsets.retain(|&o| dev[o as usize / 4] == value);
                if c.values_seen.len() < 64 {
                    c.values_seen.insert(value);
                }
            }
        }
    }

    pub fn draws(&self) -> u64 {
        self.draws.values().sum()
    }

    /// Registers outside the table that matched one device field at every discovery draw while
    /// taking at least three values: (register, offsets, values seen).
    pub fn discoveries(&self) -> Vec<(u32, Vec<u32>, usize)> {
        self.discovery
            .iter()
            .filter(|(_, c)| !c.offsets.is_empty() && c.values_seen.len() >= 3)
            .map(|(&r, c)| (r, c.offsets.clone(), c.values_seen.len()))
            .collect()
    }

    /// Registers that ever disagreed.
    pub fn mismatches(&self) -> Vec<Mismatch> {
        self.stats
            .iter()
            .filter(|(_, s)| s.mismatched != 0)
            .map(|(&register, s)| Mismatch {
                register,
                checked: s.checked,
                mismatched: s.mismatched,
                samples: s.samples.clone(),
            })
            .collect()
    }

    /// By hook: register writes by draws' packets to shadowed registers, and how many equalled
    /// the shadow.
    pub fn flush_agreement(&self) -> BTreeMap<u32, (u64, u64)> {
        let mut out: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
        for s in self.stats.values() {
            for (&h, &(n, differ)) in &s.flushed {
                let e = out.entry(h).or_default();
                e.0 += n;
                e.1 += n - differ;
            }
        }
        out
    }

    /// A multi-line summary for the log.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let hook_name = |h: u32| crate::HOOKS.get(h as usize).map_or("?", |h| h.function);
        for (h, (n, agreed)) in self.flush_agreement() {
            let differing: Vec<String> = self
                .stats
                .iter()
                .filter_map(|(r, s)| {
                    s.flushed.get(&h).filter(|f| f.1 != 0).map(|f| format!("{r:#06X} {}/{}", f.1, f.0))
                })
                .collect();
            out.push_str(&format!(
                "d3d verify: {} draws wrote shadowed registers {n} times, {agreed} with the shadow's value{}{}\n",
                hook_name(h),
                if differing.is_empty() { "" } else { "; different: " },
                differing.join(", ")
            ));
        }
        let mut by_hook: BTreeMap<u32, Vec<(u32, u64)>> = BTreeMap::new();
        for (&(h, r), &n) in &self.unshadowed_writes {
            by_hook.entry(h).or_default().push((r, n));
        }
        for (h, mut regs) in by_hook {
            regs.sort_by_key(|r| std::cmp::Reverse(r.1));
            let name = crate::HOOKS.get(h as usize).map_or("?", |h| h.function);
            let list: Vec<String> = regs.iter().take(40).map(|(r, n)| format!("{r:#06X} {n}")).collect();
            out.push_str(&format!(
                "d3d verify: {name} draws also wrote registers outside the table ({} kinds): {}\n",
                regs.len(),
                list.join(", ")
            ));
        }
        let per_hook: Vec<String> = self
            .draws
            .iter()
            .map(|(h, n)| {
                let name = crate::HOOKS.get(*h as usize).map_or("?", |h| h.function);
                format!("{name} {} of {n}", self.draws_mismatched.get(h).copied().unwrap_or(0))
            })
            .collect();
        out.push_str(&format!(
            "d3d verify: {} draws compared ({} registers each); draws with mismatches by hook: {}\n",
            self.draws(),
            device::shadowed_registers().count(),
            per_hook.join(", ")
        ));
        let mismatches = self.mismatches();
        if mismatches.is_empty() {
            out.push_str("d3d verify: every shadowed register matched the command processor\n");
        }
        for Mismatch { register: r, checked, mismatched, samples } in mismatches {
            let block = BLOCKS.iter().find(|b| b.contains(r)).map_or("", |b| b.name);
            let samples: Vec<String> = samples
                .iter()
                .map(|(s, c, h)| {
                    let hook = crate::HOOKS.get(*h as usize).map_or("?", |h| h.function);
                    format!("shadow {s:#010X} cp {c:#010X} ({hook})")
                })
                .collect();
            let by_hook: Vec<String> = self.stats[&r]
                .by_hook
                .iter()
                .map(|(h, n)| format!("{} {n}", crate::HOOKS.get(*h as usize).map_or("?", |h| h.function)))
                .collect();
            let s = &self.stats[&r];
            let writers: Vec<String> =
                s.writers.iter().map(|(k, (n, a))| format!("packet {k:#X} {n} (last at {a:#010X})")).collect();
            let written: Vec<String> =
                s.written_samples.iter().map(|(sh, w)| format!("shadow {sh:#010X} written {w:#010X}")).collect();
            out.push_str(&format!(
                "d3d verify: register {r:#06X} ({block}): {mismatched} of {checked} differ ({}); last written by {}; \
                 {} written by the draw's packets ({}); {}\n",
                by_hook.join(", "),
                writers.join(", "),
                s.written,
                written.join("; "),
                samples.join("; ")
            ));
        }
        if self.discovery_draws > 0 {
            for (r, offsets, values) in self.discoveries() {
                let offsets: Vec<String> = offsets.iter().take(4).map(|o| format!("+{o:#X}")).collect();
                out.push_str(&format!(
                    "d3d verify: register {r:#06X} (not in the table) always equalled device {} over {} draws ({values} values)\n",
                    offsets.join(" / "),
                    self.discovery_draws
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_and_discover() {
        // A fake device: every shadowed register holds its own number, the rest of the device
        // holds `0x1000_0000 | offset`, and register 0x2080 is kept at +0x3300.
        let base = 0x4000_0000u32;
        let mut dev = vec![0u32; (device::SIZE / 4) as usize];
        for (i, v) in dev.iter_mut().enumerate() {
            *v = 0x1000_0000 | (i as u32 * 4);
        }
        for (r, o) in device::shadowed_registers() {
            dev[o as usize / 4] = r;
        }
        let mut v = Verifier::default();
        for draw in 0..5u32 {
            dev[0x3300 / 4] = draw * 7;
            let snap = Snapshot::capture(draw, 0, base, true, |a| dev[((a - base) / 4) as usize]);
            let cp = |r| match r {
                0x2080 => draw * 7,
                0x2205 if draw == 3 => 0xDEAD,
                r if device::shadow_offset(r).is_some() => r,
                _ => 0x5555_5555,
            };
            v.compare(&snap, &[], cp, |_| (0x100, 0x1000));
        }
        let m = v.mismatches();
        assert_eq!(
            m,
            vec![Mismatch { register: 0x2205, checked: 5, mismatched: 1, samples: vec![(0x2205, 0xDEAD, 0)] }]
        );
        let d = v.discoveries();
        assert_eq!(d, vec![(0x2080, vec![0x3300], 5)]);
        assert!(v.report().contains("register 0x2205"));
    }
}
