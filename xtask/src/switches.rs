//! Jump-table discovery for XenonRecomp.
//!
//! Every `bctr` is classified by tracing how the CTR value was computed:
//!
//! * `lwz rT, d(rA); mtctr rT` — a call through a vtable/function pointer.
//!   XenonRecomp already emits these as indirect calls; nothing to do.
//! * `lwzx rT, rTable, rI` with a constant table — an absolute jump table.
//! * `add rT, rBase, rOff` with a constant code base and rOff loaded by
//!   `lbzx`/`lhzx` (optionally shifted) from a constant table — a relative
//!   jump table.
//!
//! The case count comes from the guarding `cmplwi crN, rX, N` before the table
//! load. Tables are written in the target-keyed form the skyrt XenonRecomp
//! patch understands (no `r` key: dispatch on the address in CTR), so the
//! index register may be clobbered before the branch. Any site that does not
//! fully resolve, or whose targets leave the containing function, is reported
//! instead of guessed.

use std::fmt::Write as _;
use std::path::Path;

const BCTR: u32 = 0x4E80_0420;
/// How far back from a bctr the tracer looks.
const WINDOW: usize = 32;

fn op(w: u32) -> u32 {
    w >> 26
}
fn rt(w: u32) -> u32 {
    (w >> 21) & 31
}
fn ra(w: u32) -> u32 {
    (w >> 16) & 31
}
fn rb(w: u32) -> u32 {
    (w >> 11) & 31
}
fn xo10(w: u32) -> u32 {
    (w >> 1) & 0x3FF
}
fn simm(w: u32) -> i32 {
    (w & 0xFFFF) as i16 as i32
}

fn is_mtctr(w: u32) -> Option<u32> {
    (op(w) == 31 && xo10(w) == 467 && ((w >> 11) & 0x3FF) == (9 << 5)).then(|| rt(w))
}

/// Calls clobber volatile registers and unconditional transfers mean the
/// preceding code is on another path; conditional branches and returns
/// (e.g. `bclr` guards) fall through, so tracing continues past them.
fn ends_trace(w: u32) -> bool {
    let link = w & 1 == 1;
    match op(w) {
        16 => link,
        18 => true,
        19 => link || is_unconditional(w),
        _ => false,
    }
}

/// `blr` / `bctr` / `b` that always transfers control.
fn is_unconditional(w: u32) -> bool {
    match op(w) {
        18 => w & 1 == 0,
        19 => matches!(xo10(w), 16 | 528) && (w >> 21) & 0x14 == 0x14 && w & 1 == 0,
        _ => false,
    }
}

/// Branch target of an in-function branch (b/bc, not calls).
fn branch_target(w: u32, at: u32) -> Option<u32> {
    if w & 1 == 1 || w & 2 == 2 {
        return None;
    }
    match op(w) {
        18 => Some(at.wrapping_add((((w & 0x03FF_FFFC) << 6) as i32 >> 6) as u32)),
        16 => Some(at.wrapping_add((w & 0xFFFC) as i16 as i32 as u32)),
        _ => None,
    }
}

/// Register copied by `mr` (or rS,rS) or `rlwinm rA,rS,0,0,31`.
fn move_source(w: u32) -> Option<u32> {
    if op(w) == 31 && xo10(w) == 444 && rt(w) == rb(w) {
        return Some(rt(w));
    }
    if op(w) == 21 && (w >> 11) & 31 == 0 && (w >> 6) & 31 == 0 && (w >> 1) & 31 == 31 {
        return Some(rt(w));
    }
    None
}

/// GPRs written by an instruction (enough coverage for compiler-generated
/// table dispatch; anything unknown is treated as writing nothing).
fn writes(w: u32) -> [Option<u32>; 2] {
    match op(w) {
        7 | 8 | 12 | 13 | 14 | 15 | 32 | 34 | 40 | 42 | 58 => [Some(rt(w)), None],
        33 | 35 | 41 | 43 => [Some(rt(w)), Some(ra(w))],
        20 | 21 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 => [Some(ra(w)), None],
        31 => match xo10(w) {
            // loads
            23 | 87 | 279 | 343 | 21 | 534 | 790 | 20 => [Some(rt(w)), None],
            55 | 119 | 311 | 375 | 53 => [Some(rt(w)), Some(ra(w))],
            // logical / shifts (RA destination)
            28 | 60 | 444 | 412 | 316 | 124 | 284 | 476 | 24 | 536 | 792 | 824 | 954 | 922 | 986 | 26 | 27 | 539
            | 794 | 58 => [Some(ra(w)), None],
            339 | 19 => [Some(rt(w)), None],
            x => {
                // XO-form arithmetic (9-bit XO, OE bit ignored).
                match x & 0x1FF {
                    266 | 40 | 8 | 10 | 138 | 136 | 202 | 200 | 234 | 232 | 104 | 235 | 75 | 11 | 491 | 459 | 233
                    | 73 | 9 | 489 | 457 => [Some(rt(w)), None],
                    _ => [None, None],
                }
            }
        },
        _ => [None, None],
    }
}

fn writes_reg(w: u32, r: u32) -> bool {
    // `nop` is `ori r0, r0, 0`, which the compiler schedules inside dispatch
    // sequences; it does not change r0.
    w != 0x6000_0000 && writes(w).contains(&Some(r))
}

struct Code<'a> {
    base: u32,
    image: &'a [u8],
}

impl Code<'_> {
    fn word(&self, addr: u32) -> Option<u32> {
        let off = addr.checked_sub(self.base)? as usize;
        self.image.get(off..off + 4).map(|b| u32::from_be_bytes(b.try_into().unwrap()))
    }

    /// Like [`Code::def_raw`] but looks through register moves.
    fn def(&self, r: u32, at: u32, lo: u32) -> Option<(u32, u32)> {
        let (a, w) = self.def_raw(r, at, lo)?;
        match move_source(w) {
            Some(src) => self.def(src, a, lo),
            None => Some((a, w)),
        }
    }

    /// Nearest instruction before `at` (bounded by `lo`) that writes `r`;
    /// stops at calls and unconditional transfers.
    fn def_raw(&self, r: u32, at: u32, lo: u32) -> Option<(u32, u32)> {
        let mut a = at;
        for _ in 0..WINDOW {
            if a < lo + 4 {
                return None;
            }
            a -= 4;
            let w = self.word(a)?;
            if writes_reg(w, r) {
                return Some((a, w));
            }
            if ends_trace(w) {
                return None;
            }
        }
        None
    }

    /// Constant value of `r` at `at` built from lis/addi/ori.
    fn constant(&self, r: u32, at: u32, lo: u32) -> Option<u32> {
        let (a, w) = self.def(r, at, lo)?;
        match op(w) {
            15 if ra(w) == 0 => Some((simm(w) as u32) << 16),
            15 => Some(self.constant(ra(w), a, lo)?.wrapping_add((simm(w) as u32) << 16)),
            14 if ra(w) == 0 => Some(simm(w) as u32),
            14 => Some(self.constant(ra(w), a, lo)?.wrapping_add(simm(w) as u32)),
            24 => Some(self.constant(rt(w), a, lo)? | (w & 0xFFFF)),
            _ => None,
        }
    }

    /// Upper bound (inclusive) on the index register used by the table load
    /// at `at`: the nearest `cmplwi crN, rX, N` on the index or on a register
    /// it was copied or scaled from. A non-copy redefinition of every tracked
    /// register ends the search.
    fn bound(&self, index: u32, at: u32, lo: u32) -> Option<u32> {
        let mut sources = vec![index];
        // Compares already passed (they executed after anything still ahead
        // in the backward walk); a later-found copy `dst = src` makes a
        // compare on `src` apply to `dst` as well.
        let mut passed: Vec<(u32, u32)> = Vec::new();
        let mut a = at;
        for _ in 0..WINDOW * 2 {
            if a < lo + 4 || sources.is_empty() {
                return None;
            }
            a -= 4;
            let w = self.word(a)?;
            // cmplwi crN, rX, UIMM (opcode 10, L=0)
            if op(w) == 10 && (w >> 21) & 1 == 0 {
                if sources.contains(&ra(w)) {
                    return Some(w & 0xFFFF);
                }
                passed.push((ra(w), w & 0xFFFF));
            }
            if let Some(src) = move_source(w)
                && writes(w).into_iter().flatten().any(|d| sources.contains(&d))
                && let Some(&(_, imm)) = passed.iter().find(|(r, _)| *r == src)
            {
                return Some(imm);
            }
            for dst in writes(w).into_iter().flatten() {
                if w == 0x6000_0000 || !sources.contains(&dst) {
                    continue;
                }
                sources.retain(|&r| r != dst);
                // Copies and scalings carry the index forward from their source.
                let scaled = op(w) == 21 && (w >> 6) & 31 == 0;
                let doubled = op(w) == 31 && xo10(w) & 0x1FF == 266 && ra(w) == rb(w);
                if let Some(src) = move_source(w).or((scaled).then(|| rt(w))).or(doubled.then(|| ra(w))) {
                    sources.push(src);
                }
            }
            if ends_trace(w) {
                return None;
            }
        }
        None
    }
}

#[derive(Default)]
pub struct Report {
    pub bctr: usize,
    pub import_thunks: usize,
    pub indirect_calls: usize,
    /// Extents computed for functions without .pdata that contain tables.
    pub functions: Vec<(u32, u32)>,
    pub tables: Vec<(u32, Vec<u32>)>,
    pub unresolved: Vec<(u32, String)>,
}

fn containing(funcs: &[(u32, u32)], addr: u32) -> Option<(u32, u32)> {
    let i = funcs.partition_point(|&(s, _)| s <= addr);
    let (s, n) = *funcs.get(i.checked_sub(1)?)?;
    (addr < s + n).then_some((s, n))
}

/// Extent of a function without .pdata that contains the jump-table bctr at
/// `bctr`. Starting from the code after the previous unconditional transfer,
/// it grows backwards over earlier blocks that branches or case labels reach,
/// and ends at the first unconditional transfer (the bctr included) past every
/// forward target, skipping inline table data.
enum Leaf {
    Call,
    Table(u32, u32),
    Unresolved(String),
}

fn leaf(code: &Code, funcs: &[(u32, u32)], entries: &[u32], bctr: u32) -> Leaf {
    match leaf_extent(code, funcs, entries, bctr) {
        Ok(Some(extent)) => Leaf::Table(extent.0, extent.1),
        Ok(None) => Leaf::Call,
        Err(e) => Leaf::Unresolved(e),
    }
}

/// Ok(None) when the bctr is a call; Ok(Some(extent)) for a jump table.
fn leaf_extent(code: &Code, funcs: &[(u32, u32)], entries: &[u32], bctr: u32) -> Result<Option<(u32, u32)>, String> {
    let debug = std::env::var_os("SKYRT_DEBUG_SWITCHES").is_some();
    let i = funcs.partition_point(|&(s, _)| s <= bctr);
    let prev_end = i.checked_sub(1).map_or(0, |i| funcs[i].0 + funcs[i].1);
    let next_start = funcs.get(i).map_or(u32::MAX, |f| f.0);
    let after_terminator_before = |mut a: u32| -> Result<u32, String> {
        while a > prev_end && bctr - a < 0x4000 {
            let w = code.word(a - 4).ok_or("unreadable code")?;
            if is_unconditional(w) || w == 0 {
                break;
            }
            a -= 4;
        }
        Ok(a)
    };

    // Labels, using the widest extent the neighbouring .pdata functions allow.
    let wide_lo = prev_end.max(bctr.saturating_sub(0x4000));
    let labels = match classify(code, funcs, bctr, wide_lo, next_start.min(bctr + 0x4000) - wide_lo) {
        Ok(Some(l)) => l,
        Ok(None) => return Ok(None),
        Err(e) => {
            if debug {
                eprintln!("leaf {bctr:08X}: {e}");
            }
            return Err(e);
        }
    };
    let table_end = {
        // Inline absolute table right after the bctr.
        let n = labels.iter().enumerate().take_while(|&(k, &l)| code.word(bctr + 4 + 4 * k as u32) == Some(l)).count();
        bctr + 4 + 4 * n as u32
    };

    // Start at the nearest known entry point (call target or address taken in
    // data) before the dispatch; fall back to the code after the previous
    // return when none is known.
    let nearest_entry = |at: u32| -> Option<u32> {
        let i = entries.partition_point(|&e| e <= at);
        i.checked_sub(1).map(|i| entries[i]).filter(|&e| e > prev_end.saturating_sub(1) && at - e < 0x4000)
    };
    let mut start = match nearest_entry(bctr) {
        Some(e) => e,
        None => after_terminator_before(bctr)?,
    };
    loop {
        let mut lowest = labels.iter().copied().min().ok_or("empty table")?.min(start);
        let mut a = start;
        while a < bctr {
            if let Some(t) = branch_target(code.word(a).ok_or("unreadable code")?, a)
                && t > prev_end
                && t < lowest
            {
                lowest = t;
            }
            a += 4;
        }
        if lowest >= start {
            break;
        }
        start = match nearest_entry(lowest) {
            Some(e) => e,
            None => after_terminator_before(lowest)?,
        };
    }

    let mut reach = labels.iter().copied().max().ok_or("empty table")?.max(bctr);
    let mut a = start;
    while a < bctr {
        if let Some(t) = branch_target(code.word(a).ok_or("unreadable code")?, a)
            && t > reach
            && t < next_start
        {
            reach = t;
        }
        a += 4;
    }
    // Walk forward from the bctr (itself a terminator), past the table.
    let mut after_terminator = true;
    let mut a = table_end;
    loop {
        // `reach` is an address that must be inside: stop only after it.
        if a > reach && after_terminator {
            if debug {
                let lo = labels.iter().min().unwrap();
                let hi = labels.iter().max().unwrap();
                eprintln!(
                    "leaf {bctr:08X}: extent {start:08X}..{a:08X} labels {lo:08X}..{hi:08X} table_end {table_end:08X} prev_end {prev_end:08X} next {next_start:08X}"
                );
            }
            return Ok(Some((start, a - start)));
        }
        if a >= next_start {
            break;
        }
        let w = code.word(a).ok_or("unreadable code")?;
        if let Some(t) = branch_target(w, a)
            && t > reach
            && t < next_start
        {
            reach = t;
        }
        after_terminator = is_unconditional(w);
        a += 4;
    }
    if debug {
        eprintln!("leaf {bctr:08X}: start {start:08X} reach {reach:08X} ran into next function {next_start:08X}");
    }
    Err("jump table in a function without .pdata bounds".into())
}

pub fn analyze(xex: &xex::Xex) -> Result<Report, String> {
    let code = Code { base: xex.image_base(), image: xex.image() };
    let funcs = xex.pdata_functions().map_err(|e| e.to_string())?;
    let thunks: Vec<u32> = xex.imports().map_err(|e| e.to_string())?.iter().filter_map(|i| i.thunk_addr).collect();
    let code_sections: Vec<(u32, u32)> = xex
        .sections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|s| s.is_code())
        .map(|s| (code.base + s.virtual_address, s.virtual_size))
        .collect();
    // Every `bl` target. XenonRecomp makes each one its own function, so an
    // entry point inside a table-dispatching function needs an extent that
    // reaches the table's cases too.
    let mut call_targets: Vec<u32> = Vec::new();
    for &(start, size) in &code_sections {
        for a in (start..start + size).step_by(4) {
            let w = code.word(a).unwrap_or(0);
            if op(w) == 18 && w & 3 == 1 {
                call_targets.push(a.wrapping_add((((w & 0x03FF_FFFC) << 6) as i32 >> 6) as u32));
            }
        }
    }
    call_targets.sort_unstable();
    call_targets.dedup();
    // Entry points: call targets, .pdata starts, and code addresses stored in
    // data sections (vtables, function-pointer tables).
    let in_code = |v: u32| code_sections.iter().any(|&(s, n)| v >= s && v < s + n) && v.is_multiple_of(4);
    let mut entries: Vec<u32> = call_targets.clone();
    entries.extend(funcs.iter().map(|f| f.0));
    for sec in xex.sections().map_err(|e| e.to_string())?.into_iter().filter(|s| !s.is_code()) {
        let start = code.base + sec.virtual_address;
        for a in (start..start + sec.virtual_size).step_by(4) {
            if let Some(v) = code.word(a).filter(|&v| in_code(v)) {
                entries.push(v);
            }
        }
    }
    entries.sort_unstable();
    entries.dedup();
    let mut report = Report::default();
    for s in xex.sections().map_err(|e| e.to_string())?.into_iter().filter(|s| s.is_code()) {
        let start = code.base + s.virtual_address;
        for a in (start..start + s.virtual_size).step_by(4) {
            if code.word(a) != Some(BCTR) {
                continue;
            }
            report.bctr += 1;
            if thunks.iter().any(|&t| a >= t && a < t + 16) {
                report.import_thunks += 1;
                continue;
            }
            // Leaf functions (typically vtable thunks) have no .pdata entry;
            // they can still be classified as indirect calls, but a jump table
            // in one needs its bounds, which are unknown.
            let pdata = containing(&funcs, a);
            let (fstart, fsize) = match pdata {
                Some(f) => f,
                None => match leaf(&code, &funcs, &entries, a) {
                    Leaf::Table(s, n) => (s, n),
                    Leaf::Call => {
                        report.indirect_calls += 1;
                        continue;
                    }
                    Leaf::Unresolved(why) => {
                        report.unresolved.push((a, why));
                        continue;
                    }
                },
            };
            match classify(&code, &funcs, a, fstart, fsize) {
                Ok(None) => report.indirect_calls += 1,
                Ok(Some(labels)) => {
                    if pdata.is_none() {
                        report.functions.push((fstart, fsize));
                    }
                    let end = fstart + fsize;
                    let lo = entries.partition_point(|&t| t <= fstart);
                    let hi = entries.partition_point(|&t| t <= a);
                    for &entry in &entries[lo..hi] {
                        report.functions.push((entry, end - entry));
                    }
                    report.tables.push((a, labels));
                }
                Err(why) => report.unresolved.push((a, why)),
            }
        }
    }
    Ok(report)
}

/// Ok(None): an indirect call. Ok(Some(labels)): a jump table.
fn classify(code: &Code, funcs: &[(u32, u32)], bctr: u32, fstart: u32, fsize: u32) -> Result<Option<Vec<u32>>, String> {
    let lo = fstart;
    // Find mtctr rT.
    let mut a = bctr;
    let target_reg = loop {
        if a < lo + 4 || bctr - a > WINDOW as u32 * 4 {
            return Err("no mtctr before bctr".into());
        }
        a -= 4;
        let w = code.word(a).ok_or("unreadable")?;
        if let Some(r) = is_mtctr(w) {
            break r;
        }
    };
    let mtctr = a;
    // No definition inside the function: the target came from the caller
    // (a tail call through a function-pointer argument).
    let Some((def_at, def)) = code.def(target_reg, mtctr, lo) else { return Ok(None) };

    let in_func = |t: u32| t >= fstart && t < fstart + fsize && t.is_multiple_of(4);
    let has_bounds = fsize != 0;
    match (op(def), xo10(def)) {
        // lwz / lwzu / ld from a displacement: call through a pointer.
        (32 | 33 | 58, _) => Ok(None),
        // lwzx rT, rA, rB: absolute table.
        (31, 23) => {
            let (table, index) = match (code.constant(ra(def), def_at, lo), code.constant(rb(def), def_at, lo)) {
                (Some(t), None) => (t, rb(def)),
                (None, Some(t)) => (t, ra(def)),
                _ => return Ok(None), // table base not constant: pointer array lookup
            };
            // A constant table of function pointers: a call through it. Jump
            // table targets sit next to their dispatch; function pointers
            // point at known functions or far away.
            let is_function = |t: u32| funcs.binary_search_by_key(&t, |f| f.0).is_ok();
            let far = |t: u32| t.abs_diff(bctr) > 0x40000;
            if code.word(table).is_some_and(|t| is_function(t) || far(t)) {
                return Ok(None);
            }
            let labels: Vec<u32> = match code.bound(index, def_at, lo) {
                Some(max) => (0..=max).map(|i| code.word(table + i * 4).unwrap_or(0)).collect(),
                // Unbounded (index range checked far away): take entries while
                // they are code addresses in this function and, for an inline
                // table, until the table would run into its own targets.
                None if has_bounds => {
                    let mut out = Vec::new();
                    let mut first_label = u32::MAX;
                    for i in 0..1024 {
                        let at = table + i * 4;
                        if at >= first_label {
                            break;
                        }
                        match code.word(at) {
                            Some(l) if in_func(l) => {
                                first_label = first_label.min(l);
                                out.push(l);
                            }
                            _ => break,
                        }
                    }
                    // Jump-table entries always target their own function, so
                    // a first entry outside it means a table of pointers.
                    if out.is_empty() {
                        return Ok(None);
                    }
                    out
                }
                None => return Err("absolute table without cmplwi bound".into()),
            };
            if !has_bounds {
                Err("jump table in a function without .pdata bounds".into())
            } else if labels.iter().all(|&l| in_func(l)) {
                Ok(Some(labels))
            } else if labels.iter().all(|&l| !in_func(l)) {
                // A constant table of function pointers (tail call through it).
                Ok(None)
            } else {
                Err("absolute table mixes in-function and external targets".into())
            }
        }
        // add rT, rA, rB: code base + scaled offset from a byte/halfword table.
        (31, x) if x & 0x1FF == 266 => {
            let (base, off_reg) = match (code.constant(ra(def), def_at, lo), code.constant(rb(def), def_at, lo)) {
                (Some(b), None) => (b, rb(def)),
                (None, Some(b)) => (b, ra(def)),
                _ => return Err("relative table without constant code base".into()),
            };
            // Optional slwi on the loaded offset.
            let (mut load_at, mut load) = code.def(off_reg, def_at, lo).ok_or("offset not traceable")?;
            let mut shift = 0;
            if op(load) == 21 {
                let sh = (load >> 11) & 31;
                let mb = (load >> 6) & 31;
                let me = (load >> 1) & 31;
                if mb != 0 || me != 31 - sh {
                    return Err("offset transformed by an unsupported rlwinm".into());
                }
                shift = sh;
                (load_at, load) = code.def(rt(load), load_at, lo).ok_or("offset load not traceable")?;
            }
            let width = match (op(load), xo10(load)) {
                (31, 87) => 1,
                (31, 279) => 2,
                (31, 23) => 4,
                _ => return Err("offset not loaded by lbzx/lhzx/lwzx".into()),
            };
            let (table, index) = match (code.constant(ra(load), load_at, lo), code.constant(rb(load), load_at, lo)) {
                (Some(t), None) => (t, rb(load)),
                (None, Some(t)) => (t, ra(load)),
                _ => return Err("offset table base not constant".into()),
            };
            let read = |i: u32| -> Option<u32> {
                let at = table + i * width;
                let w = code.word(at & !3)?;
                Some(match width {
                    1 => (w >> (24 - 8 * (at & 3))) & 0xFF,
                    2 => (w >> (16 - 8 * (at & 2))) & 0xFFFF,
                    _ => w,
                })
            };
            // bound() looks through an slwi that pre-scales the index by the
            // element width, so `max` counts entries, not bytes.
            let labels: Vec<u32> = match code.bound(index, load_at, lo) {
                Some(max) => (0..=max).map(|i| read(i).map(|o| base.wrapping_add(o << shift)).unwrap_or(0)).collect(),
                // Range checked far away: every real entry targets this
                // function, so read until one doesn't. Extra in-function
                // labels are harmless with target-keyed dispatch.
                None if has_bounds => (0..1024)
                    .map(|i| read(i).map(|o| base.wrapping_add(o << shift)).unwrap_or(0))
                    .take_while(|&l| in_func(l))
                    .collect(),
                None => return Err("relative table without cmplwi bound".into()),
            };
            if labels.is_empty() {
                return Err("relative table without cmplwi bound".into());
            }
            if !has_bounds {
                Err("jump table in a function without .pdata bounds".into())
            } else if labels.iter().all(|&l| in_func(l)) {
                Ok(Some(labels))
            } else {
                Err(format!("relative table targets leave the function (base {base:#X}, table {table:#X})"))
            }
        }
        _ => Err(format!("CTR computed by unhandled instruction {def:08X} at {def_at:08X}")),
    }
}

pub fn toml(report: &Report) -> String {
    let mut out = String::from(
        "# Generated by `cargo xtask switches`. Target-keyed tables (no `r`): the\n\
         # skyrt XenonRecomp patch dispatches on the address in CTR.\n",
    );
    let mut functions = report.functions.clone();
    functions.sort_unstable();
    functions.dedup_by_key(|f| f.0);
    for (address, size) in &functions {
        let _ = writeln!(out, "\n[[function]]\naddress = 0x{address:08X}\nsize = 0x{size:X}");
    }
    for (base, labels) in &report.tables {
        let _ = writeln!(out, "\n[[switch]]\nbase = 0x{base:08X}");
        let list: Vec<String> = labels.iter().map(|l| format!("0x{l:08X}")).collect();
        let _ = writeln!(out, "labels = [{}]", list.join(", "));
    }
    out
}

pub fn run(xex_path: &Path, out: &Path) -> Result<(), String> {
    let data = std::fs::read(xex_path).map_err(|e| format!("{}: {e}", xex_path.display()))?;
    let xex = xex::Xex::parse(&data).map_err(|e| format!("{}: {e}", xex_path.display()))?;
    let report = analyze(&xex)?;
    std::fs::write(out, toml(&report)).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "{} bctr: {} import thunks, {} indirect calls, {} jump tables ({} in leaf functions), {} unresolved -> {}",
        report.bctr,
        report.import_thunks,
        report.indirect_calls,
        report.tables.len(),
        report.functions.len(),
        report.unresolved.len(),
        out.display()
    );
    for (a, why) in &report.unresolved {
        println!("  unresolved bctr at {a:08X}: {why}");
    }
    Ok(())
}
