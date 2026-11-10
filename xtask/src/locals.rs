//! Registers in C++ locals for the functions that can have them.
//!
//! XenonRecomp can keep the registers the Xbox 360 ABI never passes between functions (r0, r2,
//! r11, r12, f0, v14-v63 and the non-volatile r14-r31, f14-f31, v64-v127) in locals of each
//! generated function instead of in `PPCContext` (`non_argument_as_local` and
//! `non_volatile_as_local`): the compiler then holds them in host registers, and the register
//! save/restore helper calls disappear (about a tenth of Skyrim's CPU time). It is all or
//! nothing there, and the title has code the ABI doesn't describe: `__finally` funclets and other
//! function chunks that use their parent's registers, helpers with conventions of their own,
//! setjmp. With every function converted Skyrim dies at boot.
//!
//! So the title is recompiled twice, with and without the two options, and the output is put
//! together per function: [`context_functions`] finds, in the output that keeps everything in
//! the context, the functions that pass values to or from another function in one of those
//! registers; they and the functions on the other side keep their context bodies, and every
//! other function gets its body from the output with locals ([`mix`]). The two kinds call each
//! other freely: a function with locals neither reads nor writes the context's copies of these
//! registers, and one without saves and restores them as the ABI says.
//!
//! A function keeps the context when it
//!
//! * reads one of the registers before writing it, other than to save it in its stack frame
//!   (a chunk continuing its parent's work): its direct callers keep the context too;
//! * reads a volatile one after a call without writing it first (the callee returned something
//!   in it): that callee keeps the context too;
//! * writes a non-volatile one and never restores it (it hands it to its caller): its direct
//!   callers keep the context too;
//! * calls setjmp or longjmp (locals changed after setjmp are lost by the jump);
//! * is one of the save/restore helpers, which work on their caller's registers.
//!
//! Function pointers are not followed: a function reached only indirectly that used one of
//! these registers as an argument would be missed. The compiler's own helpers are called
//! directly, and nothing else has a reason to.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Registers as indices: r0-r31, then f0-f31, then v0-v127.
const REGS: usize = 32 + 32 + 128;

fn reg_index(kind: u8, number: u32) -> Option<usize> {
    match kind {
        b'r' if number < 32 => Some(number as usize),
        b'f' if number < 32 => Some(32 + number as usize),
        b'v' if number < 128 => Some(64 + number as usize),
        _ => None,
    }
}

fn reg_name(index: usize) -> String {
    match index {
        0..32 => format!("r{index}"),
        32..64 => format!("f{}", index - 32),
        _ => format!("v{}", index - 64),
    }
}

/// Volatile registers that become locals: r0, r2, r11, r12, f0 and v14-v63. (The ABI calls
/// v14-v31 non-volatile, but the title's compiler never saves them and its leaf functions use
/// them freely.)
fn volatile(index: usize) -> bool {
    matches!(index, 0 | 2 | 11 | 12 | 32) || (64 + 14..64 + 64).contains(&index)
}

/// Non-volatile registers that become locals: r14-r31, f14-f31, v64-v127.
fn non_volatile(index: usize) -> bool {
    (14..32).contains(&index) || (32 + 14..64).contains(&index) || (64 + 64..REGS).contains(&index)
}

fn local(index: usize) -> bool {
    volatile(index) || non_volatile(index)
}

/// The registers `text` mentions as `ctx.<r|f|v><number>.`.
fn registers(text: &str, mut each: impl FnMut(usize)) {
    let bytes = text.as_bytes();
    let mut at = 0;
    while let Some(found) = text[at..].find("ctx.") {
        let start = at + found + 4;
        at = start;
        let Some(&kind) = bytes.get(start) else { break };
        let digits = bytes[start + 1..].iter().take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 || bytes.get(start + 1 + digits) != Some(&b'.') {
            continue;
        }
        if let Some(index) = text[start + 1..start + 1 + digits].parse().ok().and_then(|n| reg_index(kind, n)) {
            each(index);
        }
    }
}

/// `ctx.<kind><number>.` at the start of `text`: the register and the text after the dot.
fn leading_register<'a>(text: &'a str, kinds: &[u8]) -> Option<(usize, &'a str)> {
    let rest = text.strip_prefix("ctx.")?;
    let kind = *rest.as_bytes().first()?;
    if !kinds.contains(&kind) {
        return None;
    }
    let digits = rest[1..].bytes().take_while(u8::is_ascii_digit).count();
    let index = reg_index(kind, rest[1..1 + digits].parse().ok()?)?;
    Some((index, rest[1 + digits..].strip_prefix('.')?))
}

fn word(text: &str) -> usize {
    text.bytes().take_while(|b| b.is_ascii_alphanumeric() || *b == b'_').count()
}

/// `ctx.<r|f>N.field = rest`: the register and `rest`.
fn assignment(line: &str) -> Option<(usize, &str)> {
    let (index, after) = leading_register(line, b"rf")?;
    let field = word(after);
    if field == 0 {
        return None;
    }
    Some((index, after[field..].strip_prefix(" = ")?))
}

/// `simde_mm_store*((cast)ctx.vN.field, rest`: the register and `rest`.
fn vector_store(line: &str) -> Option<(usize, &str)> {
    let after = line.strip_prefix("simde_mm_store")?;
    let after = after[word(after)..].strip_prefix('(')?;
    let after = match after.strip_prefix('(') {
        Some(cast) => {
            let end = cast.find(')')?;
            if cast[..end].contains('(') {
                return None;
            }
            &cast[end + 1..]
        }
        None => after,
    };
    let (index, after) = leading_register(after, b"v")?;
    Some((index, after[word(after)..].strip_prefix(", ")?))
}

fn digits(text: &str) -> Option<&str> {
    let n = text.bytes().take_while(u8::is_ascii_digit).count();
    (n > 0).then(|| &text[n..])
}

/// `PPC_STORE_U64(ctx.r1.u32 + -N, ctx.<r|f>K.u64);`: a register saved in the stack frame.
fn stack_save(line: &str) -> Option<usize> {
    let after = digits(line.strip_prefix("PPC_STORE_U64(ctx.r1.u32 + -")?)?.strip_prefix(", ")?;
    let (index, after) = leading_register(after, b"rf")?;
    (after == "u64);").then_some(index)
}

/// `ctx.<r|f>K.u64 = PPC_LOAD_U64(ctx.r1.u32 + N);` given what follows the ` = `.
fn stack_restore(value: &str) -> bool {
    let Some(after) = value.strip_prefix("PPC_LOAD_U64(ctx.r1.u32 + ") else { return false };
    digits(after.strip_prefix('-').unwrap_or(after)) == Some(");")
}

/// `name(ctx, base);`
fn call(line: &str) -> Option<&str> {
    let n = word(line);
    (n > 0 && &line[n..] == "(ctx, base);").then(|| &line[..n])
}

/// `__save<kind>_N` / `__rest<kind>_N`: whether it restores, and the registers it covers.
fn helper(name: &str) -> Option<(bool, Vec<usize>)> {
    let rest = name.strip_prefix("__")?;
    let (restores, rest) = match rest.strip_prefix("rest") {
        Some(rest) => (true, rest),
        None => (false, rest.strip_prefix("save")?),
    };
    let (kind, first) = rest.split_once('_')?;
    if first.is_empty() || !first.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let first: usize = first.parse().ok()?;
    let covered = match kind {
        "gprlr" => (first..32).chain([12]).collect(),
        "fpr" => (first..32).map(|n| 32 + n).collect(),
        "vmx" => (first..if first < 64 { 32 } else { 128 }).map(|n| 64 + n).collect(),
        _ => return None,
    };
    Some((restores, covered))
}

/// `loc_XXXXXXXX` at the start of a label line.
fn label(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("loc_")?;
    let n = rest.bytes().take_while(|b| b.is_ascii_digit() || (b'A'..=b'F').contains(b)).count();
    (n > 0 && rest[n..].starts_with(':')).then(|| &line[..4 + n])
}

/// The targets of the `goto loc_X;` in `line`.
fn gotos(line: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(i) = line[at..].find("goto loc_") {
        let start = at + i + 5;
        let rest = &line[start + 4..];
        let n = rest.bytes().take_while(|b| b.is_ascii_digit() || (b'A'..=b'F').contains(b)).count();
        if n > 0 && rest[n..].starts_with(';') {
            found.push(&line[start..start + 4 + n]);
        }
        at = start;
    }
    found
}

/// Name of the function a `PPC_FUNC_IMPL(__imp__name) {` line starts.
fn function_start(line: &str) -> Option<&str> {
    line.strip_prefix("PPC_FUNC_IMPL(__imp__")?.strip_suffix(") {")
}

/// A call's callee: nothing a caller's registers depend on, a function called indirectly, or a
/// named one.
#[derive(Clone, Copy, PartialEq)]
enum Callee {
    None,
    Unknown,
    Named(u32),
}

struct Statement {
    reads: Vec<usize>,
    writes: Vec<usize>,
    callee: Callee,
}

/// What a function's body says about the registers that would become locals.
#[derive(Default)]
struct Flow {
    /// Read before being written.
    live_in: BTreeSet<usize>,
    /// Callees (ids; `u32::MAX` for an indirect call) a volatile register was read after.
    read_after: BTreeSet<u32>,
    /// Non-volatile registers written and never restored.
    clobbered: BTreeSet<usize>,
    callees: BTreeSet<u32>,
    setjmp: bool,
}

const UNKNOWN_CALLEE: u32 = u32::MAX;

/// Function names as ids. A callee's id is that of its body: calls name either `sub_X` or
/// `__imp__sub_X` (imports keep their `__imp__Name`, which has no body here).
#[derive(Default)]
struct Names {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl Names {
    fn id(&mut self, name: &str) -> u32 {
        let name = match name.strip_prefix("__imp__") {
            Some(body) if !name.starts_with("__imp____") => body,
            _ => name,
        };
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }
}

fn analyse(lines: &[&str], names: &mut Names) -> Flow {
    let mut flow = Flow::default();
    let mut blocks: Vec<Vec<Statement>> = vec![Vec::new()];
    let mut labels: HashMap<&str, usize> = HashMap::new();
    let mut jumps: Vec<Vec<&str>> = vec![Vec::new()];
    // Whether control can run from a block into the next (absent: yes).
    let mut falls: HashMap<usize, bool> = HashMap::new();
    let (mut wrote, mut restored) = (BTreeSet::new(), BTreeSet::new());
    for &raw in lines {
        if let Some(name) = label(raw) {
            if !blocks.last().unwrap().is_empty() || blocks.len() == 1 {
                falls.entry(blocks.len() - 1).or_insert(true);
                blocks.push(Vec::new());
                jumps.push(Vec::new());
            }
            labels.insert(name, blocks.len() - 1);
            continue;
        }
        let line = raw.trim_start();
        let block = blocks.len() - 1;
        let mut statement = Statement { reads: Vec::new(), writes: Vec::new(), callee: Callee::None };
        if raw.contains("setjmp") || raw.contains("longjmp") {
            flow.setjmp = true;
        }
        let direct = call(line).filter(|_| !["PPC_", "simde", "ctx."].iter().any(|p| line.starts_with(p)));
        if let Some(name) = direct {
            match helper(name) {
                Some((restores, covered)) => {
                    if restores {
                        restored.extend(covered.iter().copied());
                        statement.writes = covered;
                    }
                }
                None => {
                    let id = names.id(name);
                    flow.callees.insert(id);
                    statement.callee = Callee::Named(id);
                }
            }
        } else if raw.contains("PPC_CALL_INDIRECT_FUNC") {
            statement.callee = Callee::Unknown;
            registers(raw, |r| statement.reads.push(r));
        } else if let Some((target, value)) = assignment(line) {
            registers(value, |r| statement.reads.push(r));
            statement.writes.push(target);
            // Only `ctx.rK.u64 = PPC_LOAD_U64(ctx.r1.u32 + N);` puts a saved value back.
            if line[..line.len() - value.len()].ends_with(".u64 = ") && stack_restore(value) {
                restored.insert(target);
            }
        } else if let Some((target, value)) = vector_store(line) {
            registers(value, |r| statement.reads.push(r));
            statement.writes.push(target);
        } else {
            // Anything else reads what it names (element writes of a vector register too).
            let saved = stack_save(line).filter(|&r| non_volatile(r));
            registers(raw, |r| {
                if Some(r) != saved {
                    statement.reads.push(r);
                }
            });
        }
        wrote.extend(statement.writes.iter().copied().filter(|&r| non_volatile(r)));
        statement.reads.retain(|&r| local(r));
        statement.writes.retain(|&r| local(r));
        blocks[block].push(statement);
        let targets = gotos(raw);
        let top_level = !raw.starts_with("\t\t");
        if !targets.is_empty() {
            // The target gets the state here, so the block ends here. Unconditional only at
            // function level: the gotos inside a switch are its cases.
            jumps[block].extend(targets);
            falls.insert(block, !(line.starts_with("goto ") && top_level));
            blocks.push(Vec::new());
            jumps.push(Vec::new());
        } else if raw.trim() == "return;" && top_level {
            falls.insert(block, false);
            blocks.push(Vec::new());
            jumps.push(Vec::new());
        }
    }
    flow.clobbered = wrote.difference(&restored).copied().collect();

    let count = blocks.len();
    let successors: Vec<Vec<usize>> = (0..count)
        .map(|b| {
            let mut next: Vec<usize> = jumps[b].iter().filter_map(|l| labels.get(l).copied()).collect();
            if falls.get(&b).copied().unwrap_or(true) && b + 1 < count {
                next.push(b + 1);
            }
            next
        })
        .collect();
    let used: BTreeSet<usize> = blocks.iter().flatten().flat_map(|s| s.reads.iter().copied()).collect();
    if used.is_empty() {
        return flow;
    }
    // Per used register: whether it may still hold its value at entry, and the callees that may
    // have left the value it holds.
    type State = BTreeMap<usize, (bool, BTreeSet<u32>)>;
    let entry: State = used.iter().map(|&r| (r, (true, BTreeSet::new()))).collect();
    let mut states: Vec<Option<State>> = vec![None; count];
    states[0] = Some(entry);
    let mut work = vec![0];
    while let Some(b) = work.pop() {
        let mut state = states[b].clone().unwrap();
        for statement in &blocks[b] {
            for r in &statement.reads {
                let (entry, killers) = &state[r];
                if *entry {
                    flow.live_in.insert(*r);
                }
                flow.read_after.extend(killers.iter().copied());
            }
            for w in &statement.writes {
                if let Some(s) = state.get_mut(w) {
                    *s = (false, BTreeSet::new());
                }
            }
            let callee = match statement.callee {
                Callee::None => continue,
                Callee::Unknown => UNKNOWN_CALLEE,
                Callee::Named(id) => id,
            };
            // The ABI leaves the volatile ones undefined across a call.
            for (r, s) in state.iter_mut() {
                if volatile(*r) {
                    *s = (false, BTreeSet::from([callee]));
                }
            }
        }
        for &next in &successors[b] {
            match &mut states[next] {
                None => {
                    states[next] = Some(state.clone());
                    work.push(next);
                }
                Some(merged) => {
                    let mut changed = false;
                    for (r, (entry, killers)) in &state {
                        let (merged_entry, merged_killers) = merged.get_mut(r).unwrap();
                        if *entry && !*merged_entry {
                            *merged_entry = true;
                            changed = true;
                        }
                        for k in killers {
                            changed |= merged_killers.insert(*k);
                        }
                    }
                    if changed {
                        work.push(next);
                    }
                }
            }
        }
    }
    flow
}

/// The functions of one generated source file: (name, body lines without comments).
fn for_each_function(text: &str, mut each: impl FnMut(&str, &[&str])) {
    let mut current: Option<(&str, Vec<&str>)> = None;
    for line in text.lines() {
        match &mut current {
            None => {
                if let Some(name) = function_start(line) {
                    current = Some((name, Vec::new()));
                }
            }
            Some((name, lines)) => {
                if line.starts_with('}') {
                    each(name, lines);
                    current = None;
                } else if !line.trim().is_empty() && !line.trim_start().starts_with("//") {
                    lines.push(line);
                }
            }
        }
    }
}

fn sources(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut files: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("ppc_recomp.") && name.ends_with(".cpp")
        })
        .collect();
    files.sort();
    Ok(files)
}

/// The functions of the output in `dir` (generated with every register in the context) that
/// must keep the context, each with why.
pub fn context_functions(dir: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let mut names = Names::default();
    let mut flows: BTreeMap<u32, Flow> = BTreeMap::new();
    for path in sources(dir)? {
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for_each_function(&text, |name, lines| {
            let flow = analyse(lines, &mut names);
            flows.insert(names.id(name), flow);
        });
    }
    let mut callers: HashMap<u32, Vec<u32>> = HashMap::new();
    for (&id, flow) in &flows {
        for &callee in &flow.callees {
            callers.entry(callee).or_default().push(id);
        }
    }
    let list = |regs: &BTreeSet<usize>| regs.iter().map(|&r| reg_name(r)).collect::<Vec<_>>().join(",");
    let mut reasons: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for (&id, flow) in &flows {
        let name = &names.names[id as usize];
        if helper(name).is_some() {
            reasons.entry(id).or_default().push("register save/restore helper".into());
            continue;
        }
        let involve_callers = |why: &str, reasons: &mut BTreeMap<u32, Vec<String>>| {
            for caller in callers.get(&id).into_iter().flatten() {
                reasons.entry(*caller).or_default().push(format!("calls {name} ({why})"));
            }
        };
        if !flow.live_in.is_empty() {
            reasons.entry(id).or_default().push(format!("reads {} before writing", list(&flow.live_in)));
            involve_callers("which reads its caller's registers", &mut reasons);
        }
        if !flow.read_after.is_empty() {
            let callees: Vec<&str> = flow
                .read_after
                .iter()
                .map(|&c| if c == UNKNOWN_CALLEE { "an indirect call" } else { names.names[c as usize].as_str() })
                .collect();
            reasons.entry(id).or_default().push(format!("reads a volatile register after {}", callees.join(", ")));
            for &callee in flow.read_after.iter().filter(|&&c| c != UNKNOWN_CALLEE) {
                reasons.entry(callee).or_default().push(format!("returns a register to {name}"));
            }
        }
        if !flow.clobbered.is_empty() {
            reasons.entry(id).or_default().push(format!("leaves {} changed", list(&flow.clobbered)));
            involve_callers("which changes its caller's registers", &mut reasons);
        }
        if flow.setjmp {
            reasons.entry(id).or_default().push("setjmp/longjmp".into());
        }
    }
    Ok(reasons
        .into_iter()
        .filter(|(id, _)| flows.contains_key(id))
        .map(|(id, why)| (names.names[id as usize].clone(), why))
        .collect())
}

/// Replaces, in the sources of `out` (context output), the body of every function not in
/// `context` with its body from `locals` (the output with registers as locals). Returns how
/// many functions got locals.
pub fn mix(out: &Path, locals: &Path, context: &HashSet<&str>) -> Result<usize, String> {
    /// `text` cut into pieces in order; a function's piece carries its name.
    fn pieces(text: &str) -> Vec<(Option<&str>, &str)> {
        let mut pieces = Vec::new();
        let (mut start, mut at, mut name) = (0, 0, None);
        for line in text.split_inclusive('\n') {
            let end = at + line.len();
            if let Some(n) = function_start(line.trim_end()) {
                pieces.push((None, &text[start..at]));
                (start, name) = (at, Some(n));
            } else if name.is_some() && line.starts_with('}') {
                pieces.push((name.take(), &text[start..end]));
                start = end;
            }
            at = end;
        }
        pieces.push((None, &text[start..]));
        pieces
    }
    let mut converted = 0;
    for path in sources(out)? {
        let other = locals.join(path.file_name().unwrap());
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let other_text = fs::read_to_string(&other).map_err(|e| format!("{}: {e}", other.display()))?;
        let with_locals: HashMap<&str, &str> = pieces(&other_text).into_iter().filter_map(|(n, body)| Some((n?, body))).collect();
        let mut mixed = String::with_capacity(text.len());
        for (name, body) in pieces(&text) {
            match name.filter(|n| !context.contains(n)) {
                Some(name) => {
                    let body = with_locals.get(name).ok_or_else(|| format!("{}: no function {name}", other.display()))?;
                    mixed.push_str(body);
                    converted += 1;
                }
                None => mixed.push_str(body),
            }
        }
        if mixed != text {
            fs::write(&path, mixed).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    Ok(converted)
}
