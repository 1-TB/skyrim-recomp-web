//! Recovery of structured control flow: forward jumps become `if` blocks and loop_start/loop_end
//! pairs become WGSL loops when every region they delimit has a single entry. Programs where that
//! doesn't hold (calls, backward or crossing jumps) use the dispatch loop instead.

use crate::parsed::{CfCondition, ParsedCf};
use crate::program::Program;

#[derive(Clone, Debug)]
pub(super) enum Node {
    /// A control flow instruction without branching (exec, nop, alloc...).
    Cf(usize),
    /// Region executed when `condition` holds (the inverse of the jump over it).
    If { condition: CfCondition, inverted: bool, body: Vec<Node>, else_body: Vec<Node> },
    /// Jump past the end of the program: ends the shader when the condition holds.
    Return { condition: CfCondition },
    /// loop_start at `start`, loop_end at `end`.
    Loop { start: usize, end: usize, body: Vec<Node> },
}

struct Builder<'a> {
    program: &'a Program,
    /// (source, target) of every branch.
    branches: Vec<(usize, usize)>,
}

impl Builder<'_> {
    /// Whether only instructions within `[lo, hi)` (or the fall-through into `lo`, or `allowed`)
    /// branch to targets in `[lo, hi)`.
    fn single_entry(&self, lo: usize, hi: usize, allowed: Option<usize>) -> bool {
        self.branches.iter().all(|&(source, target)| {
            !(lo..hi).contains(&target) || (lo..hi).contains(&source) || Some(source) == allowed
        })
    }

    fn build(&self, start: usize, end: usize) -> Option<Vec<Node>> {
        let cf_count = self.program.cf.len();
        let mut nodes = Vec::new();
        let mut i = start;
        while i < end {
            match self.program.cf[i].cf {
                ParsedCf::Nop | ParsedCf::Exec(_) | ParsedCf::Alloc { .. } | ParsedCf::MarkVsFetchDone => {
                    nodes.push(Node::Cf(i));
                    i += 1;
                }
                ParsedCf::Jump { target, condition } => {
                    let target = target as usize;
                    if target <= i {
                        return None;
                    }
                    if target >= cf_count {
                        // Only reachable as the last region of the program.
                        if end != cf_count {
                            return None;
                        }
                        nodes.push(Node::Return { condition });
                        i += 1;
                        continue;
                    }
                    if target > end || !self.single_entry(i + 1, target, None) {
                        return None;
                    }
                    if condition == CfCondition::Always {
                        i = target;
                        continue;
                    }
                    // `(cond) jmp else; then...; jmp join; else: ...; join:` becomes if/else.
                    if let ParsedCf::Jump { target: join, condition: CfCondition::Always } =
                        self.program.cf[target - 1].cf
                    {
                        let join = join as usize;
                        if target - 1 > i && join > target && join <= end && self.single_entry(target, join, Some(i)) {
                            let body = self.build(i + 1, target - 1)?;
                            let else_body = self.build(target, join)?;
                            nodes.push(Node::If { condition, inverted: true, body, else_body });
                            i = join;
                            continue;
                        }
                    }
                    let body = self.build(i + 1, target)?;
                    nodes.push(Node::If { condition, inverted: true, body, else_body: Vec::new() });
                    i = target;
                }
                ParsedCf::LoopStart { skip_address, .. } => {
                    let loop_end = skip_address as usize;
                    if loop_end <= i + 1 || loop_end > end {
                        return None;
                    }
                    let e = loop_end - 1;
                    match self.program.cf[e].cf {
                        ParsedCf::LoopEnd { body_address, .. } if body_address as usize == i + 1 => {}
                        _ => return None,
                    }
                    if !self.single_entry(i + 1, e + 1, Some(e)) {
                        return None;
                    }
                    let body = self.build(i + 1, e)?;
                    nodes.push(Node::Loop { start: i, end: e, body });
                    i = loop_end;
                }
                ParsedCf::LoopEnd { .. } | ParsedCf::Call { .. } | ParsedCf::Return => return None,
            }
        }
        Some(nodes)
    }
}

/// Structured form of the whole program, if it has one.
pub(super) fn structure(program: &Program) -> Option<Vec<Node>> {
    let mut branches = Vec::new();
    for (i, entry) in program.cf.iter().enumerate() {
        match entry.cf {
            ParsedCf::Jump { target, .. } | ParsedCf::Call { target, .. } => branches.push((i, target as usize)),
            ParsedCf::LoopStart { skip_address, .. } => branches.push((i, skip_address as usize)),
            ParsedCf::LoopEnd { body_address, .. } => branches.push((i, body_address as usize)),
            _ => {}
        }
    }
    Builder { program, branches }.build(0, program.cf.len())
}
