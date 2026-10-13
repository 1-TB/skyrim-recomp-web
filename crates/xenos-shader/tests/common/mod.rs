//! Test support: a small Xenos microcode assembler and WGSL validation through naga.
#![allow(dead_code)]

use xenos_shader::ucode::{RawCf, pack_cf_pair};

// ---- Swizzles ---------------------------------------------------------------------------------

fn component(c: char) -> u32 {
    match c {
        'x' | 'r' => 0,
        'y' | 'g' => 1,
        'z' | 'b' => 2,
        'w' | 'a' => 3,
        _ => panic!("bad component {c}"),
    }
}

/// Relative ALU source swizzle from absolute components ("xyzw", "wzyx", "x" = "xxxx").
pub fn alu_swizzle(s: &str) -> u32 {
    let chars: Vec<char> = s.chars().collect();
    (0..4u32).fold(0, |acc, i| {
        let c = component(chars[(i as usize).min(chars.len() - 1)]);
        acc | (((c.wrapping_sub(i)) & 3) << (2 * i))
    })
}

/// Fetch destination swizzle: x y z w 0 1 and _ (keep).
pub fn fetch_dest_swizzle(s: &str) -> u32 {
    s.chars().enumerate().fold(0, |acc, (i, c)| {
        let v = match c {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            'w' => 3,
            '0' => 4,
            '1' => 5,
            '_' => 7,
            _ => panic!("bad fetch swizzle {c}"),
        };
        acc | (v << (3 * i))
    })
}

/// Absolute 2-bit-per-component swizzle (texture fetch sources).
pub fn fetch_src_swizzle(s: &str) -> u32 {
    s.chars().enumerate().fold(0, |acc, (i, c)| acc | (component(c) << (2 * i)))
}

// ---- ALU --------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Src {
    pub reg: u32,
    pub temp: bool,
    pub swizzle: u32,
    pub negate: bool,
    /// Temporaries only.
    pub abs: bool,
    /// Temporaries only: aL-relative.
    pub relative: bool,
}

pub fn r(i: u32) -> Src {
    Src { reg: i, temp: true, swizzle: 0, negate: false, abs: false, relative: false }
}

pub fn c(i: u32) -> Src {
    Src { reg: i, temp: false, swizzle: 0, negate: false, abs: false, relative: false }
}

impl Src {
    pub fn swz(mut self, s: &str) -> Self {
        self.swizzle = alu_swizzle(s);
        self
    }
    pub fn neg(mut self) -> Self {
        self.negate = true;
        self
    }
    pub fn abs(mut self) -> Self {
        self.abs = true;
        self
    }
    pub fn rel(mut self) -> Self {
        self.relative = true;
        self
    }
    fn encoded_reg(&self) -> u32 {
        if self.temp { self.reg | (self.relative as u32) << 6 | (self.abs as u32) << 7 } else { self.reg }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Alu {
    pub vector_op: u32,
    pub scalar_op: u32,
    pub vector_dest: u32,
    pub vector_dest_rel: bool,
    pub scalar_dest: u32,
    pub scalar_dest_rel: bool,
    pub vector_mask: u32,
    pub scalar_mask: u32,
    pub vector_clamp: bool,
    pub scalar_clamp: bool,
    pub export: bool,
    pub abs_constants: bool,
    pub src: [Src; 3],
    pub predicated: Option<bool>,
    pub const_a0: bool,
    pub const_0_rel: bool,
    pub const_1_rel: bool,
}

impl Default for Alu {
    fn default() -> Self {
        Alu {
            vector_op: 2, // max r0._, r0, r0 = vector nop
            scalar_op: 50,
            vector_dest: 0,
            vector_dest_rel: false,
            scalar_dest: 0,
            scalar_dest_rel: false,
            vector_mask: 0,
            scalar_mask: 0,
            vector_clamp: false,
            scalar_clamp: false,
            export: false,
            abs_constants: false,
            src: [r(0), r(0), r(0)],
            predicated: None,
            const_a0: false,
            const_0_rel: false,
            const_1_rel: false,
        }
    }
}

impl Alu {
    /// Vector operation `op dest.mask, srcs...`.
    pub fn vector(op: u32, dest: u32, mask: u32, srcs: &[Src]) -> Alu {
        let mut a = Alu { vector_op: op, vector_dest: dest, vector_mask: mask, ..Default::default() };
        for (i, s) in srcs.iter().enumerate() {
            a.src[i] = *s;
        }
        a
    }
    /// Scalar operation using source 3.
    pub fn scalar(op: u32, dest: u32, mask: u32, src3: Src) -> Alu {
        let mut a = Alu { scalar_op: op, scalar_dest: dest, scalar_mask: mask, ..Default::default() };
        a.src[2] = src3;
        a
    }
    pub fn pred(mut self, condition: bool) -> Self {
        self.predicated = Some(condition);
        self
    }
    pub fn export(mut self) -> Self {
        self.export = true;
        self
    }
    pub fn encode(&self) -> [u32; 3] {
        let d0 = self.vector_dest
            | (self.vector_dest_rel as u32) << 6
            | (self.abs_constants as u32) << 7
            | self.scalar_dest << 8
            | (self.scalar_dest_rel as u32) << 14
            | (self.export as u32) << 15
            | self.vector_mask << 16
            | self.scalar_mask << 20
            | (self.vector_clamp as u32) << 24
            | (self.scalar_clamp as u32) << 25
            | self.scalar_op << 26;
        let d1 = self.src[2].swizzle
            | self.src[1].swizzle << 8
            | self.src[0].swizzle << 16
            | (self.src[2].negate as u32) << 24
            | (self.src[1].negate as u32) << 25
            | (self.src[0].negate as u32) << 26
            | ((self.predicated == Some(true)) as u32) << 27
            | (self.predicated.is_some() as u32) << 28
            | (self.const_a0 as u32) << 29
            | (self.const_1_rel as u32) << 30
            | (self.const_0_rel as u32) << 31;
        let d2 = self.src[2].encoded_reg()
            | self.src[1].encoded_reg() << 8
            | self.src[0].encoded_reg() << 16
            | self.vector_op << 24
            | (self.src[2].temp as u32) << 29
            | (self.src[1].temp as u32) << 30
            | (self.src[0].temp as u32) << 31;
        [d0, d1, d2]
    }
}

// ---- Fetches ----------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct VFetch {
    pub dest: u32,
    pub dest_swizzle: &'static str,
    pub src: u32,
    pub src_component: u32,
    pub fetch_constant: u32,
    pub format: u32,
    pub signed: bool,
    pub integer: bool,
    pub no_zero: bool,
    pub round_index: bool,
    pub exp_adjust: i32,
    pub mini: bool,
    pub stride: u32,
    pub offset: i32,
    pub predicated: Option<bool>,
    pub prefetch: u32,
}

impl Default for VFetch {
    fn default() -> Self {
        VFetch {
            dest: 1,
            dest_swizzle: "xyzw",
            src: 0,
            src_component: 0,
            fetch_constant: 95,
            format: 38,
            signed: false,
            integer: false,
            no_zero: false,
            round_index: false,
            exp_adjust: 0,
            mini: false,
            stride: 4,
            offset: 0,
            predicated: None,
            prefetch: 0,
        }
    }
}

impl VFetch {
    pub fn encode(&self) -> [u32; 3] {
        let d0 = self.src << 5
            | self.dest << 12
            | 1 << 19
            | (self.fetch_constant / 3) << 20
            | (self.fetch_constant % 3) << 25
            | self.prefetch << 27
            | self.src_component << 30;
        let d1 = fetch_dest_swizzle(self.dest_swizzle)
            | (self.signed as u32) << 12
            | (self.integer as u32) << 13
            | (self.no_zero as u32) << 14
            | (self.round_index as u32) << 15
            | self.format << 16
            | ((self.exp_adjust as u32) & 63) << 24
            | (self.mini as u32) << 30
            | (self.predicated.is_some() as u32) << 31;
        let d2 = self.stride | ((self.offset as u32) & 0x7F_FFFF) << 8 | ((self.predicated == Some(true)) as u32) << 31;
        [d0, d1, d2]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TFetch {
    pub opcode: u32,
    pub dest: u32,
    pub dest_swizzle: &'static str,
    pub src: u32,
    pub src_swizzle: &'static str,
    pub fetch_constant: u32,
    pub dimension: u32,
    pub unnormalized: bool,
    pub fetch_valid_only: bool,
    pub mag: u32,
    pub min: u32,
    pub mip: u32,
    pub aniso: u32,
    pub vol_mag: u32,
    pub vol_min: u32,
    pub computed_lod: bool,
    pub register_lod: bool,
    pub register_gradients: bool,
    pub lod_bias_16ths: i32,
    pub offset_halves: [i32; 3],
    pub predicated: Option<bool>,
}

impl Default for TFetch {
    fn default() -> Self {
        TFetch {
            opcode: 1,
            dest: 0,
            dest_swizzle: "xyzw",
            src: 0,
            src_swizzle: "xyz",
            fetch_constant: 0,
            dimension: 1,
            unnormalized: false,
            fetch_valid_only: true,
            mag: 3,
            min: 3,
            mip: 3,
            aniso: 7,
            vol_mag: 3,
            vol_min: 3,
            computed_lod: true,
            register_lod: false,
            register_gradients: false,
            lod_bias_16ths: 0,
            offset_halves: [0; 3],
            predicated: None,
        }
    }
}

impl TFetch {
    pub fn encode(&self) -> [u32; 3] {
        let d0 = self.opcode
            | self.src << 5
            | self.dest << 12
            | (self.fetch_valid_only as u32) << 19
            | self.fetch_constant << 20
            | (self.unnormalized as u32) << 25
            | fetch_src_swizzle(self.src_swizzle) << 26;
        let d1 = fetch_dest_swizzle(self.dest_swizzle)
            | self.mag << 12
            | self.min << 14
            | self.mip << 16
            | self.aniso << 18
            | 7 << 21
            | self.vol_mag << 24
            | self.vol_min << 26
            | (self.computed_lod as u32) << 28
            | (self.register_lod as u32) << 29
            | (self.predicated.is_some() as u32) << 31;
        let d2 = self.register_gradients as u32
            | ((self.lod_bias_16ths as u32) & 0x7F) << 2
            | self.dimension << 14
            | ((self.offset_halves[0] as u32) & 31) << 16
            | ((self.offset_halves[1] as u32) & 31) << 21
            | ((self.offset_halves[2] as u32) & 31) << 26
            | ((self.predicated == Some(true)) as u32) << 31;
        [d0, d1, d2]
    }
}

// ---- Control flow and program layout ------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Ins {
    Alu([u32; 3]),
    Fetch([u32; 3]),
    /// Fetch with the serialize bit set.
    SerializedFetch([u32; 3]),
}

impl From<Alu> for Ins {
    fn from(a: Alu) -> Ins {
        Ins::Alu(a.encode())
    }
}

impl From<VFetch> for Ins {
    fn from(f: VFetch) -> Ins {
        Ins::Fetch(f.encode())
    }
}

impl From<TFetch> for Ins {
    fn from(f: TFetch) -> Ins {
        Ins::Fetch(f.encode())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Always,
    Bool(u32, bool),
    Pred(bool),
    /// Bool condition with the PredicateClean variant of the opcode.
    BoolClean(u32, bool),
}

#[derive(Clone, Debug)]
pub enum Cf {
    Exec { cond: Cond, end: bool, instrs: Vec<Ins>, yield_: bool },
    Nop,
    LoopStart { loop_id: u32, skip: u32, repeat: bool },
    LoopEnd { loop_id: u32, body: u32, pred_break: Option<bool> },
    Call { target: u32, cond: Cond },
    Ret,
    Jmp { target: u32, cond: Cond },
    Alloc { kind: u32, size: u32 },
    MarkVsFetchDone,
}

pub fn exec(instrs: Vec<Ins>) -> Cf {
    Cf::Exec { cond: Cond::Always, end: false, instrs, yield_: false }
}

pub fn exece(instrs: Vec<Ins>) -> Cf {
    Cf::Exec { cond: Cond::Always, end: true, instrs, yield_: false }
}

pub fn cexec(cond: Cond, end: bool, instrs: Vec<Ins>) -> Cf {
    Cf::Exec { cond, end, instrs, yield_: false }
}

fn raw_cf(cf: &Cf, address: u32) -> RawCf {
    let op = |o: u32| o << 12;
    match cf {
        Cf::Exec { cond, end, instrs, yield_ } => {
            let sequence = instrs.iter().enumerate().fold(0u32, |acc, (i, ins)| {
                acc | match ins {
                    Ins::Alu(_) => 0,
                    Ins::Fetch(_) => 1,
                    Ins::SerializedFetch(_) => 3,
                } << (2 * i)
            });
            let dword0 = address | (instrs.len() as u32) << 12 | (*yield_ as u32) << 15 | sequence << 16;
            let dword1 = match cond {
                Cond::Always => op(if *end { 2 } else { 1 }) | 1 << 9,
                Cond::Bool(b, v) => op(if *end { 4 } else { 3 }) | b << 2 | (*v as u32) << 10,
                Cond::BoolClean(b, v) => op(if *end { 14 } else { 13 }) | b << 2 | (*v as u32) << 10,
                Cond::Pred(v) => op(if *end { 6 } else { 5 }) | 1 << 9 | (*v as u32) << 10,
            };
            RawCf { dword0, dword1 }
        }
        Cf::Nop => RawCf { dword0: 0, dword1: 0 },
        Cf::LoopStart { loop_id, skip, repeat } => {
            RawCf { dword0: skip | (*repeat as u32) << 13 | loop_id << 16, dword1: op(7) }
        }
        Cf::LoopEnd { loop_id, body, pred_break } => RawCf {
            dword0: body | loop_id << 16 | (pred_break.is_some() as u32) << 21,
            dword1: op(8) | ((*pred_break == Some(true)) as u32) << 10,
        },
        Cf::Call { target, cond } | Cf::Jmp { target, cond } => {
            let opcode = if matches!(cf, Cf::Call { .. }) { 9 } else { 11 };
            let (d0, d1) = match cond {
                Cond::Always => (1 << 13, 0),
                Cond::Pred(v) => (1 << 14, (*v as u32) << 10),
                Cond::Bool(b, v) | Cond::BoolClean(b, v) => (0, b << 2 | (*v as u32) << 10),
            };
            RawCf { dword0: target | d0, dword1: op(opcode) | d1 }
        }
        Cf::Ret => RawCf { dword0: 0, dword1: op(10) },
        Cf::Alloc { kind, size } => RawCf { dword0: *size, dword1: op(12) | kind << 9 },
        Cf::MarkVsFetchDone => RawCf { dword0: 0, dword1: op(15) },
    }
}

/// Lays out control flow pairs followed by the exec instructions.
pub fn assemble(cfs: &[Cf]) -> Vec<u32> {
    let pairs = cfs.len().div_ceil(2).max(1) as u32;
    let mut next_address = pairs;
    let mut raws = Vec::new();
    let mut instructions = Vec::new();
    for cf in cfs {
        let address = next_address;
        if let Cf::Exec { instrs, .. } = cf {
            assert!(instrs.len() <= 6);
            next_address += instrs.len() as u32;
            for ins in instrs {
                instructions.push(match ins {
                    Ins::Alu(d) | Ins::Fetch(d) | Ins::SerializedFetch(d) => *d,
                });
            }
        }
        raws.push(raw_cf(cf, address));
    }
    if raws.len() % 2 == 1 {
        raws.push(RawCf::default());
    }
    let mut out = Vec::new();
    for pair in raws.chunks(2) {
        out.extend_from_slice(&pack_cf_pair(pair[0], pair[1]));
    }
    for i in instructions {
        out.extend_from_slice(&i);
    }
    out
}

// ---- WGSL validation ----------------------------------------------------------------------------

pub fn numbered(source: &str) -> String {
    source.lines().enumerate().map(|(i, l)| format!("{:5} {l}\n", i + 1)).collect()
}

pub fn validate_wgsl(source: &str) -> naga::Module {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => panic!("WGSL parse error:\n{}\n{}", e.emit_to_string(source), numbered(source)),
    };
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    if let Err(e) = validator.validate(&module) {
        panic!("WGSL validation error:\n{}\n{}", e.emit_to_string(source), numbered(source));
    }
    module
}
