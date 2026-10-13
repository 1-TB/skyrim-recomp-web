//! A tiny big-endian PowerPC assembler with labels, enough to write the
//! synthetic test title.

use std::collections::HashMap;

pub const R0: u32 = 0;
pub const CR6: u32 = 6;
pub const R1: u32 = 1;
pub const R3: u32 = 3;
pub const R4: u32 = 4;
pub const R5: u32 = 5;
pub const R6: u32 = 6;
pub const R7: u32 = 7;
pub const R8: u32 = 8;
pub const R9: u32 = 9;
pub const R10: u32 = 10;
pub const R11: u32 = 11;
pub const R12: u32 = 12;
pub const R31: u32 = 31;
pub const V0: u32 = 0;
pub const V1: u32 = 1;
pub const V2: u32 = 2;
pub const V3: u32 = 3;

enum Fixup {
    /// I-form branch (b/bl) to a label.
    Branch { at: usize, label: String, link: bool },
    /// B-form conditional branch to a label.
    Cond { at: usize, label: String, bo: u32, bi: u32 },
    /// lis/addi pair loading a label's absolute address.
    Addr { at: usize, label: String, reg: u32 },
    /// A raw word holding a label's address.
    Word { at: usize, label: String },
}

pub struct Asm {
    base: u32,
    words: Vec<u32>,
    labels: HashMap<String, u32>,
    fixups: Vec<Fixup>,
}

fn d16(v: i32) -> u32 {
    (v as u32) & 0xFFFF
}

impl Asm {
    pub fn new(base: u32) -> Self {
        Asm { base, words: Vec::new(), labels: HashMap::new(), fixups: Vec::new() }
    }

    pub fn pc(&self) -> u32 {
        self.base + self.words.len() as u32 * 4
    }

    pub fn label(&mut self, name: &str) -> u32 {
        let pc = self.pc();
        assert!(self.labels.insert(name.to_string(), pc).is_none(), "duplicate label {name}");
        pc
    }

    /// Defines a label at an address outside this code block (data, thunks).
    pub fn extern_label(&mut self, name: &str, addr: u32) {
        self.labels.insert(name.to_string(), addr);
    }

    pub fn word(&mut self, w: u32) {
        self.words.push(w);
    }

    pub fn addi(&mut self, rt: u32, ra: u32, si: i32) {
        self.word((14 << 26) | (rt << 21) | (ra << 16) | d16(si));
    }

    pub fn li(&mut self, rt: u32, si: i32) {
        self.addi(rt, 0, si);
    }

    pub fn lis(&mut self, rt: u32, si: i32) {
        self.word((15 << 26) | (rt << 21) | d16(si));
    }

    /// Loads a 32-bit constant.
    pub fn li32(&mut self, rt: u32, v: u32) {
        self.lis(rt, (v >> 16) as i32);
        self.word((24 << 26) | (rt << 21) | (rt << 16) | (v & 0xFFFF)); // ori
    }

    /// Loads a label's address (lis + addi, high half adjusted for sign).
    pub fn la(&mut self, rt: u32, label: &str) {
        self.fixups.push(Fixup::Addr { at: self.words.len(), label: label.to_string(), reg: rt });
        self.word(0);
        self.word(0);
    }

    pub fn lwz(&mut self, rt: u32, d: i32, ra: u32) {
        self.word((32 << 26) | (rt << 21) | (ra << 16) | d16(d));
    }

    pub fn stw(&mut self, rs: u32, d: i32, ra: u32) {
        self.word((36 << 26) | (rs << 21) | (ra << 16) | d16(d));
    }

    pub fn stwu(&mut self, rs: u32, d: i32, ra: u32) {
        self.word((37 << 26) | (rs << 21) | (ra << 16) | d16(d));
    }

    pub fn std(&mut self, rs: u32, ds: i32, ra: u32) {
        assert!(ds % 4 == 0);
        self.word((62 << 26) | (rs << 21) | (ra << 16) | (d16(ds) & 0xFFFC));
    }

    pub fn ld(&mut self, rt: u32, ds: i32, ra: u32) {
        assert!(ds % 4 == 0);
        self.word((58 << 26) | (rt << 21) | (ra << 16) | (d16(ds) & 0xFFFC));
    }

    /// cmplwi crN, rA, UIMM
    pub fn cmplwi(&mut self, cr: u32, ra: u32, ui: u32) {
        self.word((10 << 26) | (cr << 23) | (ra << 16) | (ui & 0xFFFF));
    }

    /// slwi rA, rS, n (rlwinm rA, rS, n, 0, 31-n)
    pub fn slwi(&mut self, ra: u32, rs: u32, n: u32) {
        self.word((21 << 26) | (rs << 21) | (ra << 16) | (n << 11) | (31 - n) << 1);
    }

    pub fn lwzx(&mut self, rt: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (23 << 1));
    }

    pub fn lbzx(&mut self, rt: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (87 << 1));
    }

    pub fn add(&mut self, rt: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (266 << 1));
    }

    pub fn bctr(&mut self) {
        self.word(0x4E80_0420);
    }

    /// Branch if crN.gt is set.
    pub fn bgt(&mut self, cr: u32, label: &str) {
        self.fixups.push(Fixup::Cond { at: self.words.len(), label: label.to_string(), bo: 12, bi: cr * 4 + 1 });
        self.word(0);
    }

    /// A data word holding a label's address (inline jump tables).
    pub fn addr_word(&mut self, label: &str) {
        self.fixups.push(Fixup::Word { at: self.words.len(), label: label.to_string() });
        self.word(0);
    }

    pub fn mr(&mut self, ra: u32, rs: u32) {
        // or ra, rs, rs
        self.word((31 << 26) | (rs << 21) | (ra << 16) | (rs << 11) | (444 << 1));
    }

    pub fn mflr(&mut self, rt: u32) {
        self.word((31 << 26) | (rt << 21) | (8 << 16) | (339 << 1));
    }

    pub fn mtlr(&mut self, rs: u32) {
        self.word((31 << 26) | (rs << 21) | (8 << 16) | (467 << 1));
    }

    pub fn mtctr(&mut self, rs: u32) {
        self.word((31 << 26) | (rs << 21) | (9 << 16) | (467 << 1));
    }

    pub fn bctrl(&mut self) {
        self.word(0x4E80_0421);
    }

    pub fn blr(&mut self) {
        self.word(0x4E80_0020);
    }

    pub fn stwx(&mut self, rs: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (151 << 1));
    }

    pub fn lvx(&mut self, vt: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (vt << 21) | (ra << 16) | (rb << 11) | (103 << 1));
    }

    pub fn stvx(&mut self, vs: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (vs << 21) | (ra << 16) | (rb << 11) | (231 << 1));
    }

    /// vupkd3d128 vD, vB, imm (VX128_3; vD and vB below v32).
    pub fn vupkd3d128(&mut self, vd: u32, vb: u32, imm: u32) {
        self.word((6 << 26) | (vd << 21) | (imm << 16) | (vb << 11) | 0x7F0);
    }

    /// vpkd3d128 vD, vB, type, pack, shift (VX128_4; vD and vB below v32).
    pub fn vpkd3d128(&mut self, vd: u32, vb: u32, ty: u32, pack: u32, shift: u32) {
        self.word((6 << 26) | (vd << 21) | (ty << 18) | (pack << 16) | (vb << 11) | (shift << 6) | 0x610);
    }

    pub fn blrl(&mut self) {
        self.word(0x4E80_0021);
    }

    pub fn dcbz(&mut self, ra: u32, rb: u32) {
        self.word((31 << 26) | (ra << 16) | (rb << 11) | (1014 << 1));
    }

    pub fn lwarx(&mut self, rt: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (20 << 1));
    }

    pub fn stwcx(&mut self, rs: u32, ra: u32, rb: u32) {
        self.word((31 << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (150 << 1) | 1);
    }

    pub fn bl(&mut self, label: &str) {
        self.fixups.push(Fixup::Branch { at: self.words.len(), label: label.to_string(), link: true });
        self.word(0);
    }

    pub fn b(&mut self, label: &str) {
        self.fixups.push(Fixup::Branch { at: self.words.len(), label: label.to_string(), link: false });
        self.word(0);
    }

    /// Branch if cr0.eq is clear.
    pub fn bne(&mut self, label: &str) {
        self.fixups.push(Fixup::Cond { at: self.words.len(), label: label.to_string(), bo: 4, bi: 2 });
        self.word(0);
    }

    /// Standard prologue: save LR below the frame, allocate `frame` bytes.
    pub fn prologue(&mut self, frame: i32) {
        self.mflr(R12);
        self.stw(R12, -8, R1);
        self.stwu(R1, -frame, R1);
    }

    pub fn epilogue(&mut self, frame: i32) {
        self.addi(R1, R1, frame);
        self.lwz(R12, -8, R1);
        self.mtlr(R12);
        self.blr();
    }

    pub fn finish(mut self) -> (Vec<u8>, HashMap<String, u32>) {
        for f in std::mem::take(&mut self.fixups) {
            match f {
                Fixup::Branch { at, label, link } => {
                    let target = self.labels[&label];
                    let pc = self.base + at as u32 * 4;
                    let off = target.wrapping_sub(pc);
                    self.words[at] = (18 << 26) | (off & 0x03FF_FFFC) | link as u32;
                }
                Fixup::Cond { at, label, bo, bi } => {
                    let target = self.labels[&label];
                    let pc = self.base + at as u32 * 4;
                    let off = target.wrapping_sub(pc);
                    self.words[at] = (16 << 26) | (bo << 21) | (bi << 16) | (off & 0xFFFC);
                }
                Fixup::Word { at, label } => self.words[at] = self.labels[&label],
                Fixup::Addr { at, label, reg } => {
                    let addr = self.labels[&label];
                    let lo = (addr & 0xFFFF) as i16 as i32;
                    let hi = (addr.wrapping_sub(lo as u32) >> 16) as i32;
                    self.words[at] = (15 << 26) | (reg << 21) | d16(hi);
                    self.words[at + 1] = (14 << 26) | (reg << 21) | (reg << 16) | d16(lo);
                }
            }
        }
        let bytes = self.words.iter().flat_map(|w| w.to_be_bytes()).collect();
        (bytes, self.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_encodings() {
        let mut a = Asm::new(0x8200_0000);
        a.mflr(R12);
        a.mtlr(R12);
        a.mtctr(R12);
        a.stwu(R1, -0x80, R1);
        a.li(R3, 7);
        let (bytes, _) = a.finish();
        let words: Vec<u32> = bytes.chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect();
        assert_eq!(words, [0x7D88_02A6, 0x7D88_03A6, 0x7D89_03A6, 0x9421_FF80, 0x3860_0007]);
    }

    #[test]
    fn save_restore_patterns_match_xenon_recomp_readme() {
        let mut a = Asm::new(0x8200_0000);
        a.ld(14, -0x98, R1);
        a.std(14, -0x98, R1);
        let (bytes, _) = a.finish();
        assert_eq!(bytes, [0xE9, 0xC1, 0xFF, 0x68, 0xF9, 0xC1, 0xFF, 0x68]);
    }

    #[test]
    fn branches_and_addresses() {
        let mut a = Asm::new(0x8200_1000);
        a.label("top");
        a.bl("top");
        a.extern_label("data", 0x8201_8004);
        a.la(R3, "data");
        a.bne("top");
        let (bytes, _) = a.finish();
        let w = |i: usize| u32::from_be_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(w(0), 0x4800_0001);
        assert_eq!(w(1), 0x3C60_8202); // lis r3, 0x8202 (0x8004 sign-extends negative)
        assert_eq!(w(2), 0x3863_8004); // addi r3, r3, -0x7FFC
        assert_eq!(w(3), 0x4082_FFF4);
    }
}
