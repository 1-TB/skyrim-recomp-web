//! The renderer's copy of the Xenos register file, updated from the command stream.

use std::sync::Arc;

use xenos_shader::ShaderStage;
use xenos_state::RegisterSource;

/// Registers 0x0000..0x5002 (the MMIO window plus the constant registers).
pub const REGISTER_COUNT: usize = 0x5003;

/// First float constant register (SHADER_CONSTANT_000_X); 512 vec4 constants follow.
pub const FLOAT_CONSTANTS: usize = 0x4000;
/// First fetch constant register (SHADER_CONSTANT_FETCH_00_0); 192 dwords follow.
pub const FETCH_CONSTANTS: usize = 0x4800;
/// First bool constant register (SHADER_CONSTANT_BOOL_000_031); 8 bool + 32 loop dwords follow.
pub const BOOL_LOOP_CONSTANTS: usize = 0x4900;

/// [`RegisterFile::take_dirty`] bits: constant ranges written since the last call.
pub mod dirty {
    /// Float constants c0..c255 (the vertex shader's).
    pub const FLOAT_VS: u8 = 1;
    pub const FETCH: u8 = 2;
    pub const BOOL_LOOP: u8 = 4;
    /// Float constants c256..c511 (the pixel shader's).
    pub const FLOAT_PS: u8 = 8;
}

/// First register of the pixel shader's float constants (c256).
const PIXEL_FLOAT_CONSTANTS: usize = FLOAT_CONSTANTS + 256 * 4;

pub struct RegisterFile {
    values: Box<[u32]>,
    dirty: u8,
    /// Per float constant vec4 (512): written since the renderer last compared it with what it
    /// uploaded (see [`RegisterFile::float_written`]).
    float_written: [u64; 8],
}

impl Default for RegisterFile {
    fn default() -> Self {
        RegisterFile { values: vec![0; REGISTER_COUNT].into_boxed_slice(), dirty: !0, float_written: [!0; 8] }
    }
}

impl RegisterFile {
    pub fn write(&mut self, index: u32, value: u32) {
        let i = index as usize;
        if let Some(r) = self.values.get_mut(i) {
            *r = value;
            if (FLOAT_CONSTANTS..FETCH_CONSTANTS).contains(&i) {
                let v = (i - FLOAT_CONSTANTS) / 4;
                self.float_written[v / 64] |= 1 << (v % 64);
            }
            self.dirty |= match i {
                FLOAT_CONSTANTS..PIXEL_FLOAT_CONSTANTS => dirty::FLOAT_VS,
                PIXEL_FLOAT_CONSTANTS..FETCH_CONSTANTS => dirty::FLOAT_PS,
                FETCH_CONSTANTS..BOOL_LOOP_CONSTANTS => dirty::FETCH,
                BOOL_LOOP_CONSTANTS.. => dirty::BOOL_LOOP,
                _ => 0,
            };
        }
    }

    /// Returns and clears the [`dirty`] bits.
    pub fn take_dirty(&mut self) -> u8 {
        std::mem::take(&mut self.dirty)
    }

    /// Marks constant ranges as changed (e.g. their uploaded copies were discarded).
    pub fn mark_dirty(&mut self, bits: u8) {
        self.dirty |= bits;
        if bits & (dirty::FLOAT_VS | dirty::FLOAT_PS) != 0 {
            self.float_written = [!0; 8];
        }
    }

    /// Bits of the 256 float constant vec4s of stage `stage` (0 vertex, 1 pixel) written since
    /// [`RegisterFile::clear_float_written`] cleared them.
    pub fn float_written(&self, stage: usize) -> [u64; 4] {
        self.float_written[stage * 4..stage * 4 + 4].try_into().expect("4 words")
    }

    /// Clears the written bits of stage `stage`'s vec4s `[0, count)`.
    pub fn clear_float_written(&mut self, stage: usize, count: usize) {
        for (w, word) in self.float_written[stage * 4..stage * 4 + 4].iter_mut().enumerate() {
            let first = w * 64;
            if count >= first + 64 {
                *word = 0;
            } else if count > first {
                *word &= !0u64 << (count - first);
            }
        }
    }

    /// Float constant vec4 `index` (0..512) as its little-endian bytes.
    pub fn float_vec4(&self, index: usize) -> [u8; 16] {
        let c = FLOAT_CONSTANTS + index * 4;
        let mut out = [0u8; 16];
        for (k, w) in self.values[c..c + 4].iter().enumerate() {
            out[k * 4..k * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.values
    }

    /// `count` vec4 float constants starting at vec4 `base` (wrapping within the 512).
    pub fn float_constants_window(&self, base: u32, count: u32, out: &mut Vec<u8>) {
        out.reserve(count as usize * 16);
        let mut i = 0usize;
        while i < count as usize {
            // Contiguous run up to the wrap at 512.
            let first = (base as usize + i) & 511;
            let n = (count as usize - i).min(512 - first);
            let c = FLOAT_CONSTANTS + first * 4;
            let words = &self.values[c..c + n * 4];
            if cfg!(target_endian = "little") {
                // SAFETY: u32s viewed as their (little-endian) bytes.
                out.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4)
                });
            } else {
                words.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
            }
            i += n;
        }
    }

    pub fn fetch_constants(&self) -> &[u32; 192] {
        self.values[FETCH_CONSTANTS..FETCH_CONSTANTS + 192].try_into().expect("192 fetch constant dwords")
    }

    pub fn bool_loop_constants(&self) -> &[u32] {
        &self.values[BOOL_LOOP_CONSTANTS..BOOL_LOOP_CONSTANTS + 40]
    }

    pub fn set_float_constant_dirty(&mut self) {
        self.mark_dirty(dirty::FLOAT_VS | dirty::FLOAT_PS);
    }
}

impl RegisterSource for RegisterFile {
    fn reg(&self, index: u32) -> u32 {
        self.values.get(index as usize).copied().unwrap_or(0)
    }
}

/// The microcode most recently loaded for each stage (IM_LOAD / IM_LOAD_IMMEDIATE).
#[derive(Default, Clone)]
pub struct ActiveShaders {
    /// Microcode and its hash (computed once per load, not per draw).
    pub vertex: Option<(Arc<[u32]>, u64)>,
    pub pixel: Option<(Arc<[u32]>, u64)>,
}

impl ActiveShaders {
    pub fn set(&mut self, stage: ShaderStage, code: Arc<[u32]>) {
        let hash = crate::shaders::code_hash(&code);
        match stage {
            ShaderStage::Vertex => self.vertex = Some((code, hash)),
            ShaderStage::Pixel => self.pixel = Some((code, hash)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_writes_are_tracked_per_vec4_and_stage() {
        let mut regs = RegisterFile::default();
        regs.clear_float_written(0, 256);
        regs.clear_float_written(1, 256);
        assert_eq!(regs.float_written(0), [0; 4]);
        // c5.y (vertex) and c256 + 70 (pixel c70).
        regs.write(FLOAT_CONSTANTS as u32 + 5 * 4 + 1, 7);
        regs.write(FLOAT_CONSTANTS as u32 + (256 + 70) * 4, 9);
        assert_eq!(regs.float_written(0), [1 << 5, 0, 0, 0]);
        assert_eq!(regs.float_written(1), [0, 1 << 6, 0, 0]);
        assert_eq!(regs.float_vec4(5)[4..8], 7u32.to_le_bytes());
        // Clearing the first `count` vec4s keeps the rest.
        regs.clear_float_written(1, 70);
        assert_eq!(regs.float_written(1), [0, 1 << 6, 0, 0]);
        regs.clear_float_written(1, 71);
        assert_eq!(regs.float_written(1), [0; 4]);
        regs.mark_dirty(dirty::FLOAT_PS);
        assert_eq!(regs.float_written(0), [!0; 4]);
    }
}
