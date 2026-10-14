//! How far down the render target a draw without clipping reaches, found by running its vertex
//! shader on the CPU (port of Xenia's DrawExtentEstimator::EstimateVertexMaxY).
//!
//! Screen-space draws (Direct3D 9's clears and full-screen passes) disable clipping and leave the
//! scissor at 8192x8192, so neither bounds the rows they touch; render targets would have to
//! cover all of EDRAM and every such draw would claim (and transfer) every tile. These draws have
//! a handful of vertices, so interpreting the shader is cheap.

use xenos_shader::interpreter::{Environment, Interpreter, SharedMemoryWords};
use xenos_shader::{BoolLoopConstants, Shader};
use xenos_state::bitfields::{PaSuPointMinmax, PaSuPointSize, RegisterField};
use xenos_state::primitive::DrawDescription;
use xenos_state::viewport::ViewportRegisters;
use xenos_state::xenos::{
    IndexFormat, MsaaSamples, PixelCenter, PrimitiveType, SourceSelect, TEXTURE_2D_MAX_WIDTH_HEIGHT, gpu_swap_u16,
    gpu_swap_u32,
};

use crate::PhysicalMemory;
use crate::registers::{FLOAT_CONSTANTS, RegisterFile};

/// Guest physical memory read a dword at a time.
struct Words<'a>(&'a dyn PhysicalMemory);

impl SharedMemoryWords for Words<'_> {
    fn word(&self, index: u32) -> u32 {
        if index >= (crate::shared_memory::PHYSICAL_SIZE / 4) as u32 {
            return 0;
        }
        let mut bytes = [0; 4];
        self.0.read(index * 4, &mut bytes);
        u32::from_le_bytes(bytes)
    }
}

/// D3D11 float -> 16.8 fixed point conversion (Xenia's ui::FloatToD3D11Fixed16p8).
fn float_to_fixed_16p8(f: f32) -> i32 {
    if !(f.abs() >= 1.0 / 512.0) {
        return 0;
    }
    if f >= 32768.0 - 1.0 / 256.0 {
        return (1 << 23) - 1;
    }
    if f <= -32768.0 {
        return -32768 * 256;
    }
    let bits = f.to_bits();
    let mut result = ((bits & ((1 << 23) - 1)) | (1 << 23)) as i32;
    if bits >> 31 != 0 {
        result = -result;
    }
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    let extra_bits = (15 - exponent) as u32;
    if extra_bits != 0 {
        let mut u = result as u32;
        u = u.wrapping_add((1 << (extra_bits - 1)) - 1 + ((u >> extra_bits) & 1));
        result = (u as i32) >> extra_bits;
    }
    result
}

/// One past the lowest render target row the draw's vertices can cover, or
/// [`TEXTURE_2D_MAX_WIDTH_HEIGHT`] when that can't be determined on the CPU.
pub fn vertex_max_y(
    desc: &DrawDescription,
    regs: &RegisterFile,
    viewport: &ViewportRegisters,
    shader: &Shader,
    memory: &dyn PhysicalMemory,
) -> u32 {
    let num_indices = desc.initiator.num_indices;
    if num_indices == 0 {
        return 0;
    }
    let info = shader.info();
    // Texture fetches aren't interpreted (Xenia's CanInterpretShader), and tessellation and
    // immediate indices aren't reproduced.
    if desc.tessellation.is_some()
        || !matches!(desc.source_select, SourceSelect::Dma | SourceSelect::AutoIndex)
        || info.uses_texture_fetch_results
        || !info.writes_position
    {
        return TEXTURE_2D_MAX_WIDTH_HEIGHT;
    }

    // Guest indices, already endian-swapped and masked to 24 bits.
    let indices: Option<Vec<u32>> = desc.index_buffer.map(|ib| {
        let size = ib.format.size_bytes();
        let count = num_indices.min(ib.capacity);
        let mut bytes = vec![0u8; (count * size) as usize];
        memory.read(ib.guest_base, &mut bytes);
        bytes
            .chunks_exact(size as usize)
            .map(|c| match ib.format {
                IndexFormat::Int16 => u32::from(gpu_swap_u16(u16::from_le_bytes([c[0], c[1]]), ib.endian)),
                IndexFormat::Int32 => gpu_swap_u32(u32::from_le_bytes([c[0], c[1], c[2], c[3]]), ib.endian),
            } & 0xFF_FFFF)
            .collect()
    });

    let floats: Vec<[f32; 4]> = regs.as_slice()[FLOAT_CONSTANTS..FLOAT_CONSTANTS + 256 * 4]
        .chunks_exact(4)
        .map(|c| [0, 1, 2, 3].map(|i| f32::from_bits(c[i])))
        .collect();
    let raw_bool_loop = regs.bool_loop_constants();
    let mut bool_loop = BoolLoopConstants::default();
    bool_loop.bools.copy_from_slice(&raw_bool_loop[..8]);
    bool_loop.loops.copy_from_slice(&raw_bool_loop[8..40]);
    let words = Words(memory);

    let vte = viewport.vte_cntl;
    let y_scale = if vte.vport_y_scale_ena { viewport.scale[1] } else { 1.0 };
    let y_offset = if vte.vport_y_offset_ena { viewport.offset[1] } else { 0.0 };
    let points = desc.primitive_type == PrimitiveType::PointList;
    let (point_min_diameter, point_max_diameter, point_constant_radius) = if points {
        let minmax = PaSuPointMinmax::read(regs);
        (
            (minmax.min_size as f32 * (2.0 / 16.0)).to_bits() as i32,
            (minmax.max_size as f32 * (2.0 / 16.0)).to_bits() as i32,
            PaSuPointSize::read(regs).height as f32 * (1.0 / 16.0),
        )
    } else {
        (0, 0, 0.0)
    };
    let writes_point_size = info.writes_point_size_edge_flag_kill_vertex & 1 != 0;
    let writes_kill = info.writes_point_size_edge_flag_kill_vertex & 4 != 0;

    let params = desc.index_params;
    let mut max_y = f32::MIN;
    for i in 0..num_indices {
        let index = match &indices {
            Some(indices) => {
                let index = indices.get(i as usize).copied().unwrap_or(0);
                if desc.reset_index == Some(index) {
                    continue;
                }
                index
            }
            None => i,
        };
        let index = (index.wrapping_add(params.offset) & 0xFF_FFFF).max(params.min).min(params.max);
        let env = Environment {
            float_constants: &floats,
            bool_loop: &bool_loop,
            fetch_constants: regs.fetch_constants(),
            shared_memory: &words,
        };
        let mut interpreter = Interpreter::new(shader.program(), env);
        interpreter.state.registers[0][0] = index as f32;
        interpreter.run();
        let exports = &interpreter.exports;
        if writes_kill && exports.point_size_edge_flag_kill_vertex[2].to_bits() & !(1 << 31) != 0 {
            continue;
        }
        let mut y = exports.position[1];
        if !vte.vtx_xy_fmt {
            y /= exports.position[3];
        }
        y = y * y_scale + y_offset;
        if points {
            let radius = if writes_point_size {
                let diameter = exports.point_size_edge_flag_kill_vertex[0].to_bits() as i32;
                0.5 * f32::from_bits(diameter.max(point_min_diameter).min(point_max_diameter) as u32)
            } else {
                point_constant_radius
            };
            y += radius;
        }
        // `max` that keeps the first argument for NaN, like Xenia's std::max.
        if max_y < y {
            max_y = y;
        }
    }

    let mut max_y_24p8 = float_to_fixed_16p8(max_y);
    if viewport.vtx_cntl.pix_center == PixelCenter::D3DZero {
        max_y_24p8 += 128;
    }
    if viewport.mode_cntl.vtx_window_offset_enable {
        max_y_24p8 += viewport.window_offset.window_y_offset * 256;
    }
    // Top-left rule: .5 exclusive without MSAA, 1. exclusive with it.
    let round = if viewport.msaa_samples == MsaaSamples::X1 { 127 } else { 255 };
    (max_y_24p8.max(0) as u32 + round) >> 8
}

#[cfg(test)]
mod tests {
    use super::float_to_fixed_16p8;

    #[test]
    fn fixed_16p8_matches_d3d11_rules() {
        assert_eq!(float_to_fixed_16p8(0.0), 0);
        assert_eq!(float_to_fixed_16p8(f32::NAN), 0);
        assert_eq!(float_to_fixed_16p8(1.0), 256);
        assert_eq!(float_to_fixed_16p8(720.5), 720 * 256 + 128);
        assert_eq!(float_to_fixed_16p8(-2.0), -512);
        assert_eq!(float_to_fixed_16p8(1e9), (1 << 23) - 1);
        assert_eq!(float_to_fixed_16p8(-1e9), -32768 * 256);
    }
}
