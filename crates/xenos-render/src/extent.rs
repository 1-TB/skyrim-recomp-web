//! How far down the render target a draw without clipping reaches, found by running its vertex
//! shader on the CPU (port of Xenia's DrawExtentEstimator::EstimateVertexMaxY).
//!
//! Screen-space draws (Direct3D 9's clears and full-screen passes) disable clipping and leave the
//! scissor at 8192x8192, so neither bounds the rows they touch; render targets would have to
//! cover all of EDRAM and every such draw would claim (and transfer) every tile. These draws have
//! a handful of vertices, so interpreting the shader is cheap.

use xenos_shader::interpreter::{Environment, Exports, Interpreter, SharedMemoryWords};
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

/// Runs the draw's vertex shader on the CPU for each vertex in draw order, passing `f` its
/// exports. `false` (with `f` not called for
/// every vertex) when the draw can't be interpreted.
fn for_each_vertex(
    desc: &DrawDescription,
    regs: &RegisterFile,
    shader: &Shader,
    memory: &dyn PhysicalMemory,
    mut f: impl FnMut(&Exports),
) -> bool {
    let num_indices = desc.initiator.num_indices;
    let info = shader.info();
    // Texture fetches aren't interpreted (Xenia's CanInterpretShader), and tessellation and
    // immediate indices aren't reproduced.
    if desc.tessellation.is_some()
        || !matches!(desc.source_select, SourceSelect::Dma | SourceSelect::AutoIndex)
        || info.uses_texture_fetch_results
        || !info.writes_position
    {
        return false;
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

    let params = desc.index_params;
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
        f(&interpreter.exports);
    }
    true
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
    if desc.initiator.num_indices == 0 {
        return 0;
    }
    let info = shader.info();
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

    let mut max_y = f32::MIN;
    let interpreted = for_each_vertex(desc, regs, shader, memory, |exports| {
        let (position, pts) = (exports.position, exports.point_size_edge_flag_kill_vertex);
        if writes_kill && pts[2].to_bits() & !(1 << 31) != 0 {
            return;
        }
        let mut y = position[1];
        if !vte.vtx_xy_fmt {
            y /= position[3];
        }
        y = y * y_scale + y_offset;
        if points {
            let radius = if writes_point_size {
                let diameter = pts[0].to_bits() as i32;
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
    });
    if !interpreted {
        return TEXTURE_2D_MAX_WIDTH_HEIGHT;
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

/// A draw that fills an axis-aligned rectangle with one depth value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlatRect {
    /// Pixels covered: `[x0, x1) x [y0, y1)` (before the scissor).
    pub pixels: [i64; 4],
}

/// If the draw is one screen-space rectangle (a rectangle list of one rectangle, or a strip, fan,
/// quad or two-triangle list over its four corners) whose vertices all have the same depth and
/// the same values of the interpolators in the `interpolators` mask, the pixels it covers.
/// Direct3D 9 clears are drawn like this.
pub fn flat_rect(
    desc: &DrawDescription,
    regs: &RegisterFile,
    viewport: &ViewportRegisters,
    shader: &Shader,
    memory: &dyn PhysicalMemory,
    interpolators: u32,
) -> Option<FlatRect> {
    let corners = match desc.primitive_type {
        PrimitiveType::RectangleList => 3,
        PrimitiveType::TriangleStrip | PrimitiveType::TriangleFan | PrimitiveType::QuadList => 4,
        PrimitiveType::TriangleList => 6,
        _ => return None,
    };
    if desc.initiator.num_indices != corners || shader.info().writes_point_size_edge_flag_kill_vertex & 4 != 0 {
        return None;
    }
    let vte = viewport.vte_cntl;
    let scale = [
        if vte.vport_x_scale_ena { viewport.scale[0] } else { 1.0 },
        if vte.vport_y_scale_ena { viewport.scale[1] } else { 1.0 },
    ];
    let offset = [
        if vte.vport_x_offset_ena { viewport.offset[0] } else { 0.0 },
        if vte.vport_y_offset_ena { viewport.offset[1] } else { 0.0 },
    ];
    let mut points: Vec<[f32; 2]> = Vec::with_capacity(6);
    // Bits of z, w and the interpolators that must not vary.
    let mut first: Option<Vec<u32>> = None;
    let mut flat = true;
    let interpreted = for_each_vertex(desc, regs, shader, memory, |exports| {
        let p = exports.position;
        let w = if vte.vtx_xy_fmt { 1.0 } else { p[3] };
        points.push([0, 1].map(|i| p[i] / w * scale[i] + offset[i]));
        let mut values = vec![p[2].to_bits(), p[3].to_bits()];
        for (i, v) in exports.interpolators.iter().enumerate() {
            if interpolators & (1 << i) != 0 {
                values.extend(v.map(f32::to_bits));
            }
        }
        flat &= *first.get_or_insert_with(|| values.clone()) == values;
    });
    if !interpreted || !flat || points.len() != corners as usize || points.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = points.iter().map(|p| p[1]).collect();
    let (x0, x1) = (xs.iter().copied().fold(f32::MAX, f32::min), xs.iter().copied().fold(f32::MIN, f32::max));
    let (y0, y1) = (ys.iter().copied().fold(f32::MAX, f32::min), ys.iter().copied().fold(f32::MIN, f32::max));
    // Every vertex on a corner, and (for triangles) all four corners used.
    if points.iter().any(|p| (p[0] != x0 && p[0] != x1) || (p[1] != y0 && p[1] != y1)) {
        return None;
    }
    if corners != 3 {
        let has = |x: f32, y: f32| points.iter().any(|p| p[0] == x && p[1] == y);
        if !(has(x0, y0) && has(x0, y1) && has(x1, y0) && has(x1, y1)) {
            return None;
        }
    }
    // Pixel centers at .5 (Direct3D 9's zero center puts them on the integers), covered when
    // inside the rectangle by the top-left rule.
    let center = if viewport.vtx_cntl.pix_center == PixelCenter::D3DZero { 0.0 } else { 0.5 };
    let (mut wx, mut wy) = (0i64, 0i64);
    if viewport.mode_cntl.vtx_window_offset_enable {
        wx = i64::from(viewport.window_offset.window_x_offset);
        wy = i64::from(viewport.window_offset.window_y_offset);
    }
    let edge = |v: f32| (f64::from(v) - center).ceil() as i64;
    Some(FlatRect { pixels: [edge(x0) + wx, edge(y0) + wy, edge(x1) + wx, edge(y1) + wy] })
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
