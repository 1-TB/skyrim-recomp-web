//! Primitive processing: the draw initiator, guest index buffers, and conversion of guest
//! topologies to host-supported ones (point/line/triangle lists and strips with restart), following
//! Xenia's primitive_processor.cc with WebGPU's constraints in mind (no fans, loops or quads,
//! restart always enabled for indexed strips, 0xFFFF / 0xFFFFFFFF restart values).

use crate::bitfields::*;
use crate::registers as r;
use crate::xenos::*;
use crate::{GuestMemory, RegisterSource};

/// Guest index buffer referenced by a DMA-sourced draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexBufferInfo {
    /// Physical address, aligned down to the index size.
    pub guest_base: u32,
    pub format: IndexFormat,
    /// Endian swap from VGT_DMA_SIZE, normalized for 16-bit indices (`Swap8In32` becomes `Swap8In16`,
    /// `Swap16In32` becomes `None`, like Xenia).
    pub endian: Endian,
    /// Endian swap as written in VGT_DMA_SIZE.
    pub raw_endian: Endian,
    /// Capacity of the buffer in indices (VGT_DMA_SIZE::NUM_WORDS).
    pub capacity: u32,
    /// Number of indices the draw reads (num_indices clamped to the capacity).
    pub count: u32,
}

impl IndexBufferInfo {
    /// Number of bytes the draw reads.
    pub fn size_bytes(&self) -> u32 {
        self.count * self.format.size_bytes()
    }
}

/// Vertex index post-processing applied after the primitive reset check:
/// `index = clamp((index + offset) & 0xFFFFFF, min, max)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexIndexParams {
    /// VGT_INDX_OFFSET (24-bit, wraps).
    pub offset: u32,
    /// VGT_MIN_VTX_INDX.
    pub min: u32,
    /// VGT_MAX_VTX_INDX.
    pub max: u32,
}

impl Default for VertexIndexParams {
    fn default() -> Self {
        Self { offset: 0, min: 0, max: VERTEX_INDEX_MASK }
    }
}

impl VertexIndexParams {
    /// Reads VGT_INDX_OFFSET / VGT_MIN_VTX_INDX / VGT_MAX_VTX_INDX.
    pub fn from_registers(regs: &(impl RegisterSource + ?Sized)) -> Self {
        Self {
            offset: VgtIndxOffset::read(regs).indx_offset,
            min: VgtMinVtxIndx::read(regs).min_indx,
            max: VgtMaxVtxIndx::read(regs).max_indx,
        }
    }

    /// Applies the offset, 24-bit wrap and clamp to a (24-bit) guest index.
    pub fn apply(&self, index: u32) -> u32 {
        (index.wrapping_add(self.offset) & VERTEX_INDEX_MASK).max(self.min).min(self.max)
    }

    /// Whether `apply` is the identity for all 24-bit indices.
    pub fn is_identity(&self) -> bool {
        self.offset == 0 && self.min == 0 && self.max >= VERTEX_INDEX_MASK
    }
}

/// Everything about the current draw call that the primitive processor needs, decoded from the
/// VGT registers (VGT_DRAW_INITIATOR, VGT_DMA_BASE/SIZE, reset index, index offset/clamp).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawDescription {
    pub initiator: VgtDrawInitiator,
    pub primitive_type: PrimitiveType,
    pub source_select: SourceSelect,
    /// Index (vertex) count, clamped to the index buffer capacity for DMA draws.
    pub index_count: u32,
    /// Present for `SourceSelect::Dma`.
    pub index_buffer: Option<IndexBufferInfo>,
    /// Primitive reset index (24-bit), if reset is enabled and applies to the topology and index size.
    pub reset_index: Option<u32>,
    pub index_params: VertexIndexParams,
    /// Tessellation mode, if the explicit major mode routes the draw through the tessellator.
    pub tessellation: Option<TessellationMode>,
    /// PA_SU_SC_MODE_CNTL::PROVOKING_VTX_LAST.
    pub provoking_vertex_last: bool,
}

impl DrawDescription {
    /// Decodes the draw from the registers. The command processor must have stored the packet's
    /// VGT_DRAW_INITIATOR, VGT_DMA_BASE and VGT_DMA_SIZE in the register file (as Xenia does).
    pub fn from_registers(regs: &(impl RegisterSource + ?Sized)) -> Self {
        let initiator = VgtDrawInitiator::read(regs);
        let prim = initiator.prim_type;
        let tessellation = (is_major_mode_explicit(initiator.major_mode, prim)
            && VgtOutputPathCntl::read(regs).path_select == VgtOutputPath::TessellationEnable)
            .then(|| VgtHosCntl::read(regs).tess_mode);
        let mode_cntl = PaSuScModeCntl::read(regs);
        let mut index_count = initiator.num_indices;
        let mut index_buffer = None;
        let mut reset_index = None;
        if initiator.source_select == SourceSelect::Dma {
            let dma_size = VgtDmaSize::read(regs);
            let format = initiator.index_size;
            let raw_endian = dma_size.swap_mode;
            let endian = normalize_index_endian(format, raw_endian);
            index_count = index_count.min(dma_size.num_words);
            let guest_base = regs.reg(r::VGT_DMA_BASE) & !(format.size_bytes() - 1);
            index_buffer = Some(IndexBufferInfo {
                guest_base,
                format,
                endian,
                raw_endian,
                capacity: dma_size.num_words,
                count: index_count,
            });
            let adaptive = tessellation == Some(TessellationMode::Adaptive);
            if mode_cntl.multi_prim_ib_ena && !adaptive && prim.supports_primitive_reset() {
                let reset = VgtMultiPrimIbResetIndx::read(regs).reset_indx;
                if format == IndexFormat::Int32 || reset <= u32::from(u16::MAX) {
                    reset_index = Some(reset);
                }
            }
        }
        Self {
            initiator,
            primitive_type: prim,
            source_select: initiator.source_select,
            index_count,
            index_buffer,
            reset_index,
            index_params: VertexIndexParams::from_registers(regs),
            tessellation,
            provoking_vertex_last: mode_cntl.provoking_vtx_last,
        }
    }
}

/// 32-bit swap modes on 16-bit indices are normalized like Xenia: 8-in-32 becomes 8-in-16 and
/// 16-in-32 is dropped.
pub fn normalize_index_endian(format: IndexFormat, endian: Endian) -> Endian {
    match (format, endian) {
        (IndexFormat::Int16, Endian::Swap8In32) => Endian::Swap8In16,
        (IndexFormat::Int16, Endian::Swap16In32) => Endian::None,
        (_, e) => e,
    }
}

/// Decoded PM4 draw packet (DRAW_INDX / DRAW_INDX_2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawPacket<'a> {
    /// Visibility query condition token (DRAW_INDX only, 0 for DRAW_INDX_2).
    pub viz_query_condition: u32,
    pub initiator: VgtDrawInitiator,
    /// VGT_DMA_BASE (already masked to a physical address) for DMA sources.
    pub dma_base: Option<u32>,
    /// VGT_DMA_SIZE for DMA sources.
    pub dma_size: Option<VgtDmaSize>,
    /// Remaining packet payload (immediate index data for `SourceSelect::Immediate`).
    pub trailing: &'a [u32],
}

/// PM4 type 3 opcode of DRAW_INDX.
pub const PM4_DRAW_INDX: u32 = 0x22;
/// PM4 type 3 opcode of DRAW_INDX_2.
pub const PM4_DRAW_INDX_2: u32 = 0x36;

impl<'a> DrawPacket<'a> {
    /// Parses the (host-order) payload of a DRAW_INDX or DRAW_INDX_2 packet. Returns `None` if the
    /// opcode is not a draw or the payload is too short. The caller should then write
    /// VGT_DRAW_INITIATOR, VGT_DMA_BASE and VGT_DMA_SIZE into the register file.
    pub fn parse(opcode: u32, payload: &'a [u32]) -> Option<Self> {
        let (viz_query_condition, rest) = match opcode {
            PM4_DRAW_INDX => (*payload.first()?, &payload[1..]),
            PM4_DRAW_INDX_2 => (0, payload),
            _ => return None,
        };
        let initiator = VgtDrawInitiator::decode(*rest.first()?);
        let mut rest = &rest[1..];
        let (mut dma_base, mut dma_size) = (None, None);
        if initiator.source_select == SourceSelect::Dma {
            if rest.len() < 2 {
                return None;
            }
            dma_base = Some(rest[0] & 0x1FFF_FFFF);
            dma_size = Some(VgtDmaSize::decode(rest[1]));
            rest = &rest[2..];
        }
        Some(Self { viz_query_condition, initiator, dma_base, dma_size, trailing: rest })
    }
}

/// Host primitive topology (the WebGPU set).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

impl HostTopology {
    /// Whether this is a strip topology (restart values are honored for indexed draws).
    pub fn is_strip(self) -> bool {
        matches!(self, Self::LineStrip | Self::TriangleStrip)
    }
}

/// Host index data, host-order values (write as little-endian for WebGPU).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostIndices {
    /// Non-indexed draw of `HostDraw::vertex_count` sequential vertices starting at 0; the vertex
    /// shader must still apply [`VertexIndexParams`] to the vertex index.
    None,
    /// 16-bit indices, restart value 0xFFFF.
    U16(Vec<u16>),
    /// 32-bit indices, restart value 0xFFFFFFFF.
    U32(Vec<u32>),
}

/// Vertex shader expansion the host must perform for primitives without a host equivalent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveExpansion {
    None,
    /// Each host vertex index is `point * 4 + corner` (corner 0..3 of a two-triangle strip); the vertex
    /// shader emits the corner of the sprite for guest vertex `point`.
    PointSprite,
    /// Each host vertex index is `rect * 4 + corner`; corners 0..2 are the guest vertices
    /// `3 * rect + corner`, and corner 3 is derived as the vertex completing the parallelogram
    /// opposite of corner 0 (v1 + v2 - v0), so the strip (0, 1, 2, 3) covers the rectangle.
    RectangleList,
}

/// Result of converting a guest draw into a host draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostDraw {
    pub topology: HostTopology,
    pub indices: HostIndices,
    /// Number of indices (indexed) or vertices (non-indexed) to draw.
    pub vertex_count: u32,
    /// Whether restart values are present in `indices` (only for strip topologies).
    pub primitive_restart: bool,
    pub expansion: PrimitiveExpansion,
    /// For expansions with an index buffer (or baked indices): the guest vertex index of each
    /// expanded primitive vertex (1 per point, 3 per rectangle), with [`VertexIndexParams`] applied if
    /// `index_params_applied`. `None` for auto-indexed expansion without baking - then guest vertex
    /// indices are `point` / `3 * rect + corner`.
    pub expansion_guest_indices: Option<Vec<u32>>,
    /// Whether `indices` (or `expansion_guest_indices`) already have offset and clamping applied, so
    /// the vertex shader must use the host vertex index directly.
    pub index_params_applied: bool,
}

/// Host capabilities / preferences for primitive conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConversionOptions {
    /// Apply VGT_INDX_OFFSET / min / max to the indices on the CPU (always producing an index buffer).
    pub bake_index_params: bool,
    /// Allow 16-bit host indices when all values fit (below 0xFFFF).
    pub allow_16bit_indices: bool,
    /// Expand point lists into two-triangle strips (needed when sprites larger than 1 pixel are drawn,
    /// WebGPU has no point size).
    pub expand_point_sprites: bool,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self { bake_index_params: false, allow_16bit_indices: true, expand_point_sprites: false }
    }
}

/// Primitive processing failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveError {
    /// The guest primitive type has no conversion (2D copy/fill rectangles, triangles with W flags,
    /// none, unknown values).
    UnsupportedPrimitiveType(PrimitiveType),
    /// Immediate (in-packet) indices are not supported.
    UnsupportedIndexSource(SourceSelect),
    /// Tessellated draws need a host tessellation path.
    TessellationUnsupported(TessellationMode),
    /// The index data slice is smaller than `count * index size`.
    IndexDataTooShort { needed: usize, available: usize },
    /// Guest memory read of the index buffer failed.
    IndexBufferUnreadable { base: u32, size: u32 },
}

impl core::fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedPrimitiveType(p) => write!(f, "unsupported primitive type {p:?}"),
            Self::UnsupportedIndexSource(s) => write!(f, "unsupported index source {s:?}"),
            Self::TessellationUnsupported(m) => write!(f, "tessellation ({m:?}) is not supported"),
            Self::IndexDataTooShort { needed, available } => {
                write!(f, "index data too short: {needed} bytes needed, {available} available")
            }
            Self::IndexBufferUnreadable { base, size } => {
                write!(f, "index buffer at 0x{base:08X} (0x{size:X} bytes) is not readable")
            }
        }
    }
}

impl std::error::Error for PrimitiveError {}

/// Source of guest vertex indices.
#[derive(Clone, Copy, Debug)]
pub enum IndexSource<'a> {
    /// Sequential indices 0..count.
    Auto { count: u32 },
    /// Raw guest index buffer bytes exactly as stored in guest memory, interpreted like the GPU does:
    /// little-endian load followed by the `endian` swap (so big-endian D3D data uses `Swap8In16` for
    /// 16-bit and `Swap8In32` for 32-bit indices).
    Guest { bytes: &'a [u8], format: IndexFormat, endian: Endian, count: u32 },
}

impl IndexSource<'_> {
    fn count(&self) -> u32 {
        match *self {
            Self::Auto { count } | Self::Guest { count, .. } => count,
        }
    }
}

const RESET: u32 = u32::MAX;

/// Decodes guest indices into 24-bit values, with `RESET` for primitive reset indices.
fn decode_guest_indices(source: &IndexSource<'_>, reset_index: Option<u32>) -> Result<Vec<u32>, PrimitiveError> {
    match *source {
        IndexSource::Auto { count } => Ok((0..count).collect()),
        IndexSource::Guest { bytes, format, endian, count } => {
            let size = format.size_bytes() as usize;
            let needed = count as usize * size;
            if bytes.len() < needed {
                return Err(PrimitiveError::IndexDataTooShort { needed, available: bytes.len() });
            }
            let bytes = &bytes[..needed];
            Ok(match format {
                IndexFormat::Int16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| {
                        let v = u32::from(gpu_swap_u16(u16::from_le_bytes(*c), endian));
                        if reset_index == Some(v) { RESET } else { v }
                    })
                    .collect(),
                IndexFormat::Int32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| {
                        let v = gpu_swap_u32(u32::from_le_bytes(*c), endian) & VERTEX_INDEX_MASK;
                        if reset_index == Some(v) { RESET } else { v }
                    })
                    .collect(),
            })
        }
    }
}

/// Splits an index list at reset markers, dropping the markers.
fn segments(indices: &[u32]) -> impl Iterator<Item = &[u32]> {
    indices.split(|&i| i == RESET)
}

/// Joins strips with restart markers, skipping strips shorter than `min_len`.
fn join_strips<'s>(strips: impl Iterator<Item = &'s [u32]>, min_len: usize, out: &mut Vec<u32>) {
    for strip in strips.filter(|s| s.len() >= min_len) {
        if !out.is_empty() {
            out.push(RESET);
        }
        out.extend_from_slice(strip);
    }
}

/// Triangle fan (or polygon) to triangle list, ordered (v[i-1], v[i], v[0]) like Direct3D.
pub fn triangle_fan_to_list(fan: &[u32], out: &mut Vec<u32>) {
    if fan.len() < 3 {
        return;
    }
    for i in 2..fan.len() {
        out.extend_from_slice(&[fan[i - 1], fan[i], fan[0]]);
    }
}

/// Line loop to line strip (closing vertex appended). Loops with fewer than 2 vertices draw nothing.
pub fn line_loop_to_strip(lp: &[u32], out: &mut Vec<u32>) {
    if lp.len() < 2 {
        return;
    }
    out.extend_from_slice(lp);
    out.push(lp[0]);
}

/// Quad list to triangle list: (v0, v1, v2), (v0, v2, v3) per quad; incomplete quads are dropped.
pub fn quad_list_to_triangle_list(quads: &[u32], out: &mut Vec<u32>) {
    for q in quads.as_chunks::<4>().0 {
        out.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
}

/// Builds two-triangle strips `prim * 4 + corner` separated by restarts for `count` primitives.
fn two_triangle_strips(count: u32, out: &mut Vec<u32>) {
    for prim in 0..count {
        if prim != 0 {
            out.push(RESET);
        }
        out.extend((0..4).map(|corner| prim * 4 + corner));
    }
}

/// Converts guest topology and indices into a host draw.
///
/// `reset_index` is the 24-bit primitive reset index ([`DrawDescription::reset_index`]); it is only
/// honored for topologies where reset applies. `index_params` is only used with
/// `options.bake_index_params`.
pub fn convert_primitive(
    primitive_type: PrimitiveType,
    source: IndexSource<'_>,
    reset_index: Option<u32>,
    index_params: VertexIndexParams,
    options: &ConversionOptions,
) -> Result<HostDraw, PrimitiveError> {
    use PrimitiveType as P;
    let reset_index = reset_index.filter(|_| primitive_type.supports_primitive_reset());
    let count = source.count();
    let auto = matches!(source, IndexSource::Auto { .. });
    let bake = options.bake_index_params;

    let expansion = match primitive_type {
        P::PointList if options.expand_point_sprites => PrimitiveExpansion::PointSprite,
        P::RectangleList => PrimitiveExpansion::RectangleList,
        _ => PrimitiveExpansion::None,
    };

    let (topology, native) = match primitive_type {
        P::PointList if expansion == PrimitiveExpansion::None => (HostTopology::PointList, true),
        P::LineList => (HostTopology::LineList, true),
        P::LineStrip | P::LineStrip2D => (HostTopology::LineStrip, true),
        P::TriangleList => (HostTopology::TriangleList, true),
        P::TriangleStrip | P::TriStrip2D => (HostTopology::TriangleStrip, true),
        P::PointList | P::RectangleList => (HostTopology::TriangleStrip, false),
        P::TriangleFan | P::Polygon | P::QuadList => (HostTopology::TriangleList, false),
        P::LineLoop => (HostTopology::LineStrip, false),
        P::QuadStrip => (HostTopology::TriangleStrip, false),
        other => return Err(PrimitiveError::UnsupportedPrimitiveType(other)),
    };

    // Fast path: auto-indexed native topology, nothing to bake.
    if auto && native && !bake {
        return Ok(HostDraw {
            topology,
            indices: HostIndices::None,
            vertex_count: count,
            primitive_restart: false,
            expansion,
            expansion_guest_indices: None,
            index_params_applied: false,
        });
    }

    let mut guest = decode_guest_indices(&source, reset_index)?;
    if bake {
        for i in guest.iter_mut().filter(|i| **i != RESET) {
            *i = index_params.apply(*i);
        }
    }

    let mut host = Vec::new();
    let mut expansion_guest_indices = None;
    match expansion {
        PrimitiveExpansion::PointSprite | PrimitiveExpansion::RectangleList => {
            let per_prim = if expansion == PrimitiveExpansion::RectangleList { 3 } else { 1 };
            let prims = count / per_prim;
            two_triangle_strips(prims, &mut host);
            if !auto || bake {
                expansion_guest_indices = Some(guest[..(prims * per_prim) as usize].to_vec());
            }
        }
        PrimitiveExpansion::None => match primitive_type {
            P::PointList | P::LineList | P::TriangleList => host = guest,
            P::LineStrip | P::LineStrip2D => join_strips(segments(&guest), 2, &mut host),
            P::TriangleStrip | P::TriStrip2D => join_strips(segments(&guest), 3, &mut host),
            P::TriangleFan | P::Polygon => segments(&guest).for_each(|s| triangle_fan_to_list(s, &mut host)),
            P::QuadList => quad_list_to_triangle_list(&guest, &mut host),
            P::LineLoop => {
                for s in segments(&guest).filter(|s| s.len() >= 2) {
                    if !host.is_empty() {
                        host.push(RESET);
                    }
                    line_loop_to_strip(s, &mut host);
                }
            }
            P::QuadStrip => {
                join_strips(segments(&guest).map(|s| &s[..s.len() & !1]), 4, &mut host);
            }
            _ => unreachable!("filtered by the topology match"),
        },
    }

    let primitive_restart = topology.is_strip() && host.contains(&RESET);
    let max_value = host.iter().copied().filter(|&i| i != RESET).max().unwrap_or(0);
    let vertex_count = host.len() as u32;
    let indices = if options.allow_16bit_indices && max_value < u32::from(u16::MAX) {
        HostIndices::U16(host.into_iter().map(|i| if i == RESET { u16::MAX } else { i as u16 }).collect())
    } else {
        HostIndices::U32(host)
    };
    Ok(HostDraw {
        topology,
        indices,
        vertex_count,
        primitive_restart,
        expansion,
        expansion_guest_indices,
        index_params_applied: bake,
    })
}

/// Reads the guest index buffer of `draw` (if any) and converts the draw for the host.
pub fn process_draw(
    draw: &DrawDescription,
    memory: &(impl GuestMemory + ?Sized),
    options: &ConversionOptions,
) -> Result<HostDraw, PrimitiveError> {
    if let Some(mode) = draw.tessellation {
        return Err(PrimitiveError::TessellationUnsupported(mode));
    }
    match draw.source_select {
        SourceSelect::AutoIndex => convert_primitive(
            draw.primitive_type,
            IndexSource::Auto { count: draw.index_count },
            None,
            draw.index_params,
            options,
        ),
        SourceSelect::Dma => {
            let ib = draw.index_buffer.expect("DMA draws always have an index buffer description");
            let mut bytes = vec![0u8; ib.size_bytes() as usize];
            if !memory.read_physical(ib.guest_base, &mut bytes) {
                return Err(PrimitiveError::IndexBufferUnreadable { base: ib.guest_base, size: ib.size_bytes() });
            }
            convert_primitive(
                draw.primitive_type,
                IndexSource::Guest { bytes: &bytes, format: ib.format, endian: ib.endian, count: ib.count },
                draw.reset_index,
                draw.index_params,
                options,
            )
        }
        other => Err(PrimitiveError::UnsupportedIndexSource(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_be_bytes()).collect()
    }
    fn be32(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_be_bytes()).collect()
    }
    fn u16s(d: &HostDraw) -> Vec<u16> {
        match &d.indices {
            HostIndices::U16(v) => v.clone(),
            other => panic!("expected u16 indices, got {other:?}"),
        }
    }
    fn u32s(d: &HostDraw) -> Vec<u32> {
        match &d.indices {
            HostIndices::U32(v) => v.clone(),
            other => panic!("expected u32 indices, got {other:?}"),
        }
    }
    const R16: u16 = u16::MAX;
    const R32: u32 = u32::MAX;
    fn opts() -> ConversionOptions {
        ConversionOptions::default()
    }
    fn conv(p: PrimitiveType, src: IndexSource<'_>, reset: Option<u32>) -> HostDraw {
        convert_primitive(p, src, reset, VertexIndexParams::default(), &opts()).unwrap()
    }

    #[test]
    fn auto_native_passthrough() {
        let d = conv(PrimitiveType::TriangleStrip, IndexSource::Auto { count: 7 }, None);
        assert_eq!(d.topology, HostTopology::TriangleStrip);
        assert_eq!(d.indices, HostIndices::None);
        assert_eq!(d.vertex_count, 7);
        assert!(!d.primitive_restart);
    }

    #[test]
    fn auto_fan_and_polygon() {
        for p in [PrimitiveType::TriangleFan, PrimitiveType::Polygon] {
            let d = conv(p, IndexSource::Auto { count: 5 }, None);
            assert_eq!(d.topology, HostTopology::TriangleList);
            assert_eq!(u16s(&d), vec![1, 2, 0, 2, 3, 0, 3, 4, 0]);
            assert_eq!(d.vertex_count, 9);
        }
        let d = conv(PrimitiveType::TriangleFan, IndexSource::Auto { count: 2 }, None);
        assert_eq!(d.vertex_count, 0);
    }

    #[test]
    fn fan_with_reset_16bit_be() {
        let data = be16(&[10, 11, 12, 13, 0xBEEF, 20, 21, 22, 0xBEEF, 30, 31]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::Swap8In16, count: 11 };
        let d = conv(PrimitiveType::TriangleFan, src, Some(0xBEEF));
        assert_eq!(u16s(&d), vec![11, 12, 10, 12, 13, 10, 21, 22, 20]);
        assert!(!d.primitive_restart);
    }

    #[test]
    fn quad_list() {
        let d = conv(PrimitiveType::QuadList, IndexSource::Auto { count: 9 }, None);
        assert_eq!(u16s(&d), vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        // Reset index is ignored for lists.
        let data = be32(&[0, 1, 2, 0xFFFFFF, 4, 5, 6, 7]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int32, endian: Endian::Swap8In32, count: 8 };
        let d = conv(PrimitiveType::QuadList, src, Some(0xFFFFFF));
        assert_eq!(u32s(&d), vec![0, 1, 2, 0, 2, 0xFFFFFF, 4, 5, 6, 4, 6, 7]);
    }

    #[test]
    fn quad_strip() {
        let d = conv(PrimitiveType::QuadStrip, IndexSource::Auto { count: 7 }, None);
        assert_eq!(d.topology, HostTopology::TriangleStrip);
        assert_eq!(u16s(&d), vec![0, 1, 2, 3, 4, 5]);
        let data = be16(&[0, 1, 2, 3, 0xFFFF, 5, 6, 0xFFFF, 7, 8, 9, 10, 11]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::Swap8In16, count: 13 };
        let d = conv(PrimitiveType::QuadStrip, src, Some(0xFFFF));
        assert_eq!(u16s(&d), vec![0, 1, 2, 3, R16, 7, 8, 9, 10]);
        assert!(d.primitive_restart);
    }

    #[test]
    fn line_loops() {
        let d = conv(PrimitiveType::LineLoop, IndexSource::Auto { count: 4 }, None);
        assert_eq!(d.topology, HostTopology::LineStrip);
        assert_eq!(u16s(&d), vec![0, 1, 2, 3, 0]);
        let data = be32(&[5, 6, 7, 0x00AB_CDEF, 9, 0x00AB_CDEF, 1, 2]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int32, endian: Endian::Swap8In32, count: 8 };
        let d = conv(PrimitiveType::LineLoop, src, Some(0xAB_CDEF));
        // Single-vertex loop dropped, loops separated by restart.
        assert_eq!(u16s(&d), vec![5, 6, 7, 5, R16, 1, 2, 1]);
        assert!(d.primitive_restart);
    }

    #[test]
    fn rectangle_list_expansion() {
        let d = conv(PrimitiveType::RectangleList, IndexSource::Auto { count: 7 }, None);
        assert_eq!(d.topology, HostTopology::TriangleStrip);
        assert_eq!(d.expansion, PrimitiveExpansion::RectangleList);
        assert_eq!(u16s(&d), vec![0, 1, 2, 3, R16, 4, 5, 6, 7]);
        assert_eq!(d.expansion_guest_indices, None);
        let data = be16(&[9, 8, 7, 6, 5, 4]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::Swap8In16, count: 6 };
        let d = conv(PrimitiveType::RectangleList, src, None);
        assert_eq!(d.expansion_guest_indices, Some(vec![9, 8, 7, 6, 5, 4]));
    }

    #[test]
    fn point_sprite_expansion() {
        let o = ConversionOptions { expand_point_sprites: true, ..opts() };
        let d =
            convert_primitive(PrimitiveType::PointList, IndexSource::Auto { count: 2 }, None, Default::default(), &o)
                .unwrap();
        assert_eq!(d.expansion, PrimitiveExpansion::PointSprite);
        assert_eq!(u16s(&d), vec![0, 1, 2, 3, R16, 4, 5, 6, 7]);
        let d = conv(PrimitiveType::PointList, IndexSource::Auto { count: 2 }, None);
        assert_eq!((d.topology, d.indices), (HostTopology::PointList, HostIndices::None));
    }

    #[test]
    fn strip_restart_endians_and_widths() {
        // Same logical indices encoded with each 32-bit endian mode.
        let logical = [1u32, 2, 3, 0x123456, 4, 5, 6];
        for endian in [Endian::None, Endian::Swap8In16, Endian::Swap8In32, Endian::Swap16In32] {
            // Stored value s such that swap(le_load(s)) == logical (swaps are involutions).
            let data: Vec<u8> = logical.iter().flat_map(|&v| gpu_swap_u32(v, endian).to_le_bytes()).collect();
            let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int32, endian, count: 7 };
            let d = conv(PrimitiveType::TriangleStrip, src, Some(0x123456));
            assert_eq!(u16s(&d), vec![1, 2, 3, R16, 4, 5, 6], "{endian:?}");
        }
        // Upper 8 bits ignored for the reset comparison and the index value.
        let data = be32(&[0xFF00_0001, 2, 3, 0xFFFF_FFFF, 4, 5, 6]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int32, endian: Endian::Swap8In32, count: 7 };
        let d = conv(PrimitiveType::TriangleStrip, src, Some(0xFF_FFFF));
        assert_eq!(u16s(&d), vec![1, 2, 3, R16, 4, 5, 6]);
        // Short strips are dropped entirely.
        let data = be16(&[1, 2, 0xFFFF, 3, 4, 5]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::Swap8In16, count: 6 };
        let d = conv(PrimitiveType::TriangleStrip, src, Some(0xFFFF));
        assert_eq!(u16s(&d), vec![3, 4, 5]);
        assert!(!d.primitive_restart);
        // 16-bit little-endian (no swap).
        let data: Vec<u8> = [1u16, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::None, count: 3 };
        assert_eq!(u16s(&conv(PrimitiveType::TriangleList, src, None)), vec![1, 2, 3]);
    }

    #[test]
    fn ffff_vertex_index_forces_32bit() {
        // 0xFFFF used as a real vertex with a different reset index.
        let data = be16(&[0xFFFF, 1, 2, 0x8000, 3, 4, 5]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::Swap8In16, count: 7 };
        let d = conv(PrimitiveType::TriangleStrip, src, Some(0x8000));
        assert_eq!(u32s(&d), vec![0xFFFF, 1, 2, R32, 3, 4, 5]);
        assert!(d.primitive_restart);
        let o = ConversionOptions { allow_16bit_indices: false, ..opts() };
        let d = convert_primitive(
            PrimitiveType::TriangleList,
            IndexSource::Auto { count: 3 },
            None,
            Default::default(),
            &o,
        )
        .unwrap();
        // Native auto-indexed list without baking is still non-indexed.
        assert_eq!(d.indices, HostIndices::None);
    }

    #[test]
    fn baking_offset_and_clamp() {
        let params = VertexIndexParams { offset: 0xFF_FFFE, min: 1, max: 100 };
        assert_eq!(params.apply(3), 1);
        assert_eq!(params.apply(1), 100);
        let params = VertexIndexParams { offset: 10, min: 12, max: 0xFF_FFFF };
        let o = ConversionOptions { bake_index_params: true, ..opts() };
        let d =
            convert_primitive(PrimitiveType::TriangleStrip, IndexSource::Auto { count: 4 }, None, params, &o).unwrap();
        assert_eq!(u16s(&d), vec![12, 12, 12, 13]);
        assert!(d.index_params_applied);
        // Reset check happens before offsetting.
        let data = be16(&[0, 1, 2, 5, 3, 4, 6]);
        let src = IndexSource::Guest { bytes: &data, format: IndexFormat::Int16, endian: Endian::Swap8In16, count: 7 };
        let params = VertexIndexParams { offset: 0x10000, min: 0, max: 0xFF_FFFF };
        let d = convert_primitive(PrimitiveType::TriangleStrip, src, Some(5), params, &o).unwrap();
        assert_eq!(u32s(&d), vec![0x10000, 0x10001, 0x10002, R32, 0x10003, 0x10004, 0x10006]);
    }

    #[test]
    fn unsupported() {
        for p in [PrimitiveType::TriangleWithWFlags, PrimitiveType::Copy2DRectListV0, PrimitiveType::Unknown(9)] {
            assert_eq!(
                convert_primitive(p, IndexSource::Auto { count: 3 }, None, Default::default(), &opts()),
                Err(PrimitiveError::UnsupportedPrimitiveType(p))
            );
        }
        let src = IndexSource::Guest { bytes: &[0; 3], format: IndexFormat::Int16, endian: Endian::None, count: 2 };
        assert_eq!(
            convert_primitive(PrimitiveType::TriangleList, src, None, Default::default(), &opts()),
            Err(PrimitiveError::IndexDataTooShort { needed: 4, available: 3 })
        );
    }

    #[test]
    fn draw_description_and_process() {
        let mut regs = vec![0u32; 0x5003];
        let init = VgtDrawInitiator {
            prim_type: PrimitiveType::TriangleFan,
            source_select: SourceSelect::Dma,
            major_mode: MajorMode::Implicit,
            index_size: IndexFormat::Int16,
            not_eop: false,
            num_indices: 6,
        };
        regs[r::VGT_DRAW_INITIATOR as usize] = init.encode();
        regs[r::VGT_DMA_BASE as usize] = 0x101;
        regs[r::VGT_DMA_SIZE as usize] = VgtDmaSize { num_words: 5, swap_mode: Endian::Swap8In32 }.encode();
        regs[r::PA_SU_SC_MODE_CNTL as usize] = 1 << 21;
        regs[r::VGT_MULTI_PRIM_IB_RESET_INDX as usize] = 0xFFFF;
        regs[r::VGT_MAX_VTX_INDX as usize] = 0xFF_FFFF;
        let d = DrawDescription::from_registers(&regs);
        let ib = d.index_buffer.unwrap();
        assert_eq!(ib.guest_base, 0x100);
        assert_eq!(ib.count, 5);
        assert_eq!(ib.endian, Endian::Swap8In16);
        assert_eq!(ib.raw_endian, Endian::Swap8In32);
        assert_eq!(d.reset_index, Some(0xFFFF));
        assert_eq!(d.tessellation, None);

        let mut mem = vec![0u8; 0x200];
        mem[0x100..0x10A].copy_from_slice(&be16(&[7, 8, 9, 0xFFFF, 1]));
        let host = process_draw(&d, &mem, &opts()).unwrap();
        assert_eq!(u16s(&host), vec![8, 9, 7]);

        // A reset index above 0xFFFF disables reset for 16-bit indices.
        regs[r::VGT_MULTI_PRIM_IB_RESET_INDX as usize] = 0x10000;
        assert_eq!(DrawDescription::from_registers(&regs).reset_index, None);
        // Unreadable index buffer.
        let host = process_draw(&d, &mem[..0x104], &opts());
        assert_eq!(host, Err(PrimitiveError::IndexBufferUnreadable { base: 0x100, size: 10 }));

        // Tessellation in the explicit major mode.
        regs[r::VGT_DRAW_INITIATOR as usize] = VgtDrawInitiator { major_mode: MajorMode::Explicit, ..init }.encode();
        regs[r::VGT_OUTPUT_PATH_CNTL as usize] = 1;
        regs[r::VGT_HOS_CNTL as usize] = 1;
        let d = DrawDescription::from_registers(&regs);
        assert_eq!(d.tessellation, Some(TessellationMode::Continuous));
        assert_eq!(
            process_draw(&d, &mem, &opts()),
            Err(PrimitiveError::TessellationUnsupported(TessellationMode::Continuous))
        );
    }

    #[test]
    fn draw_packets() {
        let init = VgtDrawInitiator::decode(0x04 | (1 << 11) | (3 << 16));
        let payload = [0x55, init.encode(), 0xA000_1000, 0x8000_0003];
        let p = DrawPacket::parse(PM4_DRAW_INDX, &payload).unwrap();
        assert_eq!(p.viz_query_condition, 0x55);
        assert_eq!(p.initiator, init);
        assert_eq!(p.dma_base, Some(0x0000_1000));
        assert_eq!(p.dma_size.unwrap().num_words, 3);
        assert!(p.trailing.is_empty());
        let auto = VgtDrawInitiator::decode(0x06 | (2 << 6) | (4 << 16));
        let payload2 = [auto.encode(), 1, 2];
        let p = DrawPacket::parse(PM4_DRAW_INDX_2, &payload2).unwrap();
        assert_eq!(p.dma_base, None);
        assert_eq!(p.trailing, &[1, 2]);
        assert!(DrawPacket::parse(PM4_DRAW_INDX, &[0, init.encode(), 0]).is_none());
        assert!(DrawPacket::parse(0x10, &[0]).is_none());
    }
}
