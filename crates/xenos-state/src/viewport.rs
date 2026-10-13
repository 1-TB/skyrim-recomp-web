//! Viewport, scissor, clipping and rasterizer state (Xenia draw_util.cc GetHostViewportInfo /
//! GetScissor / GetPreferredFacePolygonOffset).

use crate::RegisterSource;
use crate::bitfields::*;
use crate::registers as r;
use crate::xenos::*;

/// Host-side parameters for the viewport conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportArgs {
    /// Resolution scale (1 for native resolution).
    pub draw_resolution_scale_x: u32,
    pub draw_resolution_scale_y: u32,
    /// Whether host NDC +Y points up (Direct3D 12, Metal, WebGPU) - the guest convention is +Y down
    /// the render target, so Y is flipped.
    pub origin_bottom_left: bool,
    /// Maximum host viewport bounds (in host pixels, with resolution scale).
    pub x_max: u32,
    pub y_max: u32,
    /// Whether the host allows z_min > z_max (WebGPU does not).
    pub allow_reverse_z: bool,
    /// Depth is snapped to float24 in the pixel shader, so the depth bounds are snapped too.
    pub convert_z_to_float24: bool,
    /// Float24 depth is stored as [0, 2) remapped to [0, 1).
    pub full_float24_in_0_to_1: bool,
    pub pixel_shader_writes_depth: bool,
    /// Whether the host programs Xenos sample positions (affects resolution-scaled MSAA offsets).
    pub xenos_sample_positions: bool,
    /// Apply the Direct3D 9 half-pixel offset for `PixelCenter::D3DZero` (Xenia's half_pixel_offset).
    pub half_pixel_offset: bool,
    /// The window offset is folded into EDRAM bases instead of vertex positions.
    pub window_offset_in_edram: bool,
}

impl ViewportArgs {
    /// Defaults for a WebGPU host at native resolution with `max_dimension` (maxTextureDimension2D).
    pub fn webgpu(max_dimension: u32) -> Self {
        Self {
            draw_resolution_scale_x: 1,
            draw_resolution_scale_y: 1,
            origin_bottom_left: true,
            x_max: max_dimension,
            y_max: max_dimension,
            allow_reverse_z: false,
            convert_z_to_float24: false,
            full_float24_in_0_to_1: false,
            pixel_shader_writes_depth: false,
            xenos_sample_positions: false,
            half_pixel_offset: true,
            window_offset_in_edram: false,
        }
    }
}

/// Guest registers that affect the viewport transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportRegisters {
    pub clip_cntl: PaClClipCntl,
    pub vte_cntl: PaClVteCntl,
    pub mode_cntl: PaSuScModeCntl,
    pub vtx_cntl: PaSuVtxCntl,
    pub window_offset: PaScWindowOffset,
    /// PA_CL_VPORT_XSCALE / YSCALE / ZSCALE.
    pub scale: [f32; 3],
    /// PA_CL_VPORT_XOFFSET / YOFFSET / ZOFFSET.
    pub offset: [f32; 3],
    pub depth_format: DepthRenderTargetFormat,
    pub msaa_samples: MsaaSamples,
    /// Normalized depth test enable (see [`crate::depth_stencil::normalized_depth_control`]).
    pub z_enable: bool,
}

impl ViewportRegisters {
    /// Reads the viewport registers; `z_enable` should be the normalized depth enable.
    pub fn from_registers(regs: &(impl RegisterSource + ?Sized), z_enable: bool) -> Self {
        Self {
            clip_cntl: PaClClipCntl::read(regs),
            vte_cntl: PaClVteCntl::read(regs),
            mode_cntl: PaSuScModeCntl::read(regs),
            vtx_cntl: PaSuVtxCntl::read(regs),
            window_offset: PaScWindowOffset::read(regs),
            scale: [
                regs.reg_f32(r::PA_CL_VPORT_XSCALE),
                regs.reg_f32(r::PA_CL_VPORT_YSCALE),
                regs.reg_f32(r::PA_CL_VPORT_ZSCALE),
            ],
            offset: [
                regs.reg_f32(r::PA_CL_VPORT_XOFFSET),
                regs.reg_f32(r::PA_CL_VPORT_YOFFSET),
                regs.reg_f32(r::PA_CL_VPORT_ZOFFSET),
            ],
            depth_format: RbDepthInfo::read(regs).depth_format,
            msaa_samples: RbSurfaceInfo::read(regs).msaa_samples,
            z_enable,
        }
    }
}

/// Host viewport and the clip-space transform the vertex shader must apply.
///
/// The vertex shader output position must be transformed as
/// `pos.xyz = pos.xyz * ndc_scale + ndc_offset * pos.w` (after any guest W handling) before being
/// written to the host clip-space position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportInfo {
    /// Integer viewport origin in host render target pixels (may be negative).
    pub xy_offset: [i32; 2],
    /// Integer viewport size; zero for an empty viewport (use an empty scissor on the host).
    pub xy_extent: [u32; 2],
    pub z_min: f32,
    pub z_max: f32,
    pub ndc_scale: [f32; 3],
    pub ndc_offset: [f32; 3],
}

impl ViewportInfo {
    /// Crops the viewport to `[0, width) x [0, height)` (WebGPU requires the viewport to lie within the
    /// attachments), folding the change into `ndc_scale` / `ndc_offset` so positions are unchanged.
    pub fn crop_to_render_target(&self, width: u32, height: u32, origin_bottom_left: bool) -> Self {
        let mut out = *self;
        for axis in 0..2 {
            let limit = i64::from(if axis == 0 { width } else { height });
            let off = i64::from(self.xy_offset[axis]);
            let ext = i64::from(self.xy_extent[axis]);
            let n0 = off.clamp(0, limit);
            let n1 = (off + ext).clamp(0, limit);
            if n1 <= n0 || ext == 0 {
                out.xy_offset[axis] = n0 as i32;
                out.xy_extent[axis] = 0;
                continue;
            }
            if n0 == off && n1 == off + ext {
                continue;
            }
            let next = (n1 - n0) as f32;
            let k = ext as f32 / next;
            let c = (2.0 * (off - n0) as f32 + ext as f32) / next - 1.0;
            let flip = axis == 1 && origin_bottom_left;
            out.ndc_scale[axis] = self.ndc_scale[axis] * k;
            out.ndc_offset[axis] = if flip { self.ndc_offset[axis] * k - c } else { self.ndc_offset[axis] * k + c };
            out.xy_offset[axis] = n0 as i32;
            out.xy_extent[axis] = (n1 - n0) as u32;
        }
        out
    }
}

/// Converts the guest viewport to a host viewport plus NDC scale/offset (GetHostViewportInfo).
///
/// The result always uses Direct3D-style 0...W clip-space Z.
pub fn host_viewport_info(v: &ViewportRegisters, args: &ViewportArgs) -> ViewportInfo {
    let scale_x_res = args.draw_resolution_scale_x.max(1);
    let scale_y_res = args.draw_resolution_scale_y.max(1);
    let vte = v.vte_cntl;
    let scale_xy =
        [if vte.vport_x_scale_ena { v.scale[0] } else { 1.0 }, if vte.vport_y_scale_ena { v.scale[1] } else { 1.0 }];
    let scale_z = if vte.vport_z_scale_ena { v.scale[2] } else { 1.0 };
    let offset_base_xy = [
        if vte.vport_x_offset_ena { v.offset[0] } else { 0.0 },
        if vte.vport_y_offset_ena { v.offset[1] } else { 0.0 },
    ];
    let offset_z = if vte.vport_z_offset_ena { v.offset[2] } else { 0.0 };

    let mut offset_add_xy = [0.0f32; 2];
    if v.mode_cntl.vtx_window_offset_enable && !args.window_offset_in_edram {
        offset_add_xy[0] += v.window_offset.window_x_offset as f32;
        offset_add_xy[1] += v.window_offset.window_y_offset as f32;
    }
    if args.half_pixel_offset && v.vtx_cntl.pix_center == PixelCenter::D3DZero {
        offset_add_xy[0] += 0.5;
        offset_add_xy[1] += 0.5;
    }
    // Keep the Xenos sample offset in host pixels with resolution scaling (sample positions don't scale).
    if args.xenos_sample_positions && v.msaa_samples != MsaaSamples::X1 && (scale_x_res > 1 || scale_y_res > 1) {
        let fraction: [f32; 2] = core::array::from_fn(|i| {
            let o = offset_base_xy[i] + offset_add_xy[i];
            o - o.floor() - 0.5
        });
        let positions: &[[i8; 2]] =
            if v.msaa_samples == MsaaSamples::X4 { &XENOS_SAMPLE_POSITIONS_4X } else { &XENOS_SAMPLE_POSITIONS_2X };
        if positions.iter().any(|p| fraction[0] == f32::from(p[0]) / 16.0 && fraction[1] == f32::from(p[1]) / 16.0) {
            for (j, res) in [scale_x_res, scale_y_res].into_iter().enumerate() {
                offset_add_xy[j] -= fraction[j] * (res - 1) as f32 / res as f32;
            }
        }
    }

    let xy_max_unscaled = [(args.x_max / scale_x_res).max(1), (args.y_max / scale_y_res).max(1)];
    let res_scale = [scale_x_res, scale_y_res];
    let mut info = ViewportInfo {
        xy_offset: [0; 2],
        xy_extent: [0; 2],
        z_min: 0.0,
        z_max: 1.0,
        ndc_scale: [1.0; 3],
        ndc_offset: [0.0; 3],
    };

    if v.clip_cntl.clip_disable {
        // Huge viewport; all offsetting happens in the vertex shader.
        for i in 0..2 {
            let extent_unscaled = TEXTURE_2D_MAX_WIDTH_HEIGHT.min(xy_max_unscaled[i]);
            info.xy_offset[i] = 0;
            info.xy_extent[i] = extent_unscaled * res_scale[i];
            let extent_f = extent_unscaled as f32;
            let pixels_to_ndc = 2.0 / extent_f;
            info.ndc_scale[i] = scale_xy[i] * pixels_to_ndc;
            info.ndc_offset[i] = (offset_base_xy[i] - extent_f * 0.5 + offset_add_xy[i]) * pixels_to_ndc;
        }
        info.z_min = 0.0;
        info.z_max = 1.0;
        info.ndc_scale[2] = scale_z;
        info.ndc_offset[2] = offset_z;
    } else {
        for i in 0..2 {
            let offset_axis = offset_base_xy[i] + offset_add_xy[i];
            let scale_abs = scale_xy[i].abs();
            let max_f = xy_max_unscaled[i] as f32;
            let axis_1 = clamp_float(offset_axis + scale_abs, 0.0, max_f).floor() as i32;
            let axis_0 = clamp_float(offset_axis - scale_abs, axis_1 as f32 - max_f, max_f).floor() as i32;
            let extent = if axis_1 > 0 { (axis_1 - axis_0) as u32 } else { 0 };
            info.xy_offset[i] = axis_0 * res_scale[i] as i32;
            info.xy_extent[i] = extent * res_scale[i];
            if extent != 0 {
                let extent_f = extent as f32;
                let inv = 1.0 / extent_f;
                info.ndc_scale[i] = scale_xy[i] * 2.0 * inv;
                info.ndc_offset[i] = (offset_axis - (axis_0 as f32 + extent_f * 0.5)) * 2.0 * inv;
            } else {
                info.ndc_scale[i] = 1.0;
                info.ndc_offset[i] = 0.0;
            }
        }
        let (host_clip_offset_z, host_clip_scale_z);
        if v.clip_cntl.dx_clip_space_def {
            host_clip_offset_z = offset_z;
            host_clip_scale_z = scale_z;
            info.ndc_scale[2] = 1.0;
            info.ndc_offset[2] = 0.0;
        } else {
            // Remap OpenGL -W...W clip space to 0...W.
            host_clip_offset_z = offset_z - scale_z;
            host_clip_scale_z = scale_z * 2.0;
            info.ndc_scale[2] = 0.5;
            info.ndc_offset[2] = 0.5;
        }
        if args.pixel_shader_writes_depth {
            info.z_min = 0.0;
            info.z_max = 1.0;
        } else {
            info.z_min = saturate(host_clip_offset_z);
            info.z_max = saturate(host_clip_offset_z + host_clip_scale_z);
            if !args.allow_reverse_z && info.z_min > info.z_max {
                core::mem::swap(&mut info.z_min, &mut info.z_max);
                info.ndc_scale[2] = -info.ndc_scale[2];
                info.ndc_offset[2] = 1.0 - info.ndc_offset[2];
            }
        }
    }

    if v.z_enable && v.depth_format == DepthRenderTargetFormat::D24FS8 {
        if args.convert_z_to_float24 {
            info.z_min = float20e4_to_32(float32_to_20e4(info.z_min, true));
            info.z_max = float20e4_to_32(float32_to_20e4(info.z_max, true));
        }
        if args.full_float24_in_0_to_1 {
            info.z_min *= 0.5;
            info.z_max *= 0.5;
        }
    }

    if args.origin_bottom_left {
        info.ndc_scale[1] = -info.ndc_scale[1];
        info.ndc_offset[1] = -info.ndc_offset[1];
    }
    info
}

/// NaN-safe clamp (NaN becomes `lo`), like xe::clamp_float.
fn clamp_float(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() { lo } else { v.max(lo).min(hi) }
}

fn saturate(v: f32) -> f32 {
    clamp_float(v, 0.0, 1.0)
}

/// Integer scissor rectangle in render target pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Scissor {
    pub offset: [u32; 2],
    /// May be zero.
    pub extent: [u32; 2],
}

impl Scissor {
    /// Clamps the rectangle to `[0, width) x [0, height)` (WebGPU requires it to be within the
    /// attachments).
    pub fn clamp_to(&self, width: u32, height: u32) -> Self {
        let x0 = self.offset[0].min(width);
        let y0 = self.offset[1].min(height);
        let x1 = self.offset[0].saturating_add(self.extent[0]).min(width);
        let y1 = self.offset[1].saturating_add(self.extent[1]).min(height);
        Self { offset: [x0, y0], extent: [x1 - x0, y1 - y0] }
    }
}

/// Computes the scissor rectangle from the window scissor, window offset and screen scissor
/// (GetScissor). With `clamp_to_surface_pitch` the horizontal range is limited to the surface pitch.
pub fn scissor(
    regs: &(impl RegisterSource + ?Sized),
    clamp_to_surface_pitch: bool,
    window_offset_in_edram: bool,
) -> Scissor {
    let tl = PaScWindowScissorTl::read(regs);
    let br = PaScWindowScissorBr::read(regs);
    let window_offset = PaScWindowOffset::read(regs);
    let screen_tl = PaScScreenScissor::decode(regs.reg(r::PA_SC_SCREEN_SCISSOR_TL));
    let screen_br = PaScScreenScissor::decode(regs.reg(r::PA_SC_SCREEN_SCISSOR_BR));
    let (wx, wy) =
        if tl.window_offset_disable { (0, 0) } else { (window_offset.window_x_offset, window_offset.window_y_offset) };
    let mut tl_x = tl.tl_x as i32 + wx;
    let mut tl_y = tl.tl_y as i32 + wy;
    let mut br_x = br.br_x as i32 + wx;
    let mut br_y = br.br_y as i32 + wy;
    tl_x = tl_x.max(screen_tl.x);
    tl_y = tl_y.max(screen_tl.y);
    br_x = br_x.min(screen_br.x);
    br_y = br_y.min(screen_br.y);
    if clamp_to_surface_pitch {
        let pitch = RbSurfaceInfo::read(regs).surface_pitch as i32;
        tl_x = tl_x.min(pitch);
        br_x = br_x.min(pitch);
    }
    tl_x = tl_x.max(0);
    tl_y = tl_y.max(0);
    br_x = br_x.max(tl_x);
    br_y = br_y.max(tl_y);
    if window_offset_in_edram {
        tl_x -= wx;
        tl_y -= wy;
        br_x -= wx;
        br_y -= wy;
    }
    Scissor { offset: [tl_x as u32, tl_y as u32], extent: [(br_x - tl_x) as u32, (br_y - tl_y) as u32] }
}

/// WebGPU-style face culling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// WebGPU-style front face winding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Ccw,
    Cw,
}

/// Polygon fill mode requested by the guest (WebGPU only supports `Fill`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FillMode {
    Fill,
    Line,
    Point,
}

/// Constant + slope-scaled depth bias in Direct3D 10 / WebGPU units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthBias {
    /// Guest absolute constant offset (PA_SU_POLY_OFFSET_*_OFFSET).
    pub guest_offset: f32,
    /// Guest slope scale in 1/16 subpixel units (PA_SU_POLY_OFFSET_*_SCALE).
    pub guest_scale: f32,
    /// Host integer bias (WebGPU `depthBias`).
    pub constant: i32,
    /// Host slope scale in pixels (WebGPU `depthBiasSlopeScale`).
    pub slope_scale: f32,
}

/// Rasterizer state for the draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterizerState {
    /// Whether the primitive type is polygonal (faces exist).
    pub polygonal: bool,
    pub cull_mode: CullMode,
    /// Both faces culled - polygonal draws produce no fragments.
    pub cull_all: bool,
    pub front_face: FrontFace,
    pub fill_mode: FillMode,
    /// Front and back fill modes differ (not expressible on the host; `fill_mode` holds the front one).
    pub mixed_fill_modes: bool,
    pub depth_bias: DepthBias,
    /// Clip-space clipping disabled (needs WebGPU `unclippedDepth` for the Z part).
    pub clip_disable: bool,
    /// Enabled user clip planes (bits 0..5) and their (x, y, z, w) coefficients.
    pub user_clip_plane_mask: u32,
    pub user_clip_planes: [[f32; 4]; 6],
    pub provoking_vertex_last: bool,
    pub msaa_enable: bool,
    /// Point sprite size in pixels (PA_SU_POINT_SIZE is a 12.4 half-size).
    pub point_size: [f32; 2],
    /// Per-vertex point size clamp in pixels.
    pub point_size_min_max: [f32; 2],
    /// Line width in pixels.
    pub line_width: f32,
}

fn polygon_type_fill(t: PolygonType) -> FillMode {
    match t {
        PolygonType::Points => FillMode::Point,
        PolygonType::Lines => FillMode::Line,
        _ => FillMode::Fill,
    }
}

/// Rounds to the nearest order of magnitude (xe::roundToNearestOrderOfMagnitude), used by Xenia for
/// the front slope scale.
fn round_to_nearest_order_of_magnitude(value: f32) -> f32 {
    if value == 0.0 {
        return value;
    }
    let v = f64::from(value);
    let order = 10f64.powf(v.abs().log10().floor());
    ((v / order).round() * order) as f32
}

/// Guest polygon offset (scale, offset) of the face most likely drawn (GetPreferredFacePolygonOffset).
pub fn preferred_face_polygon_offset(regs: &(impl RegisterSource + ?Sized), polygonal: bool) -> (f32, f32) {
    let m = PaSuScModeCntl::read(regs);
    let (mut scale, mut offset) = (0.0f32, 0.0f32);
    if polygonal {
        if m.poly_offset_front_enable && !m.cull_front {
            scale = round_to_nearest_order_of_magnitude(regs.reg_f32(r::PA_SU_POLY_OFFSET_FRONT_SCALE));
            offset = regs.reg_f32(r::PA_SU_POLY_OFFSET_FRONT_OFFSET);
        }
        if m.poly_offset_back_enable && !m.cull_back && scale == 0.0 && offset == 0.0 {
            scale = regs.reg_f32(r::PA_SU_POLY_OFFSET_BACK_SCALE);
            offset = regs.reg_f32(r::PA_SU_POLY_OFFSET_BACK_OFFSET);
        }
    } else if m.poly_offset_para_enable {
        scale = regs.reg_f32(r::PA_SU_POLY_OFFSET_FRONT_SCALE);
        offset = regs.reg_f32(r::PA_SU_POLY_OFFSET_FRONT_OFFSET);
    }
    (scale, offset)
}

/// Converts the guest absolute depth offset to a Direct3D 10-style integer bias
/// (GetD3D10IntegerPolygonOffset). Assumes unorm24 host depth for D24S8 and float32 for D24FS8.
pub fn d3d10_integer_polygon_offset(depth_format: DepthRenderTargetFormat, offset: f32) -> i32 {
    let is_float24 = depth_format == DepthRenderTargetFormat::D24FS8;
    let factor = if is_float24 { (1u32 << 24) as f32 / 8.0 } else { ((1u32 << 24) - 1) as f32 };
    let mut v = (offset.abs() * factor).ceil() as i32;
    if is_float24 {
        v <<= 3;
    }
    if offset < 0.0 { -v } else { v }
}

impl RasterizerState {
    /// Decodes the rasterizer state for a draw of `primitive_type`.
    pub fn from_registers(
        regs: &(impl RegisterSource + ?Sized),
        primitive_type: PrimitiveType,
        tessellation_enabled: bool,
    ) -> Self {
        let m = PaSuScModeCntl::read(regs);
        let polygonal = primitive_type.is_polygonal(tessellation_enabled);
        let clip = PaClClipCntl::read(regs);
        let (mut cull_mode, mut cull_all, mut fill_mode, mut mixed) = (CullMode::None, false, FillMode::Fill, false);
        if polygonal {
            cull_mode = match (m.cull_front, m.cull_back) {
                (true, true) => {
                    cull_all = true;
                    CullMode::None
                }
                (true, false) => CullMode::Front,
                (false, true) => CullMode::Back,
                (false, false) => CullMode::None,
            };
            if m.poly_mode == PolygonModeEnable::DualMode {
                let front = polygon_type_fill(m.polymode_front_ptype);
                let back = polygon_type_fill(m.polymode_back_ptype);
                fill_mode = if m.cull_front { back } else { front };
                mixed = !m.cull_front && !m.cull_back && front != back;
            }
        }
        let (guest_scale, guest_offset) = preferred_face_polygon_offset(regs, polygonal);
        let depth_format = RbDepthInfo::read(regs).depth_format;
        let depth_bias = DepthBias {
            guest_offset,
            guest_scale,
            constant: d3d10_integer_polygon_offset(depth_format, guest_offset),
            slope_scale: guest_scale * POLYGON_OFFSET_SCALE_SUBPIXEL_UNIT,
        };
        let mut user_clip_planes = [[0.0f32; 4]; 6];
        for (i, plane) in user_clip_planes.iter_mut().enumerate() {
            for (j, c) in plane.iter_mut().enumerate() {
                *c = regs.reg_f32(r::PA_CL_UCP_0_X + (i * 4 + j) as u32);
            }
        }
        let point = PaSuPointSize::read(regs);
        let minmax = PaSuPointMinmax::read(regs);
        let fixed_12_4_doubled = |v: u32| v as f32 / 16.0 * 2.0;
        Self {
            polygonal,
            cull_mode,
            cull_all,
            front_face: if m.face { FrontFace::Cw } else { FrontFace::Ccw },
            fill_mode,
            mixed_fill_modes: mixed,
            depth_bias,
            clip_disable: clip.clip_disable,
            user_clip_plane_mask: if clip.clip_disable { 0 } else { clip.ucp_ena },
            user_clip_planes,
            provoking_vertex_last: m.provoking_vtx_last,
            msaa_enable: m.msaa_enable,
            point_size: [fixed_12_4_doubled(point.width), fixed_12_4_doubled(point.height)],
            point_size_min_max: [fixed_12_4_doubled(minmax.min_size), fixed_12_4_doubled(minmax.max_size)],
            line_width: fixed_12_4_doubled(PaSuLineCntl::read(regs).width),
        }
    }
}

/// Whether the draw is a VIZ query survey (counts coverage only). `viz_queries_emulated` mirrors
/// Xenia's occlusion_query_viz option.
pub fn is_viz_survey_draw(regs: &(impl RegisterSource + ?Sized), viz_queries_emulated: bool) -> bool {
    let q = PaScVizQuery::read(regs);
    viz_queries_emulated && q.viz_query_ena && q.kill_pix_post_hi_z
}

/// Whether any samples may be rasterized (IsRasterizationPotentiallyDone).
pub fn is_rasterization_potentially_done(
    regs: &(impl RegisterSource + ?Sized),
    primitive_polygonal: bool,
    viz_queries_emulated: bool,
) -> bool {
    let mode = RbModecontrol::read(regs).edram_mode;
    if mode != EdramMode::ColorDepth && mode != EdramMode::DepthOnly {
        return false;
    }
    if SqProgramCntl::read(regs).vs_export_mode == VertexShaderExportMode::Multipass
        || RbSurfaceInfo::read(regs).surface_pitch == 0
    {
        return false;
    }
    if PaScVizQuery::read(regs).kill_pix_post_hi_z && !is_viz_survey_draw(regs, viz_queries_emulated) {
        return false;
    }
    if primitive_polygonal {
        let m = PaSuScModeCntl::read(regs);
        if m.cull_front && m.cull_back {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6 * b.abs().max(1.0)
    }

    fn d3d_viewport(w: f32, h: f32) -> ViewportRegisters {
        ViewportRegisters {
            clip_cntl: PaClClipCntl { dx_clip_space_def: true, ..Default::default() },
            vte_cntl: PaClVteCntl::decode(0x3F),
            mode_cntl: PaSuScModeCntl::default(),
            vtx_cntl: PaSuVtxCntl::default(),
            window_offset: PaScWindowOffset::default(),
            scale: [w / 2.0, -h / 2.0, 1.0],
            offset: [w / 2.0, h / 2.0, 0.0],
            depth_format: DepthRenderTargetFormat::D24S8,
            msaa_samples: MsaaSamples::X1,
            z_enable: true,
        }
    }

    #[test]
    fn standard_d3d_viewport_with_half_pixel() {
        let args = ViewportArgs { origin_bottom_left: false, ..ViewportArgs::webgpu(8192) };
        let info = host_viewport_info(&d3d_viewport(1280.0, 720.0), &args);
        assert_eq!(info.xy_offset, [0, 0]);
        assert_eq!(info.xy_extent, [1280, 720]);
        assert_eq!((info.z_min, info.z_max), (0.0, 1.0));
        assert!(approx(info.ndc_scale[0], 1.0) && approx(info.ndc_scale[1], -1.0));
        assert!(approx(info.ndc_offset[0], 1.0 / 1280.0) && approx(info.ndc_offset[1], 1.0 / 720.0));
        assert_eq!((info.ndc_scale[2], info.ndc_offset[2]), (1.0, 0.0));

        // WebGPU (origin bottom-left) flips Y.
        let info = host_viewport_info(&d3d_viewport(1280.0, 720.0), &ViewportArgs::webgpu(8192));
        assert!(approx(info.ndc_scale[1], 1.0) && approx(info.ndc_offset[1], -1.0 / 720.0));

        // Without the half-pixel offset (OpenGL pixel center) the transform is identity.
        let mut v = d3d_viewport(1280.0, 720.0);
        v.vtx_cntl.pix_center = PixelCenter::OglHalf;
        let info = host_viewport_info(&v, &args);
        assert_eq!(info.ndc_offset[0], 0.0);
        assert_eq!(info.ndc_scale[0], 1.0);
    }

    #[test]
    fn opengl_clip_space_and_reverse_z() {
        let mut v = d3d_viewport(640.0, 480.0);
        v.clip_cntl.dx_clip_space_def = false;
        v.scale[2] = 0.5;
        v.offset[2] = 0.5;
        let info = host_viewport_info(&v, &ViewportArgs::webgpu(8192));
        assert_eq!((info.z_min, info.z_max), (0.0, 1.0));
        assert_eq!((info.ndc_scale[2], info.ndc_offset[2]), (0.5, 0.5));

        let mut v = d3d_viewport(640.0, 480.0);
        v.scale[2] = -1.0;
        v.offset[2] = 1.0;
        let info = host_viewport_info(&v, &ViewportArgs::webgpu(8192));
        assert_eq!((info.z_min, info.z_max), (0.0, 1.0));
        assert_eq!((info.ndc_scale[2], info.ndc_offset[2]), (-1.0, 1.0));
        let info = host_viewport_info(&v, &ViewportArgs { allow_reverse_z: true, ..ViewportArgs::webgpu(8192) });
        assert_eq!((info.z_min, info.z_max), (1.0, 0.0));
        assert_eq!((info.ndc_scale[2], info.ndc_offset[2]), (1.0, 0.0));

        // Partial depth range.
        let mut v = d3d_viewport(640.0, 480.0);
        v.scale[2] = 0.5;
        v.offset[2] = 0.25;
        let info = host_viewport_info(&v, &ViewportArgs::webgpu(8192));
        assert_eq!((info.z_min, info.z_max), (0.25, 0.75));
        let info =
            host_viewport_info(&v, &ViewportArgs { pixel_shader_writes_depth: true, ..ViewportArgs::webgpu(8192) });
        assert_eq!((info.z_min, info.z_max), (0.0, 1.0));
    }

    #[test]
    fn clip_disabled_screen_space() {
        let mut v = d3d_viewport(0.0, 0.0);
        v.clip_cntl.clip_disable = true;
        v.vte_cntl = PaClVteCntl::default();
        v.mode_cntl.vtx_window_offset_enable = true;
        v.window_offset = PaScWindowOffset { window_x_offset: 16, window_y_offset: -8 };
        let args = ViewportArgs { origin_bottom_left: false, ..ViewportArgs::webgpu(16384) };
        let info = host_viewport_info(&v, &args);
        assert_eq!(info.xy_offset, [0, 0]);
        assert_eq!(info.xy_extent, [8192, 8192]);
        assert!(approx(info.ndc_scale[0], 2.0 / 8192.0));
        assert!(approx(info.ndc_offset[0], (16.5 - 4096.0) * 2.0 / 8192.0));
        assert!(approx(info.ndc_offset[1], (-7.5 - 4096.0) * 2.0 / 8192.0));
        assert_eq!((info.ndc_scale[2], info.ndc_offset[2], info.z_min, info.z_max), (1.0, 0.0, 0.0, 1.0));
        // A pixel at (100, 50) (pre-divided, w = 1) lands at host pixel (116.5, 42.5).
        let px = (100.0 * info.ndc_scale[0] + info.ndc_offset[0] + 1.0) * 0.5 * 8192.0;
        let py = (50.0 * info.ndc_scale[1] + info.ndc_offset[1] + 1.0) * 0.5 * 8192.0;
        assert!(approx(px, 116.5) && approx(py, 42.5), "{px} {py}");
        // Window offset folded into EDRAM: not applied to vertices.
        let info = host_viewport_info(&v, &ViewportArgs { window_offset_in_edram: true, ..args });
        assert!(approx(info.ndc_offset[0], (0.5 - 4096.0) * 2.0 / 8192.0));
    }

    #[test]
    fn window_offset_negative_tile() {
        // Predicated tiling: right-hand tile drawn with x window offset -608.
        let mut v = d3d_viewport(1280.0, 720.0);
        v.mode_cntl.vtx_window_offset_enable = true;
        v.window_offset = PaScWindowOffset { window_x_offset: -608, window_y_offset: 0 };
        let args = ViewportArgs { origin_bottom_left: false, ..ViewportArgs::webgpu(8192) };
        let info = host_viewport_info(&v, &args);
        // offset_axis = 640 - 608 + 0.5 = 32.5; range [-607.5, 672.5] -> floor -> [-608, 672].
        assert_eq!(info.xy_offset[0], -608);
        assert_eq!(info.xy_extent[0], 1280);
        assert!(approx(info.ndc_scale[0], 1.0));
        assert!(approx(info.ndc_offset[0], 1.0 / 1280.0));
        // Cropping for WebGPU to a 672-wide target keeps the mapping of positions.
        let cropped = info.crop_to_render_target(672, 720, false);
        assert_eq!((cropped.xy_offset[0], cropped.xy_extent[0]), (0, 672));
        for ndc in [-1.0f32, -0.3, 0.0, 0.7, 1.0] {
            let orig = info.xy_offset[0] as f32
                + (ndc * info.ndc_scale[0] + info.ndc_offset[0] + 1.0) * 0.5 * info.xy_extent[0] as f32;
            let new = cropped.xy_offset[0] as f32
                + (ndc * cropped.ndc_scale[0] + cropped.ndc_offset[0] + 1.0) * 0.5 * cropped.xy_extent[0] as f32;
            assert!((orig - new).abs() < 1e-3, "{ndc}: {orig} vs {new}");
        }
    }

    #[test]
    fn crop_with_flipped_y() {
        let v = d3d_viewport(1280.0, 720.0);
        let info = host_viewport_info(&v, &ViewportArgs::webgpu(8192));
        let cropped = info.crop_to_render_target(1280, 600, true);
        assert_eq!(cropped.xy_extent, [1280, 600]);
        for ndc in [-1.0f32, -0.25, 0.5, 1.0] {
            // Host Y-up NDC: pixel_y = offset + (1 - ndc_host) / 2 * extent.
            let host = ndc * info.ndc_scale[1] + info.ndc_offset[1];
            let orig = info.xy_offset[1] as f32 + (1.0 - host) * 0.5 * info.xy_extent[1] as f32;
            let host_c = ndc * cropped.ndc_scale[1] + cropped.ndc_offset[1];
            let new = cropped.xy_offset[1] as f32 + (1.0 - host_c) * 0.5 * cropped.xy_extent[1] as f32;
            assert!((orig - new).abs() < 1e-3, "{ndc}: {orig} vs {new}");
        }
        let empty = info.crop_to_render_target(0, 600, true);
        assert_eq!(empty.xy_extent[0], 0);
    }

    #[test]
    fn empty_viewport() {
        let mut v = d3d_viewport(1280.0, 720.0);
        v.offset[0] = -5000.0;
        let info = host_viewport_info(&v, &ViewportArgs::webgpu(8192));
        assert_eq!(info.xy_extent[0], 0);
        assert_eq!((info.ndc_scale[0], info.ndc_offset[0]), (1.0, 0.0));
    }

    #[test]
    fn float24_depth_bounds() {
        let mut v = d3d_viewport(64.0, 64.0);
        v.depth_format = DepthRenderTargetFormat::D24FS8;
        v.scale[2] = 0.3;
        let args =
            ViewportArgs { convert_z_to_float24: true, full_float24_in_0_to_1: true, ..ViewportArgs::webgpu(8192) };
        let info = host_viewport_info(&v, &args);
        assert_eq!(info.z_min, 0.0);
        assert_eq!(info.z_max, float20e4_to_32(float32_to_20e4(0.3, true)) * 0.5);
    }

    #[test]
    fn resolution_scale_and_msaa_sample_offset() {
        let mut v = d3d_viewport(1280.0, 720.0);
        v.msaa_samples = MsaaSamples::X4;
        // UE3-style sample 0 alignment: (-0.125, -0.125) added to the offsets.
        v.offset[0] -= 0.125;
        v.offset[1] -= 0.125;
        let args = ViewportArgs {
            draw_resolution_scale_x: 2,
            draw_resolution_scale_y: 2,
            xenos_sample_positions: true,
            origin_bottom_left: false,
            ..ViewportArgs::webgpu(16384)
        };
        let info = host_viewport_info(&v, &args);
        assert_eq!(info.xy_extent, [2560, 1440]);
        // Fraction -0.125 matched sample 0 (-2/16), halved for 2x scale: 640.375 + 0.0625.
        assert!(approx(info.ndc_offset[0], (640.4375 - 640.0) * 2.0 / 1280.0), "{}", info.ndc_offset[0]);
    }

    fn scissor_regs(tl: (u32, u32, bool), br: (u32, u32), wo: (i32, i32), pitch: u32) -> Vec<u32> {
        let mut regs = vec![0u32; 0x5003];
        regs[r::PA_SC_WINDOW_SCISSOR_TL as usize] =
            PaScWindowScissorTl { tl_x: tl.0, tl_y: tl.1, window_offset_disable: tl.2 }.encode();
        regs[r::PA_SC_WINDOW_SCISSOR_BR as usize] = PaScWindowScissorBr { br_x: br.0, br_y: br.1 }.encode();
        regs[r::PA_SC_WINDOW_OFFSET as usize] =
            PaScWindowOffset { window_x_offset: wo.0, window_y_offset: wo.1 }.encode();
        regs[r::PA_SC_SCREEN_SCISSOR_TL as usize] = 0;
        regs[r::PA_SC_SCREEN_SCISSOR_BR as usize] = PaScScreenScissor { x: 8192, y: 8192 }.encode();
        regs[r::RB_SURFACE_INFO as usize] = pitch;
        regs
    }

    #[test]
    fn scissor_cases() {
        let regs = scissor_regs((0, 0, false), (1280, 720), (0, 0), 1280);
        assert_eq!(scissor(&regs, true, false), Scissor { offset: [0, 0], extent: [1280, 720] });
        // Window offset applies to the scissor, negative part dropped.
        let regs = scissor_regs((0, 0, false), (1280, 720), (-608, -16), 1280);
        assert_eq!(scissor(&regs, true, false), Scissor { offset: [0, 0], extent: [672, 704] });
        // Folded into EDRAM: moved back by the offset.
        let s = scissor(&regs, true, true);
        assert_eq!(s, Scissor { offset: [608, 16], extent: [672, 704] });
        // window_offset_disable.
        let regs = scissor_regs((8, 8, true), (100, 100), (50, 50), 1280);
        assert_eq!(scissor(&regs, true, false), Scissor { offset: [8, 8], extent: [92, 92] });
        // Surface pitch clamp.
        let regs = scissor_regs((0, 0, false), (8192, 8192), (0, 0), 640);
        assert_eq!(scissor(&regs, true, false).extent, [640, 8192]);
        assert_eq!(scissor(&regs, false, false).extent, [8192, 8192]);
        // Inverted collapses to empty.
        let regs = scissor_regs((100, 100, false), (50, 50), (0, 0), 1280);
        assert_eq!(scissor(&regs, true, false), Scissor { offset: [100, 100], extent: [0, 0] });
        // Screen scissor.
        let mut regs = scissor_regs((0, 0, false), (1280, 720), (0, 0), 1280);
        regs[r::PA_SC_SCREEN_SCISSOR_TL as usize] = PaScScreenScissor { x: 10, y: 20 }.encode();
        regs[r::PA_SC_SCREEN_SCISSOR_BR as usize] = PaScScreenScissor { x: 500, y: 400 }.encode();
        assert_eq!(scissor(&regs, true, false), Scissor { offset: [10, 20], extent: [490, 380] });
        assert_eq!(
            Scissor { offset: [10, 20], extent: [490, 380] }.clamp_to(300, 300),
            Scissor { offset: [10, 20], extent: [290, 280] }
        );
    }

    #[test]
    fn rasterizer_state() {
        let mut regs = vec![0u32; 0x5003];
        regs[r::PA_SU_SC_MODE_CNTL as usize] = PaSuScModeCntl {
            cull_back: true,
            face: true,
            poly_mode: PolygonModeEnable::DualMode,
            polymode_front_ptype: PolygonType::Lines,
            polymode_back_ptype: PolygonType::Points,
            poly_offset_front_enable: true,
            provoking_vtx_last: true,
            ..Default::default()
        }
        .encode();
        regs[r::PA_SU_POLY_OFFSET_FRONT_SCALE as usize] = 16.0f32.to_bits(); // Rounded to 20.
        regs[r::PA_SU_POLY_OFFSET_FRONT_OFFSET as usize] = 0.000002f32.to_bits();
        regs[r::RB_DEPTH_INFO as usize] = 1 << 16;
        regs[r::PA_SU_POINT_SIZE as usize] = (24 << 16) | 8;
        regs[r::PA_SU_LINE_CNTL as usize] = 8;
        regs[r::PA_CL_CLIP_CNTL as usize] = 0b11;
        regs[r::PA_CL_UCP_1_W as usize] = 3.0f32.to_bits();
        let s = RasterizerState::from_registers(&regs, PrimitiveType::TriangleList, false);
        assert!(s.polygonal && !s.cull_all && !s.mixed_fill_modes);
        assert_eq!(s.cull_mode, CullMode::Back);
        assert_eq!(s.front_face, FrontFace::Cw);
        assert_eq!(s.fill_mode, FillMode::Line);
        assert_eq!(s.depth_bias.slope_scale, 1.25);
        // float24: ceil(2e-6 * 2^21) = 5, in float32 ULPs * 8.
        assert_eq!(s.depth_bias.constant, 40);
        assert_eq!(s.point_size, [3.0, 1.0]);
        assert_eq!(s.line_width, 1.0);
        assert_eq!(s.user_clip_plane_mask, 0b11);
        assert_eq!(s.user_clip_planes[1][3], 3.0);
        assert!(s.provoking_vertex_last);

        // Lines are not polygonal: no culling, para offset only.
        let s = RasterizerState::from_registers(&regs, PrimitiveType::LineStrip, false);
        assert_eq!((s.cull_mode, s.fill_mode, s.depth_bias.constant), (CullMode::None, FillMode::Fill, 0));
        assert_eq!(d3d10_integer_polygon_offset(DepthRenderTargetFormat::D24S8, -1.0 / 16_777_215.0), -1);
    }

    #[test]
    fn rasterization_done() {
        let mut regs = vec![0u32; 0x5003];
        assert!(!is_rasterization_potentially_done(&regs, true, false));
        regs[r::RB_MODECONTROL as usize] = 4;
        regs[r::RB_SURFACE_INFO as usize] = 1280;
        assert!(is_rasterization_potentially_done(&regs, true, false));
        regs[r::PA_SU_SC_MODE_CNTL as usize] = 3;
        assert!(!is_rasterization_potentially_done(&regs, true, false));
        assert!(is_rasterization_potentially_done(&regs, false, false));
        regs[r::PA_SC_VIZ_QUERY as usize] = 1 | (1 << 7);
        assert!(!is_rasterization_potentially_done(&regs, false, false));
        assert!(is_rasterization_potentially_done(&regs, false, true));
        assert!(is_viz_survey_draw(&regs, true));
    }
}
