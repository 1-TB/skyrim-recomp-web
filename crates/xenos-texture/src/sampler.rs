//! Sampler state from texture fetch constants mapped to WebGPU sampler descriptors.

use crate::fetch::{AnisoFilter, ClampMode, TextureFetchConstant, TextureFilter};
use crate::layout::TextureSubresources;

/// `GPUAddressMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// `GPUFilterMode` / `GPUMipmapFilterMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Filter and anisotropy overrides from a fetch instruction (`UseFetchConst` defers to the
/// fetch constant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FetchInstructionFilters {
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mip_filter: TextureFilter,
    pub aniso_filter: AnisoFilter,
}

impl Default for FetchInstructionFilters {
    fn default() -> Self {
        FetchInstructionFilters {
            mag_filter: TextureFilter::UseFetchConst,
            min_filter: TextureFilter::UseFetchConst,
            mip_filter: TextureFilter::UseFetchConst,
            aniso_filter: AnisoFilter::UseFetchConst,
        }
    }
}

/// Reasons a host sampler cannot reproduce the guest sampler exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplerLoss {
    /// WebGPU has no border addressing; clamped to edge (border color must be emulated in the
    /// shader if needed, see `SamplerDescriptor::border_color`).
    BorderClampToEdge { axis: u8 },
    /// Mirror-once modes approximated with mirror-repeat.
    MirrorOnceAsMirrorRepeat { axis: u8 },
    /// GL_CLAMP-style "halfway" clamping approximated with clamp-to-edge.
    HalfwayAsClampToEdge { axis: u8 },
    /// Fetch constant filter value not meaningful there (e.g. BaseMap for min/mag).
    InvalidFilterAsNearest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDescriptor {
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    /// 1 when anisotropic filtering is disabled (WebGPU requires all-linear filters for > 1).
    pub max_anisotropy: u16,
    /// Border color when any axis uses border addressing.
    pub border_color: Option<[f32; 4]>,
    /// Guest LOD bias, which WebGPU samplers lack: apply in the shader (`textureSampleBias`).
    pub lod_bias: f32,
    pub losses: Vec<SamplerLoss>,
}

fn map_clamp(mode: ClampMode, axis: u8, losses: &mut Vec<SamplerLoss>) -> AddressMode {
    match mode {
        ClampMode::Repeat => AddressMode::Repeat,
        ClampMode::MirroredRepeat => AddressMode::MirrorRepeat,
        ClampMode::ClampToEdge => AddressMode::ClampToEdge,
        ClampMode::MirrorClampToEdge | ClampMode::MirrorClampToHalfway | ClampMode::MirrorClampToBorder => {
            losses.push(SamplerLoss::MirrorOnceAsMirrorRepeat { axis });
            AddressMode::MirrorRepeat
        }
        ClampMode::ClampToHalfway => {
            losses.push(SamplerLoss::HalfwayAsClampToEdge { axis });
            AddressMode::ClampToEdge
        }
        ClampMode::ClampToBorder => {
            losses.push(SamplerLoss::BorderClampToEdge { axis });
            AddressMode::ClampToEdge
        }
    }
}

fn resolve(instr: TextureFilter, fetch: TextureFilter) -> TextureFilter {
    if instr == TextureFilter::UseFetchConst { fetch } else { instr }
}

fn min_mag(filter: TextureFilter, losses: &mut Vec<SamplerLoss>) -> FilterMode {
    match filter {
        TextureFilter::Linear => FilterMode::Linear,
        TextureFilter::Point => FilterMode::Nearest,
        _ => {
            losses.push(SamplerLoss::InvalidFilterAsNearest);
            FilterMode::Nearest
        }
    }
}

/// Builds the host sampler for a fetch constant and fetch instruction filters. LOD clamps
/// are in host mip levels (host textures hold levels `0..=mip_max_level`).
pub fn sampler_descriptor(fetch: &TextureFetchConstant, instr: &FetchInstructionFilters) -> SamplerDescriptor {
    let mut losses = Vec::new();
    let clamp = fetch.clamp_modes_for_dimension();
    let address_mode_u = map_clamp(clamp[0], 0, &mut losses);
    let address_mode_v = map_clamp(clamp[1], 1, &mut losses);
    let address_mode_w = map_clamp(clamp[2], 2, &mut losses);
    let border_color = clamp.iter().any(|c| c.uses_border()).then(|| {
        let mut c = fetch.border_color.rgba();
        if fetch.force_bc_w_to_max {
            c[3] = 1.0;
        }
        c
    });

    let sub = TextureSubresources::from_fetch(fetch);
    let mag = resolve(instr.mag_filter, fetch.mag_filter);
    let min = resolve(instr.min_filter, fetch.min_filter);
    let mip = resolve(instr.mip_filter, fetch.mip_filter);
    let aniso = resolve_aniso(instr.aniso_filter, fetch.aniso_filter);
    let mip_base_map = mip == TextureFilter::BaseMap;
    let mut mip_min_level = sub.mip_min_level;
    if mip_base_map && sub.base_page != 0 {
        mip_min_level = 0;
    }

    let (mag_filter, min_filter, mipmap_filter, max_anisotropy) = match aniso.max_anisotropy() {
        Some(n) if n > 1 && !mip_base_map => (FilterMode::Linear, FilterMode::Linear, FilterMode::Linear, n),
        _ => {
            let mipmap = match mip {
                TextureFilter::Linear => FilterMode::Linear,
                TextureFilter::Point | TextureFilter::BaseMap => FilterMode::Nearest,
                TextureFilter::UseFetchConst => {
                    losses.push(SamplerLoss::InvalidFilterAsNearest);
                    FilterMode::Nearest
                }
            };
            (min_mag(mag, &mut losses), min_mag(min, &mut losses), mipmap, 1)
        }
    };
    let lod_min_clamp = mip_min_level as f32;
    // Base-map sampling: clamp to the first level while keeping the magnification vs.
    // minification decision (Vulkan/Direct3D GL_NEAREST emulation trick).
    let lod_max_clamp = if mip_base_map { lod_min_clamp + 0.25 } else { sub.mip_max_level.max(mip_min_level) as f32 };
    SamplerDescriptor {
        address_mode_u,
        address_mode_v,
        address_mode_w,
        mag_filter,
        min_filter,
        mipmap_filter,
        lod_min_clamp,
        lod_max_clamp,
        max_anisotropy,
        border_color,
        lod_bias: fetch.lod_bias_f32(),
        losses,
    }
}

fn resolve_aniso(instr: AnisoFilter, fetch: AnisoFilter) -> AnisoFilter {
    let a = if instr == AnisoFilter::UseFetchConst { fetch } else { instr };
    a.min(AnisoFilter::Max16to1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::{BorderColor, DataDimension};
    use crate::format::TextureFormat;

    fn fetch() -> TextureFetchConstant {
        let mut f = TextureFetchConstant::new_2d(TextureFormat::k_DXT1, 256, 256);
        f.base_page = 0x100;
        f.mip_page = 0x140;
        f.mip_max_level = 8;
        f
    }

    #[test]
    fn basic_trilinear() {
        let mut f = fetch();
        f.mip_filter = TextureFilter::Linear;
        f.lod_bias = 16;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(
            (s.mag_filter, s.min_filter, s.mipmap_filter),
            (FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
        );
        assert_eq!((s.lod_min_clamp, s.lod_max_clamp), (0.0, 8.0));
        assert_eq!(s.address_mode_u, AddressMode::Repeat);
        assert_eq!(s.address_mode_w, AddressMode::ClampToEdge);
        assert_eq!(s.lod_bias, 0.5);
        assert!(s.losses.is_empty());
        assert_eq!(s.border_color, None);
    }

    #[test]
    fn instruction_overrides_and_aniso() {
        let mut f = fetch();
        f.aniso_filter = AnisoFilter::Max8to1;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(s.max_anisotropy, 8);
        assert_eq!(s.mipmap_filter, FilterMode::Linear);
        let instr = FetchInstructionFilters {
            mag_filter: TextureFilter::Point,
            min_filter: TextureFilter::Point,
            mip_filter: TextureFilter::Point,
            aniso_filter: AnisoFilter::Disabled,
        };
        let s = sampler_descriptor(&f, &instr);
        assert_eq!(s.max_anisotropy, 1);
        assert_eq!((s.mag_filter, s.mipmap_filter), (FilterMode::Nearest, FilterMode::Nearest));
        f.aniso_filter = AnisoFilter::Max1to1;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(s.max_anisotropy, 1);
    }

    #[test]
    fn base_map_and_lod_clamps() {
        let mut f = fetch();
        f.mip_filter = TextureFilter::BaseMap;
        f.mip_min_level = 2;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!((s.lod_min_clamp, s.lod_max_clamp), (0.0, 0.25));
        assert_eq!(s.mipmap_filter, FilterMode::Nearest);
        f.mip_filter = TextureFilter::Point;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!((s.lod_min_clamp, s.lod_max_clamp), (2.0, 8.0));
        // No base: min level becomes at least 1.
        f.base_page = 0;
        f.mip_min_level = 0;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(s.lod_min_clamp, 1.0);
    }

    #[test]
    fn address_modes_and_losses() {
        let mut f = fetch();
        f.dimension = DataDimension::D3;
        f.clamp = [ClampMode::ClampToBorder, ClampMode::MirrorClampToEdge, ClampMode::ClampToHalfway];
        f.border_color = BorderColor::AbgrWhite;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(
            (s.address_mode_u, s.address_mode_v, s.address_mode_w),
            (AddressMode::ClampToEdge, AddressMode::MirrorRepeat, AddressMode::ClampToEdge)
        );
        assert_eq!(s.border_color, Some([1.0; 4]));
        assert_eq!(
            s.losses,
            vec![
                SamplerLoss::BorderClampToEdge { axis: 0 },
                SamplerLoss::MirrorOnceAsMirrorRepeat { axis: 1 },
                SamplerLoss::HalfwayAsClampToEdge { axis: 2 }
            ]
        );
        f.clamp = [ClampMode::MirrorClampToBorder, ClampMode::Repeat, ClampMode::Repeat];
        f.border_color = BorderColor::AbgrBlack;
        f.force_bc_w_to_max = true;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(s.border_color, Some([0.0, 0.0, 0.0, 1.0]));
        f.dimension = DataDimension::Cube;
        let s = sampler_descriptor(&f, &FetchInstructionFilters::default());
        assert_eq!(s.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(s.border_color, None);
    }
}
