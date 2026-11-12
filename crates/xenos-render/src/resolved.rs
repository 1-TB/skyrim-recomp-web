//! Resolves kept as textures (`--gpu d3d`).
//!
//! A resolve normally converts the surface into the GPU copy of guest memory (two compute
//! dispatches), and the texture the title then samples is loaded from there (a third): three
//! passes over the picture for what is, most of the time, a plain copy. Skyrim resolves some
//! thirty times a frame, nearly all of them an 8-bit RGBA surface into an 8-bit RGBA texture of
//! the same size, and those passes took a third of the frame's GPU time.
//!
//! Such a resolve is instead one copy of the surface's rectangle into a texture kept per
//! destination ([`ResolvedTextures::resolve`]), which the texture cache binds as the guest
//! texture ([`ResolvedTextures::lookup`]): the red/blue swap and the byte orders the resolve
//! and the texture disagree on only permute the components, which the view's swizzle does.
//! Later resolves into the destination update the texture in place, so the texture stays bound.
//!
//! Depth resolves (the depth buffer as a 24:8 texture) are kept the same way, as two textures a
//! small pass fills from the depth surface: the texels' words as bytes, for textures reading
//! that memory as 8-bit RGBA (Skyrim draws its depth texture into a color surface over the depth
//! buffer's EDRAM to restore the depth buffer), and the depth a 24:8 texture of it loads as.
//!
//! The GPU copy of guest memory isn't written, only marked as if it were. What else would read
//! it there (a texture of another shape or format over the same memory, a resolve that overlaps
//! it) first has the kept resolves stored the usual way, from the texture
//! ([`ResolvedTextures::store`]), so it sees what it would have without any of this.

use gpu_webgpu::*;
use xenos_edram::{DestKey, EdramCache, ExternalSource, ExternalViews, SharedMemoryTarget};
use xenos_state::resolve::ResolveEdramInfo;
use xenos_state::resolve::ResolveInfo;
use xenos_state::xenos::{
    CopyCommand, CopySampleSelect, MsaaSamples, SurfaceNumberFormat, TextureFormat as GuestFormat,
};

/// Frames a destination is kept without being resolved to or bound.
const MAX_IDLE_FRAMES: u64 = 600;

/// Times the textures bound of a destination may change size before it is left to the usual
/// path (two textures of different sizes over the same memory would replace it every frame).
const MAX_RESIZES: u32 = 4;

/// What a destination's textures hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A color surface's pixels as they are; the resolves swap red and blue storing them.
    Color { swap: bool },
    /// A depth surface's EDRAM words as bytes, and the depth a texture of them loads as
    /// (`float24`: D24FS8, else D24S8).
    Depth { float24: bool },
}

/// A resolve [`ResolvedTextures::resolve`] can do, from [`plan`] or [`plan_depth`].
pub(crate) struct Plan {
    key: DestKey,
    /// Where the pixels go in the destination (x, y, width, height).
    rect: [u32; 4],
    /// The rectangle's origin in the surface (host pixels).
    source: [u32; 2],
    kind: Kind,
}

/// `info` as a copy into a destination's texture, if it is one: the whole rectangle of a
/// single-sampled 8-bit RGBA surface `surface` (its size) into an 8-bit RGBA 2D destination,
/// with nothing converted. `offset` is where EDRAM's origin is in the surface (see
/// `xenos_edram::ExternalSource::offset`).
pub(crate) fn plan(info: &ResolveInfo, surface: (u32, u32), offset: [u32; 2]) -> Option<Plan> {
    let source = info.source_edram()?;
    let dest = info.copy_dest_info;
    // The byte orders of 32-bit words: none, 8 in 16, 8 in 32, 16 in 32.
    let endian = dest.copy_dest_endian.raw();
    let plain = !info.is_copying_depth()
        && source.format == 0
        && matches!(info.copy_control.copy_command, CopyCommand::Raw | CopyCommand::Convert)
        && info.dest_format == GuestFormat::K8888
        && !dest.copy_dest_array
        && dest.copy_dest_exp_bias == 0
        && dest.copy_dest_number == SurfaceNumberFormat::UnsignedRepeatingFraction
        && endian <= 3
        && info.copy_dest_extent_length != 0
        && !info.rect.is_empty();
    if !plain {
        return None;
    }
    let rect = [info.dest_offset_x_div_8 << 3, info.dest_offset_y_div_8 << 3, info.rect.width, info.rect.height];
    let from = [(info.edram_offset_x_div_8 << 3) + offset[0], (info.edram_offset_y_div_8 << 3) + offset[1]];
    let key = DestKey {
        base: info.copy_dest_base & 0x1FFF_FFFF,
        pitch: info.dest_pitch_aligned,
        height: info.dest_height_aligned,
        bytes_per_element: 4,
        endian,
    };
    let fits = from[0] + rect[2] <= surface.0
        && from[1] + rect[3] <= surface.1
        && rect[0] + rect[2] <= key.pitch
        && rect[1] + rect[3] <= key.height;
    fits.then_some(Plan { key, rect, source: from, kind: Kind::Color { swap: dest.copy_dest_swap } })
}

/// `info` as a copy of a depth surface into a destination's textures, if it is one: the whole
/// rectangle of a single-sampled depth surface `surface` (its size; `float24`: D24FS8) into a
/// 24:8 2D destination of its format. `offset` as for [`plan`].
pub(crate) fn plan_depth(info: &ResolveInfo, surface: (u32, u32), offset: [u32; 2], float24: bool) -> Option<Plan> {
    let source = info.source_edram()?;
    let dest = info.copy_dest_info;
    let endian = dest.copy_dest_endian.raw();
    let format = if float24 { GuestFormat::K248Float } else { GuestFormat::K248 };
    let plain = info.is_copying_depth()
        && source.format == u32::from(float24)
        && info.copy_control.copy_command == CopyCommand::Raw
        && info.dest_format == format
        && !dest.copy_dest_array
        && endian <= 3
        && info.copy_dest_extent_length != 0
        && !info.rect.is_empty();
    if !plain {
        return None;
    }
    let rect = [info.dest_offset_x_div_8 << 3, info.dest_offset_y_div_8 << 3, info.rect.width, info.rect.height];
    let from = [(info.edram_offset_x_div_8 << 3) + offset[0], (info.edram_offset_y_div_8 << 3) + offset[1]];
    let key = DestKey {
        base: info.copy_dest_base & 0x1FFF_FFFF,
        pitch: info.dest_pitch_aligned,
        height: info.dest_height_aligned,
        bytes_per_element: 4,
        endian,
    };
    let fits = from[0] + rect[2] <= surface.0
        && from[1] + rect[3] <= surface.1
        && rect[0] + rect[2] <= key.pitch
        && rect[1] + rect[3] <= key.height;
    fits.then_some(Plan { key, rect, source: from, kind: Kind::Depth { float24 } })
}

/// A destination's textures for the texture cache.
pub struct Resolved {
    /// The destination's texels as 8-bit RGBA.
    pub texture: Texture,
    /// The resolves' red/blue swap and byte order (`Endian128`), which with the texture's own
    /// byte order give the order its components are read in (see [`swizzle`]).
    pub swap: bool,
    pub endian: u32,
    /// For a depth destination: the depth a 24:8 texture (floating-point if the flag is set)
    /// with the resolves' byte order loads, as a 32-bit float texture.
    pub depth: Option<(Texture, bool)>,
    /// The guest memory the resolves were to write (start, length): the texture stands for it
    /// only while that is still marked as written by the GPU.
    pub extent: (u32, u32),
}

/// The components a texture of byte order `fetch_endian` (0-3: none, 8 in 16, 8 in 32, 16 in
/// 32) reads of a destination resolved with `swap` and byte order `endian`, in terms of the
/// surface's: component `i` is the surface's component `swizzle[i]`. The resolve packs the
/// surface's components into a word (red lowest, after its swap) and stores it in its byte
/// order; the texture reads the word in its own and takes its bytes in memory order.
pub fn swizzle(swap: bool, endian: u32, fetch_endian: u32) -> [u8; 4] {
    let reorder = |bytes: [u8; 4], endian: u32| match endian {
        1 => [bytes[1], bytes[0], bytes[3], bytes[2]],
        2 => [bytes[3], bytes[2], bytes[1], bytes[0]],
        3 => [bytes[2], bytes[3], bytes[0], bytes[1]],
        _ => bytes,
    };
    let packed = if swap { [2, 1, 0, 3] } else { [0, 1, 2, 3] };
    reorder(reorder(packed, endian), fetch_endian)
}

struct Dest {
    key: DestKey,
    kind: Kind,
    /// What textures bind: as large as the texture last asked for (`wanted`), or as the first
    /// resolve's rectangle (the whole destination for depth) until one is.
    texture: Texture,
    width: u32,
    height: u32,
    /// A depth destination's depth texture and the two as the copy pass's targets.
    depth: Option<(Texture, [TextureView; 2])>,
    /// The whole destination, kept as well once a resolve writes outside `texture`: what
    /// [`ResolvedTextures::store`] then reads.
    full: Option<Texture>,
    /// Views of `texture` and `full` for the resolve shaders (made when first stored from).
    views: [Option<ExternalViews>; 2],
    /// Rectangles of `texture` resolves filled (x, y, width, height).
    rects: Vec<[u32; 4]>,
    /// Resolves not in the GPU copy of guest memory yet, oldest first, each with where its
    /// pixels are in the destination.
    unstored: Vec<(ResolveInfo, [u32; 4])>,
    /// Guest memory the resolves were to write: first and last byte's end.
    extent: (u32, u32),
    /// The size of the texture last looked up, when not `texture`'s: the next resolve makes
    /// `texture` that size.
    wanted: Option<(u32, u32)>,
    resizes: u32,
    last_used: u64,
}

impl Dest {
    fn extent(&self) -> (u32, u32) {
        (self.extent.0, self.extent.1 - self.extent.0)
    }

    fn overlaps(&self, span: (u32, u32)) -> bool {
        xenos_edram::spans_overlap(self.key.span(), span) || xenos_edram::spans_overlap(self.extent(), span)
    }
}

fn texture(device: &Device, width: u32, height: u32) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: "d3d resolved texture",
        size: extent(width, height, 1),
        format: TextureFormat::RGBA8Unorm,
        usage: TextureUsage::TextureBinding | TextureUsage::CopyDst | TextureUsage::CopySrc,
        ..Default::default()
    })
}

/// A depth destination's textures: the bytes (as [`texture`]), and the depth with both as the
/// copy pass's targets.
fn depth_textures(device: &Device, width: u32, height: u32) -> (Texture, (Texture, [TextureView; 2])) {
    let make = |format| {
        device.create_texture(&TextureDescriptor {
            label: "d3d resolved depth",
            size: extent(width, height, 1),
            format,
            usage: TextureUsage::TextureBinding | TextureUsage::RenderAttachment,
            ..Default::default()
        })
    };
    let [bytes, depth] = xenos_edram::DEPTH_COPY_FORMATS.map(make);
    let views = [bytes.create_view(None), depth.create_view(None)];
    (bytes, (depth, views))
}

fn inside(rect: [u32; 4], width: u32, height: u32) -> bool {
    rect[0] + rect[2] <= width && rect[1] + rect[3] <= height
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResolvedStats {
    /// Resolves copied into a destination's texture.
    pub copies: u64,
    /// Of those, also into the whole destination's.
    pub full_copies: u64,
    /// Resolves stored into the GPU copy of guest memory after all.
    pub stored: u64,
    /// Destinations' textures made (first use or another size).
    pub textures: u64,
    pub dropped: u64,
}

#[derive(Default)]
pub(crate) struct ResolvedTextures {
    dests: Vec<Dest>,
    /// Destinations left to the usual path (see [`MAX_RESIZES`]).
    unsuited: Vec<DestKey>,
    /// Textures no destination uses anymore: destroyed two frame ends later, when nothing
    /// recorded or shown uses them.
    retired: [Vec<Texture>; 2],
    /// Guest memory (start, length) of the textures dropped or replaced since
    /// [`ResolvedTextures::take_dropped`]: what the texture cache bound of them must go.
    dropped: Vec<(u32, u32)>,
    /// The textures the last resolve was copied into, if it was done here.
    last: Vec<Texture>,
    frame: u64,
    pub stats: ResolvedStats,
}

impl ResolvedTextures {
    /// Does the copy `plan` describes for `info`, from `surface` (the source of a depth copy
    /// as the EDRAM shaders take it). `false` (nothing done) for a destination left to the
    /// usual path. Other destinations it writes over are stored and dropped first.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        edram: &mut EdramCache,
        shared: &SharedMemoryTarget<'_>,
        info: &ResolveInfo,
        plan: Plan,
        surface: &ExternalSource<'_>,
    ) -> bool {
        self.last.clear();
        let same = self.dests.iter().filter(|d| d.kind == plan.kind).map(|d| d.key);
        let (key, rect) = xenos_edram::canonical_dest(same, plan.key, plan.rect);
        if self.unsuited.contains(&key) {
            return false;
        }
        self.drop_overlapping(device, encoder, edram, Some(shared), key.span(), Some((key, plan.kind)));
        let frame = self.frame;
        let is_depth = matches!(plan.kind, Kind::Depth { .. });
        let make = |device: &Device, width: u32, height: u32| {
            if is_depth {
                let (bytes, depth) = depth_textures(device, width, height);
                (bytes, Some(depth))
            } else {
                (texture(device, width, height), None)
            }
        };
        let index = match self.dests.iter().position(|d| d.key == key && d.kind == plan.kind) {
            Some(i) => i,
            None => {
                // (A depth destination's has no second texture for what doesn't fit: it starts
                // out whole.)
                let (width, height) =
                    if is_depth { (key.pitch, key.height) } else { (rect[0] + rect[2], rect[1] + rect[3]) };
                let (texture, depth) = make(device, width, height);
                self.stats.textures += 1;
                self.dests.push(Dest {
                    key,
                    kind: plan.kind,
                    texture,
                    width,
                    height,
                    depth,
                    full: None,
                    views: [None, None],
                    rects: Vec::new(),
                    unstored: Vec::new(),
                    extent: (info.copy_dest_extent_start & 0x1FFF_FFFF, info.copy_dest_extent_start & 0x1FFF_FFFF),
                    wanted: None,
                    resizes: 0,
                    last_used: frame,
                });
                self.dests.len() - 1
            }
        };
        // Another size was asked for: a new texture, once what only the old one holds is stored.
        if let Some((width, height)) = self.dests[index].wanted.take() {
            self.store_dest(device, encoder, edram, shared, index);
            let d = &mut self.dests[index];
            d.resizes += 1;
            if d.resizes > MAX_RESIZES {
                log::info!(
                    "d3d renderer: resolves to {:#010X} are stored again (textures of several sizes read them)",
                    key.base
                );
                self.unsuited.push(key);
                self.remove(index);
                return false;
            }
            self.dropped.push(d.extent());
            let (texture, depth) = make(device, width, height);
            self.retired[0].push(std::mem::replace(&mut d.texture, texture));
            self.retired[0].extend(std::mem::replace(&mut d.depth, depth).map(|(texture, _)| texture));
            (d.width, d.height, d.views[0]) = (width, height, None);
            d.rects.clear();
            self.stats.textures += 1;
        }
        let d = &mut self.dests[index];
        d.last_used = frame;
        if let (Kind::Depth { float24 }, Some((_, targets))) = (plan.kind, &d.depth) {
            if !inside(rect, d.width, d.height) {
                // Textures smaller than what is resolved read it: not kept as textures.
                self.store_dest(device, encoder, edram, shared, index);
                self.unsuited.push(key);
                self.remove(index);
                return false;
            }
            edram.copy_depth_external(device, encoder, surface, plan.source, float24, &targets[0], &targets[1], rect);
            self.stats.copies += 1;
            d.rects.retain(|r| !xenos_edram::rect_contains(rect, *r));
            d.rects.push(rect);
            d.unstored.retain(|(_, r)| !xenos_edram::rect_contains(rect, *r));
            d.unstored.push((info.clone(), rect));
            let start = info.copy_dest_extent_start & 0x1FFF_FFFF;
            d.extent = (d.extent.0.min(start), d.extent.1.max(start + info.copy_dest_extent_length));
            self.last.push(d.texture.clone());
            self.last.extend(d.depth.as_ref().map(|(texture, _)| texture.clone()));
            return true;
        }
        if !inside(rect, d.width, d.height) && d.full.is_none() {
            // What is unstored so far lies inside the texture, which keeps holding it.
            d.full = Some(texture(device, key.pitch, key.height));
            self.stats.textures += 1;
        }
        let copy = |encoder: &mut CommandEncoder, to: &Texture, width: u32, height: u32| {
            let (w, h) = (rect[2].min(width.saturating_sub(rect[0])), rect[3].min(height.saturating_sub(rect[1])));
            if w == 0 || h == 0 {
                return None;
            }
            encoder.copy_texture_to_texture(
                &TexelCopyTextureInfo {
                    texture: surface.texture,
                    mip_level: 0,
                    origin: origin(plan.source[0], plan.source[1], 0),
                    aspect: TextureAspect::All,
                },
                &TexelCopyTextureInfo {
                    texture: to,
                    mip_level: 0,
                    origin: origin(rect[0], rect[1], 0),
                    aspect: TextureAspect::All,
                },
                extent(w, h, 1),
            );
            Some([rect[0], rect[1], w, h])
        };
        let filled = copy(encoder, &d.texture, d.width, d.height);
        if let Some(full) = &d.full {
            copy(encoder, full, key.pitch, key.height);
            self.stats.full_copies += 1;
        }
        self.stats.copies += 1;
        if let Some(filled) = filled {
            d.rects.retain(|r| !xenos_edram::rect_contains(filled, *r));
            d.rects.push(filled);
        }
        // A resolve over all of an earlier one replaces it.
        d.unstored.retain(|(_, r)| !xenos_edram::rect_contains(rect, *r));
        d.unstored.push((info.clone(), rect));
        let start = info.copy_dest_extent_start & 0x1FFF_FFFF;
        d.extent = (d.extent.0.min(start), d.extent.1.max(start + info.copy_dest_extent_length));
        self.last.push(d.texture.clone());
        true
    }

    /// The textures of the last [`ResolvedTextures::resolve`] done here.
    pub fn last(&self) -> Vec<Texture> {
        self.last.clone()
    }

    /// A resolve is done the usual way: it isn't one into a texture here.
    pub fn resolved_elsewhere(&mut self) {
        self.last.clear();
    }

    /// The texture standing for the 2D texture of `width` x `height` pixels at physical address
    /// `base` with rows of `pitch` pixels, if resolves filled all of one that size.
    pub fn lookup(&mut self, base: u32, pitch: u32, width: u32, height: u32) -> Option<Resolved> {
        let frame = self.frame;
        let d = self.dests.iter_mut().find(|d| d.key.base == base && d.key.pitch == pitch)?;
        if (d.width, d.height) != (width, height) {
            if width <= d.key.pitch && height <= d.key.height {
                d.wanted = Some((width, height));
            }
            return None;
        }
        d.wanted = None;
        if !xenos_edram::rects_cover(&d.rects, width, height) {
            return None;
        }
        d.last_used = frame;
        let (swap, float24) = match d.kind {
            Kind::Color { swap } => (swap, false),
            Kind::Depth { float24 } => (false, float24),
        };
        Some(Resolved {
            texture: d.texture.clone(),
            swap,
            endian: d.key.endian,
            depth: d.depth.as_ref().map(|(texture, _)| (texture.clone(), float24)),
            extent: d.extent(),
        })
    }

    /// The guest memory (start, length) of the destinations with resolves not stored yet that
    /// overlap `[address, address + len)`.
    pub fn unstored(&self, address: u32, len: u32) -> Vec<(u32, u32)> {
        self.dests
            .iter()
            .filter(|d| !d.unstored.is_empty() && xenos_edram::spans_overlap(d.extent(), (address, len)))
            .map(Dest::extent)
            .collect()
    }

    /// The first byte and the end of the guest memory resolves not stored yet were to write.
    pub fn unstored_bounds(&self) -> Option<(u32, u32)> {
        self.dests
            .iter()
            .filter(|d| !d.unstored.is_empty())
            .map(|d| d.extent)
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
    }

    /// Stores the resolves not stored yet of the destinations overlapping
    /// `[address, address + len)` into the GPU copy of guest memory.
    pub fn store(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        edram: &mut EdramCache,
        shared: &SharedMemoryTarget<'_>,
        address: u32,
        len: u32,
    ) {
        for i in 0..self.dests.len() {
            if xenos_edram::spans_overlap(self.dests[i].extent(), (address, len)) {
                self.store_dest(device, encoder, edram, shared, i);
            }
        }
    }

    /// Stores destination `index`'s unstored resolves: each is done again as recorded, with the
    /// destination's texture, where its pixels are, as the surface. A depth resolve is done
    /// again as a color resolve of the bytes texture into 8-bit RGBA without a swap, which
    /// stores the same words.
    fn store_dest(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        edram: &mut EdramCache,
        shared: &SharedMemoryTarget<'_>,
        index: usize,
    ) {
        let d = &mut self.dests[index];
        for (mut info, rect) in std::mem::take(&mut d.unstored) {
            // The whole destination's texture holds every resolve since it was made, and the
            // bound one all those inside it.
            let whole = !inside(rect, d.width, d.height);
            let (source, width, height) = match (&d.full, whole) {
                (Some(full), true) => (full, d.key.pitch, d.key.height),
                _ => (&d.texture, d.width, d.height),
            };
            let views = d.views[usize::from(whole)].get_or_insert_with(|| ExternalViews::new(source, false));
            info.copy_control.color_clear_enable = false;
            info.copy_control.depth_clear_enable = false;
            (info.edram_offset_x_div_8, info.edram_offset_y_div_8) = (0, 0);
            if matches!(d.kind, Kind::Depth { .. }) {
                info.copy_control.copy_src_select = 0;
                info.copy_control.copy_command = CopyCommand::Raw;
                info.copy_control.copy_sample_select = CopySampleSelect::S0;
                info.copy_sample_select = CopySampleSelect::S0;
                info.depth_edram = None;
                info.color_edram = Some(ResolveEdramInfo {
                    pitch_tiles: width.div_ceil(80),
                    msaa_samples: MsaaSamples::X1,
                    ..Default::default()
                });
                info.color_original_base = 0;
                info.dest_format = GuestFormat::K8888;
                info.copy_dest_info.copy_dest_format = GuestFormat::K8888;
                info.copy_dest_info.copy_dest_number = SurfaceNumberFormat::UnsignedRepeatingFraction;
                info.copy_dest_info.copy_dest_exp_bias = 0;
                info.copy_dest_info.copy_dest_swap = false;
            }
            let external = ExternalSource {
                texture: source,
                views,
                format: TextureFormat::RGBA8Unorm,
                width,
                height,
                samples: 1,
                depth: false,
                base_tiles: info.color_original_base,
                offset: [rect[0], rect[1]],
            };
            if let Err(e) = edram.resolve_external(device, encoder, &info, shared, Some(&external), None) {
                log::warn!("d3d renderer: storing a resolve kept as a texture: {e}");
            }
            self.stats.stored += 1;
        }
    }

    /// Guest memory `span` (start, length) is written by something other than a resolve into
    /// `except` (a destination's key and kind): the destinations it overlaps are stored (with
    /// `shared`) and dropped.
    pub fn drop_overlapping(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        edram: &mut EdramCache,
        shared: Option<&SharedMemoryTarget<'_>>,
        span: (u32, u32),
        except: Option<(DestKey, Kind)>,
    ) {
        let mut i = 0;
        while i < self.dests.len() {
            let d = &self.dests[i];
            if Some((d.key, d.kind)) == except || !d.overlaps(span) {
                i += 1;
                continue;
            }
            if let Some(shared) = shared {
                self.store_dest(device, encoder, edram, shared, i);
            }
            self.remove(i);
        }
    }

    fn remove(&mut self, index: usize) {
        let d = self.dests.swap_remove(index);
        self.dropped.push(d.extent());
        self.retired[0].push(d.texture);
        self.retired[0].extend(d.full);
        self.retired[0].extend(d.depth.map(|(texture, _)| texture));
        self.stats.dropped += 1;
    }

    /// Drops the destinations idle for [`MAX_IDLE_FRAMES`], stored first.
    pub fn trim(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        edram: &mut EdramCache,
        shared: &SharedMemoryTarget<'_>,
    ) {
        let frame = self.frame;
        let mut i = 0;
        while i < self.dests.len() {
            if frame - self.dests[i].last_used > MAX_IDLE_FRAMES {
                self.store_dest(device, encoder, edram, shared, i);
                self.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Whether a destination went unused long enough for [`ResolvedTextures::trim`] to drop it.
    pub fn has_idle(&self) -> bool {
        self.dests.iter().any(|d| self.frame - d.last_used > MAX_IDLE_FRAMES)
    }

    /// Everything goes (the surfaces are made again, see `D3dRenderer::reconfigure`): what
    /// wasn't stored is lost with the surfaces' contents.
    pub fn clear(&mut self) {
        while !self.dests.is_empty() {
            self.remove(0);
        }
        self.unsuited.clear();
        self.last.clear();
    }

    /// The guest memory of the textures dropped or replaced since the last call.
    pub fn take_dropped(&mut self) -> Vec<(u32, u32)> {
        std::mem::take(&mut self.dropped)
    }

    /// Frame boundary, with everything recorded submitted.
    pub fn end_frame(&mut self) {
        self.frame += 1;
        for texture in self.retired[1].drain(..) {
            texture.destroy();
        }
        self.retired.swap(0, 1);
    }

    pub fn count(&self) -> usize {
        self.dests.len()
    }
}

#[cfg(test)]
mod tests {
    use super::swizzle;

    #[test]
    fn swizzle_follows_the_swap_and_both_byte_orders() {
        // Same byte order on both sides: only the resolve's red/blue swap shows.
        assert_eq!(swizzle(false, 2, 2), [0, 1, 2, 3]);
        assert_eq!(swizzle(true, 2, 2), [2, 1, 0, 3]);
        assert_eq!(swizzle(true, 0, 0), [2, 1, 0, 3]);
        // Stored as is, read byte-swapped: the word's bytes reversed.
        assert_eq!(swizzle(false, 0, 2), [3, 2, 1, 0]);
        assert_eq!(swizzle(true, 0, 2), [3, 0, 1, 2]);
        // 8 in 16 against 16 in 32 is a full reversal too.
        assert_eq!(swizzle(false, 1, 3), [3, 2, 1, 0]);
    }
}
