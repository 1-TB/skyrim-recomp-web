//! Pipelines remembered across sessions. Browsers compile pipelines slowly (seconds for a
//! title's first minutes on D3D12), and draws whose pipeline isn't ready are skipped. Every
//! pipeline created is recorded with what it was made from (microcode, translation options and
//! render state), and the next session starts creating them all at boot, before they are needed.
//!
//! The file is a list of little-endian words: a header, the shaders, then the pipelines referring
//! to shaders by position. Anything unreadable (another version, a truncated write) is ignored.

use std::sync::Arc;

use gpu_webgpu::*;
use rustc_hash::FxHashMap;
use xenos_shader::{TranslationOptions, VertexExpansion};

use super::{ColorKey, DepthKey, PipelineKey, SlotKind, Stage};

const MAGIC: u32 = 0x534B_5043; // "SKPC"
/// Bumped when the encoding or what a key means changes.
const VERSION: u32 = 1;
/// Pipelines kept at most (the oldest are dropped first).
const MAX_PIPELINES: usize = 8192;
const NONE: u32 = u32::MAX;

/// A shader as the executor translates it.
#[derive(Clone)]
pub(super) struct ShaderSource {
    pub stage: Stage,
    pub options: TranslationOptions,
    pub code: Arc<[u32]>,
}

/// A pipeline in terms that outlive the session: shader sources instead of shader ids, texture
/// bind group slots instead of a layout id.
#[derive(Clone)]
pub(super) struct StoredPipeline {
    pub vertex: ShaderSource,
    pub pixel: Option<ShaderSource>,
    pub slots: Vec<(u32, SlotKind)>,
    /// The key with `vertex`, `pixel` and `texture_layout` to be filled in.
    pub key: PipelineKey,
}

/// The pipelines of this and earlier sessions, and where they are kept.
pub(super) struct PipelineStore {
    path: std::path::PathBuf,
    pipelines: Vec<StoredPipeline>,
    /// Encoded records already present (so a pipeline is recorded once).
    seen: std::collections::HashSet<Vec<u32>>,
    dirty: bool,
    last_save: std::time::Instant,
}

fn encode_options(o: &TranslationOptions, w: &mut Vec<u32>) {
    w.extend([
        o.vs_output_interpolators,
        o.vs_point_size_location.unwrap_or(NONE),
        o.ps_interpolator_count.unwrap_or(NONE),
        o.ps_param_gen as u32,
        o.texture_binding_base,
        match o.vs_expansion {
            VertexExpansion::None => 0,
            VertexExpansion::PointSprite => 1,
            VertexExpansion::RectangleList => 2,
        },
    ]);
}

fn encode_shader(s: &ShaderSource, w: &mut Vec<u32>) {
    w.push(match s.stage {
        Stage::Vertex => 0,
        Stage::Pixel => 1,
    });
    encode_options(&s.options, w);
    w.push(s.code.len() as u32);
    w.extend_from_slice(&s.code);
}

/// The pipeline's own words (shaders as indices into `shaders`).
fn encode_pipeline(p: &StoredPipeline, vertex: u32, pixel: u32, w: &mut Vec<u32>) {
    let k = &p.key;
    w.extend([vertex, pixel, p.slots.len() as u32]);
    for &(binding, kind) in &p.slots {
        match kind {
            SlotKind::Texture { dimension, filterable } => w.extend([binding, 0, dimension.0, filterable as u32]),
            SlotKind::Sampler { filtering } => w.extend([binding, 1, filtering as u32, 0]),
        }
    }
    w.extend([
        k.topology.0,
        k.strip_index_format.0,
        k.cull.0,
        k.front_face.0,
        k.color_slots as u32,
        k.sample_count,
        k.sample_mask,
        k.alpha_to_coverage as u32,
    ]);
    for c in &k.colors {
        match c {
            None => w.push(0),
            Some(c) => {
                w.extend([1, c.format.0, c.write_mask]);
                match c.blend {
                    None => w.push(0),
                    Some(b) => {
                        w.push(1);
                        for (op, src, dst) in b {
                            w.extend([op.0, src.0, dst.0]);
                        }
                    }
                }
            }
        }
    }
    match &k.depth {
        None => w.push(0),
        Some(d) => {
            w.extend([1, d.format.0, d.write as u32, d.compare.0]);
            for (compare, fail, depth_fail, pass) in d.stencil {
                w.extend([compare.0, fail.0, depth_fail.0, pass.0]);
            }
            w.extend([d.read_mask, d.write_mask, d.bias as u32, d.slope_scale_bits]);
        }
    }
}

/// Reads words in order; `None` past the end.
struct Reader<'a> {
    words: &'a [u32],
    at: usize,
}

impl Reader<'_> {
    fn next(&mut self) -> Option<u32> {
        let w = *self.words.get(self.at)?;
        self.at += 1;
        Some(w)
    }

    fn take(&mut self, n: usize) -> Option<&[u32]> {
        let s = self.words.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }

    fn opt(&mut self) -> Option<Option<u32>> {
        self.next().map(|v| (v != NONE).then_some(v))
    }

    fn shader(&mut self) -> Option<ShaderSource> {
        let stage = match self.next()? {
            0 => Stage::Vertex,
            1 => Stage::Pixel,
            _ => return None,
        };
        let options = TranslationOptions {
            vs_output_interpolators: self.next()?,
            vs_point_size_location: self.opt()?,
            ps_interpolator_count: self.opt()?,
            ps_param_gen: self.next()? != 0,
            texture_binding_base: self.next()?,
            vs_expansion: match self.next()? {
                0 => VertexExpansion::None,
                1 => VertexExpansion::PointSprite,
                2 => VertexExpansion::RectangleList,
                _ => return None,
            },
        };
        let len = self.next()? as usize;
        let code: Arc<[u32]> = self.take(len)?.into();
        Some(ShaderSource { stage, options, code })
    }

    fn pipeline(&mut self, shaders: &[ShaderSource]) -> Option<StoredPipeline> {
        let vertex = shaders.get(self.next()? as usize)?.clone();
        let pixel = match self.opt()? {
            Some(i) => Some(shaders.get(i as usize)?.clone()),
            None => None,
        };
        let slot_count = self.next()? as usize;
        let mut slots = Vec::with_capacity(slot_count.min(64));
        for _ in 0..slot_count {
            let s = self.take(4)?;
            let kind = match s[1] {
                0 => SlotKind::Texture { dimension: TextureViewDimension(s[2]), filterable: s[3] != 0 },
                1 => SlotKind::Sampler { filtering: s[2] != 0 },
                _ => return None,
            };
            slots.push((s[0], kind));
        }
        let s = self.take(8)?.to_vec();
        let mut colors: [Option<ColorKey>; 4] = Default::default();
        for color in &mut colors {
            if self.next()? != 0 {
                let (format, write_mask) = (TextureFormat(self.next()?), self.next()?);
                let blend = if self.next()? != 0 {
                    let b = self.take(6)?;
                    Some([0, 1].map(|i| {
                        (BlendOperation(b[i * 3]), BlendFactor(b[i * 3 + 1]), BlendFactor(b[i * 3 + 2]))
                    }))
                } else {
                    None
                };
                *color = Some(ColorKey { format, blend, write_mask });
            }
        }
        let depth = if self.next()? != 0 {
            let d = self.take(15)?;
            let face = |i: usize| {
                let f = &d[3 + i * 4..7 + i * 4];
                (CompareFunction(f[0]), StencilOperation(f[1]), StencilOperation(f[2]), StencilOperation(f[3]))
            };
            Some(DepthKey {
                format: TextureFormat(d[0]),
                write: d[1] != 0,
                compare: CompareFunction(d[2]),
                stencil: [face(0), face(1)],
                read_mask: d[11],
                write_mask: d[12],
                bias: d[13] as i32,
                slope_scale_bits: d[14],
            })
        } else {
            None
        };
        if s[4] > 4 {
            return None;
        }
        let key = PipelineKey {
            vertex: 0,
            pixel: None,
            texture_layout: 0,
            topology: PrimitiveTopology(s[0]),
            strip_index_format: IndexFormat(s[1]),
            cull: CullMode(s[2]),
            front_face: FrontFace(s[3]),
            colors,
            depth,
            color_slots: s[4] as usize,
            sample_count: s[5],
            sample_mask: s[6],
            alpha_to_coverage: s[7] != 0,
        };
        Some(StoredPipeline { vertex, pixel, slots, key })
    }
}

fn decode(words: &[u32]) -> Option<Vec<StoredPipeline>> {
    let mut r = Reader { words, at: 0 };
    if r.next()? != MAGIC || r.next()? != VERSION {
        return None;
    }
    let shader_count = r.next()? as usize;
    let mut shaders = Vec::with_capacity(shader_count.min(65536));
    for _ in 0..shader_count {
        shaders.push(r.shader()?);
    }
    let count = r.next()? as usize;
    let mut pipelines = Vec::with_capacity(count.min(MAX_PIPELINES));
    for _ in 0..count {
        pipelines.push(r.pipeline(&shaders)?);
    }
    Some(pipelines)
}

/// Key of a shader source for deduplication in the file.
fn shader_identity(s: &ShaderSource) -> Vec<u32> {
    let mut w = Vec::new();
    encode_shader(s, &mut w);
    w
}

fn encode(pipelines: &[StoredPipeline]) -> Vec<u32> {
    let mut shader_words = Vec::new();
    let mut index: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut shader_count = 0u32;
    let mut index_of = |s: &ShaderSource, shader_words: &mut Vec<u32>| -> u32 {
        let id = shader_identity(s);
        *index.entry(id.clone()).or_insert_with(|| {
            shader_words.extend_from_slice(&id);
            shader_count += 1;
            shader_count - 1
        })
    };
    let mut pipeline_words = Vec::new();
    for p in pipelines {
        let vertex = index_of(&p.vertex, &mut shader_words);
        let pixel = p.pixel.as_ref().map_or(NONE, |s| index_of(s, &mut shader_words));
        encode_pipeline(p, vertex, pixel, &mut pipeline_words);
    }
    let mut words = vec![MAGIC, VERSION, shader_count];
    words.extend(shader_words);
    words.push(pipelines.len() as u32);
    words.extend(pipeline_words);
    words
}

/// What identifies a pipeline record (its encoding with the shaders inline).
fn identity(p: &StoredPipeline) -> Vec<u32> {
    let mut w = shader_identity(&p.vertex);
    if let Some(ps) = &p.pixel {
        w.extend(shader_identity(ps));
    }
    encode_pipeline(p, 0, 0, &mut w);
    w
}

impl PipelineStore {
    /// The store kept at `path`, with the pipelines recorded there before.
    pub fn open(path: std::path::PathBuf) -> Self {
        let pipelines = match std::fs::read(&path) {
            Ok(bytes) => {
                let words: Vec<u32> =
                    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                decode(&words).unwrap_or_else(|| {
                    log::warn!("renderer: {} is not a pipeline list of this version; starting over", path.display());
                    Vec::new()
                })
            }
            Err(_) => Vec::new(),
        };
        let seen = pipelines.iter().map(identity).collect();
        PipelineStore { path, pipelines, seen, dirty: false, last_save: std::time::Instant::now() }
    }

    /// The pipelines of earlier sessions.
    pub fn pipelines(&self) -> &[StoredPipeline] {
        &self.pipelines
    }

    /// Records a pipeline created this session (if it's new).
    pub fn record(&mut self, pipeline: StoredPipeline) {
        if self.seen.insert(identity(&pipeline)) {
            self.pipelines.push(pipeline);
            if self.pipelines.len() > MAX_PIPELINES {
                let dropped = self.pipelines.len() - MAX_PIPELINES;
                for p in self.pipelines.drain(..dropped) {
                    self.seen.remove(&identity(&p));
                }
            }
            self.dirty = true;
        }
    }

    /// Writes the list if pipelines were added and the last write was over `interval` ago.
    pub fn save_if_due(&mut self, interval: std::time::Duration) {
        if !self.dirty || self.last_save.elapsed() < interval {
            return;
        }
        self.dirty = false;
        self.last_save = std::time::Instant::now();
        let bytes: Vec<u8> = encode(&self.pipelines).iter().flat_map(|w| w.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(&self.path, &bytes) {
            log::warn!("renderer: writing {}: {e}", self.path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let vs = ShaderSource {
            stage: Stage::Vertex,
            options: TranslationOptions { vs_point_size_location: Some(3), ..Default::default() },
            code: vec![1, 2, 3].into(),
        };
        let ps = ShaderSource { stage: Stage::Pixel, options: TranslationOptions::default(), code: vec![9; 6].into() };
        let key = PipelineKey {
            vertex: 0,
            pixel: None,
            texture_layout: 0,
            topology: PrimitiveTopology::TriangleStrip,
            strip_index_format: IndexFormat::Uint16,
            cull: CullMode::Back,
            front_face: FrontFace::CW,
            colors: [
                Some(ColorKey {
                    format: TextureFormat::RGBA8Unorm,
                    blend: Some([(BlendOperation::Add, BlendFactor::One, BlendFactor::Zero); 2]),
                    write_mask: 0xF,
                }),
                None,
                Some(ColorKey { format: TextureFormat::RGBA16Float, blend: None, write_mask: 3 }),
                None,
            ],
            depth: Some(DepthKey {
                format: TextureFormat::Depth32FloatStencil8,
                write: true,
                compare: CompareFunction::GreaterEqual,
                stencil: [(
                    CompareFunction::Always,
                    StencilOperation::Keep,
                    StencilOperation::Replace,
                    StencilOperation::Keep,
                ); 2],
                read_mask: 0xFF,
                write_mask: 0x0F,
                bias: -12,
                slope_scale_bits: 1.5f32.to_bits(),
            }),
            color_slots: 3,
            sample_count: 4,
            sample_mask: 0b1001,
            alpha_to_coverage: true,
        };
        let slots = vec![
            (96, SlotKind::Texture { dimension: TextureViewDimension::D2, filterable: true }),
            (97, SlotKind::Sampler { filtering: false }),
        ];
        let pipelines = vec![
            StoredPipeline { vertex: vs.clone(), pixel: Some(ps), slots: slots.clone(), key: key.clone() },
            StoredPipeline { vertex: vs, pixel: None, slots, key },
        ];
        let words = encode(&pipelines);
        let decoded = decode(&words).expect("decodes");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.iter().map(identity).collect::<Vec<_>>(), pipelines.iter().map(identity).collect::<Vec<_>>());
        assert!(decode(&words[..words.len() - 1]).is_none());
    }
}
