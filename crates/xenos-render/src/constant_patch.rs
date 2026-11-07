//! Constant blocks built on the GPU from what changed.
//!
//! Every draw binds a block of up to 256 float constants per shader stage, plus blocks of fetch,
//! system and bool/loop constants, and draws in one submission each need their own blocks.
//! Titles rewrite constants before every draw, but in Skyrim only about 5% of the float vec4s of
//! a new block differ from the stage's previous block (and a draw changes one or two of the 32
//! fetch constants), so instead of uploading whole blocks (megabytes per frame, which browsers
//! copy slowly) each block is recorded as a patch on the previous one of its kind ("stage"): a
//! 256-bit mask of the changed vec4s and their values. Before the submission's own commands, one
//! compute dispatch replays the patches into the blocks: one workgroup per stage and one
//! invocation per vec4 index, walking the stage's blocks in order and carrying the last value
//! forward (indices are independent, so nothing synchronizes). The records are grouped by stage
//! when uploaded, so each workgroup walks only its own stage's (the walk is serial, and with
//! every stage's records in it the dispatch took ~0.6 ms on an M2 GPU).
//!
//! The first block of a stage in a submission has no previous block: all of it is sent.

use gpu_webgpu::*;

/// Bytes of the block buffer (blocks are sub-allocated like the constant ring; it rewinds after
/// each submission, and a submission is made first when it's full). Skyrim submits every ~20
/// draws on average (at each resolve), so 1024 blocks are rarely all used.
pub const BLOCK_BUFFER_BYTES: u64 = 4 << 20;
/// Bytes of each set's block buffer with direct writes: sets are made per submission in
/// flight, so they're kept smaller (a submission that fills one is made early).
const DIRECT_BLOCK_BUFFER_BYTES: u64 = 1 << 20;
/// Patch records per submission at most (more needs a submission first).
const MAX_RECORDS: usize = 16384;
/// Changed vec4s per submission at most.
const MAX_VALUES: usize = 1 << 17;
/// u32s per record: stage, block offset (vec4s), vec4 count, first value, 8 mask words.
const RECORD_WORDS: usize = 12;
/// Vec4s per block at most (one invocation each).
const INDICES: usize = 256;
/// Kinds of blocks, each patched on its own previous block: vertex and pixel shader float
/// constants (0 and 1), then the [`FETCH`], [`SYSTEM`] and [`BOOL_LOOP`] blocks.
const STAGES: u32 = 5;
pub const FETCH: u32 = 2;
pub const SYSTEM: u32 = 3;
pub const BOOL_LOOP: u32 = 4;

/// Words before the first record: where each stage's records start, then the record count.
const HEADER_WORDS: usize = STAGES as usize + 1;

const SHADER: &str = r#"
// records[s] (s < 5) is the index of stage s's first record and records[5] the record count;
// record r starts at 6 + 12 r. Records are grouped by stage, in submission order within one.
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> blocks: array<vec4<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let stage = group.x;
    let first = records[stage];
    let end = records[stage + 1u];
    let word_index = t / 32u;
    let bit = 1u << (t % 32u);
    var current = vec4<u32>(0u);
    for (var r = first; r < end; r++) {
        let b = 6u + r * 12u;
        let mask = records[b + 4u + word_index];
        if ((mask & bit) != 0u) {
            // This vec4's value is the n-th of the record's values, n = changed vec4s before it.
            var n = countOneBits(mask & (bit - 1u));
            for (var w = 0u; w < word_index; w++) {
                n += countOneBits(records[b + 4u + w]);
            }
            current = values[records[b + 3u] + n];
        }
        if (t < records[b + 2u]) {
            blocks[records[b + 1u] + t] = current;
        }
    }
}
"#;

/// Buffer sets used in turn, one per submission: a submission's patch dispatch then writes
/// blocks that the previous submission's draws aren't reading, so it needn't wait for them
/// (with one set, every submission waited for the GPU to finish the previous one). With direct
/// writes there are as many as submissions in flight need (see [`ConstantPatcher::rewind`]).
pub const SETS: usize = 3;
/// Most sets with direct writes (1 MiB of blocks each, 32 MiB in all); past that a submission
/// waits. How far ahead of the GPU the renderer may record matters: 16 sets cost ~10% fps.
const MAX_SETS: usize = 32;

struct Set {
    /// The blocks, bound to the draws' float constant bindings.
    blocks: Buffer,
    records: Buffer,
    values: Buffer,
    group: BindGroup,
    /// Completion of the last submission that used the set (direct writes wait for it).
    done: Option<Pending<gpu_webgpu::Result<()>>>,
}

pub struct ConstantPatcher {
    sets: Vec<Set>,
    /// The set this submission uses.
    current: usize,
    /// Blocks are written straight into the set's buffer by the CPU (memory shared with the
    /// GPU, as on Apple GPUs) instead of patched on the GPU: no dispatch at the start of every
    /// submission for its draws to wait for.
    direct: bool,
    alignment: u64,
    cursor: u64,
    /// The records, in submission order.
    records: Vec<u32>,
    values: Vec<u8>,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
    /// Time spent waiting for a set to come free (the GPU behind by [`MAX_SETS`] submissions).
    pub waited: std::time::Duration,
}

/// A buffer set. With direct writes (`direct`) the records and values are never used: they get
/// minimal buffers (sets are made per submission in flight, and memory is short on phones).
fn new_set(device: &Device, layout: &BindGroupLayout, direct: bool) -> Set {
    let buffer =
        |label, size, usage| device.create_buffer(&BufferDescriptor { label, size, usage, mapped_at_creation: false });
    // Created mapped so that it's zeroed right away by the CPU: a buffer created unmapped may be
    // cleared by the GPU at the next submission, over blocks written directly before it.
    let blocks = device.create_buffer(&BufferDescriptor {
        label: "xenos constant blocks",
        size: if direct { DIRECT_BLOCK_BUFFER_BYTES } else { BLOCK_BUFFER_BYTES },
        usage: BufferUsage::Storage | BufferUsage::Uniform | BufferUsage::CopySrc,
        mapped_at_creation: true,
    });
    blocks.unmap();
    let (records_size, values_size) =
        if direct { (16, 16) } else { (((HEADER_WORDS + MAX_RECORDS * RECORD_WORDS) * 4) as u64, (MAX_VALUES * 16) as u64) };
    let records = buffer("xenos constant patches", records_size, BufferUsage::Storage | BufferUsage::CopyDst);
    let values = buffer("xenos constant values", values_size, BufferUsage::Storage | BufferUsage::CopyDst);
    let whole = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
    let group = device.create_bind_group(
        "xenos constant patches",
        layout,
        &[
            BindGroupEntry { binding: 0, resource: whole(&records) },
            BindGroupEntry { binding: 1, resource: whole(&values) },
            BindGroupEntry { binding: 2, resource: whole(&blocks) },
        ],
    );
    Set { blocks, records, values, group, done: None }
}

impl ConstantPatcher {
    pub fn new(device: &Device, alignment: u64) -> Self {
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStage::Compute,
            ty: BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: 0 },
        };
        let layout = device.create_bind_group_layout(
            "xenos constant patches",
            &[
                entry(0, BufferBindingType::ReadOnlyStorage),
                entry(1, BufferBindingType::ReadOnlyStorage),
                entry(2, BufferBindingType::Storage),
            ],
        );
        let module = device.create_shader_module_wgsl("xenos constant patches", SHADER);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "xenos constant patches",
            layout: Some(&device.create_pipeline_layout("xenos constant patches", &[&layout])),
            stage: ProgrammableStage::new(&module, "main"),
        });
        // An empty write succeeds where direct writes are possible.
        let probe = device.create_buffer(&BufferDescriptor {
            label: "xenos constant probe",
            size: 16,
            usage: BufferUsage::Uniform,
            mapped_at_creation: false,
        });
        let direct = probe.write_unsynchronized(0, &[]);
        probe.destroy();
        let sets: Vec<Set> = (0..SETS).map(|_| new_set(device, &layout, direct)).collect();
        ConstantPatcher {
            sets,
            current: 0,
            direct,
            alignment: alignment.max(16),
            cursor: 0,
            records: Vec::new(),
            values: Vec::new(),
            layout,
            pipeline,
            waited: std::time::Duration::ZERO,
        }
    }

    /// Whether blocks are written directly (then [`ConstantPatcher::push_full`] replaces
    /// [`ConstantPatcher::push_masked`]).
    pub fn direct(&self) -> bool {
        self.direct
    }

    /// Bytes of each set's block buffer.
    fn capacity(&self) -> u64 {
        if self.direct { DIRECT_BLOCK_BUFFER_BYTES } else { BLOCK_BUFFER_BYTES }
    }

    /// Sets so far (each needs its own bind group for the draws).
    pub fn sets(&self) -> usize {
        self.sets.len()
    }

    /// The block buffer of set `set` (each needs its own bind group for the draws).
    pub fn blocks(&self, set: usize) -> &Buffer {
        &self.sets[set].blocks
    }

    /// The set the draws recorded now read their blocks from.
    pub fn current(&self) -> usize {
        self.current
    }

    /// Whether blocks bound with `windows` bytes each, with all their vec4s changed, still fit
    /// in this submission (else submit first).
    pub fn has_room(&self, windows: &[u64]) -> bool {
        // Each block may start up to an alignment past the previous window's end.
        let bytes: u64 = windows.iter().map(|w| w.next_multiple_of(self.alignment)).sum();
        let vec4s: usize = windows.iter().map(|&w| (w as usize / 16).min(INDICES)).sum();
        self.cursor.next_multiple_of(self.alignment) + bytes <= self.capacity()
            && (self.direct
                || (self.records.len() / RECORD_WORDS + windows.len() <= MAX_RECORDS
                    && self.values.len() / 16 + vec4s <= MAX_VALUES))
    }

    /// Adds `data` (vec4s of stage `stage`) as a block bound with `window` bytes, given the
    /// stage's previous block of this submission. Returns its offset. Check
    /// [`ConstantPatcher::has_room`] first.
    pub fn push(&mut self, stage: u32, previous: Option<&[u8]>, data: &[u8], window: u64) -> u32 {
        debug_assert!(data.len().is_multiple_of(16) && data.len() <= INDICES * 16);
        if self.direct {
            return self.push_full(data, window);
        }
        let offset = self.cursor.next_multiple_of(self.alignment);
        assert!(offset + window <= self.capacity(), "float constant blocks full");
        self.cursor = offset + (data.len() as u64).max(16);
        let first_value = self.values.len() / 16;
        let mut mask = [0u32; INDICES / 32];
        for (i, v) in data.chunks(16).enumerate() {
            if previous.and_then(|p| p.get(i * 16..i * 16 + 16)) != Some(v) {
                mask[i / 32] |= 1 << (i % 32);
                self.values.extend_from_slice(v);
            }
        }
        self.records.extend([stage, (offset / 16) as u32, (data.len() / 16) as u32, first_value as u32]);
        self.records.extend(mask);
        offset as u32
    }

    /// With direct writes: writes the whole block `data`, bound with `window` bytes, into the
    /// current set. Returns its offset.
    pub fn push_full(&mut self, data: &[u8], window: u64) -> u32 {
        debug_assert!(self.direct);
        let offset = self.cursor.next_multiple_of(self.alignment);
        assert!(offset + window <= self.capacity(), "float constant blocks full");
        self.cursor = offset + (data.len() as u64).max(16);
        let written = self.sets[self.current].blocks.write_unsynchronized(offset, data);
        debug_assert!(written, "constant block out of bounds");
        offset as u32
    }

    /// Like [`ConstantPatcher::push`] for a block of `count` vec4s whose changed vec4s (in `mask`,
    /// with their bytes in index order in `values`) are already known: the others are the
    /// stage's previous block's, which must cover them.
    pub fn push_masked(
        &mut self,
        stage: u32,
        count: usize,
        mask: [u32; INDICES / 32],
        values: &[u8],
        window: u64,
    ) -> u32 {
        debug_assert!(
            count <= INDICES && values.len() == mask.iter().map(|m| m.count_ones() as usize).sum::<usize>() * 16
        );
        let offset = self.cursor.next_multiple_of(self.alignment);
        assert!(offset + window <= self.capacity(), "float constant blocks full");
        self.cursor = offset + (count as u64 * 16).max(16);
        let first_value = self.values.len() / 16;
        self.values.extend_from_slice(values);
        self.records.extend([stage, (offset / 16) as u32, count as u32, first_value as u32]);
        self.records.extend(mask);
        offset as u32
    }

    /// Uploads the patches and returns the command buffer that applies them (to submit before
    /// the draws that read the blocks), with the bytes uploaded; `None` if there are none.
    pub fn encode(&mut self, device: &Device) -> Option<(CommandBuffer, u64)> {
        if self.records.is_empty() {
            return None;
        }
        // Grouped by stage (stable: a stage's records stay in order), after the header.
        let count = self.records.len() / RECORD_WORDS;
        let mut words = Vec::with_capacity(HEADER_WORDS + self.records.len());
        words.resize(HEADER_WORDS, 0);
        for stage in 0..STAGES {
            words[stage as usize] = ((words.len() - HEADER_WORDS) / RECORD_WORDS) as u32;
            for record in self.records.chunks_exact(RECORD_WORDS).filter(|r| r[0] == stage) {
                words.extend_from_slice(record);
            }
        }
        words[STAGES as usize] = count as u32;
        let records: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let set = &self.sets[self.current];
        device.queue().write_buffer(&set.records, 0, &records);
        if !self.values.is_empty() {
            device.queue().write_buffer(&set.values, 0, &self.values);
        }
        let mut encoder = device.create_command_encoder("xenos constant patches");
        {
            let mut pass = encoder.begin_compute_pass("xenos constant patches");
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &set.group, &[]);
            pass.dispatch_workgroups(STAGES, 1, 1);
            pass.end();
        }
        Some((encoder.finish("xenos constant patches"), (records.len() + self.values.len()) as u64))
    }

    /// The submission went out: blocks and patches start over, in the next set. With direct
    /// writes that is a set no submission still running uses (the CPU writes it while the GPU
    /// runs), a new one if all are busy.
    pub fn rewind(&mut self, device: &Device) {
        if self.direct {
            self.sets[self.current].done = Some(device.queue().on_submitted_work_done());
            let n = self.sets.len();
            let free = (1..=n)
                .map(|k| (self.current + k) % n)
                .find(|&i| self.sets[i].done.as_ref().is_none_or(Pending::is_ready));
            self.current = match free {
                Some(i) => i,
                None if self.sets.len() < MAX_SETS => {
                    self.sets.push(new_set(device, &self.layout, true));
                    self.sets.len() - 1
                }
                None => {
                    // As many submissions in flight as sets: wait for the oldest's set.
                    let next = (self.current + 1) % n;
                    let done = self.sets[next].done.take();
                    // Direct writes are native only (the browser patches on the GPU).
                    #[cfg(not(target_os = "emscripten"))]
                    if let Some(done) = done {
                        let started = std::time::Instant::now();
                        let _ = device.block_on(done);
                        self.waited += started.elapsed();
                    }
                    #[cfg(target_os = "emscripten")]
                    drop(done);
                    next
                }
            };
            self.sets[self.current].done = None;
        } else {
            self.current = (self.current + 1) % SETS;
        }
        self.cursor = 0;
        self.records.clear();
        self.values.clear();
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::*;

    /// Blocks of every stage with varying lengths and a few changed vec4s each (including NaN
    /// patterns, which must survive bit for bit) come out of the GPU exactly as pushed.
    #[test]
    fn patches_rebuild_every_block_exactly() {
        let instance = Instance::new().expect("instance");
        let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions::default())).expect("adapter");
        let device = instance.block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");
        let mut patcher = ConstantPatcher::new(&device, 256);
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut expected = Vec::new();
        let mut previous: [Option<Vec<u8>>; STAGES as usize] = Default::default();
        for i in 0..300 {
            let stage = (next() % STAGES) as usize;
            let len = [256usize, 256, 40, 3, 256, 128][i % 6];
            let mut data = previous[stage].clone().unwrap_or_default();
            data.resize(len * 16, 0);
            for _ in 0..(next() % 12) {
                let v = (next() as usize % len) * 16;
                let word = if next() % 7 == 0 { 0x7FC0_0001 } else { next() };
                data[v..v + 4].copy_from_slice(&word.to_le_bytes());
            }
            assert!(patcher.has_room(&[4096]));
            let offset = patcher.push(stage as u32, previous[stage].as_deref(), &data, 4096);
            expected.push((offset, data.clone()));
            previous[stage] = Some(data);
        }
        // With direct writes the blocks are written already and there's nothing to patch.
        let commands = patcher.encode(&device).map(|(commands, bytes)| {
            let total: usize = expected.iter().map(|(_, d)| d.len()).sum();
            assert!(bytes < total as u64 / 2, "patches ({bytes} bytes) should be far smaller than the blocks ({total})");
            commands
        });
        assert_eq!(commands.is_none(), patcher.direct());
        let readback = device.create_buffer(&BufferDescriptor {
            label: "readback",
            size: patcher.blocks(patcher.current()).size(),
            usage: BufferUsage::MapRead | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder("readback");
        encoder.copy_buffer_to_buffer(patcher.blocks(patcher.current()), 0, &readback, 0, readback.size());
        device.queue().submit(commands.into_iter().chain([encoder.finish("readback")]));
        device.block_on(readback.map_async(MapMode::Read, 0, None)).expect("map");
        for (offset, data) in expected {
            let mut got = vec![0u8; data.len()];
            readback.read_mapped(u64::from(offset), &mut got).expect("read");
            assert_eq!(got, data, "block at {offset}");
        }
    }
}
