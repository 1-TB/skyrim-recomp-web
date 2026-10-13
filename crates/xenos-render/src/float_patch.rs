//! Float constant blocks built on the GPU from what changed.
//!
//! Every draw binds a block of up to 256 float constants per shader stage, and draws in one
//! submission each need their own block. Titles rewrite constants before every draw, but in
//! Skyrim only about 5% of the vec4s of a new block differ from the stage's previous block, so
//! instead of uploading whole blocks (megabytes per frame, which browsers copy slowly) each block
//! is recorded as a patch on the previous one: a 256-bit mask of the changed vec4s and their
//! values. Before the submission's own commands, one compute dispatch replays the patches into
//! the blocks: one invocation per constant index, walking the blocks in order and carrying the
//! last value forward (indices are independent, so nothing synchronizes).
//!
//! The first block of a stage in a submission has no previous block: all of it is sent.

use gpu_webgpu::*;

/// Bytes of the block buffer (blocks are sub-allocated like the constant ring; it rewinds after
/// each submission).
pub const BLOCK_BUFFER_BYTES: u64 = 16 << 20;
/// Patch records per submission at most (more needs a submission first).
const MAX_RECORDS: usize = 16384;
/// Changed vec4s per submission at most.
const MAX_VALUES: usize = 1 << 17;
/// u32s per record: stage, block offset (vec4s), vec4 count, first value, 8 mask words.
const RECORD_WORDS: usize = 12;
/// Constant indices per stage (one invocation each).
const INDICES: usize = 256;

const SHADER: &str = r#"
// records[0] is the record count; record r starts at 1 + 12 r.
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> blocks: array<vec4<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let stage = group.x;
    let count = records[0];
    let word_index = t / 32u;
    let bit = 1u << (t % 32u);
    var current = vec4<u32>(0u);
    for (var r = 0u; r < count; r++) {
        let b = 1u + r * 12u;
        if (records[b] != stage) {
            continue;
        }
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

pub struct FloatPatcher {
    /// The blocks, bound to the draws' float constant bindings.
    pub buffer: Buffer,
    alignment: u64,
    cursor: u64,
    /// Record count followed by the records.
    records: Vec<u32>,
    values: Vec<u8>,
    records_buffer: Buffer,
    values_buffer: Buffer,
    pipeline: ComputePipeline,
    group: BindGroup,
}

impl FloatPatcher {
    pub fn new(device: &Device, alignment: u64) -> Self {
        let buffer = |label, size, usage| {
            device.create_buffer(&BufferDescriptor { label, size, usage, mapped_at_creation: false })
        };
        let blocks = buffer("xenos float constants", BLOCK_BUFFER_BYTES, BufferUsage::Storage | BufferUsage::CopySrc);
        let records_buffer = buffer(
            "xenos float constant patches",
            ((1 + MAX_RECORDS * RECORD_WORDS) * 4) as u64,
            BufferUsage::Storage | BufferUsage::CopyDst,
        );
        let values_buffer = buffer(
            "xenos float constant values",
            (MAX_VALUES * 16) as u64,
            BufferUsage::Storage | BufferUsage::CopyDst,
        );
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStage::Compute,
            ty: BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: 0 },
        };
        let layout = device.create_bind_group_layout(
            "xenos float constant patches",
            &[
                entry(0, BufferBindingType::ReadOnlyStorage),
                entry(1, BufferBindingType::ReadOnlyStorage),
                entry(2, BufferBindingType::Storage),
            ],
        );
        let module = device.create_shader_module_wgsl("xenos float constant patches", SHADER);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "xenos float constant patches",
            layout: Some(&device.create_pipeline_layout("xenos float constant patches", &[&layout])),
            stage: ProgrammableStage::new(&module, "main"),
        });
        let whole = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
        let group = device.create_bind_group(
            "xenos float constant patches",
            &layout,
            &[
                BindGroupEntry { binding: 0, resource: whole(&records_buffer) },
                BindGroupEntry { binding: 1, resource: whole(&values_buffer) },
                BindGroupEntry { binding: 2, resource: whole(&blocks) },
            ],
        );
        FloatPatcher {
            buffer: blocks,
            alignment: alignment.max(16),
            cursor: 0,
            records: vec![0],
            values: Vec::new(),
            records_buffer,
            values_buffer,
            pipeline,
            group,
        }
    }

    /// Whether a block of up to `window` bytes with all its vec4s changed still fits in this
    /// submission (else submit first).
    pub fn has_room(&self, window: u64) -> bool {
        self.cursor.next_multiple_of(self.alignment) + window <= BLOCK_BUFFER_BYTES
            && (self.records.len() - 1) / RECORD_WORDS < MAX_RECORDS
            && self.values.len() / 16 + INDICES <= MAX_VALUES
    }

    /// Adds `data` (vec4s of stage `stage`, 0 vertex or 1 pixel) as a block bound with `window`
    /// bytes, given the stage's previous block of this submission. Returns its offset. Check
    /// [`FloatPatcher::has_room`] first.
    pub fn push(&mut self, stage: u32, previous: Option<&[u8]>, data: &[u8], window: u64) -> u32 {
        debug_assert!(data.len() % 16 == 0 && data.len() <= INDICES * 16);
        let offset = self.cursor.next_multiple_of(self.alignment);
        assert!(offset + window <= BLOCK_BUFFER_BYTES, "float constant blocks full");
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
        self.records[0] += 1;
        offset as u32
    }

    /// Uploads the patches and returns the command buffer that applies them (to submit before
    /// the draws that read the blocks), with the bytes uploaded; `None` if there are none.
    pub fn encode(&mut self, device: &Device) -> Option<(CommandBuffer, u64)> {
        if self.records[0] == 0 {
            return None;
        }
        let records: Vec<u8> = self.records.iter().flat_map(|w| w.to_le_bytes()).collect();
        device.queue().write_buffer(&self.records_buffer, 0, &records);
        if !self.values.is_empty() {
            device.queue().write_buffer(&self.values_buffer, 0, &self.values);
        }
        let mut encoder = device.create_command_encoder("xenos float constant patches");
        {
            let mut pass = encoder.begin_compute_pass("xenos float constant patches");
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(2, 1, 1);
            pass.end();
        }
        Some((encoder.finish("xenos float constant patches"), (records.len() + self.values.len()) as u64))
    }

    /// The submission went out: blocks and patches start over.
    pub fn rewind(&mut self) {
        self.cursor = 0;
        self.records.truncate(1);
        self.records[0] = 0;
        self.values.clear();
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::*;

    /// Blocks of both stages with varying lengths and a few changed vec4s each (including NaN
    /// patterns, which must survive bit for bit) come out of the GPU exactly as pushed.
    #[test]
    fn patches_rebuild_every_block_exactly() {
        let instance = Instance::new().expect("instance");
        let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions::default())).expect("adapter");
        let device = instance.block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");
        let mut patcher = FloatPatcher::new(&device, 256);
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut expected = Vec::new();
        let mut previous: [Option<Vec<u8>>; 2] = [None, None];
        for i in 0..300 {
            let stage = (next() % 2) as usize;
            let len = [256usize, 256, 40, 3, 256, 128][i % 6];
            let mut data = previous[stage].clone().unwrap_or_default();
            data.resize(len * 16, 0);
            for _ in 0..(next() % 12) {
                let v = (next() as usize % len) * 16;
                let word = if next() % 7 == 0 { 0x7FC0_0001 } else { next() };
                data[v..v + 4].copy_from_slice(&word.to_le_bytes());
            }
            assert!(patcher.has_room(4096));
            let offset = patcher.push(stage as u32, previous[stage].as_deref(), &data, 4096);
            expected.push((offset, data.clone()));
            previous[stage] = Some(data);
        }
        let (commands, bytes) = patcher.encode(&device).expect("patches");
        let total: usize = expected.iter().map(|(_, d)| d.len()).sum();
        assert!(bytes < total as u64 / 2, "patches ({bytes} bytes) should be far smaller than the blocks ({total})");
        let readback = device.create_buffer(&BufferDescriptor {
            label: "readback",
            size: BLOCK_BUFFER_BYTES,
            usage: BufferUsage::MapRead | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder("readback");
        encoder.copy_buffer_to_buffer(&patcher.buffer, 0, &readback, 0, BLOCK_BUFFER_BYTES);
        device.queue().submit([commands, encoder.finish("readback")]);
        device.block_on(readback.map_async(MapMode::Read, 0, None)).expect("map");
        for (offset, data) in expected {
            let mut got = vec![0u8; data.len()];
            readback.read_mapped(u64::from(offset), &mut got).expect("read");
            assert_eq!(got, data, "block at {offset}");
        }
    }
}
