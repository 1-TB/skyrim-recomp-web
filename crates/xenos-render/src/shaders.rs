//! Translated shader cache: microcode + translation options → WGSL module and its metadata.

use rustc_hash::FxHashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;

use gpu_webgpu::{Device, ShaderModule};
use xenos_shader::{Shader, ShaderStage, Translation, TranslationOptions};

pub struct CachedShader {
    pub module: ShaderModule,
    pub translation: Translation,
    /// The decoded microcode (vertex shaders are run on the CPU to bound unclipped draws).
    pub shader: Shader,
    /// Identifies this (microcode, options) pair in pipeline keys.
    pub id: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    stage: ShaderStage,
    code_hash: u64,
    code_len: usize,
    options: TranslationOptions,
}

#[derive(Default)]
pub struct ShaderCache {
    /// `None` records a shader that failed to translate or compile, so it isn't retried.
    shaders: FxHashMap<Key, Option<Rc<CachedShader>>>,
    next_id: u64,
    /// Time spent translating microcode and creating shader modules (for the stats log).
    pub translate_time: std::time::Duration,
    pub module_time: std::time::Duration,
}

/// Identifies microcode in the cache.
pub fn code_hash(code: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    code.hash(&mut h);
    h.finish()
}

impl ShaderCache {
    pub fn get(
        &mut self,
        device: &Device,
        stage: ShaderStage,
        code: &[u32],
        hash: u64,
        options: &TranslationOptions,
    ) -> Option<Rc<CachedShader>> {
        let key = Key { stage, code_hash: hash, code_len: code.len(), options: options.clone() };
        if let Some(entry) = self.shaders.get(&key) {
            return entry.clone();
        }
        let entry = self.build(device, stage, code, options, key.code_hash);
        self.shaders.insert(key, entry.clone());
        entry
    }

    fn build(
        &mut self,
        device: &Device,
        stage: ShaderStage,
        code: &[u32],
        options: &TranslationOptions,
        hash: u64,
    ) -> Option<Rc<CachedShader>> {
        let shader = match Shader::new(stage, code) {
            Ok(s) => s,
            Err(e) => {
                log::error!("shader {stage:?} {hash:016X}: microcode not decodable: {e}");
                return None;
            }
        };
        let started = std::time::Instant::now();
        let translation = shader.translate(options);
        let translated = std::time::Instant::now();
        self.translate_time += translated - started;
        for w in &translation.warnings {
            log::debug!("shader {stage:?} {hash:016X}: {w}");
        }
        // The translator's output is validated with naga in its tests; anything the device
        // still rejects is reported through its uncaptured-error log.
        let module = device.create_shader_module_wgsl(&format!("{stage:?} {hash:016X}"), &translation.wgsl);
        self.module_time += translated.elapsed();
        self.next_id += 1;
        Some(Rc::new(CachedShader { module, translation, shader, id: self.next_id }))
    }
}
