//! Translates every compiled Xbox 360 shader found in a game's shader archive and validates the
//! WGSL. Game data is never part of the repository: point `XENOS_SHADER_SOURCE` at a BSA archive
//! (e.g. `Skyrim - Shaders.bsa`) or a directory of extracted files, then run
//! `cargo test -p xenos-shader --test game_shaders -- --ignored --nocapture`.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use xenos_shader::{Shader, ShaderStage, TranslationOptions};

fn be32(data: &[u8], offset: usize) -> Option<u32> {
    data.get(offset..offset + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn le32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Files of a TES5 (version 104) BSA archive.
fn read_bsa(path: &Path) -> Vec<(String, Vec<u8>)> {
    let data = std::fs::read(path).expect("read BSA");
    assert_eq!(&data[0..4], b"BSA\0", "not a BSA archive");
    let offset = le32(&data, 8) as usize;
    let archive_flags = le32(&data, 12);
    let folder_count = le32(&data, 16) as usize;
    let mut pos = offset;
    let mut folders = Vec::new();
    for _ in 0..folder_count {
        folders.push(le32(&data, pos + 8) as usize);
        pos += 16;
    }
    let mut files = Vec::new();
    for count in folders {
        let len = data[pos] as usize;
        let name = String::from_utf8_lossy(&data[pos + 1..pos + len]).trim_end_matches('\0').to_string();
        pos += 1 + len;
        for _ in 0..count {
            files.push((name.clone(), le32(&data, pos + 8), le32(&data, pos + 12) as usize));
            pos += 16;
        }
    }
    let mut out = Vec::new();
    for (folder, size, file_offset) in files {
        let end = data[pos..].iter().position(|&b| b == 0).unwrap();
        let name = String::from_utf8_lossy(&data[pos..pos + end]).to_string();
        pos += end + 1;
        let compressed = (size & 0x4000_0000 != 0) != (archive_flags & 4 != 0);
        let mut body = &data[file_offset..file_offset + (size & 0x3FFF_FFFF) as usize];
        if archive_flags & 0x100 != 0 {
            let n = body[0] as usize;
            body = &body[1 + n..];
        }
        let content = if compressed {
            miniz_oxide::inflate::decompress_to_vec_zlib(&body[4..]).expect("inflate")
        } else {
            body.to_vec()
        };
        out.push((format!("{folder}\\{name}"), content));
    }
    out
}

fn read_dir(dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
    for entry in std::fs::read_dir(dir).expect("read dir") {
        let path: PathBuf = entry.unwrap().path();
        if path.is_dir() {
            read_dir(&path, out);
        } else {
            out.push((path.display().to_string(), std::fs::read(&path).unwrap()));
        }
    }
}

/// Microcode of every Xbox 360 shader container (magic 0x102A1100 pixel / 0x102A1101 vertex).
fn extract_shaders(data: &[u8]) -> Vec<(usize, ShaderStage, Vec<u32>)> {
    let mut shaders = Vec::new();
    let mut i = 0;
    while i + 0x60 <= data.len() {
        let magic = be32(data, i).unwrap();
        if magic & !1 != 0x102A_1100 {
            i += 1;
            continue;
        }
        let parse = || -> Option<Vec<u32>> {
            let virtual_size = be32(data, i + 4)? as usize;
            let physical_size = be32(data, i + 8)? as usize;
            let shader_offset = be32(data, i + 0x18)? as usize;
            let physical_offset = be32(data, i + shader_offset)? as usize;
            let size = be32(data, i + shader_offset + 4)? as usize;
            if shader_offset >= virtual_size
                || size == 0
                || !size.is_multiple_of(12)
                || physical_offset + size > physical_size
            {
                return None;
            }
            let start = i + virtual_size + physical_offset;
            let bytes = data.get(start..start + size)?;
            Some(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_be_bytes(*c)).collect())
        };
        match parse() {
            Some(ucode) => {
                let stage = if magic & 1 != 0 { ShaderStage::Vertex } else { ShaderStage::Pixel };
                shaders.push((i, stage, ucode));
                i += 4;
            }
            None => i += 1,
        }
    }
    shaders
}

#[test]
#[ignore]
fn translate_game_shaders() {
    let Ok(source) = std::env::var("XENOS_SHADER_SOURCE") else {
        eprintln!("XENOS_SHADER_SOURCE not set, skipping");
        return;
    };
    let source = PathBuf::from(source);
    let mut files = Vec::new();
    if source.is_dir() {
        read_dir(&source, &mut files);
    } else {
        files = read_bsa(&source);
    }
    std::panic::set_hook(Box::new(|_| {}));
    let dump_dir = std::env::var("XENOS_SHADER_DUMP").ok().map(PathBuf::from);
    let mut seen = std::collections::HashSet::new();
    let (mut total, mut unique, mut vertex, mut pixel) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    let mut warnings: BTreeMap<String, usize> = BTreeMap::new();
    let mut features: BTreeMap<&str, usize> = BTreeMap::new();
    for (name, data) in &files {
        for (offset, stage, ucode) in extract_shaders(data) {
            total += 1;
            if !seen.insert((stage, ucode.clone())) {
                continue;
            }
            unique += 1;
            match stage {
                ShaderStage::Vertex => vertex += 1,
                ShaderStage::Pixel => pixel += 1,
            }
            let id = format!("{name}@{offset:#x}");
            let shader = match Shader::new(stage, &ucode) {
                Ok(s) => s,
                Err(e) => {
                    failures.push(format!("{id}: {e}"));
                    continue;
                }
            };
            let disassembly = shader.disassemble();
            let translation = shader.translate(&TranslationOptions::default());
            if let Some(dir) = &dump_dir {
                let base = dir.join(format!("{unique:05}_{:?}", stage));
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(base.with_extension("txt"), &disassembly).unwrap();
                std::fs::write(base.with_extension("wgsl"), &translation.wgsl).unwrap();
            }
            for w in &translation.warnings {
                *warnings.entry(w.clone()).or_default() += 1;
            }
            let info = shader.info();
            for (feature, used) in [
                ("loops", info.uses_loops),
                ("calls", info.uses_calls),
                ("jumps", info.uses_jumps),
                ("predicates", info.uses_predicate),
                ("kill", info.kills_pixels),
                ("depth write", info.writes_depth),
                ("memexport", info.uses_memexport()),
                ("dynamic registers", info.uses_register_dynamic_addressing),
                ("dynamic constants", info.float_dynamic_addressing),
                ("texture fetch", !info.texture_fetches.is_empty()),
                ("vertex fetch", !info.vertex_bindings.is_empty()),
            ] {
                if used {
                    *features.entry(feature).or_default() += 1;
                }
            }
            if !translation.structured_control_flow {
                *features.entry("dispatch loop").or_default() += 1;
            }
            let result = std::panic::catch_unwind(|| common::validate_wgsl(&translation.wgsl));
            if let Err(e) = result {
                let message = e
                    .downcast_ref::<String>()
                    .map(|s| s.lines().take(8).collect::<Vec<_>>().join("\n"))
                    .unwrap_or_default();
                failures.push(format!("{id} ({stage:?}): {message}"));
            }
        }
    }
    let _ = std::panic::take_hook();
    println!("{total} shader blobs, {unique} unique ({vertex} vertex, {pixel} pixel)");
    println!("features: {features:?}");
    for (w, n) in &warnings {
        println!("warning x{n}: {w}");
    }
    for f in failures.iter().take(20) {
        println!("FAILED {f}");
    }
    assert!(failures.is_empty(), "{} of {unique} shaders failed", failures.len());
}
