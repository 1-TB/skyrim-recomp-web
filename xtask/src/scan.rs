//! Finds the addresses config/skyrim.toml.template asks for: the compiler's
//! register save/restore helpers (fixed byte patterns) and candidates for
//! longjmp (functions that call RtlUnwind).

use std::path::Path;

/// (config key, pattern) from XenonRecomp's README.
const PATTERNS: &[(&str, &[u8])] = &[
    ("restgprlr_14_address", &[0xE9, 0xC1, 0xFF, 0x68]),
    ("savegprlr_14_address", &[0xF9, 0xC1, 0xFF, 0x68]),
    ("restfpr_14_address", &[0xC9, 0xCC, 0xFF, 0x70]),
    ("savefpr_14_address", &[0xD9, 0xCC, 0xFF, 0x70]),
    ("restvmx_14_address", &[0x39, 0x60, 0xFE, 0xE0, 0x7D, 0xCB, 0x60, 0xCE]),
    ("savevmx_14_address", &[0x39, 0x60, 0xFE, 0xE0, 0x7D, 0xCB, 0x61, 0xCE]),
    ("restvmx_64_address", &[0x39, 0x60, 0xFC, 0x00, 0x10, 0x0B, 0x60, 0xCB]),
    ("savevmx_64_address", &[0x39, 0x60, 0xFC, 0x00, 0x10, 0x0B, 0x61, 0xCB]),
];

const RTL_UNWIND_ORDINAL: u16 = 0x147;

pub fn scan(xex_path: &Path) -> Result<(), String> {
    let data = std::fs::read(xex_path).map_err(|e| format!("{}: {e}", xex_path.display()))?;
    let xex = xex::Xex::parse(&data).map_err(|e| format!("{}: {e}", xex_path.display()))?;
    let base = xex.image_base();
    let image = xex.image();
    let code: Vec<(u32, u32)> = xex
        .sections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|s| s.is_code())
        .map(|s| (s.virtual_address, s.virtual_size))
        .collect();

    println!("# candidates for config/skyrim.toml (verify each in a disassembler)");
    for (key, pattern) in PATTERNS {
        let mut hits = Vec::new();
        for &(rva, size) in &code {
            let start = rva as usize;
            let end = (start + size as usize).min(image.len());
            let section = &image[start..end];
            let mut off = 0;
            while off + pattern.len() <= section.len() {
                if &section[off..off + pattern.len()] == *pattern {
                    hits.push(base + rva + off as u32);
                }
                off += 4;
            }
        }
        match hits.as_slice() {
            [one] => println!("{key} = {one:#010X}"),
            [] => println!("# {key}: pattern not found"),
            many => println!(
                "# {key}: {} matches: {}",
                many.len(),
                many.iter().map(|h| format!("{h:#010X}")).collect::<Vec<_>>().join(", ")
            ),
        }
    }

    // bl instructions that target the RtlUnwind import thunk.
    let imports = xex.imports().map_err(|e| e.to_string())?;
    let thunk = imports
        .iter()
        .find(|i| i.library.eq_ignore_ascii_case("xboxkrnl.exe") && i.ordinal == RTL_UNWIND_ORDINAL)
        .and_then(|i| i.thunk_addr);
    let Some(thunk) = thunk else {
        println!("# RtlUnwind is not imported; the title may not use setjmp/longjmp");
        return Ok(());
    };
    let mut callers = Vec::new();
    for &(rva, size) in &code {
        let mut off = 0u32;
        while off + 4 <= size {
            let addr = base + rva + off;
            let at = (rva + off) as usize;
            if at + 4 > image.len() {
                break;
            }
            let insn = u32::from_be_bytes(image[at..at + 4].try_into().unwrap());
            // I-form branch with LK=1, AA=0.
            if insn >> 26 == 18 && insn & 3 == 1 {
                let disp = ((insn & 0x03FF_FFFC) as i32) << 6 >> 6;
                if addr.wrapping_add(disp as u32) == thunk {
                    callers.push(addr);
                }
            }
            off += 4;
        }
    }
    println!(
        "# RtlUnwind thunk {thunk:#010X} is called from: {} \
         (the function containing such a call is usually longjmp; setjmp typically follows it)",
        callers.iter().map(|c| format!("{c:#010X}")).collect::<Vec<_>>().join(", ")
    );
    Ok(())
}
