mod common;

use common::{guest, kernel, scratch};
use kernel_hle::Kernel;
use kernel_hle::dispatch::ModuleId;
use kernel_hle::status;
use runtime_core::PPCContext;

// --- LZX (LDI*) ------------------------------------------------------------------

/// Writes LZX bits the way the decoder reads them: 16-bit little-endian words,
/// most significant bit first.
#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    word: u16,
    used: u32,
}

impl BitWriter {
    fn bits(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.word = (self.word << 1) | ((value >> i) & 1) as u16;
            self.used += 1;
            if self.used == 16 {
                self.out.extend_from_slice(&self.word.to_le_bytes());
                self.word = 0;
                self.used = 0;
            }
        }
    }

    fn align(&mut self) {
        if self.used != 0 {
            self.bits(0, 16 - self.used);
        }
    }
}

/// One LZX block of stored (uncompressed) data, as a compressor emits for
/// incompressible input: the stream header on the first block, block type 3,
/// the 24-bit size, alignment, the repeat offsets R0-R2, then the bytes. A
/// stored block of odd length is followed by a padding byte, which the decoder
/// skips when it starts the next block (here, the next call's input).
fn stored_block(data: &[u8], first: bool, after_odd: bool) -> Vec<u8> {
    let mut w = BitWriter::default();
    if after_odd {
        w.out.push(0);
    }
    if first {
        w.bits(0, 1); // no E8 translation
    }
    w.bits(3, 3);
    w.bits(data.len() as u32, 24);
    w.align();
    let mut out = w.out;
    for _ in 0..3 {
        out.extend_from_slice(&1u32.to_le_bytes());
    }
    out.extend_from_slice(data);
    out
}

const MDI_BAD_PARAMETERS: u64 = 2;
const MDI_FAILED: u64 = 4;
const MDI_CONFIGURATION: u64 = 5;

#[test]
fn ldi_decompresses_consecutive_blocks_of_one_stream() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x10000);
    let mem = k.mem();
    let (block_max, config, src_min, handle_ptr, dst_len) = (io, io + 0x10, io + 0x20, io + 0x24, io + 0x28);
    let (src, dst) = (io + 0x1000, io + 0x8000);

    // Size query: no handle, block size defaults to an LZX frame.
    mem.write_u32(block_max, 0);
    mem.write_u32(config, 0x10000);
    mem.write_u32(config + 4, 0);
    assert_eq!(k.call_export("LDICreateDecompression", &mut ctx, &[block_max, config, 0, 0, src_min, 0]), 0);
    assert_eq!(mem.read_u32(block_max), 0x8000);
    assert_eq!(mem.read_u32(src_min), 0x8000 + 6144);
    // Window sizes outside LZX's range are a configuration error.
    mem.write_u32(config, 12345);
    let r = k.call_export("LDICreateDecompression", &mut ctx, &[block_max, config, 0, 0, src_min, handle_ptr]);
    assert_eq!(r, MDI_CONFIGURATION);
    mem.write_u32(config, 0x10000);
    assert_eq!(k.call_export("LDICreateDecompression", &mut ctx, &[block_max, config, 0, 0, src_min, handle_ptr]), 0);
    let h = mem.read_u32(handle_ptr);
    assert_ne!(h, 0);

    let first: Vec<u8> = (0..1001u32).map(|i| (i * 7) as u8).collect();
    let second = b"the second block continues the stream".to_vec();
    for (data, is_first) in [(&first, true), (&second, false)] {
        let block = stored_block(data, is_first, !is_first);
        mem.write_bytes(src, &block);
        mem.write_u32(dst_len, data.len() as u32);
        let r = k.call_export("LDIDecompress", &mut ctx, &[h, src, block.len() as u32, dst, dst_len]);
        assert_eq!(r, 0);
        assert_eq!(mem.read_u32(dst_len), data.len() as u32);
        assert_eq!(&mem.read_vec(dst, data.len() as u32), data);
    }

    // After a reset the next block starts a stream again (with its header).
    assert_eq!(k.call_export("LDIResetDecompression", &mut ctx, &[h]), 0);
    let block = stored_block(&second, true, false);
    mem.write_bytes(src, &block);
    mem.write_u32(dst_len, second.len() as u32);
    assert_eq!(k.call_export("LDIDecompress", &mut ctx, &[h, src, block.len() as u32, dst, dst_len]), 0);
    assert_eq!(mem.read_vec(dst, second.len() as u32), second);

    // Garbage fails the call, not the thread.
    mem.fill(src, 64, 0xFF);
    mem.write_u32(dst_len, 0x100);
    assert_eq!(k.call_export("LDIDecompress", &mut ctx, &[h, src, 64, dst, dst_len]), MDI_FAILED);
    assert_eq!(mem.read_u32(dst_len), 0);
    // More output than a block holds is refused.
    mem.write_u32(dst_len, 0x8001);
    assert_eq!(k.call_export("LDIDecompress", &mut ctx, &[h, src, 64, dst, dst_len]), MDI_BAD_PARAMETERS);

    assert_eq!(k.call_export("LDIDestroyDecompression", &mut ctx, &[h]), 0);
    assert_eq!(k.call_export("LDIDestroyDecompression", &mut ctx, &[h]), MDI_BAD_PARAMETERS);
    mem.write_u32(dst_len, 4);
    assert_eq!(k.call_export("LDIDecompress", &mut ctx, &[h, src, 64, dst, dst_len]), MDI_BAD_PARAMETERS);
}

// --- Strings, characters, randomness -----------------------------------------------

fn wstr(k: &Kernel, at: u32, s: &str) -> u32 {
    let mut units: Vec<u16> = s.encode_utf16().collect();
    units.push(0);
    k.mem().write_utf16(at, &units);
    at
}

/// _snwprintf(buffer, count, format, varargs...) with 64-bit vararg slots: r6-r10,
/// then the caller's parameter area.
fn snwprintf(k: &'static Kernel, ctx: &mut PPCContext, buffer: u32, count: i32, format: u32, args: &[u64]) -> i32 {
    ctx.set_gpr(3, buffer as u64);
    ctx.set_gpr(4, count as u32 as u64);
    ctx.set_gpr(5, format as u64);
    for (i, &a) in args.iter().enumerate() {
        let n = 3 + i;
        if n < 8 {
            ctx.set_gpr(3 + n, a);
        } else {
            k.mem().write_u64(ctx.gpr32(1) + 0x50 + (n as u32 - 8) * 8, a);
        }
    }
    k.dispatch(ModuleId::XBOXKRNL, 0x13D, ctx);
    ctx.gpr32(3) as i32
}

#[test]
fn snwprintf_formats_wide_with_msvc_string_conventions() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x2000);
    let mem = k.mem();
    let buf = io + 0x1000;
    let name = wstr(k, io, "Whiterun");
    let narrow = io + 0x100;
    mem.write_bytes(narrow, b"abc\0");
    let fmt = wstr(k, io + 0x200, "%s=%d %S %5.2f|%-4c|%hs");
    let n = snwprintf(
        k,
        &mut ctx,
        buf,
        64,
        fmt,
        &[name as u64, -42i64 as u64, narrow as u64, 1.23456f64.to_bits(), 'x' as u64, narrow as u64],
    );
    let expected = "Whiterun=-42 abc  1.23|x   |abc";
    assert_eq!(n, expected.len() as i32);
    let units: Vec<u16> = expected.encode_utf16().chain([0]).collect();
    assert_eq!(mem.read_utf16(buf, units.len() as u32), units);

    // C float forms: %e has a signed, two-digit exponent; %g drops trailing zeros.
    let fmt = wstr(k, io + 0x200, "%e %g %g %G %.3g");
    let args = [1500.0f64, 0.0001, 1234567.0, 1e-5, 2.5].map(|v: f64| v.to_bits());
    let n = snwprintf(k, &mut ctx, buf, 64, fmt, &args);
    let expected = "1.500000e+03 0.0001 1.23457e+06 1E-05 2.5";
    assert_eq!(String::from_utf16(&mem.read_utf16(buf, n as u32)).unwrap(), expected);

    // Exactly full: no terminator, length returned. Too long: -1, truncated.
    let fmt = wstr(k, io + 0x200, "%d");
    mem.write_u16(buf + 8, 0xAAAA);
    assert_eq!(snwprintf(k, &mut ctx, buf, 4, fmt, &[1234]), 4);
    assert_eq!(mem.read_u16(buf + 8), 0xAAAA, "no terminator past the buffer");
    assert_eq!(snwprintf(k, &mut ctx, buf, 3, fmt, &[98765]), -1);
    assert_eq!(mem.read_utf16(buf, 3), "987".encode_utf16().collect::<Vec<_>>());
    assert_eq!(snwprintf(k, &mut ctx, 0, 3, fmt, &[1]), -1);
}

#[test]
fn downcase_and_random() {
    let k = kernel();
    let mut ctx = guest(k);
    for (from, to) in [('A', 'a'), ('Ä', 'ä'), ('Ω', 'ω'), ('1', '1'), ('z', 'z')] {
        assert_eq!(k.call_export("RtlDowncaseUnicodeChar", &mut ctx, &[from as u32]), to as u64, "{from}");
    }

    let io = scratch(k, 0x1000);
    let mem = k.mem();
    mem.fill(io, 0x100, 0xEE);
    k.call_export("XeCryptRandom", &mut ctx, &[io, 61]);
    let a = mem.read_vec(io, 61);
    assert_eq!(mem.read_u8(io + 61), 0xEE, "wrote past the buffer");
    k.call_export("XeCryptRandom", &mut ctx, &[io, 61]);
    let b = mem.read_vec(io, 61);
    assert_ne!(a, b);
    assert!(a.iter().any(|&x| x != a[0]), "not random: {a:?}");
}

// --- Symbolic links -------------------------------------------------------------------

/// OBJECT_ATTRIBUTES naming `name` (an ANSI_STRING), case insensitive.
fn object_attributes(k: &Kernel, at: u32, name: &str) -> u32 {
    let mem = k.mem();
    let (attrs, string, text) = (at, at + 0x10, at + 0x20);
    mem.write_bytes(text, name.as_bytes());
    mem.write_u16(string, name.len() as u16);
    mem.write_u16(string + 2, name.len() as u16);
    mem.write_u32(string + 4, text);
    mem.write_u32(attrs, 0);
    mem.write_u32(attrs + 4, string);
    mem.write_u32(attrs + 8, 0x40);
    attrs
}

#[test]
fn symbolic_links_open_and_report_their_target() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    let handle_ptr = io;

    let attrs = object_attributes(k, io + 0x100, "\\??\\update:");
    assert_eq!(k.call_export("NtOpenSymbolicLinkObject", &mut ctx, &[handle_ptr, attrs]) as u32, status::NO_SUCH_FILE);

    k.vfs.add_symlink("update:", "\\Device\\Cdrom0\\$TitleUpdate");
    assert_eq!(k.call_export("NtOpenSymbolicLinkObject", &mut ctx, &[handle_ptr, attrs]), 0);
    let h = mem.read_u32(handle_ptr);
    let (target, text) = (io + 0x200, io + 0x300);
    let expected = b"\\Device\\Cdrom0\\$TitleUpdate";
    mem.write_u16(target, 0);
    mem.write_u16(target + 2, 0x40);
    mem.write_u32(target + 4, text);
    assert_eq!(k.call_export("NtQuerySymbolicLinkObject", &mut ctx, &[h, target]), 0);
    assert_eq!(mem.read_u16(target) as usize, expected.len());
    assert_eq!(mem.read_vec(text, expected.len() as u32), expected);
    // A short buffer gets the truncated target.
    mem.write_u16(target + 2, 8);
    assert_eq!(k.call_export("NtQuerySymbolicLinkObject", &mut ctx, &[h, target]), 0);
    assert_eq!(mem.read_u16(target), 8);
    assert_eq!(k.call_export("NtClose", &mut ctx, &[h]), 0);
    // The game drive is a link too.
    let attrs = object_attributes(k, io + 0x100, "game:");
    assert_eq!(k.call_export("NtOpenSymbolicLinkObject", &mut ctx, &[handle_ptr, attrs]), 0);
}
