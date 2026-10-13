//! Runtime library exports: strings, memory helpers, time conversion, XEX
//! header lookup.
#![allow(non_snake_case)]

use crate::dispatch::{Call, Export, ModuleId};
use crate::status;
use crate::strings::{narrow, read_ansi, read_unicode, widen, write_string_header};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x12C, RtlInitAnsiString),
    export!(K, 0x12D, RtlInitUnicodeString),
    export!(K, 0x127, RtlFreeAnsiString),
    export!(K, 0x128, RtlFreeUnicodeString),
    export!(K, 0x142, RtlUnicodeStringToAnsiString),
    export!(K, 0x114, RtlAnsiStringToUnicodeString),
    export!(K, 0x133, RtlMultiByteToUnicodeN),
    export!(K, 0x143, RtlUnicodeToMultiByteN),
    export!(K, 0x11C, RtlCompareString),
    export!(K, 0x11D, RtlCompareStringN),
    export!(K, 0x121, RtlCopyString),
    export!(K, 0x149, RtlUpcaseUnicodeChar),
    export!(K, 0x11A, RtlCompareMemory),
    export!(K, 0x11B, RtlCompareMemoryUlong),
    export!(K, 0x126, RtlFillMemoryUlong),
    export!(K, 0x135, RtlNtStatusToDosError),
    export!(K, 0x12B, RtlImageXexHeaderField),
    export!(K, 0x140, RtlTimeToTimeFields),
    export!(K, 0x13F, RtlTimeFieldsToTime),
];

/// Longest string the Init routines will measure (MAXUSHORT-sized buffers).
const MAX_STRING_SCAN: u32 = 0xFFFE;

fn RtlInitAnsiString(c: &mut Call) -> u64 {
    let (dest, src) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    if src == 0 {
        write_string_header(mem, dest, 0, 0, 0);
    } else {
        let len = mem.read_cstr(src, MAX_STRING_SCAN).len() as u16;
        write_string_header(mem, dest, len, len + 1, src);
    }
    0
}

fn RtlInitUnicodeString(c: &mut Call) -> u64 {
    let (dest, src) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    if src == 0 {
        write_string_header(mem, dest, 0, 0, 0);
    } else {
        let mut units = 0u32;
        while units < MAX_STRING_SCAN / 2 && mem.read_u16(src + units * 2) != 0 {
            units += 1;
        }
        let bytes = (units * 2) as u16;
        write_string_header(mem, dest, bytes, bytes + 2, src);
    }
    0
}

/// Frees a buffer allocated by an Rtl conversion with AllocateDestination set.
fn free_string(c: &Call, string_ptr: u32) {
    let mem = c.mem();
    let buf = mem.read_u32(string_ptr + 4);
    if buf != 0
        && let Err(e) = c.kernel.rt.space.system_free(buf)
    {
        log::warn!("RtlFree*String({string_ptr:#010X}): buffer {buf:#010X} was not allocated by Rtl: {e}");
    }
    write_string_header(mem, string_ptr, 0, 0, 0);
}

fn RtlFreeAnsiString(c: &mut Call) -> u64 {
    free_string(c, c.arg(0));
    0
}

fn RtlFreeUnicodeString(c: &mut Call) -> u64 {
    free_string(c, c.arg(0));
    0
}

/// Writes `data` (already terminated) into `dest`, allocating if asked.
/// `len` is the string length in bytes excluding the terminator.
fn store_converted(c: &Call, dest: u32, data: &[u8], len: u16, allocate: bool) -> u32 {
    let mem = c.mem();
    if allocate {
        let Ok(buf) = c.kernel.rt.space.system_alloc(data.len() as u32, 4) else {
            return status::NO_MEMORY;
        };
        mem.write_bytes(buf, data);
        write_string_header(mem, dest, len, data.len() as u16, buf);
        return status::SUCCESS;
    }
    let max = mem.read_u16(dest + 2) as usize;
    let buf = mem.read_u32(dest + 4);
    if max < data.len() {
        return status::BUFFER_OVERFLOW;
    }
    mem.write_bytes(buf, data);
    mem.write_u16(dest, len);
    status::SUCCESS
}

fn RtlUnicodeStringToAnsiString(c: &mut Call) -> u64 {
    let (dest, src, allocate) = (c.arg(0), c.arg(1), c.arg(2) != 0);
    let mut bytes = narrow(&read_unicode(c.mem(), src));
    log::debug!("RtlUnicodeStringToAnsiString: {:?}", String::from_utf8_lossy(&bytes));
    let len = bytes.len() as u16;
    bytes.push(0);
    store_converted(c, dest, &bytes, len, allocate) as u64
}

fn RtlAnsiStringToUnicodeString(c: &mut Call) -> u64 {
    let (dest, src, allocate) = (c.arg(0), c.arg(1), c.arg(2) != 0);
    let units = widen(&read_ansi(c.mem(), src));
    let mut bytes: Vec<u8> = units.iter().flat_map(|u| u.to_be_bytes()).collect();
    let len = bytes.len() as u16;
    bytes.extend_from_slice(&[0, 0]);
    store_converted(c, dest, &bytes, len, allocate) as u64
}

fn RtlMultiByteToUnicodeN(c: &mut Call) -> u64 {
    let (out, out_max_bytes, written_ptr, src, src_bytes) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let mem = c.mem();
    let n = (out_max_bytes / 2).min(src_bytes);
    for i in 0..n {
        mem.write_u16(out + i * 2, mem.read_u8(src + i) as u16);
    }
    if written_ptr != 0 {
        mem.write_u32(written_ptr, n * 2);
    }
    status::SUCCESS as u64
}

fn RtlUnicodeToMultiByteN(c: &mut Call) -> u64 {
    let (out, out_max_bytes, written_ptr, src, src_bytes) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let mem = c.mem();
    let n = out_max_bytes.min(src_bytes / 2);
    for i in 0..n {
        let u = mem.read_u16(src + i * 2);
        mem.write_u8(out + i, if u <= 0xFF { u as u8 } else { b'?' });
    }
    if written_ptr != 0 {
        mem.write_u32(written_ptr, n);
    }
    status::SUCCESS as u64
}

fn compare_bytes(a: &[u8], b: &[u8], case_insensitive: bool) -> i32 {
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = if case_insensitive { (x.to_ascii_uppercase(), y.to_ascii_uppercase()) } else { (x, y) };
        if x != y {
            return x as i32 - y as i32;
        }
    }
    a.len() as i32 - b.len() as i32
}

fn RtlCompareString(c: &mut Call) -> u64 {
    let (a, b, ci) = (c.arg(0), c.arg(1), c.arg(2) != 0);
    compare_bytes(&read_ansi(c.mem(), a), &read_ansi(c.mem(), b), ci) as u32 as u64
}

/// (const char* s1, u32 len1, const char* s2, u32 len2, BOOLEAN case_insensitive)
fn RtlCompareStringN(c: &mut Call) -> u64 {
    let (a, alen, b, blen, ci) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4) != 0);
    let mem = c.mem();
    let sa = mem.read_cstr(a, alen);
    let sb = mem.read_cstr(b, blen);
    compare_bytes(&sa, &sb, ci) as u32 as u64
}

fn RtlCopyString(c: &mut Call) -> u64 {
    let (dest, src) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    if src == 0 {
        mem.write_u16(dest, 0);
        return 0;
    }
    let data = read_ansi(mem, src);
    let max = mem.read_u16(dest + 2) as usize;
    let n = data.len().min(max);
    mem.write_bytes(mem.read_u32(dest + 4), &data[..n]);
    mem.write_u16(dest, n as u16);
    0
}

fn RtlUpcaseUnicodeChar(c: &mut Call) -> u64 {
    let ch = c.arg(0) as u16;
    char::from_u32(ch as u32)
        .and_then(|c| {
            let mut up = c.to_uppercase();
            let first = up.next()?;
            (up.next().is_none() && (first as u32) <= 0xFFFF).then_some(first as u32)
        })
        .unwrap_or(ch as u32) as u64
}

fn RtlCompareMemory(c: &mut Call) -> u64 {
    let (a, b, len) = (c.arg(0), c.arg(1), c.arg(2));
    let mem = c.mem();
    (0..len).take_while(|&i| mem.read_u8(a + i) == mem.read_u8(b + i)).count() as u64
}

fn RtlCompareMemoryUlong(c: &mut Call) -> u64 {
    let (src, len, pattern) = (c.arg(0), c.arg(1), c.arg(2));
    let mem = c.mem();
    let words = len / 4;
    let matching = (0..words).take_while(|&i| mem.read_u32(src + i * 4) == pattern).count() as u32;
    (matching * 4) as u64
}

fn RtlFillMemoryUlong(c: &mut Call) -> u64 {
    let (dest, len, pattern) = (c.arg(0), c.arg(1), c.arg(2));
    let mem = c.mem();
    for i in 0..len / 4 {
        mem.write_u32(dest + i * 4, pattern);
    }
    0
}

fn RtlNtStatusToDosError(c: &mut Call) -> u64 {
    status::to_dos_error(c.arg(0)) as u64
}

/// Reads the optional-header table of any XEX header in guest memory.
fn RtlImageXexHeaderField(c: &mut Call) -> u64 {
    let (header, key) = (c.arg(0), c.arg(1));
    if header == 0 {
        return 0;
    }
    let mem = c.mem();
    let count = mem.read_u32(header + 0x14);
    for i in 0..count.min(0x100) {
        let entry = header + xex::OPT_HEADER_TABLE_OFFSET + i * 8;
        if mem.read_u32(entry) != key {
            continue;
        }
        return match key & 0xFF {
            0x00 => mem.read_u32(entry + 4),
            0x01 => entry + 4,
            _ => header + mem.read_u32(entry + 4),
        } as u64;
    }
    0
}

// --- TIME_FIELDS <-> FILETIME ---------------------------------------------------
//
// TIME_FIELDS: u16 year, month, day, hour, minute, second, milliseconds, weekday.

const TICKS_PER_MS: u64 = 10_000;
const TICKS_PER_DAY: u64 = 86_400_000 * TICKS_PER_MS;
/// Days from 1601-01-01 to 1970-01-01.
const EPOCH_1601_TO_1970_DAYS: i64 = 134_774;

/// Days since 1970-01-01 to (year, month, day) — Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn RtlTimeToTimeFields(c: &mut Call) -> u64 {
    let (time_ptr, fields) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    let t = mem.read_u64(time_ptr);
    let days = (t / TICKS_PER_DAY) as i64;
    let rem = t % TICKS_PER_DAY;
    let (y, m, d) = civil_from_days(days - EPOCH_1601_TO_1970_DAYS);
    let ms_of_day = rem / TICKS_PER_MS;
    // 1601-01-01 was a Monday.
    let weekday = ((days + 1) % 7) as u16;
    for (i, v) in [
        y as u16,
        m as u16,
        d as u16,
        (ms_of_day / 3_600_000) as u16,
        (ms_of_day / 60_000 % 60) as u16,
        (ms_of_day / 1000 % 60) as u16,
        (ms_of_day % 1000) as u16,
        weekday,
    ]
    .into_iter()
    .enumerate()
    {
        mem.write_u16(fields + i as u32 * 2, v);
    }
    0
}

fn RtlTimeFieldsToTime(c: &mut Call) -> u64 {
    let (fields, time_ptr) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    let f: Vec<u16> = (0..7).map(|i| mem.read_u16(fields + i * 2)).collect();
    let (year, month, day, hour, minute, second, ms) = (f[0], f[1], f[2], f[3], f[4], f[5], f[6]);
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if year < 1601
        || !(1..=12).contains(&month)
        || day < 1
        || day > month_days[month as usize - 1]
        || hour > 23
        || minute > 59
        || second > 59
        || ms > 999
    {
        return 0;
    }
    let days = days_from_civil(year as i64, month as u32, day as u32) + EPOCH_1601_TO_1970_DAYS;
    let ms_total = ((hour as u64 * 60 + minute as u64) * 60 + second as u64) * 1000 + ms as u64;
    mem.write_u64(time_ptr, days as u64 * TICKS_PER_DAY + ms_total * TICKS_PER_MS);
    1
}
