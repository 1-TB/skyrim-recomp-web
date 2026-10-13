//! printf-style formatting of guest varargs (DbgPrint and friends).
//!
//! On the Xbox 360 every variadic argument occupies one 64-bit GPR slot
//! (doubles included), continuing from the registers into the caller's
//! parameter area, so arguments are consumed as a flat sequence of u64s.

use runtime_core::memory::GuestMemory;

pub fn format(mem: &GuestMemory, fmt: &[u8], mut next: impl FnMut() -> u64) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < fmt.len() {
        let c = fmt[i];
        i += 1;
        if c != b'%' {
            out.push(c as char);
            continue;
        }
        if i < fmt.len() && fmt[i] == b'%' {
            out.push('%');
            i += 1;
            continue;
        }

        let mut left = false;
        let mut zero = false;
        let mut plus = false;
        let mut space = false;
        let mut alt = false;
        while i < fmt.len() {
            match fmt[i] {
                b'-' => left = true,
                b'0' => zero = true,
                b'+' => plus = true,
                b' ' => space = true,
                b'#' => alt = true,
                _ => break,
            }
            i += 1;
        }
        let mut width = 0usize;
        if i < fmt.len() && fmt[i] == b'*' {
            width = next() as i32 as usize;
            i += 1;
        } else {
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                width = width * 10 + (fmt[i] - b'0') as usize;
                i += 1;
            }
        }
        let mut precision: Option<usize> = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            let mut p = 0usize;
            if i < fmt.len() && fmt[i] == b'*' {
                p = next() as i32 as usize;
                i += 1;
            } else {
                while i < fmt.len() && fmt[i].is_ascii_digit() {
                    p = p * 10 + (fmt[i] - b'0') as usize;
                    i += 1;
                }
            }
            precision = Some(p);
        }

        // Length modifiers: h, hh, l, ll, w, I64, I32, z.
        let mut long64 = false;
        let mut wide = false;
        let mut short = false;
        loop {
            if i >= fmt.len() {
                break;
            }
            match fmt[i] {
                b'h' => short = true,
                b'l' => {
                    if i + 1 < fmt.len() && fmt[i + 1] == b'l' {
                        long64 = true;
                        i += 1;
                    } else {
                        wide = true;
                    }
                }
                b'w' => wide = true,
                b'z' => {}
                b'I' if fmt[i..].starts_with(b"I64") => {
                    long64 = true;
                    i += 2;
                }
                b'I' if fmt[i..].starts_with(b"I32") => i += 2,
                _ => break,
            }
            i += 1;
        }
        if i >= fmt.len() {
            break;
        }
        let conv = fmt[i];
        i += 1;

        let body = match conv {
            b'd' | b'i' => {
                let raw = next();
                let v = if long64 {
                    raw as i64
                } else if short {
                    raw as i16 as i64
                } else {
                    raw as i32 as i64
                };
                let mut s = v.unsigned_abs().to_string();
                if let Some(p) = precision {
                    s = format!("{s:0>p$}");
                }
                let sign = if v < 0 {
                    "-"
                } else if plus {
                    "+"
                } else if space {
                    " "
                } else {
                    ""
                };
                pad_number(sign, &s, width, left, zero && precision.is_none())
            }
            b'u' | b'x' | b'X' | b'o' => {
                let raw = next();
                let v = if long64 {
                    raw
                } else if short {
                    raw as u16 as u64
                } else {
                    raw as u32 as u64
                };
                let mut s = match conv {
                    b'u' => v.to_string(),
                    b'x' => format!("{v:x}"),
                    b'X' => format!("{v:X}"),
                    _ => format!("{v:o}"),
                };
                if let Some(p) = precision {
                    s = format!("{s:0>p$}");
                }
                let prefix = match (alt && v != 0, conv) {
                    (true, b'x') => "0x",
                    (true, b'X') => "0X",
                    (true, b'o') => "0",
                    _ => "",
                };
                pad_number(prefix, &s, width, left, zero && precision.is_none())
            }
            b'p' => pad(&format!("{:08X}", next() as u32), width, left),
            b'c' | b'C' => {
                let v = next() as u32;
                let ch =
                    if wide || conv == b'C' { char::from_u32(v & 0xFFFF).unwrap_or('?') } else { (v as u8) as char };
                pad(&ch.to_string(), width, left)
            }
            b's' | b'S' => {
                let ptr = next() as u32;
                let wide_str = (conv == b's' && wide) || (conv == b'S' && !short);
                let mut s = if ptr == 0 {
                    "(null)".to_string()
                } else if wide_str {
                    read_wide(mem, ptr, precision)
                } else {
                    let limit = precision.unwrap_or(4096) as u32;
                    String::from_utf8_lossy(&mem.read_cstr(ptr, limit)).into_owned()
                };
                if let Some(p) = precision {
                    s = s.chars().take(p).collect();
                }
                pad(&s, width, left)
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                let v = f64::from_bits(next());
                let p = precision.unwrap_or(6);
                let s = match conv {
                    b'e' => format!("{v:.p$e}"),
                    b'E' => format!("{v:.p$E}"),
                    b'g' | b'G' => {
                        let s = format!("{v}");
                        if conv == b'G' { s.to_uppercase() } else { s }
                    }
                    _ => format!("{v:.p$}"),
                };
                let sign = if plus && v >= 0.0 { "+" } else { "" };
                pad_number(sign, &s, width, left, zero)
            }
            b'n' => {
                next();
                String::new()
            }
            other => {
                // Unknown conversion: echo it so the log still shows the format.
                format!("%{}", other as char)
            }
        };
        out.push_str(&body);
    }
    out
}

fn read_wide(mem: &GuestMemory, ptr: u32, limit: Option<usize>) -> String {
    let max = limit.unwrap_or(4096);
    let mut units = Vec::new();
    while units.len() < max {
        let u = mem.read_u16(ptr + units.len() as u32 * 2);
        if u == 0 {
            break;
        }
        units.push(u);
    }
    String::from_utf16_lossy(&units)
}

fn pad(s: &str, width: usize, left: bool) -> String {
    if left { format!("{s:<width$}") } else { format!("{s:>width$}") }
}

fn pad_number(prefix: &str, digits: &str, width: usize, left: bool, zero: bool) -> String {
    let len = prefix.len() + digits.len();
    if len >= width {
        return format!("{prefix}{digits}");
    }
    let fill = width - len;
    if left {
        format!("{prefix}{digits}{}", " ".repeat(fill))
    } else if zero {
        format!("{prefix}{}{digits}", "0".repeat(fill))
    } else {
        format!("{}{prefix}{digits}", " ".repeat(fill))
    }
}
