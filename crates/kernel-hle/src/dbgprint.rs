//! printf-style formatting of guest varargs (DbgPrint, _snwprintf).
//!
//! On the Xbox 360 every variadic argument occupies one 64-bit GPR slot
//! (doubles included), continuing from the registers into the caller's
//! parameter area, so arguments are consumed as a flat sequence of u64s.
//!
//! Conversions follow the MSVC CRT: in a wide format (`wprintf` family) `%s`
//! and `%c` take wide arguments and `%S` / `%C` narrow ones, the reverse of
//! a narrow format; `h` forces narrow and `l` / `w` force wide in both.

use runtime_core::memory::GuestMemory;

/// Formats a narrow (`printf`) format string.
pub fn format(mem: &GuestMemory, fmt: &[u8], next: impl FnMut() -> u64) -> String {
    let chars: Vec<char> = fmt.iter().map(|&b| b as char).collect();
    format_core(mem, &chars, false, next)
}

/// Formats a wide (`wprintf`) format string, returning UTF-16.
pub fn format_wide(mem: &GuestMemory, fmt: &[u16], next: impl FnMut() -> u64) -> Vec<u16> {
    let chars: Vec<char> = char::decode_utf16(fmt.iter().copied()).map(|c| c.unwrap_or('\u{FFFD}')).collect();
    format_core(mem, &chars, true, next).encode_utf16().collect()
}

fn format_core(mem: &GuestMemory, fmt: &[char], wide_format: bool, mut next: impl FnMut() -> u64) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < fmt.len() {
        let c = fmt[i];
        i += 1;
        if c != '%' {
            out.push(c);
            continue;
        }
        if i < fmt.len() && fmt[i] == '%' {
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
                '-' => left = true,
                '0' => zero = true,
                '+' => plus = true,
                ' ' => space = true,
                '#' => alt = true,
                _ => break,
            }
            i += 1;
        }
        let mut width = 0usize;
        if i < fmt.len() && fmt[i] == '*' {
            let w = next() as i32;
            // A negative width argument means left alignment.
            if w < 0 {
                left = true;
            }
            width = w.unsigned_abs() as usize;
            i += 1;
        } else {
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                width = width * 10 + fmt[i].to_digit(10).unwrap() as usize;
                i += 1;
            }
        }
        let mut precision: Option<usize> = None;
        if i < fmt.len() && fmt[i] == '.' {
            i += 1;
            let mut p = 0usize;
            if i < fmt.len() && fmt[i] == '*' {
                // A negative precision argument is taken as omitted.
                let v = next() as i32;
                i += 1;
                precision = (v >= 0).then_some(v as usize);
            } else {
                while i < fmt.len() && fmt[i].is_ascii_digit() {
                    p = p * 10 + fmt[i].to_digit(10).unwrap() as usize;
                    i += 1;
                }
                precision = Some(p);
            }
        }

        // Length modifiers: h, hh, l, ll, w, I64, I32, z.
        let mut long64 = false;
        let mut wide = false;
        let mut short = false;
        let starts = |at: usize, s: &str| s.chars().enumerate().all(|(k, ch)| fmt.get(at + k) == Some(&ch));
        loop {
            if i >= fmt.len() {
                break;
            }
            match fmt[i] {
                'h' => short = true,
                'l' => {
                    if i + 1 < fmt.len() && fmt[i + 1] == 'l' {
                        long64 = true;
                        i += 1;
                    } else {
                        wide = true;
                    }
                }
                'w' => wide = true,
                'z' => {}
                'I' if starts(i, "I64") => {
                    long64 = true;
                    i += 2;
                }
                'I' if starts(i, "I32") => i += 2,
                _ => break,
            }
            i += 1;
        }
        if i >= fmt.len() {
            break;
        }
        let conv = fmt[i];
        i += 1;
        // Whether a %c / %s argument is wide.
        let wide_arg = |upper: bool| {
            if short {
                false
            } else if wide {
                true
            } else {
                upper ^ wide_format
            }
        };

        let body = match conv {
            'd' | 'i' => {
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
            'u' | 'x' | 'X' | 'o' => {
                let raw = next();
                let v = if long64 {
                    raw
                } else if short {
                    raw as u16 as u64
                } else {
                    raw as u32 as u64
                };
                let mut s = match conv {
                    'u' => v.to_string(),
                    'x' => format!("{v:x}"),
                    'X' => format!("{v:X}"),
                    _ => format!("{v:o}"),
                };
                if let Some(p) = precision {
                    s = format!("{s:0>p$}");
                }
                let prefix = match (alt && v != 0, conv) {
                    (true, 'x') => "0x",
                    (true, 'X') => "0X",
                    (true, 'o') => "0",
                    _ => "",
                };
                pad_number(prefix, &s, width, left, zero && precision.is_none())
            }
            'p' => pad(&format!("{:08X}", next() as u32), width, left),
            'c' | 'C' => {
                let v = next() as u32;
                let ch =
                    if wide_arg(conv == 'C') { char::from_u32(v & 0xFFFF).unwrap_or('?') } else { (v as u8) as char };
                pad(&ch.to_string(), width, left)
            }
            's' | 'S' => {
                let ptr = next() as u32;
                let mut s = if ptr == 0 {
                    "(null)".to_string()
                } else if wide_arg(conv == 'S') {
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
            'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                let v = f64::from_bits(next());
                let p = precision.unwrap_or(6);
                let s = match conv {
                    _ if !v.is_finite() => {
                        let s = if v.is_nan() { "nan" } else { "inf" };
                        if conv.is_ascii_uppercase() { s.to_uppercase() } else { s.to_string() }
                    }
                    'e' | 'E' => format_exp(v.abs(), p, conv == 'E', alt),
                    'g' | 'G' => format_general(v.abs(), p, conv == 'G', alt),
                    _ => {
                        let s = format!("{:.p$}", v.abs());
                        if alt && p == 0 { s + "." } else { s }
                    }
                };
                let sign = if v.is_sign_negative() && !v.is_nan() {
                    "-"
                } else if plus {
                    "+"
                } else if space {
                    " "
                } else {
                    ""
                };
                pad_number(sign, &s, width, left, zero && v.is_finite())
            }
            'n' => {
                next();
                String::new()
            }
            other => {
                // Unknown conversion: echo it so the log still shows the format.
                format!("%{other}")
            }
        };
        out.push_str(&body);
    }
    out
}

/// `%e` of a non-negative finite value: mantissa with `p` decimals, then an
/// exponent of at least two digits (`1.500000e+01`).
fn format_exp(v: f64, p: usize, upper: bool, alt: bool) -> String {
    let s = format!("{v:.p$e}");
    let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let point = if alt && p == 0 { "." } else { "" };
    let e = if upper { 'E' } else { 'e' };
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mantissa}{point}{e}{sign}{:02}", exp.unsigned_abs())
}

/// `%g` of a non-negative finite value: `p` significant digits, in `%e` form
/// when the exponent is below -4 or not below the precision, else `%f`;
/// trailing zeros (and a bare point) removed unless `#`.
fn format_general(v: f64, p: usize, upper: bool, alt: bool) -> String {
    let p = p.max(1);
    // The exponent after rounding to p significant digits.
    let rounded = format!("{v:.prec$e}", prec = p - 1);
    let exp: i32 = rounded.split_once('e').and_then(|(_, e)| e.parse().ok()).unwrap_or(0);
    let mut s = if exp < -4 || exp >= p as i32 {
        format_exp(v, p - 1, upper, alt)
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        let s = format!("{v:.decimals$}");
        if alt && decimals == 0 { s + "." } else { s }
    };
    if !alt {
        let (number, exponent) = match s.find(['e', 'E']) {
            Some(at) => (s[..at].to_string(), s[at..].to_string()),
            None => (s.clone(), String::new()),
        };
        let number =
            if number.contains('.') { number.trim_end_matches('0').trim_end_matches('.').to_string() } else { number };
        s = number + &exponent;
    }
    s
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
    let len = prefix.len() + digits.chars().count();
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
