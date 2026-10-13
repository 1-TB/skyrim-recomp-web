//! Guest path parsing.
//!
//! Guest paths look like `game:\Data\Skyrim.esm`, `\??\d:\foo`, or
//! `\Device\Cdrom0\default.xex`. Separators may be `\` or `/`. Comparison is
//! ASCII case-insensitive, as on the console.

/// Splits a path into non-empty components with `.` removed and `..` applied.
/// `..` never climbs above the first component (the device root).
pub fn normalize_components(path: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    for part in path.split(['\\', '/']) {
        match part {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            p => out.push(p),
        }
    }
    out
}

/// Removes the NT object-manager prefixes `\??\` and `\DosDevices\`.
pub fn strip_object_prefix(path: &str) -> &str {
    for prefix in ["\\??\\", "/??/", "\\DosDevices\\"] {
        if path.len() >= prefix.len() && path[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return &path[prefix.len()..];
        }
    }
    path
}

/// Splits `name:` off the front of a path (`game:\x` -> (`game:`, `\x`)).
pub fn split_drive(path: &str) -> Option<(&str, &str)> {
    let colon = path.find(':')?;
    let drive = &path[..colon];
    if drive.is_empty() || drive.contains(['\\', '/']) {
        return None;
    }
    Some((&path[..=colon], &path[colon + 1..]))
}

/// `*` / `?` wildcard match, ASCII case-insensitive. Also accepts the NT DOS
/// wildcards `<` (as `*`), `>` (as `?`) and `"` (as `.`).
pub fn matches_pattern(name: &str, pattern: &str) -> bool {
    fn norm(c: u8) -> u8 {
        c.to_ascii_lowercase()
    }
    let n = name.as_bytes();
    let p: Vec<u8> = pattern
        .bytes()
        .map(|c| match c {
            b'<' => b'*',
            b'>' => b'?',
            b'"' => b'.',
            c => c,
        })
        .collect();
    if p == b"*" || p == b"*.*" {
        return true;
    }
    // Iterative glob with single-star backtracking.
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while i < n.len() {
        if j < p.len() && (p[j] == b'?' || norm(p[j]) == norm(n[i])) {
            i += 1;
            j += 1;
        } else if j < p.len() && p[j] == b'*' {
            star = Some(j);
            mark = i;
            j += 1;
        } else if let Some(s) = star {
            j = s + 1;
            mark += 1;
            i = mark;
        } else {
            return false;
        }
    }
    while j < p.len() && p[j] == b'*' {
        j += 1;
    }
    j == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components() {
        assert_eq!(normalize_components("\\a\\.\\b\\..\\c//d"), ["a", "c", "d"]);
        assert_eq!(normalize_components("..\\..\\x"), ["x"]);
    }

    #[test]
    fn drives() {
        assert_eq!(split_drive("game:\\x"), Some(("game:", "\\x")));
        assert_eq!(split_drive("\\Device\\Cdrom0"), None);
        assert_eq!(strip_object_prefix("\\??\\D:\\a"), "D:\\a");
    }

    #[test]
    fn wildcards() {
        assert!(matches_pattern("Skyrim.esm", "*.ESM"));
        assert!(matches_pattern("abc", "a?c"));
        assert!(!matches_pattern("abc", "a?d"));
        assert!(matches_pattern("anything", "*"));
        assert!(matches_pattern("a.b.c", "*.c"));
        assert!(!matches_pattern("a.b.c", "*.b"));
    }
}
