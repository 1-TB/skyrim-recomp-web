//! Minimal stderr logger used by the host binaries: elapsed time, thread name,
//! level, and `guest:` for text the title printed itself (DbgPrint).

use std::io::Write;
use std::time::Instant;

use log::{Level, LevelFilter, Log, Metadata, Record};

struct Logger {
    start: Instant,
    level: LevelFilter,
    /// (target prefix, level), longest prefix first.
    overrides: Vec<(String, LevelFilter)>,
}

impl Logger {
    fn level_for(&self, target: &str) -> LevelFilter {
        self.overrides.iter().find(|(p, _)| target.starts_with(p.as_str())).map_or(self.level, |&(_, l)| l)
    }
}

impl Log for Logger {
    fn enabled(&self, m: &Metadata) -> bool {
        m.level() <= self.level_for(m.target())
    }

    fn log(&self, r: &Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let t = self.start.elapsed();
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("host");
        let level = match r.level() {
            Level::Error => "ERROR",
            Level::Warn => "WARN ",
            Level::Info => "INFO ",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        };
        let prefix = if r.target() == "guest" { "guest: " } else { "" };
        let line = format!("[{:>9.3} {name:<11} {level}] {prefix}{}\n", t.as_secs_f64(), r.args());
        let _ = std::io::stderr().lock().write_all(line.as_bytes());
    }

    fn flush(&self) {
        let _ = std::io::stderr().flush();
    }
}

/// Installs the logger. `spec` is a default level (error, warn, info, debug,
/// trace) optionally followed by per-target levels, e.g.
/// `info,kernel_hle::gpu=trace`.
pub fn init(spec: &str) -> Result<(), String> {
    let (level, overrides) = parse_spec(spec)?;
    let max = overrides.iter().map(|&(_, l)| l).fold(level, Ord::max);
    let logger = Box::leak(Box::new(Logger { start: Instant::now(), level, overrides }));
    log::set_logger(logger).map_err(|e| e.to_string())?;
    log::set_max_level(max);
    Ok(())
}

fn parse_spec(spec: &str) -> Result<(LevelFilter, Vec<(String, LevelFilter)>), String> {
    let parse_level = |s: &str| s.parse::<LevelFilter>().map_err(|_| format!("unknown log level {s:?}"));
    let mut level = LevelFilter::Info;
    let mut overrides = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match item.split_once('=') {
            Some((target, l)) => overrides.push((target.to_string(), parse_level(l)?)),
            None => level = parse_level(item)?,
        }
    }
    overrides.sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));
    Ok((level, overrides))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_with_target_overrides() {
        let (level, overrides) = parse_spec("warn,kernel_hle=info,kernel_hle::gpu=trace").unwrap();
        assert_eq!(level, LevelFilter::Warn);
        let logger = Logger { start: Instant::now(), level, overrides };
        assert_eq!(logger.level_for("kernel_hle::gpu"), LevelFilter::Trace);
        assert_eq!(logger.level_for("kernel_hle::dispatch"), LevelFilter::Info);
        assert_eq!(logger.level_for("vfs"), LevelFilter::Warn);
        assert!(parse_spec("loud").is_err());
    }
}
