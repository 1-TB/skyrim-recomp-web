//! GPU time per pass label. The Metal backend has no timestamp queries yet, so [`enable`]
//! reports them unavailable (as for a device without `FeatureName::TimestampQuery`) and nothing
//! is ever collected.

use crate::Device;

#[derive(Clone, Copy, Debug, Default)]
pub struct LabelTime {
    /// Sum of the passes' GPU time.
    pub nanos: u64,
    pub passes: u64,
}

/// `false`: pass timing needs timestamp queries.
pub fn enable(_device: &Device) -> bool {
    false
}

pub fn enabled() -> bool {
    false
}

/// Nothing to collect without timestamp queries.
pub fn collect() {}

/// Per-label totals and untimed passes since the last call (always empty here).
pub fn take() -> (Vec<(String, LabelTime)>, u64) {
    (Vec::new(), 0)
}
