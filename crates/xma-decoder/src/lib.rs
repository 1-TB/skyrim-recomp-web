// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Licensing: wmapro.rs, tables.rs, vlc.rs, mdct.rs and bits.rs are ports of
// FFmpeg code (LGPL-2.1-or-later; see their headers). context.rs is ported
// from Xenia and keeps its BSD 3-Clause notice. The crate as a whole is LGPL-2.1-or-later.

//! Xbox 360 XMA audio decoding.
//!
//! * [`FrameDecoder`] decodes single XMA frames (WMA Pro bitstream, 512
//!   samples per channel) to float PCM. It is a port of FFmpeg's wmaprodec.c
//!   and is licensed LGPL-2.1-or-later, like the whole crate.
//! * [`Context`] is the hardware context processor: it walks a guest
//!   XMA_CONTEXT_DATA's input packet buffers, decodes frames and fills the
//!   output ring with big-endian 16-bit PCM. It is a port of Xenia's
//!   XmaContextNew (BSD-licensed).

mod bits;
pub mod context;
mod mdct;
mod tables;
mod vlc;
mod wmapro;

pub use context::{Context, FrameDecoding, XmaMemory, convert_to_be_s16, field, float_to_s16, merge_decoder_fields};
pub use wmapro::{DecodeError, FrameDecoder, FrameInfo, MAX_CHANNELS, MAX_FRAME_BITS, SAMPLES_PER_FRAME};
