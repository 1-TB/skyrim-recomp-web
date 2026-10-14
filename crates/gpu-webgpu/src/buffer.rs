use std::cell::Cell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;

use crate::handle::{label, string, wbool};
use crate::pending::{Pending, complete};
use crate::{BufferUsage, Device, Error, MapMode, Result, ffi};

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferDescriptor<'a> {
    pub label: &'a str,
    pub size: u64,
    pub usage: BufferUsage,
    pub mapped_at_creation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Map {
    Unmapped,
    Pending,
    /// The whole mapped range was fetched once: emdawnwebgpu hands out a fresh copy per
    /// `GetMappedRange` call and the browser rejects overlapping ranges, so it is cached here.
    Mapped {
        write: bool,
        start: u64,
        end: u64,
        data: *mut u8,
    },
}

/// Map state is tracked here rather than queried: wgpu-native lacks `wgpuBufferGetMapState` and
/// aborts (instead of returning null) on mapped-range calls outside a mapping.
struct State {
    size: u64,
    usage: BufferUsage,
    map: Cell<Map>,
    /// Bumped per map request so a stale callback (after an unmap) cannot clobber a newer one.
    generation: Cell<u64>,
}

pub struct Buffer {
    raw: ffi::WGPUBuffer,
    state: Rc<State>,
    _thread: PhantomData<*const ()>,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        unsafe { ffi::wgpuBufferAddRef(self.raw) };
        Self { raw: self.raw, state: self.state.clone(), _thread: PhantomData }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { ffi::wgpuBufferRelease(self.raw) };
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({:p}, {} bytes)", self.raw, self.state.size)
    }
}

impl Device {
    pub fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Buffer {
        crate::counters::add(&crate::counters::count::buffers, 1);
        let raw_desc = ffi::WGPUBufferDescriptor {
            label: label(desc.label),
            usage: desc.usage,
            size: desc.size,
            mappedAtCreation: wbool(desc.mapped_at_creation),
            ..Default::default()
        };
        // Validation failures yield an error object (reported through the error callbacks),
        // never null - except that a buffer mapped at creation that can't be allocated is null
        // in browsers (JS throws a RangeError). Then an unmapped buffer of the same size is made
        // instead: also out of memory, it comes back as an error object, reported like any other.
        let mut raw = unsafe { ffi::wgpuDeviceCreateBuffer(self.raw(), &raw_desc) };
        let mut mapped = desc.mapped_at_creation;
        if raw.is_null() && mapped {
            mapped = false;
            let unmapped = ffi::WGPUBufferDescriptor { mappedAtCreation: wbool(false), ..raw_desc };
            raw = unsafe { ffi::wgpuDeviceCreateBuffer(self.raw(), &unmapped) };
        }
        assert!(!raw.is_null(), "wgpuDeviceCreateBuffer returned null");
        crate::memory::track(raw as usize, "buffer", desc.label, desc.size);
        let state = Rc::new(State {
            size: desc.size,
            usage: desc.usage,
            map: Cell::new(Map::Unmapped),
            generation: Cell::new(0),
        });
        let buffer = Buffer { raw, state, _thread: PhantomData };
        if mapped {
            let data = unsafe { ffi::wgpuBufferGetMappedRange(raw, 0, desc.size as usize) };
            if !data.is_null() {
                buffer.state.map.set(Map::Mapped { write: true, start: 0, end: desc.size, data: data.cast() });
            }
        }
        buffer
    }

    /// A buffer initialised with `data` (zero-padded to a multiple of 4 bytes).
    pub fn create_buffer_init(&self, label: &str, usage: BufferUsage, data: &[u8]) -> Buffer {
        let size = (data.len() as u64).div_ceil(4).max(1) * 4;
        let buffer = self.create_buffer(&BufferDescriptor { label, size, usage, mapped_at_creation: true });
        let mut padded = data.to_vec();
        padded.resize(size as usize, 0);
        // Only fails when creation itself failed (an error object), which was already reported.
        let _ = buffer.write_mapped(0, &padded);
        buffer.unmap();
        buffer
    }
}

impl Buffer {
    unsafe fn adopt(raw: ffi::WGPUBuffer, state: Rc<State>) -> Self {
        Self { raw, state, _thread: PhantomData }
    }

    pub fn raw(&self) -> ffi::WGPUBuffer {
        self.raw
    }

    pub fn size(&self) -> u64 {
        self.state.size
    }

    pub fn usage(&self) -> BufferUsage {
        self.state.usage
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.state.map.get(), Map::Mapped { .. })
    }

    /// Maps `offset..offset + size` (`size: None` = to the end; offset a multiple of 8, size of 4).
    /// Resolves once the GPU is done with the buffer; then use [`Buffer::read_mapped`] /
    /// [`Buffer::write_mapped`] and finally [`Buffer::unmap`].
    pub fn map_async(&self, mode: MapMode, offset: u64, size: Option<u64>) -> Pending<Result<()>> {
        struct Request {
            buffer: ffi::WGPUBuffer,
            state: Rc<State>,
            generation: u64,
            write: bool,
            start: u64,
            end: u64,
            userdata: *mut c_void,
        }
        unsafe extern "C" fn done(
            status: ffi::WGPUMapAsyncStatus,
            message: ffi::WGPUStringView,
            userdata: *mut c_void,
            _: *mut c_void,
        ) {
            let req = unsafe { Box::from_raw(userdata.cast::<Request>()) };
            let _keep = unsafe { Buffer::adopt(req.buffer, req.state.clone()) };
            let current = req.state.generation.get() == req.generation;
            let result = if status == ffi::WGPUMapAsyncStatus::Success && current {
                let len = (req.end - req.start) as usize;
                let data = unsafe {
                    if req.write {
                        ffi::wgpuBufferGetMappedRange(req.buffer, req.start as usize, len)
                    } else {
                        ffi::wgpuBufferGetConstMappedRange(req.buffer, req.start as usize, len).cast_mut()
                    }
                };
                if data.is_null() && len > 0 {
                    req.state.map.set(Map::Unmapped);
                    Err(Error::Map("mapped range unavailable".into()))
                } else {
                    let (write, start, end) = (req.write, req.start, req.end);
                    req.state.map.set(Map::Mapped { write, start, end, data: data.cast() });
                    Ok(())
                }
            } else {
                if current {
                    req.state.map.set(Map::Unmapped);
                }
                Err(Error::Map(format!("status {}: {}", status.0, string(message))))
            };
            unsafe { complete(req.userdata, result) };
        }

        let write = mode.contains(MapMode::Write);
        let size = size.unwrap_or_else(|| self.state.size.saturating_sub(offset));
        let needed = if write { BufferUsage::MapWrite } else { BufferUsage::MapRead };
        let problem = if self.state.map.get() != Map::Unmapped {
            Some("buffer is already mapped or being mapped".to_string())
        } else if mode != MapMode::Read && mode != MapMode::Write {
            Some(format!("map mode must be Read or Write, got {:?}", mode))
        } else if !self.state.usage.contains(needed) {
            Some(format!("buffer usage {:?} lacks {:?}", self.state.usage, needed))
        } else if !offset.is_multiple_of(8)
            || !size.is_multiple_of(4)
            || offset.checked_add(size).is_none_or(|e| e > self.state.size)
        {
            Some(format!("range {offset}+{size} is misaligned or exceeds {} bytes", self.state.size))
        } else {
            None
        };
        if let Some(problem) = problem {
            return Pending::ready(Err(Error::Map(problem)));
        }
        let generation = self.state.generation.get() + 1;
        // The request holds a reference so the callback can fetch the range even if every
        // `Buffer` handle is dropped meanwhile.
        unsafe { ffi::wgpuBufferAddRef(self.raw) };
        self.state.generation.set(generation);
        self.state.map.set(Map::Pending);
        let (pending, userdata) = Pending::new();
        let req = Box::new(Request {
            buffer: self.raw,
            state: self.state.clone(),
            generation,
            write,
            start: offset,
            end: offset + size,
            userdata,
        });
        let info = ffi::WGPUBufferMapCallbackInfo {
            mode: ffi::WGPUCallbackMode::AllowProcessEvents,
            callback: Some(done),
            userdata1: Box::into_raw(req).cast(),
            ..Default::default()
        };
        unsafe { ffi::wgpuBufferMapAsync(self.raw, mode, offset as usize, size as usize, info) };
        pending
    }

    fn mapped_ptr(&self, offset: u64, len: usize, need_write: bool) -> Result<*mut u8> {
        match self.state.map.get() {
            Map::Mapped { write, start, end, data }
                if (write || !need_write) && offset >= start && offset + len as u64 <= end =>
            {
                Ok(unsafe { data.add((offset - start) as usize) })
            }
            Map::Mapped { write: false, .. } if need_write => Err(Error::Map("buffer is mapped for reading".into())),
            Map::Mapped { start, end, .. } => {
                Err(Error::Map(format!("range {offset}+{len} is outside the mapped range {start}..{end}")))
            }
            _ => Err(Error::Map("buffer is not mapped".into())),
        }
    }

    /// Copies out of the mapped range (any alignment).
    pub fn read_mapped(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let src = self.mapped_ptr(offset, dst.len(), false)?;
        unsafe { ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    /// Copies into a range mapped for writing (or mapped at creation). On the web the mapped
    /// memory starts zeroed, not with the buffer's contents.
    pub fn write_mapped(&self, offset: u64, src: &[u8]) -> Result<()> {
        let dst = self.mapped_ptr(offset, src.len(), true)?;
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        Ok(())
    }

    /// Unmaps (cancelling a pending map, whose callback then reports an abort).
    pub fn unmap(&self) {
        self.state.generation.set(self.state.generation.get() + 1);
        self.state.map.set(Map::Unmapped);
        unsafe { ffi::wgpuBufferUnmap(self.raw) };
    }

    pub fn destroy(&self) {
        self.state.generation.set(self.state.generation.get() + 1);
        self.state.map.set(Map::Unmapped);
        crate::memory::untrack(self.raw as usize);
        unsafe { ffi::wgpuBufferDestroy(self.raw) };
    }
}
