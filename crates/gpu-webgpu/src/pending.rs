use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;

type Slot<T> = Rc<RefCell<Option<T>>>;

/// The result of an asynchronous webgpu.h operation, filled in by its callback.
///
/// webgpu.h guarantees each callback runs exactly once (with a cancellation status if the
/// instance goes away first), which is what frees the callback's reference to the slot.
pub struct Pending<T> {
    slot: Slot<T>,
    /// Produces the value once it exists (the Metal backend's background work).
    #[cfg(metal_backend)]
    poll: Option<Box<dyn Fn() -> Option<T>>>,
}

impl<T> Pending<T> {
    /// A pending value plus the `userdata1` pointer to hand to the callback, which must
    /// eventually pass it to [`complete`] with the same `T`.
    #[cfg_attr(metal_backend, allow(dead_code))]
    pub(crate) fn new() -> (Self, *mut c_void) {
        let slot: Slot<T> = Rc::new(RefCell::new(None));
        let userdata = Box::into_raw(Box::new(slot.clone())).cast::<c_void>();
        (Self::with_slot(slot), userdata)
    }

    fn with_slot(slot: Slot<T>) -> Self {
        Self {
            slot,
            #[cfg(metal_backend)]
            poll: None,
        }
    }

    pub(crate) fn ready(value: T) -> Self {
        Self::with_slot(Rc::new(RefCell::new(Some(value))))
    }

    pub fn is_ready(&self) -> bool {
        self.poll_now();
        self.slot.borrow().is_some()
    }

    /// Takes the value once the callback has run.
    pub fn try_take(&self) -> Option<T> {
        self.poll_now();
        self.slot.borrow_mut().take()
    }

    fn poll_now(&self) {
        #[cfg(metal_backend)]
        if let Some(poll) = &self.poll
            && self.slot.borrow().is_none()
            && let Some(v) = poll()
        {
            *self.slot.borrow_mut() = Some(v);
        }
    }
}

impl<T> std::fmt::Debug for Pending<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").field("ready", &self.is_ready()).finish()
    }
}

/// The completing side of a [`Pending`] made by [`Pending::pair`] (the Metal backend resolves
/// its operations from Rust rather than through C callbacks).
#[cfg(metal_backend)]
pub(crate) struct Completer<T> {
    slot: Slot<T>,
}

#[cfg(metal_backend)]
impl<T> Completer<T> {
    pub(crate) fn complete(self, value: T) {
        *self.slot.borrow_mut() = Some(value);
    }
}

#[cfg(metal_backend)]
impl<T> Pending<T> {
    pub(crate) fn pair() -> (Self, Completer<T>) {
        let slot: Slot<T> = Rc::new(RefCell::new(None));
        (Self::with_slot(slot.clone()), Completer { slot })
    }

    /// A value `poll` produces once it exists (asked whenever the value is).
    pub(crate) fn polled(poll: impl Fn() -> Option<T> + 'static) -> Self {
        Self { slot: Rc::new(RefCell::new(None)), poll: Some(Box::new(poll)) }
    }
}

/// Stores a callback's result. `userdata` must come from [`Pending::new`] for the same `T` and
/// is consumed.
#[cfg_attr(metal_backend, allow(dead_code))]
pub(crate) unsafe fn complete<T>(userdata: *mut c_void, value: T) {
    let slot = unsafe { Box::from_raw(userdata.cast::<Slot<T>>()) };
    *slot.borrow_mut() = Some(value);
}
