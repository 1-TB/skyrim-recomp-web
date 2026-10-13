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
}

impl<T> Pending<T> {
    /// A pending value plus the `userdata1` pointer to hand to the callback, which must
    /// eventually pass it to [`complete`] with the same `T`.
    pub(crate) fn new() -> (Self, *mut c_void) {
        let slot: Slot<T> = Rc::new(RefCell::new(None));
        let userdata = Box::into_raw(Box::new(slot.clone())).cast::<c_void>();
        (Self { slot }, userdata)
    }

    pub(crate) fn ready(value: T) -> Self {
        Self { slot: Rc::new(RefCell::new(Some(value))) }
    }

    pub fn is_ready(&self) -> bool {
        self.slot.borrow().is_some()
    }

    /// Takes the value once the callback has run.
    pub fn try_take(&self) -> Option<T> {
        self.slot.borrow_mut().take()
    }
}

impl<T> std::fmt::Debug for Pending<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").field("ready", &self.is_ready()).finish()
    }
}

/// Stores a callback's result. `userdata` must come from [`Pending::new`] for the same `T` and
/// is consumed.
pub(crate) unsafe fn complete<T>(userdata: *mut c_void, value: T) {
    let slot = unsafe { Box::from_raw(userdata.cast::<Slot<T>>()) };
    *slot.borrow_mut() = Some(value);
}
