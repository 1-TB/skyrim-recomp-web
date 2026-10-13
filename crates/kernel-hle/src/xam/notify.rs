//! XNotify listeners: waitable queues of system notifications.
//! Behavior follows Xenia's xam_notify.cc / xnotifylistener.cc.
#![allow(non_snake_case)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::Kernel;
use crate::dispatch::{Call, Export, ModuleId};
use crate::objects::{KObject, ObjectType, Payload, SyncState};
use crate::status;

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x28A, XamNotifyCreateListener),
    export!(X, 0x28B, XNotifyGetNext),
    export!(X, 0x28C, XNotifyPositionUI),
    export!(X, 0x28D, XNotifyDelayUI),
    export!(X, 0x28E, XNotifyBroadcast),
];

/// Notification ids (area in bits 25-30, version in bits 16-24).
pub mod id {
    pub const SYSTEM_UI: u32 = 0x0000_0009;
    pub const SYSTEM_SIGNIN_CHANGED: u32 = 0x0000_000A;
    pub const SYSTEM_INPUT_DEVICES_CHANGED: u32 = 0x0000_0012;
    pub const LIVE_CONNECTION_CHANGED: u32 = 0x0200_0001;
    pub const LIVE_LINK_STATE_CHANGED: u32 = 0x0200_0003;
}

/// Listener area masks (XNOTIFY_SYSTEM, XNOTIFY_LIVE).
const AREA_SYSTEM: u64 = 1;
const AREA_LIVE: u64 = 2;
/// X_ONLINE_S_LOGON_DISCONNECTED.
const LOGON_DISCONNECTED: u32 = 0x0015_10F1;

pub struct Listener {
    mask: u64,
    max_version: u32,
    queue: Mutex<VecDeque<(u32, u32)>>,
}

impl Listener {
    fn accepts(&self, notification: u32) -> bool {
        let area = (notification >> 25) & 0x3F;
        let version = (notification >> 16) & 0x1FF;
        self.mask & (1u64 << area) != 0 && version <= self.max_version
    }
}

#[derive(Default)]
pub struct Notifications {
    listeners: Mutex<Vec<Weak<KObject>>>,
    startup: Mutex<(bool, bool)>,
    ui_position: Mutex<u32>,
    /// System dialogs currently open (XamIsUIActive).
    ui_open: AtomicU32,
}

fn listener_of(obj: &KObject) -> Option<&Arc<Listener>> {
    match &obj.payload {
        Payload::NotifyListener(l) => Some(l),
        _ => None,
    }
}

impl Kernel {
    /// Queues `notification` on one listener if its mask and version accept it.
    fn enqueue_notification(&self, obj: &KObject, notification: u32, data: u32) {
        let Some(l) = listener_of(obj) else { return };
        if !l.accepts(notification) {
            return;
        }
        l.queue.lock().unwrap().push_back((notification, data));
        self.objects.set_event(self.mem(), obj);
    }

    /// A system dialog opens: counted for XamIsUIActive and announced with
    /// XN_SYS_UI(TRUE).
    pub(crate) fn system_ui_shown(&self) {
        self.notifications.ui_open.fetch_add(1, Ordering::AcqRel);
        self.broadcast_notification(id::SYSTEM_UI, 1);
    }

    /// The dialog closes: XN_SYS_UI(FALSE) once the last one has.
    pub(crate) fn system_ui_hidden(&self) {
        let prev = self.notifications.ui_open.try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1));
        if prev == Ok(1) {
            self.broadcast_notification(id::SYSTEM_UI, 0);
        }
    }

    pub(crate) fn system_ui_active(&self) -> bool {
        self.notifications.ui_open.load(Ordering::Acquire) != 0
    }

    /// XNotifyBroadcast: every live listener whose mask accepts it.
    pub fn broadcast_notification(&self, notification: u32, data: u32) {
        let listeners: Vec<Arc<KObject>> = {
            let mut list = self.notifications.listeners.lock().unwrap();
            list.retain(|w| w.strong_count() > 0);
            list.iter().filter_map(Weak::upgrade).collect()
        };
        for obj in listeners {
            self.enqueue_notification(&obj, notification, data);
        }
    }

    /// Creates a listener and returns its handle.
    pub fn create_notify_listener(&self, mask: u64, max_version: u32) -> Result<u32, u32> {
        let listener =
            Arc::new(Listener { mask, max_version: max_version.min(10), queue: Mutex::new(VecDeque::new()) });
        let obj = self.objects.create(
            &self.rt.space,
            ObjectType::Event,
            Some(SyncState::Event { manual_reset: true, signaled: false }),
            Payload::NotifyListener(listener),
            None,
        )?;
        let handle = self.objects.insert_handle(&obj);
        self.notifications.listeners.lock().unwrap().push(Arc::downgrade(&obj));
        // Titles expect a few notifications at startup, on the first listener
        // of each area only (Xenia KernelState::RegisterNotifyListener).
        let mut startup = self.notifications.startup.lock().unwrap();
        if !startup.0 && mask & AREA_SYSTEM != 0 {
            startup.0 = true;
            self.enqueue_notification(&obj, id::SYSTEM_UI, 0);
            self.enqueue_notification(&obj, id::SYSTEM_SIGNIN_CHANGED, 1);
        }
        if !startup.1 && mask & AREA_LIVE != 0 {
            startup.1 = true;
            self.enqueue_notification(&obj, id::LIVE_CONNECTION_CHANGED, LOGON_DISCONNECTED);
            self.enqueue_notification(&obj, id::LIVE_LINK_STATE_CHANGED, 0);
        }
        Ok(handle)
    }

    /// Takes the oldest notification (or the oldest with id `only`).
    pub fn next_notification(&self, obj: &KObject, only: Option<u32>) -> Option<(u32, u32)> {
        let l = listener_of(obj)?;
        let mut q = l.queue.lock().unwrap();
        let pos = match only {
            Some(want) => q.iter().position(|&(n, _)| n == want)?,
            None => 0,
        };
        let item = q.remove(pos)?;
        if q.is_empty() {
            self.objects.reset_event(self.mem(), obj);
        }
        Some(item)
    }
}

/// (u64 AreaMask, u32 MaxVersion) -> handle (0 on failure)
fn XamNotifyCreateListener(c: &mut Call) -> u64 {
    let (mask, max_version) = (c.arg64(0), c.arg(1));
    match c.kernel.create_notify_listener(mask, max_version) {
        Ok(h) => h as u64,
        Err(s) => {
            log::error!("XamNotifyCreateListener: {s:#010X}");
            0
        }
    }
}

/// (HANDLE Listener, u32 MatchId, u32* Id, u32* Param) -> BOOL
fn XNotifyGetNext(c: &mut Call) -> u64 {
    let (handle, match_id, id_ptr, param_ptr) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3));
    let mem = c.mem();
    if param_ptr != 0 {
        mem.write_u32(param_ptr, 0);
    }
    if id_ptr == 0 {
        return 0;
    }
    mem.write_u32(id_ptr, 0);
    let Some(obj) = c.kernel.objects.lookup(handle, None) else { return 0 };
    match c.kernel.next_notification(&obj, (match_id != 0).then_some(match_id)) {
        Some((n, data)) => {
            mem.write_u32(id_ptr, n);
            if param_ptr != 0 {
                mem.write_u32(param_ptr, data);
            }
            1
        }
        None => 0,
    }
}

/// Where system notification toasts appear; there is no system UI here.
fn XNotifyPositionUI(c: &mut Call) -> u64 {
    *c.kernel.notifications.ui_position.lock().unwrap() = c.arg(0);
    0
}

fn XNotifyDelayUI(_c: &mut Call) -> u64 {
    0
}

/// (u32 Notification, u32 Data)
fn XNotifyBroadcast(c: &mut Call) -> u64 {
    c.kernel.broadcast_notification(c.arg(0), c.arg(1));
    status::SUCCESS as u64
}
