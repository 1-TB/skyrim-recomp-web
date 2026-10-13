mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{RECORDER, game_dir, guest, kernel, scratch};
use kernel_hle::input::{Gamepad, buttons};
use kernel_hle::status::{self, hresult, win32};

fn cstr(k: &kernel_hle::Kernel, at: u32, s: &str) -> u32 {
    k.mem().write_bytes(at, s.as_bytes());
    k.mem().write_u8(at + s.len() as u32, 0);
    at
}

#[test]
fn signed_in_profile_identity() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    assert_eq!(k.call_export("XamUserGetSigninState", &mut ctx, &[0]), 1);
    assert_eq!(k.call_export("XamUserGetSigninState", &mut ctx, &[1]), 0);
    assert_eq!(k.call_export("XamUserGetXUID", &mut ctx, &[0, 1, io]), 0);
    let xuid = mem.read_u64(io);
    assert_eq!(xuid >> 52, 0xE03);
    assert_eq!(k.call_export("XamUserGetName", &mut ctx, &[0, io + 0x10, 16]), 0);
    assert_eq!(mem.read_cstr(io + 0x10, 16), b"Player");
    assert_eq!(k.call_export("XamUserGetSigninInfo", &mut ctx, &[0, 0, io + 0x40]), 0);
    assert_eq!(mem.read_u64(io + 0x40), xuid);
    assert_eq!(mem.read_u32(io + 0x4C), 1, "signed in locally");
    assert_eq!(mem.read_cstr(io + 0x58, 16), b"Player");
    assert_eq!(k.call_export("XamUserGetXUID", &mut ctx, &[2, 1, io]) as u32, hresult::from_win32(win32::NO_SUCH_USER));
}

#[test]
fn controller_state_and_capabilities() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    assert_eq!(k.call_export("XamInputGetState", &mut ctx, &[0, 0, io]) as u32, win32::DEVICE_NOT_CONNECTED);
    k.input.update(
        0,
        Gamepad { buttons: buttons::A | buttons::START, right_trigger: 200, thumb_ly: -32768, ..Default::default() },
    );
    assert_eq!(k.call_export("XamInputGetState", &mut ctx, &[0, 1, io]), 0);
    let packet = mem.read_u32(io);
    assert_eq!(mem.read_u16(io + 4), buttons::A | buttons::START);
    assert_eq!(mem.read_u8(io + 7), 200);
    assert_eq!(mem.read_u16(io + 10) as i16, -32768);
    // XUSER_INDEX_ANY reads port 0.
    assert_eq!(k.call_export("XamInputGetState", &mut ctx, &[0xFF, 1, io]), 0);
    assert_eq!(mem.read_u32(io), packet, "unchanged state keeps the packet number");

    assert_eq!(k.call_export("XamInputGetCapabilities", &mut ctx, &[0, 1, io + 0x20]), 0);
    assert_eq!((mem.read_u8(io + 0x20), mem.read_u8(io + 0x21)), (1, 1));
    assert_eq!(mem.read_u16(io + 0x24), 0xF3FF);
    mem.write_u16(io + 0x40, 0x1234);
    mem.write_u16(io + 0x42, 0x5678);
    assert_eq!(k.call_export("XamInputSetState", &mut ctx, &[0, 0, io + 0x40]), 0);
    assert_eq!(k.input.vibration(0), Some((0x1234, 0x5678)));
    assert_eq!(k.call_export("XamInputSetState", &mut ctx, &[2, 0, io + 0x40]) as u32, win32::DEVICE_NOT_CONNECTED);
}

#[test]
fn notification_listener_queue() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    // XNOTIFY_SYSTEM | XNOTIFY_LIVE, version 10.
    let h = k.call_export("XamNotifyCreateListener", &mut ctx, &[3, 10]) as u32;
    assert_ne!(h, 0);
    // Startup notifications arrive in order; the listener is signaled until drained.
    let next = |ctx: &mut runtime_core::PPCContext, matching: u32| {
        let ok = k.call_export("XNotifyGetNext", ctx, &[h, matching, io, io + 4]);
        (ok, mem.read_u32(io), mem.read_u32(io + 4))
    };
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, 0]) as u32, status::SUCCESS);
    // Ask for the Live link state first, out of order.
    assert_eq!(next(&mut ctx, 0x0200_0003), (1, 0x0200_0003, 0));
    assert_eq!(next(&mut ctx, 0), (1, 0x9, 0));
    assert_eq!(next(&mut ctx, 0), (1, 0xA, 1));
    assert_eq!(next(&mut ctx, 0), (1, 0x0200_0001, 0x0015_10F1));
    assert_eq!(next(&mut ctx, 0).0, 0);
    let t = io + 0x20;
    mem.write_i64(t, 0);
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, t]) as u32, status::TIMEOUT);
    // Broadcasts reach listeners whose area mask matches.
    k.call_export("XNotifyBroadcast", &mut ctx, &[0x12, 7]);
    k.call_export("XNotifyBroadcast", &mut ctx, &[0x0400_0001, 1]); // friends: not subscribed
    assert_eq!(next(&mut ctx, 0), (1, 0x12, 7));
    assert_eq!(next(&mut ctx, 0).0, 0);
    // A second system listener gets no startup notifications.
    let h2 = k.call_export("XamNotifyCreateListener", &mut ctx, &[1, 10]) as u32;
    assert_eq!(k.call_export("XNotifyGetNext", &mut ctx, &[h2, 0, io, io + 4]), 0);
}

#[test]
fn save_content_round_trip() {
    let k = kernel();
    let mut ctx = guest(k);
    let root = game_dir().join("content");
    k.content.set_root(&root);
    let io = scratch(k, 0x2000);
    let mem = k.mem();
    let (size_ptr, handle_ptr, data, disp, lic, buf, count) =
        (io, io + 4, io + 0x100, io + 8, io + 12, io + 0x400, io + 16);

    // Disc device ids other than HDD/ODD are rejected, as in Xenia.
    let r = k.call_export("XamContentCreateEnumerator", &mut ctx, &[0, 0xFFFF_FFFF, 1, 0, 1, size_ptr, handle_ptr]);
    assert_eq!(r as u32, hresult::E_INVALIDARG);

    // Nothing saved yet.
    assert_eq!(k.call_export("XamContentCreateEnumerator", &mut ctx, &[0, 0, 1, 0, 4, size_ptr, handle_ptr]), 0);
    assert_eq!(mem.read_u32(size_ptr), 4 * 0x134);
    let e = mem.read_u32(handle_ptr);
    assert_eq!(k.call_export("XamEnumerate", &mut ctx, &[e, 0, buf, 4 * 0x134, count, 0]) as u32, win32::NO_MORE_FILES);
    k.call_export("NtClose", &mut ctx, &[e]);

    // XCONTENT_DATA { device 1, type 1 (saved game), "Quick Save", "save1" }.
    mem.fill(data, 0x134, 0);
    mem.write_u32(data, 1);
    mem.write_u32(data + 4, 1);
    let display: Vec<u16> = "Quick Save".encode_utf16().collect();
    mem.write_utf16(data + 8, &display);
    mem.write_bytes(data + 0x108, b"save1");
    let root_name = cstr(k, io + 0x300, "save");
    assert_eq!(k.call_export("XamContentCreate", &mut ctx, &[0, root_name, data, 1, disp, lic, 0]), 0);
    assert_eq!(mem.read_u32(disp), 1, "created");
    // CREATE_NEW again fails while it exists.
    k.call_export("XamContentClose", &mut ctx, &[root_name, 0]);
    assert_eq!(
        k.call_export("XamContentCreate", &mut ctx, &[0, root_name, data, 1, disp, lic, 0]) as u32,
        win32::ALREADY_EXISTS
    );
    // OPEN_EXISTING mounts it as save:, which the title writes through.
    assert_eq!(k.call_export("XamContentCreate", &mut ctx, &[0, root_name, data, 3, disp, lic, 0]), 0);
    assert_eq!(mem.read_u32(disp), 2, "opened");
    let resolved = k.vfs.resolve("save:\\Skyrim.ess").unwrap();
    std::fs::write(&resolved.host_path, b"ESS").unwrap();
    // A second mount under the same name is refused until closed.
    assert_eq!(
        k.call_export("XamContentCreate", &mut ctx, &[0, root_name, data, 3, disp, lic, 0]) as u32,
        win32::INVALID_PARAMETER
    );
    // Close completes through an XOVERLAPPED.
    let ov = io + 0x600;
    mem.fill(ov, 0x1C, 0);
    assert_eq!(k.call_export("XamContentClose", &mut ctx, &[root_name, ov]) as u32, win32::IO_PENDING);
    assert_eq!(mem.read_u32(ov), 0);
    assert!(k.vfs.resolve("save:\\Skyrim.ess").is_err());

    // The save is listed with its display name, one per XamEnumerate call.
    assert_eq!(k.call_export("XamContentCreateEnumerator", &mut ctx, &[0, 0, 1, 0, 1, size_ptr, handle_ptr]), 0);
    let e = mem.read_u32(handle_ptr);
    assert_eq!(k.call_export("XamEnumerate", &mut ctx, &[e, 0, buf, 0x134, count, 0]), 0);
    assert_eq!(mem.read_u32(count), 1);
    assert_eq!(mem.read_u32(buf + 4), 1);
    assert_eq!(mem.read_utf16(buf + 8, display.len() as u32), display);
    assert_eq!(mem.read_cstr(buf + 0x108, 42), b"save1");
    assert_eq!(k.call_export("XamEnumerate", &mut ctx, &[e, 0, buf, 0x134, count, 0]) as u32, win32::NO_MORE_FILES);

    let xuid = k.users.get(0).unwrap().xuid;
    let package = root.join(format!("{xuid:016X}")).join("00000000").join("00000001").join("save1");
    assert_eq!(std::fs::read(package.join("Skyrim.ess")).unwrap(), b"ESS");
    assert_eq!(k.call_export("XamContentDelete", &mut ctx, &[0, data, 0]), 0);
    assert!(!package.exists());
}

#[test]
fn app_messages_and_overlapped_requests() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    // XMPGetPlaybackController: the title owns background music.
    mem.write_u32(io, 2);
    mem.write_u32(io + 4, io + 0x10);
    mem.write_u32(io + 8, io + 0x14);
    mem.write_u32(io + 0x10, 0xFF);
    assert_eq!(k.call_export("XMsgInProcessCall", &mut ctx, &[0xFA, 0x7001B, io, 0]), 0);
    assert_eq!((mem.read_u32(io + 0x10), mem.read_u32(io + 0x14)), (0, 0));
    assert_eq!(k.call_export("XMsgInProcessCall", &mut ctx, &[0x42, 1, 0, 0]) as u32, hresult::E_NOTFOUND);

    // XContentQueryVolumeDeviceType on the disc root: not a content package.
    cstr(k, io + 0x100, "game");
    let ov = io + 0x200;
    mem.fill(ov, 0x1C, 0);
    let r = k.call_export("XMsgStartIORequest", &mut ctx, &[0xFE, 0x20021, ov, io + 0x100, 0x4C]);
    assert_eq!(r as u32, win32::IO_PENDING);
    assert_eq!(mem.read_u32(ov), hresult::E_INVALIDARG);
    assert_eq!(k.call_export("XamGetOverlappedResult", &mut ctx, &[ov, io + 0x300, 0]) as u32, hresult::E_INVALIDARG);
    // Synchronous form returns the result directly.
    let r = k.call_export("XMsgStartIORequest", &mut ctx, &[0xFE, 0x20021, 0, io + 0x100, 0x4C]);
    assert_eq!(r as u32, hresult::E_INVALIDARG);
}

#[test]
fn audio_client_is_pumped_and_frames_drain() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x2000);
    let mem = k.mem();
    const ARG: u32 = 0x00A0_D100;
    let played: Arc<Mutex<Vec<f32>>> = Arc::default();
    let sink = played.clone();
    k.audio.set_output(Arc::new(move |_client, frame| sink.lock().unwrap().push(frame[0])));

    mem.write_u32(io, RECORDER);
    mem.write_u32(io + 4, ARG);
    assert_eq!(k.call_export("XAudioRegisterRenderDriverClient", &mut ctx, &[io, io + 8]), 0);
    let driver = mem.read_u32(io + 8);
    assert_eq!(driver & 0xFFFF_0000, 0x4155_0000);

    // The callback receives a pointer to a word holding the argument.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let calls = common::RECORDED.lock().unwrap().clone();
        if calls.iter().any(|c| c.args[0] != 0 && mem.read_u32(c.args[0]) == ARG) {
            break;
        }
        assert!(Instant::now() < deadline, "audio callback never ran");
        std::thread::sleep(Duration::from_millis(2));
    }

    // A submitted frame is played (handed to the output) in real time.
    let samples = io + 0x100;
    for i in 0..6 * 256 {
        mem.write_f32(samples + i * 4, 0.25);
    }
    assert_eq!(k.call_export("XAudioSubmitRenderDriverFrame", &mut ctx, &[driver, samples]), 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    while played.lock().unwrap().is_empty() {
        assert!(Instant::now() < deadline, "frame never played");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(played.lock().unwrap()[0], 0.25);
    assert_eq!(k.call_export("XAudioUnregisterRenderDriverClient", &mut ctx, &[driver]), 0);
    assert_eq!(k.audio.client_count(), 0);

    assert_eq!(k.call_export("XAudioGetVoiceCategoryVolume", &mut ctx, &[0, io]), 0);
    assert_eq!(mem.read_f32(io), 1.0);
}

#[test]
fn thread_naming_exception_and_misc() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    // EXCEPTION_RECORD { 0x406D1388, ..., 4 params: THREADNAME_INFO { 0x1000, name, -1, 0 } }
    mem.write_u32(io, 0x406D_1388);
    mem.write_u32(io + 0x10, 4);
    mem.write_u32(io + 0x14, 0x1000);
    mem.write_u32(io + 0x18, cstr(k, io + 0x100, "Render"));
    mem.write_u32(io + 0x1C, u32::MAX);
    k.call_export("RtlRaiseException", &mut ctx, &[io]);
    let t = k.adopt_current_thread().unwrap();
    assert_eq!(t.name.lock().unwrap().as_deref(), Some("Render"));

    // KeEnableFpuExceptions records the flag in the KTHREAD.
    k.call_export("KeEnableFpuExceptions", &mut ctx, &[1]);
    assert_eq!(mem.read_u8(t.block.kthread + 0x71), 1);

    // XWSADATA reports the requested version.
    assert_eq!(k.call_export("NetDll_WSAStartup", &mut ctx, &[0, 0x0202, io + 0x200]), 0);
    assert_eq!(mem.read_u16(io + 0x200), 0x0202);
    assert_eq!(k.call_export("XamLoaderGetLaunchDataSize", &mut ctx, &[io]) as u32, win32::NOT_FOUND);
    assert_eq!(mem.read_u32(io), 0);

    // Title terminate registration and removal.
    mem.write_u32(io + 0x300, 0x8200_1000);
    mem.write_u32(io + 0x304, 1);
    k.call_export("ExRegisterTitleTerminateNotification", &mut ctx, &[io + 0x300, 1]);
    k.call_export("ExRegisterTitleTerminateNotification", &mut ctx, &[io + 0x300, 0]);
    assert_eq!(k.call_export("FscSetCacheElementCount", &mut ctx, &[0, 0x100]), 0);
}

#[test]
fn profile_settings_defaults_and_title_writes() {
    let k = kernel();
    let mut ctx = guest(k);
    let root = game_dir().join("content");
    k.content.set_root(&root);
    let io = scratch(k, 0x4000);
    let mem = k.mem();
    const VIBRATION: u32 = 0x1004_0003;
    const TITLE_SPECIFIC1: u32 = 0x63E8_3FFF;
    let (ids, size_ptr, buf, setting) = (io, io + 0x10, io + 0x100, io + 0x80);
    mem.write_u32(ids, VIBRATION);
    mem.write_u32(ids + 4, TITLE_SPECIFIC1);

    // Size query: header + 2 settings + room for the 1000-byte blob.
    mem.write_u32(size_ptr, 0);
    let r = k.call_export("XamUserReadProfileSettings", &mut ctx, &[0, 0, 0, 0, 2, ids, size_ptr, 0, 0]);
    assert_eq!(r as u32, win32::INSUFFICIENT_BUFFER);
    assert_eq!(mem.read_u32(size_ptr), 8 + 2 * 40 + 0x3E8);

    let read = |ctx: &mut runtime_core::PPCContext| {
        let r = k.call_export("XamUserReadProfileSettings", ctx, &[0, 0, 0, 0, 2, ids, size_ptr, buf, 0]);
        assert_eq!(r, 0);
        assert_eq!(mem.read_u32(buf), 2);
        mem.read_u32(buf + 4)
    };
    let s = read(&mut ctx);
    assert_eq!(mem.read_u32(s), 1, "default source");
    assert_eq!(mem.read_u32(s + 8), 0, "user index");
    assert_eq!(mem.read_u32(s + 0x10), VIBRATION);
    assert_eq!((mem.read_u8(s + 0x18), mem.read_u32(s + 0x20)), (1, 3));
    assert_eq!(mem.read_u32(s + 40 + 0x20), 0, "empty title blob");

    // The title stores its options blob, then reads it back.
    let blob = io + 0x1000;
    mem.write_bytes(blob, b"options-v1");
    mem.fill(setting, 40, 0);
    mem.write_u32(setting + 0x10, TITLE_SPECIFIC1);
    mem.write_u8(setting + 0x18, 6);
    mem.write_u32(setting + 0x20, 10);
    mem.write_u32(setting + 0x24, blob);
    assert_eq!(k.call_export("XamUserWriteProfileSettings", &mut ctx, &[0, 0, 1, setting, 0]), 0);
    let s = read(&mut ctx);
    let t = s + 40;
    assert_eq!(mem.read_u32(t), 2, "title source");
    assert_eq!(mem.read_u32(t + 0x20), 10);
    assert_eq!(mem.read_vec(mem.read_u32(t + 0x24), 10), b"options-v1");

    // Persisted for the next run.
    let xuid = k.users.get(0).unwrap().xuid;
    let file = root.join(format!("{xuid:016X}")).join("00000000").join("profile_settings.bin");
    assert!(std::fs::read(file).unwrap().windows(10).any(|w| w == b"options-v1"));
}

#[test]
fn device_selector_answers_hard_drive_asynchronously() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    let listener = k.call_export("XamNotifyCreateListener", &mut ctx, &[1, 10]) as u32;
    // Drain the startup notifications.
    while k.call_export("XNotifyGetNext", &mut ctx, &[listener, 0, io + 0x100, io + 0x104]) != 0 {}

    let (device, ov, event_handle) = (io, io + 0x20, io + 0x40);
    k.call_export("NtCreateEvent", &mut ctx, &[event_handle, 0, 0, 0]);
    let ev = mem.read_u32(event_handle);
    mem.fill(ov, 0x1C, 0);
    mem.write_u32(ov + 0xC, ev);
    let r = k.call_export("XamShowDeviceSelectorUI", &mut ctx, &[0, 1, 0, 0, device, ov]);
    assert_eq!(r as u32, win32::IO_PENDING);
    assert_eq!(mem.read_u32(ov), win32::IO_PENDING);
    let t = io + 0x60;
    mem.write_i64(t, -50_000_000); // 5 s
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[ev, 1, 0, t]) as u32, status::SUCCESS);
    assert_eq!(mem.read_u32(ov), 0);
    assert_eq!(mem.read_u32(device), 1, "hard drive");
    // System UI opened, then closes shortly after.
    assert_eq!(k.call_export("XNotifyGetNext", &mut ctx, &[listener, 0, io + 0x100, io + 0x104]), 1);
    assert_eq!((mem.read_u32(io + 0x100), mem.read_u32(io + 0x104)), (0x9, 1));
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(k.call_export("XNotifyGetNext", &mut ctx, &[listener, 0, io + 0x100, io + 0x104]), 1);
    assert_eq!((mem.read_u32(io + 0x100), mem.read_u32(io + 0x104)), (0x9, 0));
    // XCONTENT_DEVICE_DATA for the chosen device.
    assert_eq!(k.call_export("XamContentGetDeviceData", &mut ctx, &[1, io + 0x200]), 0);
    assert_eq!((mem.read_u32(io + 0x200), mem.read_u32(io + 0x204)), (1, 1));
    assert_eq!(mem.read_u64(io + 0x210), 10 << 30);
    assert_eq!(mem.read_utf16(io + 0x218, 3), "Dum".encode_utf16().collect::<Vec<_>>());
    assert_eq!(
        k.call_export("XamContentGetDeviceData", &mut ctx, &[9, io + 0x200]) as u32,
        win32::DEVICE_NOT_CONNECTED
    );
    // No overlapped: invalid.
    assert_eq!(
        k.call_export("XamShowDeviceSelectorUI", &mut ctx, &[0, 1, 0, 0, device, 0]) as u32,
        win32::INVALID_PARAMETER
    );
}

#[test]
fn keyboard_ui_asks_the_host_and_fills_the_buffer() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    let wstr = |at: u32, s: &str| {
        let mut units: Vec<u16> = s.encode_utf16().collect();
        units.push(0);
        mem.write_utf16(at, &units);
    };
    wstr(io + 0x300, "Prisoner");
    wstr(io + 0x400, "Name");
    kernel_hle::xam::set_text_input(|p| {
        assert_eq!((p.title.as_str(), p.default_text.as_str(), p.max_chars), ("Name", "Prisoner", 6));
        Some(format!("{}!", p.default_text))
    });
    let (buffer, ov, event_handle) = (io, io + 0x20, io + 0x40);
    k.call_export("NtCreateEvent", &mut ctx, &[event_handle, 0, 0, 0]);
    let ev = mem.read_u32(event_handle);
    mem.fill(ov, 0x1C, 0);
    mem.write_u32(ov + 0xC, ev);
    let r = k.call_export("XamShowKeyboardUI", &mut ctx, &[0, 0, io + 0x300, io + 0x400, 0, buffer, 6, ov]);
    assert_eq!(r as u32, win32::IO_PENDING);
    let t = io + 0x60;
    mem.write_i64(t, -50_000_000); // 5 s
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[ev, 1, 0, t]) as u32, status::SUCCESS);
    assert_eq!(mem.read_u32(ov), 0);
    // Truncated to the buffer and terminated.
    assert_eq!(mem.read_utf16(buffer, 6), "Priso\0".encode_utf16().collect::<Vec<_>>());
    assert_eq!(
        k.call_export("XamShowKeyboardUI", &mut ctx, &[0, 0, 0, 0, 0, 0, 6, ov]) as u32,
        win32::INVALID_PARAMETER
    );
}

#[test]
fn title_update_queries_report_no_kinect_and_plain_accounts() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    // The synthetic title has no execution info.
    assert_eq!(k.call_export("XamGetCurrentTitleId", &mut ctx, &[]), 0);
    // US / English offline account; nobody in slot 1.
    assert_eq!(k.call_export("XamUserGetCachedUserFlags", &mut ctx, &[0]), 0x0200_6700);
    assert_eq!(k.call_export("XamUserGetCachedUserFlags", &mut ctx, &[1]), 0);

    mem.fill(io, 0x10, 0xAA);
    assert_eq!(k.call_export("XamNuiGetDeviceStatus", &mut ctx, &[io]), 0xC005_0006);
    assert_eq!(mem.read_vec(io, 0x10), vec![0; 0x10]);
    mem.write_u32(io, 0x50);
    mem.write_u32(io + 4, 0x5874_5373);
    assert_eq!(k.call_export("XamNuiCameraTiltGetStatus", &mut ctx, &[io]) as u32, hresult::E_FAIL);
    assert_eq!(k.call_export("XamVoiceGetMicArrayStatus", &mut ctx, &[]), 0);
    assert_eq!(k.call_export("XamXStudioRequest", &mut ctx, &[6, io]) as u32, hresult::E_FAIL);
    assert_eq!(k.call_export("XamShowNuiTroubleshooterUI", &mut ctx, &[0xFF, 0, 0]), 0);
    mem.fill(io, 0xC, 0xAA);
    k.call_export("XamGetActiveDashAppInfo", &mut ctx, &[io]);
    assert_eq!(mem.read_vec(io, 0xC), vec![0; 0xC]);
}

#[test]
fn ui_is_active_while_a_system_dialog_is_open() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let mem = k.mem();
    assert_eq!(k.call_export("XamIsUIActive", &mut ctx, &[]), 0);
    let (device, ov) = (io, io + 0x100);
    mem.fill(ov, 0x1C, 0);
    let r = k.call_export("XamShowDeviceSelectorUI", &mut ctx, &[0, 1, 0, 0, device, ov]);
    assert_eq!(r as u32, win32::IO_PENDING);
    // Open from just before the answer until a moment after it.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut seen_active = false;
    while mem.read_u32(device) != 1 && Instant::now() < deadline {
        seen_active |= k.call_export("XamIsUIActive", &mut ctx, &[]) == 1;
        std::thread::sleep(Duration::from_millis(1));
    }
    seen_active |= k.call_export("XamIsUIActive", &mut ctx, &[]) == 1;
    assert!(seen_active, "dialog never reported active");
    while k.call_export("XamIsUIActive", &mut ctx, &[]) == 1 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(k.call_export("XamIsUIActive", &mut ctx, &[]), 0);
}

#[test]
fn device_enumerator_and_internal_content() {
    let k = kernel();
    let mut ctx = guest(k);
    let root = game_dir().join("content");
    k.content.set_root(&root);
    let io = scratch(k, 0x2000);
    let mem = k.mem();
    let (size_ptr, handle_ptr, count, buf) = (io, io + 4, io + 8, io + 0x400);

    assert_eq!(k.call_export("XamContentCreateDeviceEnumerator", &mut ctx, &[1, 0, 4, size_ptr, handle_ptr]), 0);
    assert_eq!(mem.read_u32(size_ptr), 4 * 0x50);
    let e = mem.read_u32(handle_ptr);
    assert_eq!(k.call_export("XamEnumerate", &mut ctx, &[e, 0, buf, 4 * 0x50, count, 0]), 0);
    assert_eq!(mem.read_u32(count), 2);
    assert_eq!((mem.read_u32(buf), mem.read_u32(buf + 4)), (1, 1), "hard drive");
    assert_eq!(mem.read_u64(buf + 8), 20 << 30);
    assert_eq!(mem.read_utf16(buf + 0x18, 9), "Dummy HDD".encode_utf16().collect::<Vec<_>>());
    assert_eq!(mem.read_u32(buf + 0x50), 2, "disc drive");
    k.call_export("NtClose", &mut ctx, &[e]);
    // Without read-only devices, only the hard drive.
    assert_eq!(
        k.call_export("XamContentCreateDeviceEnumerator", &mut ctx, &[1, 0x8000_0000, 4, size_ptr, handle_ptr]),
        0
    );
    let e = mem.read_u32(handle_ptr);
    assert_eq!(k.call_export("XamEnumerate", &mut ctx, &[e, 0, buf, 4 * 0x50, count, 0]), 0);
    assert_eq!(mem.read_u32(count), 1);

    // XCONTENT_DATA_INTERNAL for another title: shared content (no user) filed under it.
    let (data, disp, lic) = (io + 0x100, io + 12, io + 16);
    mem.fill(data, 0x200, 0);
    mem.write_u32(data, 1);
    mem.write_u32(data + 4, 2);
    mem.write_bytes(data + 0x108, b"pkg");
    mem.write_u32(data + 0x140, 0x1234_5678);
    let root_name = cstr(k, io + 0x380, "dlc");
    assert_eq!(k.call_export("XamContentCreateInternal", &mut ctx, &[root_name, data, 1, disp, lic, 0, 0, 0]), 0);
    assert_eq!(mem.read_u32(disp), 1);
    let package = root.join("0000000000000000").join("12345678").join("00000002").join("pkg");
    assert!(package.is_dir(), "{}", package.display());
    k.call_export("XamContentClose", &mut ctx, &[root_name, 0]);
    assert_eq!(k.call_export("XamContentDeleteInternal", &mut ctx, &[data, 0]), 0);
    assert!(!package.exists());
    assert_eq!(k.call_export("XamContentDeleteInternal", &mut ctx, &[data, 0]) as u32, win32::FILE_NOT_FOUND);

    let path = cstr(k, io + 0x3C0, "update:\\x.pkg");
    let r = k.call_export("XamContentOpenFile", &mut ctx, &[0, root_name, path, 3, disp, lic, 0]);
    assert_eq!(r as u32, win32::FILE_NOT_FOUND);
}
