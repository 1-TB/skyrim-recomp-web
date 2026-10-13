use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use vfs::{CreateAction, Disposition, Kind, Node, OpenRequest, Vfs, VfsError};

/// A fresh directory under the target dir with a small fake title tree.
fn fixture() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "vfs-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("Data/Interface")).unwrap();
    fs::write(dir.join("default.xex"), b"XEX2").unwrap();
    fs::write(dir.join("Data/Skyrim.esm"), b"TES4data").unwrap();
    fs::write(dir.join("Data/Interface/Fonts.swf"), b"FWS").unwrap();
    dir
}

#[test]
fn game_and_d_resolve_to_the_disc_root() {
    let root = fixture();
    let v = Vfs::with_game_root(&root);
    let a = v.resolve("game:\\Data\\Skyrim.esm").unwrap();
    let b = v.resolve("d:\\Data\\Skyrim.esm").unwrap();
    let c = v.resolve("\\Device\\Cdrom0\\Data\\Skyrim.esm").unwrap();
    assert_eq!(a.host_path, root.join("Data/Skyrim.esm"));
    assert_eq!(a.host_path, b.host_path);
    assert_eq!(a.host_path, c.host_path);
    assert_eq!(a.guest_path, "\\Device\\Cdrom0\\Data\\Skyrim.esm");
}

#[test]
fn resolution_is_case_insensitive_and_keeps_host_casing() {
    let root = fixture();
    let v = Vfs::with_game_root(&root);
    for p in [
        "GAME:\\data\\SKYRIM.ESM",
        "game:/DATA/skyrim.esm",
        "\\??\\D:\\Data\\skyrim.Esm",
        "\\device\\CDROM0\\data\\skyrim.esm",
    ] {
        assert_eq!(v.resolve(p).unwrap().host_path, root.join("Data/Skyrim.esm"), "{p}");
    }
}

#[test]
fn dot_components_cannot_escape_the_device() {
    let root = fixture();
    let v = Vfs::with_game_root(&root);
    assert_eq!(v.resolve("game:\\Data\\.\\Interface\\..\\Skyrim.esm").unwrap().host_path, root.join("Data/Skyrim.esm"));
    let escaped = v.resolve("game:\\..\\..\\..\\etc\\passwd").unwrap();
    assert!(escaped.host_path.starts_with(&root), "{:?}", escaped.host_path);
    assert_eq!(escaped.host_path, root.join("etc/passwd"));
}

#[test]
fn unknown_devices_are_rejected() {
    let v = Vfs::with_game_root(fixture());
    assert!(matches!(v.resolve("cache:\\x"), Err(VfsError::NoSuchDevice(_))));
    assert!(matches!(v.resolve("\\Device\\Harddisk0\\Partition1\\x"), Err(VfsError::NoSuchDevice(_))));
}

#[test]
fn extra_devices_and_links_can_be_mounted() {
    let root = fixture();
    let cache = root.join("cache");
    fs::create_dir_all(&cache).unwrap();
    let v = Vfs::with_game_root(&root);
    v.mount("\\Device\\Harddisk0\\Cache0", &cache, false);
    v.add_symlink("cache", "\\Device\\Harddisk0\\Cache0");
    let (node, action) = v
        .open(
            "cache:\\new.bin",
            OpenRequest { write: true, disposition: Disposition::Create, kind: Kind::NonDirectory },
        )
        .unwrap();
    assert_eq!(action, CreateAction::Created);
    let Node::File(f) = node else { panic!("expected file") };
    f.write_at(b"hello", 0).unwrap();
    assert_eq!(fs::read(cache.join("new.bin")).unwrap(), b"hello");
    assert!(v.remove_symlink("cache:"));
    assert!(v.resolve("cache:\\new.bin").is_err());
}

#[test]
fn open_reads_files_and_lists_directories() {
    let root = fixture();
    let v = Vfs::with_game_root(&root);
    let (node, action) = v.open("game:\\DATA\\skyrim.esm", OpenRequest::read()).unwrap();
    assert_eq!(action, CreateAction::Opened);
    let Node::File(f) = node else { panic!("expected file") };
    assert_eq!(f.len().unwrap(), 8);
    let mut buf = [0u8; 4];
    assert_eq!(f.read_at(&mut buf, 4).unwrap(), 4);
    assert_eq!(&buf, b"data");
    assert_eq!(f.read_at(&mut buf, 6).unwrap(), 2, "short read at EOF");

    let (node, _) = v.open("game:\\data", OpenRequest::read()).unwrap();
    let Node::Directory(d) = node else { panic!("expected directory") };
    let names: Vec<_> = d.entries().unwrap().into_iter().map(|e| e.name).collect();
    assert_eq!(names, ["Interface", "Skyrim.esm"]);
}

#[test]
fn disc_device_is_read_only() {
    let root = fixture();
    let v = Vfs::with_game_root(&root);
    let w = |d| OpenRequest { write: true, disposition: d, kind: Kind::Any };
    assert!(matches!(v.open("game:\\Data\\Skyrim.esm", w(Disposition::Open)), Err(VfsError::AccessDenied)));
    assert!(matches!(v.open("game:\\Data\\new.txt", w(Disposition::Create)), Err(VfsError::AccessDenied)));
    let (node, _) = v.open("game:\\Data\\Skyrim.esm", OpenRequest::read()).unwrap();
    let Node::File(f) = node else { panic!() };
    assert!(f.write_at(b"x", 0).is_err());
}

#[test]
fn missing_files_distinguish_name_and_path() {
    let v = Vfs::with_game_root(fixture());
    assert!(matches!(v.open("game:\\Data\\nope.esp", OpenRequest::read()), Err(VfsError::NotFound)));
    assert!(matches!(v.open("game:\\NoDir\\nope.esp", OpenRequest::read()), Err(VfsError::PathNotFound)));
    assert!(matches!(v.metadata("game:\\NoDir\\x"), Err(VfsError::PathNotFound)));
}

#[test]
fn directory_and_file_kind_checks() {
    let v = Vfs::with_game_root(fixture());
    let dir_only = OpenRequest { write: false, disposition: Disposition::Open, kind: Kind::Directory };
    let file_only = OpenRequest { write: false, disposition: Disposition::Open, kind: Kind::NonDirectory };
    assert!(matches!(v.open("game:\\Data\\Skyrim.esm", dir_only), Err(VfsError::NotADirectory)));
    assert!(matches!(v.open("game:\\Data", file_only), Err(VfsError::IsADirectory)));
    let m = v.metadata("game:\\Data").unwrap();
    assert!(m.is_dir);
    assert!(m.read_only);
}
