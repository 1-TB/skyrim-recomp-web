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

mod remote_game {
    use std::io;
    use std::sync::Arc;

    use vfs::remote::{RangeReader, RemoteTree};
    use vfs::{Disposition, GameRoot, Kind, Node, OpenRequest, Vfs, VfsError};

    /// Byte `o` of every file is `o as u8`.
    struct Counting;

    impl RangeReader for Counting {
        fn read(&self, _path: &str, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (offset + i as u64) as u8;
            }
            Ok(buf.len())
        }
    }

    #[test]
    fn a_streamed_game_reads_like_a_local_one_and_is_read_only() {
        let tree = Arc::new(RemoteTree::from_manifest("1000\tData/Skyrim.esm\n4\tdefault.xex\n", Arc::new(Counting), 8));
        let game = GameRoot::Remote(tree);
        let vfs = Vfs::with_game(&game);
        assert_eq!(game.read("DEFAULT.XEX").unwrap(), [0, 1, 2, 3]);

        let Ok((Node::File(f), _)) = vfs.open("game:\\data\\SKYRIM.ESM", OpenRequest::read()) else {
            panic!("file not opened");
        };
        assert_eq!(f.len().unwrap(), 1000);
        let mut buf = [0u8; 4];
        assert_eq!(f.read_at(&mut buf, 998).unwrap(), 2);
        assert_eq!(&buf[..2], &[(998u32 % 256) as u8, (999u32 % 256) as u8]);

        let Ok((Node::Directory(d), _)) = vfs.open("d:\\Data", OpenRequest::read()) else { panic!("dir not opened") };
        let names: Vec<String> = d.entries().unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names, ["Skyrim.esm"]);

        assert!(vfs.metadata("game:\\data\\skyrim.esm").unwrap().read_only);
        assert!(matches!(vfs.metadata("game:\\data\\nope.esp"), Err(VfsError::NotFound)));
        assert!(matches!(vfs.metadata("game:\\nodir\\nope.esp"), Err(VfsError::PathNotFound)));
        let write = OpenRequest { write: true, disposition: Disposition::OpenIf, kind: Kind::NonDirectory };
        assert!(matches!(vfs.open("game:\\data\\skyrim.esm", write), Err(VfsError::AccessDenied)));
        assert!(matches!(vfs.open("game:\\new.txt", write), Err(VfsError::AccessDenied)));
    }
}
