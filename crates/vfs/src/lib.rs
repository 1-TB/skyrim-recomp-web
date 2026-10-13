//! Virtual filesystem for guest device paths.
//!
//! Devices (`\Device\Cdrom0`, ...) are mounted onto host directories, and
//! symbolic links (`game:`, `d:`) point at devices, mirroring how the Xbox
//! object manager resolves paths. Lookups are case-insensitive even when the
//! host filesystem is not.
//!
//! In the browser the host directory is an OPFS mount provided by Emscripten's
//! WasmFS (see [`opfs`]), so the same `std::fs` code serves both builds. A device
//! can instead be a [`remote::RemoteTree`] (game files streamed over HTTP).

#[cfg(target_os = "emscripten")]
pub mod opfs;
pub mod path;
pub mod remote;

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

pub use path::matches_pattern;

/// The device the game disc is mounted on.
pub const CDROM_DEVICE: &str = "\\Device\\Cdrom0";

#[derive(Debug)]
pub enum VfsError {
    /// No device or symbolic link matches the path.
    NoSuchDevice(String),
    /// The final component does not exist.
    NotFound,
    /// An intermediate directory does not exist.
    PathNotFound,
    AlreadyExists,
    AccessDenied,
    NotADirectory,
    IsADirectory,
    Io(io::Error),
}

impl fmt::Display for VfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VfsError::NoSuchDevice(p) => write!(f, "no device for path {p:?}"),
            VfsError::NotFound => f.write_str("file not found"),
            VfsError::PathNotFound => f.write_str("path not found"),
            VfsError::AlreadyExists => f.write_str("file exists"),
            VfsError::AccessDenied => f.write_str("access denied"),
            VfsError::NotADirectory => f.write_str("not a directory"),
            VfsError::IsADirectory => f.write_str("is a directory"),
            VfsError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for VfsError {}

impl From<io::Error> for VfsError {
    fn from(e: io::Error) -> Self {
        match e.kind() {
            io::ErrorKind::NotFound => VfsError::NotFound,
            io::ErrorKind::PermissionDenied => VfsError::AccessDenied,
            io::ErrorKind::AlreadyExists => VfsError::AlreadyExists,
            _ => VfsError::Io(e),
        }
    }
}

pub type Result<T> = std::result::Result<T, VfsError>;

/// NtCreateFile CreateDisposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    Supersede,
    Open,
    Create,
    OpenIf,
    Overwrite,
    OverwriteIf,
}

impl Disposition {
    pub fn from_nt(v: u32) -> Option<Self> {
        Some(match v {
            0 => Disposition::Supersede,
            1 => Disposition::Open,
            2 => Disposition::Create,
            3 => Disposition::OpenIf,
            4 => Disposition::Overwrite,
            5 => Disposition::OverwriteIf,
            _ => return None,
        })
    }
}

/// What NtCreateFile reports in IO_STATUS_BLOCK.Information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateAction {
    Superseded = 0,
    Opened = 1,
    Created = 2,
    Overwritten = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Any,
    /// FILE_DIRECTORY_FILE
    Directory,
    /// FILE_NON_DIRECTORY_FILE
    NonDirectory,
}

#[derive(Debug, Clone, Copy)]
pub struct OpenRequest {
    pub write: bool,
    pub disposition: Disposition,
    pub kind: Kind,
}

impl OpenRequest {
    pub fn read() -> Self {
        OpenRequest { write: false, disposition: Disposition::Open, kind: Kind::Any }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
    pub is_dir: bool,
    pub size: u64,
    pub read_only: bool,
    pub created: SystemTime,
    pub accessed: SystemTime,
    pub modified: SystemTime,
}

impl Metadata {
    fn remote(kind: remote::RemoteKind) -> Self {
        let t = SystemTime::UNIX_EPOCH;
        let (is_dir, size) = match kind {
            remote::RemoteKind::File { size } => (false, size),
            remote::RemoteKind::Dir => (true, 0),
        };
        Metadata { is_dir, size, read_only: true, created: t, accessed: t, modified: t }
    }

    fn from_fs(m: &fs::Metadata, device_read_only: bool) -> Self {
        let modified = m.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        Metadata {
            is_dir: m.is_dir(),
            size: if m.is_dir() { 0 } else { m.len() },
            read_only: device_read_only || m.permissions().readonly(),
            created: m.created().unwrap_or(modified),
            accessed: m.accessed().unwrap_or(modified),
            modified,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub metadata: Metadata,
}

pub struct Device {
    pub name: String,
    pub root: PathBuf,
    pub read_only: bool,
    /// Files come from this tree instead of `root` (which is then unused).
    pub remote: Option<Arc<remote::RemoteTree>>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name)
            .field("root", &self.root)
            .field("read_only", &self.read_only)
            .field("remote", &self.remote.is_some())
            .finish()
    }
}

/// Where the title's files come from.
#[derive(Clone)]
pub enum GameRoot {
    Dir(PathBuf),
    Remote(Arc<remote::RemoteTree>),
}

impl GameRoot {
    /// Reads a whole file below the root (`/`-separated relative path).
    pub fn read(&self, rel: &str) -> io::Result<Vec<u8>> {
        match self {
            GameRoot::Dir(root) => fs::read(resolve_case_insensitive(root, &rel.split('/').collect::<Vec<_>>())),
            GameRoot::Remote(tree) => tree.read_file(rel),
        }
    }
}

impl fmt::Display for GameRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameRoot::Dir(root) => write!(f, "{}", root.display()),
            GameRoot::Remote(_) => f.write_str("(streamed from the server)"),
        }
    }
}

/// A resolved guest path.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub device: Arc<Device>,
    /// Guest-visible path, `\Device\X\a\b` with original casing of the request.
    pub guest_path: String,
    /// Host path. Components that exist on the host use the host's casing.
    pub host_path: PathBuf,
}

pub enum Node {
    File(VfsFile),
    Directory(VfsDirectory),
}

enum FileImpl {
    Host(File),
    Remote { tree: Arc<remote::RemoteTree>, path: String, size: u64 },
}

pub struct VfsFile {
    file: FileImpl,
    pub guest_path: String,
    pub host_path: PathBuf,
    pub read_only: bool,
}

impl VfsFile {
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let file = match &self.file {
            FileImpl::Host(f) => f,
            FileImpl::Remote { tree, path, size } => return tree.read_at(path, *size, offset, buf),
        };
        let mut done = 0;
        while done < buf.len() {
            match file.read_at(&mut buf[done..], offset + done as u64) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(done)
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        if self.read_only {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        match &self.file {
            FileImpl::Host(f) => f.write_all_at(buf, offset)?,
            FileImpl::Remote { .. } => return Err(io::Error::from(io::ErrorKind::PermissionDenied)),
        }
        Ok(buf.len())
    }

    pub fn len(&self) -> io::Result<u64> {
        match &self.file {
            FileImpl::Host(f) => Ok(f.metadata()?.len()),
            FileImpl::Remote { size, .. } => Ok(*size),
        }
    }

    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn set_len(&self, len: u64) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        match &self.file {
            FileImpl::Host(f) => f.set_len(len),
            FileImpl::Remote { .. } => Err(io::Error::from(io::ErrorKind::PermissionDenied)),
        }
    }

    pub fn flush(&self) -> io::Result<()> {
        match &self.file {
            FileImpl::Host(f) if !self.read_only => f.sync_data(),
            _ => Ok(()),
        }
    }

    pub fn metadata(&self) -> io::Result<Metadata> {
        match &self.file {
            FileImpl::Host(f) => Ok(Metadata::from_fs(&f.metadata()?, self.read_only)),
            FileImpl::Remote { size, .. } => Ok(Metadata::remote(remote::RemoteKind::File { size: *size })),
        }
    }
}

pub struct VfsDirectory {
    pub guest_path: String,
    pub host_path: PathBuf,
    pub read_only: bool,
    remote: Option<Arc<remote::RemoteTree>>,
}

impl VfsDirectory {
    /// Entries sorted case-insensitively (stable enumeration order across hosts).
    pub fn entries(&self) -> io::Result<Vec<DirEntry>> {
        let mut out = Vec::new();
        if let Some(tree) = &self.remote {
            let listed = tree.list(&remote_path(&self.host_path)).ok_or(io::ErrorKind::NotFound)?;
            out.extend(listed.into_iter().map(|(name, kind)| DirEntry { name, metadata: Metadata::remote(kind) }));
            out.sort_by_key(|a| a.name.to_ascii_lowercase());
            return Ok(out);
        }
        for e in fs::read_dir(&self.host_path)? {
            let e = e?;
            let Ok(name) = e.file_name().into_string() else { continue };
            let Ok(m) = e.metadata() else { continue };
            out.push(DirEntry { name, metadata: Metadata::from_fs(&m, self.read_only) });
        }
        out.sort_by_key(|a| a.name.to_ascii_lowercase());
        Ok(out)
    }

    pub fn metadata(&self) -> io::Result<Metadata> {
        if self.remote.is_some() {
            return Ok(Metadata::remote(remote::RemoteKind::Dir));
        }
        Ok(Metadata::from_fs(&fs::metadata(&self.host_path)?, self.read_only))
    }
}

/// A remote device's paths are stored as `/`-separated relative paths in `host_path`.
fn remote_path(host_path: &Path) -> String {
    host_path.to_string_lossy().replace('\\', "/")
}

#[derive(Default)]
pub struct Vfs {
    devices: RwLock<Vec<Arc<Device>>>,
    /// Lower-cased link name (`game:`) -> target path (`\Device\Cdrom0`).
    links: RwLock<HashMap<String, String>>,
}

impl Vfs {
    pub fn new() -> Self {
        Self::default()
    }

    /// The standard title layout: the disc device at `root`, reachable as
    /// `game:` and `d:`, read-only like the physical disc.
    pub fn with_game_root(root: impl Into<PathBuf>) -> Self {
        Self::with_game(&GameRoot::Dir(root.into()))
    }

    /// [`Vfs::with_game_root`] for either kind of game root.
    pub fn with_game(game: &GameRoot) -> Self {
        let vfs = Vfs::new();
        match game {
            GameRoot::Dir(root) => vfs.mount(CDROM_DEVICE, root.clone(), true),
            GameRoot::Remote(tree) => vfs.mount_remote(CDROM_DEVICE, tree.clone()),
        }
        vfs.add_symlink("game:", CDROM_DEVICE);
        vfs.add_symlink("d:", CDROM_DEVICE);
        vfs
    }

    pub fn mount(&self, device: &str, root: impl Into<PathBuf>, read_only: bool) {
        self.add_device(Device {
            name: device.trim_end_matches(['\\', '/']).to_string(),
            root: root.into(),
            read_only,
            remote: None,
        });
    }

    /// Mounts `device` read-only onto a remote tree.
    pub fn mount_remote(&self, device: &str, tree: Arc<remote::RemoteTree>) {
        self.add_device(Device {
            name: device.trim_end_matches(['\\', '/']).to_string(),
            root: PathBuf::new(),
            read_only: true,
            remote: Some(tree),
        });
    }

    fn add_device(&self, dev: Device) {
        let dev = Arc::new(dev);
        let mut devices = self.devices.write().unwrap();
        devices.retain(|d| !d.name.eq_ignore_ascii_case(&dev.name));
        devices.push(dev);
        // Longest name first so nested device names resolve correctly.
        devices.sort_by_key(|d| std::cmp::Reverse(d.name.len()));
    }

    /// Removes a device; returns whether it was mounted.
    pub fn unmount(&self, device: &str) -> bool {
        let name = device.trim_end_matches(['\\', '/']);
        let mut devices = self.devices.write().unwrap();
        let before = devices.len();
        devices.retain(|d| !d.name.eq_ignore_ascii_case(name));
        devices.len() != before
    }

    /// Target of a symbolic link (`save:` -> `\Device\...`), if registered.
    pub fn symlink_target(&self, link: &str) -> Option<String> {
        let mut key = link.to_ascii_lowercase();
        if !key.ends_with(':') {
            key.push(':');
        }
        self.links.read().unwrap().get(&key).cloned()
    }

    pub fn add_symlink(&self, link: &str, target: &str) {
        let mut key = link.to_ascii_lowercase();
        if !key.ends_with(':') {
            key.push(':');
        }
        self.links.write().unwrap().insert(key, target.trim_end_matches(['\\', '/']).to_string());
    }

    pub fn remove_symlink(&self, link: &str) -> bool {
        let mut key = link.to_ascii_lowercase();
        if !key.ends_with(':') {
            key.push(':');
        }
        self.links.write().unwrap().remove(&key).is_some()
    }

    /// Resolves a guest path to a device and host path. Does not require the
    /// path to exist.
    pub fn resolve(&self, guest: &str) -> Result<Resolved> {
        let stripped = path::strip_object_prefix(guest);
        let expanded = match path::split_drive(stripped) {
            Some((drive, rest)) => {
                let links = self.links.read().unwrap();
                let target =
                    links.get(&drive.to_ascii_lowercase()).ok_or_else(|| VfsError::NoSuchDevice(guest.to_string()))?;
                format!("{target}\\{rest}")
            }
            None => stripped.to_string(),
        };
        // Match the device on the raw components, then apply `.`/`..` only to the
        // remainder so they can never climb out of the device root.
        let raw: Vec<&str> = expanded.split(['\\', '/']).filter(|c| !c.is_empty() && *c != ".").collect();

        let devices = self.devices.read().unwrap();
        for dev in devices.iter() {
            let dev_parts = path::normalize_components(&dev.name);
            if raw.len() < dev_parts.len() || !raw.iter().zip(&dev_parts).all(|(a, b)| a.eq_ignore_ascii_case(b)) {
                continue;
            }
            let rest_joined = raw[dev_parts.len()..].join("\\");
            let rest = path::normalize_components(&rest_joined);
            let rest = rest.as_slice();
            let mut guest_path = dev.name.clone();
            for c in rest {
                guest_path.push('\\');
                guest_path.push_str(c);
            }
            let host_path = match &dev.remote {
                // The listed spelling where the path exists, else the request's.
                Some(tree) => {
                    let joined = rest.join("/");
                    PathBuf::from(tree.lookup(&joined).map_or(joined, |(_, listed)| listed))
                }
                None => resolve_case_insensitive(&dev.root, rest),
            };
            return Ok(Resolved { device: dev.clone(), guest_path, host_path });
        }
        Err(VfsError::NoSuchDevice(guest.to_string()))
    }

    pub fn metadata(&self, guest: &str) -> Result<Metadata> {
        let r = self.resolve(guest)?;
        if let Some(tree) = &r.device.remote {
            let path = remote_path(&r.host_path);
            return match tree.lookup(&path) {
                Some((kind, _)) => Ok(Metadata::remote(kind)),
                None => Err(remote_missing(tree, &path)),
            };
        }
        match fs::metadata(&r.host_path) {
            Ok(m) => Ok(Metadata::from_fs(&m, r.device.read_only)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(missing_kind(&r.host_path)),
            Err(e) => Err(e.into()),
        }
    }

    /// NtCreateFile semantics.
    pub fn open(&self, guest: &str, req: OpenRequest) -> Result<(Node, CreateAction)> {
        let r = self.resolve(guest)?;
        if let Some(tree) = r.device.remote.clone() {
            return open_remote(tree, r, req);
        }
        let read_only = r.device.read_only;
        let existing = fs::metadata(&r.host_path).ok();
        let creates = !matches!(req.disposition, Disposition::Open | Disposition::Overwrite);
        let truncates =
            matches!(req.disposition, Disposition::Supersede | Disposition::Overwrite | Disposition::OverwriteIf);

        match existing {
            Some(m) if m.is_dir() => {
                if req.kind == Kind::NonDirectory {
                    return Err(VfsError::IsADirectory);
                }
                if req.disposition == Disposition::Create {
                    return Err(VfsError::AlreadyExists);
                }
                if truncates {
                    return Err(VfsError::IsADirectory);
                }
                Ok((
                    Node::Directory(VfsDirectory {
                        guest_path: r.guest_path,
                        host_path: r.host_path,
                        read_only,
                        remote: None,
                    }),
                    CreateAction::Opened,
                ))
            }
            Some(_) => {
                if req.kind == Kind::Directory {
                    return Err(VfsError::NotADirectory);
                }
                if req.disposition == Disposition::Create {
                    return Err(VfsError::AlreadyExists);
                }
                if (req.write || truncates) && read_only {
                    return Err(VfsError::AccessDenied);
                }
                let file = OpenOptions::new()
                    .read(true)
                    .write(req.write || truncates)
                    .truncate(truncates)
                    .open(&r.host_path)?;
                let action = match req.disposition {
                    Disposition::Supersede => CreateAction::Superseded,
                    Disposition::Overwrite | Disposition::OverwriteIf => CreateAction::Overwritten,
                    _ => CreateAction::Opened,
                };
                Ok((
                    Node::File(VfsFile {
                        file: FileImpl::Host(file),
                        guest_path: r.guest_path,
                        host_path: r.host_path,
                        read_only: read_only || (!req.write && !truncates),
                    }),
                    action,
                ))
            }
            None => {
                if !creates {
                    return Err(missing_kind(&r.host_path));
                }
                if read_only {
                    return Err(VfsError::AccessDenied);
                }
                let parent_ok = r.host_path.parent().is_some_and(|p| p.is_dir());
                if !parent_ok {
                    return Err(VfsError::PathNotFound);
                }
                if req.kind == Kind::Directory {
                    fs::create_dir(&r.host_path)?;
                    return Ok((
                        Node::Directory(VfsDirectory {
                            guest_path: r.guest_path,
                            host_path: r.host_path,
                            read_only,
                            remote: None,
                        }),
                        CreateAction::Created,
                    ));
                }
                let file = OpenOptions::new().read(true).write(true).create_new(true).open(&r.host_path)?;
                Ok((
                    Node::File(VfsFile {
                        file: FileImpl::Host(file),
                        guest_path: r.guest_path,
                        host_path: r.host_path,
                        read_only: false,
                    }),
                    CreateAction::Created,
                ))
            }
        }
    }
}

/// NtCreateFile on a remote (read-only) device.
fn open_remote(tree: Arc<remote::RemoteTree>, r: Resolved, req: OpenRequest) -> Result<(Node, CreateAction)> {
    let path = remote_path(&r.host_path);
    let truncates =
        matches!(req.disposition, Disposition::Supersede | Disposition::Overwrite | Disposition::OverwriteIf);
    match tree.lookup(&path) {
        None if matches!(req.disposition, Disposition::Open | Disposition::Overwrite) => {
            Err(remote_missing(&tree, &path))
        }
        None => Err(VfsError::AccessDenied),
        Some(_) if req.disposition == Disposition::Create => Err(VfsError::AlreadyExists),
        Some((remote::RemoteKind::Dir, _)) => {
            if req.kind == Kind::NonDirectory || truncates {
                return Err(VfsError::IsADirectory);
            }
            Ok((
                Node::Directory(VfsDirectory {
                    guest_path: r.guest_path,
                    host_path: r.host_path,
                    read_only: true,
                    remote: Some(tree),
                }),
                CreateAction::Opened,
            ))
        }
        Some((remote::RemoteKind::File { size }, listed)) => {
            if req.kind == Kind::Directory {
                return Err(VfsError::NotADirectory);
            }
            if req.write || truncates {
                return Err(VfsError::AccessDenied);
            }
            Ok((
                Node::File(VfsFile {
                    file: FileImpl::Remote { tree, path: listed, size },
                    guest_path: r.guest_path,
                    host_path: r.host_path,
                    read_only: true,
                }),
                CreateAction::Opened,
            ))
        }
    }
}

/// [`missing_kind`] for a remote path.
fn remote_missing(tree: &remote::RemoteTree, path: &str) -> VfsError {
    let parent = path.rsplit_once('/').map_or("", |(p, _)| p);
    if matches!(tree.lookup(parent), Some((remote::RemoteKind::Dir, _))) {
        VfsError::NotFound
    } else {
        VfsError::PathNotFound
    }
}

/// NotFound if the parent directory exists, PathNotFound otherwise (the
/// distinction NT makes between OBJECT_NAME_NOT_FOUND and OBJECT_PATH_NOT_FOUND).
fn missing_kind(host_path: &Path) -> VfsError {
    if host_path.parent().is_some_and(|p| p.is_dir()) { VfsError::NotFound } else { VfsError::PathNotFound }
}

/// Maps guest components onto host entries, matching case-insensitively when
/// the exact name doesn't exist. Components that don't exist are appended as-is.
fn resolve_case_insensitive(root: &Path, components: &[&str]) -> PathBuf {
    let mut path = root.to_path_buf();
    let mut exists = true;
    for c in components {
        if exists {
            let exact = path.join(c);
            if fs::symlink_metadata(&exact).is_ok() {
                path = exact;
                continue;
            }
            let found = fs::read_dir(&path).ok().and_then(|rd| {
                rd.filter_map(|e| e.ok())
                    .find(|e| e.file_name().to_str().is_some_and(|n| n.eq_ignore_ascii_case(c)))
                    .map(|e| e.file_name())
            });
            match found {
                Some(name) => {
                    path.push(name);
                    continue;
                }
                None => exists = false,
            }
        }
        path.push(c);
    }
    path
}
