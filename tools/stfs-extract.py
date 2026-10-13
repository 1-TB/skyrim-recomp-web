#!/usr/bin/env python3
"""Extract an Xbox 360 STFS package (LIVE / PIRS / CON), such as a title update.

    tools/stfs-extract.py PACKAGE [OUT_DIR]      extract every file (verifying block hashes)
    tools/stfs-extract.py --info PACKAGE         print the package header and file list

Every data block is checked against its SHA-1 in the package's level-0 hash
tables, so a corrupt or misparsed package fails loudly instead of producing
bad files. Read-only ("female") packages, which is what Microsoft-signed
LIVE/PIRS content such as title updates uses, are fully supported. Writable
("male") CON packages keep two copies of each hash table; for those the
active copy is chosen from the level-1 table's status bits (packages up to
~115 MB only).
"""

import hashlib
import os
import sys

BLOCK = 0x1000
PER_TABLE = 0xAA  # hash entries per hash table block
LEVEL1_SPAN = PER_TABLE * PER_TABLE  # 0x70E4 data blocks under one level-1 table

CONTENT_TYPES = {
    0x00001: "saved game", 0x00002: "marketplace content", 0x10000: "profile",
    0x80000: "game demo", 0xB0000: "installer (title update)", 0xD0000: "arcade title",
    0x7000: "games on demand", 0x2000000: "theme", 0x4000: "installed game",
}


def be(data, off, size=4):
    return int.from_bytes(data[off:off + size], "big")


def le(data, off, size):
    return int.from_bytes(data[off:off + size], "little")


def version(v):
    return f"{v >> 28}.{(v >> 24) & 0xF}.{(v >> 8) & 0xFFFF}.{v & 0xFF}"


def utf16(data):
    return data.decode("utf-16-be", "replace").split("\0", 1)[0]


class StfsError(Exception):
    pass


class Package:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size
        h = self.f.read(0xA000 + 0x1000)
        self.header = h
        self.magic = h[:4]
        if self.magic not in (b"LIVE", b"PIRS", b"CON "):
            raise StfsError(f"{path}: not an STFS package (magic {self.magic!r})")
        self.header_size = be(h, 0x340)
        self.content_type = be(h, 0x344)
        self.media_id = be(h, 0x354)
        self.version = be(h, 0x358)
        self.base_version = be(h, 0x35C)
        self.title_id = be(h, 0x360)
        if h[0x379] != 0x24:
            raise StfsError("package has no STFS volume descriptor (an SVOD package?)")
        self.read_only = h[0x37B] & 1  # "block separation" bit 0: 1 = female, one hash table copy
        self.ft_blocks = le(h, 0x37C, 2)
        self.ft_start = le(h, 0x37E, 3)
        self.allocated = be(h, 0x395)
        self.display_name = utf16(h[0x411:0x411 + 0x80])
        self.title_name = utf16(h[0x1691:0x1691 + 0x80])
        self.installer = None
        if self.content_type == 0xB0000:
            kind = h[0x971A:0x971E]
            if kind in (b"TUPD", b"SUPD"):
                self.installer = (kind.decode(), be(h, 0x971E), be(h, 0x9722))
        self.data_start = (self.header_size + BLOCK - 1) & ~(BLOCK - 1)
        self.shift = 0 if self.read_only else 1  # hash tables per slot: 1 (female) or 2 (male)
        if self.shift and self.allocated > LEVEL1_SPAN:
            raise StfsError("writable packages larger than 0x70E4 blocks (~115 MB) are not supported")
        self._tables = {}

    # Layout: each run of 170 data blocks is preceded by its level-0 hash table;
    # each run of 170 level-0 tables by a level-1 table (if the package needs one),
    # and so on. Male packages store every hash table twice.
    def backing_block(self, block):
        n = block + (((block + PER_TABLE) // PER_TABLE) << self.shift)
        if block < PER_TABLE:
            return n
        n += ((block + LEVEL1_SPAN) // LEVEL1_SPAN) << self.shift
        if block < LEVEL1_SPAN:
            return n
        return n + (1 << self.shift)

    def level0_table_block(self, block):
        group = block // PER_TABLE
        n = group * (PER_TABLE + (1 << self.shift))
        if group == 0:
            return n
        n += ((block // LEVEL1_SPAN) + 1) << self.shift
        if block < LEVEL1_SPAN:
            return n
        return n + (1 << self.shift)

    def level1_table_block(self, block):
        # Only reached for male packages, which __init__ limits to one level-1 table.
        return PER_TABLE + (1 << self.shift)

    def read_backing(self, n):
        off = self.data_start + n * BLOCK
        if off + BLOCK > self.size and off >= self.size:
            raise StfsError(f"block at {off:#x} is past the end of the package")
        self.f.seek(off)
        return self.f.read(BLOCK).ljust(BLOCK, b"\0")

    def level0_table(self, block):
        n = self.level0_table_block(block)
        if self.shift and self.allocated > PER_TABLE:
            # Male package with a level-1 table: its entry for this level-0 table says
            # which of the two copies is current (status bit 6).
            l1 = self.read_backing(self.level1_table_block(block))
            entry = (block // PER_TABLE) % PER_TABLE
            if l1[entry * 0x18 + 0x14] & 0x40:
                n += 1
        if n not in self._tables:
            self._tables[n] = self.read_backing(n)
        return self._tables[n]

    def hash_entry(self, block):
        t = self.level0_table(block)
        e = (block % PER_TABLE) * 0x18
        return t[e:e + 0x14], t[e + 0x14], be(t, e + 0x15, 3)

    def read_block(self, block):
        if block >= self.allocated:
            raise StfsError(f"block {block} is beyond the {self.allocated} allocated blocks")
        data = self.read_backing(self.backing_block(block))
        digest, _status, _next = self.hash_entry(block)
        if hashlib.sha1(data).digest() != digest:
            raise StfsError(f"SHA-1 mismatch in block {block}: package is corrupt or misparsed")
        return data

    def chain(self, start, count, consecutive):
        block = start
        for i in range(count):
            yield block
            if consecutive:
                block += 1
            else:
                block = self.hash_entry(block)[2]
                if block == 0xFFFFFF and i + 1 < count:
                    raise StfsError(f"block chain from {start} ends after {i + 1} of {count} blocks")

    def files(self):
        table = b"".join(self.read_block(b) for b in self.chain(self.ft_start, self.ft_blocks, False))
        entries = []
        for off in range(0, len(table), 0x40):
            e = table[off:off + 0x40]
            flags = e[0x28]
            name_len = flags & 0x3F
            if name_len == 0:
                break
            entries.append({
                "name": e[:name_len].decode("ascii", "replace"),
                "dir": bool(flags & 0x80),
                "consecutive": bool(flags & 0x40),
                "blocks": le(e, 0x29, 3),
                "start": le(e, 0x2F, 3),
                "parent": int.from_bytes(e[0x32:0x34], "big", signed=True),
                "size": be(e, 0x34),
            })
        for e in entries:
            parts, seen, cur = [e["name"]], set(), e
            while cur["parent"] != -1:
                if cur["parent"] in seen or not 0 <= cur["parent"] < len(entries):
                    raise StfsError(f"bad parent index for {e['name']}")
                seen.add(cur["parent"])
                cur = entries[cur["parent"]]
                parts.append(cur["name"])
            for p in parts:
                if p in ("", ".", "..") or "/" in p or "\\" in p:
                    raise StfsError(f"unsafe file name {p!r} in package")
            e["path"] = "/".join(reversed(parts))
        return entries

    def extract(self, entry, out):
        remaining = entry["size"]
        with open(out, "wb") as f:
            for b in self.chain(entry["start"], entry["blocks"], entry["consecutive"]):
                if remaining <= 0:
                    break
                data = self.read_block(b)
                f.write(data[:min(BLOCK, remaining)])
                remaining -= BLOCK
        if remaining > 0:
            raise StfsError(f"{entry['path']}: ran out of blocks")

    def describe(self):
        kind = CONTENT_TYPES.get(self.content_type, f"{self.content_type:#x}")
        lines = [
            f"{self.magic.decode().strip()} package: {self.display_name!r} ({self.title_name})",
            f"  title ID {self.title_id:08X}, media ID {self.media_id:08X}, content type: {kind}",
            f"  {'read-only' if self.read_only else 'writable'} STFS, {self.allocated} blocks",
        ]
        if self.installer:
            k, base, cur = self.installer
            lines.append(f"  {k}: installer base version {version(base)}, version {version(cur)}")
        return "\n".join(lines)


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 else 2
    info = argv[1] == "--info"
    args = argv[2:] if info else argv[1:]
    if not args:
        print("missing PACKAGE", file=sys.stderr)
        return 2
    try:
        pkg = Package(args[0])
        files = pkg.files()
        print(pkg.describe())
        if info:
            for e in files:
                print(f"  {'<dir>' if e['dir'] else e['size']:>10}  {e['path']}")
            return 0
        out_dir = args[1] if len(args) > 1 else os.path.splitext(os.path.basename(args[0]))[0]
        for e in files:
            dest = os.path.join(out_dir, *e["path"].split("/"))
            if e["dir"]:
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            pkg.extract(e, dest)
            print(f"  {e['size']:>10}  {e['path']}")
        print(f"extracted {sum(not e['dir'] for e in files)} files to {out_dir} (all block hashes verified)")
        return 0
    except (StfsError, OSError) as err:
        print(f"stfs-extract: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
