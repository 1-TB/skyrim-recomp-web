#!/usr/bin/env python3
"""Generates src/ffi.rs: Rust declarations for the subset of webgpu.h that is
identical in wgpu-native's and emdawnwebgpu's headers.

usage: gen_ffi.py <wgpu-native webgpu.h> <emdawnwebgpu webgpu.h> <out.rs>

Items are emitted in the order of the first header. Anything whose normalized
definition differs between the two headers (or exists in only one) is left out
and listed on stderr; such items belong in the per-platform extension modules.
"""
import re
import sys

PRIMS = {
    'uint8_t': 'u8', 'uint16_t': 'u16', 'uint32_t': 'u32', 'uint64_t': 'u64', 'int32_t': 'i32',
    'int64_t': 'i64', 'size_t': 'usize', 'float': 'f32', 'double': 'f64', 'int': 'core::ffi::c_int',
    'char': 'core::ffi::c_char', 'void': 'core::ffi::c_void',
}
KEYWORDS = {'type', 'loop', 'ref', 'match', 'mod', 'move', 'struct', 'fn', 'impl', 'use', 'where', 'self'}
CONSTS = {
    'WGPU_TRUE': ('WGPUBool', '1'),
    'WGPU_FALSE': ('WGPUBool', '0'),
    'WGPU_ARRAY_LAYER_COUNT_UNDEFINED': ('u32', 'u32::MAX'),
    'WGPU_COPY_STRIDE_UNDEFINED': ('u32', 'u32::MAX'),
    'WGPU_DEPTH_CLEAR_VALUE_UNDEFINED': ('f32', 'f32::NAN'),
    'WGPU_DEPTH_SLICE_UNDEFINED': ('u32', 'u32::MAX'),
    'WGPU_LIMIT_U32_UNDEFINED': ('u32', 'u32::MAX'),
    'WGPU_LIMIT_U64_UNDEFINED': ('u64', 'u64::MAX'),
    'WGPU_MIP_LEVEL_COUNT_UNDEFINED': ('u32', 'u32::MAX'),
    'WGPU_QUERY_SET_INDEX_UNDEFINED': ('u32', 'u32::MAX'),
    'WGPU_STRLEN': ('usize', 'usize::MAX'),
    'WGPU_WHOLE_MAP_SIZE': ('usize', 'usize::MAX'),
    'WGPU_WHOLE_SIZE': ('u64', 'u64::MAX'),
}


def strip(src):
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    src = re.sub(r'//[^\n]*', '', src)
    src = src.replace('WGPU_NULLABLE ', '')
    return src


def norm(s):
    return ' '.join(s.split())


class Header:
    def __init__(self, path):
        raw = open(path).read()
        self.inits = {}
        for m in re.finditer(r'#define WGPU_\w+_INIT _wgpu_MAKE_INIT_STRUCT\((WGPU\w+), \{(.*?)\}\)\n', raw, re.S):
            body = norm(m.group(2).replace('\\\n', '\n'))
            body = re.sub(r'_wgpu_MAKE_INIT_STRUCT\(WGPUChainedStruct, \{ /\*\.next=\*/NULL _wgpu_COMMA '
                          r'/\*\.sType=\*/WGPUSType_(\w+) _wgpu_COMMA \}\)', r'CHAIN(\1)', body)
            vals = re.findall(r'/\*\.(\w+)=\*/(.*?)\s*_wgpu_COMMA', body)
            self.inits[m.group(1)] = [(k, norm(v)) for k, v in vals]
        src = strip(raw)
        src = re.sub(r'#define[^\n]*(\\\n[^\n]*)*', '', src)
        self.items = []  # (kind, name, data)
        pos = 0
        pat = re.compile(
            r'typedef struct (WGPU\w+)Impl\* (WGPU\w+) WGPU_OBJECT_ATTRIBUTE;'
            r'|typedef enum (WGPU\w+) \{(.*?)\} \3 WGPU_ENUM_ATTRIBUTE;'
            r'|typedef struct (WGPU\w+) \{(.*?)\} \5 WGPU_STRUCTURE_ATTRIBUTE;'
            r'|typedef (WGPUFlags|uint64_t|uint32_t) (WGPU\w+);'
            r'|static const (WGPU\w+) (WGPU\w+?)_(\w+) = (\w+);'
            r'|typedef void \(\*(WGPU\w+)\)\((.*?)\) WGPU_FUNCTION_ATTRIBUTE;'
            r'|WGPU_EXPORT ([\w\s\*]+?)\s*\b(wgpu\w+)\((.*?)\) WGPU_FUNCTION_ATTRIBUTE;', re.S)
        for m in pat.finditer(src, pos):
            g = m.groups()
            if g[0]:
                self.items.append(('handle', g[1], None))
            elif g[2]:
                vals = []
                for line in g[3].split(','):
                    line = norm(line)
                    if not line:
                        continue
                    k, v = [x.strip() for x in line.split('=')]
                    vals.append((k[len(g[2]) + 1:], v))
                self.items.append(('enum', g[2], vals))
            elif g[4]:
                fields = []
                for decl in g[5].split(';'):
                    decl = norm(decl)
                    if decl:
                        fields.append(decl)
                self.items.append(('struct', g[4], fields))
            elif g[7]:
                self.items.append(('alias', g[7], g[6]))
            elif g[8]:
                self.items.append(('flagval', g[9], (g[10], g[11])))
            elif g[12]:
                if g[12].startswith('WGPUProc') and g[12] != 'WGPUProc':
                    continue
                self.items.append(('callback', g[12], norm(g[13])))
            elif g[15]:
                self.items.append(('func', g[15], (norm(g[14]), norm(g[16]))))
        self.flagvals = {}
        for kind, name, data in self.items:
            if kind == 'flagval':
                self.flagvals.setdefault(name, []).append(data)
        self.by_name = {}
        for kind, name, data in self.items:
            if kind == 'flagval':
                continue
            self.by_name[name] = (kind, data)


def rust_type(ctype, handles):
    """Maps a C declarator type (without the name) to Rust."""
    t = ctype.replace('struct ', '').strip()
    # pointer chains, e.g. "WGPUFoo const *", "char const *", "void *", "WGPUFoo *"
    stars = []
    while t.endswith('*'):
        t = t[:-1].strip()
        if t.endswith('const'):
            t = t[:-5].strip()
            stars.append('*const ')
        else:
            stars.append('*mut ')
    if t.startswith('const '):
        t = t[6:].strip()
        if stars:
            stars[-1] = '*const '
    base = PRIMS.get(t, t)
    return ''.join(reversed(stars)) + base if stars else base


def split_decl(decl):
    m = re.match(r'(.*?)\s*\b(\w+)$', decl)
    return m.group(1), m.group(2)


def field_name(n):
    return f'r#{n}' if n in KEYWORDS else n


def variant_name(n):
    # `2DArray` -> `D2Array`, as wgpu spells texture dimensions.
    return f'D{n[0]}{n[2:]}' if re.match(r'\dD', n) else n


def main():
    a, b = Header(sys.argv[1]), Header(sys.argv[2])
    common = set()
    for name, (kind, data) in a.by_name.items():
        other = b.by_name.get(name)
        if kind == 'enum' and other and other[0] == 'enum':
            bv = dict(other[1])
            if all(bv.get(k, v) == v for k, v in data):
                common.add(name)
                extra = sorted(set(bv) ^ set(dict(data)))
                if extra:
                    print(f'enum {name}: values in one header only: {extra}', file=sys.stderr)
                continue
        if other == (kind, data):
            if kind == 'alias' and a.flagvals.get(name) != b.flagvals.get(name):
                print(f'differs: {name} (flag values)', file=sys.stderr)
                continue
            common.add(name)
        else:
            print(f'{"differs" if other else "only in A"}: {name}', file=sys.stderr)
    for name in b.by_name:
        if name not in a.by_name:
            print(f'only in B: {name}', file=sys.stderr)

    # A struct is only usable if every type it names is common too.
    def deps_ok(text):
        return all(t in common or t in ('WGPUFlags', 'WGPUBool') for t in re.findall(r'\bWGPU\w+', text))

    changed = True
    while changed:
        changed = False
        for name in list(common):
            kind, data = a.by_name[name]
            text = ' '.join(data) if kind == 'struct' else (' '.join(data) if kind == 'func' else str(data))
            if kind in ('struct', 'callback', 'func') and not deps_ok(text.replace(name, '')):
                common.discard(name)
                print(f'dropped (dependency): {name}', file=sys.stderr)
                changed = True

    handles = {n for n, (k, _) in a.by_name.items() if k == 'handle'}
    enums = {n for n, (k, _) in a.by_name.items() if k == 'enum'}
    flags = {n for n, (k, d) in a.by_name.items() if k == 'alias' and d == 'WGPUFlags'}
    struct_fields = {}
    out = []
    w = out.append
    w('// @generated by tools/gen_ffi.py from wgpu-native v29.0.1.1 and emdawnwebgpu v20260423.175430\n')
    w('// webgpu.h (webgpu-native/webgpu-headers, BSD-3-Clause). Only items identical in both are present.\n')
    w('#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, clippy::all)]\n\n')
    for cname, (ty, val) in CONSTS.items():
        w(f'pub const {cname}: {ty} = {val};\n')
    w('\n')

    funcs = []
    for kind, name, data in a.items:
        if name not in common or kind == 'flagval':
            continue
        if kind == 'handle':
            w(f'#[repr(C)]\npub struct {name}Impl {{\n    _private: [u8; 0],\n}}\n')
            w(f'pub type {name} = *mut {name}Impl;\n\n')
        elif kind == 'alias':
            if data == 'WGPUFlags':
                w('#[repr(transparent)]\n#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]\n')
                w(f'pub struct {name}(pub u64);\nimpl {name} {{\n')
                for vname, v in a.flagvals.get(name, []):
                    w(f'    pub const {variant_name(vname)}: Self = Self({v});\n')
                w('}\n')
                w(f'flag_ops!({name});\n\n')
            else:
                w(f'pub type {name} = {PRIMS[data]};\n\n')
        elif kind == 'enum':
            w('#[repr(transparent)]\n#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]\n')
            w(f'pub struct {name}(pub u32);\nimpl {name} {{\n')
            bv = dict(b.by_name[name][1])
            for vname, v in data:
                if vname == 'Force32' or vname not in bv:
                    continue
                w(f'    pub const {variant_name(vname)}: Self = Self({v});\n')
            w('}\n\n')
        elif kind == 'callback':
            params = [] if data == 'void' else [rust_type(split_decl(p)[0], handles) for p in data.split(', ')]
            w(f'pub type {name} = Option<unsafe extern "C" fn({", ".join(params)})>;\n\n')
        elif kind == 'struct':
            w(f'#[repr(C)]\n#[derive(Copy, Clone, Debug)]\npub struct {name} {{\n')
            fl = []
            for decl in data:
                ctype, fname = split_decl(decl)
                rt = rust_type(ctype, handles)
                fl.append((fname, rt))
                w(f'    pub {field_name(fname)}: {rt},\n')
            struct_fields[name] = fl
            w('}\n\n')
        elif kind == 'func':
            ret, params = data
            ps = []
            if params != 'void':
                for p in params.split(', '):
                    ctype, pname = split_decl(p)
                    ps.append(f'{field_name(pname)}: {rust_type(ctype, handles)}')
            r = rust_type(ret, handles)
            funcs.append(f'    pub fn {name}({", ".join(ps)}){"" if r == "core::ffi::c_void" else " -> " + r};\n')

    # Default impls mirroring the WGPU_*_INIT macros (identical in both headers).
    init_macro_to_struct = {}
    for sname in a.inits:
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', sname[4:]).upper()
        init_macro_to_struct[f'WGPU_{snake}_INIT'] = sname
    for sname, fl in struct_fields.items():
        if sname not in a.inits or a.inits[sname] != b.inits.get(sname):
            w(f'impl Default for {sname} {{\n    fn default() -> Self {{\n        unsafe {{ core::mem::zeroed() }}\n    }}\n}}\n\n')
            continue
        vals = dict(a.inits[sname])
        w(f'impl Default for {sname} {{\n    fn default() -> Self {{\n        Self {{\n')
        for fname, rt in fl:
            v = vals[fname]
            if v == 'NULL':
                if rt.startswith('*mut'):
                    e = 'core::ptr::null_mut()'
                elif rt.startswith('*const'):
                    e = 'core::ptr::null()'
                elif rt in handles:
                    e = 'core::ptr::null_mut()'
                else:
                    e = 'None'
            elif v.startswith('CHAIN('):
                e = f'WGPUChainedStruct {{ next: core::ptr::null_mut(), sType: WGPUSType::{v[6:-1]} }}'
            elif v.startswith('_wgpu_ENUM_ZERO_INIT'):
                e = f'{rt}(0)'
            elif v == '_wgpu_STRUCT_ZERO_INIT':
                e = 'unsafe { core::mem::zeroed() }'
            elif v in CONSTS:
                e = v
            elif re.match(r'WGPU_\w+_INIT$', v):
                e = 'Default::default()'
            elif re.match(r'WGPU\w+?_\w+$', v) and (rt in enums or rt in flags):
                e = f'{rt}::{variant_name(v[len(rt) + 1:])}'
            elif re.match(r'^[0-9.]+f?$|^0x[0-9A-Fa-f]+$', v):
                num = v.rstrip('f')
                if rt in ('f32', 'f64'):
                    e = num + ('0' if num.endswith('.') else ('' if '.' in num else '.0'))
                elif rt in enums or rt in flags:
                    e = f'{rt}({num})'
                else:
                    e = num
            else:
                raise SystemExit(f'unhandled init value {sname}.{fname} = {v}')
            w(f'            {field_name(fname)}: {e},\n')
        w('        }\n    }\n}\n\n')

    w('unsafe extern "C" {\n')
    out.extend(funcs)
    w('}\n')
    with open(sys.argv[3], 'w') as f:
        f.write(''.join(out))


if __name__ == '__main__':
    main()
