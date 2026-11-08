"""PE32 image access for the recompiler: sections, memory-mapped bytes, imports, entry point.

Everything specific to one executable comes from the game's config JSON:

{
  "sha256": "...",                  # required: refuse any other build
  "entry": "0x...",                 # optional: start here instead of the PE entry point
                                    #   (e.g. the original entry behind a DRM wrapper)
  "code_sections": [".text"],       # optional: sections to recompile (default: executable ones)
  "iat": {                          # optional: use this import table instead of the PE's
    "addr": "0x...", "size": "0x...",   # e.g. the program's own IAT that a wrapper fills in
    "dlls": ["kernel32.dll", ...]       # DLL of each zero-terminated group, in order
  },
  "noreturn": ["name", ...],        # optional: extra imports that never return
  "replace": ["0x...", ...],        # optional: functions the project rewrites by hand; recomp.py
                                    #   emits no body for them, the project defines f_XXXXXXXX
  "trace": ["0x...", ...],          # optional: functions that call fn_trace_hook(c, addr) at entry
  "hooks": ["0x...", ...]           # optional: instructions preceded by insn_hook(c, addr) with the
                                    #   registers synced, so the project can read or change state;
                                    #   a nonzero return skips the instruction
}
"""
import hashlib
import json
import struct

import pefile

IMAGE_SCN_MEM_EXECUTE = 0x20000000


def load_config(path):
    cfg = json.load(open(path))
    for k in ("entry",):
        if k in cfg:
            cfg[k] = int(cfg[k], 16)
    if "iat" in cfg:
        cfg["iat"]["addr"] = int(cfg["iat"]["addr"], 16)
        cfg["iat"]["size"] = int(cfg["iat"]["size"], 16)
    return cfg


class Image:
    def __init__(self, path, config):
        cfg = load_config(config) if isinstance(config, str) else config
        self.cfg = cfg
        raw = open(path, "rb").read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != cfg["sha256"]:
            raise SystemExit(f"{path}: sha256 {digest} does not match the config ({cfg['sha256']})")
        self.pe = pefile.PE(data=raw, fast_load=True)
        if self.pe.FILE_HEADER.Machine != 0x14c:
            raise SystemExit(f"{path}: not an i386 PE")
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.end = self.base + self.pe.OPTIONAL_HEADER.SizeOfImage
        self.mem = self.pe.get_memory_mapped_image()
        self.sections = {}
        self.section_flags = {}
        for s in self.pe.sections:
            name = s.Name.rstrip(b"\0").decode()
            start = self.base + s.VirtualAddress
            self.sections[name] = (start, start + s.Misc_VirtualSize)
            self.section_flags[name] = s.Characteristics
        names = cfg.get("code_sections") or [n for n, f in self.section_flags.items() if f & IMAGE_SCN_MEM_EXECUTE]
        self.code = [self.sections[n] for n in names]
        self.text = (min(lo for lo, _ in self.code), max(hi for _, hi in self.code))
        self.data_sections = [n for n, f in self.section_flags.items() if n not in names and not f & IMAGE_SCN_MEM_EXECUTE]
        self.entry = cfg.get("entry", self.base + self.pe.OPTIONAL_HEADER.AddressOfEntryPoint)
        self.imports = self._iat_override(cfg["iat"]) if "iat" in cfg else self._pe_imports()
        self.is_dll = bool(self.pe.FILE_HEADER.Characteristics & 0x2000)
        self.exports = self._pe_exports()
        self.relocs = self._pe_relocs()

    def in_text(self, a):
        for lo, hi in self.code:
            if lo <= a < hi:
                return True
        return False

    def bytes(self, a, n):
        o = a - self.base
        return self.mem[o:o + n]

    def u8(self, a):
        return self.mem[a - self.base]

    def u32(self, a):
        return struct.unpack_from("<I", self.mem, a - self.base)[0]

    def cstr(self, a):
        o = a - self.base
        return self.mem[o:self.mem.index(b"\0", o)].decode("latin1")

    def _pe_imports(self):
        """slot address -> (dll, name or '#ordinal') from the PE import directory"""
        self.pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        out = {}
        for e in getattr(self.pe, "DIRECTORY_ENTRY_IMPORT", []):
            for i in e.imports:
                out[i.address] = (e.dll.decode().lower(), i.name.decode() if i.name else f"#{i.ordinal}")
        return out

    def _pe_relocs(self):
        """addresses of the dwords that hold absolute addresses (HIGHLOW base relocations)"""
        self.pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"]])
        out = []
        for blk in getattr(self.pe, "DIRECTORY_ENTRY_BASERELOC", []):
            for e in blk.entries:
                if e.type == 3:
                    out.append(self.base + e.rva)
        return out

    def _pe_exports(self):
        """[(name or None, ordinal, address, forwarder or None)]"""
        self.pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
        out = []
        d = getattr(self.pe, "DIRECTORY_ENTRY_EXPORT", None)
        for s in (d.symbols if d else []):
            out.append((s.name.decode() if s.name else None, s.ordinal, self.base + s.address,
                        s.forwarder.decode() if s.forwarder else None))
        return out

    def _iat_override(self, iat):
        """an IAT given by address whose slots still hold hint/name RVAs (or ordinals) on disk"""
        groups = iat["dlls"]
        out, g = {}, 0
        for k in range(iat["size"] // 4):
            slot = iat["addr"] + k * 4
            v = self.u32(slot)
            if v == 0:
                g += 1
                continue
            name = f"#{v & 0xffff}" if v & 0x80000000 else self.cstr(self.base + v + 2)
            out[slot] = (groups[g], name)
        return out
