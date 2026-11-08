"""Translate a PE32 program to C.

usage: recomp.py CONFIG.json PROGRAM.exe FUNCTIONS_JSON OUT_DIR [--module NAME] [--only ADDR,...]

--module NAME (default "main") names the generated tables, so several translated modules
(an executable and the DLLs it ships with or links to) can be linked into one program.
table.c defines `const struct guest_module guest_module_NAME` (see runtime/table.h).

Writes OUT_DIR/funcs_NNN.c (chunks of functions), funcs.h (prototypes), table.c (sorted
address -> function table for dispatch), imports.txt (import shims referenced) and
report.txt (unsupported instructions by mnemonic).
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pe import Image  # noqa: E402
from discover import Discover  # noqa: E402
from x86 import FuncTranslator, fn_name  # noqa: E402

CHUNK = 250


def c_ident(name):
    out = "".join(ch if ch.isalnum() else "_" for ch in name)
    return out


class Ctx:
    def __init__(self, img, funcs, disc):
        self.img = img
        self.funcs = funcs
        self.disc = disc
        self.starts = set(funcs)
        self.noreturn = disc.noreturn
        self.tables = {}
        self.labels = {}
        self.simd_used = set()
        self.import_names = {slot: c_ident(f"{dll.split('.')[0]}_{name}") for slot, (dll, name) in img.imports.items()}

    def is_func(self, a):
        return a in self.starts

    def import_name(self, slot):
        return self.import_names.get(slot)

    def labels_of(self, f):
        return self.labels[f]


def main():
    img = Image(sys.argv[2], sys.argv[1])
    fj = json.load(open(sys.argv[3]))
    out = sys.argv[4]
    only = None
    mod = sys.argv[sys.argv.index("--module") + 1] if "--module" in sys.argv else "main"
    if "--only" in sys.argv:
        only = {int(x, 16) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
    os.makedirs(out, exist_ok=True)
    disc = Discover(img)
    starts = sorted(int(k, 16) for k in fj)
    disc.starts = set(starts)
    # noreturn: functions that never return (recomputed the same way discover.py does)
    walks = {f: disc.walk(f) for f in starts}
    disc.noreturn = {f for f, r in walks.items() if not r["returns"] and not
                     any(isinstance(t, int) for t in r["tails"]) and not any(isinstance(t, tuple) and t[0] == "ind" for t in r["tails"])}
    ctx = Ctx(img, starts, disc)
    for f, r in walks.items():
        ctx.labels[f] = r["labels"]
        for t in r["tables"]:
            ctx.tables[t["at"]] = t["targets"]
    unsupported = collections.Counter()
    examples = {}
    imports = set()
    replaced = {int(a, 16) for a in img.cfg.get("replace", [])}
    ctx.traced = {int(a, 16) for a in img.cfg.get("trace", [])}
    ctx.hooked = {int(a, 16) for a in img.cfg.get("hooks", [])}
    missing = replaced - set(starts)
    if missing:
        raise SystemExit(f"replace: not function starts: {sorted(hex(a) for a in missing)}")
    todo = [f for f in starts if only is None or f in only]
    chunks = [todo[k:k + CHUNK] for k in range(0, len(todo), CHUNK)]
    for n, chunk in enumerate(chunks):
        parts = ['#include "cpu.h"\n#include "funcs.h"\n#include "imports.h"\n#include "simd.h"\n']
        for f in chunk:
            if f in replaced:
                parts.append(f"/* {fn_name(f)}: replaced by a hand-written version (config \"replace\") */\n")
                continue
            tr = FuncTranslator(ctx, f)
            parts.append(tr.translate(walks[f]["insns"], walks[f]["labels"], disc.d))
            imports |= tr.imports
            for a, m, why in tr.unsupported:
                unsupported[why] += 1
                examples.setdefault(why, f"{a:#x}")
        with open(os.path.join(out, f"funcs_{n:03d}.c"), "w") as fh:
            fh.write("\n".join(parts))
    with open(os.path.join(out, "funcs.h"), "w") as fh:
        fh.write('#pragma once\n#include "cpu.h"\n')
        fh.write("".join(f"void {fn_name(f)}(struct cpu *c);\n" for f in starts))
    with open(os.path.join(out, "table.c"), "w") as fh:
        fh.write('#include "funcs.h"\n#include "table.h"\nstatic const struct func_entry funcs[] = {\n')
        fh.write("".join(f"  {{0x{f:x}u, {fn_name(f)}}},\n" for f in starts))
        fh.write("};\nstatic const struct export_entry exports[] = {\n")
        for name, ordinal, addr, fwd in img.exports:
            if fwd:
                continue
            nm = '"' + name.replace('\\', '\\\\').replace('"', '\\"') + '"' if name else "0"
            fh.write(f"  {{{nm}, {ordinal}u, 0x{addr:x}u}},\n")
        fh.write("  {0, 0, 0}\n};\n")
        fh.write(f"extern const struct import_entry guest_imports_{mod}[];\n")
        fh.write(f"const struct guest_module guest_module_{mod} = {{\n"
                 f"  \"{mod}\", funcs, {len(starts)}u, guest_imports_{mod}, {len(img.imports)}u,\n"
                 f"  exports, {sum(1 for e in img.exports if not e[3])}u,\n"
                 f"  0x{img.entry:x}u, 0x{img.base:x}u, 0x{img.end:x}u, {int(img.is_dll)}\n}};\n")
    allimp = sorted(set(ctx.import_names.values()))
    with open(os.path.join(out, "imports.txt"), "w") as fh:
        for slot, (dll, name) in sorted(img.imports.items()):
            fh.write(f"{slot:#x} {dll} {name} {ctx.import_names[slot]} {'used' if ctx.import_names[slot] in imports else 'unused'}\n")
    with open(os.path.join(out, "imports.h"), "w") as fh:
        fh.write('#pragma once\n#include "cpu.h"\n/* native shims for the game\'s imports (runtime/imports/) */\n')
        fh.write("".join(f"void imp_{n}(struct cpu *c);\n" for n in allimp))
    # weak placeholder shims (the runtime's real ones override them) and the import slot table
    with open(os.path.join(out, "import_stubs.c"), "w") as fh:
        fh.write('#include "cpu.h"\n#include "imports.h"\n#include "table.h"\n')
        fh.write("void import_missing(struct cpu *c, const char *dll, const char *name);\n")
        names = {}
        for slot, (dll, name) in sorted(img.imports.items()):
            names.setdefault(ctx.import_names[slot], (dll, name))
        for ident, (dll, name) in sorted(names.items()):
            fh.write(f'__attribute__((weak)) void imp_{ident}(struct cpu *c) {{ import_missing(c, "{dll}", "{name}"); }}\n')
        fh.write(f"const struct import_entry guest_imports_{mod}[] = {{\n")
        for slot, (dll, name) in sorted(img.imports.items()):
            fh.write(f'  {{0x{slot:x}u, "{dll}", "{name}", imp_{ctx.import_names[slot]}}},\n')
        fh.write("  {0, 0, 0, 0}\n};\n")
    with open(os.path.join(out, "simd_used.txt"), "w") as fh:
        fh.write("".join(f"{n}\n" for n in sorted(ctx.simd_used)))
    with open(os.path.join(out, "report.txt"), "w") as fh:
        for why, k in unsupported.most_common():
            fh.write(f"{k:7d} {why}  e.g. {examples[why]}\n")
    print(f"{len(todo)} functions in {len(chunks)} files; {len(imports)} of {len(allimp)} imports called directly; "
          f"{sum(unsupported.values())} unsupported instructions ({len(unsupported)} kinds); {len(ctx.simd_used)} simd forms")


if __name__ == "__main__":
    main()
