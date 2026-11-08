"""Find every function in a PE32's code sections and the jump tables and calls inside it.

usage: discover.py CONFIG.json PROGRAM.exe OUT_DIR

Seeds: the original entry, call targets, and addresses of .text code found as immediates in
code or as dwords in .rdata/.data (vtables, callbacks, command-buffer thunks). Each function is
walked by recursive descent. A jmp to another function's start is a tail call. MSVC jump tables
(`jmp [r*4+T]`, optionally with a `movzx r, byte [r2+I]` index table) are sized from the bounds
check before them.

Writes OUT_DIR/functions.json (per function: blocks, jump tables, calls, issues) and prints
coverage of .text.
"""
import collections
import json
import os
import sys

import capstone
from capstone import x86 as X

sys.path.insert(0, os.path.dirname(__file__))
from pe import Image  # noqa: E402

md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
md.detail = True

NORETURN_IMPORTS = {"exit", "_exit", "_amsg_exit", "abort", "_CxxThrowException", "terminate",
                    "ExitProcess", "_invalid_parameter_noinfo_noreturn",
                    "?_Xlength_error@std@@YAXPBD@Z", "?_Xout_of_range@std@@YAXPBD@Z",
                    "?_Xbad_alloc@std@@YAXXZ", "?terminate@@YAXXZ", "_invoke_watson",
                    "?_Xinvalid_argument@std@@YAXPBD@Z", "?_Xruntime_error@std@@YAXPBD@Z"}
JCC = {X.X86_INS_JA, X.X86_INS_JAE, X.X86_INS_JB, X.X86_INS_JBE, X.X86_INS_JE, X.X86_INS_JNE,
       X.X86_INS_JG, X.X86_INS_JGE, X.X86_INS_JL, X.X86_INS_JLE, X.X86_INS_JO, X.X86_INS_JNO,
       X.X86_INS_JP, X.X86_INS_JNP, X.X86_INS_JS, X.X86_INS_JNS, X.X86_INS_JCXZ, X.X86_INS_JECXZ,
       X.X86_INS_LOOP, X.X86_INS_LOOPE, X.X86_INS_LOOPNE}


def is_data_export(name):
    """MSVC-mangled data: vftables (??_7), vbtables (??_8), static members and globals (@@2/@@3)"""
    return bool(name) and name.startswith("?") and (name.startswith(("??_7", "??_8")) or "@@2" in name or "@@3" in name)


class Disasm:
    def __init__(self, img):
        self.img = img
        self.cache = {}

    def at(self, a):
        i = self.cache.get(a)
        if i is None:
            if not self.img.in_text(a):
                return None
            i = next(md.disasm(self.img.bytes(a, 16), a, 1), None)
            self.cache[a] = i if i is not None else False
        return i or None


def imm_target(insn):
    """direct branch/call target or None"""
    op = insn.operands[0] if insn.operands else None
    return op.imm & 0xffffffff if op is not None and op.type == X.X86_OP_IMM else None


def mem_abs(insn):
    """[disp] operand with no base/index -> disp"""
    for op in insn.operands:
        if op.type == X.X86_OP_MEM and op.mem.base == 0 and op.mem.index == 0:
            return op.mem.disp & 0xffffffff
    return None


class Discover:
    def __init__(self, img):
        self.img = img
        self.d = Disasm(img)
        self.data_exports = {a for n, o, a, f in img.exports if not f and is_data_export(n)}
        self.starts = set()
        self.funcs = {}
        self.noreturn = set()

    def plausible_start(self, a):
        """function starts in this binary follow int3/nop padding or a ret, or are called"""
        if not self.img.in_text(a):
            return False
        i = self.d.at(a)
        if i is None:
            return False
        prev = self.img.u8(a - 1)
        if prev in (0xcc, 0x90, 0xc3) or self.img.u8(a - 3) == 0xc2:
            return True
        # an import stub `jmp [iat slot]` is a function wherever it sits
        if i.id == X.X86_INS_JMP and mem_abs(i) in self.img.imports:
            return True
        # right after an unconditional jump: import stubs `jmp [iat]`, tail jumps, `jmp [reg]` thunks
        for back in range(2, 8):
            j = self.d.at(a - back)
            if j is not None and j.size == back and j.id == X.X86_INS_JMP and j.bytes[0] in (0xe9, 0xeb, 0xff):
                return True
        head = self.img.bytes(a, 5)
        return a % 16 == 0 and (head[:3] == b"\x55\x8b\xec" or head[:3] == b"\x8b\xff\x55")

    def seed(self):
        img = self.img
        if not img.is_dll or img.pe.OPTIONAL_HEADER.AddressOfEntryPoint:
            self.starts.add(img.entry)
        for name, ordinal, addr, fwd in img.exports:
            if not fwd and img.in_text(addr) and not is_data_export(name):
                self.starts.add(addr)
        # with relocations we know exactly which dwords are addresses (vtables in code sections)
        for r in img.relocs:
            v = img.u32(r)
            if img.in_text(v) and v not in self.data_exports and self.plausible_start(v):
                self.starts.add(v)
        # linear pass over .text for call rel32 / immediates (cheap, refined later by walking)
        lo, hi = img.text
        a = lo
        imms = collections.Counter()
        while a < hi:
            i = self.d.at(a)
            if i is None:
                a += 1
                continue
            if i.id == X.X86_INS_CALL:
                t = imm_target(i)
                if t is not None and img.in_text(t):
                    self.starts.add(t)
            elif i.id in (X.X86_INS_PUSH, X.X86_INS_MOV):
                for op in i.operands:
                    if op.type == X.X86_OP_IMM and img.in_text(op.imm & 0xffffffff):
                        imms[op.imm & 0xffffffff] += 1
            a += i.size
        for t in imms:
            if self.plausible_start(t):
                self.starts.add(t)
        # function pointers in data
        for sec in img.data_sections:
            s, e = img.sections[sec]
            e = min(e, img.base + len(img.mem))
            for p in range(s, e - 3, 4):
                v = img.u32(p)
                if img.in_text(v) and self.plausible_start(v):
                    self.starts.add(v)

    def table_bound(self, path, reg_name):
        """walk back over the straight-line path to the bounds check `cmp reg, N; ja`"""
        for k in range(len(path) - 1, max(-1, len(path) - 12), -1):
            i = path[k]
            if i.id == X.X86_INS_CMP and len(i.operands) == 2 and i.operands[0].type == X.X86_OP_REG \
                    and i.reg_name(i.operands[0].reg) == reg_name and i.operands[1].type == X.X86_OP_IMM:
                return i.operands[1].imm + 1
        return None

    def jump_table(self, insn, path):
        """returns list of targets or None"""
        op = insn.operands[0]
        if op.type != X.X86_OP_MEM or op.mem.scale != 4 or op.mem.base != 0 or op.mem.index == 0:
            return None
        table = op.mem.disp & 0xffffffff
        reg = insn.reg_name(op.mem.index)
        count = None
        # two-level: movzx reg, byte ptr [r2 + index_table]
        for k in range(len(path) - 1, max(-1, len(path) - 6), -1):
            i = path[k]
            if i.id == X.X86_INS_MOVZX and i.reg_name(i.operands[0].reg) == reg and i.operands[1].type == X.X86_OP_MEM:
                m = i.operands[1].mem
                if m.base and not m.index:
                    r2 = i.reg_name(m.base)
                    n = self.table_bound(path[:k], r2)
                    if n:
                        idx = self.img.bytes(m.disp & 0xffffffff, n)
                        count = max(idx) + 1
                        break
        if count is None:
            count = self.table_bound(path, reg)
        if count is None:
            # unbounded: read while entries point into .text, max 512
            count = 0
            while count < 512 and self.img.in_text(self.img.u32(table + 4 * count)):
                count += 1
        targets = [self.img.u32(table + 4 * n) for n in range(count)]
        if not all(self.img.in_text(t) for t in targets):
            return None
        return table, count, targets

    def walk(self, start):
        """recursive descent over one function: instruction set, branch labels, calls, tables"""
        img, d = self.img, self.d
        insns, labels = set(), {start}
        calls, tails, tables, issues = set(), set(), [], []
        data = set()      # byte ranges inside .text that are jump/index tables
        icalls, returns = 0, False
        work = [start]
        while work:
            a = work.pop()
            path = []
            while a not in insns:
                i = d.at(a)
                if i is None:
                    issues.append(f"bad decode at {a:#x}")
                    break
                insns.add(a)
                path.append(i)
                nxt = a + i.size
                iid = i.id
                if iid in (X.X86_INS_RET, X.X86_INS_RETF):
                    returns = True
                    break
                if iid in (X.X86_INS_INT3, X.X86_INS_HLT, X.X86_INS_UD2):
                    break
                if iid == X.X86_INS_JMP:
                    t = imm_target(i)
                    if t is not None:
                        if t in self.starts and t != start:
                            tails.add(t)
                        elif img.in_text(t):
                            labels.add(t)
                            work.append(t)
                        else:
                            issues.append(f"jmp out of .text {t:#x} at {a:#x}")
                    else:
                        slot = mem_abs(i)
                        if slot is not None and slot in img.imports:
                            tails.add(("imp", slot))
                            if img.imports[slot][1] not in NORETURN_IMPORTS:
                                returns = True
                        else:
                            jt = self.jump_table(i, path)
                            if jt:
                                table, count, targets = jt
                                tables.append({"at": a, "table": table, "count": count, "targets": targets})
                                data.add((table, table + 4 * count))
                                labels.update(targets)
                                work.extend(targets)
                            else:
                                tails.add(("ind", a))   # virtual tail call: through dispatch()
                    break
                if iid in JCC:
                    t = imm_target(i)
                    if t in self.starts and t != start:
                        tails.add(t)          # conditional tail call
                    else:
                        labels.add(t)
                        work.append(t)
                elif iid == X.X86_INS_CALL:
                    t = imm_target(i)
                    nr = False
                    if t is not None:
                        calls.add(t)
                        nr = t in self.noreturn
                    else:
                        slot = mem_abs(i)
                        if slot is not None and slot in img.imports:
                            calls.add(("imp", slot))
                            nr = img.imports[slot][1] in NORETURN_IMPORTS
                        else:
                            icalls += 1
                    if nr or img.u8(nxt) == 0xcc or (nxt in self.starts and nxt != start):
                        break
                if nxt in self.starts and nxt != start:
                    tails.add(nxt)            # falls through into the next function
                    break
                a = nxt
        return {"insns": insns, "labels": labels, "calls": calls, "tails": tails, "tables": tables,
                "issues": issues, "returns": returns, "icalls": icalls, "data": data}

    def run(self):
        self.seed()
        # noreturn fixpoint: walk all, mark functions that never return, re-walk until stable
        for rnd in range(4):
            self.funcs = {}
            queue = sorted(self.starts)
            while queue:
                f = queue.pop()
                if f in self.funcs:
                    continue
                r = self.walk(f)
                self.funcs[f] = r
                for t in list(r["calls"]) + list(r["tails"]):
                    if isinstance(t, int) and self.img.in_text(t) and t not in self.starts:
                        self.starts.add(t)
                        queue.append(t)
            nr = {f for f, r in self.funcs.items() if not r["returns"] and not
                  any(isinstance(t, int) and t not in self.noreturn for t in r["tails"])}
            if nr == self.noreturn:
                break
            self.noreturn = nr
        return self.funcs


def coverage(img, disc):
    lo, hi = img.text
    cov = bytearray(hi - lo)
    for f, r in disc.funcs.items():
        for a in r["insns"]:
            i = disc.d.at(a)
            for k in range(i.size):
                if lo <= a + k < hi:
                    cov[a + k - lo] = 1
        for s, e in r["data"]:
            for k in range(s, e):
                if lo <= k < hi:
                    cov[k - lo] = 2
    gaps = []
    a = 0
    while a < len(cov):
        if cov[a] == 0 and img.u8(lo + a) not in (0xcc,):
            s = a
            while a < len(cov) and cov[a] == 0:
                a += 1
            raw = img.bytes(lo + s, a - s)
            if not all(c in (0xcc, 0x90, 0x00) for c in raw):
                gaps.append((lo + s, a - s))
        else:
            a += 1
    return cov, gaps


def main():
    img = Image(sys.argv[2], sys.argv[1])
    out = sys.argv[3]
    NORETURN_IMPORTS.update(img.cfg.get("noreturn", []))
    os.makedirs(out, exist_ok=True)
    disc = Discover(img)
    funcs = disc.run()
    cov, gaps = coverage(img, disc)
    # unreferenced code with a function-start shape (dead code, or reached through pointers the
    # seed scan missed): translate it too, so a stray pointer still lands in a known function
    for rnd in range(1000):   # until no gap starts like a function
        extra = {a for a, n in ((int(g[0]), g[1]) for g in gaps)
                 if (disc.plausible_start(a) and img.bytes(a, 1)[0] != 0x00)
                 or img.bytes(a, 3) in (b"\x55\x8b\xec", b"\x8b\xff\x55")}
        extra -= disc.starts
        if not extra:
            break
        print(f"round {rnd}: {len(extra)} unreferenced function starts in gaps")
        disc.starts |= extra
        funcs = disc.run()
        cov, gaps = coverage(img, disc)
    code = sum(1 for c in cov if c == 1)
    data = sum(1 for c in cov if c == 2)
    gapbytes = sum(n for _, n in gaps)
    issues = collections.Counter()
    for f, r in funcs.items():
        for s in r["issues"]:
            issues[s.split(" at ")[0].split(":")[0]] += 1
    print(f"functions {len(funcs)}  noreturn {len(disc.noreturn)}  jump tables "
          f"{sum(len(r['tables']) for r in funcs.values())}  indirect calls {sum(r['icalls'] for r in funcs.values())}")
    print(f"code {sum(hi - lo for lo, hi in img.code)} bytes: code {code}, tables {data}, unaccounted non-padding {gapbytes} in {len(gaps)} gaps")
    print("issues:", dict(issues))
    gaps.sort(key=lambda g: -g[1])
    print("largest gaps:", [(hex(a), n) for a, n in gaps[:15]])

    def enc(x):
        return f"{x[0]}:{x[1]:#x}" if isinstance(x, tuple) else f"{x:#x}"
    js = {f"{f:#x}": {"insns": len(r["insns"]), "end": f"{max(r['insns']) if r['insns'] else f:#x}",
                      "labels": sorted(f"{a:#x}" for a in r["labels"] if a is not None),
                      "calls": sorted(enc(c) for c in r["calls"]), "tails": sorted(enc(t) for t in r["tails"]),
                      "tables": [{"at": f"{t['at']:#x}", "table": f"{t['table']:#x}", "count": t["count"],
                                  "targets": [f"{x:#x}" for x in t["targets"]]} for t in r["tables"]],
                      "issues": r["issues"], "returns": r["returns"]}
          for f, r in sorted(funcs.items())}
    json.dump(js, open(os.path.join(out, "functions.json"), "w"))
    json.dump([(hex(a), n) for a, n in gaps], open(os.path.join(out, "gaps.json"), "w"))


if __name__ == "__main__":
    main()
