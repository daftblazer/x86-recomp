"""Differential test of the translator against Unicorn, on instruction forms taken from a program.

usage: fuzz.py CONFIG.json PROGRAM.exe FUNCTIONS_JSON WORK_DIR [--per-form N] [--states K] [--only MNEMONIC,...]

The program image is mapped at its own base in both, so absolute addresses work; scratch,
stack and code areas sit at 0x20000000-0x31000fff, which must not overlap the image.

For each distinct instruction form (mnemonic + operand shapes) up to N real instances are
picked; each runs from K random input states through
  - Unicorn (reference): prelude loading x87/MMX state, the original bytes, a postlude saving it
  - the translator's C for that single instruction, compiled into a harness (WORK_DIR/harness)
and the final registers, defined flags, memory writes, x87 stack/condition codes and vector
registers are compared. Writes WORK_DIR/fuzz_report.txt.
"""
import collections
import json
import os
import random
import struct
import subprocess
import sys

import capstone
from capstone import x86 as X
import unicorn as U
from unicorn import x86_const as UX

sys.path.insert(0, os.path.dirname(__file__))
from pe import Image  # noqa: E402
from discover import Discover  # noqa: E402
from x86 import FuncTranslator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRATCH, SCRATCH_SIZE = 0x20000000, 0x100000
ADDR_BASE = SCRATCH + 0x80000
STACK = SCRATCH + 0xc0000
IO = 0x31000000
CODE = 0x30000000
IMG_LO = IMG_HI = 0        # set from the image in main()

SKIP = {"call", "jmp", "ret", "retf", "int3", "hlt", "ud2", "loop", "loope", "loopne", "jecxz", "jcxz",
        "lcall", "ljmp", "iretd", "int", "int1", "into", "in", "out", "insb", "insd", "outsb", "outsd",
        "cli", "sti", "ltr", "bound", "arpl", "les", "lds", "salc", "aaa", "aas", "daa", "das", "aam", "aad",
        "fsetpm", "wait", "xlatb", "pushal", "popal", "enter", "leave", "prefetcht1"}
REGS = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
UREG = [UX.UC_X86_REG_EAX, UX.UC_X86_REG_ECX, UX.UC_X86_REG_EDX, UX.UC_X86_REG_EBX,
        UX.UC_X86_REG_ESP, UX.UC_X86_REG_EBP, UX.UC_X86_REG_ESI, UX.UC_X86_REG_EDI]
CF, PF, ZF, SF, OF = 1, 4, 0x40, 0x80, 0x800
ALL = CF | PF | ZF | SF | OF
POOL = [0, 1, 2, 3, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000,
        0x80000001, 0xfffffffe, 0xffffffff, 31, 32, 33]


def form_key(i):
    parts = [i.mnemonic]
    for op in i.operands:
        if op.type == X.X86_OP_REG:
            n = i.reg_name(op.reg)
            parts.append("x" if n.startswith("xmm") else "m" if n.startswith("mm") else
                         "st" if n.startswith("st") else f"r{op.size}")
        elif op.type == X.X86_OP_IMM:
            parts.append(f"i{op.size}")
        else:
            parts.append(f"M{op.size}" + ("fs" if op.mem.segment else "") + ("a" if not op.mem.base and not op.mem.index else ""))
    return " ".join(parts)


def flag_mask(i, state):
    m = i.mnemonic.split()[-1]
    ops = i.operands
    if m in ("add", "sub", "cmp", "adc", "sbb", "neg", "xadd", "cmpxchg", "and", "or", "xor", "test",
             "inc", "dec", "scasb", "scasd", "cmpsb", "cmpsd"):
        return ALL
    if m in ("shl", "sal", "shr", "sar", "shld", "shrd"):
        cnt_op = ops[-1] if (m in ("shld", "shrd") or len(ops) > 1) else None
        if cnt_op is None:
            n = 1
        elif cnt_op.type == X.X86_OP_IMM:
            n = cnt_op.imm & 31
        else:
            n = state["regs"][1] & 31
        if n == 0:
            return ALL
        return (CF | ZF | SF | PF | (OF if n == 1 else 0))
    if m in ("rol", "ror", "rcl", "rcr"):
        n = ops[1].imm & 31 if len(ops) > 1 and ops[1].type == X.X86_OP_IMM else (1 if len(ops) == 1 else state["regs"][1] & 31)
        return ALL if n == 0 else (CF | ZF | SF | PF | (OF if n == 1 else 0))
    if m in ("mul", "imul"):
        return CF | OF
    if m in ("div", "idiv"):
        return 0
    if m in ("bt", "bts", "btr", "btc"):
        return CF
    if m in ("bsf", "bsr"):
        return ZF
    return ALL      # everything else leaves the flags alone (or defines all of them)


def fp_bytes_to_float(b):
    mant, se = struct.unpack("<QH", b)
    exp = se & 0x7fff
    sign = -1.0 if se & 0x8000 else 1.0
    if exp == 0 and mant == 0:
        return sign * 0.0
    if exp == 0x7fff:
        return float("nan") if (mant << 1) & 0xffffffffffffffff else sign * float("inf")
    import math
    return sign * math.ldexp(mant, exp - 16383 - 63)


class Test:
    def __init__(self, n, insn, state):
        self.n, self.insn, self.state = n, insn, state


def make_state(rng, i):
    regs = [rng.choice(POOL) if rng.random() < 0.5 else rng.getrandbits(32) for _ in range(8)]
    addr_regs = {4, 5}   # esp, ebp always point at the stack area
    for op in i.operands:
        if op.type == X.X86_OP_MEM:
            if op.mem.base:
                addr_regs.add(REGS.index(i.reg_name(op.mem.base)) if i.reg_name(op.mem.base) in REGS else 0)
            if op.mem.index:
                regs[REGS.index(i.reg_name(op.mem.index))] = rng.randrange(0, 64)
    m = i.mnemonic.split()[-1]
    if m.startswith(("movs", "stos", "lods", "scas", "cmps")):
        addr_regs |= {6, 7}
        regs[1] = rng.randrange(0, 40)   # ecx for rep
    for r in addr_regs:
        if r == 4:
            regs[r] = STACK + rng.randrange(0, 256) * 4
        elif r == 5:
            regs[r] = STACK + 0x8000 + rng.randrange(0, 256) * 4
        else:
            regs[r] = ADDR_BASE + rng.randrange(0, 0x4000) * 4
    # memory operands that use base and index registers: keep the index small when it is also a base
    flags = 0x2 | sum(f for f in (CF, PF, ZF, SF, OF) if rng.random() < 0.5)
    fpu = [struct.unpack("<f", struct.pack("<f", rng.uniform(-1000, 1000)))[0] for _ in range(3)]
    if rng.random() < 0.2:
        fpu[0] = float(rng.randrange(-100, 100))
    if rng.random() < 0.15:
        fpu[1] = fpu[0]
    mm = [rng.getrandbits(64) for _ in range(8)]
    xmm = []
    for _ in range(8):
        if rng.random() < 0.6:
            lanes = [struct.unpack("<I", struct.pack("<f", rng.uniform(-100, 100)))[0] for _ in range(4)]
            xmm.append(sum(v << (32 * k) for k, v in enumerate(lanes)))
        else:
            xmm.append(rng.getrandbits(128))
    return {"regs": regs, "flags": flags, "fpu": fpu, "mm": mm, "xmm": xmm}


def uses_mmx(i):
    return any(op.type == X.X86_OP_REG and i.reg_name(op.reg).startswith("mm") for op in i.operands)


# ---- reference ---------------------------------------------------------------------------
def u_prelude(mmx):
    b = bytearray()
    if mmx:
        for k in range(8):
            b += bytes([0x0f, 0x6f, (k << 3) | 5]) + struct.pack("<I", IO + 0x40 + 8 * k)
    else:
        b += b"\xdb\xe3" + b"\xd9\x2d" + struct.pack("<I", IO)                    # fninit; fldcw [IO]
        for k in (2, 1, 0):
            b += b"\xdd\x05" + struct.pack("<I", IO + 8 + 8 * k)                  # fld qword
    return bytes(b)


def u_postlude(mmx):
    b = bytearray()
    if mmx:
        for k in range(8):
            b += bytes([0x0f, 0x7f, (k << 3) | 5]) + struct.pack("<I", IO + 0x200 + 8 * k)
    else:
        b += b"\xdd\x35" + struct.pack("<I", IO + 0x100)                           # fnsave
    return bytes(b)


class Ref:
    def __init__(self, img, scratch):
        self.uc = U.Uc(U.UC_ARCH_X86, U.UC_MODE_32)
        uc = self.uc
        uc.mem_map(SCRATCH, SCRATCH_SIZE)
        uc.mem_map(IO, 0x1000)
        uc.mem_map(CODE, 0x1000)
        uc.mem_map(IMG_LO, IMG_HI - IMG_LO)
        self.image = bytes(img.mem[:IMG_HI - IMG_LO]).ljust(IMG_HI - IMG_LO, b"\0")
        uc.mem_write(IMG_LO, self.image)
        self.scratch = scratch

    def run(self, t):
        uc, s, i = self.uc, t.state, t.insn
        mmx = uses_mmx(i)
        uc.mem_write(SCRATCH, self.scratch)
        io = bytearray(0x1000)
        struct.pack_into("<H", io, 0, 0x27f)
        for k, v in enumerate(s["fpu"]):
            struct.pack_into("<d", io, 8 + 8 * k, v)
        for k, v in enumerate(s["mm"]):
            struct.pack_into("<Q", io, 0x40 + 8 * k, v)
        uc.mem_write(IO, bytes(io))
        absaddrs = abs_addrs(i)
        saved_abs = {a: bytes(uc.mem_read(a, 16)) for a in absaddrs}
        code = u_prelude(mmx) + bytes(i.bytes) + u_postlude(mmx)
        uc.mem_write(CODE, code)
        uc.ctl_remove_cache(CODE, CODE + 0x1000)    # code page is rewritten for every test
        for r, v in zip(UREG, s["regs"]):
            uc.reg_write(r, v)
        uc.reg_write(UX.UC_X86_REG_EFLAGS, s["flags"])
        for k in range(8):
            uc.reg_write(UX.UC_X86_REG_XMM0 + k, s["xmm"][k])
        try:
            uc.emu_start(CODE, CODE + len(code), count=64)
        except U.UcError as e:
            for a, b in saved_abs.items():
                uc.mem_write(a, b)
            return {"error": str(e)}
        out = {"regs": [uc.reg_read(r) for r in UREG], "flags": uc.reg_read(UX.UC_X86_REG_EFLAGS),
               "xmm": [uc.reg_read(UX.UC_X86_REG_XMM0 + k) for k in range(8)]}
        if uc.reg_read(UX.UC_X86_REG_EIP) != CODE + len(code):
            out["error"] = "did not reach the end"
        if mmx:
            out["mm"] = list(struct.unpack("<8Q", bytes(uc.mem_read(IO + 0x200, 64))))
        else:
            sv = bytes(uc.mem_read(IO + 0x100, 108))
            fsw = struct.unpack_from("<H", sv, 4)[0]
            ftw = struct.unpack_from("<H", sv, 8)[0]
            top = (fsw >> 11) & 7
            st = []
            for k in range(8):
                phys = (top + k) & 7
                if (ftw >> (2 * phys)) & 3 != 3:
                    st.append(fp_bytes_to_float(sv[28 + 10 * k:38 + 10 * k]))
                else:
                    st.append(None)
            out["fsw_cc"] = fsw & 0x4700
            out["ftop"] = top
            out["st"] = st
        mem = bytes(uc.mem_read(SCRATCH, SCRATCH_SIZE))
        out["memdiff"] = memdiff(self.scratch, mem)
        out["abs"] = {a: bytes(uc.mem_read(a, 16)) for a in absaddrs}
        for a, b in saved_abs.items():
            uc.mem_write(a, b)
        return out


def abs_addrs(i):
    out = []
    for op in i.operands:
        if op.type == X.X86_OP_MEM and not op.mem.base and IMG_LO <= (op.mem.disp & 0xffffffff) < IMG_HI - 16:
            out.append(op.mem.disp & 0xffffffff)   # abs or abs+small index: the index is < 64
    return out


def memdiff(a, b):
    if a == b:
        return []
    d = []
    for k in range(len(a)):
        if a[k] != b[k]:
            d.append((SCRATCH + k, b[k]))
            if len(d) >= 64:
                break
    return d


# ---- translated side ----------------------------------------------------------------------
class FakeCtx:
    noreturn = set()
    tables = {}
    simd_used = set()

    def is_func(self, a):
        return False

    def import_name(self, slot):
        return None

    def labels_of(self, f):
        return set()


HARNESS_MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <signal.h>
#include "cpu.h"
#include "simd.h"
static sigjmp_buf jb; static const char *trapped; static char segvmsg[64];
_Noreturn void trap(struct cpu *c, u32 at, const char *what) { (void)c; (void)at; trapped = what; siglongjmp(jb, 1); }
void dispatch(struct cpu *c, u32 t) { (void)c; (void)t; trapped = "dispatch"; siglongjmp(jb, 1); }
static void on_segv(int sig, siginfo_t *si, void *u) { (void)sig; (void)u;
    snprintf(segvmsg, sizeof segvmsg, "segv at %p", si->si_addr); trapped = segvmsg; siglongjmp(jb, 1); }
typedef void (*tfn)(struct cpu *);
extern tfn tests[];
extern int ntests;
static void *map(u32 a, u32 n) {
    void *p = mmap((void *)(uintptr_t)a, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p != (void *)(uintptr_t)a) { perror("mmap"); exit(1); }
    return p;
}
int main(int argc, char **argv) {
    FILE *img = fopen(argv[1], "rb"), *scr = fopen(argv[2], "rb"), *in = fopen(argv[3], "rb");
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_segv; sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigaction(SIGSEGV, &sa, 0); sigaction(SIGBUS, &sa, 0); sigaction(SIGFPE, &sa, 0);
    u32 img_lo = strtoul(argv[4], 0, 0), img_hi = strtoul(argv[5], 0, 0);
    u8 *image = map(img_lo, img_hi - img_lo);
    fread(image, 1, img_hi - img_lo, img);
    u8 *scratch = map(0x20000000, 0x100000);
    static u8 pristine[0x100000];
    fread(pristine, 1, sizeof pristine, scr);
    for (int t = 0; t < ntests; t++) {
        u32 hdr[3]; /* test index, nabs, mmx */
        if (fread(hdr, 4, 3, in) != 3) break;
        u32 regs[8], flags; double fpu[3]; u64 mm[8]; xmm_t xmm[8]; u32 absa[4];
        fread(regs, 4, 8, in); fread(&flags, 4, 1, in); fread(fpu, 8, 3, in); fread(mm, 8, 8, in);
        fread(xmm, 16, 8, in); fread(absa, 4, 4, in);
        memcpy(scratch, pristine, sizeof pristine);
        u8 absave[4][16];
        for (u32 k = 0; k < hdr[1]; k++) memcpy(absave[k], HOST(absa[k]), 16);
        struct cpu c; memset(&c, 0, sizeof c);
        memcpy(&c.eax, regs, 32); c.fr = flags; c.df = 0;
        fldcw(&c, 0x27f);
        c.ftop = 5; c.st[5] = fpu[0]; c.st[6] = fpu[1]; c.st[7] = fpu[2];
        memcpy(c.mm, mm, 64); memcpy(c.xmm, xmm, 128);
        trapped = 0;
        if (!sigsetjmp(jb, 1)) tests[hdr[0]](&c);
        printf("T %u\n", hdr[0]);
        if (trapped) { printf("trap %s\n", trapped); }
        printf("regs"); for (int k = 0; k < 8; k++) printf(" %u", (&c.eax)[k]); printf("\n");
        printf("flags %u\n", c.fr);
        printf("ftop %u cc %u\n", c.ftop, c.fsw_cc);
        printf("st"); for (int k = 0; k < 8; k++) printf(" %.17g", c.st[(c.ftop + k) & 7]); printf("\n");
        printf("mm"); for (int k = 0; k < 8; k++) printf(" %llu", (unsigned long long)c.mm[k].q); printf("\n");
        printf("xmm"); for (int k = 0; k < 8; k++) printf(" %016llx%016llx", (unsigned long long)c.xmm[k].q[1], (unsigned long long)c.xmm[k].q[0]); printf("\n");
        printf("mem"); int nd = 0;
        for (u32 k = 0; k < sizeof pristine && nd < 64; k++) if (scratch[k] != pristine[k]) { printf(" %u:%u", 0x20000000 + k, scratch[k]); nd++; }
        printf("\n");
        printf("abs"); for (u32 k = 0; k < hdr[1]; k++) { printf(" "); for (int b = 0; b < 16; b++) printf("%02x", ((u8 *)HOST(absa[k]))[b]); memcpy(HOST(absa[k]), absave[k], 16); }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
"""


def c_test(n, insn):
    tr = FuncTranslator(FakeCtx(), 0)
    body = tr.insn(insn, None)
    return (f"static void t_{n}(struct cpu *c) {{\n"
            "  u32 eax=c->eax,ecx=c->ecx,edx=c->edx,ebx=c->ebx,esp=c->esp,ebp=c->ebp,esi=c->esi,edi=c->edi;\n"
            "  u32 c_fop,c_fa,c_fb,c_fr,c_fc,c_fo; flags_set_l(c->fr,&c_fop,&c_fa,&c_fb,&c_fr,&c_fc,&c_fo,&c->df);\n"
            f"  {body}\n"
            "  c->eax=eax;c->ecx=ecx;c->edx=edx;c->ebx=ebx;c->esp=esp;c->ebp=ebp;c->esi=esi;c->edi=edi;\n"
            "  c->fr=flags_get_l(c_fop,c_fa,c_fb,c_fr,c_fc,c_fo,c->df);\n}\n"), tr.unsupported


def main():
    global IMG_LO, IMG_HI
    img = Image(sys.argv[2], sys.argv[1])
    fj = json.load(open(sys.argv[3]))
    work = sys.argv[4]
    IMG_LO, IMG_HI = img.base, (img.end + 0xfff) & ~0xfff
    if IMG_LO < IO + 0x1000 and IMG_HI > SCRATCH:
        raise SystemExit("image overlaps the fuzz scratch/code area (0x20000000-0x31000fff)")
    per_form = int(sys.argv[sys.argv.index("--per-form") + 1]) if "--per-form" in sys.argv else 3
    states = int(sys.argv[sys.argv.index("--states") + 1]) if "--states" in sys.argv else 4
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    os.makedirs(work, exist_ok=True)
    rng = random.Random(1234)
    disc = Discover(img)
    disc.starts = {int(k, 16) for k in fj}
    forms = collections.defaultdict(list)
    for f in sorted(disc.starts):
        for a in sorted(disc.walk(f)["insns"]):
            i = disc.d.at(a)
            m = i.mnemonic.split()[-1]
            if m in SKIP or i.id in (X.X86_INS_JMP,) or m.startswith("j") or (only and m not in only):
                continue
            if any(op.type == X.X86_OP_MEM and op.mem.segment for op in i.operands):
                continue
            if any(op.type == X.X86_OP_MEM and any(r and i.reg_name(r) not in REGS for r in (op.mem.base, op.mem.index))
                   for op in i.operands):
                continue   # 16-bit addressing: decoded data, not real code
            k = form_key(i)
            if len(forms[k]) < per_form * 4:
                forms[k].append(i)
    tests, src, skipped = [], [], collections.Counter()
    for k, lst in sorted(forms.items()):
        for i in rng.sample(lst, min(per_form, len(lst))):
            code, unsup = c_test(len(tests), i)
            if unsup:
                skipped[k] += 1
                continue
            for _ in range(states):
                n = len(tests)
                code, _ = c_test(n, i)
                src.append(code)
                tests.append(Test(n, i, make_state(rng, i)))
    print(f"{len(forms)} forms, {len(tests)} tests, {sum(skipped.values())} instances not translatable")
    with open(os.path.join(work, "tests.c"), "w") as fh:
        fh.write('#include "cpu.h"\n#include "simd.h"\n')
        fh.write("\n".join(src))
        fh.write("typedef void (*tfn)(struct cpu *);\ntfn tests[] = {" + ",".join(f"t_{t.n}" for t in tests) + "};\n")
        fh.write(f"int ntests = {len(tests)};\n")
    with open(os.path.join(work, "main.c"), "w") as fh:
        fh.write(HARNESS_MAIN)
    rt = os.path.join(ROOT, "runtime")
    exe = os.path.join(work, "harness")
    subprocess.run(["clang", "-O1", "-std=gnu11", "-w", "-I", rt, os.path.join(work, "main.c"), os.path.join(work, "tests.c"),
                    os.path.join(rt, "cpu.c"), "-lm", "-o", exe], check=True)
    scratch = bytes(rng.getrandbits(8) for _ in range(SCRATCH_SIZE))
    open(os.path.join(work, "scratch.bin"), "wb").write(scratch)
    open(os.path.join(work, "image.bin"), "wb").write(bytes(img.mem[:IMG_HI - IMG_LO]).ljust(IMG_HI - IMG_LO, b"\0"))
    with open(os.path.join(work, "input.bin"), "wb") as fh:
        for t in tests:
            s = t.state
            ab = abs_addrs(t.insn)[:4]
            fh.write(struct.pack("<3I", t.n, len(ab), uses_mmx(t.insn)))
            fh.write(struct.pack("<8I", *s["regs"]) + struct.pack("<I", s["flags"]) + struct.pack("<3d", *s["fpu"]))
            fh.write(struct.pack("<8Q", *s["mm"]))
            for v in s["xmm"]:
                fh.write(v.to_bytes(16, "little"))
            fh.write(struct.pack("<4I", *(ab + [0] * (4 - len(ab)))))
    res = subprocess.run([exe, os.path.join(work, "image.bin"), os.path.join(work, "scratch.bin"),
                          os.path.join(work, "input.bin"), hex(IMG_LO), hex(IMG_HI)], capture_output=True, text=True)
    mine = parse_harness(res.stdout)
    ref = Ref(img, scratch)
    fails = collections.defaultdict(list)
    ref_errors, ref_example = collections.Counter(), {}
    passed = noref = 0
    for t in tests:
        r = ref.run(t)
        m = mine.get(t.n)
        key = form_key(t.insn)
        if "error" in r:
            noref += 1
            ref_errors[key] += 1
            ref_example.setdefault(key, (t, r["error"], m.get("trap") if m else None))
            continue
        if m is None:
            fails[key].append((t, "harness produced nothing"))
            continue
        why = compare(t, r, m)
        if why:
            fails[key].append((t, why))
        else:
            passed += 1
    rep = os.path.join(work, "fuzz_report.txt")
    with open(rep, "w") as fh:
        fh.write(f"passed {passed}, failed {sum(len(v) for v in fails.values())} in {len(fails)} forms, "
                 f"reference faulted {noref}\n\n")
        for k, lst in sorted(fails.items(), key=lambda x: -len(x[1])):
            t, why = lst[0]
            fh.write(f"{len(lst):4d}  {k:40s}  {t.insn.address:#x}: {t.insn.mnemonic} {t.insn.op_str}\n      {why}\n")
        fh.write("\nreference faults by form:\n")
        for k, n in ref_errors.most_common(40):
            t, err, mt = ref_example[k]
            fh.write(f"{n:4d}  {k:40s}  {t.insn.mnemonic} {t.insn.op_str}  regs {[hex(x) for x in t.state['regs']]}  {err}  ours: {mt}\n")
    print(open(rep).read()[:4000])


def parse_harness(text):
    out, cur = {}, None
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "T":
            cur = {"n": int(w[1])}
            out[cur["n"]] = cur
        elif w[0] == "trap":
            cur["trap"] = line[5:]
        elif w[0] == "regs":
            cur["regs"] = [int(x) for x in w[1:]]
        elif w[0] == "flags":
            cur["flags"] = int(w[1])
        elif w[0] == "ftop":
            cur["ftop"], cur["cc"] = int(w[1]), int(w[3])
        elif w[0] == "st":
            cur["st"] = [float(x) for x in w[1:]]
        elif w[0] == "mm":
            cur["mm"] = [int(x) for x in w[1:]]
        elif w[0] == "xmm":
            cur["xmm"] = [int(x, 16) for x in w[1:]]
        elif w[0] == "mem":
            cur["mem"] = [(int(a), int(b)) for a, b in (x.split(":") for x in w[1:])]
        elif w[0] == "abs":
            cur["abs"] = [bytes.fromhex(x) for x in w[1:]]
    return out


def close(a, b):
    if a != a and b != b:
        return True
    if a == b:
        return True
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b))


def compare(t, r, m):
    i = t.insn
    if "trap" in m:
        return "trap: " + m["trap"]
    for k in range(8):
        if r["regs"][k] != m["regs"][k]:
            return f"{REGS[k]}: ref {r['regs'][k]:#x} ours {m['regs'][k]:#x} (in {t.state['regs'][k]:#x})"
    fm = flag_mask(i, t.state)
    if (r["flags"] ^ m["flags"]) & fm:
        return f"flags: ref {r['flags'] & fm:#x} ours {m['flags'] & fm:#x} mask {fm:#x} (in {t.state['flags']:#x})"
    if r["memdiff"] != m["mem"]:
        return f"memory: ref {r['memdiff'][:4]} ours {m['mem'][:4]}"
    if [bytes(b) for b in r["abs"].values()] != m["abs"]:
        return "absolute memory differs"
    if "st" in r:
        # ftop: unicorn after fnsave reinit -> compare the saved top
        if r["ftop"] != m["ftop"]:
            return f"x87 top: ref {r['ftop']} ours {m['ftop']}"
        mn = i.mnemonic
        if mn.startswith("f") and ("com" in mn or mn == "ftst") and r["fsw_cc"] & 0x4500 != m["cc"] & 0x4500:
            return f"x87 cc: ref {r['fsw_cc']:#x} ours {m['cc']:#x}"
        for k, v in enumerate(r["st"]):
            if v is not None and not close(v, m["st"][k]):
                return f"ST({k}): ref {v!r} ours {m['st'][k]!r}"
    else:
        if r["mm"] != m["mm"]:
            k = next(k for k in range(8) if r["mm"][k] != m["mm"][k])
            return f"mm{k}: ref {r['mm'][k]:#018x} ours {m['mm'][k]:#018x}"
    for k in range(8):
        if r["xmm"][k] != m["xmm"][k]:
            return f"xmm{k}: ref {r['xmm'][k]:#034x} ours {m['xmm'][k]:#034x}"
    return None


if __name__ == "__main__":
    main()
