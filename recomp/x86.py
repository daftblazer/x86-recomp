"""x86-32 instruction -> C translation for one function at a time.

Registers live in C locals (eax..edi) inside a generated function and are synced with
`struct cpu` around calls (`SYNC`/`LOAD`). Flags are the lazy c_fop/c_fa/c_fb/c_fr/c_fc/c_fo
locals of cpu.h; x87, MMX and SSE state stay in `struct cpu`.
"""
import capstone
from capstone import x86 as X

REG32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
REG16 = {"ax": "eax", "cx": "ecx", "dx": "edx", "bx": "ebx", "sp": "esp", "bp": "ebp", "si": "esi", "di": "edi"}
REG8L = {"al": "eax", "cl": "ecx", "dl": "edx", "bl": "ebx"}
REG8H = {"ah": "eax", "ch": "ecx", "dh": "edx", "bh": "ebx"}

# Flags travel with the registers: some callees read the caller's flags (MSVC's _RTC_CheckEsp
# does `cmp esi, esp; call _RTC_CheckEsp` and branches on them inside).
SYNC = ("c->eax=eax;c->ecx=ecx;c->edx=edx;c->ebx=ebx;c->esp=esp;c->ebp=ebp;c->esi=esi;c->edi=edi;"
        "c->fop=c_fop;c->fa=c_fa;c->fb=c_fb;c->fr=c_fr;c->fc=c_fc;c->fo=c_fo;")
LOAD = ("eax=c->eax;ecx=c->ecx;edx=c->edx;ebx=c->ebx;esp=c->esp;ebp=c->ebp;esi=c->esi;edi=c->edi;"
        "c_fop=c->fop;c_fa=c->fa;c_fb=c->fb;c_fr=c->fr;c_fc=c->fc;c_fo=c->fo;")

JCC = {
    X.X86_INS_JO: "C_O", X.X86_INS_JNO: "C_NO", X.X86_INS_JB: "C_B", X.X86_INS_JAE: "C_AE",
    X.X86_INS_JE: "C_E", X.X86_INS_JNE: "C_NE", X.X86_INS_JBE: "C_BE", X.X86_INS_JA: "C_A",
    X.X86_INS_JS: "C_S", X.X86_INS_JNS: "C_NS", X.X86_INS_JP: "C_P", X.X86_INS_JNP: "C_NP",
    X.X86_INS_JL: "C_L", X.X86_INS_JGE: "C_GE", X.X86_INS_JLE: "C_LE", X.X86_INS_JG: "C_G",
}
SETCC = {
    "seto": "C_O", "setno": "C_NO", "setb": "C_B", "setae": "C_AE", "sete": "C_E", "setne": "C_NE",
    "setbe": "C_BE", "seta": "C_A", "sets": "C_S", "setns": "C_NS", "setp": "C_P", "setnp": "C_NP",
    "setl": "C_L", "setge": "C_GE", "setle": "C_LE", "setg": "C_G",
}
CMOV = {"cmov" + k[3:]: v for k, v in SETCC.items()}
MASK = {1: "0xffu", 2: "0xffffu", 4: "0xffffffffu"}


class Unsupported(Exception):
    pass


def fn_name(a):
    return f"f_{a:08x}"


class FuncTranslator:
    """Translates the instructions of one function. ctx supplies: is_func(a), import_name(slot),
    tables{at: [targets]}, labels(set), noreturn(set)."""

    def __init__(self, ctx, start):
        self.ctx = ctx
        self.start = start
        self.out = []
        self.calls = set()
        self.imports = set()
        self.unsupported = []

    # ---- operands -------------------------------------------------------------------------
    def reg(self, r, insn):
        n = insn.reg_name(r)
        if n in REG32:
            return n, 4, 0
        if n in REG16:
            return REG16[n], 2, 0
        if n in REG8L:
            return REG8L[n], 1, 0
        if n in REG8H:
            return REG8H[n], 1, 8
        raise Unsupported(f"register {n}")

    def addr(self, op, insn):
        m = op.mem
        parts = []
        if m.segment != 0:
            seg = insn.reg_name(m.segment)
            if seg == "fs":
                parts.append("c->fs_base")
            elif seg not in ("ds", "ss", "es", "cs"):
                raise Unsupported(f"segment {seg}")
        if m.base != 0:
            parts.append(self.reg(m.base, insn)[0])
        if m.index != 0:
            ix = self.reg(m.index, insn)[0]
            parts.append(ix if m.scale == 1 else f"{ix}*{m.scale}u")
        if m.disp or not parts:
            parts.append(f"0x{m.disp & 0xffffffff:x}u")
        return "(u32)(" + "+".join(parts) + ")"

    def read(self, op, insn, size=None):
        size = size or op.size
        if op.type == X.X86_OP_REG:
            r, rs, sh = self.reg(op.reg, insn)
            if rs == 4:
                return r
            return f"(({r}>>{sh})&{MASK[rs]})" if sh else f"({r}&{MASK[rs]})"
        if op.type == X.X86_OP_IMM:
            return f"0x{op.imm & int(MASK[size][:-1], 16):x}u"
        if op.type == X.X86_OP_MEM:
            return {1: "ld8", 2: "ld16", 4: "ld32"}[size] + "(" + self.addr(op, insn) + ")"
        raise Unsupported("operand")

    def write(self, op, insn, val, size=None):
        size = size or op.size
        if op.type == X.X86_OP_REG:
            r, rs, sh = self.reg(op.reg, insn)
            if rs == 4:
                return f"{r}=(u32)({val});"
            if sh:
                return f"{r}=({r}&0xffff00ffu)|((((u32)({val}))&0xffu)<<8);"
            return f"{r}=({r}&~{MASK[rs]})|(((u32)({val}))&{MASK[rs]});"
        if op.type == X.X86_OP_MEM:
            return {1: "st8", 2: "st16", 4: "st32"}[size] + "(" + self.addr(op, insn) + f",(u32)({val}));"
        raise Unsupported("write operand")

    def sx(self, val, size):
        return {1: f"(u32)(s32)(s8)({val})", 2: f"(u32)(s32)(s16)({val})", 4: f"({val})"}[size]

    # ---- helpers --------------------------------------------------------------------------
    def emit(self, s):
        self.out.append(s)

    def flags(self, kind, size, a, b, r, cf=None, of=None):
        s = f"c_fop={kind}|{size};c_fa={a};c_fb={b};c_fr={r};"
        if cf is not None:
            s += f"c_fc={cf};"
        if of is not None:
            s += f"c_fo={of};"
        return s

    def goto(self, t):
        return f"goto L_{t:x};"

    def call_direct(self, t, ret_addr):
        if not self.ctx.is_func(t):
            return f"{SYNC}trap(c,0x{ret_addr:x}u,\"call to non-function 0x{t:x}\");"
        self.calls.add(t)
        return f"esp-=4;st32(esp,0x{ret_addr:x}u);{SYNC}{fn_name(t)}(c);{LOAD}"

    def tail_direct(self, t):
        self.calls.add(t)
        return f"{SYNC}{fn_name(t)}(c);return;"

    def call_import(self, slot, ret_addr):
        name = self.ctx.import_name(slot)
        self.imports.add(name)
        return f"IMPORT_HOOK(\"{name}\",0x{ret_addr:x}u);esp-=4;st32(esp,0x{ret_addr:x}u);{SYNC}imp_{name}(c);{LOAD}"

    def call_indirect(self, target, ret_addr):
        return f"{{u32 t_=(u32)({target});esp-=4;st32(esp,0x{ret_addr:x}u);{SYNC}dispatch(c,t_);{LOAD}}}"

    # ---- one instruction ------------------------------------------------------------------
    def insn(self, i, nxt_in_order):
        """returns C for one instruction. nxt_in_order: address emitted next (for fallthrough)."""
        m = i.mnemonic.split()[-1]          # "rep movsd", "lock xadd" -> prefix is in i.prefix
        ops = i.operands
        a = i.address
        nxt = a + i.size
        try:
            s = self._insn(i, m, ops, a, nxt)
        except Unsupported as e:
            self.unsupported.append((a, m, str(e)))
            s = f"{SYNC}trap(c,0x{a:x}u,\"{m} {i.op_str} ({e})\");"
        return s

    def _insn(self, i, m, ops, a, nxt):
        rd, wr, ctx = self.read, self.write, self.ctx
        o0 = ops[0] if ops else None
        o1 = ops[1] if len(ops) > 1 else None
        sz = o0.size if o0 is not None else 4

        if i.prefix[0] == 0xf0:   # lock
            if m == "xadd":
                return f"{{u32 v_={rd(o1, i)};u32 old_=__atomic_fetch_add((u{sz*8}*)HOST({self.addr(o0, i)}),(u{sz*8})v_,__ATOMIC_SEQ_CST);" \
                       f"{wr(o1, i, 'old_')}" + self.flags("F_ADD", sz, "old_", "v_", "old_+v_") + "}"
            if m == "cmpxchg":
                t = f"u{sz*8}"
                return (f"{{{t} exp_=({t})eax;{t} des_=({t}){rd(o1, i)};"
                        f"__atomic_compare_exchange_n(({t}*)HOST({self.addr(o0, i)}),&exp_,des_,0,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);"
                        f"u32 old_=(u32)exp_,a_=eax&{MASK[sz]};"
                        + self.flags("F_SUB", sz, "a_", "old_", "a_-old_")
                        + f"eax=(eax&~{MASK[sz]})|old_;}}")
            raise Unsupported("lock " + m)

        # -------- moves --------
        if m == "mov":
            return wr(o0, i, rd(o1, i))
        if m == "movzx":
            return wr(o0, i, rd(o1, i))
        if m == "movsx":
            return wr(o0, i, self.sx(rd(o1, i), o1.size))
        if m == "lea":
            return wr(o0, i, self.addr(o1, i))
        if m == "xchg":
            if o0.type == X.X86_OP_REG and o1.type == X.X86_OP_REG and o0.reg == o1.reg:
                return ";"
            # xchg with memory is implicitly locked on x86 (spinlocks are built on it)
            mem, reg = (o0, o1) if o0.type == X.X86_OP_MEM else (o1, o0) if o1.type == X.X86_OP_MEM else (None, None)
            if mem is not None:
                t = f"u{sz*8}"
                return (f"{{{t} n_=({t}){rd(reg, i)};{t} o_=__atomic_exchange_n(({t}*)HOST({self.addr(mem, i)}),n_,__ATOMIC_SEQ_CST);"
                        f"{wr(reg, i, 'o_')}}}")
            return f"{{u32 t_={rd(o0, i)};{wr(o0, i, rd(o1, i))}{wr(o1, i, 't_')}}}"
        if m == "push":
            v = rd(o0, i, 4) if o0.type != X.X86_OP_IMM else f"0x{o0.imm & 0xffffffff:x}u"
            if o0.size == 2:
                return f"{{u32 v_={rd(o0, i)};esp-=2;st16(esp,v_);}}"
            return f"{{u32 v_={v};esp-=4;st32(esp,v_);}}"
        if m == "pop":
            if o0.type == X.X86_OP_REG and i.reg_name(o0.reg) == "esp":
                return "esp=ld32(esp);"
            if o0.size == 2:
                return f"{{u32 v_=ld16(esp);esp+=2;{wr(o0, i, 'v_')}}}"
            return f"{{u32 v_=ld32(esp);esp+=4;{wr(o0, i, 'v_')}}}"
        if m == "pushal":
            return "{u32 s_=esp;esp-=32;st32(esp+28,eax);st32(esp+24,ecx);st32(esp+20,edx);st32(esp+16,ebx);" \
                   "st32(esp+12,s_);st32(esp+8,ebp);st32(esp+4,esi);st32(esp,edi);}"
        if m == "popal":
            return "edi=ld32(esp);esi=ld32(esp+4);ebp=ld32(esp+8);ebx=ld32(esp+16);edx=ld32(esp+20);" \
                   "ecx=ld32(esp+24);eax=ld32(esp+28);esp+=32;"
        if m == "pushfd":
            return f"{{u32 f_=flags_get_l(FL,c_fo,c->df);esp-=4;st32(esp,f_);}}"
        if m == "popfd":
            return "{u32 f_=ld32(esp);esp+=4;flags_set_l(f_,&c_fop,&c_fa,&c_fb,&c_fr,&c_fc,&c_fo,&c->df);}"
        if m == "lahf":
            return "{u32 f_=flags_get_l(FL,c_fo,c->df);eax=(eax&0xffff00ffu)|((f_&0xffu)<<8);}"
        if m == "sahf":
            return "{u32 f_=(flags_get_l(FL,c_fo,c->df)&~0xffu)|((eax>>8)&0xd5u);flags_set_l(f_,&c_fop,&c_fa,&c_fb,&c_fr,&c_fc,&c_fo,&c->df);}"
        if m == "cdq":
            return "edx=(u32)((s32)eax>>31);"
        if m == "cwde":
            return "eax=(u32)(s32)(s16)eax;"
        if m == "cbw":
            return "eax=(eax&0xffff0000u)|((u32)(s16)(s8)eax&0xffffu);"
        if m == "leave":
            return "esp=ebp;ebp=ld32(esp);esp+=4;"
        if m == "enter":
            if o1.imm != 0:
                raise Unsupported("enter with nesting")
            return f"esp-=4;st32(esp,ebp);ebp=esp;esp-=0x{o0.imm & 0xffff:x}u;"
        if m in ("nop", "wait", "fnop", "prefetcht0", "prefetcht1", "prefetcht2", "prefetchnta", "emms"):
            return ";"
        if m == "cld":
            return "c->df=0;"
        if m == "std":
            return "c->df=1;"
        if m == "clc":
            return "{u32 z_=ZF,s_=SF,o_=OF,p_=PF;" + self._set_flags_from("0", "z_", "s_", "o_", "p_") + "}"
        if m == "stc":
            return "{u32 z_=ZF,s_=SF,o_=OF,p_=PF;" + self._set_flags_from("1", "z_", "s_", "o_", "p_") + "}"
        if m == "cmc":
            return "{u32 cf_=!CF,z_=ZF,s_=SF,o_=OF,p_=PF;" + self._set_flags_from("cf_", "z_", "s_", "o_", "p_") + "}"

        # -------- arithmetic --------
        if m in ("add", "sub", "cmp", "and", "or", "xor", "test", "adc", "sbb"):
            A, B = rd(o0, i), rd(o1, i)
            if o1.type == X.X86_OP_IMM:
                B = f"0x{o1.imm & int(MASK[sz][:-1], 16):x}u"
            if m in ("xor", "sub") and o0.type == X.X86_OP_REG and o1.type == X.X86_OP_REG and o0.reg == o1.reg:
                return wr(o0, i, "0") + self.flags("F_LOGIC", sz, "0", "0", "0")
            body = {"add": "a_+b_", "sub": "a_-b_", "cmp": "a_-b_", "and": "a_&b_", "or": "a_|b_",
                    "xor": "a_^b_", "test": "a_&b_"}
            if m in ("adc", "sbb"):
                if m == "adc":
                    s = (f"{{u32 a_={A},b_={B},ci_=CF;u64 w_=(u64)a_+b_+ci_;u32 r_=(u32)w_;"
                         f"u32 cf_=(u32)((w_>>{sz*8})&1);u32 of_=((a_^r_)&(b_^r_)&{hex(1 << (sz*8-1))}u)!=0;")
                else:
                    s = (f"{{u32 a_={A},b_={B},ci_=CF;u64 w_=(u64)(a_&{MASK[sz]})-(b_&{MASK[sz]})-ci_;u32 r_=(u32)w_;"
                         f"u32 cf_=(u32)((w_>>63)&1);u32 of_=((a_^b_)&(a_^r_)&{hex(1 << (sz*8-1))}u)!=0;")
                return s + wr(o0, i, "r_") + self.flags("F_EAGER", sz, "a_", "b_", "r_", "cf_", "of_") + "}"
            kind = {"add": "F_ADD", "sub": "F_SUB", "cmp": "F_SUB"}.get(m, "F_LOGIC")
            s = f"{{u32 a_={A},b_={B},r_={body[m]};"
            if m not in ("cmp", "test"):
                s += wr(o0, i, "r_")
            return s + self.flags(kind, sz, "a_", "b_", "r_") + "}"
        if m in ("inc", "dec"):
            op = "+" if m == "inc" else "-"
            return (f"{{u32 cf_=CF;u32 a_={rd(o0, i)},r_=a_{op}1u;{wr(o0, i, 'r_')}"
                    + self.flags("F_INC" if m == "inc" else "F_DEC", sz, "a_", "1u", "r_", "cf_") + "}")
        if m == "neg":
            return f"{{u32 a_={rd(o0, i)},r_=0u-a_;{wr(o0, i, 'r_')}" + self.flags("F_NEG", sz, "a_", "0u", "r_") + "}"
        if m == "not":
            return wr(o0, i, f"~{rd(o0, i)}")
        if m in ("shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr"):
            return self.shift(i, m, o0, o1, sz)
        if m in ("shld", "shrd"):
            cnt = rd(ops[2], i, 1) + "&31u" if ops[2].type == X.X86_OP_IMM else "(ecx&31u)"
            if ops[2].type == X.X86_OP_IMM:
                cnt = f"{ops[2].imm & 31}u"
            if sz != 4:
                raise Unsupported("16-bit shld/shrd")
            if m == "shld":
                e = "(n_?((d_<<n_)|(s_>>(32u-n_))):d_)"
                cf = "(n_?((d_>>(32u-n_))&1u):CF)"
            else:
                e = "(n_?((d_>>n_)|(s_<<(32u-n_))):d_)"
                cf = "(n_?((d_>>(n_-1u))&1u):CF)"
            return (f"{{u32 n_={cnt},d_={rd(o0, i)},s_={rd(o1, i)};u32 cf_={cf};u32 r_={e};{wr(o0, i, 'r_')}"
                    f"if(n_){{{self.flags('F_EAGER', 4, 'd_', 's_', 'r_', 'cf_', '((r_^d_)>>31)')}}}}}")
        if m == "imul":
            if len(ops) == 1:
                if sz == 4:
                    return ("{s64 p_=(s64)(s32)eax*(s32)" + rd(o0, i) + ";eax=(u32)p_;edx=(u32)((u64)p_>>32);"
                            "u32 o_=(p_!=(s64)(s32)p_);" + self.flags("F_EAGER", 4, "0", "0", "eax", "o_", "o_") + "}")
                if sz == 1:
                    return ("{s32 p_=(s32)(s8)eax*(s32)(s8)" + rd(o0, i) + ";eax=(eax&0xffff0000u)|((u32)p_&0xffffu);"
                            "u32 o_=(p_!=(s32)(s8)p_);" + self.flags("F_EAGER", 1, "0", "0", "(u32)p_", "o_", "o_") + "}")
                if sz == 2:
                    return ("{s32 p_=(s32)(s16)eax*(s32)(s16)" + rd(o0, i) + ";eax=(eax&0xffff0000u)|((u32)p_&0xffffu);"
                            "edx=(edx&0xffff0000u)|(((u32)p_>>16)&0xffffu);u32 o_=(p_!=(s32)(s16)p_);"
                            + self.flags("F_EAGER", 2, "0", "0", "(u32)p_", "o_", "o_") + "}")
            src1 = rd(ops[1], i) if len(ops) == 3 else rd(o0, i)
            src2 = (f"0x{ops[2].imm & 0xffffffff:x}u" if len(ops) == 3 else rd(o1, i))
            if sz == 2:
                return (f"{{s32 p_=(s32)(s16){src1}*(s32)(s16)({src2});u32 r_=(u32)p_;{wr(o0, i, 'r_')}"
                        "u32 o_=(p_!=(s32)(s16)p_);" + self.flags("F_EAGER", 2, "0", "0", "r_", "o_", "o_") + "}")
            return (f"{{s64 p_=(s64)(s32){src1}*(s64)(s32)({src2});u32 r_=(u32)p_;{wr(o0, i, 'r_')}"
                    "u32 o_=(p_!=(s64)(s32)p_);" + self.flags("F_EAGER", 4, "0", "0", "r_", "o_", "o_") + "}")
        if m == "mul":
            if sz == 4:
                return ("{u64 p_=(u64)eax*" + rd(o0, i) + ";eax=(u32)p_;edx=(u32)(p_>>32);u32 o_=edx!=0;"
                        + self.flags("F_EAGER", 4, "0", "0", "eax", "o_", "o_") + "}")
            if sz == 1:
                return ("{u32 p_=(eax&0xffu)*" + rd(o0, i) + ";eax=(eax&0xffff0000u)|p_;u32 o_=(p_>>8)!=0;"
                        + self.flags("F_EAGER", 1, "0", "0", "p_", "o_", "o_") + "}")
            raise Unsupported("mul16")
        if m in ("div", "idiv"):
            if sz == 4:
                if m == "div":
                    return ("{u64 n_=((u64)edx<<32)|eax;u32 d_=" + rd(o0, i) + ";if(!d_)" + f"{{{SYNC}trap(c,0x{a:x}u,\"divide by zero\");}}"
                            "eax=(u32)(n_/d_);edx=(u32)(n_%d_);}")
                return ("{s64 n_=(s64)(((u64)edx<<32)|eax);s32 d_=(s32)" + rd(o0, i) + ";if(!d_)" + f"{{{SYNC}trap(c,0x{a:x}u,\"divide by zero\");}}"
                        "eax=(u32)(s32)(n_/d_);edx=(u32)(s32)(n_%d_);}")
            if sz == 1:
                if m == "div":
                    return ("{u32 n_=eax&0xffffu,d_=" + rd(o0, i) + ";if(!d_)" + f"{{{SYNC}trap(c,0x{a:x}u,\"divide by zero\");}}"
                            "eax=(eax&0xffff0000u)|((n_%d_)<<8)|(n_/d_&0xffu);}")
                return ("{s32 n_=(s16)eax,d_=(s8)" + rd(o0, i) + ";if(!d_)" + f"{{{SYNC}trap(c,0x{a:x}u,\"divide by zero\");}}"
                        "eax=(eax&0xffff0000u)|(((u32)(n_%d_)&0xffu)<<8)|((u32)(n_/d_)&0xffu);}")
            raise Unsupported("div16")

        # -------- conditions --------
        if m in SETCC:
            return wr(o0, i, f"({SETCC[m]})?1u:0u")
        if m in CMOV:
            return f"if({CMOV[m]}){{{wr(o0, i, rd(o1, i))}}}"

        # -------- bit ops --------
        if m in ("bt", "bts", "btr", "btc"):
            if o0.type == X.X86_OP_MEM and o1.type == X.X86_OP_REG:
                raise Unsupported("bt mem,reg")
            n = f"({rd(o1, i)}&{sz*8-1}u)"
            s = f"{{u32 v_={rd(o0, i)},n_={n};u32 cf_=(v_>>n_)&1u;"
            if m == "bts":
                s += wr(o0, i, "v_|(1u<<n_)")
            elif m == "btr":
                s += wr(o0, i, "v_&~(1u<<n_)")
            elif m == "btc":
                s += wr(o0, i, "v_^(1u<<n_)")
            # only CF is defined; keep ZF from a nonzero result by reusing r=1
            return s + self.flags("F_EAGER", sz, "0", "0", "1u", "cf_", "0") + "}"
        if m in ("bsf", "bsr"):
            f = "__builtin_ctz(v_)" if m == "bsf" else "(31u-__builtin_clz(v_))"
            return (f"{{u32 v_={rd(o1, i)};if(v_){{{wr(o0, i, f)}}}"
                    + self.flags("F_LOGIC", 4, "0", "0", "v_") + "}")

        # -------- strings --------
        has_xmm = any(op.type == X.X86_OP_REG and i.reg_name(op.reg).startswith(("xmm", "mm")) for op in ops)
        if m in ("movsd", "movsb", "movsw") and not has_xmm:
            n = {"movsd": 4, "movsb": 1, "movsw": 2}[m]
            if i.prefix[0] in (0xf3,):
                return f"{SYNC}rep_movs(c,{n});{LOAD}"
            ld, st = {1: ("ld8", "st8"), 2: ("ld16", "st16"), 4: ("ld32", "st32")}[n]
            return f"{st}(edi,{ld}(esi));if(c->df){{esi-={n};edi-={n};}}else{{esi+={n};edi+={n};}}"
        if m in ("stosd", "stosb", "stosw"):
            n = {"stosd": 4, "stosb": 1, "stosw": 2}[m]
            if i.prefix[0] == 0xf3:
                return f"{SYNC}rep_stos(c,{n});{LOAD}"
            st = {1: "st8", 2: "st16", 4: "st32"}[n]
            return f"{st}(edi,eax);if(c->df)edi-={n};else edi+={n};"
        if m in ("lodsb", "lodsd", "lodsw"):
            n = {"lodsd": 4, "lodsb": 1, "lodsw": 2}[m]
            if i.prefix[0] == 0xf3:
                raise Unsupported("rep lods")
            r = {1: "eax=(eax&~0xffu)|ld8(esi);", 2: "eax=(eax&~0xffffu)|ld16(esi);", 4: "eax=ld32(esi);"}[n]
            return r + f"if(c->df)esi-={n};else esi+={n};"
        if m in ("scasb", "scasd", "cmpsb", "cmpsd") and not has_xmm:
            if i.prefix[0] in (0xf2, 0xf3):
                return self.rep_cmp(i, m)
            raise Unsupported("single " + m)

        # -------- control flow --------
        if m == "call":
            if o0.type == X.X86_OP_IMM:
                t = o0.imm & 0xffffffff
                s = self.call_direct(t, nxt)
                if t in ctx.noreturn:
                    s += f"{SYNC}trap(c,0x{a:x}u,\"noreturn function returned\");"
                return s
            if o0.type == X.X86_OP_MEM and o0.mem.base == 0 and o0.mem.index == 0 and o0.mem.segment == 0:
                slot = o0.mem.disp & 0xffffffff
                if ctx.import_name(slot):
                    return self.call_import(slot, nxt)
            return self.call_indirect(rd(o0, i), nxt)
        if m == "ret":
            n = o0.imm if o0 is not None else 0
            return f"esp+={4 + n}u;{SYNC}return;"
        if m == "jmp":
            if o0.type == X.X86_OP_IMM:
                t = o0.imm & 0xffffffff
                if t in ctx.labels_of(self.start):
                    return self.goto(t)
                if ctx.is_func(t):
                    return self.tail_direct(t)
                return f"{SYNC}trap(c,0x{a:x}u,\"jmp to unknown 0x{t:x}\");"
            if a in ctx.tables:
                targets = ctx.tables[a]
                ix = self.reg(o0.mem.index, i)[0]
                cases = "".join(f"case {k}:goto L_{t:x};" for k, t in enumerate(targets))
                return f"switch({ix}){{{cases}default:{SYNC}trap(c,0x{a:x}u,\"jump table index\");}}"
            if o0.type == X.X86_OP_MEM and o0.mem.base == 0 and o0.mem.index == 0 and o0.mem.segment == 0:
                slot = o0.mem.disp & 0xffffffff
                if ctx.import_name(slot):
                    name = ctx.import_name(slot)
                    self.imports.add(name)
                    return f"IMPORT_HOOK(\"{name}\",ld32(esp));{SYNC}imp_{name}(c);return;"
            return f"{{u32 t_=(u32)({rd(o0, i)});{SYNC}dispatch(c,t_);return;}}"
        if i.id in JCC:
            t = o0.imm & 0xffffffff
            cond = JCC[i.id]
            if t in ctx.labels_of(self.start):
                return f"if({cond}){self.goto(t)}"
            if ctx.is_func(t):
                return f"if({cond}){{{self.tail_direct(t)}}}"
            return f"if({cond}){{{SYNC}trap(c,0x{a:x}u,\"jcc to unknown\");}}"
        if m in ("jecxz", "jcxz"):
            t = o0.imm & 0xffffffff
            return f"if(!ecx){self.goto(t)}"
        if m in ("loop", "loope", "loopne"):
            t = o0.imm & 0xffffffff
            extra = {"loop": "", "loope": "&&ZF", "loopne": "&&!ZF"}[m]
            return f"ecx--;if(ecx{extra}){self.goto(t)}"
        if m in ("int3", "hlt", "ud2"):
            return f"{SYNC}trap(c,0x{a:x}u,\"{m}\");"

        # -------- x87 / SIMD --------
        if m.startswith("f"):
            return self.x87(i, m, ops)
        return self.simd(i, m, ops)

    def _set_flags_from(self, cf, zf, sf, of, pf):
        """rebuild lazy flags from individual bits (rare: clc/stc/cmc)"""
        return f"flags_set_l(flags_from_bits({cf},{zf},{sf},{of},{pf}),&c_fop,&c_fa,&c_fb,&c_fr,&c_fc,&c_fo,&c->df);"

    def shift(self, i, m, o0, o1, sz):
        bits = sz * 8
        if o1 is None:
            cnt = "1u"
        elif o1.type == X.X86_OP_IMM:
            cnt = f"{o1.imm & 31}u"
        else:
            cnt = "(ecx&31u)"
        v = self.read(o0, i)
        msk = MASK[sz]
        top = f"{bits-1}u"
        if m in ("shl", "sal"):
            r, cf, of = f"(v_<<n_)", f"((v_>>({bits}u-n_))&1u)", f"(((r_>>{top})&1u)^cf_)"
        elif m == "shr":
            r, cf, of = f"((v_&{msk})>>n_)", "((v_>>(n_-1u))&1u)", f"((v_>>{top})&1u)"
        elif m == "sar":
            sx = self.sx("v_", sz)
            r, cf, of = f"(u32)((s32){sx}>>n_)", f"(u32)(((s32){sx}>>(n_-1u))&1)", "0u"
        elif m in ("rol", "ror"):
            n = f"(n_%{bits}u)"
            if m == "rol":
                r = f"(((v_&{msk})<<{n})|((v_&{msk})>>(({bits}u-{n})%{bits}u)))"
                cf, of = "(r_&1u)", f"(((r_>>{top})&1u)^(r_&1u))"
            else:
                r = f"(((v_&{msk})>>{n})|((v_&{msk})<<(({bits}u-{n})%{bits}u)))"
                cf, of = f"((r_>>{top})&1u)", f"(((r_>>{top})^(r_>>{bits-2}u))&1u)"
            return (f"{{u32 v_={v},n_={cnt};if(n_){{u32 r_={r};{self.write(o0, i, 'r_')}"
                    f"u32 z_=ZF,s_=SF,p_=PF;" + self._set_flags_from(cf, "z_", "s_", of, "p_") + "}}")
        else:  # rcl / rcr, count 1 only in practice
            if cnt != "1u":
                raise Unsupported(m + " count")
            keep = "u32 z_=ZF,s_=SF,p_=PF;"
            if m == "rcl":
                return (f"{{u32 v_={v},ci_=CF;u32 r_=((v_<<1)|ci_)&{msk};{self.write(o0, i, 'r_')}{keep}u32 cf_=(v_>>{top})&1u;"
                        + self._set_flags_from("cf_", "z_", "s_", f"((r_>>{top})&1u)^cf_", "p_") + "}")
            return (f"{{u32 v_={v},ci_=CF;u32 r_=((v_&{msk})>>1)|(ci_<<{top});{self.write(o0, i, 'r_')}{keep}u32 cf_=v_&1u;"
                    + self._set_flags_from("cf_", "z_", "s_", f"((r_>>{top})^(r_>>{bits-2}u))&1u", "p_") + "}")
        return (f"{{u32 v_={v},n_={cnt};if(n_){{u32 r_={r};{self.write(o0, i, 'r_')}u32 cf_={cf};"
                + self.flags("F_EAGER", sz, "v_", "n_", "r_", "cf_", of) + "}}")

    def rep_cmp(self, i, m):
        n = 1 if m.endswith("b") else 4
        ld = "ld8" if n == 1 else "ld32"
        stop = "!ZF" if i.prefix[0] == 0xf3 else "ZF"   # repe stops on mismatch, repne on match
        if m.startswith("scas"):
            body = f"u32 a_=eax&{MASK[n]},b_={ld}(edi);c_fop=F_SUB|{n};c_fa=a_;c_fb=b_;c_fr=a_-b_;edi+=c->df?-{n}:{n};"
        else:
            body = f"u32 a_={ld}(esi),b_={ld}(edi);c_fop=F_SUB|{n};c_fa=a_;c_fb=b_;c_fr=a_-b_;esi+=c->df?-{n}:{n};edi+=c->df?-{n}:{n};"
        return f"while(ecx){{{body}ecx--;if({stop})break;}}"

    # ---- x87 ------------------------------------------------------------------------------
    def fmem(self, op, i):
        a = self.addr(op, i)
        return {4: f"(double)ldf({a})", 8: f"ldd({a})", 10: f"ld80({a})"}[op.size]

    def fimem(self, op, i):
        a = self.addr(op, i)
        return {2: f"(double)(s16)ld16({a})", 4: f"(double)(s32)ld32({a})", 8: f"(double)(s64)ld64({a})"}[op.size]

    def sti(self, op, i):
        n = i.reg_name(op.reg)
        if not n.startswith("st"):
            raise Unsupported("x87 reg " + n)
        k = int(n[3]) if "(" in n else 0
        return k

    def x87(self, i, m, ops):
        o0 = ops[0] if ops else None
        o1 = ops[1] if len(ops) > 1 else None
        R = "fround(c,{})"
        arith = {"add": "{a}+{b}", "sub": "{a}-{b}", "subr": "{b}-{a}", "mul": "{a}*{b}", "div": "{a}/{b}", "divr": "{b}/{a}"}
        if m in ("fld",):
            if o0.type == X.X86_OP_MEM:
                return f"fpush(c,{self.fmem(o0, i)});"
            return f"{{double v_=ST({self.sti(o0, i)});fpush(c,v_);}}"
        if m == "fild":
            return f"fpush(c,{self.fimem(o0, i)});"
        if m in ("fldz", "fld1", "fldpi", "fldl2e", "fldln2", "fldlg2", "fldl2t"):
            v = {"fldz": "0.0", "fld1": "1.0", "fldpi": "M_PI", "fldl2e": "M_LOG2E", "fldln2": "M_LN2",
                 "fldlg2": "0.30102999566398119521", "fldl2t": "3.32192809488736234787"}[m]
            return f"fpush(c,{v});"
        if m in ("fst", "fstp"):
            pop = "fpop(c);" if m == "fstp" else ""
            if o0.type == X.X86_OP_MEM:
                a = self.addr(o0, i)
                st = {4: f"stf({a},(float)ST(0));", 8: f"std_({a},ST(0));", 10: f"st80({a},ST(0));"}[o0.size]
                return st + pop
            k = self.sti(o0, i)
            return f"ST({k})=ST(0);" + pop
        if m in ("fist", "fistp"):
            a = self.addr(o0, i)
            st = {2: "st16", 4: "st32", 8: "st64"}[o0.size]
            v = f"fist(c,ST(0),{o0.size*8})"
            s = f"{st}({a},(u{max(o0.size*8,32)}){v});" if o0.size != 8 else f"st64({a},(u64){v});"
            return s + ("fpop(c);" if m == "fistp" else "")
        if m == "fxch":
            k = self.sti(ops[-1], i) if ops else 1
            return f"{{double t_=ST(0);ST(0)=ST({k});ST({k})=t_;}}"
        if m == "fchs":
            return "ST(0)=-ST(0);"
        if m == "fabs":
            return "ST(0)=fabs(ST(0));"
        if m == "fsqrt":
            return "ST(0)=fround(c,sqrt(ST(0)));"
        if m == "frndint":
            return "ST(0)=frndint_(c,ST(0));"
        if m == "fprem1":
            return "ST(0)=remainder(ST(0),ST(1));c->fsw_cc&=~0x0400u;"
        if m == "fprem":
            return "ST(0)=fmod(ST(0),ST(1));c->fsw_cc&=~0x0400u;"
        for base, expr in arith.items():
            if m == "f" + base or m == "f" + base + "p":
                pop = m == "f" + base + "p"
                if not ops:   # faddp with no operands = faddp st(1), st
                    return f"ST(1)={R.format(expr.format(a='ST(1)', b='ST(0)'))};fpop(c);"
                if len(ops) == 1 and o0.type == X.X86_OP_MEM:
                    return f"ST(0)={R.format(expr.format(a='ST(0)', b=self.fmem(o0, i)))};"
                if len(ops) == 1 and pop:   # faddp st(i): st(i) = st(i) op st0, pop
                    k = self.sti(o0, i)
                    return f"ST({k})={R.format(expr.format(a=f'ST({k})', b='ST(0)'))};fpop(c);"
                if len(ops) == 1:   # fadd st(i): st0 = st0 op st(i)
                    k = self.sti(o0, i)
                    return f"ST(0)={R.format(expr.format(a='ST(0)', b=f'ST({k})'))};"
                d, s = self.sti(o0, i), self.sti(o1, i)
                out = f"ST({d})={R.format(expr.format(a=f'ST({d})', b=f'ST({s})'))};"
                return out + ("fpop(c);" if pop else "")
            if m == "fi" + base:
                return f"ST(0)={R.format(expr.format(a='ST(0)', b=self.fimem(o0, i)))};"
        if m in ("fcom", "fcomp", "fcompp", "fucom", "fucomp", "fucompp", "ficom", "ficomp"):
            if m in ("ficom", "ficomp"):
                b = self.fimem(o0, i)
            elif o0 is None:
                b = "ST(1)"
            elif o0.type == X.X86_OP_MEM:
                b = self.fmem(o0, i)
            else:
                b = f"ST({self.sti(o0, i)})"
            pops = 2 if m.endswith("pp") else 1 if m.endswith("p") else 0
            return "fcom(c,ST(0)," + b + ");" + "fpop(c);" * pops
        if m in ("fcomi", "fcomip", "fucomi", "fucomip"):
            k = self.sti(o1, i) if o1 is not None else 1
            pop = "fpop(c);" if m.endswith("p") else ""
            # ZF PF CF from the compare: model as an unsigned compare of ordered values
            return (f"{{double a_=ST(0),b_=ST({k});u32 un_=(a_!=a_||b_!=b_);"
                    f"c_fop=F_FLAGS|4;c_fr=flags_from_bits(un_||a_<b_,un_||a_==b_,0,0,un_);c_fc=c_fr&1;c_fo=0;c_fa=0;c_fb=0;}}" + pop)
        if m == "ftst":
            return "fcom(c,ST(0),0.0);"
        if m in ("fnstsw", "fstsw"):
            if o0.type == X.X86_OP_REG:
                return "eax=(eax&0xffff0000u)|(fsw(c)&0xffffu);"
            return f"st16({self.addr(o0, i)},fsw(c));"
        if m in ("fnstcw", "fstcw"):
            return f"st16({self.addr(o0, i)},c->fcw);"
        if m == "fldcw":
            return f"fldcw(c,ld16({self.addr(o0, i)}));"
        if m in ("fnclex", "fclex", "fninit", "finit", "fwait", "fsetpm", "ffree"):
            if m in ("fninit", "finit"):
                return "c->ftop=0;fldcw(c,0x37f);c->fsw_cc=0;"
            return ";"
        if m in ("fsin", "fcos", "fpatan", "fptan", "fyl2x", "f2xm1", "fscale", "fxtract"):
            raise Unsupported("x87 " + m)
        raise Unsupported("x87 " + m)

    # ---- MMX / SSE ------------------------------------------------------------------------
    def vreg(self, op, i):
        n = i.reg_name(op.reg)
        if n.startswith("xmm"):
            return f"c->xmm[{n[3]}]", 16
        if n.startswith("mm"):
            return f"c->mm[{n[2]}]", 8
        raise Unsupported("vector reg " + n)

    def simd(self, i, m, ops):
        """MMX/SSE through helpers in runtime/simd.h: each op works on a temporary copy."""
        o0 = ops[0] if ops else None
        o1 = ops[1] if len(ops) > 1 else None

        def src(op, width):
            if op.type == X.X86_OP_REG:
                n = i.reg_name(op.reg)
                if n.startswith("xmm") or n.startswith("mm"):
                    r, w = self.vreg(op, i)
                    return f"&{r}", w
                return None, 4
            a = self.addr(op, i)
            return f"(const void*)HOST({a})", width

        if m in ("movaps", "movups", "movdqa", "movdqu", "movq", "movd", "movss", "movsd"):
            if m == "movd":
                if o0.type == X.X86_OP_REG and not i.reg_name(o0.reg)[0] in "xm":
                    r, w = self.vreg(o1, i)
                    return self.write(o0, i, f"{r}.d[0]")
                if o0.type == X.X86_OP_MEM:
                    r, w = self.vreg(o1, i)
                    return f"st32({self.addr(o0, i)},{r}.d[0]);"
                r, w = self.vreg(o0, i)
                val = self.read(o1, i, 4)
                return f"memset(&{r},0,{w});{r}.d[0]={val};"
            n = {"movaps": 16, "movups": 16, "movdqa": 16, "movdqu": 16, "movq": 8, "movss": 4, "movsd": 8}[m]
            if o0.type == X.X86_OP_MEM:
                r, w = self.vreg(o1, i)
                return f"memcpy(HOST({self.addr(o0, i)}),&{r},{n});"
            r, w = self.vreg(o0, i)
            if o1.type == X.X86_OP_MEM:
                zero = f"memset(&{r},0,{w});" if (n < w and (m in ("movss", "movsd") or (m == "movq" and w == 16))) else ""
                return f"{zero}memcpy(&{r},HOST({self.addr(o1, i)}),{n});"
            s, _ = self.vreg(o1, i)
            if m == "movq" and w == 16:
                return f"{{xmm_t t_;memset(&t_,0,16);memcpy(&t_,&{s},8);{r}=t_;}}"
            return f"memmove(&{r},&{s},{n});"
        if m.startswith("cvt"):
            # register-to-gp forms
            if m in ("cvttsd2si", "cvtsd2si", "cvttss2si", "cvtss2si"):
                r, _ = self.vreg(o1, i) if o1.type == X.X86_OP_REG else (None, 0)
                if m.endswith("sd2si"):
                    v = f"{r}.g[0]" if r else f"ldd({self.addr(o1, i)})"
                else:
                    v = f"(double){r}.f[0]" if r else f"(double)ldf({self.addr(o1, i)})"
                conv = f"cvt_trunc32({v})" if m.startswith("cvtt") else f"cvt_round32(c,{v})"
                return self.write(o0, i, conv)
        r0, w0 = self.vreg(o0, i) if o0 is not None and o0.type == X.X86_OP_REG else (None, 0)
        if r0 is None:
            raise Unsupported("simd " + m + " to memory")
        s1, _ = src(o1, w0) if o1 is not None and o1.type != X.X86_OP_IMM else (None, 0)
        imm = ops[2].imm if len(ops) > 2 and ops[2].type == X.X86_OP_IMM else (o1.imm if o1 is not None and o1.type == X.X86_OP_IMM else 0)
        if m in ("cvtsi2ss", "cvtsi2sd", "pinsrw"):      # integer source
            s1 = self.read(o1, i, 4 if m != "pinsrw" or o1.type == X.X86_OP_REG else 2)
        immonly = o1 is not None and o1.type == X.X86_OP_IMM
        args = [f"&{r0}"] + ([s1] if s1 else []) + ([str(imm)] if (len(ops) > 2 or immonly) else [])
        name = f"simd_{m}_{w0}{'i' if immonly else ''}"
        self.ctx.simd_used.add(name)
        return f"{name}({','.join(args)});"

    # ---- whole function -------------------------------------------------------------------
    def translate(self, insns, labels, disasm):
        order = sorted(insns)
        body = []
        if order and order[0] != self.start:
            # blocks below the entry (reached by backward jumps): start at the entry, not the top
            labels = set(labels) | {self.start}
            body.append(f"  goto L_{self.start:x};")
        for k, a in enumerate(order):
            i = disasm.at(a)
            if a in labels:
                body.append(f"L_{a:x}:")
            if a in getattr(self.ctx, "hooked", ()):
                # config "hooks": the project's insn_hook(c, addr) runs first and may change any
                # state; a nonzero return skips the instruction
                body.append(f"  {{int skip_;{SYNC}skip_=INSN_HOOK(c,0x{a:x}u);{LOAD}"
                            f"if(!skip_){{{self.insn(i, order[k + 1] if k + 1 < len(order) else None)}}}}}")
            else:
                body.append("  " + self.insn(i, order[k + 1] if k + 1 < len(order) else None))
            nxt = a + i.size
            falls = not (i.mnemonic in ("ret", "jmp", "int3", "hlt", "ud2") or
                         (i.mnemonic == "call" and i.operands[0].type == X.X86_OP_IMM and
                          (i.operands[0].imm & 0xffffffff) in self.ctx.noreturn))
            if falls:
                following = order[k + 1] if k + 1 < len(order) else None
                if following != nxt:
                    if nxt in labels:
                        body.append(f"  goto L_{nxt:x};")
                    elif self.ctx.is_func(nxt):
                        body.append("  " + self.tail_direct(nxt))
                    else:
                        body.append(f"  {SYNC}trap(c,0x{nxt:x}u,\"fell off function\");")
        for x in sorted(labels):
            if x is not None and x not in insns:
                body.append(f"L_{x:x}:  {SYNC}trap(c,0x{x:x}u,\"branch into undecoded bytes\");")
        text = "\n".join(body)
        used = {f"L_{x:x}" for x in labels if x is not None and f"L_{x:x};" in text}
        body = [b for b in body if not ((b.endswith(":") and b[:-1] not in used) or
                                        (b.startswith("L_") and ":  " in b and b.split(":")[0] not in used))]
        hook = f"  TRACE_FN(c, 0x{self.start:x}u);\n" if self.start in getattr(self.ctx, "traced", ()) else ""
        head = (f"void {fn_name(self.start)}(struct cpu *c) {{\n{hook}"
                "  u32 eax=c->eax,ecx=c->ecx,edx=c->edx,ebx=c->ebx,esp=c->esp,ebp=c->ebp,esi=c->esi,edi=c->edi;\n"
                "  u32 c_fop=c->fop,c_fa=c->fa,c_fb=c->fb,c_fr=c->fr,c_fc=c->fc,c_fo=c->fo;\n"
                "  (void)c_fop;(void)c_fa;(void)c_fb;(void)c_fr;(void)c_fc;(void)c_fo;\n")
        return head + "\n".join(body) + "\n}\n"
