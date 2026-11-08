/* Guest CPU state and the helpers the generated code uses.
 *
 * Guest addresses are host addresses: the runtime maps the image and every guest allocation in
 * the low 4 GB, so a guest pointer is (void *)(uintptr_t)addr. All access goes through the
 * ld/st helpers so a base offset could be added in one place.
 */
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

#define GUEST_BASE 0
#define HOST(a) ((void *)(uintptr_t)(GUEST_BASE + (u32)(a)))

typedef union {
    u8 b[16]; u16 w[8]; u32 d[4]; u64 q[2];
    s8 sb[16]; s16 sw[8]; s32 sd[4]; s64 sq[2];
    float f[4]; double g[2];
} xmm_t;
typedef union { u8 b[8]; u16 w[4]; u32 d[2]; u64 q; s8 sb[8]; s16 sw[4]; s32 sd[2]; } mmx_t;

struct cpu {
    u32 eax, ecx, edx, ebx, esp, ebp, esi, edi;
    /* lazy flags, see cc_* below */
    u32 fop, fa, fb, fr, fc, fo;
    u32 df;                 /* direction flag (0 = forward) */
    u32 fs_base;            /* guest TEB */
    /* x87 */
    double st[8];
    u32 ftop;
    u32 fsw_cc;             /* C0 (bit 8), C1 (9), C2 (10), C3 (14) as in the status word */
    u16 fcw;                /* control word: PC bits 8-9, RC bits 10-11 */
    u8 fpc24;               /* precision control is 24-bit: round results to float */
    u8 _pad;
    xmm_t xmm[8];
    u32 mxcsr;
    mmx_t mm[8];
    void *host;             /* runtime: thread data */
};

/* ---- memory ---------------------------------------------------------------------------- */
static inline u8 ld8(u32 a) { return *(volatile u8 *)HOST(a); }
static inline u16 ld16(u32 a) { u16 v; memcpy(&v, HOST(a), 2); return v; }
static inline u32 ld32(u32 a) { u32 v; memcpy(&v, HOST(a), 4); return v; }
static inline u64 ld64(u32 a) { u64 v; memcpy(&v, HOST(a), 8); return v; }
static inline float ldf(u32 a) { float v; memcpy(&v, HOST(a), 4); return v; }
static inline double ldd(u32 a) { double v; memcpy(&v, HOST(a), 8); return v; }
static inline void st8(u32 a, u32 v) { *(volatile u8 *)HOST(a) = (u8)v; }
static inline void st16(u32 a, u32 v) { u16 x = (u16)v; memcpy(HOST(a), &x, 2); }
static inline void st32(u32 a, u32 v) { memcpy(HOST(a), &v, 4); }
static inline void st64(u32 a, u64 v) { memcpy(HOST(a), &v, 8); }
static inline void stf(u32 a, float v) { memcpy(HOST(a), &v, 4); }
static inline void std_(u32 a, double v) { memcpy(HOST(a), &v, 8); }
static inline void ld128(u32 a, xmm_t *x) { memcpy(x, HOST(a), 16); }
static inline void st128(u32 a, const xmm_t *x) { memcpy(HOST(a), x, 16); }
double ld80(u32 a);
void st80(u32 a, double v);

/* ---- lazy flags ------------------------------------------------------------------------
 * fop = kind | size (size: 1, 2, 4 in the low 3 bits). fa/fb are the operands, fr the result.
 * EAGER: CF in fc, OF in fo (shifts, rotates, mul, adc/sbb, bt).
 * INC/DEC keep the previous CF in fc.
 */
enum { F_ADD = 0x10, F_SUB = 0x20, F_LOGIC = 0x30, F_INC = 0x40, F_DEC = 0x50, F_NEG = 0x60,
       F_EAGER = 0x70, F_FLAGS = 0x80 /* fr holds an EFLAGS value */ };
#define F_SIZE(op) ((op) & 7)
static inline u32 f_msb(u32 op) { return F_SIZE(op) == 1 ? 0x80u : F_SIZE(op) == 2 ? 0x8000u : 0x80000000u; }
static inline u32 f_mask(u32 op) { return F_SIZE(op) == 1 ? 0xffu : F_SIZE(op) == 2 ? 0xffffu : 0xffffffffu; }

static inline int cc_cf(u32 op, u32 a, u32 b, u32 r, u32 c) {
    u32 m = f_mask(op);
    switch (op & 0xf0) {
    case F_ADD: return (r & m) < (a & m);
    case F_SUB: return (a & m) < (b & m);
    case F_LOGIC: return 0;
    case F_NEG: return (a & m) != 0;
    case F_FLAGS: return r & 1;
    default: return c & 1;            /* INC, DEC, EAGER */
    }
}
static inline int cc_zf(u32 op, u32 r) { return (op & 0xf0) == F_FLAGS ? (r >> 6) & 1 : (r & f_mask(op)) == 0; }
static inline int cc_sf(u32 op, u32 r) { return (op & 0xf0) == F_FLAGS ? (r >> 7) & 1 : (r & f_msb(op)) != 0; }
static inline int cc_of(u32 op, u32 a, u32 b, u32 r, u32 o) {
    u32 s = f_msb(op), m = f_mask(op);
    switch (op & 0xf0) {
    case F_ADD: return ((a ^ r) & (b ^ r) & s) != 0;
    case F_SUB: return ((a ^ b) & (a ^ r) & s) != 0;
    case F_LOGIC: return 0;
    case F_INC: return (r & m) == s;
    case F_DEC: return (r & m) == s - 1;
    case F_NEG: return (r & m) == s;
    case F_FLAGS: return (r >> 11) & 1;
    default: return o & 1;
    }
}
static inline int cc_pf(u32 op, u32 r) { return (op & 0xf0) == F_FLAGS ? (r >> 2) & 1 : !__builtin_parity(r & 0xff); }

#define FL c_fop, c_fa, c_fb, c_fr, c_fc
#define CF cc_cf(c_fop, c_fa, c_fb, c_fr, c_fc)
#define ZF cc_zf(c_fop, c_fr)
#define SF cc_sf(c_fop, c_fr)
#define OF cc_of(c_fop, c_fa, c_fb, c_fr, c_fo)
#define PF cc_pf(c_fop, c_fr)
/* condition codes, x86 names */
#define C_O OF
#define C_NO (!OF)
#define C_B CF
#define C_AE (!CF)
#define C_E ZF
#define C_NE (!ZF)
#define C_BE (CF || ZF)
#define C_A (!CF && !ZF)
#define C_S SF
#define C_NS (!SF)
#define C_P PF
#define C_NP (!PF)
#define C_L (SF != OF)
#define C_GE (SF == OF)
#define C_LE (ZF || SF != OF)
#define C_G (!ZF && SF == OF)

/* EFLAGS <-> lazy flags (pushfd/popfd, lahf/sahf, clc/stc/cmc, rotates) */
static inline u32 flags_get_l(u32 op, u32 a, u32 b, u32 r, u32 cin, u32 o, u32 df) {
    return 0x2u | (u32)cc_cf(op, a, b, r, cin) | ((u32)cc_pf(op, r) << 2) | ((u32)cc_zf(op, r) << 6) |
           ((u32)cc_sf(op, r) << 7) | ((df & 1) << 10) | ((u32)cc_of(op, a, b, r, o) << 11);
}
static inline u32 flags_from_bits(u32 cf, u32 zf, u32 sf, u32 of, u32 pf) {
    return 0x2u | (cf & 1) | ((pf & 1) << 2) | ((zf & 1) << 6) | ((sf & 1) << 7) | ((of & 1) << 11);
}
static inline void flags_set_l(u32 f, u32 *op, u32 *a, u32 *b, u32 *r, u32 *cin, u32 *o, u32 *df) {
    *op = F_FLAGS | 4; *a = 0; *b = 0; *r = f; *cin = f & 1; *o = (f >> 11) & 1; *df = (f >> 10) & 1;
}
#define c_df (c->df)

/* ---- x87 ------------------------------------------------------------------------------- */
#define ST(i) (c->st[(c->ftop + (i)) & 7])
static inline void fpush(struct cpu *c, double v) { c->ftop = (c->ftop - 1) & 7; c->st[c->ftop] = v; }
static inline void fpop(struct cpu *c) { c->ftop = (c->ftop + 1) & 7; }
static inline double fround(struct cpu *c, double v) { return c->fpc24 ? (double)(float)v : v; }
static inline void fcom(struct cpu *c, double a, double b) {
    if (a != a || b != b) c->fsw_cc = 0x4500;          /* unordered: C3 C2 C0 */
    else if (a > b) c->fsw_cc = 0;
    else if (a < b) c->fsw_cc = 0x0100;                /* C0 */
    else c->fsw_cc = 0x4000;                           /* C3 */
}
static inline u32 fsw(struct cpu *c) { return c->fsw_cc | ((c->ftop & 7) << 11); }
void fldcw(struct cpu *c, u16 cw);
double frndint_(struct cpu *c, double v);     /* round with RC */
s64 fist(struct cpu *c, double v, int bits);   /* round with RC, x87 integer-indefinite on overflow */

/* ---- calls ----------------------------------------------------------------------------- */
/* called before every direct import call when set (tracing); one load and branch otherwise */
extern void (*import_hook)(const char *name, u32 ret);
/* functions listed under "trace" in the config call this at entry (when set) */
extern void (*fn_trace_hook)(struct cpu *c, u32 addr);
#define TRACE_FN(c, a) do { if (fn_trace_hook) fn_trace_hook(c, a); } while (0)
/* config "hooks": called before each listed instruction (the project sets it); a nonzero return
 * skips the instruction (a skipped call pushes nothing and calls nothing) */
extern int (*insn_hook)(struct cpu *c, u32 addr);
#define INSN_HOOK(c, a) (insn_hook ? insn_hook(c, a) : 0)
#define IMPORT_HOOK(name, ret) do { if (__builtin_expect(import_hook != 0, 0)) import_hook(name, ret); } while (0)
typedef void (*guest_fn)(struct cpu *);
void dispatch(struct cpu *c, u32 target);           /* indirect call/jmp target */
_Noreturn void trap(struct cpu *c, u32 at, const char *what);
void rep_movs(struct cpu *c, int size);
void rep_stos(struct cpu *c, int size);
