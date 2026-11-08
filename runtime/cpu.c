/* CPU helpers used by the generated code: x87 control word, rounding, 80-bit memory format,
 * rep string ops. */
#include "cpu.h"
#include <fenv.h>

void fldcw(struct cpu *c, u16 cw) {
    c->fcw = cw;
    c->fpc24 = ((cw >> 8) & 3) == 0;     /* PC = 00: 24-bit */
}

static double round_rc(struct cpu *c, double v) {
    switch ((c->fcw >> 10) & 3) {
    case 0: return nearbyint(v);          /* host default mode is nearest-even */
    case 1: return floor(v);
    case 2: return ceil(v);
    default: return trunc(v);
    }
}

double frndint_(struct cpu *c, double v) { return round_rc(c, v); }

s64 fist(struct cpu *c, double v, int bits) {
    double r = round_rc(c, v);
    if (bits == 16) return (r != r || r > 32767.0 || r < -32768.0) ? (s16)0x8000 : (s64)r;
    if (bits == 32) return (r != r || r > 2147483647.0 || r < -2147483648.0) ? (s32)0x80000000 : (s64)r;
    return (r != r || r >= 9223372036854775808.0 || r < -9223372036854775808.0) ? (s64)0x8000000000000000ull : (s64)r;
}

/* 80-bit extended <-> double */
double ld80(u32 a) {
    u64 mant = ld64(a);
    u16 se = ld16(a + 8);
    int sign = se >> 15, exp = se & 0x7fff;
    double v;
    if (exp == 0 && mant == 0) v = 0.0;
    else if (exp == 0x7fff) v = (mant << 1) ? NAN : INFINITY;
    else v = ldexp((double)mant, exp - 16383 - 63);
    return sign ? -v : v;
}

void st80(u32 a, double v) {
    u64 mant = 0;
    u16 se = signbit(v) ? 0x8000 : 0;
    if (v != v) { mant = 0xc000000000000000ull; se |= 0x7fff; }
    else if (isinf(v)) { mant = 0x8000000000000000ull; se |= 0x7fff; }
    else if (v != 0) {
        int e;
        double f = frexp(fabs(v), &e);              /* f in [0.5, 1) */
        mant = (u64)ldexp(f, 64);
        se |= (u16)(e - 1 + 16383);
    }
    st64(a, mant);
    st16(a + 8, se);
}

void rep_movs(struct cpu *c, int size) {
    u32 n = c->ecx;
    if (!n) return;
    if (!c->df) {
        /* forward copy, element by element semantics (overlap like the hardware) */
        if (c->edi <= c->esi || c->edi >= c->esi + n * (u32)size) memmove(HOST(c->edi), HOST(c->esi), (size_t)n * size);
        else for (u32 k = 0; k < n * (u32)size; k++) st8(c->edi + k, ld8(c->esi + k));
        c->esi += n * size; c->edi += n * size;
    } else {
        for (u32 k = 0; k < n; k++) {
            if (size == 4) st32(c->edi, ld32(c->esi)); else if (size == 2) st16(c->edi, ld16(c->esi)); else st8(c->edi, ld8(c->esi));
            c->esi -= size; c->edi -= size;
        }
    }
    c->ecx = 0;
}

void rep_stos(struct cpu *c, int size) {
    u32 n = c->ecx;
    if (!n) return;
    if (size == 1 && !c->df) { memset(HOST(c->edi), (int)(c->eax & 0xff), n); c->edi += n; }
    else for (u32 k = 0; k < n; k++) {
        if (size == 4) st32(c->edi, c->eax); else if (size == 2) st16(c->edi, c->eax); else st8(c->edi, c->eax);
        c->edi += c->df ? -size : size;
    }
    c->ecx = 0;
}
