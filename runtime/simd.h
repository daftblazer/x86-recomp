/* MMX / SSE helpers for the generated code: simd_<mnemonic>_<register bytes>[i](dst, src[, imm]).
 * Each copies its operands first, so dst and src may be the same register.
 * Only the forms needed so far are here: the translator lists what a program uses in
 * simd_used.txt, and a missing helper shows up as a compile error. Add them as programs need them.
 */
#pragma once
#include "cpu.h"

#define V_IN(W) xmm_t d, s; memcpy(&d, dp, W); memcpy(&s, sp, W)
#define V_OUT(W) memcpy(dp, &d, W)
#define LANES(name, W, N, F, EXPR) \
    static inline void name(void *dp, const void *sp) { V_IN(W); for (int k = 0; k < N; k++) d.F[k] = (EXPR); V_OUT(W); }

static inline s16 sat16(s32 v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (s16)v; }
static inline u8 satu8(s32 v) { return v > 255 ? 255 : v < 0 ? 0 : (u8)v; }
static inline s32 cvt_trunc32(double v) { return (v != v || v >= 2147483648.0 || v < -2147483648.0) ? (s32)0x80000000 : (s32)v; }
static inline s32 cvt_round32(struct cpu *c, double v) { (void)c; v = rint(v); return cvt_trunc32(v); }

/* ---- float packed ---- */
LANES(simd_addps_16, 16, 4, f, d.f[k] + s.f[k])
LANES(simd_subps_16, 16, 4, f, d.f[k] - s.f[k])
LANES(simd_mulps_16, 16, 4, f, d.f[k] * s.f[k])
LANES(simd_divps_16, 16, 4, f, d.f[k] / s.f[k])
LANES(simd_maxps_16, 16, 4, f, d.f[k] > s.f[k] ? d.f[k] : s.f[k])
LANES(simd_sqrtps_16, 16, 4, f, sqrtf(s.f[k]))
LANES(simd_andps_16, 16, 4, d, d.d[k] & s.d[k])
LANES(simd_orps_16, 16, 4, d, d.d[k] | s.d[k])
LANES(simd_xorps_16, 16, 4, d, d.d[k] ^ s.d[k])
LANES(simd_cvtdq2ps_16, 16, 4, f, (float)s.sd[k])
LANES(simd_cvttps2dq_16, 16, 4, sd, cvt_trunc32(s.f[k]))
/* scalar (memory sources are only 4/8 bytes: read just that much) */
#define SCALAR(name, n, F, EXPR) \
    static inline void name(void *dp, const void *sp) { xmm_t d, s; memcpy(&d, dp, 16); memcpy(&s, sp, n); d.F[0] = (EXPR); memcpy(dp, &d, 16); }
SCALAR(simd_addss_16, 4, f, d.f[0] + s.f[0])
SCALAR(simd_subss_16, 4, f, d.f[0] - s.f[0])
SCALAR(simd_mulss_16, 4, f, d.f[0] * s.f[0])
SCALAR(simd_mulsd_16, 8, g, d.g[0] * s.g[0])
SCALAR(simd_divsd_16, 8, g, d.g[0] / s.g[0])
SCALAR(simd_cvtss2sd_16, 4, g, (double)s.f[0])
static inline void simd_cvtsi2ss_16(void *dp, u32 v) { xmm_t d; memcpy(&d, dp, 16); d.f[0] = (float)(s32)v; memcpy(dp, &d, 16); }
static inline void simd_cvtps2pd_16(void *dp, const void *sp) {
    xmm_t d, s; memcpy(&s, sp, 8); d.g[0] = s.f[0]; d.g[1] = s.f[1]; memcpy(dp, &d, 16);
}
static inline void simd_cvtps2pi_8(void *dp, const void *sp) {
    xmm_t d, s; memcpy(&s, sp, 8); d.sd[0] = cvt_round32(0, s.f[0]); d.sd[1] = cvt_round32(0, s.f[1]); memcpy(dp, &d, 8);
}
static inline void simd_shufps_16(void *dp, const void *sp, int imm) {
    V_IN(16); xmm_t r;
    r.f[0] = d.f[imm & 3]; r.f[1] = d.f[(imm >> 2) & 3]; r.f[2] = s.f[(imm >> 4) & 3]; r.f[3] = s.f[(imm >> 6) & 3];
    memcpy(dp, &r, 16);
}
static inline void simd_unpcklps_16(void *dp, const void *sp) {
    V_IN(16); xmm_t r; r.f[0] = d.f[0]; r.f[1] = s.f[0]; r.f[2] = d.f[1]; r.f[3] = s.f[1]; memcpy(dp, &r, 16);
}

/* ---- integer, both widths ---- */
#define INT2(op, N8, N16, F, EXPR) LANES(simd_##op##_8, 8, N8, F, EXPR) LANES(simd_##op##_16, 16, N16, F, EXPR)
INT2(paddd, 2, 4, d, d.d[k] + s.d[k])
INT2(psubd, 2, 4, d, d.d[k] - s.d[k])
INT2(paddw, 4, 8, w, (u16)(d.w[k] + s.w[k]))
INT2(paddsw, 4, 8, sw, sat16((s32)d.sw[k] + s.sw[k]))
INT2(psubsw, 4, 8, sw, sat16((s32)d.sw[k] - s.sw[k]))
INT2(pmulhw, 4, 8, sw, (s16)(((s32)d.sw[k] * s.sw[k]) >> 16))
INT2(por, 1, 2, q, d.q[k] | s.q[k])
INT2(pxor, 1, 2, q, d.q[k] ^ s.q[k])
LANES(simd_pand_8, 8, 1, q, d.q[k] & s.q[k])
LANES(simd_pandn_8, 8, 1, q, ~d.q[k] & s.q[k])
LANES(simd_paddq_16, 16, 2, q, d.q[k] + s.q[k])
LANES(simd_paddusb_8, 8, 8, b, satu8((s32)d.b[k] + s.b[k]))
LANES(simd_psubb_8, 8, 8, b, (u8)(d.b[k] - s.b[k]))
LANES(simd_pavgb_8, 8, 8, b, (u8)(((u32)d.b[k] + s.b[k] + 1) >> 1))

static inline void pmaddwd_(void *dp, const void *sp, int W) {
    xmm_t d, s, r; memcpy(&d, dp, W); memcpy(&s, sp, W);
    for (int k = 0; k < W / 4; k++) r.sd[k] = (s32)d.sw[2 * k] * s.sw[2 * k] + (s32)d.sw[2 * k + 1] * s.sw[2 * k + 1];
    memcpy(dp, &r, W);
}
static inline void simd_pmaddwd_8(void *dp, const void *sp) { pmaddwd_(dp, sp, 8); }
static inline void simd_pmaddwd_16(void *dp, const void *sp) { pmaddwd_(dp, sp, 16); }

static inline void packssdw_(void *dp, const void *sp, int W) {
    xmm_t d, s, r; int h = W / 4; memcpy(&d, dp, W); memcpy(&s, sp, W);
    for (int k = 0; k < h; k++) { r.sw[k] = sat16(d.sd[k]); r.sw[h + k] = sat16(s.sd[k]); }
    memcpy(dp, &r, W);
}
static inline void simd_packssdw_8(void *dp, const void *sp) { packssdw_(dp, sp, 8); }
static inline void simd_packssdw_16(void *dp, const void *sp) { packssdw_(dp, sp, 16); }
static inline void packuswb_(void *dp, const void *sp, int W) {
    xmm_t d, s, r; int h = W / 2; memcpy(&d, dp, W); memcpy(&s, sp, W);
    for (int k = 0; k < h; k++) { r.b[k] = satu8(d.sw[k]); r.b[h + k] = satu8(s.sw[k]); }
    memcpy(dp, &r, W);
}
static inline void simd_packuswb_8(void *dp, const void *sp) { packuswb_(dp, sp, 8); }
static inline void simd_packuswb_16(void *dp, const void *sp) { packuswb_(dp, sp, 16); }
static inline void punpcklbw_(void *dp, const void *sp, int W, int hi) {
    xmm_t d, s, r; int h = W / 2; memcpy(&d, dp, W); memcpy(&s, sp, W);
    for (int k = 0; k < h; k++) { r.b[2 * k] = d.b[k + hi * h]; r.b[2 * k + 1] = s.b[k + hi * h]; }
    memcpy(dp, &r, W);
}
static inline void simd_punpcklbw_8(void *dp, const void *sp) { punpcklbw_(dp, sp, 8, 0); }
static inline void simd_punpcklbw_16(void *dp, const void *sp) { punpcklbw_(dp, sp, 16, 0); }
static inline void simd_punpckhbw_8(void *dp, const void *sp) { punpcklbw_(dp, sp, 8, 1); }

/* shuffles */
static inline void simd_pshufd_16(void *dp, const void *sp, int imm) {
    xmm_t s, r; memcpy(&s, sp, 16); for (int k = 0; k < 4; k++) r.d[k] = s.d[(imm >> (2 * k)) & 3]; memcpy(dp, &r, 16);
}
static inline void simd_pshuflw_16(void *dp, const void *sp, int imm) {
    xmm_t s, r; memcpy(&s, sp, 16); r.q[1] = s.q[1]; for (int k = 0; k < 4; k++) r.w[k] = s.w[(imm >> (2 * k)) & 3]; memcpy(dp, &r, 16);
}
static inline void simd_pshufhw_16(void *dp, const void *sp, int imm) {
    xmm_t s, r; memcpy(&s, sp, 16); r.q[0] = s.q[0]; for (int k = 0; k < 4; k++) r.w[4 + k] = s.w[4 + ((imm >> (2 * k)) & 3)]; memcpy(dp, &r, 16);
}
static inline void simd_pshufw_8(void *dp, const void *sp, int imm) {
    xmm_t s, r; memcpy(&s, sp, 8); for (int k = 0; k < 4; k++) r.w[k] = s.w[(imm >> (2 * k)) & 3]; memcpy(dp, &r, 8);
}
static inline void simd_pinsrw_8(void *dp, u32 v, int imm) { mmx_t d; memcpy(&d, dp, 8); d.w[imm & 3] = (u16)v; memcpy(dp, &d, 8); }
static inline void simd_pinsrw_16(void *dp, u32 v, int imm) { xmm_t d; memcpy(&d, dp, 16); d.w[imm & 7] = (u16)v; memcpy(dp, &d, 16); }

/* shifts by immediate */
#define SHIFTI(name, W, N, F, EXPR) \
    static inline void name(void *dp, int n) { xmm_t d; memcpy(&d, dp, W); for (int k = 0; k < N; k++) d.F[k] = (EXPR); memcpy(dp, &d, W); }
SHIFTI(simd_psrad_8i, 8, 2, sd, d.sd[k] >> (n > 31 ? 31 : n))
SHIFTI(simd_psrad_16i, 16, 4, sd, d.sd[k] >> (n > 31 ? 31 : n))
SHIFTI(simd_psraw_8i, 8, 4, sw, (s16)(d.sw[k] >> (n > 15 ? 15 : n)))
SHIFTI(simd_psraw_16i, 16, 8, sw, (s16)(d.sw[k] >> (n > 15 ? 15 : n)))
SHIFTI(simd_psrlw_8i, 8, 4, w, (u16)(n > 15 ? 0 : d.w[k] >> n))
SHIFTI(simd_psrlq_8i, 8, 1, q, n > 63 ? 0 : d.q[k] >> n)
