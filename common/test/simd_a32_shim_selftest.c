/* Selftest for the AArch32 stand-ins in utility/simd_kernels.h: each AArch32 stand-in against a lane-by-lane C reference, and
 * on AArch64 additionally against the real AArch64 intrinsic it replaces. Random vectors plus the
 * special values (signed zeros, infinities, NaN, extreme magnitudes; subnormals only where the vector
 * unit does not flush them, i.e. not on 32-bit ARM). A NaN result must be a NaN on both sides; every
 * other result must match bit for bit. Compile with -ffp-contract=off. */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if !defined(__ARM_NEON)

int main(void) {
    printf("simd_a32_shim_selftest: SKIP (no NEON on this target)\n");
    return 0;
}

#else

#if defined(__aarch64__)
#define SK_A32_SHIM_HOST_TEST 1
#endif
#include "utility/simd_kernels.h"

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static uint32_t rnd32(void) {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return (uint32_t)(g_state >> 16);
}

static float bits_to_f(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }
static uint32_t f_to_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }

/* One float drawn from the classes the kernels see. */
static float draw(void) {
    uint32_t c = rnd32() % 16u;
    if (c == 0) return 0.0f;
    if (c == 1) return -0.0f;
    if (c == 2) return INFINITY;
    if (c == 3) return -INFINITY;
    if (c == 4) return bits_to_f(0x7fc00000u);
    if (c == 5) return 3.4028234e38f;
    if (c == 6) return -3.4028234e38f;
#if !defined(__arm__)
    if (c == 7) return bits_to_f(rnd32() & 0x007fffffu);
    if (c == 8) return bits_to_f(0x80000000u | (rnd32() & 0x007fffffu));
#endif
    if (c == 9) return 1.17549435e-38f;
    {
        int e = (int)(rnd32() % 61u) - 30;
        float m = (float)(rnd32() & 0xffffffu) / 16777216.0f + 0.5f;
        float v = ldexpf(m, e);
        return (rnd32() & 1u) ? -v : v;
    }
}

static int same_float(float a, float b) {
    if (isnan(a) || isnan(b)) return isnan(a) && isnan(b);
    return f_to_bits(a) == f_to_bits(b);
}

static long g_checks, g_fail;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fail; if (g_fail < 20) printf("FAIL %s (iteration %ld)\n", what, g_checks); } } while (0)

static void load4(float32x4_t *v, float out[4]) {
    int k;
    for (k = 0; k < 4; ++k) out[k] = draw();
    *v = vld1q_f32(out);
}

static float ref_max4(const float l[4]) {
    float r = l[0];
    int k;
    for (k = 1; k < 4; ++k) {
        if (isnan(r) || isnan(l[k])) return NAN;
        if (l[k] > r || (l[k] == 0.0f && r == 0.0f && !signbit(l[k]))) r = l[k];
    }
    return r;
}

int main(void) {
    long it;
    for (it = 0; it < 200000; ++it) {
        float a[4], b[4], pos[4], got[4], want[4];
        float32x4_t va, vb, vr;
        int k;
        load4(&va, a);
        load4(&vb, b);

        /* division and square root: the correctly rounded per-lane scalar operations */
        vr = sk_a32_vdivq_f32(va, vb);
        vst1q_f32(got, vr);
        for (k = 0; k < 4; ++k) CHECK(same_float(got[k], a[k] / b[k]), "vdivq_f32 vs scalar /");
        for (k = 0; k < 4; ++k) pos[k] = fabsf(a[k]);
        vr = sk_a32_vsqrtq_f32(vld1q_f32(pos));
        vst1q_f32(got, vr);
        for (k = 0; k < 4; ++k) CHECK(same_float(got[k], sqrtf(pos[k])), "vsqrtq_f32 vs sqrtf");
        vr = sk_a32_vrndmq_f32(va);
        vst1q_f32(got, vr);
        for (k = 0; k < 4; ++k) CHECK(same_float(got[k], floorf(a[k])), "vrndmq_f32 vs floorf");

        /* across-lane reductions: (l0 + l1) + (l2 + l3), and max */
        CHECK(same_float(sk_a32_vaddvq_f32(va), (a[0] + a[1]) + (a[2] + a[3])), "vaddvq_f32 order");
        CHECK(same_float(sk_a32_vmaxvq_f32(va), ref_max4(a)), "vmaxvq_f32");

        /* pairwise add of two vectors, and zips */
        vst1q_f32(got, sk_a32_vpaddq_f32(va, vb));
        want[0] = a[0] + a[1]; want[1] = a[2] + a[3]; want[2] = b[0] + b[1]; want[3] = b[2] + b[3];
        for (k = 0; k < 4; ++k) CHECK(same_float(got[k], want[k]), "vpaddq_f32");
        vst1q_f32(got, sk_a32_vzip1q_f32(va, vb));
        want[0] = a[0]; want[1] = b[0]; want[2] = a[1]; want[3] = b[1];
        for (k = 0; k < 4; ++k) CHECK(f_to_bits(got[k]) == f_to_bits(want[k]), "vzip1q_f32");
        vst1q_f32(got, sk_a32_vzip2q_f32(va, vb));
        want[0] = a[2]; want[1] = b[2]; want[2] = a[3]; want[3] = b[3];
        for (k = 0; k < 4; ++k) CHECK(f_to_bits(got[k]) == f_to_bits(want[k]), "vzip2q_f32");

        /* integer reductions and the zero compare */
        {
            uint32_t u[4], ur;
            int32_t s[4];
            uint32_t eq[4];
            uint8_t by[16];
            uint32_t sum = 0, mx, mn;
            uint8_t bmin = 255;
            for (k = 0; k < 4; ++k) {
                u[k] = (rnd32() % 4u == 0) ? 0u : rnd32();
                s[k] = (rnd32() % 3u == 0) ? 0 : (int32_t)rnd32();
            }
            for (k = 0; k < 16; ++k) by[k] = (uint8_t)(rnd32() % 5u == 0 ? 255u : rnd32());
            mx = mn = u[0];
            for (k = 0; k < 4; ++k) { sum += u[k]; if (u[k] > mx) mx = u[k]; if (u[k] < mn) mn = u[k]; }
            ur = sk_a32_vaddvq_u32(vld1q_u32(u));
            CHECK(ur == sum, "vaddvq_u32");
            CHECK(sk_a32_vmaxvq_u32(vld1q_u32(u)) == mx, "vmaxvq_u32");
            CHECK(sk_a32_vminvq_u32(vld1q_u32(u)) == mn, "vminvq_u32");
            vst1q_u32(eq, sk_a32_vceqzq_s32(vld1q_s32(s)));
            for (k = 0; k < 4; ++k) CHECK(eq[k] == (s[k] == 0 ? 0xffffffffu : 0u), "vceqzq_s32");
            sum = 0;
            for (k = 0; k < 16; ++k) { sum += by[k]; if (by[k] < bmin) bmin = by[k]; }
            CHECK(sk_a32_vminvq_u8(vld1q_u8(by)) == bmin, "vminvq_u8");
            CHECK(sk_a32_vaddlvq_u8(vld1q_u8(by)) == (uint16_t)sum, "vaddlvq_u8");
        }

#if defined(__aarch64__)
        /* the real AArch64 intrinsics */
        {
            float ga[4], gb[4];
            uint32_t u[4];
            vst1q_f32(ga, sk_a32_vdivq_f32(va, vb));
            vst1q_f32(gb, vdivq_f32(va, vb));
            for (k = 0; k < 4; ++k) CHECK(same_float(ga[k], gb[k]), "vdivq_f32 vs AArch64");
            vst1q_f32(ga, sk_a32_vsqrtq_f32(vabsq_f32(va)));
            vst1q_f32(gb, vsqrtq_f32(vabsq_f32(va)));
            for (k = 0; k < 4; ++k) CHECK(same_float(ga[k], gb[k]), "vsqrtq_f32 vs AArch64");
            vst1q_f32(ga, sk_a32_vrndmq_f32(va));
            vst1q_f32(gb, vrndmq_f32(va));
            for (k = 0; k < 4; ++k) CHECK(same_float(ga[k], gb[k]), "vrndmq_f32 vs AArch64");
            CHECK(same_float(sk_a32_vaddvq_f32(va), vaddvq_f32(va)), "vaddvq_f32 vs AArch64");
            CHECK(same_float(sk_a32_vmaxvq_f32(va), vmaxvq_f32(va)), "vmaxvq_f32 vs AArch64");
            vst1q_f32(ga, sk_a32_vpaddq_f32(va, vb));
            vst1q_f32(gb, vpaddq_f32(va, vb));
            for (k = 0; k < 4; ++k) CHECK(same_float(ga[k], gb[k]), "vpaddq_f32 vs AArch64");
            vst1q_f32(ga, sk_a32_vzip1q_f32(va, vb));
            vst1q_f32(gb, vzip1q_f32(va, vb));
            for (k = 0; k < 4; ++k) CHECK(f_to_bits(ga[k]) == f_to_bits(gb[k]), "vzip1q_f32 vs AArch64");
            vst1q_f32(ga, sk_a32_vzip2q_f32(va, vb));
            vst1q_f32(gb, vzip2q_f32(va, vb));
            for (k = 0; k < 4; ++k) CHECK(f_to_bits(ga[k]) == f_to_bits(gb[k]), "vzip2q_f32 vs AArch64");
            for (k = 0; k < 4; ++k) u[k] = rnd32();
            CHECK(sk_a32_vaddvq_u32(vld1q_u32(u)) == vaddvq_u32(vld1q_u32(u)), "vaddvq_u32 vs AArch64");
            CHECK(sk_a32_vmaxvq_u32(vld1q_u32(u)) == vmaxvq_u32(vld1q_u32(u)), "vmaxvq_u32 vs AArch64");
            CHECK(sk_a32_vminvq_u32(vld1q_u32(u)) == vminvq_u32(vld1q_u32(u)), "vminvq_u32 vs AArch64");
        }
#endif
    }
    printf("simd_a32_shim_selftest: %ld checks, %ld failures%s\n", g_checks, g_fail,
#if defined(__aarch64__)
           " (reference and AArch64 intrinsics)"
#else
           " (reference)"
#endif
    );
    return g_fail ? 1 : 0;
}

#endif /* __ARM_NEON */
