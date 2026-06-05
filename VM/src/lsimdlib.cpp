// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lualib.h"

#include <math.h>
#include <string.h>

// Interpreter implementations of the simd library. Each value is a boxed 128-bit object holding four
// uint32 lanes. These are the correct-but-slow fallbacks; native code generation keeps the same values
// in registers and lowers the operations to packed SSE instructions.

static int simd_create(lua_State* L)
{
    uint32_t a = luaL_checkunsigned(L, 1);
    uint32_t b = luaL_checkunsigned(L, 2);
    uint32_t c = luaL_checkunsigned(L, 3);
    uint32_t d = luaL_checkunsigned(L, 4);

    uint32_t* r = lua_newsimd(L);
    r[0] = a;
    r[1] = b;
    r[2] = c;
    r[3] = d;
    return 1;
}

static int simd_splat(lua_State* L)
{
    uint32_t v = luaL_checkunsigned(L, 1);

    uint32_t* r = lua_newsimd(L);
    r[0] = v;
    r[1] = v;
    r[2] = v;
    r[3] = v;
    return 1;
}

// Read a SIMD lane index, rejecting non-finite and out-of-range values uniformly across platforms (avoids the
// undefined behavior of (int)(double) when the index is NaN or out of int range).
static int simdLaneIndex(lua_State* L, int idx, int lanes)
{
    double i = luaL_checknumber(L, idx);
    luaL_argcheck(L, i >= 0.0 && i < double(lanes), idx, lanes == 4 ? "lane index out of range [0, 3]" : "lane index out of range [0, 7]");
    return int(i);
}

static int simd_extract(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    int i = simdLaneIndex(L, 2, 4);

    lua_pushunsigned(L, v[i]);
    return 1;
}

#define SIMD_BINOP(name, expr) \
    static int simd_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 4; i++) \
        { \
            uint32_t x = a[i], y = b[i]; \
            r[i] = (expr); \
        } \
        return 1; \
    }

SIMD_BINOP(add, x + y)
SIMD_BINOP(sub, x - y)
SIMD_BINOP(mul, x * y)
SIMD_BINOP(band, x & y)
SIMD_BINOP(bor, x | y)
SIMD_BINOP(bxor, x ^ y)

#undef SIMD_BINOP

static int simd_bnot(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = ~a[i];
    return 1;
}

// Read a SIMD shift count. Reading via a double avoids the undefined behavior of (int)(double) when the count is
// NaN or out of int range (which also diverged between x86 and ARM). Any count outside [0, 31], including non-finite,
// is normalized to 32, which the shl/shr ops treat as "every bit shifted out" -> 0.
static unsigned simdShiftCount(lua_State* L, int idx)
{
    double n = luaL_checknumber(L, idx);
    return (n >= 0.0 && n < 32.0) ? unsigned(n) : 32u;
}

// Read a SIMD rotate count, taken modulo 32. Finite, in-int32-range counts keep their two's-complement modulo-32
// meaning (so a negative count rotates the other way); a non-finite or out-of-range count means no rotation.
static unsigned simdRotateCount(lua_State* L, int idx)
{
    double n = luaL_checknumber(L, idx);
    return (n == n && n >= -2147483648.0 && n < 2147483648.0) ? (unsigned(int(n)) & 31u) : 0u;
}

// Read a SIMD shuffle/permute selector, rejecting non-finite and out-of-range values uniformly across platforms.
static int simdShuffleControl(lua_State* L, int idx)
{
    double c = luaL_checknumber(L, idx);
    luaL_argcheck(L, c >= 0.0 && c < 256.0, idx, "shuffle selector out of range [0, 255]");
    return int(c);
}

static int simd_shl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    unsigned n = simdShiftCount(L, 2);

    uint32_t* r = lua_newsimd(L);
    // a count outside [0, 31] shifts every bit out and returns 0, matching the hardware shift and bit32
    for (int i = 0; i < 4; i++)
        r[i] = n < 32 ? a[i] << n : 0;
    return 1;
}

static int simd_shr(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    unsigned n = simdShiftCount(L, 2);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = n < 32 ? a[i] >> n : 0;
    return 1;
}

static int simd_rotl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    unsigned n = simdRotateCount(L, 2);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = n == 0 ? a[i] : (a[i] << n) | (a[i] >> (32 - n));
    return 1;
}

// Float lane operations reinterpret the four 32-bit lanes as IEEE-754 single-precision values.
// 'fmin'/'fmax' match the packed SSE min/max semantics ((x op y) ? x : y), not C fmin/fmax NaN handling.
#define SIMD_FBINOP(name, op) \
    static int simd_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 4; i++) \
        { \
            float x, y, z; \
            memcpy(&x, &a[i], sizeof(float)); \
            memcpy(&y, &b[i], sizeof(float)); \
            z = (op); \
            memcpy(&r[i], &z, sizeof(float)); \
        } \
        return 1; \
    }

SIMD_FBINOP(fadd, x + y)
SIMD_FBINOP(fsub, x - y)
SIMD_FBINOP(fmul, x* y)
SIMD_FBINOP(fdiv, x / y)
SIMD_FBINOP(fmin, x < y ? x : y)
SIMD_FBINOP(fmax, x > y ? x : y)

#undef SIMD_FBINOP

static int simd_fsqrt(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
    {
        float x, z;
        memcpy(&x, &a[i], sizeof(float));
        z = sqrtf(x);
        memcpy(&r[i], &z, sizeof(float));
    }
    return 1;
}

static int simd_fma(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    const uint32_t* b = luaL_checksimd(L, 2);
    const uint32_t* c = luaL_checksimd(L, 3);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
    {
        float x, y, w, z;
        memcpy(&x, &a[i], sizeof(float));
        memcpy(&y, &b[i], sizeof(float));
        memcpy(&w, &c[i], sizeof(float));
        // fused multiply-add: matches vfmadd213ps (single rounding) on FMA3 hardware
        z = fmaf(x, y, w);
        memcpy(&r[i], &z, sizeof(float));
    }
    return 1;
}

static int simd_tofloat(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
    {
        int32_t x;
        memcpy(&x, &a[i], sizeof(int32_t));
        float z = (float)x;
        memcpy(&r[i], &z, sizeof(float));
    }
    return 1;
}

static int simd_toint(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
    {
        float x;
        memcpy(&x, &a[i], sizeof(float));
        // Saturating truncate toward zero, matching ARM fcvtzs and WASM trunc_sat_s: NaN becomes 0 and out-of-range
        // values clamp to the signed 32-bit ends. (Bare x86 cvttps2dq returns 0x80000000 for all of these, so the
        // x64 native lowering fixes it up to agree with this.)
        int32_t z;
        if (x != x)
            z = 0;
        else if (x >= 2147483648.0f)
            z = 2147483647;
        else if (x < -2147483648.0f)
            z = (int32_t)0x80000000;
        else
            z = (int32_t)x;
        memcpy(&r[i], &z, sizeof(int32_t));
    }
    return 1;
}

static int simd_shuffle(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int control = simdShuffleControl(L, 2);

    // vpshufd semantics: lane i of the result is lane ((control >> (i*2)) & 3) of the source
    uint32_t src[4];
    for (int i = 0; i < 4; i++)
        src[i] = a[i];

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = src[(control >> (i * 2)) & 3];
    return 1;
}

static int simd_fcreate(lua_State* L)
{
    // validate all arguments before allocating, so a missing argument reports the right index instead of seeing
    // the freshly pushed result box in its stack slot
    float f[4];
    for (int i = 0; i < 4; i++)
        f[i] = (float)luaL_checknumber(L, i + 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        memcpy(&r[i], &f[i], sizeof(float));
    return 1;
}

static int simd_fextract(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    int i = simdLaneIndex(L, 2, 4);

    float f;
    memcpy(&f, &v[i], sizeof(float));
    lua_pushnumber(L, f);
    return 1;
}

static int simd_fsplat(lua_State* L)
{
    float f = (float)luaL_checknumber(L, 1);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = bits;
    return 1;
}

// 8-wide (256-bit) interpreter implementations: the same integer lane ops over all 8 lanes. The boxed
// value uses all 8 lanes of the shared 32-byte object. Native code keeps these in ymm registers.
#define SIMD256_BINOP(name, expr) \
    static int simd256_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 8; i++) \
        { \
            uint32_t x = a[i], y = b[i]; \
            r[i] = (expr); \
        } \
        return 1; \
    }

SIMD256_BINOP(add, x + y)
SIMD256_BINOP(sub, x - y)
SIMD256_BINOP(mul, x * y)
SIMD256_BINOP(band, x & y)
SIMD256_BINOP(bor, x | y)
SIMD256_BINOP(bxor, x ^ y)

#undef SIMD256_BINOP

static int simd256_bnot(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = ~a[i];
    return 1;
}

static int simd256_shl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    unsigned n = simdShiftCount(L, 2);
    uint32_t* r = lua_newsimd(L);
    // a count outside [0, 31] shifts every bit out and returns 0, matching the hardware shift and bit32
    for (int i = 0; i < 8; i++)
        r[i] = n < 32 ? a[i] << n : 0;
    return 1;
}

static int simd256_shr(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    unsigned n = simdShiftCount(L, 2);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = n < 32 ? a[i] >> n : 0;
    return 1;
}

static int simd256_rotl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    unsigned n = simdRotateCount(L, 2);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = n == 0 ? a[i] : (a[i] << n) | (a[i] >> (32 - n));
    return 1;
}

static int simd256_shuffle(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int control = simdShuffleControl(L, 2);

    // vpshufd on ymm shuffles within each 128-bit lane independently using the same control byte
    uint32_t src[8];
    for (int i = 0; i < 8; i++)
        src[i] = a[i];

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
    {
        unsigned sel = (control >> (i * 2)) & 3;
        r[i] = src[sel];
        r[i + 4] = src[4 + sel];
    }
    return 1;
}

#define SIMD256_FBINOP(name, op) \
    static int simd256_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 8; i++) \
        { \
            float x, y, z; \
            memcpy(&x, &a[i], sizeof(float)); \
            memcpy(&y, &b[i], sizeof(float)); \
            z = (op); \
            memcpy(&r[i], &z, sizeof(float)); \
        } \
        return 1; \
    }

SIMD256_FBINOP(fadd, x + y)
SIMD256_FBINOP(fsub, x - y)
SIMD256_FBINOP(fmul, x* y)
SIMD256_FBINOP(fdiv, x / y)
SIMD256_FBINOP(fmin, x < y ? x : y)
SIMD256_FBINOP(fmax, x > y ? x : y)

#undef SIMD256_FBINOP

static int simd256_fsqrt(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
    {
        float x, z;
        memcpy(&x, &a[i], sizeof(float));
        z = sqrtf(x);
        memcpy(&r[i], &z, sizeof(float));
    }
    return 1;
}

static int simd256_fma(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    const uint32_t* b = luaL_checksimd(L, 2);
    const uint32_t* c = luaL_checksimd(L, 3);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
    {
        float x, y, w, z;
        memcpy(&x, &a[i], sizeof(float));
        memcpy(&y, &b[i], sizeof(float));
        memcpy(&w, &c[i], sizeof(float));
        z = fmaf(x, y, w);
        memcpy(&r[i], &z, sizeof(float));
    }
    return 1;
}

static int simd256_tofloat(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
    {
        int32_t x;
        memcpy(&x, &a[i], sizeof(int32_t));
        float z = (float)x;
        memcpy(&r[i], &z, sizeof(float));
    }
    return 1;
}

static int simd256_toint(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
    {
        float x;
        memcpy(&x, &a[i], sizeof(float));
        // saturating truncate toward zero (see simd_toint): NaN -> 0, out-of-range clamps to the signed 32-bit ends
        int32_t z;
        if (x != x)
            z = 0;
        else if (x >= 2147483648.0f)
            z = 2147483647;
        else if (x < -2147483648.0f)
            z = (int32_t)0x80000000;
        else
            z = (int32_t)x;
        memcpy(&r[i], &z, sizeof(int32_t));
    }
    return 1;
}

static int simd256_create(lua_State* L)
{
    // validate all arguments before allocating (see simd_fcreate)
    uint32_t v[8];
    for (int i = 0; i < 8; i++)
        v[i] = luaL_checkunsigned(L, i + 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = v[i];
    return 1;
}

static int simd256_splat(lua_State* L)
{
    uint32_t v = luaL_checkunsigned(L, 1);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = v;
    return 1;
}

static int simd256_extract(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    int i = simdLaneIndex(L, 2, 8);

    lua_pushunsigned(L, v[i]);
    return 1;
}

static int simd256_fcreate(lua_State* L)
{
    // validate all arguments before allocating (see simd_fcreate)
    float f[8];
    for (int i = 0; i < 8; i++)
        f[i] = (float)luaL_checknumber(L, i + 1);

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        memcpy(&r[i], &f[i], sizeof(float));
    return 1;
}

static int simd256_fsplat(lua_State* L)
{
    float f = (float)luaL_checknumber(L, 1);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = bits;
    return 1;
}

static int simd256_fextract(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    int i = simdLaneIndex(L, 2, 8);

    float f;
    memcpy(&f, &v[i], sizeof(float));
    lua_pushnumber(L, f);
    return 1;
}

// Extract every lane at once, returning them as a tuple, so a value does not have to be taken apart with a
// separate extract call per lane.
static int simd_unpack(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    for (int i = 0; i < 4; i++)
        lua_pushunsigned(L, v[i]);
    return 4;
}

static int simd_funpack(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    for (int i = 0; i < 4; i++)
    {
        float f;
        memcpy(&f, &v[i], sizeof(float));
        lua_pushnumber(L, f);
    }
    return 4;
}

static int simd256_unpack(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    for (int i = 0; i < 8; i++)
        lua_pushunsigned(L, v[i]);
    return 8;
}

static int simd256_funpack(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    for (int i = 0; i < 8; i++)
    {
        float f;
        memcpy(&f, &v[i], sizeof(float));
        lua_pushnumber(L, f);
    }
    return 8;
}

// Float lane compares produce an all-ones / all-zero mask per lane (ordered: false for any NaN operand),
// matching the native vcmpps / fcm result. The masks feed select() or band/bor/bnot.
#define SIMD_FCMP(name, cmp) \
    static int simd_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 4; i++) \
        { \
            float x, y; \
            memcpy(&x, &a[i], sizeof(float)); \
            memcpy(&y, &b[i], sizeof(float)); \
            r[i] = (cmp) ? 0xFFFFFFFFu : 0u; \
        } \
        return 1; \
    }

SIMD_FCMP(feq, x == y)
SIMD_FCMP(flt, x < y)
SIMD_FCMP(fgt, x > y)

#undef SIMD_FCMP

#define SIMD256_FCMP(name, cmp) \
    static int simd256_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 8; i++) \
        { \
            float x, y; \
            memcpy(&x, &a[i], sizeof(float)); \
            memcpy(&y, &b[i], sizeof(float)); \
            r[i] = (cmp) ? 0xFFFFFFFFu : 0u; \
        } \
        return 1; \
    }

SIMD256_FCMP(feq, x == y)
SIMD256_FCMP(flt, x < y)
SIMD256_FCMP(fgt, x > y)

#undef SIMD256_FCMP

// Bitwise blend: per bit, take a where the mask bit is set, b where it is clear. With a full-lane mask (from a
// compare) this is a lanewise select(mask, a, b) = mask ? a : b. Element-agnostic, so it works for int or float lanes.
static int simd_select(lua_State* L)
{
    const uint32_t* m = luaL_checksimd(L, 1);
    const uint32_t* a = luaL_checksimd(L, 2);
    const uint32_t* b = luaL_checksimd(L, 3);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = (a[i] & m[i]) | (b[i] & ~m[i]);
    return 1;
}

static int simd256_select(lua_State* L)
{
    const uint32_t* m = luaL_checksimd(L, 1);
    const uint32_t* a = luaL_checksimd(L, 2);
    const uint32_t* b = luaL_checksimd(L, 3);
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = (a[i] & m[i]) | (b[i] & ~m[i]);
    return 1;
}

// Integer lane compares producing an all-ones / all-zero mask. Ordering is UNSIGNED (uint32_t < / > in C),
// matching the native cmhi / sign-bias path; equality is sign-agnostic.
#define SIMD_ICMP(name, cmp) \
    static int simd_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 4; i++) \
            r[i] = (a[i] cmp b[i]) ? 0xFFFFFFFFu : 0u; \
        return 1; \
    }

SIMD_ICMP(ieq, ==)
SIMD_ICMP(ilt, <)
SIMD_ICMP(igt, >)

#undef SIMD_ICMP

#define SIMD256_ICMP(name, cmp) \
    static int simd256_##name(lua_State* L) \
    { \
        const uint32_t* a = luaL_checksimd(L, 1); \
        const uint32_t* b = luaL_checksimd(L, 2); \
        uint32_t* r = lua_newsimd(L); \
        for (int i = 0; i < 8; i++) \
            r[i] = (a[i] cmp b[i]) ? 0xFFFFFFFFu : 0u; \
        return 1; \
    }

SIMD256_ICMP(ieq, ==)
SIMD256_ICMP(ilt, <)
SIMD256_ICMP(igt, >)

#undef SIMD256_ICMP

// Horizontal reductions: collapse every lane to a single scalar number. This is normally the END of a SIMD
// pipeline (accumulate lanewise across the data in registers, then reduce once), so a boxing lib call here is
// cheap relative to the work it summarizes. Integer min/max are UNSIGNED; bitwise reductions pair with the compare
// masks (e.g. hbor(eq(a,b)) != 0 = "any lane equal"). Float reductions accumulate left-to-right in single precision.
#define SIMD_IREDUCE(name, init, op) \
    static int simd_##name(lua_State* L) \
    { \
        const uint32_t* v = luaL_checksimd(L, 1); \
        uint32_t a = (init); \
        for (int i = 0; i < 4; i++) \
        { \
            uint32_t x = v[i]; \
            a = (op); \
        } \
        lua_pushunsigned(L, a); \
        return 1; \
    }

#define SIMD256_IREDUCE(name, init, op) \
    static int simd256_##name(lua_State* L) \
    { \
        const uint32_t* v = luaL_checksimd(L, 1); \
        uint32_t a = (init); \
        for (int i = 0; i < 8; i++) \
        { \
            uint32_t x = v[i]; \
            a = (op); \
        } \
        lua_pushunsigned(L, a); \
        return 1; \
    }

SIMD_IREDUCE(sum, 0u, a + x)
SIMD_IREDUCE(hmin, 0xFFFFFFFFu, x < a ? x : a)
SIMD_IREDUCE(hmax, 0u, x > a ? x : a)
SIMD_IREDUCE(hband, 0xFFFFFFFFu, a & x)
SIMD_IREDUCE(hbor, 0u, a | x)
SIMD_IREDUCE(hbxor, 0u, a ^ x)
SIMD256_IREDUCE(sum, 0u, a + x)
SIMD256_IREDUCE(hmin, 0xFFFFFFFFu, x < a ? x : a)
SIMD256_IREDUCE(hmax, 0u, x > a ? x : a)
SIMD256_IREDUCE(hband, 0xFFFFFFFFu, a & x)
SIMD256_IREDUCE(hbor, 0u, a | x)
SIMD256_IREDUCE(hbxor, 0u, a ^ x)

#undef SIMD_IREDUCE
#undef SIMD256_IREDUCE

#define SIMD_FREDUCE(name, init, op) \
    static int simd_##name(lua_State* L) \
    { \
        const uint32_t* v = luaL_checksimd(L, 1); \
        float a = (init); \
        for (int i = 0; i < 4; i++) \
        { \
            float x; \
            memcpy(&x, &v[i], sizeof(float)); \
            a = (op); \
        } \
        lua_pushnumber(L, a); \
        return 1; \
    }

#define SIMD256_FREDUCE(name, init, op) \
    static int simd256_##name(lua_State* L) \
    { \
        const uint32_t* v = luaL_checksimd(L, 1); \
        float a = (init); \
        for (int i = 0; i < 8; i++) \
        { \
            float x; \
            memcpy(&x, &v[i], sizeof(float)); \
            a = (op); \
        } \
        lua_pushnumber(L, a); \
        return 1; \
    }

SIMD_FREDUCE(fsum, 0.0f, a + x)
SIMD_FREDUCE(fhmin, INFINITY, x < a ? x : a)
SIMD_FREDUCE(fhmax, -INFINITY, x > a ? x : a)
SIMD256_FREDUCE(fsum, 0.0f, a + x)
SIMD256_FREDUCE(fhmin, INFINITY, x < a ? x : a)
SIMD256_FREDUCE(fhmax, -INFINITY, x > a ? x : a)

#undef SIMD_FREDUCE
#undef SIMD256_FREDUCE

static const luaL_Reg simd256lib[] = {
    {"create", simd256_create},
    {"splat", simd256_splat},
    {"fsplat", simd256_fsplat},
    {"extract", simd256_extract},
    {"unpack", simd256_unpack},
    {"feq", simd256_feq},
    {"flt", simd256_flt},
    {"fgt", simd256_fgt},
    {"select", simd256_select},
    {"eq", simd256_ieq},
    {"lt", simd256_ilt},
    {"gt", simd256_igt},
    {"sum", simd256_sum},
    {"hmin", simd256_hmin},
    {"hmax", simd256_hmax},
    {"hband", simd256_hband},
    {"hbor", simd256_hbor},
    {"hbxor", simd256_hbxor},
    {"fsum", simd256_fsum},
    {"fhmin", simd256_fhmin},
    {"fhmax", simd256_fhmax},
    {"add", simd256_add},
    {"sub", simd256_sub},
    {"mul", simd256_mul},
    {"band", simd256_band},
    {"bor", simd256_bor},
    {"bxor", simd256_bxor},
    {"bnot", simd256_bnot},
    {"shl", simd256_shl},
    {"shr", simd256_shr},
    {"rotl", simd256_rotl},
    {"shuffle", simd256_shuffle},
    {"fadd", simd256_fadd},
    {"fsub", simd256_fsub},
    {"fmul", simd256_fmul},
    {"fdiv", simd256_fdiv},
    {"fmin", simd256_fmin},
    {"fmax", simd256_fmax},
    {"fsqrt", simd256_fsqrt},
    {"fma", simd256_fma},
    {"tofloat", simd256_tofloat},
    {"toint", simd256_toint},
    {NULL, NULL},
};

static const luaL_Reg simdlib[] = {
    {"create", simd_create},
    {"splat", simd_splat},
    {"fsplat", simd_fsplat},
    {"extract", simd_extract},
    {"unpack", simd_unpack},
    {"add", simd_add},
    {"sub", simd_sub},
    {"mul", simd_mul},
    {"band", simd_band},
    {"bor", simd_bor},
    {"bxor", simd_bxor},
    {"bnot", simd_bnot},
    {"shl", simd_shl},
    {"shr", simd_shr},
    {"rotl", simd_rotl},
    {"fadd", simd_fadd},
    {"fsub", simd_fsub},
    {"fmul", simd_fmul},
    {"fdiv", simd_fdiv},
    {"fmin", simd_fmin},
    {"fmax", simd_fmax},
    {"fsqrt", simd_fsqrt},
    {"fma", simd_fma},
    {"tofloat", simd_tofloat},
    {"toint", simd_toint},
    {"shuffle", simd_shuffle},
    {"fcreate", simd_fcreate},
    {"fextract", simd_fextract},
    {"funpack", simd_funpack},
    {"feq", simd_feq},
    {"flt", simd_flt},
    {"fgt", simd_fgt},
    {"select", simd_select},
    {"eq", simd_ieq},
    {"lt", simd_ilt},
    {"gt", simd_igt},
    {"sum", simd_sum},
    {"hmin", simd_hmin},
    {"hmax", simd_hmax},
    {"hband", simd_hband},
    {"hbor", simd_hbor},
    {"hbxor", simd_hbxor},
    {"fsum", simd_fsum},
    {"fhmin", simd_fhmin},
    {"fhmax", simd_fhmax},
    {NULL, NULL},
};

// Element-typed namespaces matching the buffer suffixes (u32x4/f32x4/u32x8/f32x8). These reuse the same boxed
// values and C implementations as simd/simd256; the element type is in the namespace, so the float ops drop the
// 'f' prefix (f32x4.add, not simd.fadd). simd/simd256 stay registered as deprecated aliases.
static const luaL_Reg u32x4lib[] = {
    {"create", simd_create},
    {"splat", simd_splat},
    {"extract", simd_extract},
    {"unpack", simd_unpack},
    {"select", simd_select},
    {"eq", simd_ieq},
    {"lt", simd_ilt},
    {"gt", simd_igt},
    {"sum", simd_sum},
    {"hmin", simd_hmin},
    {"hmax", simd_hmax},
    {"hband", simd_hband},
    {"hbor", simd_hbor},
    {"hbxor", simd_hbxor},
    {"add", simd_add},
    {"sub", simd_sub},
    {"mul", simd_mul},
    {"band", simd_band},
    {"bor", simd_bor},
    {"bxor", simd_bxor},
    {"bnot", simd_bnot},
    {"shl", simd_shl},
    {"shr", simd_shr},
    {"rotl", simd_rotl},
    {"shuffle", simd_shuffle},
    {"tofloat", simd_tofloat},
    {NULL, NULL},
};

static const luaL_Reg f32x4lib[] = {
    {"create", simd_fcreate},
    {"splat", simd_fsplat},
    {"extract", simd_fextract},
    {"unpack", simd_funpack},
    {"eq", simd_feq},
    {"lt", simd_flt},
    {"gt", simd_fgt},
    {"sum", simd_fsum},
    {"hmin", simd_fhmin},
    {"hmax", simd_fhmax},
    {"select", simd_select},
    {"add", simd_fadd},
    {"sub", simd_fsub},
    {"mul", simd_fmul},
    {"div", simd_fdiv},
    {"min", simd_fmin},
    {"max", simd_fmax},
    {"sqrt", simd_fsqrt},
    {"fma", simd_fma},
    {"toint", simd_toint},
    {NULL, NULL},
};

static const luaL_Reg u32x8lib[] = {
    {"create", simd256_create},
    {"splat", simd256_splat},
    {"extract", simd256_extract},
    {"unpack", simd256_unpack},
    {"select", simd256_select},
    {"eq", simd256_ieq},
    {"lt", simd256_ilt},
    {"gt", simd256_igt},
    {"sum", simd256_sum},
    {"hmin", simd256_hmin},
    {"hmax", simd256_hmax},
    {"hband", simd256_hband},
    {"hbor", simd256_hbor},
    {"hbxor", simd256_hbxor},
    {"add", simd256_add},
    {"sub", simd256_sub},
    {"mul", simd256_mul},
    {"band", simd256_band},
    {"bor", simd256_bor},
    {"bxor", simd256_bxor},
    {"bnot", simd256_bnot},
    {"shl", simd256_shl},
    {"shr", simd256_shr},
    {"rotl", simd256_rotl},
    {"shuffle", simd256_shuffle},
    {"tofloat", simd256_tofloat},
    {NULL, NULL},
};

static const luaL_Reg f32x8lib[] = {
    {"create", simd256_fcreate},
    {"splat", simd256_fsplat},
    {"extract", simd256_fextract},
    {"unpack", simd256_funpack},
    {"eq", simd256_feq},
    {"lt", simd256_flt},
    {"gt", simd256_fgt},
    {"sum", simd256_fsum},
    {"hmin", simd256_fhmin},
    {"hmax", simd256_fhmax},
    {"select", simd256_select},
    {"add", simd256_fadd},
    {"sub", simd256_fsub},
    {"mul", simd256_fmul},
    {"div", simd256_fdiv},
    {"min", simd256_fmin},
    {"max", simd256_fmax},
    {"sqrt", simd256_fsqrt},
    {"fma", simd256_fma},
    {"toint", simd256_toint},
    {NULL, NULL},
};

int luaopen_simd(lua_State* L)
{
    // element-typed namespaces (preferred)
    luaL_register(L, "u32x4", u32x4lib);
    lua_pop(L, 1);
    luaL_register(L, "f32x4", f32x4lib);
    lua_pop(L, 1);
    luaL_register(L, "u32x8", u32x8lib);
    lua_pop(L, 1);
    luaL_register(L, "f32x8", f32x8lib);
    lua_pop(L, 1);

    // deprecated aliases
    luaL_register(L, "simd256", simd256lib);
    lua_pop(L, 1);

    luaL_register(L, LUA_SIMDLIBNAME, simdlib);
    return 1;
}
