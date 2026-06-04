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

static int simd_extract(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    int i = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(i) < 4, 2, "lane index out of range [0, 3]");

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

static int simd_shl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(n) < 32, 2, "shift count out of range [0, 31]");

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = a[i] << n;
    return 1;
}

static int simd_shr(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(n) < 32, 2, "shift count out of range [0, 31]");

    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
        r[i] = a[i] >> n;
    return 1;
}

static int simd_rotl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(n) < 32, 2, "rotate count out of range [0, 31]");

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
        // Match vcvttps2dq: truncate toward zero; NaN or out-of-range lanes become 0x80000000
        int32_t z;
        if (x >= -2147483648.0f && x < 2147483648.0f)
            z = (int32_t)x;
        else
            z = (int32_t)0x80000000;
        memcpy(&r[i], &z, sizeof(int32_t));
    }
    return 1;
}

static int simd_shuffle(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int control = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(control) < 256, 2, "shuffle selector out of range [0, 255]");

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
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 4; i++)
    {
        float f = (float)luaL_checknumber(L, i + 1);
        memcpy(&r[i], &f, sizeof(float));
    }
    return 1;
}

static int simd_fextract(lua_State* L)
{
    const uint32_t* v = luaL_checksimd(L, 1);
    int i = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(i) < 4, 2, "lane index out of range [0, 3]");

    float f;
    memcpy(&f, &v[i], sizeof(float));
    lua_pushnumber(L, f);
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
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(n) < 32, 2, "shift count out of range [0, 31]");
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = a[i] << n;
    return 1;
}

static int simd256_shr(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(n) < 32, 2, "shift count out of range [0, 31]");
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = a[i] >> n;
    return 1;
}

static int simd256_rotl(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(n) < 32, 2, "rotate count out of range [0, 31]");
    uint32_t* r = lua_newsimd(L);
    for (int i = 0; i < 8; i++)
        r[i] = n == 0 ? a[i] : (a[i] << n) | (a[i] >> (32 - n));
    return 1;
}

static int simd256_shuffle(lua_State* L)
{
    const uint32_t* a = luaL_checksimd(L, 1);
    int control = luaL_checkinteger(L, 2);
    luaL_argcheck(L, unsigned(control) < 256, 2, "shuffle selector out of range [0, 255]");

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

static const luaL_Reg simd256lib[] = {
    {"add", simd256_add},
    {"sub", simd256_sub},
    {"band", simd256_band},
    {"bor", simd256_bor},
    {"bxor", simd256_bxor},
    {"bnot", simd256_bnot},
    {"shl", simd256_shl},
    {"shr", simd256_shr},
    {"rotl", simd256_rotl},
    {"shuffle", simd256_shuffle},
    {NULL, NULL},
};

static const luaL_Reg simdlib[] = {
    {"create", simd_create},
    {"splat", simd_splat},
    {"extract", simd_extract},
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
    {NULL, NULL},
};

int luaopen_simd(lua_State* L)
{
    luaL_register(L, LUA_SIMDLIBNAME, simdlib);

    luaL_register(L, "simd256", simd256lib);
    lua_pop(L, 1); // leave only the simd table as the result; simd256 stays registered as a global

    return 1;
}
