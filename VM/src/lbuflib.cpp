// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lualib.h"

#include "lcommon.h"
#include "lbuffer.h"

#if defined(LUAU_BIG_ENDIAN)
#include <endian.h>
#endif

LUAU_FASTFLAG(LuauIntegerLibrary)

#include <math.h>
#include <string.h>

#ifdef LUAU_TARGET_AVX2
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

// while C API returns 'size_t' for binary compatibility in case of future extensions,
// in the current implementation, length and offset are limited to 31 bits
// because offset is limited to an integer, a single 64bit comparison can be used and will not overflow
#define isoutofbounds(offset, len, accessize) (uint64_t(unsigned(offset)) + (accessize) > uint64_t(len))

static_assert(MAX_BUFFER_SIZE <= INT_MAX, "current implementation can't handle a larger limit");

#if defined(LUAU_BIG_ENDIAN)
template<typename T>
inline T buffer_swapbe(T v)
{
    if (sizeof(T) == 8)
        return htole64(v);
    else if (sizeof(T) == 4)
        return htole32(v);
    else if (sizeof(T) == 2)
        return htole16(v);
    else
        return v;
}
#endif

static int buffer_create(lua_State* L)
{
    int size = luaL_checkinteger(L, 1);

    luaL_argcheck(L, size >= 0, 1, "size");

    lua_newbuffer(L, size);
    return 1;
}

static int buffer_fromstring(lua_State* L)
{
    size_t len = 0;
    const char* val = luaL_checklstring(L, 1, &len);

    void* data = lua_newbuffer(L, len);
    memcpy(data, val, len);
    return 1;
}

static int buffer_tostring(lua_State* L)
{
    size_t len = 0;
    void* data = luaL_checkbuffer(L, 1, &len);

    lua_pushlstring(L, (char*)data, len);
    return 1;
}

template<typename T>
static int buffer_readinteger(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    if (isoutofbounds(offset, len, sizeof(T)))
        luaL_error(L, "buffer access out of bounds");

    T val;
    memcpy(&val, (char*)buf + offset, sizeof(T));

#if defined(LUAU_BIG_ENDIAN)
    val = buffer_swapbe(val);
#endif

    lua_pushnumber(L, double(val));
    return 1;
}

template<typename T>
static int buffer_writeinteger(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    int value = luaL_checkunsigned(L, 3);

    if (isoutofbounds(offset, len, sizeof(T)))
        luaL_error(L, "buffer access out of bounds");

    T val = T(value);

#if defined(LUAU_BIG_ENDIAN)
    val = buffer_swapbe(val);
#endif

    memcpy((char*)buf + offset, &val, sizeof(T));
    return 0;
}

static int buffer_readlong(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    if (isoutofbounds(offset, len, sizeof(uint64_t)))
        luaL_error(L, "buffer access out of bounds");

    int64_t val;
    memcpy(&val, (char*)buf + offset, sizeof(int64_t));

#if defined(LUAU_BIG_ENDIAN)
    val = buffer_swapbe(val);
#endif

    lua_pushinteger64(L, val);
    return 1;
}

static int buffer_writelong(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    int64_t value = luaL_checkinteger64(L, 3);

    if (isoutofbounds(offset, len, sizeof(int64_t)))
        luaL_error(L, "buffer access out of bounds");

#if defined(LUAU_BIG_ENDIAN)
    value = buffer_swapbe(value);
#endif

    memcpy((char*)buf + offset, &value, sizeof(int64_t));
    return 0;
}

template<typename T, typename StorageType>
static int buffer_readfp(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    if (isoutofbounds(offset, len, sizeof(T)))
        luaL_error(L, "buffer access out of bounds");

    T val;

#if defined(LUAU_BIG_ENDIAN)
    static_assert(sizeof(T) == sizeof(StorageType), "type size must match to reinterpret data");
    StorageType tmp;
    memcpy(&tmp, (char*)buf + offset, sizeof(tmp));
    tmp = buffer_swapbe(tmp);

    memcpy(&val, &tmp, sizeof(tmp));
#else
    memcpy(&val, (char*)buf + offset, sizeof(T));
#endif

    lua_pushnumber(L, double(val));
    return 1;
}

template<typename T, typename StorageType>
static int buffer_writefp(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    double value = luaL_checknumber(L, 3);

    if (isoutofbounds(offset, len, sizeof(T)))
        luaL_error(L, "buffer access out of bounds");

    T val = T(value);

#if defined(LUAU_BIG_ENDIAN)
    static_assert(sizeof(T) == sizeof(StorageType), "type size must match to reinterpret data");
    StorageType tmp;
    memcpy(&tmp, &val, sizeof(tmp));
    tmp = buffer_swapbe(tmp);

    memcpy((char*)buf + offset, &tmp, sizeof(tmp));
#else
    memcpy((char*)buf + offset, &val, sizeof(T));
#endif

    return 0;
}

static int buffer_readstring(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    int size = luaL_checkinteger(L, 3);

    luaL_argcheck(L, size >= 0, 3, "size");

    if (isoutofbounds(offset, len, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    lua_pushlstring(L, (char*)buf + offset, size);
    return 1;
}

static int buffer_writestring(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    size_t size = 0;
    const char* val = luaL_checklstring(L, 3, &size);
    int count = luaL_optinteger(L, 4, int(size));

    luaL_argcheck(L, count >= 0, 4, "count");

    if (size_t(count) > size)
        luaL_error(L, "string length overflow");

    // string size can't exceed INT_MAX at this point
    if (isoutofbounds(offset, len, unsigned(count)))
        luaL_error(L, "buffer access out of bounds");

    memcpy((char*)buf + offset, val, count);
    return 0;
}

static int buffer_len(lua_State* L)
{
    size_t len = 0;
    luaL_checkbuffer(L, 1, &len);

    lua_pushnumber(L, double(unsigned(len)));
    return 1;
}

static int buffer_copy(lua_State* L)
{
    size_t tlen = 0;
    void* tbuf = luaL_checkbuffer(L, 1, &tlen);
    int toffset = luaL_checkinteger(L, 2);

    size_t slen = 0;
    void* sbuf = luaL_checkbuffer(L, 3, &slen);
    int soffset = luaL_optinteger(L, 4, 0);

    int size = luaL_optinteger(L, 5, int(slen) - soffset);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(soffset, slen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(toffset, tlen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    memmove((char*)tbuf + toffset, (char*)sbuf + soffset, size);
    return 0;
}

static int buffer_fill(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    unsigned value = luaL_checkunsigned(L, 3);
    int size = luaL_optinteger(L, 4, int(len) - offset);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(offset, len, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    memset((char*)buf + offset, value & 0xff, size);
    return 0;
}

// Element-wise bitwise combinations of byte ranges, performed in place: dst[i] = op(dst[i], src[i]).
// The wide registers live only inside these kernels, the data stays in buffer memory. SSE2 is part of the
// x86-64 baseline and AVX2 is detected at runtime, with a portable scalar fallback that also handles the
// sub-register tail. Each operation is a small tag supplying the combine at every width.
//
// OVERLAP CONTRACT (applies to every in-place dst/src kernel below, including the integer, float and fma ops):
// the destination and source ranges must either be identical or not overlap at all. A partial overlap is
// unsupported and gives a result that depends on the chosen vector width (the kernels process front to back),
// the same restriction memcpy has versus memmove. Identical ranges (dst == src, same offset) are always fine.
#ifdef LUAU_TARGET_AVX2
static bool buffer_hasavx2()
{
#if defined(_MSC_VER)
    int info1[4] = {};
    __cpuid(info1, 1);

    // both OSXSAVE (bit 27) and AVX (bit 28) must be set before any YMM state can be touched
    if ((info1[2] & (1 << 27)) == 0 || (info1[2] & (1 << 28)) == 0)
        return false;

    // the OS must have enabled saving of XMM (bit 1) and YMM (bit 2) register state
    if ((_xgetbv(0) & 0x6) != 0x6)
        return false;

    int info7[4] = {};
    __cpuidex(info7, 7, 0);

    // AVX2 is reported in leaf 7, EBX bit 5
    return (info7[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

// FMA3 (fused multiply-add) support, checked independently of AVX2 so buffer.fma only takes the fused vfmadd path
// on hardware that has it. Callers also require buffer_hasavx2() before touching ymm state.
static bool buffer_hasfma3()
{
#if defined(_MSC_VER)
    int info1[4] = {};
    __cpuid(info1, 1);

    // FMA3 is reported in leaf 1, ECX bit 12
    return (info1[2] & (1 << 12)) != 0;
#else
    return __builtin_cpu_supports("fma");
#endif
}
#endif

struct buffer_op_xor
{
    static unsigned char byte(unsigned char a, unsigned char b) { return a ^ b; }
    static uint64_t word(uint64_t a, uint64_t b) { return a ^ b; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i a, __m256i b) { return _mm256_xor_si256(a, b); }
#endif
};

struct buffer_op_and
{
    static unsigned char byte(unsigned char a, unsigned char b) { return a & b; }
    static uint64_t word(uint64_t a, uint64_t b) { return a & b; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i a, __m128i b) { return _mm_and_si128(a, b); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i a, __m256i b) { return _mm256_and_si256(a, b); }
#endif
};

struct buffer_op_or
{
    static unsigned char byte(unsigned char a, unsigned char b) { return a | b; }
    static uint64_t word(uint64_t a, uint64_t b) { return a | b; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i a, __m128i b) { return _mm_or_si128(a, b); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i a, __m256i b) { return _mm256_or_si256(a, b); }
#endif
};

template<typename Op>
static void buffer_binop_scalar(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;

    for (; i + 8 <= len; i += 8)
    {
        uint64_t d, s;
        memcpy(&d, dst + i, 8);
        memcpy(&s, src + i, 8);
        d = Op::word(d, s);
        memcpy(dst + i, &d, 8);
    }

    for (; i < len; i++)
        dst[i] = char(Op::byte((unsigned char)dst[i], (unsigned char)src[i]));
}

#ifdef LUAU_TARGET_AVX2
template<typename Op>
LUAU_TARGET_AVX2 static void buffer_binop_avx2(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;

    for (; i + 32 <= len; i += 32)
    {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((const __m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), Op::wide(d, s));
    }

    buffer_binop_scalar<Op>(dst + i, src + i, len - i);
}

template<typename Op>
static void buffer_binop_sse2(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;

    for (; i + 16 <= len; i += 16)
    {
        __m128i d = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i s = _mm_loadu_si128((const __m128i*)(src + i));
        _mm_storeu_si128((__m128i*)(dst + i), Op::half(d, s));
    }

    buffer_binop_scalar<Op>(dst + i, src + i, len - i);
}
#endif

template<typename Op>
static void buffer_binop_bytes(char* dst, const char* src, unsigned len)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();

    if (hasavx2)
        buffer_binop_avx2<Op>(dst, src, len);
    else
        buffer_binop_sse2<Op>(dst, src, len);
#else
    buffer_binop_scalar<Op>(dst, src, len);
#endif
}

template<typename Op>
static int buffer_binop(lua_State* L)
{
    size_t tlen = 0;
    void* tbuf = luaL_checkbuffer(L, 1, &tlen);
    int toffset = luaL_checkinteger(L, 2);

    size_t slen = 0;
    void* sbuf = luaL_checkbuffer(L, 3, &slen);
    int soffset = luaL_optinteger(L, 4, 0);

    int size = luaL_optinteger(L, 5, int(slen) - soffset);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(soffset, slen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(toffset, tlen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    buffer_binop_bytes<Op>((char*)tbuf + toffset, (const char*)sbuf + soffset, unsigned(size));
    return 0;
}

// Bitwise complement in place: dst[i] = ~dst[i]. The wide path xors against an all-ones register.
static void buffer_bnot_scalar(char* dst, unsigned len)
{
    unsigned i = 0;

    for (; i + 8 <= len; i += 8)
    {
        uint64_t d;
        memcpy(&d, dst + i, 8);
        d = ~d;
        memcpy(dst + i, &d, 8);
    }

    for (; i < len; i++)
        dst[i] = char(~(unsigned char)dst[i]);
}

#ifdef LUAU_TARGET_AVX2
LUAU_TARGET_AVX2 static void buffer_bnot_avx2(char* dst, unsigned len)
{
    unsigned i = 0;
    __m256i ones = _mm256_set1_epi8((char)0xff);

    for (; i + 32 <= len; i += 32)
    {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_xor_si256(d, ones));
    }

    buffer_bnot_scalar(dst + i, len - i);
}

static void buffer_bnot_sse2(char* dst, unsigned len)
{
    unsigned i = 0;
    __m128i ones = _mm_set1_epi8((char)0xff);

    for (; i + 16 <= len; i += 16)
    {
        __m128i d = _mm_loadu_si128((const __m128i*)(dst + i));
        _mm_storeu_si128((__m128i*)(dst + i), _mm_xor_si128(d, ones));
    }

    buffer_bnot_scalar(dst + i, len - i);
}
#endif

static void buffer_bnot_bytes(char* dst, unsigned len)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();

    if (hasavx2)
        buffer_bnot_avx2(dst, len);
    else
        buffer_bnot_sse2(dst, len);
#else
    buffer_bnot_scalar(dst, len);
#endif
}

static int buffer_bnot(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    int size = luaL_optinteger(L, 3, int(len) - offset);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(offset, len, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    buffer_bnot_bytes((char*)buf + offset, unsigned(size));
    return 0;
}

// Lanewise 32-bit integer arithmetic in place: dst[i] = dst[i] OP src[i] over each u32 lane (with wraparound).
// Unlike the bitwise ops this cannot be done bytewise (carry/borrow crosses byte boundaries), so it has its own
// u32-granular kernels via an op tag. Multiply uses _mm_mullo_epi32 / _mm256_mullo_epi32 (SSE4.1 / AVX2).
struct buffer_iop_add
{
    static uint32_t lane(uint32_t a, uint32_t b) { return a + b; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i a, __m128i b) { return _mm_add_epi32(a, b); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }
#endif
};
struct buffer_iop_sub
{
    static uint32_t lane(uint32_t a, uint32_t b) { return a - b; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i a, __m128i b) { return _mm_sub_epi32(a, b); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i a, __m256i b) { return _mm256_sub_epi32(a, b); }
#endif
};
struct buffer_iop_mul
{
    static uint32_t lane(uint32_t a, uint32_t b) { return a * b; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i a, __m128i b) { return _mm_mullo_epi32(a, b); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i a, __m256i b) { return _mm256_mullo_epi32(a, b); }
#endif
};

template<typename Op>
static void buffer_ibinop_scalar(char* dst, const char* src, unsigned len)
{
    for (unsigned i = 0; i + 4 <= len; i += 4)
    {
        uint32_t d, s;
        memcpy(&d, dst + i, 4);
        memcpy(&s, src + i, 4);
        d = Op::lane(d, s);
        memcpy(dst + i, &d, 4);
    }
}

#ifdef LUAU_TARGET_AVX2
template<typename Op>
LUAU_TARGET_AVX2 static void buffer_ibinop_avx2(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;
    for (; i + 32 <= len; i += 32)
    {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((const __m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), Op::wide(d, s));
    }
    buffer_ibinop_scalar<Op>(dst + i, src + i, len - i);
}

template<typename Op>
static void buffer_ibinop_sse2(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;
    for (; i + 16 <= len; i += 16)
    {
        __m128i d = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i s = _mm_loadu_si128((const __m128i*)(src + i));
        _mm_storeu_si128((__m128i*)(dst + i), Op::half(d, s));
    }
    buffer_ibinop_scalar<Op>(dst + i, src + i, len - i);
}
#endif

template<typename Op>
static void buffer_ibinop_bytes(char* dst, const char* src, unsigned len)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();
    if (hasavx2)
        buffer_ibinop_avx2<Op>(dst, src, len);
    else
        buffer_ibinop_sse2<Op>(dst, src, len);
#else
    buffer_ibinop_scalar<Op>(dst, src, len);
#endif
}

template<typename Op>
static int buffer_ibinop(lua_State* L)
{
    size_t tlen = 0;
    void* tbuf = luaL_checkbuffer(L, 1, &tlen);
    int toffset = luaL_checkinteger(L, 2);

    size_t slen = 0;
    void* sbuf = luaL_checkbuffer(L, 3, &slen);
    int soffset = luaL_optinteger(L, 4, 0);

    int size = luaL_optinteger(L, 5, int(slen) - soffset);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");
    if (size % 4 != 0)
        luaL_error(L, "buffer integer op size must be a multiple of 4");
    if (isoutofbounds(soffset, slen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");
    if (isoutofbounds(toffset, tlen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    buffer_ibinop_bytes<Op>((char*)tbuf + toffset, (const char*)sbuf + soffset, unsigned(size));
    return 0;
}

// Lanewise 32-bit rotate-left in place by a compile-time-known count in [1, 31]: dst[i] = rotl(dst[i], n).
static void buffer_rotl_u32_scalar(char* dst, unsigned len, int n)
{
    unsigned i = 0;

    for (; i + 4 <= len; i += 4)
    {
        uint32_t d;
        memcpy(&d, dst + i, 4);
        d = (d << n) | (d >> (32 - n));
        memcpy(dst + i, &d, 4);
    }
}

#ifdef LUAU_TARGET_AVX2
LUAU_TARGET_AVX2 static void buffer_rotl_u32_avx2(char* dst, unsigned len, int n)
{
    unsigned i = 0;

    for (; i + 32 <= len; i += 32)
    {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i r = _mm256_or_si256(_mm256_slli_epi32(d, n), _mm256_srli_epi32(d, 32 - n));
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }

    buffer_rotl_u32_scalar(dst + i, len - i, n);
}

static void buffer_rotl_u32_sse2(char* dst, unsigned len, int n)
{
    unsigned i = 0;

    for (; i + 16 <= len; i += 16)
    {
        __m128i d = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i r = _mm_or_si128(_mm_slli_epi32(d, n), _mm_srli_epi32(d, 32 - n));
        _mm_storeu_si128((__m128i*)(dst + i), r);
    }

    buffer_rotl_u32_scalar(dst + i, len - i, n);
}
#endif

static void buffer_rotl_u32_bytes(char* dst, unsigned len, int n)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();

    if (hasavx2)
        buffer_rotl_u32_avx2(dst, len, n);
    else
        buffer_rotl_u32_sse2(dst, len, n);
#else
    buffer_rotl_u32_scalar(dst, len, n);
#endif
}

static int buffer_rotl(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    int n = luaL_checkinteger(L, 3);

    int size = luaL_optinteger(L, 4, int(len) - offset);

    if (n < 0 || n > 31)
        luaL_error(L, "buffer.rotl amount must be in [0, 31]");

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");

    if (size % 4 != 0)
        luaL_error(L, "buffer.rotl size must be a multiple of 4");

    if (isoutofbounds(offset, len, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    if (n != 0)
        buffer_rotl_u32_bytes((char*)buf + unsigned(offset), unsigned(size), n);
    return 0;
}

// Lanewise 32-bit FLOAT arithmetic in place: dst[i] = dst[i] OP src[i] over each f32 lane. Like the u32 add, the
// lanes are 4-byte granular (size must be a multiple of 4); the op tag supplies scalar/SSE/AVX2 float kernels.
struct buffer_fop_add
{
    static float lane(float a, float b) { return a + b; }
#ifdef LUAU_TARGET_AVX2
    static __m128 half(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
    LUAU_TARGET_AVX2 static __m256 wide(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
#endif
};
struct buffer_fop_sub
{
    static float lane(float a, float b) { return a - b; }
#ifdef LUAU_TARGET_AVX2
    static __m128 half(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
    LUAU_TARGET_AVX2 static __m256 wide(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
#endif
};
struct buffer_fop_mul
{
    static float lane(float a, float b) { return a * b; }
#ifdef LUAU_TARGET_AVX2
    static __m128 half(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
    LUAU_TARGET_AVX2 static __m256 wide(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
#endif
};
struct buffer_fop_div
{
    static float lane(float a, float b) { return a / b; }
#ifdef LUAU_TARGET_AVX2
    static __m128 half(__m128 a, __m128 b) { return _mm_div_ps(a, b); }
    LUAU_TARGET_AVX2 static __m256 wide(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
#endif
};
struct buffer_fop_min
{
    static float lane(float a, float b) { return a < b ? a : b; }
#ifdef LUAU_TARGET_AVX2
    static __m128 half(__m128 a, __m128 b) { return _mm_min_ps(a, b); }
    LUAU_TARGET_AVX2 static __m256 wide(__m256 a, __m256 b) { return _mm256_min_ps(a, b); }
#endif
};
struct buffer_fop_max
{
    static float lane(float a, float b) { return a > b ? a : b; }
#ifdef LUAU_TARGET_AVX2
    static __m128 half(__m128 a, __m128 b) { return _mm_max_ps(a, b); }
    LUAU_TARGET_AVX2 static __m256 wide(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
#endif
};

template<typename Op>
static void buffer_fbinop_scalar(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;
    for (; i + 4 <= len; i += 4)
    {
        float a, b;
        memcpy(&a, dst + i, 4);
        memcpy(&b, src + i, 4);
        a = Op::lane(a, b);
        memcpy(dst + i, &a, 4);
    }
}

#ifdef LUAU_TARGET_AVX2
template<typename Op>
LUAU_TARGET_AVX2 static void buffer_fbinop_avx2(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;
    for (; i + 32 <= len; i += 32)
    {
        __m256 d = _mm256_loadu_ps((const float*)(dst + i));
        __m256 s = _mm256_loadu_ps((const float*)(src + i));
        _mm256_storeu_ps((float*)(dst + i), Op::wide(d, s));
    }
    buffer_fbinop_scalar<Op>(dst + i, src + i, len - i);
}

template<typename Op>
static void buffer_fbinop_sse2(char* dst, const char* src, unsigned len)
{
    unsigned i = 0;
    for (; i + 16 <= len; i += 16)
    {
        __m128 d = _mm_loadu_ps((const float*)(dst + i));
        __m128 s = _mm_loadu_ps((const float*)(src + i));
        _mm_storeu_ps((float*)(dst + i), Op::half(d, s));
    }
    buffer_fbinop_scalar<Op>(dst + i, src + i, len - i);
}
#endif

template<typename Op>
static void buffer_fbinop_bytes(char* dst, const char* src, unsigned len)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();
    if (hasavx2)
        buffer_fbinop_avx2<Op>(dst, src, len);
    else
        buffer_fbinop_sse2<Op>(dst, src, len);
#else
    buffer_fbinop_scalar<Op>(dst, src, len);
#endif
}

template<typename Op>
static int buffer_fbinop(lua_State* L)
{
    size_t tlen = 0;
    void* tbuf = luaL_checkbuffer(L, 1, &tlen);
    int toffset = luaL_checkinteger(L, 2);

    size_t slen = 0;
    void* sbuf = luaL_checkbuffer(L, 3, &slen);
    int soffset = luaL_optinteger(L, 4, 0);

    int size = luaL_optinteger(L, 5, int(slen) - soffset);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");

    if (size % 4 != 0)
        luaL_error(L, "buffer float op size must be a multiple of 4");

    if (isoutofbounds(soffset, slen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    if (isoutofbounds(toffset, tlen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    buffer_fbinop_bytes<Op>((char*)tbuf + toffset, (const char*)sbuf + soffset, unsigned(size));
    return 0;
}

// Lanewise 32-bit logical shift in place by a runtime count in [0, 31]: dst[i] <<= n (shl) or >>= n (shr).
struct buffer_sop_shl
{
    static uint32_t lane(uint32_t x, int n) { return x << n; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i x, int n) { return _mm_slli_epi32(x, n); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i x, int n) { return _mm256_slli_epi32(x, n); }
#endif
};
struct buffer_sop_shr
{
    static uint32_t lane(uint32_t x, int n) { return x >> n; }
#ifdef LUAU_TARGET_AVX2
    static __m128i half(__m128i x, int n) { return _mm_srli_epi32(x, n); }
    LUAU_TARGET_AVX2 static __m256i wide(__m256i x, int n) { return _mm256_srli_epi32(x, n); }
#endif
};

template<typename Op>
static void buffer_sunop_scalar(char* dst, unsigned len, int n)
{
    for (unsigned i = 0; i + 4 <= len; i += 4)
    {
        uint32_t d;
        memcpy(&d, dst + i, 4);
        d = Op::lane(d, n);
        memcpy(dst + i, &d, 4);
    }
}

#ifdef LUAU_TARGET_AVX2
template<typename Op>
LUAU_TARGET_AVX2 static void buffer_sunop_avx2(char* dst, unsigned len, int n)
{
    unsigned i = 0;
    for (; i + 32 <= len; i += 32)
    {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        _mm256_storeu_si256((__m256i*)(dst + i), Op::wide(d, n));
    }
    buffer_sunop_scalar<Op>(dst + i, len - i, n);
}

template<typename Op>
static void buffer_sunop_sse2(char* dst, unsigned len, int n)
{
    unsigned i = 0;
    for (; i + 16 <= len; i += 16)
    {
        __m128i d = _mm_loadu_si128((const __m128i*)(dst + i));
        _mm_storeu_si128((__m128i*)(dst + i), Op::half(d, n));
    }
    buffer_sunop_scalar<Op>(dst + i, len - i, n);
}
#endif

template<typename Op>
static void buffer_sunop_bytes(char* dst, unsigned len, int n)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();
    if (hasavx2)
        buffer_sunop_avx2<Op>(dst, len, n);
    else
        buffer_sunop_sse2<Op>(dst, len, n);
#else
    buffer_sunop_scalar<Op>(dst, len, n);
#endif
}

template<typename Op>
static int buffer_sunop(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    int n = luaL_checkinteger(L, 3);

    int size = luaL_optinteger(L, 4, int(len) - offset);

    if (n < 0 || n > 31)
        luaL_error(L, "buffer shift amount must be in [0, 31]");
    if (size < 0)
        luaL_error(L, "buffer access out of bounds");
    if (size % 4 != 0)
        luaL_error(L, "buffer shift size must be a multiple of 4");
    if (isoutofbounds(offset, len, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    buffer_sunop_bytes<Op>((char*)buf + unsigned(offset), unsigned(size), n);
    return 0;
}

// Lanewise f32 fused multiply-add in place: dst[i] = dst[i] * mul[i] + add[i], in ONE memory pass with a single
// fused rounding on every platform (matching the value-type f32x4.fma and the interpreter). fmaf is used so the
// result is deterministic regardless of whether the toolchain would contract d * x + y on its own. The motivating
// case is an audio mix / image composite out = out * gain + src without two separate passes.
static void buffer_fma_scalar(char* dst, const char* m, const char* a, unsigned len)
{
    for (unsigned i = 0; i + 4 <= len; i += 4)
    {
        float d, x, y;
        memcpy(&d, dst + i, 4);
        memcpy(&x, m + i, 4);
        memcpy(&y, a + i, 4);
        d = fmaf(d, x, y);
        memcpy(dst + i, &d, 4);
    }
}

#ifdef LUAU_TARGET_AVX2
// Fused (single-rounding) vfmadd path. Only reached when FMA3 is present (see buffer_fma_bytes), so _mm*_fmadd_ps
// is safe; the LUAU_TARGET_FMA attribute lets it compile when fma is not in the baseline compiler settings.
LUAU_TARGET_FMA static void buffer_fma_avx2(char* dst, const char* m, const char* a, unsigned len)
{
    unsigned i = 0;
    for (; i + 32 <= len; i += 32)
    {
        __m256 d = _mm256_loadu_ps((const float*)(dst + i));
        __m256 x = _mm256_loadu_ps((const float*)(m + i));
        __m256 y = _mm256_loadu_ps((const float*)(a + i));
        _mm256_storeu_ps((float*)(dst + i), _mm256_fmadd_ps(d, x, y));
    }
    for (; i + 16 <= len; i += 16)
    {
        __m128 d = _mm_loadu_ps((const float*)(dst + i));
        __m128 x = _mm_loadu_ps((const float*)(m + i));
        __m128 y = _mm_loadu_ps((const float*)(a + i));
        _mm_storeu_ps((float*)(dst + i), _mm_fmadd_ps(d, x, y));
    }
    buffer_fma_scalar(dst + i, m + i, a + i, len - i);
}
#endif

static void buffer_fma_bytes(char* dst, const char* m, const char* a, unsigned len)
{
#ifdef LUAU_TARGET_AVX2
    // fma stays a single fused rounding on every device: the vectorized vfmadd path only when FMA3 is present,
    // otherwise the scalar fmaf path. Both round once, matching f32x4.fma and the interpreter.
    static const bool hasfma3 = buffer_hasavx2() && buffer_hasfma3();
    if (hasfma3)
        buffer_fma_avx2(dst, m, a, len);
    else
        buffer_fma_scalar(dst, m, a, len);
#else
    buffer_fma_scalar(dst, m, a, len);
#endif
}

static int buffer_fma(lua_State* L)
{
    size_t dlen = 0;
    void* dbuf = luaL_checkbuffer(L, 1, &dlen);
    int doff = luaL_checkinteger(L, 2);
    size_t mlen = 0;
    void* mbuf = luaL_checkbuffer(L, 3, &mlen);
    int moff = luaL_optinteger(L, 4, 0);
    size_t alen = 0;
    void* abuf = luaL_checkbuffer(L, 5, &alen);
    int aoff = luaL_optinteger(L, 6, 0);

    int size = luaL_optinteger(L, 7, int(dlen) - doff);

    if (size < 0)
        luaL_error(L, "buffer access out of bounds");
    if (size % 4 != 0)
        luaL_error(L, "buffer.fma size must be a multiple of 4");
    if (isoutofbounds(doff, dlen, unsigned(size)) || isoutofbounds(moff, mlen, unsigned(size)) ||
        isoutofbounds(aoff, alen, unsigned(size)))
        luaL_error(L, "buffer access out of bounds");

    buffer_fma_bytes((char*)dbuf + doff, (const char*)mbuf + moff, (const char*)abuf + aoff, unsigned(size));
    return 0;
}

// Transpose a Rows-by-Columns matrix of u32 from src into dst (which becomes Columns-by-Rows). dst and src must not
// overlap. The AVX2 path tiles into 8x8 blocks and transposes each in registers; the scalar path is the fallback.
static void buffer_transpose_scalar(uint32_t* dst, const uint32_t* src, unsigned rows, unsigned cols)
{
    for (unsigned r = 0; r < rows; r++)
        for (unsigned c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

#ifdef LUAU_TARGET_AVX2
LUAU_TARGET_AVX2 static void buffer_transpose8x8(uint32_t* dst, const uint32_t* src, unsigned dstStride, unsigned srcStride)
{
    __m256i r0 = _mm256_loadu_si256((const __m256i*)(src + 0 * srcStride));
    __m256i r1 = _mm256_loadu_si256((const __m256i*)(src + 1 * srcStride));
    __m256i r2 = _mm256_loadu_si256((const __m256i*)(src + 2 * srcStride));
    __m256i r3 = _mm256_loadu_si256((const __m256i*)(src + 3 * srcStride));
    __m256i r4 = _mm256_loadu_si256((const __m256i*)(src + 4 * srcStride));
    __m256i r5 = _mm256_loadu_si256((const __m256i*)(src + 5 * srcStride));
    __m256i r6 = _mm256_loadu_si256((const __m256i*)(src + 6 * srcStride));
    __m256i r7 = _mm256_loadu_si256((const __m256i*)(src + 7 * srcStride));

    __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
    __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
    __m256i t3 = _mm256_unpackhi_epi32(r2, r3);
    __m256i t4 = _mm256_unpacklo_epi32(r4, r5);
    __m256i t5 = _mm256_unpackhi_epi32(r4, r5);
    __m256i t6 = _mm256_unpacklo_epi32(r6, r7);
    __m256i t7 = _mm256_unpackhi_epi32(r6, r7);

    __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi64(t5, t7);

    _mm256_storeu_si256((__m256i*)(dst + 0 * dstStride), _mm256_permute2x128_si256(u0, u4, 0x20));
    _mm256_storeu_si256((__m256i*)(dst + 1 * dstStride), _mm256_permute2x128_si256(u1, u5, 0x20));
    _mm256_storeu_si256((__m256i*)(dst + 2 * dstStride), _mm256_permute2x128_si256(u2, u6, 0x20));
    _mm256_storeu_si256((__m256i*)(dst + 3 * dstStride), _mm256_permute2x128_si256(u3, u7, 0x20));
    _mm256_storeu_si256((__m256i*)(dst + 4 * dstStride), _mm256_permute2x128_si256(u0, u4, 0x31));
    _mm256_storeu_si256((__m256i*)(dst + 5 * dstStride), _mm256_permute2x128_si256(u1, u5, 0x31));
    _mm256_storeu_si256((__m256i*)(dst + 6 * dstStride), _mm256_permute2x128_si256(u2, u6, 0x31));
    _mm256_storeu_si256((__m256i*)(dst + 7 * dstStride), _mm256_permute2x128_si256(u3, u7, 0x31));
}

LUAU_TARGET_AVX2 static void buffer_transpose_avx2(uint32_t* dst, const uint32_t* src, unsigned rows, unsigned cols)
{
    for (unsigned tr = 0; tr < rows; tr += 8)
        for (unsigned tc = 0; tc < cols; tc += 8)
            buffer_transpose8x8(dst + tc * rows + tr, src + tr * cols + tc, rows, cols);
}
#endif

static void buffer_transpose_bytes(uint32_t* dst, const uint32_t* src, unsigned rows, unsigned cols)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();
    if (hasavx2 && (rows % 8) == 0 && (cols % 8) == 0)
        buffer_transpose_avx2(dst, src, rows, cols);
    else
        buffer_transpose_scalar(dst, src, rows, cols);
#else
    buffer_transpose_scalar(dst, src, rows, cols);
#endif
}

static int buffer_transpose(lua_State* L)
{
    size_t dlen = 0;
    void* dbuf = luaL_checkbuffer(L, 1, &dlen);
    int doff = luaL_checkinteger(L, 2);
    size_t slen = 0;
    void* sbuf = luaL_checkbuffer(L, 3, &slen);
    int soff = luaL_checkinteger(L, 4);
    int rows = luaL_checkinteger(L, 5);
    int cols = luaL_checkinteger(L, 6);

    if (rows <= 0 || cols <= 0)
        luaL_error(L, "buffer.transpose rows and columns must be positive");
    if ((doff % 4) != 0 || (soff % 4) != 0)
        luaL_error(L, "buffer.transpose offsets must be a multiple of 4");

    uint64_t bytes = uint64_t(unsigned(rows)) * unsigned(cols) * 4;
    if (isoutofbounds(doff, dlen, bytes) || isoutofbounds(soff, slen, bytes))
        luaL_error(L, "buffer access out of bounds");

    buffer_transpose_bytes((uint32_t*)((char*)dbuf + doff), (const uint32_t*)((char*)sbuf + soff), unsigned(rows), unsigned(cols));
    return 0;
}

// dst = data XOR transpose(src): transpose a Rows-by-Columns u32 matrix and XOR it onto a Columns-by-Rows data block
// in one pass. dst may alias data (in-place), but src must not overlap dst. This fuses the transpose and the XOR so a
// transposed keystream never has to round-trip through a buffer (avoiding a store-to-load stall across the call).
static void buffer_transposexor_scalar(uint32_t* dst, const uint32_t* data, const uint32_t* src, unsigned rows, unsigned cols)
{
    for (unsigned r = 0; r < rows; r++)
        for (unsigned c = 0; c < cols; c++)
            dst[c * rows + r] = data[c * rows + r] ^ src[r * cols + c];
}

#ifdef LUAU_TARGET_AVX2
LUAU_TARGET_AVX2 static void buffer_transposexor8x8(
    uint32_t* dst, const uint32_t* data, const uint32_t* src, unsigned dstStride, unsigned srcStride
)
{
    __m256i r0 = _mm256_loadu_si256((const __m256i*)(src + 0 * srcStride));
    __m256i r1 = _mm256_loadu_si256((const __m256i*)(src + 1 * srcStride));
    __m256i r2 = _mm256_loadu_si256((const __m256i*)(src + 2 * srcStride));
    __m256i r3 = _mm256_loadu_si256((const __m256i*)(src + 3 * srcStride));
    __m256i r4 = _mm256_loadu_si256((const __m256i*)(src + 4 * srcStride));
    __m256i r5 = _mm256_loadu_si256((const __m256i*)(src + 5 * srcStride));
    __m256i r6 = _mm256_loadu_si256((const __m256i*)(src + 6 * srcStride));
    __m256i r7 = _mm256_loadu_si256((const __m256i*)(src + 7 * srcStride));

    __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
    __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
    __m256i t3 = _mm256_unpackhi_epi32(r2, r3);
    __m256i t4 = _mm256_unpacklo_epi32(r4, r5);
    __m256i t5 = _mm256_unpackhi_epi32(r4, r5);
    __m256i t6 = _mm256_unpacklo_epi32(r6, r7);
    __m256i t7 = _mm256_unpackhi_epi32(r6, r7);

    __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi64(t5, t7);

    __m256i o0 = _mm256_permute2x128_si256(u0, u4, 0x20);
    __m256i o1 = _mm256_permute2x128_si256(u1, u5, 0x20);
    __m256i o2 = _mm256_permute2x128_si256(u2, u6, 0x20);
    __m256i o3 = _mm256_permute2x128_si256(u3, u7, 0x20);
    __m256i o4 = _mm256_permute2x128_si256(u0, u4, 0x31);
    __m256i o5 = _mm256_permute2x128_si256(u1, u5, 0x31);
    __m256i o6 = _mm256_permute2x128_si256(u2, u6, 0x31);
    __m256i o7 = _mm256_permute2x128_si256(u3, u7, 0x31);

    _mm256_storeu_si256((__m256i*)(dst + 0 * dstStride), _mm256_xor_si256(o0, _mm256_loadu_si256((const __m256i*)(data + 0 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 1 * dstStride), _mm256_xor_si256(o1, _mm256_loadu_si256((const __m256i*)(data + 1 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 2 * dstStride), _mm256_xor_si256(o2, _mm256_loadu_si256((const __m256i*)(data + 2 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 3 * dstStride), _mm256_xor_si256(o3, _mm256_loadu_si256((const __m256i*)(data + 3 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 4 * dstStride), _mm256_xor_si256(o4, _mm256_loadu_si256((const __m256i*)(data + 4 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 5 * dstStride), _mm256_xor_si256(o5, _mm256_loadu_si256((const __m256i*)(data + 5 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 6 * dstStride), _mm256_xor_si256(o6, _mm256_loadu_si256((const __m256i*)(data + 6 * dstStride))));
    _mm256_storeu_si256((__m256i*)(dst + 7 * dstStride), _mm256_xor_si256(o7, _mm256_loadu_si256((const __m256i*)(data + 7 * dstStride))));
}

LUAU_TARGET_AVX2 static void buffer_transposexor_avx2(uint32_t* dst, const uint32_t* data, const uint32_t* src, unsigned rows, unsigned cols)
{
    for (unsigned tr = 0; tr < rows; tr += 8)
        for (unsigned tc = 0; tc < cols; tc += 8)
            buffer_transposexor8x8(dst + tc * rows + tr, data + tc * rows + tr, src + tr * cols + tc, rows, cols);
}
#endif

static void buffer_transposexor_bytes(uint32_t* dst, const uint32_t* data, const uint32_t* src, unsigned rows, unsigned cols)
{
#ifdef LUAU_TARGET_AVX2
    static const bool hasavx2 = buffer_hasavx2();
    if (hasavx2 && (rows % 8) == 0 && (cols % 8) == 0)
        buffer_transposexor_avx2(dst, data, src, rows, cols);
    else
        buffer_transposexor_scalar(dst, data, src, rows, cols);
#else
    buffer_transposexor_scalar(dst, data, src, rows, cols);
#endif
}

static int buffer_transposexor(lua_State* L)
{
    size_t dlen = 0;
    void* dbuf = luaL_checkbuffer(L, 1, &dlen);
    int doff = luaL_checkinteger(L, 2);
    size_t alen = 0;
    void* abuf = luaL_checkbuffer(L, 3, &alen);
    int aoff = luaL_checkinteger(L, 4);
    size_t slen = 0;
    void* sbuf = luaL_checkbuffer(L, 5, &slen);
    int soff = luaL_checkinteger(L, 6);
    int rows = luaL_checkinteger(L, 7);
    int cols = luaL_checkinteger(L, 8);

    if (rows <= 0 || cols <= 0)
        luaL_error(L, "buffer.transposexor rows and columns must be positive");
    if ((doff % 4) != 0 || (aoff % 4) != 0 || (soff % 4) != 0)
        luaL_error(L, "buffer.transposexor offsets must be a multiple of 4");

    uint64_t bytes = uint64_t(unsigned(rows)) * unsigned(cols) * 4;
    if (isoutofbounds(doff, dlen, bytes) || isoutofbounds(aoff, alen, bytes) || isoutofbounds(soff, slen, bytes))
        luaL_error(L, "buffer access out of bounds");

    buffer_transposexor_bytes(
        (uint32_t*)((char*)dbuf + doff), (const uint32_t*)((char*)abuf + aoff), (const uint32_t*)((char*)sbuf + soff), unsigned(rows), unsigned(cols)
    );
    return 0;
}

static int buffer_readu32x4(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    if (isoutofbounds(offset, len, 16))
        luaL_error(L, "buffer access out of bounds");

    uint32_t* r = lua_newsimd(L);
    memcpy(r, (char*)buf + unsigned(offset), 16);
    return 1;
}

static int buffer_writeu32x4(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    const uint32_t* v = luaL_checksimd(L, 3);

    if (isoutofbounds(offset, len, 16))
        luaL_error(L, "buffer access out of bounds");

    memcpy((char*)buf + unsigned(offset), v, 16);
    return 0;
}

static int buffer_readu32x8(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    if (isoutofbounds(offset, len, 32))
        luaL_error(L, "buffer access out of bounds");

    uint32_t* r = lua_newsimd(L);
    memcpy(r, (char*)buf + unsigned(offset), 32);
    return 1;
}

static int buffer_writeu32x8(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    const uint32_t* v = luaL_checksimd(L, 3);

    if (isoutofbounds(offset, len, 32))
        luaL_error(L, "buffer access out of bounds");

    memcpy((char*)buf + unsigned(offset), v, 32);
    return 0;
}

// Bridge the vector type to buffer storage (vertex/particle data is buffer-backed): read/write LUA_VECTOR_SIZE
// consecutive f32 lanes as a single vector value, no per-component readf32/writef32 juggling.
static int buffer_readvector(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);

    if (isoutofbounds(offset, len, LUA_VECTOR_SIZE * 4))
        luaL_error(L, "buffer access out of bounds");

    float v[LUA_VECTOR_SIZE];
    memcpy(v, (char*)buf + unsigned(offset), LUA_VECTOR_SIZE * 4);
#if LUA_VECTOR_SIZE == 4
    lua_pushvector(L, v[0], v[1], v[2], v[3]);
#else
    lua_pushvector(L, v[0], v[1], v[2]);
#endif
    return 1;
}

static int buffer_writevector(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    const float* v = luaL_checkvector(L, 3);

    if (isoutofbounds(offset, len, LUA_VECTOR_SIZE * 4))
        luaL_error(L, "buffer access out of bounds");

    memcpy((char*)buf + unsigned(offset), v, LUA_VECTOR_SIZE * 4);
    return 0;
}

static int buffer_readbits(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int64_t bitoffset = (int64_t)luaL_checknumber(L, 2);
    int bitcount = luaL_checkinteger(L, 3);

    if (bitoffset < 0)
        luaL_error(L, "buffer access out of bounds");

    if (unsigned(bitcount) > 32)
        luaL_error(L, "bit count is out of range of [0; 32]");

    if (uint64_t(bitoffset + bitcount) > uint64_t(len) * 8)
        luaL_error(L, "buffer access out of bounds");

    unsigned startbyte = unsigned(bitoffset / 8);
    unsigned endbyte = unsigned((bitoffset + bitcount + 7) / 8);

    uint64_t data = 0;

#if defined(LUAU_BIG_ENDIAN)
    for (int i = int(endbyte) - 1; i >= int(startbyte); i--)
        data = (data << 8) + uint8_t(((char*)buf)[i]);
#else
    memcpy(&data, (char*)buf + startbyte, endbyte - startbyte);
#endif

    uint64_t subbyteoffset = bitoffset & 0x7;
    uint64_t mask = (1ull << bitcount) - 1;

    lua_pushunsigned(L, unsigned((data >> subbyteoffset) & mask));
    return 1;
}

static int buffer_writebits(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int64_t bitoffset = (int64_t)luaL_checknumber(L, 2);
    int bitcount = luaL_checkinteger(L, 3);
    unsigned value = luaL_checkunsigned(L, 4);

    if (bitoffset < 0)
        luaL_error(L, "buffer access out of bounds");

    if (unsigned(bitcount) > 32)
        luaL_error(L, "bit count is out of range of [0; 32]");

    if (uint64_t(bitoffset + bitcount) > uint64_t(len) * 8)
        luaL_error(L, "buffer access out of bounds");

    unsigned startbyte = unsigned(bitoffset / 8);
    unsigned endbyte = unsigned((bitoffset + bitcount + 7) / 8);

    uint64_t data = 0;

#if defined(LUAU_BIG_ENDIAN)
    for (int i = int(endbyte) - 1; i >= int(startbyte); i--)
        data = data * 256 + uint8_t(((char*)buf)[i]);
#else
    memcpy(&data, (char*)buf + startbyte, endbyte - startbyte);
#endif

    uint64_t subbyteoffset = bitoffset & 0x7;
    uint64_t mask = ((1ull << bitcount) - 1) << subbyteoffset;

    data = (data & ~mask) | ((uint64_t(value) << subbyteoffset) & mask);

#if defined(LUAU_BIG_ENDIAN)
    for (int i = int(startbyte); i < int(endbyte); i++)
    {
        ((char*)buf)[i] = data & 0xff;
        data >>= 8;
    }
#else
    memcpy((char*)buf + startbyte, &data, endbyte - startbyte);
#endif
    return 0;
}

static const luaL_Reg bufferlib[] = {
    {"create", buffer_create},
    {"fromstring", buffer_fromstring},
    {"tostring", buffer_tostring},
    {"readi8", buffer_readinteger<int8_t>},
    {"readu8", buffer_readinteger<uint8_t>},
    {"readi16", buffer_readinteger<int16_t>},
    {"readu16", buffer_readinteger<uint16_t>},
    {"readi32", buffer_readinteger<int32_t>},
    {"readu32", buffer_readinteger<uint32_t>},
    {"readf32", buffer_readfp<float, uint32_t>},
    {"readf64", buffer_readfp<double, uint64_t>},
    {"writei8", buffer_writeinteger<int8_t>},
    {"writeu8", buffer_writeinteger<uint8_t>},
    {"writei16", buffer_writeinteger<int16_t>},
    {"writeu16", buffer_writeinteger<uint16_t>},
    {"writei32", buffer_writeinteger<int32_t>},
    {"writeu32", buffer_writeinteger<uint32_t>},
    {"writef32", buffer_writefp<float, uint32_t>},
    {"writef64", buffer_writefp<double, uint64_t>},
    {"readstring", buffer_readstring},
    {"writestring", buffer_writestring},
    {"len", buffer_len},
    {"copy", buffer_copy},
    {"fill", buffer_fill},
    {"bxor", buffer_binop<buffer_op_xor>},
    {"band", buffer_binop<buffer_op_and>},
    {"bor", buffer_binop<buffer_op_or>},
    {"bnot", buffer_bnot},
    {"add", buffer_ibinop<buffer_iop_add>},
    {"sub", buffer_ibinop<buffer_iop_sub>},
    {"mul", buffer_ibinop<buffer_iop_mul>},
    {"shl", buffer_sunop<buffer_sop_shl>},
    {"shr", buffer_sunop<buffer_sop_shr>},
    {"rotl", buffer_rotl},
    {"fadd", buffer_fbinop<buffer_fop_add>},
    {"fsub", buffer_fbinop<buffer_fop_sub>},
    {"fmul", buffer_fbinop<buffer_fop_mul>},
    {"fdiv", buffer_fbinop<buffer_fop_div>},
    {"fmin", buffer_fbinop<buffer_fop_min>},
    {"fmax", buffer_fbinop<buffer_fop_max>},
    {"fma", buffer_fma},
    {"transpose", buffer_transpose},
    {"transposexor", buffer_transposexor},
    {"readvector", buffer_readvector},
    {"writevector", buffer_writevector},
    {"readu32x4", buffer_readu32x4},
    {"writeu32x4", buffer_writeu32x4},
    {"readu32x8", buffer_readu32x8},
    {"writeu32x8", buffer_writeu32x8},
    {"readbits", buffer_readbits},
    {"writebits", buffer_writebits},
    {"readinteger", buffer_readlong},
    {"writeinteger", buffer_writelong},
    {NULL, NULL},
};

static const luaL_Reg bufferlib_NOINTEGER[] = {
    {"create", buffer_create},
    {"fromstring", buffer_fromstring},
    {"tostring", buffer_tostring},
    {"readi8", buffer_readinteger<int8_t>},
    {"readu8", buffer_readinteger<uint8_t>},
    {"readi16", buffer_readinteger<int16_t>},
    {"readu16", buffer_readinteger<uint16_t>},
    {"readi32", buffer_readinteger<int32_t>},
    {"readu32", buffer_readinteger<uint32_t>},
    {"readf32", buffer_readfp<float, uint32_t>},
    {"readf64", buffer_readfp<double, uint64_t>},
    {"writei8", buffer_writeinteger<int8_t>},
    {"writeu8", buffer_writeinteger<uint8_t>},
    {"writei16", buffer_writeinteger<int16_t>},
    {"writeu16", buffer_writeinteger<uint16_t>},
    {"writei32", buffer_writeinteger<int32_t>},
    {"writeu32", buffer_writeinteger<uint32_t>},
    {"writef32", buffer_writefp<float, uint32_t>},
    {"writef64", buffer_writefp<double, uint64_t>},
    {"readstring", buffer_readstring},
    {"writestring", buffer_writestring},
    {"len", buffer_len},
    {"copy", buffer_copy},
    {"fill", buffer_fill},
    {"bxor", buffer_binop<buffer_op_xor>},
    {"band", buffer_binop<buffer_op_and>},
    {"bor", buffer_binop<buffer_op_or>},
    {"bnot", buffer_bnot},
    {"add", buffer_ibinop<buffer_iop_add>},
    {"sub", buffer_ibinop<buffer_iop_sub>},
    {"mul", buffer_ibinop<buffer_iop_mul>},
    {"shl", buffer_sunop<buffer_sop_shl>},
    {"shr", buffer_sunop<buffer_sop_shr>},
    {"rotl", buffer_rotl},
    {"fadd", buffer_fbinop<buffer_fop_add>},
    {"fsub", buffer_fbinop<buffer_fop_sub>},
    {"fmul", buffer_fbinop<buffer_fop_mul>},
    {"fdiv", buffer_fbinop<buffer_fop_div>},
    {"fmin", buffer_fbinop<buffer_fop_min>},
    {"fmax", buffer_fbinop<buffer_fop_max>},
    {"fma", buffer_fma},
    {"transpose", buffer_transpose},
    {"transposexor", buffer_transposexor},
    {"readvector", buffer_readvector},
    {"writevector", buffer_writevector},
    {"readu32x4", buffer_readu32x4},
    {"writeu32x4", buffer_writeu32x4},
    {"readu32x8", buffer_readu32x8},
    {"writeu32x8", buffer_writeu32x8},
    {"readbits", buffer_readbits},
    {"writebits", buffer_writebits},
    {NULL, NULL},
};

int luaopen_buffer(lua_State* L)
{
    if (FFlag::LuauIntegerLibrary)
        luaL_register(L, LUA_BUFFERLIBNAME, bufferlib);
    else
        luaL_register(L, LUA_BUFFERLIBNAME, bufferlib_NOINTEGER);

    return 1;
}
