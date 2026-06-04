// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lualib.h"

#include "lcommon.h"
#include "lbuffer.h"

#if defined(LUAU_BIG_ENDIAN)
#include <endian.h>
#endif

LUAU_FASTFLAG(LuauIntegerLibrary)

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
    {"readu32x4", buffer_readu32x4},
    {"writeu32x4", buffer_writeu32x4},
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
    {"readu32x4", buffer_readu32x4},
    {"writeu32x4", buffer_writeu32x4},
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
