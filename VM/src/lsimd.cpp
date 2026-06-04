// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lsimd.h"

#include "lgc.h"
#include "lmem.h"

#include <string.h>

Simd* luaSimd_new(lua_State* L)
{
    Simd* s = luaM_newgco(L, Simd, sizesimd(), L->activememcat);
    luaC_init(L, s, LUA_TSIMD);
    memset(s->data, 0, sizeof(s->data));
    return s;
}

// Returns a box to write a SIMD value into a VM register slot: if the slot already holds a LUA_TSIMD object the
// existing box is reused (its lane data is then overwritten in place by the caller), otherwise a fresh box is
// allocated and stored. Native code only uses this for slots whose box provably never escapes, so reusing the
// box is observationally identical to allocating a new value while avoiding an allocation per loop iteration.
Simd* luaSimd_storeReuse(lua_State* L, TValue* slot)
{
    if (ttissimd(slot))
        return simdvalue(slot);

    Simd* s = luaSimd_new(L);
    setsimdvalue(L, slot, s);
    return s;
}

void luaSimd_free(lua_State* L, Simd* s, lua_Page* page)
{
    luaM_freegco(L, s, sizesimd(), s->memcat, page);
}
