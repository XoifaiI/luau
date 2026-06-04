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

void luaSimd_free(lua_State* L, Simd* s, lua_Page* page)
{
    luaM_freegco(L, s, sizesimd(), s->memcat, page);
}
