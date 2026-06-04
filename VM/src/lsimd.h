// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "lobject.h"

#define sizesimd() (sizeof(Simd))

LUAI_FUNC Simd* luaSimd_new(lua_State* L);
LUAI_FUNC void luaSimd_free(lua_State* L, Simd* s, struct lua_Page* page);
