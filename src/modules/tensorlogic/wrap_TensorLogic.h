/**
 * Copyright (c) 2006-2025 LOVE Development Team
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 **/

#pragma once

#include "TensorLogic.h"
#include "common/runtime.h"

namespace love
{
namespace tensorlogic
{

extern "C" int luaopen_love_tensorlogic(lua_State *L);

// Module functions
int w_newTensor(lua_State *L);
int w_newBoolTensor(lua_State *L);
int w_einsum(lua_State *L);
int w_logicalAnd(lua_State *L);
int w_logicalOr(lua_State *L);
int w_logicalNot(lua_State *L);

// Tensor methods
int w_Tensor_getDimensions(lua_State *L);
int w_Tensor_getRank(lua_State *L);
int w_Tensor_getSize(lua_State *L);
int w_Tensor_getType(lua_State *L);
int w_Tensor_get(lua_State *L);
int w_Tensor_set(lua_State *L);
int w_Tensor_add(lua_State *L);
int w_Tensor_multiply(lua_State *L);
int w_Tensor_matmul(lua_State *L);
int w_Tensor_reshape(lua_State *L);
int w_Tensor_transpose(lua_State *L);
int w_Tensor_sum(lua_State *L);
int w_Tensor_mean(lua_State *L);
int w_Tensor_max(lua_State *L);
int w_Tensor_min(lua_State *L);
int w_Tensor_fill(lua_State *L);
int w_Tensor_randomize(lua_State *L);
int w_Tensor_toString(lua_State *L);

extern const luaL_Reg w_TensorLogic_functions[];
extern const lua_CFunction w_TensorLogic_types[];

} // namespace tensorlogic
} // namespace love
