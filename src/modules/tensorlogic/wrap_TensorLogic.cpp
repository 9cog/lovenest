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

#include "wrap_TensorLogic.h"
#include "common/wrap_Module.h"
#include <memory>

namespace love
{
namespace tensorlogic
{

#define instance() (Module::getInstance<TensorLogic>(Module::M_TENSORLOGIC))

// Helper to get tensor from Lua stack
static Tensor *luax_checktensor(lua_State *L, int idx)
{
	return luax_checktype<Tensor>(L, idx);
}

int w_newTensor(lua_State *L)
{
	// Get dimensions from table
	luaL_checktype(L, 1, LUA_TTABLE);
	
	std::vector<int> dims;
	int n = (int)luax_objlen(L, 1);
	for (int i = 1; i <= n; ++i)
	{
		lua_rawgeti(L, 1, i);
		dims.push_back((int)luaL_checkinteger(L, -1));
		lua_pop(L, 1);
	}
	
	// Optional data array
	std::vector<float> data;
	if (lua_istable(L, 2))
	{
		int dataLen = (int)luax_objlen(L, 2);
		for (int i = 1; i <= dataLen; ++i)
		{
			lua_rawgeti(L, 2, i);
			data.push_back((float)luaL_checknumber(L, -1));
			lua_pop(L, 1);
		}
	}
	
	auto tensor = instance()->newTensor(dims, data);
	luax_pushtype(L, tensor.get());
	tensor->retain();
	return 1;
}

int w_newBoolTensor(lua_State *L)
{
	// Get dimensions from table
	luaL_checktype(L, 1, LUA_TTABLE);
	
	std::vector<int> dims;
	int n = (int)luax_objlen(L, 1);
	for (int i = 1; i <= n; ++i)
	{
		lua_rawgeti(L, 1, i);
		dims.push_back((int)luaL_checkinteger(L, -1));
		lua_pop(L, 1);
	}
	
	// Optional data array
	std::vector<bool> data;
	if (lua_istable(L, 2))
	{
		int dataLen = (int)luax_objlen(L, 2);
		for (int i = 1; i <= dataLen; ++i)
		{
			lua_rawgeti(L, 2, i);
			data.push_back(lua_toboolean(L, -1) != 0);
			lua_pop(L, 1);
		}
	}
	
	auto tensor = instance()->newBoolTensor(dims, data);
	luax_pushtype(L, tensor.get());
	tensor->retain();
	return 1;
}

int w_einsum(lua_State *L)
{
	const char *equation = luaL_checkstring(L, 1);
	luaL_checktype(L, 2, LUA_TTABLE);
	
	std::vector<std::shared_ptr<Tensor>> tensors;
	int n = (int)luax_objlen(L, 2);
	for (int i = 1; i <= n; ++i)
	{
		lua_rawgeti(L, 2, i);
		Tensor *t = luax_checktensor(L, -1);
		t->retain(); // Retain for this operation
		tensors.push_back(std::shared_ptr<Tensor>(t, [](Tensor* p) { p->release(); })); // Custom deleter
		lua_pop(L, 1);
	}
	
	auto result = instance()->einsum(equation, tensors);
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_logicalAnd(lua_State *L)
{
	Tensor *a = luax_checktensor(L, 1);
	Tensor *b = luax_checktensor(L, 2);
	
	a->retain();
	b->retain();
	auto result = instance()->logicalAnd(
		std::shared_ptr<Tensor>(a, [](Tensor* p) { p->release(); }),
		std::shared_ptr<Tensor>(b, [](Tensor* p) { p->release(); })
	);
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_logicalOr(lua_State *L)
{
	Tensor *a = luax_checktensor(L, 1);
	Tensor *b = luax_checktensor(L, 2);
	
	a->retain();
	b->retain();
	auto result = instance()->logicalOr(
		std::shared_ptr<Tensor>(a, [](Tensor* p) { p->release(); }),
		std::shared_ptr<Tensor>(b, [](Tensor* p) { p->release(); })
	);
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_logicalNot(lua_State *L)
{
	Tensor *a = luax_checktensor(L, 1);
	
	a->retain();
	auto result = instance()->logicalNot(
		std::shared_ptr<Tensor>(a, [](Tensor* p) { p->release(); })
	);
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

// Tensor methods

int w_Tensor_getDimensions(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	const auto& dims = t->getDimensions();
	
	lua_createtable(L, (int)dims.size(), 0);
	for (size_t i = 0; i < dims.size(); ++i)
	{
		lua_pushinteger(L, dims[i]);
		lua_rawseti(L, -2, i + 1);
	}
	
	return 1;
}

int w_Tensor_getRank(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushinteger(L, (lua_Integer)t->getRank());
	return 1;
}

int w_Tensor_getSize(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushinteger(L, (lua_Integer)t->getSize());
	return 1;
}

int w_Tensor_getType(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	switch (t->getDataType())
	{
	case Tensor::DataType::FLOAT32:
		lua_pushstring(L, "float32");
		break;
	case Tensor::DataType::BOOL:
		lua_pushstring(L, "bool");
		break;
	case Tensor::DataType::INT32:
		lua_pushstring(L, "int32");
		break;
	default:
		lua_pushstring(L, "unknown");
		break;
	}
	return 1;
}

int w_Tensor_get(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	int index = (int)luaL_checkinteger(L, 2) - 1; // Lua is 1-indexed
	
	lua_pushnumber(L, t->getFloat(index));
	return 1;
}

int w_Tensor_set(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	int index = (int)luaL_checkinteger(L, 2) - 1; // Lua is 1-indexed
	float value = (float)luaL_checknumber(L, 3);
	
	t->setFloat(index, value);
	return 0;
}

int w_Tensor_add(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	Tensor *other = luax_checktensor(L, 2);
	
	other->retain();
	auto result = t->add(std::shared_ptr<Tensor>(other, [](Tensor* p) { p->release(); }));
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_Tensor_multiply(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	Tensor *other = luax_checktensor(L, 2);
	
	other->retain();
	auto result = t->multiply(std::shared_ptr<Tensor>(other, [](Tensor* p) { p->release(); }));
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_Tensor_matmul(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	Tensor *other = luax_checktensor(L, 2);
	
	other->retain();
	auto result = t->matmul(std::shared_ptr<Tensor>(other, [](Tensor* p) { p->release(); }));
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_Tensor_reshape(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	luaL_checktype(L, 2, LUA_TTABLE);
	
	std::vector<int> dims;
	int n = (int)luax_objlen(L, 2);
	for (int i = 1; i <= n; ++i)
	{
		lua_rawgeti(L, 2, i);
		dims.push_back((int)luaL_checkinteger(L, -1));
		lua_pop(L, 1);
	}
	
	auto result = t->reshape(dims);
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_Tensor_transpose(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	
	std::vector<int> axes;
	if (lua_istable(L, 2))
	{
		int n = (int)luax_objlen(L, 2);
		for (int i = 1; i <= n; ++i)
		{
			lua_rawgeti(L, 2, i);
			axes.push_back((int)luaL_checkinteger(L, -1));
			lua_pop(L, 1);
		}
	}
	
	auto result = t->transpose(axes);
	luax_pushtype(L, result.get());
	result->retain();
	return 1;
}

int w_Tensor_sum(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushnumber(L, t->sum());
	return 1;
}

int w_Tensor_mean(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushnumber(L, t->mean());
	return 1;
}

int w_Tensor_max(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushnumber(L, t->max());
	return 1;
}

int w_Tensor_min(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushnumber(L, t->min());
	return 1;
}

int w_Tensor_fill(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	float value = (float)luaL_checknumber(L, 2);
	
	t->fill(value);
	return 0;
}

int w_Tensor_randomize(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	float min = (float)luaL_optnumber(L, 2, 0.0);
	float max = (float)luaL_optnumber(L, 3, 1.0);
	
	t->randomize(min, max);
	return 0;
}

int w_Tensor_toString(lua_State *L)
{
	Tensor *t = luax_checktensor(L, 1);
	lua_pushstring(L, t->toString().c_str());
	return 1;
}

static const luaL_Reg w_Tensor_functions[] =
{
	{ "getDimensions", w_Tensor_getDimensions },
	{ "getRank", w_Tensor_getRank },
	{ "getSize", w_Tensor_getSize },
	{ "getType", w_Tensor_getType },
	{ "get", w_Tensor_get },
	{ "set", w_Tensor_set },
	{ "add", w_Tensor_add },
	{ "multiply", w_Tensor_multiply },
	{ "matmul", w_Tensor_matmul },
	{ "reshape", w_Tensor_reshape },
	{ "transpose", w_Tensor_transpose },
	{ "sum", w_Tensor_sum },
	{ "mean", w_Tensor_mean },
	{ "max", w_Tensor_max },
	{ "min", w_Tensor_min },
	{ "fill", w_Tensor_fill },
	{ "randomize", w_Tensor_randomize },
	{ "__tostring", w_Tensor_toString },
	{ 0, 0 }
};

const luaL_Reg w_TensorLogic_functions[] =
{
	{ "newTensor", w_newTensor },
	{ "newBoolTensor", w_newBoolTensor },
	{ "einsum", w_einsum },
	{ "logicalAnd", w_logicalAnd },
	{ "logicalOr", w_logicalOr },
	{ "logicalNot", w_logicalNot },
	{ 0, 0 }
};

extern "C" int luaopen_love_tensorlogic(lua_State *L)
{
	TensorLogic *instance = instance();
	if (instance == nullptr)
	{
		luax_catchexcept(L, [&](){ instance = new TensorLogic(); });
	}
	else
		instance->retain();

	WrappedModule w;
	w.module = instance;
	w.name = "tensorlogic";
	w.type = &Module::type;
	w.functions = w_TensorLogic_functions;
	w.types = nullptr;

	int ret = luax_register_module(L, w);

	// Register Tensor type
	luax_registertype(L, &Tensor::type, w_Tensor_functions);

	return ret;
}

} // namespace tensorlogic
} // namespace love
