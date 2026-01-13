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

#include "common/Module.h"
#include "common/StringMap.h"
#include <vector>
#include <memory>
#include <string>

namespace love
{
namespace tensorlogic
{

class Tensor;

/**
 * TensorLogic module provides neural-symbolic AI capabilities
 * based on Pedro Domingos' Tensor Logic framework.
 * 
 * This module bridges symbolic reasoning and neural networks
 * by representing both as tensor operations.
 */
class TensorLogic : public Module
{
public:
	TensorLogic();
	virtual ~TensorLogic();

	// Module interface
	const char *getName() const override { return "love.tensorlogic"; }
	ModuleType getModuleType() const override { return M_TENSORLOGIC; }

	/**
	 * Create a new tensor with specified dimensions
	 * @param dims Vector of dimension sizes
	 * @param data Optional initial data
	 * @return Shared pointer to created tensor
	 */
	std::shared_ptr<Tensor> newTensor(const std::vector<int>& dims, const std::vector<float>& data = {});

	/**
	 * Create a boolean tensor for logical operations
	 * @param dims Vector of dimension sizes
	 * @param data Optional initial boolean data
	 * @return Shared pointer to created boolean tensor
	 */
	std::shared_ptr<Tensor> newBoolTensor(const std::vector<int>& dims, const std::vector<bool>& data = {});

	/**
	 * Perform Einstein summation on tensors
	 * @param equation Einstein summation equation (e.g., "ij,jk->ik")
	 * @param tensors Input tensors for the operation
	 * @return Result tensor
	 */
	std::shared_ptr<Tensor> einsum(const std::string& equation, const std::vector<std::shared_ptr<Tensor>>& tensors);

	/**
	 * Apply logical AND operation (element-wise)
	 */
	std::shared_ptr<Tensor> logicalAnd(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

	/**
	 * Apply logical OR operation (element-wise)
	 */
	std::shared_ptr<Tensor> logicalOr(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

	/**
	 * Apply logical NOT operation (element-wise)
	 */
	std::shared_ptr<Tensor> logicalNot(std::shared_ptr<Tensor> a);

	/**
	 * Perform tensor contraction (generalized matrix multiplication)
	 */
	std::shared_ptr<Tensor> contract(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, 
	                                  const std::vector<int>& a_dims, const std::vector<int>& b_dims);

private:
	// Helper methods for tensor operations
	bool validateDimensions(const std::vector<int>& dims) const;
	size_t computeTensorSize(const std::vector<int>& dims) const;
};

/**
 * Tensor class represents multi-dimensional arrays for tensor logic operations
 */
class Tensor : public love::Object
{
public:
	static love::Type type;

	enum class DataType
	{
		FLOAT32,
		BOOL,
		INT32
	};

	Tensor(const std::vector<int>& dims, DataType type = DataType::FLOAT32);
	virtual ~Tensor();

	// Getters
	const std::vector<int>& getDimensions() const { return dimensions; }
	size_t getRank() const { return dimensions.size(); }
	size_t getSize() const { return size; }
	DataType getDataType() const { return dataType; }

	// Data access
	float getFloat(size_t index) const;
	bool getBool(size_t index) const;
	int getInt(size_t index) const;

	void setFloat(size_t index, float value);
	void setBool(size_t index, bool value);
	void setInt(size_t index, int value);

	// Tensor operations
	std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other) const;
	std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> other) const;
	std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other) const;
	
	// Shape operations
	std::shared_ptr<Tensor> reshape(const std::vector<int>& newDims) const;
	std::shared_ptr<Tensor> transpose(const std::vector<int>& axes = {}) const;
	
	// Statistical operations
	float sum() const;
	float mean() const;
	float max() const;
	float min() const;

	// Utilities
	std::string toString() const;
	void fill(float value);
	void randomize(float min = 0.0f, float max = 1.0f);

private:
	std::vector<int> dimensions;
	size_t size;
	DataType dataType;
	std::vector<float> floatData;
	std::vector<bool> boolData;
	std::vector<int> intData;

	size_t computeFlatIndex(const std::vector<int>& indices) const;
	std::vector<int> computeMultiIndex(size_t flatIndex) const;
};

} // namespace tensorlogic
} // namespace love
