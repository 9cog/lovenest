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

#include "TensorLogic.h"
#include "common/Exception.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>

namespace love
{
namespace tensorlogic
{

// Type definitions
love::Type Tensor::type("Tensor", &Object::type);

// ============================================================================
// TensorLogic Implementation
// ============================================================================

TensorLogic::TensorLogic()
	: Module(M_TENSORLOGIC, "love.tensorlogic")
{
}

TensorLogic::~TensorLogic()
{
}

std::shared_ptr<Tensor> TensorLogic::newTensor(const std::vector<int>& dims, const std::vector<float>& data)
{
	if (!validateDimensions(dims))
		throw love::Exception("Invalid tensor dimensions");

	auto tensor = std::make_shared<Tensor>(dims, Tensor::DataType::FLOAT32);
	
	if (!data.empty())
	{
		size_t expectedSize = computeTensorSize(dims);
		if (data.size() != expectedSize)
			throw love::Exception("Data size mismatch: expected %zu, got %zu", expectedSize, data.size());
		
		for (size_t i = 0; i < data.size(); ++i)
			tensor->setFloat(i, data[i]);
	}
	
	return tensor;
}

std::shared_ptr<Tensor> TensorLogic::newBoolTensor(const std::vector<int>& dims, const std::vector<bool>& data)
{
	if (!validateDimensions(dims))
		throw love::Exception("Invalid tensor dimensions");

	auto tensor = std::make_shared<Tensor>(dims, Tensor::DataType::BOOL);
	
	if (!data.empty())
	{
		size_t expectedSize = computeTensorSize(dims);
		if (data.size() != expectedSize)
			throw love::Exception("Data size mismatch: expected %zu, got %zu", expectedSize, data.size());
		
		for (size_t i = 0; i < data.size(); ++i)
			tensor->setBool(i, data[i]);
	}
	
	return tensor;
}

std::shared_ptr<Tensor> TensorLogic::einsum(const std::string& equation, 
                                             const std::vector<std::shared_ptr<Tensor>>& tensors)
{
	// Basic Einstein summation implementation
	// Format: "ij,jk->ik" for matrix multiplication
	
	if (tensors.empty())
		throw love::Exception("No tensors provided for einsum");
	
	// Parse equation
	size_t arrowPos = equation.find("->");
	if (arrowPos == std::string::npos)
		throw love::Exception("Invalid einsum equation: missing '->'");
	
	std::string inputPart = equation.substr(0, arrowPos);
	std::string outputPart = equation.substr(arrowPos + 2);
	
	// For now, implement basic matrix multiplication case: "ij,jk->ik"
	if (equation == "ij,jk->ik" && tensors.size() == 2)
	{
		return tensors[0]->matmul(tensors[1]);
	}
	
	throw love::Exception("Einsum equation '%s' not yet supported", equation.c_str());
}

std::shared_ptr<Tensor> TensorLogic::logicalAnd(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
	if (a->getDataType() != Tensor::DataType::BOOL || b->getDataType() != Tensor::DataType::BOOL)
		throw love::Exception("Logical operations require boolean tensors");
	
	if (a->getDimensions() != b->getDimensions())
		throw love::Exception("Tensor dimensions must match for logical AND");
	
	auto result = std::make_shared<Tensor>(a->getDimensions(), Tensor::DataType::BOOL);
	
	for (size_t i = 0; i < a->getSize(); ++i)
	{
		result->setBool(i, a->getBool(i) && b->getBool(i));
	}
	
	return result;
}

std::shared_ptr<Tensor> TensorLogic::logicalOr(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
	if (a->getDataType() != Tensor::DataType::BOOL || b->getDataType() != Tensor::DataType::BOOL)
		throw love::Exception("Logical operations require boolean tensors");
	
	if (a->getDimensions() != b->getDimensions())
		throw love::Exception("Tensor dimensions must match for logical OR");
	
	auto result = std::make_shared<Tensor>(a->getDimensions(), Tensor::DataType::BOOL);
	
	for (size_t i = 0; i < a->getSize(); ++i)
	{
		result->setBool(i, a->getBool(i) || b->getBool(i));
	}
	
	return result;
}

std::shared_ptr<Tensor> TensorLogic::logicalNot(std::shared_ptr<Tensor> a)
{
	if (a->getDataType() != Tensor::DataType::BOOL)
		throw love::Exception("Logical operations require boolean tensors");
	
	auto result = std::make_shared<Tensor>(a->getDimensions(), Tensor::DataType::BOOL);
	
	for (size_t i = 0; i < a->getSize(); ++i)
	{
		result->setBool(i, !a->getBool(i));
	}
	
	return result;
}

std::shared_ptr<Tensor> TensorLogic::contract(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                                               const std::vector<int>& a_dims, const std::vector<int>& b_dims)
{
	// Tensor contraction implementation
	// This is a generalization of matrix multiplication
	throw love::Exception("Tensor contraction not yet implemented");
}

bool TensorLogic::validateDimensions(const std::vector<int>& dims) const
{
	if (dims.empty())
		return false;
	
	for (int dim : dims)
	{
		if (dim <= 0)
			return false;
	}
	
	return true;
}

size_t TensorLogic::computeTensorSize(const std::vector<int>& dims) const
{
	size_t size = 1;
	for (int dim : dims)
		size *= dim;
	return size;
}

// ============================================================================
// Tensor Implementation
// ============================================================================

Tensor::Tensor(const std::vector<int>& dims, DataType type)
	: dimensions(dims)
	, dataType(type)
{
	size = 1;
	for (int dim : dimensions)
		size *= dim;
	
	switch (dataType)
	{
	case DataType::FLOAT32:
		floatData.resize(size, 0.0f);
		break;
	case DataType::BOOL:
		boolData.resize(size, false);
		break;
	case DataType::INT32:
		intData.resize(size, 0);
		break;
	}
}

Tensor::~Tensor()
{
}

float Tensor::getFloat(size_t index) const
{
	if (index >= size)
		throw love::Exception("Index out of bounds");
	
	switch (dataType)
	{
	case DataType::FLOAT32:
		return floatData[index];
	case DataType::BOOL:
		return boolData[index] ? 1.0f : 0.0f;
	case DataType::INT32:
		return static_cast<float>(intData[index]);
	default:
		return 0.0f;
	}
}

bool Tensor::getBool(size_t index) const
{
	if (index >= size)
		throw love::Exception("Index out of bounds");
	
	switch (dataType)
	{
	case DataType::BOOL:
		return boolData[index];
	case DataType::FLOAT32:
		return floatData[index] != 0.0f;
	case DataType::INT32:
		return intData[index] != 0;
	default:
		return false;
	}
}

int Tensor::getInt(size_t index) const
{
	if (index >= size)
		throw love::Exception("Index out of bounds");
	
	switch (dataType)
	{
	case DataType::INT32:
		return intData[index];
	case DataType::FLOAT32:
		return static_cast<int>(floatData[index]);
	case DataType::BOOL:
		return boolData[index] ? 1 : 0;
	default:
		return 0;
	}
}

void Tensor::setFloat(size_t index, float value)
{
	if (index >= size)
		throw love::Exception("Index out of bounds");
	
	switch (dataType)
	{
	case DataType::FLOAT32:
		floatData[index] = value;
		break;
	case DataType::BOOL:
		boolData[index] = (value != 0.0f);
		break;
	case DataType::INT32:
		intData[index] = static_cast<int>(value);
		break;
	}
}

void Tensor::setBool(size_t index, bool value)
{
	if (index >= size)
		throw love::Exception("Index out of bounds");
	
	switch (dataType)
	{
	case DataType::BOOL:
		boolData[index] = value;
		break;
	case DataType::FLOAT32:
		floatData[index] = value ? 1.0f : 0.0f;
		break;
	case DataType::INT32:
		intData[index] = value ? 1 : 0;
		break;
	}
}

void Tensor::setInt(size_t index, int value)
{
	if (index >= size)
		throw love::Exception("Index out of bounds");
	
	switch (dataType)
	{
	case DataType::INT32:
		intData[index] = value;
		break;
	case DataType::FLOAT32:
		floatData[index] = static_cast<float>(value);
		break;
	case DataType::BOOL:
		boolData[index] = (value != 0);
		break;
	}
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other) const
{
	if (dimensions != other->dimensions)
		throw love::Exception("Tensor dimensions must match for addition");
	
	auto result = std::make_shared<Tensor>(dimensions, DataType::FLOAT32);
	
	for (size_t i = 0; i < size; ++i)
	{
		result->setFloat(i, getFloat(i) + other->getFloat(i));
	}
	
	return result;
}

std::shared_ptr<Tensor> Tensor::multiply(std::shared_ptr<Tensor> other) const
{
	if (dimensions != other->dimensions)
		throw love::Exception("Tensor dimensions must match for element-wise multiplication");
	
	auto result = std::make_shared<Tensor>(dimensions, DataType::FLOAT32);
	
	for (size_t i = 0; i < size; ++i)
	{
		result->setFloat(i, getFloat(i) * other->getFloat(i));
	}
	
	return result;
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) const
{
	// Matrix multiplication for 2D tensors
	if (dimensions.size() != 2 || other->dimensions.size() != 2)
		throw love::Exception("Matrix multiplication requires 2D tensors");
	
	if (dimensions[1] != other->dimensions[0])
		throw love::Exception("Inner dimensions must match for matrix multiplication");
	
	int m = dimensions[0];
	int n = dimensions[1];
	int p = other->dimensions[1];
	
	std::vector<int> resultDims = {m, p};
	auto result = std::make_shared<Tensor>(resultDims, DataType::FLOAT32);
	
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < p; ++j)
		{
			float sum = 0.0f;
			for (int k = 0; k < n; ++k)
			{
				sum += getFloat(i * n + k) * other->getFloat(k * p + j);
			}
			result->setFloat(i * p + j, sum);
		}
	}
	
	return result;
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int>& newDims) const
{
	size_t newSize = 1;
	for (int dim : newDims)
		newSize *= dim;
	
	if (newSize != size)
		throw love::Exception("New dimensions must have the same total size");
	
	auto result = std::make_shared<Tensor>(newDims, dataType);
	
	// Copy data
	for (size_t i = 0; i < size; ++i)
	{
		switch (dataType)
		{
		case DataType::FLOAT32:
			result->setFloat(i, getFloat(i));
			break;
		case DataType::BOOL:
			result->setBool(i, getBool(i));
			break;
		case DataType::INT32:
			result->setInt(i, getInt(i));
			break;
		}
	}
	
	return result;
}

std::shared_ptr<Tensor> Tensor::transpose(const std::vector<int>& axes) const
{
	// For 2D tensors, simple transpose
	if (dimensions.size() == 2 && axes.empty())
	{
		std::vector<int> newDims = {dimensions[1], dimensions[0]};
		auto result = std::make_shared<Tensor>(newDims, dataType);
		
		for (int i = 0; i < dimensions[0]; ++i)
		{
			for (int j = 0; j < dimensions[1]; ++j)
			{
				size_t oldIdx = i * dimensions[1] + j;
				size_t newIdx = j * dimensions[0] + i;
				
				switch (dataType)
				{
				case DataType::FLOAT32:
					result->setFloat(newIdx, getFloat(oldIdx));
					break;
				case DataType::BOOL:
					result->setBool(newIdx, getBool(oldIdx));
					break;
				case DataType::INT32:
					result->setInt(newIdx, getInt(oldIdx));
					break;
				}
			}
		}
		
		return result;
	}
	
	throw love::Exception("General transpose not yet implemented");
}

float Tensor::sum() const
{
	float result = 0.0f;
	for (size_t i = 0; i < size; ++i)
	{
		result += getFloat(i);
	}
	return result;
}

float Tensor::mean() const
{
	if (size == 0)
		return 0.0f;
	return sum() / static_cast<float>(size);
}

float Tensor::max() const
{
	if (size == 0)
		return 0.0f;
	
	float maxVal = getFloat(0);
	for (size_t i = 1; i < size; ++i)
	{
		float val = getFloat(i);
		if (val > maxVal)
			maxVal = val;
	}
	return maxVal;
}

float Tensor::min() const
{
	if (size == 0)
		return 0.0f;
	
	float minVal = getFloat(0);
	for (size_t i = 1; i < size; ++i)
	{
		float val = getFloat(i);
		if (val < minVal)
			minVal = val;
	}
	return minVal;
}

std::string Tensor::toString() const
{
	std::ostringstream oss;
	oss << "Tensor(";
	for (size_t i = 0; i < dimensions.size(); ++i)
	{
		oss << dimensions[i];
		if (i < dimensions.size() - 1)
			oss << "x";
	}
	oss << ", type=";
	switch (dataType)
	{
	case DataType::FLOAT32:
		oss << "float32";
		break;
	case DataType::BOOL:
		oss << "bool";
		break;
	case DataType::INT32:
		oss << "int32";
		break;
	}
	oss << ")";
	return oss.str();
}

void Tensor::fill(float value)
{
	for (size_t i = 0; i < size; ++i)
	{
		setFloat(i, value);
	}
}

void Tensor::randomize(float min, float max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	
	for (size_t i = 0; i < size; ++i)
	{
		setFloat(i, dis(gen));
	}
}

size_t Tensor::computeFlatIndex(const std::vector<int>& indices) const
{
	if (indices.size() != dimensions.size())
		throw love::Exception("Index dimensions mismatch");
	
	size_t flatIndex = 0;
	size_t multiplier = 1;
	
	for (int i = dimensions.size() - 1; i >= 0; --i)
	{
		if (indices[i] < 0 || indices[i] >= dimensions[i])
			throw love::Exception("Index out of bounds");
		
		flatIndex += indices[i] * multiplier;
		multiplier *= dimensions[i];
	}
	
	return flatIndex;
}

std::vector<int> Tensor::computeMultiIndex(size_t flatIndex) const
{
	std::vector<int> indices(dimensions.size());
	
	for (int i = dimensions.size() - 1; i >= 0; --i)
	{
		indices[i] = flatIndex % dimensions[i];
		flatIndex /= dimensions[i];
	}
	
	return indices;
}

} // namespace tensorlogic
} // namespace love
