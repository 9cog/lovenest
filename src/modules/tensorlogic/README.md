# Tensor Logic Module for LÖVE

## Overview

The Tensor Logic module brings neural-symbolic AI capabilities to LÖVE, implementing concepts from Pedro Domingos' [Tensor Logic framework](https://tensor-logic.org/). This module bridges symbolic reasoning (traditional AI) with neural network operations (deep learning) by representing both as tensor operations.

## Key Concepts

### What is Tensor Logic?

Tensor Logic is a unified framework that represents both:
- **Neural computations** (matrix multiplications, embeddings)
- **Symbolic reasoning** (logical rules, knowledge graphs)

...using the same mathematical foundation: **tensors**.

The core insight is that logical rules can be expressed as Einstein summation operations over Boolean tensors, making them compatible with gradient-based learning.

### Why Tensor Logic?

1. **Unification**: One language for both symbolic and subsymbolic AI
2. **Scalability**: Tensor operations leverage modern GPU acceleration
3. **Differentiability**: All operations support gradient-based learning
4. **Composability**: Complex operations built from simple primitives

## API Reference

### Module Functions

#### `love.tensorlogic.newTensor(dimensions, [data])`

Creates a new floating-point tensor.

**Parameters:**
- `dimensions` (table): Array of dimension sizes, e.g., `{2, 3}` for 2x3 matrix
- `data` (table, optional): Initial values in row-major order

**Returns:**
- `Tensor`: New tensor object

**Example:**
```lua
-- Create a 2x3 tensor with initial values
local t = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})
```

#### `love.tensorlogic.newBoolTensor(dimensions, [data])`

Creates a new Boolean tensor for logical operations.

**Parameters:**
- `dimensions` (table): Array of dimension sizes
- `data` (table of booleans, optional): Initial Boolean values

**Returns:**
- `Tensor`: New Boolean tensor

**Example:**
```lua
local t = love.tensorlogic.newBoolTensor({2, 2}, {true, false, true, false})
```

#### `love.tensorlogic.einsum(equation, tensors)`

Performs Einstein summation (generalized tensor contraction).

**Parameters:**
- `equation` (string): Einstein summation notation, e.g., `"ij,jk->ik"`
- `tensors` (table): Array of input tensors

**Returns:**
- `Tensor`: Result tensor

**Example:**
```lua
-- Matrix multiplication
local a = love.tensorlogic.newTensor({2, 3}, {...})
local b = love.tensorlogic.newTensor({3, 2}, {...})
local c = love.tensorlogic.einsum("ij,jk->ik", {a, b})
```

#### Logical Operations

**`love.tensorlogic.logicalAnd(a, b)`** - Element-wise AND  
**`love.tensorlogic.logicalOr(a, b)`** - Element-wise OR  
**`love.tensorlogic.logicalNot(a)`** - Element-wise NOT

These operate on Boolean tensors.

### Tensor Methods

#### Shape and Properties

- **`tensor:getDimensions()`** - Returns array of dimension sizes
- **`tensor:getRank()`** - Returns number of dimensions
- **`tensor:getSize()`** - Returns total number of elements
- **`tensor:getType()`** - Returns data type ("float32", "bool", or "int32")

#### Data Access

- **`tensor:get(index)`** - Gets value at index (1-indexed)
- **`tensor:set(index, value)`** - Sets value at index (1-indexed)

#### Arithmetic Operations

- **`tensor:add(other)`** - Element-wise addition
- **`tensor:multiply(other)`** - Element-wise multiplication
- **`tensor:matmul(other)`** - Matrix multiplication (for 2D tensors)

#### Shape Operations

- **`tensor:reshape(newDimensions)`** - Reshapes tensor (preserves data)
- **`tensor:transpose([axes])`** - Transposes tensor

#### Statistical Operations

- **`tensor:sum()`** - Sum of all elements
- **`tensor:mean()`** - Mean of all elements
- **`tensor:max()`** - Maximum element
- **`tensor:min()`** - Minimum element

#### Utilities

- **`tensor:fill(value)`** - Fill all elements with value
- **`tensor:randomize([min], [max])`** - Fill with random values
- **`tostring(tensor)`** - Get string representation

## Examples

### Example 1: Neural Network Layer

Simulate a simple neural network layer using matrix multiplication:

```lua
-- Input: 2 samples, 3 features each
local inputs = love.tensorlogic.newTensor({2, 3}, {
    1, 2, 3,  -- sample 1
    4, 5, 6   -- sample 2
})

-- Weights: 3 inputs -> 2 outputs
local weights = love.tensorlogic.newTensor({3, 2}, {
    0.1, 0.2,
    0.3, 0.4,
    0.5, 0.6
})

-- Forward pass
local outputs = inputs:matmul(weights)
print("Layer outputs:", outputs:sum())
```

### Example 2: Logical Rules

Represent and evaluate logical expressions:

```lua
-- Propositions
local A = love.tensorlogic.newBoolTensor({1, 4}, {true, true, false, false})
local B = love.tensorlogic.newBoolTensor({1, 4}, {true, false, true, false})

-- Logical rule: A AND B
local result = love.tensorlogic.logicalAnd(A, B)

-- Check results
for i = 1, result:getSize() do
    print(string.format("Case %d: %s", i, result:get(i) == 1 and "true" or "false"))
end
```

### Example 3: Knowledge Graph

Represent entity relationships as tensor operations:

```lua
-- Entities: Dog, Cat, Animal, Mammal
-- Adjacency matrix for "is-a" relationships
local knowledge = love.tensorlogic.newBoolTensor({4, 4}, {
    -- Dog  Cat  Animal Mammal
    false, false, true,  true,   -- Dog is-a Animal, Mammal
    false, false, true,  true,   -- Cat is-a Animal, Mammal
    false, false, false, false,  -- Animal
    false, false, true,  false   -- Mammal is-a Animal
})

-- Query: Is Dog an Animal?
local is_animal = knowledge:get(3)  -- Row 0, Col 2
print("Dog is Animal:", is_animal == 1)
```

### Example 4: Tensor Manipulation

```lua
-- Create and manipulate tensors
local t = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})

print("Original:", tostring(t))

-- Reshape
local reshaped = t:reshape({3, 2})
print("Reshaped:", tostring(reshaped))

-- Transpose
local transposed = t:transpose()
print("Transposed:", tostring(transposed))

-- Statistics
print("Sum:", t:sum())
print("Mean:", t:mean())
```

## Implementation Details

### Architecture

The module is implemented in C++ with Lua bindings:

- **`TensorLogic.h/cpp`**: Core module and tensor operations
- **`wrap_TensorLogic.h/cpp`**: Lua API bindings
- **Tensor class**: Multi-dimensional array with type support (float32, bool, int32)

### Performance Considerations

- Operations are performed on CPU (no GPU acceleration yet)
- Memory is managed using LÖVE's reference counting
- Suitable for prototyping and small-scale experimentation

### Limitations

- Limited einsum equation support (mainly matrix multiplication)
- No automatic differentiation (yet)
- No GPU acceleration
- Limited to relatively small tensors

## Future Enhancements

Planned features for future versions:

1. **Automatic Differentiation**: Enable gradient-based learning
2. **GPU Acceleration**: Leverage GPU for large-scale computations
3. **Advanced Einsum**: Support more complex tensor contractions
4. **Rule Inference Engine**: Automated logical reasoning
5. **Knowledge Graph Operations**: Graph traversal and pattern matching
6. **Integration with Neural Networks**: Pre-built layers and architectures

## References

- [Tensor Logic: The Language of AI](https://tensor-logic.org/) - Pedro Domingos
- [Tensor Logic Paper](https://arxiv.org/abs/2510.12269) - arXiv preprint
- [Ben Goertzel's Article](https://bengoertzel.substack.com/p/tensor-logic-for-bridging-neural) - Neural-Symbolic Integration

## Testing

Comprehensive unit tests are available in `testing/tests/tensorlogic.lua`. To run tests:

```bash
love testing/
```

Example programs demonstrating usage can be found in `testing/examples/tensorlogic_example.lua`.

## License

This module is part of LÖVE and follows the same license (zlib/libpng license).

## Contributing

Contributions are welcome! Areas that need work:

- Extended einsum support
- Performance optimizations
- Additional neural network operations
- Knowledge graph query operations
- Documentation improvements

---

For questions or issues, please use the LÖVE GitHub repository issue tracker.
