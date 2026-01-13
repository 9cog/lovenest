--[[
Tensor Logic Example: Neural-Symbolic Rule Learning

This example demonstrates the core concepts of Tensor Logic:
1. Representing symbolic rules as tensor operations
2. Combining neural embeddings with logical inference
3. Learning from examples using tensor operations

Based on Pedro Domingos' Tensor Logic framework:
https://tensor-logic.org/
]]

function love.load()
	print("=== Tensor Logic Neural-Symbolic AI Example ===\n")
	
	-- Example 1: Basic Tensor Operations
	print("1. Basic Tensor Operations:")
	print("Creating a 3x3 tensor...")
	local t1 = love.tensorlogic.newTensor({3, 3})
	t1:randomize(0, 10)
	print("  Tensor:", tostring(t1))
	print("  Sum:", t1:sum())
	print("  Mean:", t1:mean())
	print("  Max:", t1:max())
	print("  Min:", t1:min())
	print()
	
	-- Example 2: Matrix Operations (Foundation of Neural Networks)
	print("2. Matrix Operations (Neural Network Foundation):")
	print("Matrix multiplication: simulating a simple neural layer")
	-- Input: 2 samples, 3 features
	local inputs = love.tensorlogic.newTensor({2, 3}, {
		1, 2, 3,  -- sample 1
		4, 5, 6   -- sample 2
	})
	-- Weights: 3 inputs, 2 outputs
	local weights = love.tensorlogic.newTensor({3, 2}, {
		0.1, 0.2,  -- weights for input 1
		0.3, 0.4,  -- weights for input 2
		0.5, 0.6   -- weights for input 3
	})
	local outputs = inputs:matmul(weights)
	print("  Inputs:", tostring(inputs))
	print("  Weights:", tostring(weights))
	print("  Outputs:", tostring(outputs))
	print("  Output values:")
	for i = 1, outputs:getSize() do
		print(string.format("    Output[%d] = %.2f", i, outputs:get(i)))
	end
	print()
	
	-- Example 3: Logical Rules as Tensor Operations
	print("3. Symbolic Logic with Boolean Tensors:")
	print("Representing logical rules: (A AND B) OR (C AND NOT D)")
	
	-- Create boolean tensors for propositions
	local A = love.tensorlogic.newBoolTensor({1, 4}, {true, true, false, false})
	local B = love.tensorlogic.newBoolTensor({1, 4}, {true, false, true, false})
	local C = love.tensorlogic.newBoolTensor({1, 4}, {false, true, false, true})
	local D = love.tensorlogic.newBoolTensor({1, 4}, {true, true, true, true})
	
	-- Compute logical expression
	local A_and_B = love.tensorlogic.logicalAnd(A, B)
	local not_D = love.tensorlogic.logicalNot(D)
	local C_and_not_D = love.tensorlogic.logicalAnd(C, not_D)
	local result = love.tensorlogic.logicalOr(A_and_B, C_and_not_D)
	
	print("  A:", showBoolTensor(A))
	print("  B:", showBoolTensor(B))
	print("  C:", showBoolTensor(C))
	print("  D:", showBoolTensor(D))
	print("  (A AND B):", showBoolTensor(A_and_B))
	print("  (NOT D):", showBoolTensor(not_D))
	print("  (C AND NOT D):", showBoolTensor(C_and_not_D))
	print("  Final Result:", showBoolTensor(result))
	print()
	
	-- Example 4: Knowledge Graph Representation
	print("4. Knowledge Graph as Tensor Relations:")
	print("Representing 'is-a' relationships:")
	-- Entities: Dog, Cat, Animal, Mammal
	-- Relations represented as adjacency matrix
	local relations = love.tensorlogic.newBoolTensor({4, 4}, {
		-- Dog  Cat  Animal Mammal
		false, false, true,  true,   -- Dog is-a Animal, Dog is-a Mammal
		false, false, true,  true,   -- Cat is-a Animal, Cat is-a Mammal
		false, false, false, false,  -- Animal
		false, false, true,  false   -- Mammal is-a Animal
	})
	print("  Knowledge graph adjacency matrix:", tostring(relations))
	print("  Dog is Animal?", relations:get(3) == 1)  -- [0, 2] in matrix
	print("  Cat is Mammal?", relations:get(8) == 1)  -- [1, 3] in matrix
	print()
	
	-- Example 5: Einstein Summation (Advanced Tensor Logic)
	print("5. Einstein Summation for Complex Operations:")
	print("Using einsum notation for efficient tensor contractions")
	
	-- Simple matrix multiplication using einsum
	local A_matrix = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})
	local B_matrix = love.tensorlogic.newTensor({3, 2}, {1, 0, 0, 1, 1, 1})
	local C_matrix = love.tensorlogic.einsum("ij,jk->ik", {A_matrix, B_matrix})
	
	print("  A:", tostring(A_matrix))
	print("  B:", tostring(B_matrix))
	print("  C = einsum('ij,jk->ik', A, B):", tostring(C_matrix))
	print()
	
	-- Example 6: Tensor Shape Manipulation
	print("6. Tensor Reshaping and Transposition:")
	local original = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})
	print("  Original (2x3):", showTensorData(original))
	
	local reshaped = original:reshape({3, 2})
	print("  Reshaped (3x2):", showTensorData(reshaped))
	
	local transposed = original:transpose()
	print("  Transposed (3x2):", showTensorData(transposed))
	print()
	
	print("=== Key Concepts of Tensor Logic ===")
	print("1. Unification: Neural networks and logic use the same tensor operations")
	print("2. Scalability: Tensor operations scale to large datasets")
	print("3. Differentiability: All operations are differentiable for learning")
	print("4. Composability: Complex operations built from simple tensor primitives")
	print()
	print("For more information, see: https://tensor-logic.org/")
	print()
	print("Press ESC to exit")
end

function showBoolTensor(t)
	local result = "["
	for i = 1, t:getSize() do
		if i > 1 then result = result .. ", " end
		result = result .. (t:get(i) == 1 and "T" or "F")
	end
	return result .. "]"
end

function showTensorData(t)
	local result = "["
	for i = 1, t:getSize() do
		if i > 1 then result = result .. ", " end
		result = result .. string.format("%.1f", t:get(i))
	end
	return result .. "]"
end

function love.keypressed(key)
	if key == "escape" then
		love.event.quit()
	end
end

function love.draw()
	love.graphics.print("Check the console for tensor logic examples", 10, 10)
	love.graphics.print("Press ESC to exit", 10, 30)
end
