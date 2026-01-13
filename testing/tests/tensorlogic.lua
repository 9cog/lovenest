-- love.tensorlogic tests

local love_tensorlogic = {
	-- Tests for module existence
	isModuleAvailable = function(test)
		test:assertNotNil(love.tensorlogic)
	end,

	-- Tests for tensor creation
	newTensor = function(test)
		local t = love.tensorlogic.newTensor({2, 3})
		test:assertNotNil(t)
		test:assertEquals('Tensor', t:type())
		
		local dims = t:getDimensions()
		test:assertEquals(2, #dims, 'check dimensions length')
		test:assertEquals(2, dims[1], 'check first dimension')
		test:assertEquals(3, dims[2], 'check second dimension')
		
		test:assertEquals(2, t:getRank(), 'check tensor rank')
		test:assertEquals(6, t:getSize(), 'check tensor size')
	end,

	newTensorWithData = function(test)
		local data = {1, 2, 3, 4}
		local t = love.tensorlogic.newTensor({2, 2}, data)
		test:assertNotNil(t)
		
		test:assertEquals(1, t:get(1), 'check element at index 1')
		test:assertEquals(2, t:get(2), 'check element at index 2')
		test:assertEquals(3, t:get(3), 'check element at index 3')
		test:assertEquals(4, t:get(4), 'check element at index 4')
	end,

	newBoolTensor = function(test)
		local t = love.tensorlogic.newBoolTensor({2, 2})
		test:assertNotNil(t)
		test:assertEquals('bool', t:getType(), 'check tensor type')
		
		local dims = t:getDimensions()
		test:assertEquals(2, #dims, 'check dimensions length')
	end,

	-- Tests for tensor operations
	tensorAdd = function(test)
		local a = love.tensorlogic.newTensor({2, 2}, {1, 2, 3, 4})
		local b = love.tensorlogic.newTensor({2, 2}, {2, 3, 4, 5})
		
		local c = a:add(b)
		test:assertNotNil(c)
		
		test:assertEquals(3, c:get(1), 'check sum element 1')
		test:assertEquals(5, c:get(2), 'check sum element 2')
		test:assertEquals(7, c:get(3), 'check sum element 3')
		test:assertEquals(9, c:get(4), 'check sum element 4')
	end,

	tensorMultiply = function(test)
		local a = love.tensorlogic.newTensor({2, 2}, {2, 3, 4, 5})
		local b = love.tensorlogic.newTensor({2, 2}, {2, 2, 2, 2})
		
		local c = a:multiply(b)
		test:assertNotNil(c)
		
		test:assertEquals(4, c:get(1), 'check product element 1')
		test:assertEquals(6, c:get(2), 'check product element 2')
		test:assertEquals(8, c:get(3), 'check product element 3')
		test:assertEquals(10, c:get(4), 'check product element 4')
	end,

	tensorMatmul = function(test)
		-- Test matrix multiplication: [2x2] @ [2x2]
		local a = love.tensorlogic.newTensor({2, 2}, {1, 2, 3, 4})
		local b = love.tensorlogic.newTensor({2, 2}, {5, 6, 7, 8})
		
		local c = a:matmul(b)
		test:assertNotNil(c)
		
		-- Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
		--                = [[19, 22], [43, 50]]
		test:assertEquals(19, c:get(1), 'check matmul element [0,0]')
		test:assertEquals(22, c:get(2), 'check matmul element [0,1]')
		test:assertEquals(43, c:get(3), 'check matmul element [1,0]')
		test:assertEquals(50, c:get(4), 'check matmul element [1,1]')
	end,

	-- Tests for tensor shape operations
	tensorReshape = function(test)
		local t = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})
		
		local reshaped = t:reshape({3, 2})
		test:assertNotNil(reshaped)
		
		local dims = reshaped:getDimensions()
		test:assertEquals(3, dims[1], 'check reshaped dimension 1')
		test:assertEquals(2, dims[2], 'check reshaped dimension 2')
		
		-- Data should remain the same
		test:assertEquals(1, reshaped:get(1))
		test:assertEquals(6, reshaped:get(6))
	end,

	tensorTranspose = function(test)
		local t = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})
		
		local transposed = t:transpose()
		test:assertNotNil(transposed)
		
		local dims = transposed:getDimensions()
		test:assertEquals(3, dims[1], 'check transposed dimension 1')
		test:assertEquals(2, dims[2], 'check transposed dimension 2')
	end,

	-- Tests for statistical operations
	tensorSum = function(test)
		local t = love.tensorlogic.newTensor({2, 2}, {1, 2, 3, 4})
		
		local sum = t:sum()
		test:assertEquals(10, sum, 'check tensor sum')
	end,

	tensorMean = function(test)
		local t = love.tensorlogic.newTensor({2, 2}, {2, 4, 6, 8})
		
		local mean = t:mean()
		test:assertEquals(5, mean, 'check tensor mean')
	end,

	tensorMax = function(test)
		local t = love.tensorlogic.newTensor({2, 2}, {1, 5, 3, 2})
		
		local max = t:max()
		test:assertEquals(5, max, 'check tensor max')
	end,

	tensorMin = function(test)
		local t = love.tensorlogic.newTensor({2, 2}, {5, 2, 8, 3})
		
		local min = t:min()
		test:assertEquals(2, min, 'check tensor min')
	end,

	-- Tests for tensor modification
	tensorSet = function(test)
		local t = love.tensorlogic.newTensor({2, 2})
		
		t:set(1, 10)
		t:set(2, 20)
		t:set(3, 30)
		t:set(4, 40)
		
		test:assertEquals(10, t:get(1), 'check set element 1')
		test:assertEquals(20, t:get(2), 'check set element 2')
		test:assertEquals(30, t:get(3), 'check set element 3')
		test:assertEquals(40, t:get(4), 'check set element 4')
	end,

	tensorFill = function(test)
		local t = love.tensorlogic.newTensor({2, 2})
		t:fill(7.5)
		
		test:assertEquals(7.5, t:get(1), 'check filled element 1')
		test:assertEquals(7.5, t:get(2), 'check filled element 2')
		test:assertEquals(7.5, t:get(3), 'check filled element 3')
		test:assertEquals(7.5, t:get(4), 'check filled element 4')
	end,

	tensorRandomize = function(test)
		local t = love.tensorlogic.newTensor({3, 3})
		t:randomize(0, 10)
		
		-- Check that all values are in range
		for i = 1, 9 do
			local val = t:get(i)
			test:assertRange(val, 0, 10, 'check randomized value in range')
		end
	end,

	-- Tests for logical operations
	logicalAnd = function(test)
		local a = love.tensorlogic.newBoolTensor({2, 2}, {true, true, false, false})
		local b = love.tensorlogic.newBoolTensor({2, 2}, {true, false, true, false})
		
		local c = love.tensorlogic.logicalAnd(a, b)
		test:assertNotNil(c)
		
		-- Expected: {true, false, false, false}
		test:assertEquals(1, c:get(1), 'check AND result 1')
		test:assertEquals(0, c:get(2), 'check AND result 2')
		test:assertEquals(0, c:get(3), 'check AND result 3')
		test:assertEquals(0, c:get(4), 'check AND result 4')
	end,

	logicalOr = function(test)
		local a = love.tensorlogic.newBoolTensor({2, 2}, {true, true, false, false})
		local b = love.tensorlogic.newBoolTensor({2, 2}, {true, false, true, false})
		
		local c = love.tensorlogic.logicalOr(a, b)
		test:assertNotNil(c)
		
		-- Expected: {true, true, true, false}
		test:assertEquals(1, c:get(1), 'check OR result 1')
		test:assertEquals(1, c:get(2), 'check OR result 2')
		test:assertEquals(1, c:get(3), 'check OR result 3')
		test:assertEquals(0, c:get(4), 'check OR result 4')
	end,

	logicalNot = function(test)
		local a = love.tensorlogic.newBoolTensor({2, 2}, {true, false, true, false})
		
		local b = love.tensorlogic.logicalNot(a)
		test:assertNotNil(b)
		
		-- Expected: {false, true, false, true}
		test:assertEquals(0, b:get(1), 'check NOT result 1')
		test:assertEquals(1, b:get(2), 'check NOT result 2')
		test:assertEquals(0, b:get(3), 'check NOT result 3')
		test:assertEquals(1, b:get(4), 'check NOT result 4')
	end,

	-- Tests for Einstein summation
	einsumMatmul = function(test)
		-- Test matrix multiplication using einsum notation
		local a = love.tensorlogic.newTensor({2, 3}, {1, 2, 3, 4, 5, 6})
		local b = love.tensorlogic.newTensor({3, 2}, {7, 8, 9, 10, 11, 12})
		
		local c = love.tensorlogic.einsum("ij,jk->ik", {a, b})
		test:assertNotNil(c)
		
		-- Should result in a 2x2 matrix
		local dims = c:getDimensions()
		test:assertEquals(2, dims[1], 'check result dimension 1')
		test:assertEquals(2, dims[2], 'check result dimension 2')
	end,

	-- Tests for tensor string representation
	tensorToString = function(test)
		local t = love.tensorlogic.newTensor({2, 3})
		local str = tostring(t)
		test:assertNotNil(str)
		-- Should contain dimension information
		test:assertTrue(string.find(str, "Tensor") ~= nil, 'check string contains "Tensor"')
		test:assertTrue(string.find(str, "2") ~= nil, 'check string contains first dimension')
		test:assertTrue(string.find(str, "3") ~= nil, 'check string contains second dimension')
	end,
}

return love_tensorlogic
