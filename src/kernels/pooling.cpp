/*
 * pooling.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

#include "../vectors/simd_macros.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_poolingForward(const ContextDescriptor &context, const PoolingDescriptor &config, const void *alpha, const TensorDescriptor &xDesc,
			const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t cpu_poolingBackward(const ContextDescriptor &context, const PoolingDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
			const void *beta, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

