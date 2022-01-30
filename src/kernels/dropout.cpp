/*
 * dropout.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "../vectors/simd_macros.hpp"

namespace
{
	using namespace avocado::backend;

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_dropoutForward(const ContextDescriptor &context, const DropoutDescriptor &config, const TensorDescriptor &xDesc,
			const MemoryDescriptor &xMem, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &states)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t cpu_dropoutBackward(const ContextDescriptor &context, const DropoutDescriptor &config, const TensorDescriptor &dyDesc,
			const MemoryDescriptor &dyMem, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const MemoryDescriptor &states)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

