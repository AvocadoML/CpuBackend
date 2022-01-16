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

	avStatus_t dropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
			const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t dropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
			const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

