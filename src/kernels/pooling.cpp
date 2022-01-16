/*
 * pooling.cpp
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

	avStatus_t poolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha, const avTensorDescriptor_t xDesc,
			const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t poolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha, const avTensorDescriptor_t xDesc,
			const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
			const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

