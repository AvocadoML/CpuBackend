/*
 * convolution.cpp
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

	avStatus_t convolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
			const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
			const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
			const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
			const avActivationType_t activation, avMemoryDescriptor_t workspace)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t convolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
			const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
			const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t convolutionBackward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
			const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
			const void *beta, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, avMemoryDescriptor_t workspaceMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t convolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
			const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem,
			const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

