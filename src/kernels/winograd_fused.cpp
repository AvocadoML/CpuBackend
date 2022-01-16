/*
 * winograd_fused.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"

namespace
{
	using namespace avocado::backend;

}

namespace SIMD_NAMESPACE
{
		avStatus_t winogradFusedForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
} /* namespace avocado */

