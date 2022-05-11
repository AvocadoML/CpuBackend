/*
 * conv2d_implicit_gemm.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_convolutionImplicitGemmForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc,
			const MemoryDescriptor &zMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
} /* namespace avocado */

