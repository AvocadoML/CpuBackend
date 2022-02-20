/*
 * kernel_definitions.hpp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef KERNELS_KERNEL_DEFINITIONS_HPP_
#define KERNELS_KERNEL_DEFINITIONS_HPP_

#include <CpuBackend/cpu_backend.h>
#include <backend_descriptors.hpp>

#include "vectors/simd_macros.hpp"
#include "utils.hpp"

#if DYNAMIC_ARCH

namespace NAMESPACE_AVX2
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_AVX
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_SSE41
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_SSE2
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_NO_SIMD
{
#include "kernels/kernel_definitions.ipp"
}

#else

namespace SIMD_NAMESPACE
{
#include "kernels/kernel_definitions.ipp"
}

#endif /* DYNAMIC_ARCH */

namespace avocado
{
	namespace backend
	{
		avStatus_t cpu_gemm(const cpu::ContextDescriptor &context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const cpu::TensorDescriptor &aDesc, const cpu::MemoryDescriptor &aMem, const cpu::TensorDescriptor &bDesc,
				const cpu::MemoryDescriptor &bMem, const void *beta, const cpu::TensorDescriptor &cDesc, cpu::MemoryDescriptor &cMem);

		avStatus_t cpu_gemmBatched(const cpu::ContextDescriptor &context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const cpu::TensorDescriptor &aDesc, const cpu::MemoryDescriptor &aMem, const cpu::TensorDescriptor &bDesc,
				const cpu::MemoryDescriptor &bMem, const void *beta, const cpu::TensorDescriptor &cDesc, cpu::MemoryDescriptor &cMem);

		/*
		 *
		 */

		bool is_conv(int expectedSize, const cpu::TensorDescriptor &wDesc) noexcept;
		cpu::ConvolutionDescriptor getBackwardConfig(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &wDesc);
		avConvolutionAlgorithm_t getConvolutionAlgorithm(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
				const cpu::TensorDescriptor &wDesc);
		int getWinogradTransformSize(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
				const cpu::TensorDescriptor &wDesc);
		std::array<cpu::TensorDescriptor, 3> getWinogradMatricesShape(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
				const cpu::TensorDescriptor &wDesc, int transformSize);
		cpu::TensorDescriptor getExplicitGemmMatrixShape(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
				const cpu::TensorDescriptor &wDesc);
	} /* namespace backend */
} /* namespace avocado */

#endif /* KERNELS_KERNEL_DEFINITIONS_HPP_ */
