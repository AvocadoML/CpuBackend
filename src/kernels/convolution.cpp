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

	bool is_conv(int expectedSize, const cpu::TensorDescriptor &wDesc)
	{
		for (int i = 0; i < wDesc.nbDims() - 2; i++)
			if (wDesc.dimension(1 + i) != expectedSize)
				return false;
		return true;
	}

	std::array<int, 3> getWinogradWorkspace(int transformSize, const cpu::TensorDescriptor &xDesc, const std::vector<int> &yDesc,
			const cpu::TensorDescriptor &wDesc)
	{
		assert(wDesc.dimension(1) == wDesc.dimension(2));
		assert(wDesc.nbDims() == 3 || wDesc.nbDims() == 4 || wDesc.nbDims() == 5);
		int tile_size = transformSize + wDesc.dimension(1) - 1;
		int nb_of_matrices = tile_size * tile_size;

		int nb_of_tiles = xDesc.firstDim();
		for (int i = 1; i < xDesc.nbDims() - 1; i++)
			nb_of_tiles *= (xDesc.dimension(i) + transformSize - 1) / transformSize;

		int weight_matrix_size = nb_of_matrices * cpu::dataTypeSize(wDesc.dtype()) * wDesc.firstDim() * wDesc.lastDim();
		int input_matrix_size = nb_of_matrices * cpu::dataTypeSize(xDesc.dtype()) * nb_of_tiles * xDesc.lastDim();
		int output_matrix_size = nb_of_matrices * cpu::dataTypeSize(xDesc.dtype()) * nb_of_tiles * yDesc.back();

		return std::array<int, 3>( { weight_matrix_size, input_matrix_size, output_matrix_size });
	}
	avConvolutionAlgorithm_t getConvolutionAlgorithm(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
			const cpu::TensorDescriptor &wDesc)
	{
		if (config.algorithm == AVOCADO_CONVOLUTION_ALGORITHM_AUTO)
		{
//			if (is_conv(3, wDesc))
//			{
//				if (wDesc.lastDim() > 4)
//					return AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED;
//				else
//					return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
//			}
//			if (is_conv(5, wDesc))
//			{
//				if (wDesc.lastDim() > 4)
//					return AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED;
//				else
//					return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
//			}
			return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
		}
		else
			return config.algorithm;
	}

	std::array<int, 2> getExplicitGemmMatrixShape(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
			const cpu::TensorDescriptor &wDesc)
	{
		std::vector<int> output_shape = config.getOutputShape(xDesc, wDesc);
		int output_tiles = 1;
		for (size_t i = 0; i < output_shape.size() - 1; i++)
			output_tiles *= output_shape[i];

		int filter_tiles = 1;
		for (int i = 1; i < wDesc.nbDims(); i++)
			filter_tiles *= wDesc.dimension(i);
		return std::array<int, 2>( { output_tiles, filter_tiles });
	}
}

#if DYNAMIC_ARCH == 0 or (DYNAMIC_ARCH == 1 and defined(COMPILE_COMMON_CODE))
namespace avocado
{
	namespace backend
	{
		avStatus_t cpu_getConvolutionWorkspaceSize(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
				const cpu::TensorDescriptor & wDesc, avSize_t *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			result[0] = 0;

			avConvolutionAlgorithm_t algorithm = getConvolutionAlgorithm(config, xDesc, wDesc);

			std::vector<int> output_shape = config.getOutputShape(xDesc, wDesc);
			if (algorithm == AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED)
			{
				switch (wDesc.dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
					case AVOCADO_DTYPE_BFLOAT16:
					case AVOCADO_DTYPE_FLOAT32:
					case AVOCADO_DTYPE_FLOAT64:
					{
						if (is_conv(3, wDesc))
						{
							std::array<int, 3> tmp = getWinogradWorkspace(4, xDesc, output_shape, wDesc);
							result[0] = tmp[0] + tmp[1] + tmp[2];
						}
						if (is_conv(5, wDesc))
						{
							std::array<int, 3> tmp = getWinogradWorkspace(2, xDesc, output_shape, wDesc);
							result[0] = tmp[0] + tmp[1] + tmp[2];
						}
						break;
					}
					default:
						break;
				}
			}
			if (algorithm == AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM)
			{
				int output_tiles = 1;
				for (size_t i = 0; i < output_shape.size() - 1; i++)
					output_tiles *= output_shape[i];

				int filters_tiles = 1;
				for (int i = 1; i < wDesc.nbDims(); i++)
					filters_tiles *= wDesc.dimension(i);
				result[0] = output_tiles * filters_tiles * cpu::dataTypeSize(wDesc.dtype());
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */
#endif

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_convolutionBiasActivationForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc,
			const MemoryDescriptor &zMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation,
			MemoryDescriptor &workspaceMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t cpu_convolutionForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &workspaceMem)
	{
		avConvolutionAlgorithm_t algorithm = getConvolutionAlgorithm(config, xDesc, wDesc);
		switch (algorithm)
		{
			case AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM:
			{
				avTensorDescriptor_t matrix_desc, filter_desc, output_desc;
				cpuCreateTensorDescriptor(&matrix_desc);
				cpuCreateTensorDescriptor(&filter_desc);
				cpuCreateTensorDescriptor(&output_desc);

//				cpuIm2Row(context, config, wDesc, xDesc, xMem, matrix_desc, workspaceMem);
//				cpuGemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, alpha, matrix_desc, workspaceMem, filter_desc, wMem, beta,
//						output_desc, yMem);
				break;
			}
			case AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED:
			{

				break;
			}
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t cpu_convolutionBackward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *beta,
			const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, MemoryDescriptor &workspaceMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t cpu_convolutionUpdate(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
			const void *beta, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, MemoryDescriptor &workspaceMem)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

