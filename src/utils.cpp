/*
 * utils.cpp
 *
 *  Created on: Jan 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include "kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

namespace
{
	using namespace avocado::backend;
	bool supports_simd(avDeviceProperty_t prop)
	{
		bool result = false;
		avStatus_t status = cpuGetDeviceProperty(prop, &result);
		assert(status == AVOCADO_STATUS_SUCCESS);
		return result;
	}
	SimdLevel check_supported_simd_level()
	{
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ))
//			return SimdLevel::AVX512VL_BW_DQ;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_AVX512_F))
//			return SimdLevel::AVX512F;
		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_AVX2))
			return SimdLevel::AVX2;
		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION))
			return SimdLevel::F16C;
		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_AVX))
			return SimdLevel::AVX;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE42))
//			return SimdLevel::SSE42;
		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE41))
			return SimdLevel::SSE41;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSSE3))
//			return SimdLevel::SSSE3;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE3))
//			return SimdLevel::SSE3;
		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE2))
			return SimdLevel::SSE2;
//		if (supports_simd(AVOCADO_DEVICE_SUPPORTS_SSE))
//			return SimdLevel::SSE;
		return SimdLevel::NONE;
	}

}

namespace avocado
{
	namespace backend
	{
//		avStatus_t cpu_getConvolutionWorkspaceSize(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
//				const cpu::TensorDescriptor &wDesc, bool inferenceOnly, avSize_t *result)
//		{
//			if (result == nullptr)
//				return AVOCADO_STATUS_BAD_PARAM;
//			result[0] = 0;
//
//			avConvolutionAlgorithm_t algorithm = getConvolutionAlgorithm(config, xDesc, wDesc);
//			if (algorithm == AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED)
//			{
//				switch (wDesc.dtype())
//				{
//					case AVOCADO_DTYPE_FLOAT16:
//					case AVOCADO_DTYPE_BFLOAT16:
//					case AVOCADO_DTYPE_FLOAT32:
//					case AVOCADO_DTYPE_FLOAT64:
//					{
//						if (is_conv(3, wDesc))
//						{
//							std::array<cpu::TensorDescriptor, 3> tmp = getWinogradMatricesShape(config, xDesc, wDesc, 4);
//							result[0] = tmp[0].sizeInBytes() + tmp[1].sizeInBytes() + tmp[2].sizeInBytes();
//						}
//						if (is_conv(5, wDesc))
//						{
//							std::array<cpu::TensorDescriptor, 3> tmp = getWinogradMatricesShape(config, xDesc, wDesc, 2);
//							result[0] = tmp[0].sizeInBytes() + tmp[1].sizeInBytes() + tmp[2].sizeInBytes();
//						}
//						break;
//					}
//					default:
//						break;
//				}
//			}
//			if (algorithm == AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM)
//			{
//				// forward workspace
//				cpu::TensorDescriptor input_matrix_shape = getExplicitGemmMatrixShape(config, xDesc, wDesc);
//				cpu::TensorDescriptor output_shape = config.getOutputShape(xDesc, wDesc);
//
//				// backward workspace
//				cpu::ConvolutionDescriptor cfg = getBackwardConfig(config, wDesc);
//
//				cpu::TensorDescriptor backward_wdesc(wDesc);
//				std::swap(backward_wdesc[0], backward_wdesc[backward_wdesc.nbDims() - 1]);
//				cpu::TensorDescriptor output_matrix_shape = getExplicitGemmMatrixShape(cfg, output_shape, backward_wdesc);
//
//				// for forward - input_matrix_shape + output_shape
//				// for backward - output_matrix_shape + weight_shape
//				int forward_workspace = input_matrix_shape.sizeInBytes() + output_shape.sizeInBytes();
//				if (inferenceOnly)
//					result[0] = forward_workspace;
//				else
//				{
//					int backward_workspace = output_matrix_shape.sizeInBytes() + wDesc.sizeInBytes();
//					int update_workspace = input_matrix_shape.sizeInBytes();
//					result[0] = std::max(forward_workspace, std::max(backward_workspace, update_workspace));
//				}
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}

		/*
		 *
		 */
		bool is_conv(int expectedSize, const cpu::TensorDescriptor &wDesc) noexcept
		{
			for (int i = 0; i < wDesc.nbDims() - 2; i++)
				if (wDesc.dimension(1 + i) != expectedSize)
					return false;
			return true;
		}
//		cpu::ConvolutionDescriptor getBackwardConfig(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &wDesc)
//		{
//			cpu::ConvolutionDescriptor result(config);
//			if (result.mode == AVOCADO_CONVOLUTION_MODE)
//				result.mode = AVOCADO_CROSS_CORRELATION_MODE;
//			else
//				result.mode = AVOCADO_CONVOLUTION_MODE;
//			for (int i = 0; i < wDesc.nbDims() - 2; i++)
//				result.padding[i] = -(wDesc.dimension(1 + i) - 1) * result.dilation[i] - result.padding[i];
//			return result;
//		}
//		avConvolutionAlgorithm_t getConvolutionAlgorithm(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
//				const cpu::TensorDescriptor &wDesc)
//		{
//			if (config.algorithm == AVOCADO_CONVOLUTION_ALGORITHM_AUTO)
//			{
//				if (config.isStrided() or config.isDilated())
//					return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
//
//				if (is_conv(3, wDesc))
//				{
//					if (wDesc.lastDim() > 4)
//						return AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED;
//					else
//						return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
//				}
//				if (is_conv(5, wDesc))
//				{
//					if (wDesc.lastDim() > 4)
//						return AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED;
//					else
//						return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
//				}
//				return AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM;
//			}
//			else
//				return config.algorithm;
//		}
//		int getWinogradTransformSize(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc, const cpu::TensorDescriptor &wDesc)
//		{
//			if (is_conv(3, wDesc))
//			{
//				return 4; // TODO maybe add dispatching of other tile sizes
//			}
//			if (is_conv(5, wDesc))
//				return 2;
//			return 0;
//		}
//		std::array<cpu::TensorDescriptor, 3> getWinogradMatricesShape(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
//				const cpu::TensorDescriptor &wDesc, int transformSize)
//		{
//			assert(wDesc.dimension(1) == wDesc.dimension(2));
//			assert(wDesc.nbDims() == 3 || wDesc.nbDims() == 4 || wDesc.nbDims() == 5);
//			int tile_size = transformSize + wDesc.dimension(1) - 1;
//			int nb_of_matrices = tile_size * tile_size;
//
//			int nb_of_tiles = xDesc.firstDim();
//			for (int i = 1; i < xDesc.nbDims() - 1; i++)
//				nb_of_tiles *= (xDesc.dimension(i) + transformSize - 1) / transformSize;
//
//			std::array<cpu::TensorDescriptor, 3> result;
//			result[0] = cpu::TensorDescriptor( { nb_of_matrices, wDesc.firstDim(), wDesc.lastDim() }, wDesc.dtype());
//			result[1] = cpu::TensorDescriptor( { nb_of_matrices, nb_of_tiles, wDesc.lastDim() }, wDesc.dtype());
//			result[2] = cpu::TensorDescriptor( { nb_of_matrices, nb_of_tiles, wDesc.firstDim() }, wDesc.dtype());
//			return result;
//		}
//		cpu::TensorDescriptor getExplicitGemmMatrixShape(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
//				const cpu::TensorDescriptor &wDesc)
//		{
//			cpu::TensorDescriptor output_shape = config.getOutputShape(xDesc, wDesc);
//			int output_tiles = 1;
//			for (int i = 0; i < output_shape.nbDims() - 1; i++)
//				output_tiles *= output_shape.dimension(i);
//
//			int filter_tiles = 1;
//			for (int i = 1; i < wDesc.nbDims(); i++)
//				filter_tiles *= wDesc.dimension(i);
//			return cpu::TensorDescriptor( { output_tiles, filter_tiles }, xDesc.dtype());
//		}

		std::string toString(SimdLevel sl)
		{
			switch (sl)
			{
				default:
				case SimdLevel::NONE:
					return "NONE";
				case SimdLevel::SSE:
					return "SSE";
				case SimdLevel::SSE2:
					return "SSE2";
				case SimdLevel::SSE3:
					return "SSE3";
				case SimdLevel::SSSE3:
					return "SSSE3";
				case SimdLevel::SSE41:
					return "SSE41";
				case SimdLevel::SSE42:
					return "SSE42";
				case SimdLevel::AVX:
					return "AVX";
				case SimdLevel::F16C:
					return "F16C";
				case SimdLevel::AVX2:
					return "AVX2";
				case SimdLevel::AVX512F:
					return "AVX512F";
				case SimdLevel::AVX512VL_BW_DQ:
					return "AVX512VL_BW_DQ";
			}
		}
		SimdLevel simdLevelFromString(const std::string &str)
		{
			if (str == "AVX512VL_BW_DQ")
				return SimdLevel::AVX512VL_BW_DQ;
			if (str == "AVX512F")
				return SimdLevel::AVX512F;
			if (str == "AVX2")
				return SimdLevel::AVX2;
			if (str == "F16C")
				return SimdLevel::F16C;
			if (str == "AVX")
				return SimdLevel::AVX;
			if (str == "SSE42")
				return SimdLevel::SSE42;
			if (str == "SSE41")
				return SimdLevel::SSE41;
			if (str == "SSSE3")
				return SimdLevel::SSSE3;
			if (str == "SSE3")
				return SimdLevel::SSE3;
			if (str == "SSE2")
				return SimdLevel::SSE2;
			if (str == "SSE")
				return SimdLevel::SSE;
			return SimdLevel::NONE;
		}

		SimdLevel getSimdSupport() noexcept
		{
			static const SimdLevel supported_simd_level = check_supported_simd_level();
			return supported_simd_level;
		}
	}
}

