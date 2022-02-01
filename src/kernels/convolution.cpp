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
	using namespace avocado::backend::cpu;

	// add bias    : y = alpha3 * activation(alpha1 * x + alpha2 * b + beta1 * z) + beta2 * z
	// convolution : y = activation(alpha1 * conv(x, w) + alpha2 * z + b) + beta * y

	avStatus_t winograd_nonfused_forward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc,
			const MemoryDescriptor &zMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation,
			MemoryDescriptor &workspaceMem)
	{
		const int transform_size = getWinogradTransformSize(config, xDesc, wDesc);
		std::array<cpu::TensorDescriptor, 3> matrix_shape = getWinogradMatricesShape(config, xDesc, wDesc, transform_size);

		MemoryDescriptor weight_matrix(workspaceMem, matrix_shape[0].sizeInBytes(), 0);
		MemoryDescriptor input_matrix(workspaceMem, matrix_shape[1].sizeInBytes(), weight_matrix.size());
		MemoryDescriptor output_matrix(workspaceMem, matrix_shape[2].sizeInBytes(), weight_matrix.size() + input_matrix.size());

		avStatus_t status;
		status = SIMD_NAMESPACE::cpu_winogradWeightTransform(context, config, wDesc, wMem, matrix_shape[0], weight_matrix);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

		status = SIMD_NAMESPACE::cpu_winogradInputTransform(context, config, xDesc, xMem, matrix_shape[1], input_matrix, wDesc);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

		status = cpu_gemmBatched(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, matrix_shape[1], input_matrix, matrix_shape[0],
				weight_matrix, nullptr, matrix_shape[2], output_matrix);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

		status = SIMD_NAMESPACE::cpu_winogradOutputTransform(context, config, alpha1, matrix_shape[2], output_matrix, yDesc, yMem, bDesc, bMem,
				alpha2, zDesc, zMem, beta, activation, wDesc);
		return status;
	}
	avStatus_t winograd_nonfused_forward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &workspaceMem)
	{
		const int transform_size = getWinogradTransformSize(config, xDesc, wDesc);
		std::array<cpu::TensorDescriptor, 3> matrix_shape = getWinogradMatricesShape(config, xDesc, wDesc, transform_size);

		MemoryDescriptor weight_matrix(workspaceMem, matrix_shape[0].sizeInBytes(), 0);
		MemoryDescriptor input_matrix(workspaceMem, matrix_shape[1].sizeInBytes(), weight_matrix.size());
		MemoryDescriptor output_matrix(workspaceMem, matrix_shape[2].sizeInBytes(), weight_matrix.size() + input_matrix.size());

//		MemoryDescriptor weight_matrix(matrix_shape[0].sizeInBytes());
//		MemoryDescriptor input_matrix(matrix_shape[1].sizeInBytes());
//		MemoryDescriptor output_matrix(matrix_shape[2].sizeInBytes());

		std::cout << workspaceMem.size() << " " << weight_matrix.size() << " " << input_matrix.size() << " " << output_matrix.size() << '\n';

		std::cout << workspaceMem.data() << '\n' << weight_matrix.data() << '\n' << input_matrix.data() << '\n' << output_matrix.data() << '\n';

		avStatus_t status;
		status = SIMD_NAMESPACE::cpu_winogradWeightTransform(context, config, wDesc, wMem, matrix_shape[0], weight_matrix);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;
//		for (int i = 0; i < 6; i++)
//		{
//			for (int j = 0; j < 6; j++)
//				std::cout << weight_matrix.data<float>()[matrix_shape[0].getIndex( { i * 6 + j, 0, 0 })] << ' ';
//			std::cout << '\n';
//		}

		status = SIMD_NAMESPACE::cpu_winogradInputTransform(context, config, xDesc, xMem, matrix_shape[1], input_matrix, wDesc);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;
//		std::cout << "-----------------------------------------------------------------\n";
//		for (int i = 0; i < 6; i++)
//		{
//			for (int j = 0; j < 6; j++)
//				std::cout << input_matrix.data<float>()[matrix_shape[1].getIndex( { i * 6 + j, 0, 0 })] << ' ';
//			std::cout << '\n';
//		}

//		std::cout << config.getOutputShape(xDesc, wDesc).toString() << std::endl;
//		for (int i = 0; i < 3; i++)
//			std::cout << matrix_shape[i].toString() << std::endl;
		status = cpu_gemmBatched(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, matrix_shape[1], input_matrix, matrix_shape[0],
				weight_matrix, nullptr, matrix_shape[2], output_matrix);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

//		std::cout << "-----------------------------------------------------------------\n";
//		for (int i = 0; i < 6; i++)
//		{
//			for (int j = 0; j < 6; j++)
//				std::cout << output_matrix.data<float>()[matrix_shape[2].getIndex( { i * 6 + j, 0, 0 })] << ' ';
//			std::cout << '\n';
//		}

		TensorDescriptor empty_desc;
		MemoryDescriptor empty_mem;
		status = SIMD_NAMESPACE::cpu_winogradOutputTransform(context, config, alpha, matrix_shape[2], output_matrix, yDesc, yMem, empty_desc,
				empty_mem, nullptr, empty_desc, empty_mem, beta, AVOCADO_ACTIVATION_LINEAR, wDesc);
		return status;
	}
	avStatus_t winograd_nonfused_backward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *beta,
			const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, MemoryDescriptor &workspaceMem)
	{
		const int transform_size = getWinogradTransformSize(config, dxDesc, wDesc);
		std::array<cpu::TensorDescriptor, 3> matrix_shape = getWinogradMatricesShape(config, dxDesc, wDesc, transform_size);

		MemoryDescriptor weight_matrix(workspaceMem, matrix_shape[0].sizeInBytes(), 0);
		MemoryDescriptor input_matrix(workspaceMem, matrix_shape[1].sizeInBytes(), weight_matrix.size());
		MemoryDescriptor output_matrix(workspaceMem, matrix_shape[2].sizeInBytes(), weight_matrix.size() + input_matrix.size());

		avStatus_t status;
		status = SIMD_NAMESPACE::cpu_winogradWeightTransform(context, config, wDesc, wMem, matrix_shape[0], weight_matrix);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

		status = SIMD_NAMESPACE::cpu_winogradInputTransform(context, config, dyDesc, dyMem, matrix_shape[2], output_matrix, wDesc);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

		status = cpu_gemmBatched(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_N, nullptr, matrix_shape[2], output_matrix,
				matrix_shape[0], weight_matrix, nullptr, matrix_shape[1], input_matrix);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

		TensorDescriptor emptyDesc;
		MemoryDescriptor emptyMem;
		status = SIMD_NAMESPACE::cpu_winogradOutputTransform(context, config, alpha, matrix_shape[1], input_matrix, dxDesc, dxMem, emptyDesc,
				emptyMem, nullptr, emptyDesc, emptyMem, beta, AVOCADO_ACTIVATION_LINEAR, wDesc);
		return status;
	}
	avStatus_t winograd_nonfused_update(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
			const void *beta, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, MemoryDescriptor &workspaceMem)
	{
		const int transform_size = getWinogradTransformSize(config, xDesc, dwDesc);
		std::array<cpu::TensorDescriptor, 3> matrix_shape = getWinogradMatricesShape(config, xDesc, dwDesc, transform_size);

		MemoryDescriptor weight_matrix(workspaceMem, matrix_shape[0].sizeInBytes(), 0);
		MemoryDescriptor input_matrix(workspaceMem, matrix_shape[1].sizeInBytes(), weight_matrix.size());
		MemoryDescriptor output_matrix(workspaceMem, matrix_shape[2].sizeInBytes(), weight_matrix.size() + input_matrix.size());

		avStatus_t status;
		status = SIMD_NAMESPACE::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matrix_shape[2], output_matrix, dwDesc);
		if (status != AVOCADO_STATUS_SUCCESS)
			return status;

//		status = SIMD_NAMESPACE::cpu_winogradInputTransform(context, config, xDesc, xMem, matrix_shape[1], input_matrix, dwDesc);
//		if (status != AVOCADO_STATUS_SUCCESS)
//			return status;
//
//		status = cpu_gemmBatched(context, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_N, nullptr, matrix_shape[0], weight_matrix,
//				matrix_shape[2], output_matrix, nullptr, matrix_shape[1], input_matrix);
//		if (status != AVOCADO_STATUS_SUCCESS)
//			return status;
//
//		status = SIMD_NAMESPACE::cpu_winogradUpdateTransform(context, config, alpha, matrix_shape[0], weight_matrix, beta, dwDesc, dwMem);
		return status;
	}

	avStatus_t explicit_gemm_forward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const MemoryDescriptor &zMem, const void *beta,
			const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation, MemoryDescriptor &workspaceMem)
	{
		TensorDescriptor input_desc = getExplicitGemmMatrixShape(config, xDesc, wDesc);
		TensorDescriptor filter_desc( { wDesc.firstDim(), wDesc.volumeWithoutFirstDim() }, wDesc.dtype());
		TensorDescriptor output_desc( { yDesc.volumeWithoutLastDim(), yDesc.lastDim() }, yDesc.dtype());

		MemoryDescriptor input_matrix(workspaceMem, input_desc.sizeInBytes(), 0);

		const bool beta_is_zero = (getBetaValue<uint32_t>(beta) != 0u);
		if (beta_is_zero) // can write directly to yMem
		{
			if (is_conv(1, wDesc))
				cpu_gemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, input_desc, xMem, filter_desc, wMem, nullptr,
						output_desc, yMem);
			else
			{
				SIMD_NAMESPACE::cpu_im2row(context, config, wDesc, xDesc, xMem, input_desc, input_matrix);
				cpu_gemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, input_desc, input_matrix, filter_desc, wMem, nullptr,
						output_desc, yMem);
			}
			SIMD_NAMESPACE::cpu_addBias(context, nullptr, alpha1, yDesc, yMem, alpha2, bDesc, bMem, yDesc, yMem, nullptr, beta, zMem, activation);
		}
		else
		{
			MemoryDescriptor output_matrix(workspaceMem, output_desc.sizeInBytes(), input_matrix.size());
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
	avStatus_t explicit_gemm_forward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &workspaceMem)
	{
		TensorDescriptor input_desc = getExplicitGemmMatrixShape(config, xDesc, wDesc);
		TensorDescriptor filter_desc( { wDesc.firstDim(), wDesc.volumeWithoutFirstDim() }, wDesc.dtype());
		TensorDescriptor output_desc( { yDesc.volumeWithoutLastDim(), yDesc.lastDim() }, yDesc.dtype());

		MemoryDescriptor input_matrix(workspaceMem, input_desc.sizeInBytes(), 0);

		avStatus_t status;
		if (is_conv(1, wDesc))
		{
			status = cpu_gemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, alpha, input_desc, xMem, filter_desc, wMem, beta,
					output_desc, yMem);
		}
		else
		{
			status = SIMD_NAMESPACE::cpu_im2row(context, config, wDesc, xDesc, xMem, input_desc, input_matrix);
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			status = cpu_gemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, alpha, input_desc, input_matrix, filter_desc, wMem, beta,
					output_desc, yMem);
		}
		return status;
	}
	avStatus_t explicit_gemm_backward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *beta,
			const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, MemoryDescriptor &workspaceMem)
	{
		TensorDescriptor filter_desc(wDesc);
		std::swap(filter_desc[0], filter_desc[filter_desc.nbDims() - 1]);

		TensorDescriptor grad_next_desc = getExplicitGemmMatrixShape(config, dyDesc, filter_desc);
		TensorDescriptor grad_prev_desc( { dxDesc.volumeWithoutLastDim(), dxDesc.lastDim() }, dxDesc.dtype());

		avStatus_t status;
		if (is_conv(1, wDesc))
		{
			TensorDescriptor filter_matrix_desc( { wDesc.firstDim(), wDesc.volumeWithoutFirstDim() }, wDesc.dtype());
			status = cpu_gemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_N, alpha, grad_next_desc, dyMem, filter_matrix_desc, wMem,
					beta, grad_prev_desc, dxMem);
		}
		else
		{
			TensorDescriptor filter_matrix_desc( { filter_desc.firstDim(), filter_desc.volumeWithoutFirstDim() }, wDesc.dtype());
			MemoryDescriptor grad_next_matrix(workspaceMem, grad_next_desc.sizeInBytes(), 0);
			MemoryDescriptor filter_matrix(workspaceMem, filter_desc.sizeInBytes(), grad_next_matrix.size());

			std::array<int, 4> new_dim_order = { 3, 1, 2, 0 };
			status = SIMD_NAMESPACE::cpu_transpose(context, filter_desc, filter_matrix, wDesc, wMem, new_dim_order.data());
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			status = SIMD_NAMESPACE::cpu_im2row(context, config, filter_desc, dyDesc, dyMem, grad_next_desc, grad_next_matrix);
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			status = cpu_gemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, alpha, grad_next_desc, grad_next_matrix,
					filter_matrix_desc, filter_matrix, beta, grad_prev_desc, dxMem);
		}
		return status;
	}
	avStatus_t explicit_gemm_update(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
			const void *beta, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, MemoryDescriptor &workspaceMem)
	{
		TensorDescriptor input_matrix_desc = getExplicitGemmMatrixShape(config, xDesc, dwDesc);
		TensorDescriptor grad_next_desc( { dyDesc.volumeWithoutLastDim(), dyDesc.lastDim() }, dyDesc.dtype());
		TensorDescriptor weight_update_desc( { dwDesc.firstDim(), dwDesc.volumeWithoutFirstDim() }, dwDesc.dtype());

		avStatus_t status;
		if (is_conv(1, dwDesc))
		{
			status = cpu_gemm(context, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_N, alpha, grad_next_desc, dyMem, input_matrix_desc, xMem,
					beta, weight_update_desc, dwMem);
		}
		else
		{
			MemoryDescriptor input_matrix(workspaceMem, input_matrix_desc.sizeInBytes(), 0);

			status = SIMD_NAMESPACE::cpu_im2row(context, config, dwDesc, xDesc, xMem, input_matrix_desc, input_matrix);
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			status = cpu_gemm(context, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_N, alpha, grad_next_desc, dyMem, input_matrix_desc,
					input_matrix, beta, weight_update_desc, dwMem);
		}
		return status;
	}

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_convolutionBiasActivationForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc,
			const MemoryDescriptor &zMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation,
			MemoryDescriptor &workspaceMem)
	{
		if (config.groups > 1)
			return AVOCADO_STATUS_NOT_SUPPORTED;

		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	// convolution : y = alpha * conv(x, w) + beta * y
	avStatus_t cpu_convolutionForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
			const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &workspaceMem)
	{
		if (config.groups > 1)
			return AVOCADO_STATUS_NOT_SUPPORTED;
		avConvolutionAlgorithm_t algorithm = getConvolutionAlgorithm(config, xDesc, wDesc);
		switch (algorithm)
		{
			case AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM:
				return explicit_gemm_forward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem, workspaceMem);
			case AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED:
				return winograd_nonfused_forward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem, workspaceMem);
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	// convolution : dx = alpha * conv(dy, w) + beta * dx
	avStatus_t cpu_convolutionBackward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *beta,
			const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, MemoryDescriptor &workspaceMem)
	{
		if (config.isStrided() or config.groups > 1)
			return AVOCADO_STATUS_NOT_SUPPORTED; // TODO as for now stride is not supported

		ConvolutionDescriptor cfg(config);
		if (cfg.mode == AVOCADO_CONVOLUTION_MODE)
			cfg.mode = AVOCADO_CROSS_CORRELATION_MODE;
		else
			cfg.mode = AVOCADO_CONVOLUTION_MODE;
		for (int i = 0; i < wDesc.nbDims() - 2; i++)
			cfg.padding[i] = -(wDesc.dimension(1 + i) - 1) * cfg.dilation[i] - cfg.padding[i];

		avConvolutionAlgorithm_t algorithm = getConvolutionAlgorithm(cfg, dyDesc, wDesc);
		switch (algorithm)
		{
			case AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM:
				return explicit_gemm_backward(context, cfg, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
			case AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED:
			{

				break;
			}
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	avStatus_t cpu_convolutionUpdate(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
			const void *beta, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, MemoryDescriptor &workspaceMem)
	{
		if (config.groups > 1)
			return AVOCADO_STATUS_NOT_SUPPORTED;
		avConvolutionAlgorithm_t algorithm = getConvolutionAlgorithm(config, dyDesc, dwDesc);
		switch (algorithm)
		{
			case AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM:
				return explicit_gemm_update(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem, workspaceMem);
			case AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED:
				return winograd_nonfused_update(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem, workspaceMem);
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

