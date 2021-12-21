/*
 * dispatcher.cpp
 *
 *  Created on: Nov 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cpu_backend.h>
#include "kernel_definitions.hpp"
#include "context.hpp"

namespace avocado
{
	namespace backend
	{
		avStatus_t cpuScaleTensor(avContext_t context, avTensor_t dst, const avScalar_t src)
		{
//			switch (getSimdSupport(context))
//			{
//				case SimdLevel::AVX2:
//					return ns_avx2::cpuScaleTensor(context, dst, src);
//				case SimdLevel::F16C:
//					return ns_f16c::cpuScaleTensor(context, dst, src);
//				case SimdLevel::AVX:
//					return ns_avx::cpuScaleTensor(context, dst, src);
//				case SimdLevel::SSE41:
//					return ns_sse41::cpuScaleTensor(context, dst, src);
//				case SimdLevel::SSE2:
//					return ns_sse2::cpuScaleTensor(context, dst, src);
//				default:
//					return ns_none::cpuScaleTensor(context, dst, src);
//			}
		}
		avStatus_t cpuAddScalarToTensor(avContext_t context, avTensor_t dst, const avScalar_t src)
		{
		}
		avStatus_t cpuOpTensor(avContext_t context, avOpTensorOp_t operation, const avScalar_t alpha1, const avTensor_t input1,
				const avScalar_t alpha2, const avTensor_t input2, const avScalar_t beta, avTensor_t output)
		{
		}
		avStatus_t cpuReduceTensor(avContext_t context, avReduceTensorOp_t operation, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output)
		{
		}
		avStatus_t cpuAddTensors(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
				avActivationType_t activation)
		{
		}

		avStatus_t cpuActivationForward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha1, const avScalar_t alpha2,
				const avScalar_t beta, const avTensor_t input, avTensor_t output)
		{
		}

		avStatus_t cpuActivationBackward(avContext_t context, const avActivationType_t activation, const void *alpha, const void *beta,
				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output)
		{
		}
		avStatus_t cpuSoftmaxForward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output)
		{
		}
		avStatus_t cpuSoftmaxBackward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output)
		{
		}

		avStatus_t cpuAffineForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
				const avTensor_t weight, const avTensor_t bias, const avActivationType_t activation)
		{
		}
		avStatus_t cpuBatchNormInference(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t scale, const avTensor_t bias, const avTensor_t estimatedMean, const avTensor_t estimatedVariance,
				const avScalar_t epsilon, const avActivationType_t activation)
		{
		}
		avStatus_t cpuBatchNormForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
				const avTensor_t scale, const avTensor_t bias, avTensor_t savedMean, avTensor_t savedVariance, const avScalar_t epsilon,
				const avActivationType_t activation)
		{
		}
		avStatus_t cpuBatchNormBackward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, const avTensor_t output, avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t scale,
				const avTensor_t savedMean, const avTensor_t savedVariance, const avScalar_t epsilon)
		{
		}
		avStatus_t cpuBatchNormUpdate(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				const avTensor_t gradientNext, avTensor_t scaleUpdate, avTensor_t biasUpdate, const avTensor_t savedMean,
				const avTensor_t savedVariance, const avScalar_t epsilon)
		{
		}

		avStatus_t cpuDropoutForward(avContext_t context, const avDropout_t config, const avTensor_t input, avTensor_t output, avTensor_t states)
		{
		}
		avStatus_t cpuDropoutBackward(avContext_t context, const avDropout_t config, avTensor_t gradientPrev, const avTensor_t gradientNext,
				const avTensor_t states)
		{
		}

		avStatus_t cpuPoolingForward(avContext_t context, const avPooling_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output)
		{
		}
		avStatus_t cpuPoolingBackward(avContext_t context, const avPooling_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t gradientPrev, const avTensor_t gradientNext)
		{
		}

		avStatus_t cpuIm2Col(avContext_t context, const avConvolution_t config, const avTensor_t input, avTensor_t output);

		avStatus_t cpuWinogradWeightTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t weights,
				avTensor_t matrices)
		{
		}
		avStatus_t cpuWinogradInputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t input,
				avTensor_t matrices, const avTensor_t bias, const avActivationType_t activation)
		{
		}
		avStatus_t cpuWinogradOutputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t matrices, avTensor_t output)
		{
		}
		avStatus_t cpuWinogradGradientTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t gradient,
				avTensor_t matrices)
		{
		}
		avStatus_t cpuWinogradUpdateTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t matrices, avTensor_t update)
		{
		}

		avStatus_t cpuConvolutionBiasActivationForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha1,
				const avScalar_t beta, const avTensor_t input, avTensor_t output, const avTensor_t weights, const avTensor_t bias,
				const avActivationType_t activation, const avScalar_t alpha2, const avTensor_t add)
		{
		}
		avStatus_t cpuConvolutionForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output, const avTensor_t weights)
		{
		}
		avStatus_t cpuConvolutionBackward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t output, const avTensor_t weights,
				const avActivationType_t activation)
		{
		}
		avStatus_t cpuConvolutionUpdate(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, const avTensor_t gradientNext, avTensor_t weightUpdate, avTensor_t biasUpdate)
		{
		}

		avStatus_t cpuMetricFunction(avContext_t context, avMetricType_t metricType, avScalar_t result, const avTensor_t output,
				const avTensor_t target)
		{
		}

		avStatus_t cpuLossFunction(avContext_t context, avLossType_t lossType, avScalar_t result, const avTensor_t output, const avTensor_t target)
		{
		}
		avStatus_t cpuLossGradient(avContext_t context, avLossType_t lossType, const avScalar_t alpha, const avScalar_t beta, avTensor_t gradient,
				const avTensor_t output, const avTensor_t target, bool isFused)
		{
		}

		avStatus_t cpuOptimizerLearn(avContext_t context, const avOptimizer_t optimizer, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t weight, const avTensor_t update, avTensor_t workspace1, avTensor_t workspace2)
		{
		}

		avStatus_t cpuRegularizerL2(avContext_t context, avTensor_t gradient, const avTensor_t weight, const avScalar_t coefficient,
				const avScalar_t offset)
		{
		}
	} /* namespace backend */
} /* namespace avocado */

