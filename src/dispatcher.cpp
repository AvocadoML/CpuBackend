/*
 * dispatcher.cpp
 *
 *  Created on: Nov 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CpuBackend/cpu_backend.h>
#include "kernel_definitions.hpp"

namespace avocado
{
	namespace backend
	{
		avStatus_t cpuChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
				avDataType_t srcType, avSize_t elements)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::AVX:
					return ns_avx::changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::SSE41:
					return ns_sse41::changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::SSE2:
					return ns_sse2::changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::NONE:
					return ns_none::changeType(context, dst, dstType, src, srcType, elements);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::changeType(context, dst, dstType, src, srcType, elements);
#endif
		}

		avStatus_t cpuConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::AVX:
					return ns_avx::concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE41:
					return ns_sse41::concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE2:
					return ns_sse2::concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::NONE:
					return ns_none::concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
#endif
		}

		avStatus_t cpuSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::AVX:
					return ns_avx::splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE41:
					return ns_sse41::splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE2:
					return ns_sse2::splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::NONE:
					return ns_none::splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
#endif
		}

		avStatus_t cpuTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[])
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::AVX:
					return ns_avx::transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::SSE41:
					return ns_sse41::transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::SSE2:
					return ns_sse2::transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::NONE:
					return ns_none::transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
#endif
		}

		avStatus_t cpuScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::AVX:
					return ns_avx::scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::SSE41:
					return ns_sse41::scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::SSE2:
					return ns_sse2::scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::NONE:
					return ns_none::scaleTensor(context, cDesc, cMem, alpha);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::scaleTensor(context, cDesc, cMem, alpha);
#endif
		}

		avStatus_t cpuAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *scalar)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::AVX:
					return ns_avx::addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::SSE41:
					return ns_sse41::addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::SSE2:
					return ns_sse2::addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::NONE:
					return ns_none::addScalarToTensor(context, cDesc, cMem, scalar);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::addScalarToTensor(context, cDesc, cMem, scalar);
#endif
		}

		avStatus_t cpuBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::AVX:
					return ns_avx::binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::SSE41:
					return ns_sse41::binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::SSE2:
					return ns_sse2::binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::NONE:
					return ns_none::binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
#endif
		}

		avStatus_t cpuUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::AVX:
					return ns_avx::unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE41:
					return ns_sse41::unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE2:
					return ns_sse2::unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::NONE:
					return ns_none::unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
#endif
		}

		avStatus_t cpuReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::AVX:
					return ns_avx::reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE41:
					return ns_sse41::reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE2:
					return ns_sse2::reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::NONE:
					return ns_none::reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
#endif
		}

		avStatus_t cpuAddBias(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::AVX:
					return ns_avx::addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::NONE:
					return ns_none::addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
#endif
		}

		avStatus_t cpuActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
#endif
		}

		avStatus_t cpuActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::AVX:
					return ns_avx::activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::NONE:
					return ns_none::activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
#endif
		}

		avStatus_t cpuSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
#endif
		}

		avStatus_t cpuSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::AVX:
					return ns_avx::softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::NONE:
					return ns_none::softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
#endif
		}

		avStatus_t cpuAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
#endif
		}

		avStatus_t cpuBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
					biasMem, meanMem, varianceMem, epsilon);
#endif
		}

		avStatus_t cpuBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
					biasMem, meanMem, varianceMem, epsilon);
#endif
		}

		avStatus_t cpuBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
				const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
				avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, const void *alpha2, const void *beta2,
				avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, double epsilon)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
					scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
#endif
		}

		avStatus_t cpuDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::AVX:
					return ns_avx::dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::SSE41:
					return ns_sse41::dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::SSE2:
					return ns_sse2::dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::NONE:
					return ns_none::dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
#endif
		}

		avStatus_t cpuDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::AVX:
					return ns_avx::dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::SSE41:
					return ns_sse41::dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::SSE2:
					return ns_sse2::dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::NONE:
					return ns_none::dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
#endif
		}

		avStatus_t cpuPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
#endif
		}

		avStatus_t cpuPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::AVX:
					return ns_avx::poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::NONE:
					return ns_none::poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
#endif
		}

		avStatus_t cpuIm2Row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc,
				avMemoryDescriptor_t colMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::AVX:
					return ns_avx::im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::SSE41:
					return ns_sse41::im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::SSE2:
					return ns_sse2::im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::NONE:
					return ns_none::im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
#endif
		}

		avStatus_t cpuGetConvolutionWorkspaceSize(const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, avSize_t *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cpuConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation, avMemoryDescriptor_t workspace)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::AVX:
					return ns_avx::convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::SSE41:
					return ns_sse41::convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::SSE2:
					return ns_sse2::convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::NONE:
					return ns_none::convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
					zMem, beta, yDesc, yMem, activation, workspace);
#endif
		}

		avStatus_t cpuConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
#endif
		}

		avStatus_t cpuConvolutionBackward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, avMemoryDescriptor_t workspaceMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::AVX:
					return ns_avx::convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::SSE41:
					return ns_sse41::convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::SSE2:
					return ns_sse2::convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::NONE:
					return ns_none::convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
#endif
		}

		avStatus_t cpuConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::AVX:
					return ns_avx::convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE41:
					return ns_sse41::convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE2:
					return ns_sse2::convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::NONE:
					return ns_none::convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
#endif
		}

		avStatus_t cpuMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::AVX:
					return ns_avx::metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE41:
					return ns_sse41::metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE2:
					return ns_sse2::metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::NONE:
					return ns_none::metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
#endif
		}

		avStatus_t cpuLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::AVX:
					return ns_avx::lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE41:
					return ns_sse41::lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE2:
					return ns_sse2::lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::NONE:
					return ns_none::lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
#endif
		}

		avStatus_t cpuLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::AVX:
					return ns_avx::lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::SSE41:
					return ns_sse41::lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::SSE2:
					return ns_sse2::lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::NONE:
					return ns_none::lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
					gradientMem, isFused);
#endif
		}

		avStatus_t cpuGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, avSize_t *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cpuOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const avTensorDescriptor_t wDesc,
				avMemoryDescriptor_t wMem, const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem, avMemoryDescriptor_t workspace)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::AVX:
					return ns_avx::optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::SSE41:
					return ns_sse41::optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::SSE2:
					return ns_sse2::optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::NONE:
					return ns_none::optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
#endif
		}

		avStatus_t cpuRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
				const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::AVX:
					return ns_avx::regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::SSE41:
					return ns_sse41::regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::SSE2:
					return ns_sse2::regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::NONE:
					return ns_none::regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
#endif
		}

		avStatus_t convolution2dImplicitGemm(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::AVX:
					return ns_avx::convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::NONE:
					return ns_none::convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem,
					beta, yDesc, yMem, activation);
#endif
		}

		avStatus_t winogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::AVX:
					return ns_avx::winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::SSE41:
					return ns_sse41::winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::SSE2:
					return ns_sse2::winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::NONE:
					return ns_none::winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
#endif
		}

		avStatus_t winogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t wDesc)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::AVX:
					return ns_avx::winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE41:
					return ns_sse41::winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE2:
					return ns_sse2::winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::NONE:
					return ns_none::winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
#endif
		}

		avStatus_t winogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2,
				const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation,
				const avTensorDescriptor_t wDesc)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::AVX:
					return ns_avx::winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::SSE41:
					return ns_sse41::winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::SSE2:
					return ns_sse2::winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::NONE:
					return ns_none::winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2,
					zDesc, zMem, beta, activation, wDesc);
#endif
		}

		avStatus_t winogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t wDesc)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::AVX:
					return ns_avx::winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE41:
					return ns_sse41::winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE2:
					return ns_sse2::winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::NONE:
					return ns_none::winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
#endif
		}

		avStatus_t winogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
				avMemoryDescriptor_t dwMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::AVX:
					return ns_avx::winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE41:
					return ns_sse41::winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE2:
					return ns_sse2::winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::NONE:
					return ns_none::winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
#endif
		}

		avStatus_t winogradFusedForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::AVX:
					return ns_avx::winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::NONE:
					return ns_none::winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
					yDesc, yMem, activation);
#endif
		}

	} /* namespace backend */
} /* namespace avocado */

