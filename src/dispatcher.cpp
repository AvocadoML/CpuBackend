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
					return ns_avx2::cpu_changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::AVX:
					return ns_avx::cpu_changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_changeType(context, dst, dstType, src, srcType, elements);
				case SimdLevel::NONE:
					return ns_none::cpu_changeType(context, dst, dstType, src, srcType, elements);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_changeType(context, dst, dstType, src, srcType, elements);
#endif
		}

		avStatus_t cpuConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::AVX:
					return ns_avx::cpu_concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::NONE:
					return ns_none::cpu_concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_concatTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
#endif
		}

		avStatus_t cpuSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::AVX:
					return ns_avx::cpu_splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				case SimdLevel::NONE:
					return ns_none::cpu_splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_splitTensors(context, cDesc, cMem, aDesc, aMem, nbTensors);
#endif
		}

		avStatus_t cpuTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[])
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::AVX:
					return ns_avx::cpu_transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				case SimdLevel::NONE:
					return ns_none::cpu_transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_transpose(context, cDesc, cMem, aDesc, aMem, newDimOrder);
#endif
		}

		avStatus_t cpuScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::AVX:
					return ns_avx::cpu_scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_scaleTensor(context, cDesc, cMem, alpha);
				case SimdLevel::NONE:
					return ns_none::cpu_scaleTensor(context, cDesc, cMem, alpha);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_scaleTensor(context, cDesc, cMem, alpha);
#endif
		}

		avStatus_t cpuAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *scalar)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::AVX:
					return ns_avx::cpu_addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_addScalarToTensor(context, cDesc, cMem, scalar);
				case SimdLevel::NONE:
					return ns_none::cpu_addScalarToTensor(context, cDesc, cMem, scalar);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_addScalarToTensor(context, cDesc, cMem, scalar);
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
					return ns_avx2::cpu_binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_binaryOp(context, operation, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem);
#endif
		}

		avStatus_t cpuUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_unaryOp(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
#endif
		}

		avStatus_t cpuReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_reduceTensor(context, operation, alpha, aDesc, aMem, beta, cDesc, cMem);
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
					return ns_avx2::cpu_addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_addBias(context, alpha3, alpha1, aDesc, aMem, alpha2, bDesc, bMem, beta, cDesc, cMem, activation);
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
					return ns_avx2::cpu_activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_activationForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem);
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
					return ns_avx2::cpu_activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::NONE:
					return ns_none::cpu_activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_activationBackward(context, activation, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
#endif
		}

		avStatus_t cpuSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_softmaxForward(context, mode, alpha, xDesc, xMem, beta, yDesc, yMem);
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
					return ns_avx2::cpu_softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::NONE:
					return ns_none::cpu_softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_softmaxBackward(context, mode, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
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
					return ns_avx2::cpu_affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_affineForward(context, activation, wDesc, wMem, bDesc, bMem, alpha, xDesc, xMem, beta, yDesc, yMem);
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
					return ns_avx2::cpu_batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::cpu_batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::cpu_batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_batchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
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
					return ns_avx2::cpu_batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::cpu_batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::cpu_batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
							biasMem, meanMem, varianceMem, epsilon);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_batchNormForward(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem,
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
					return ns_avx2::cpu_batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::cpu_batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::cpu_batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
							scaleMeanVarDesc, scaleMem, meanMem, varianceMem, alpha2, beta2, scaleUpdateMem, biasUpdateMem, epsilon);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_batchNormBackward(context, activation, alpha, xDesc, xMem, yDesc, yMem, beta, dxDesc, dxMem, dyDesc, dyMem,
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
					return ns_avx2::cpu_dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::AVX:
					return ns_avx::cpu_dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				case SimdLevel::NONE:
					return ns_none::cpu_dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_dropoutForward(context, config, xDesc, xMem, yDesc, yMem, states);
#endif
		}

		avStatus_t cpuDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::AVX:
					return ns_avx::cpu_dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				case SimdLevel::NONE:
					return ns_none::cpu_dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_dropoutBackward(context, config, dyDesc, dyMem, dxDesc, dxMem, states);
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
					return ns_avx2::cpu_poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_poolingForward(context, config, alpha, xDesc, xMem, beta, yDesc, yMem);
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
					return ns_avx2::cpu_poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				case SimdLevel::NONE:
					return ns_none::cpu_poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_poolingBackward(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dxDesc, dxMem);
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
					return ns_avx2::cpu_im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				case SimdLevel::NONE:
					return ns_none::cpu_im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_im2row(context, config, filterDesc, srcDesc, srcMem, colDesc, colMem);
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
					return ns_avx2::cpu_convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				case SimdLevel::NONE:
					return ns_none::cpu_convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
							zMem, beta, yDesc, yMem, activation, workspace);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_convolutionBiasActivationForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc,
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
					return ns_avx2::cpu_convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_convolutionForward(context, config, alpha, xDesc, xMem, wDesc, wMem, beta, yDesc, yMem);
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
					return ns_avx2::cpu_convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				case SimdLevel::NONE:
					return ns_none::cpu_convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_convolutionBackward(context, config, alpha, dxDesc, dxMem, wDesc, wMem, beta, dyDesc, dyMem, workspaceMem);
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
					return ns_avx2::cpu_convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				case SimdLevel::NONE:
					return ns_none::cpu_convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_convolutionUpdate(context, config, alpha, xDesc, xMem, dyDesc, dyMem, beta, dwDesc, dwMem);
#endif
		}

		avStatus_t cpuMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::AVX:
					return ns_avx::cpu_metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::NONE:
					return ns_none::cpu_metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_metricFunction(context, metricType, outputDesc, outputMem, targetDesc, targetMem, result);
#endif
		}

		avStatus_t cpuLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::AVX:
					return ns_avx::cpu_lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				case SimdLevel::NONE:
					return ns_none::cpu_lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_lossFunction(context, lossType, outputDesc, outputMem, targetDesc, targetMem, result);
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
					return ns_avx2::cpu_lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::AVX:
					return ns_avx::cpu_lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				case SimdLevel::NONE:
					return ns_none::cpu_lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
							gradientMem, isFused);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_lossGradient(context, lossType, alpha, outputDesc, outputMem, targetDesc, targetMem, beta, gradientDesc,
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
					return ns_avx2::cpu_optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::AVX:
					return ns_avx::cpu_optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				case SimdLevel::NONE:
					return ns_none::cpu_optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_optimizerLearn(context, config, wDesc, wMem, dwDesc, dwMem, workspace);
#endif
		}

		avStatus_t cpuRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
				const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::AVX:
					return ns_avx::cpu_regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				case SimdLevel::NONE:
					return ns_none::cpu_regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_regularizerL2(context, gradientDesc, gradientMem, weightDesc, weightMem, coefficient, offset, loss);
#endif
		}

		avStatus_t cpuConvolution2dImplicitGemm(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta, yDesc, yMem, activation);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_convolution2dImplicitGemm(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem,
					beta, yDesc, yMem, activation);
#endif
		}

		avStatus_t cpuWinogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_winogradWeightTransform(context, config, wDesc, wMem, matricesDesc, matricesMem);
#endif
		}

		avStatus_t cpuWinogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t wDesc)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_winogradInputTransform(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
#endif
		}

		avStatus_t cpuWinogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2,
				const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation,
				const avTensorDescriptor_t wDesc)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc, zMem, beta, activation, wDesc);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_winogradOutputTransform(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2,
					zDesc, zMem, beta, activation, wDesc);
#endif
		}

		avStatus_t cpuWinogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t wDesc)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_winogradGradientTransform(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
#endif
		}

		avStatus_t cpuWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
				avMemoryDescriptor_t dwMem)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_winogradUpdateTransform(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
#endif
		}

		avStatus_t cpuWinogradFusedForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
							yDesc, yMem, activation);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_winogradFusedForward(context, config, alpha1, xDesc, xMem, wDesc, wMem, bDesc, bMem, alpha2, zDesc, zMem, beta,
					yDesc, yMem, activation);
#endif
		}

	} /* namespace backend */
} /* namespace avocado */

