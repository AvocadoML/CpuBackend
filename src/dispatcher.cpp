/*
 * dispatcher.cpp
 *
 *  Created on: Nov 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cpu_backend.h>
#include "kernel_definitions.hpp"

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;
		/*
		 *
		 * Tensor operations.
		 *
		 */
		avStatus_t cpuChangeTypeHost(avContextDescriptor_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType,
				av_int64 elements)
		{
			const ContextDescriptor &cpu_context = getContext(context);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_changeTypeHost(cpu_context, dst, dstType, src, srcType, elements);
				case SimdLevel::AVX:
					return ns_avx::cpu_changeTypeHost(cpu_context, dst, dstType, src, srcType, elements);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_changeTypeHost(cpu_context, dst, dstType, src, srcType, elements);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_changeTypeHost(cpu_context, dst, dstType, src, srcType, elements);
				case SimdLevel::NONE:
					return ns_none::cpu_changeTypeHost(cpu_context, dst, dstType, src, srcType, elements);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
		#else
			return SIMD_NAMESPACE::cpu_changeTypeHost(cpu_context, dst, dstType, src, srcType, elements);
#endif
		}
		avStatus_t cpuChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
				avDataType_t srcType, av_int64 elements)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const MemoryDescriptor &cpu_src = getMemory(src);
			MemoryDescriptor &cpu_dst = getMemory(dst);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_changeType(cpu_context, cpu_dst, dstType, cpu_src, srcType, elements);
				case SimdLevel::AVX:
					return ns_avx::cpu_changeType(cpu_context, cpu_dst, dstType, cpu_src, srcType, elements);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_changeType(cpu_context, cpu_dst, dstType, cpu_src, srcType, elements);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_changeType(cpu_context, cpu_dst, dstType, cpu_src, srcType, elements);
				case SimdLevel::NONE:
					return ns_none::cpu_changeType(cpu_context, cpu_dst, dstType, cpu_src, srcType, elements);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_changeType(cpu_context, cpu_dst, dstType, cpu_src, srcType, elements);
#endif
		}
		avStatus_t cpuConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
			std::vector<const TensorDescriptor*> cpu_aDesc(nbTensors);
			std::vector<const MemoryDescriptor*> cpu_aMem(nbTensors);
			for (int i = 0; i < nbTensors; i++)
			{
				cpu_aDesc[i] = &(getTensor(aDesc[i]));
				cpu_aMem[i] = &(getMemory(aMem[i]));
			}

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_concatTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_concatTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_concatTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_concatTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::NONE:
					return ns_none::cpu_concatTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_concatTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
#endif
		}
		avStatus_t cpuSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
			std::vector<const TensorDescriptor*> cpu_cDesc(nbTensors);
			std::vector<MemoryDescriptor*> cpu_cMem(nbTensors);
			for (int i = 0; i < nbTensors; i++)
			{
				cpu_cDesc[i] = &(getTensor(cDesc[i]));
				cpu_cMem[i] = &(getMemory(cMem[i]));
			}

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_splitTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_splitTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_splitTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_splitTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				case SimdLevel::NONE:
					return ns_none::cpu_splitTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
				default:
					break;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
#else
			return SIMD_NAMESPACE::cpu_splitTensors(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem);
#endif
		}
		avStatus_t cpuTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[])
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_transpose(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem, newDimOrder);
				case SimdLevel::AVX:
					return ns_avx::cpu_transpose(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem, newDimOrder);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_transpose(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem, newDimOrder);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_transpose(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem, newDimOrder);
				case SimdLevel::NONE:
					return ns_none::cpu_transpose(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem, newDimOrder);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_transpose(cpu_context, cpu_cDesc, cpu_cMem, cpu_aDesc, cpu_aMem, newDimOrder);
#endif
		}
		avStatus_t cpuScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *alpha,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_scaleTensor(cpu_context, cpu_aDesc, cpu_aMem, alpha, cpu_cDesc, cpu_cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_scaleTensor(cpu_context, cpu_aDesc, cpu_aMem, alpha, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_scaleTensor(cpu_context, cpu_aDesc, cpu_aMem, alpha, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_scaleTensor(cpu_context, cpu_aDesc, cpu_aMem, alpha, cpu_cDesc, cpu_cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_scaleTensor(cpu_context, cpu_aDesc, cpu_aMem, alpha, cpu_cDesc, cpu_cMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_scaleTensor(cpu_context, cpu_aDesc, cpu_aMem, alpha, cpu_cDesc, cpu_cMem);
#endif
		}
		avStatus_t cpuAddTensors(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_addTensors(cpu_context, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_addTensors(cpu_context, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_addTensors(cpu_context, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_addTensors(cpu_context, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_addTensors(cpu_context, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_addTensors(cpu_context, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
#endif
		}
		avStatus_t cpuAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem,
				const void *scalar, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_addScalarToTensor(cpu_context, cpu_aDesc, cpu_aMem, scalar, cpu_cDesc, cpu_cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_addScalarToTensor(cpu_context, cpu_aDesc, cpu_aMem, scalar, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_addScalarToTensor(cpu_context, cpu_aDesc, cpu_aMem, scalar, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_addScalarToTensor(cpu_context, cpu_aDesc, cpu_aMem, scalar, cpu_cDesc, cpu_cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_addScalarToTensor(cpu_context, cpu_aDesc, cpu_aMem, scalar, cpu_cDesc, cpu_cMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_addScalarToTensor(cpu_context, cpu_aDesc, cpu_aMem, scalar, cpu_cDesc, cpu_cMem);
#endif
		}
		avStatus_t cpuAddBias(avContextDescriptor_t context, const void *alpha1, const void *alpha2, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const void *beta1, const void *beta2, const void *beta3, const avMemoryDescriptor_t zMem,
				avActivationType_t activation)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
			const MemoryDescriptor &cpu_zMem = getMemory(zMem);

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_addBias(cpu_context, alpha1, alpha2, cpu_xDesc, cpu_xMem, cpu_bDesc, cpu_bMem, cpu_yDesc, cpu_yMem, beta1,
							beta2, beta3, cpu_zMem, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_addBias(cpu_context, alpha1, alpha2, cpu_xDesc, cpu_xMem, cpu_bDesc, cpu_bMem, cpu_yDesc, cpu_yMem, beta1,
							beta2, beta3, cpu_zMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_addBias(cpu_context, alpha1, alpha2, cpu_xDesc, cpu_xMem, cpu_bDesc, cpu_bMem, cpu_yDesc, cpu_yMem, beta1,
							beta2, beta3, cpu_zMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_addBias(cpu_context, alpha1, alpha2, cpu_xDesc, cpu_xMem, cpu_bDesc, cpu_bMem, cpu_yDesc, cpu_yMem, beta1,
							beta2, beta3, cpu_zMem, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_addBias(cpu_context, alpha1, alpha2, cpu_xDesc, cpu_xMem, cpu_bDesc, cpu_bMem, cpu_yDesc, cpu_yMem, beta1,
							beta2, beta3, cpu_zMem, activation);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_addBias(cpu_context, alpha1, alpha2, cpu_xDesc, cpu_xMem, cpu_bDesc, cpu_bMem, cpu_yDesc, cpu_yMem, beta1,
					beta2, beta3, cpu_zMem, activation);
#endif
		}
		avStatus_t cpuBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_binaryOp(cpu_context, operation, alpha1, cpu_aDesc, cpu_aMem, alpha2, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_binaryOp(cpu_context, operation, alpha1, cpu_aDesc, cpu_aMem, alpha2, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_binaryOp(cpu_context, operation, alpha1, cpu_aDesc, cpu_aMem, alpha2, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_binaryOp(cpu_context, operation, alpha1, cpu_aDesc, cpu_aMem, alpha2, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_binaryOp(cpu_context, operation, alpha1, cpu_aDesc, cpu_aMem, alpha2, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_binaryOp(cpu_context, operation, alpha1, cpu_aDesc, cpu_aMem, alpha2, cpu_bDesc, cpu_bMem, beta, cpu_cDesc,
					cpu_cMem);
#endif
		}
		avStatus_t cpuUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_unaryOp(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_unaryOp(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_unaryOp(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_unaryOp(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_unaryOp(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_unaryOp(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
#endif
		}
		avStatus_t cpuReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_reduceTensor(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_reduceTensor(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_reduceTensor(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_reduceTensor(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				case SimdLevel::NONE:
					return ns_none::cpu_reduceTensor(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_reduceTensor(cpu_context, operation, alpha, cpu_aDesc, cpu_aMem, beta, cpu_cDesc, cpu_cMem);
#endif
		}
		avStatus_t cpuGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);

			return cpu_gemm(cpu_context, aOp, bOp, alpha, cpu_aDesc, cpu_aMem, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
		}
		avStatus_t cpuGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_aDesc = getTensor(aDesc);
			const MemoryDescriptor &cpu_aMem = getMemory(aMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_cDesc = getTensor(cDesc);
			MemoryDescriptor &cpu_cMem = getMemory(cMem);

			return cpu_gemmBatched(cpu_context, aOp, bOp, alpha, cpu_aDesc, cpu_aMem, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
		}

		/*
		 *
		 * Activation functions.
		 *
		 */

		avStatus_t cpuActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_activationForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_activationForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_activationForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_activationForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_activationForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_activationForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
#endif
		}
		avStatus_t cpuActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			const MemoryDescriptor &cpu_yMem = getMemory(yMem);
			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
			const TensorDescriptor &cpu_dxDesc = getTensor(dxDesc);
			MemoryDescriptor &cpu_dxMem = getMemory(dxMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_activationBackward(cpu_context, activation, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_activationBackward(cpu_context, activation, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_activationBackward(cpu_context, activation, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_activationBackward(cpu_context, activation, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::NONE:
					return ns_none::cpu_activationBackward(cpu_context, activation, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_activationBackward(cpu_context, activation, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta,
					cpu_dxDesc, cpu_dxMem);
#endif
		}
		avStatus_t cpuSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_softmaxForward(cpu_context, mode, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_softmaxForward(cpu_context, mode, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_softmaxForward(cpu_context, mode, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_softmaxForward(cpu_context, mode, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_softmaxForward(cpu_context, mode, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_softmaxForward(cpu_context, mode, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
#endif
		}
		avStatus_t cpuSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			const MemoryDescriptor &cpu_yMem = getMemory(yMem);
			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
			const TensorDescriptor &cpu_dxDesc = getTensor(dxDesc);
			MemoryDescriptor &cpu_dxMem = getMemory(dxMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_softmaxBackward(cpu_context, mode, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_softmaxBackward(cpu_context, mode, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_softmaxBackward(cpu_context, mode, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_softmaxBackward(cpu_context, mode, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				case SimdLevel::NONE:
					return ns_none::cpu_softmaxBackward(cpu_context, mode, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc, cpu_dxMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_softmaxBackward(cpu_context, mode, alpha, cpu_yDesc, cpu_yMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
					cpu_dxMem);
#endif
		}

		/*
		 *
		 * Batch normalization and affine transform.
		 *
		 */

		avStatus_t cpuAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_affineForward(cpu_context, activation, cpu_wDesc, cpu_wMem, cpu_bDesc, cpu_bMem, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_affineForward(cpu_context, activation, cpu_wDesc, cpu_wMem, cpu_bDesc, cpu_bMem, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_affineForward(cpu_context, activation, cpu_wDesc, cpu_wMem, cpu_bDesc, cpu_bMem, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_affineForward(cpu_context, activation, cpu_wDesc, cpu_wMem, cpu_bDesc, cpu_bMem, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_affineForward(cpu_context, activation, cpu_wDesc, cpu_wMem, cpu_bDesc, cpu_bMem, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_affineForward(cpu_context, activation, cpu_wDesc, cpu_wMem, cpu_bDesc, cpu_bMem, alpha, cpu_xDesc, cpu_xMem,
					beta, cpu_yDesc, cpu_yMem);
#endif
		}
		avStatus_t cpuBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_scaleBiasMeanVarDesc = getTensor(scaleBiasMeanVarDesc);
			const MemoryDescriptor &cpu_scaleMem = getMemory(scaleMem);
			const MemoryDescriptor &cpu_biasMem = getMemory(biasMem);
			const MemoryDescriptor &cpu_meanMem = getMemory(meanMem);
			const MemoryDescriptor &cpu_varianceMem = getMemory(varianceMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_batchNormInference(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc, cpu_scaleMem,
							cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::cpu_batchNormInference(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc, cpu_scaleMem,
							cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_batchNormInference(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc, cpu_scaleMem,
							cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_batchNormInference(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc, cpu_scaleMem,
							cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::cpu_batchNormInference(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc, cpu_scaleMem,
							cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_batchNormInference(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem,
					cpu_scaleBiasMeanVarDesc, cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
#endif
		}
		avStatus_t cpuBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_scaleBiasMeanVarDesc = getTensor(scaleBiasMeanVarDesc);
			const MemoryDescriptor &cpu_scaleMem = getMemory(scaleMem);
			const MemoryDescriptor &cpu_biasMem = getMemory(biasMem);
			MemoryDescriptor &cpu_meanMem = getMemory(meanMem);
			MemoryDescriptor &cpu_varianceMem = getMemory(varianceMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_batchNormForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc,
							cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::cpu_batchNormForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc,
							cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_batchNormForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc,
							cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_batchNormForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc,
							cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::cpu_batchNormForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem, cpu_scaleBiasMeanVarDesc,
							cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_batchNormForward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem,
					cpu_scaleBiasMeanVarDesc, cpu_scaleMem, cpu_biasMem, cpu_meanMem, cpu_varianceMem, epsilon);
#endif
		}
		avStatus_t cpuBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
				const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
				avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, const void *alpha2, const void *beta2,
				avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, double epsilon)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			const MemoryDescriptor &cpu_yMem = getMemory(yMem);
			const TensorDescriptor &cpu_dxDesc = getTensor(dxDesc);
			MemoryDescriptor &cpu_dxMem = getMemory(dxMem);
			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
			MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
			const TensorDescriptor &cpu_scaleMeanVarDesc = getTensor(scaleMeanVarDesc);
			const MemoryDescriptor &cpu_scaleMem = getMemory(scaleMem);
			const MemoryDescriptor &cpu_meanMem = getMemory(meanMem);
			const MemoryDescriptor &cpu_varianceMem = getMemory(varianceMem);
			MemoryDescriptor &cpu_scaleUpdateMem = getMemory(scaleUpdateMem);
			MemoryDescriptor &cpu_biasUpdateMem = getMemory(biasUpdateMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_batchNormBackward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, beta, cpu_dxDesc,
							cpu_dxMem, cpu_dyDesc, cpu_dyMem, cpu_scaleMeanVarDesc, cpu_scaleMem, cpu_meanMem, cpu_varianceMem, alpha2, beta2,
							cpu_scaleUpdateMem, cpu_biasUpdateMem, epsilon);
				case SimdLevel::AVX:
					return ns_avx::cpu_batchNormBackward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, beta, cpu_dxDesc,
							cpu_dxMem, cpu_dyDesc, cpu_dyMem, cpu_scaleMeanVarDesc, cpu_scaleMem, cpu_meanMem, cpu_varianceMem, alpha2, beta2,
							cpu_scaleUpdateMem, cpu_biasUpdateMem, epsilon);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_batchNormBackward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, beta, cpu_dxDesc,
							cpu_dxMem, cpu_dyDesc, cpu_dyMem, cpu_scaleMeanVarDesc, cpu_scaleMem, cpu_meanMem, cpu_varianceMem, alpha2, beta2,
							cpu_scaleUpdateMem, cpu_biasUpdateMem, epsilon);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_batchNormBackward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, beta, cpu_dxDesc,
							cpu_dxMem, cpu_dyDesc, cpu_dyMem, cpu_scaleMeanVarDesc, cpu_scaleMem, cpu_meanMem, cpu_varianceMem, alpha2, beta2,
							cpu_scaleUpdateMem, cpu_biasUpdateMem, epsilon);
				case SimdLevel::NONE:
					return ns_none::cpu_batchNormBackward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, beta, cpu_dxDesc,
							cpu_dxMem, cpu_dyDesc, cpu_dyMem, cpu_scaleMeanVarDesc, cpu_scaleMem, cpu_meanMem, cpu_varianceMem, alpha2, beta2,
							cpu_scaleUpdateMem, cpu_biasUpdateMem, epsilon);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_batchNormBackward(cpu_context, activation, alpha, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, beta, cpu_dxDesc,
					cpu_dxMem, cpu_dyDesc, cpu_dyMem, cpu_scaleMeanVarDesc, cpu_scaleMem, cpu_meanMem, cpu_varianceMem, alpha2, beta2,
					cpu_scaleUpdateMem, cpu_biasUpdateMem, epsilon);
#endif
		}

		/*
		 *
		 * Dropout.
		 *
		 */

		avStatus_t cpuDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const DropoutDescriptor &cpu_config = getDropout(config);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
			MemoryDescriptor &cpu_states = getMemory(states);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_dropoutForward(cpu_context, cpu_config, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, cpu_states);
				case SimdLevel::AVX:
					return ns_avx::cpu_dropoutForward(cpu_context, cpu_config, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, cpu_states);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_dropoutForward(cpu_context, cpu_config, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, cpu_states);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_dropoutForward(cpu_context, cpu_config, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, cpu_states);
				case SimdLevel::NONE:
					return ns_none::cpu_dropoutForward(cpu_context, cpu_config, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, cpu_states);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_dropoutForward(cpu_context, cpu_config, cpu_xDesc, cpu_xMem, cpu_yDesc, cpu_yMem, cpu_states);
#endif
		}
		avStatus_t cpuDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avMemoryDescriptor_t states)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const DropoutDescriptor &cpu_config = getDropout(config);
			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
			const TensorDescriptor &cpu_dxDesc = getTensor(dxDesc);
			MemoryDescriptor &cpu_dxMem = getMemory(dxMem);
			const MemoryDescriptor &cpu_states = getMemory(states);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_dropoutBackward(cpu_context, cpu_config, cpu_dyDesc, cpu_dyMem, cpu_dxDesc, cpu_dxMem, cpu_states);
				case SimdLevel::AVX:
					return ns_avx::cpu_dropoutBackward(cpu_context, cpu_config, cpu_dyDesc, cpu_dyMem, cpu_dxDesc, cpu_dxMem, cpu_states);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_dropoutBackward(cpu_context, cpu_config, cpu_dyDesc, cpu_dyMem, cpu_dxDesc, cpu_dxMem, cpu_states);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_dropoutBackward(cpu_context, cpu_config, cpu_dyDesc, cpu_dyMem, cpu_dxDesc, cpu_dxMem, cpu_states);
				case SimdLevel::NONE:
					return ns_none::cpu_dropoutBackward(cpu_context, cpu_config, cpu_dyDesc, cpu_dyMem, cpu_dxDesc, cpu_dxMem, cpu_states);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_dropoutBackward(cpu_context, cpu_config, cpu_dyDesc, cpu_dyMem, cpu_dxDesc, cpu_dxMem, cpu_states);
#endif
		}

		/*
		 *
		 * Pooling.
		 *
		 */

		avStatus_t cpuPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const PoolingDescriptor &cpu_config = getPooling(config);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_poolingForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_poolingForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_poolingForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_poolingForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				case SimdLevel::NONE:
					return ns_none::cpu_poolingForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_poolingForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, beta, cpu_yDesc, cpu_yMem);
#endif
		}
		avStatus_t cpuPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const PoolingDescriptor &cpu_config = getPooling(config);
			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_dxDesc = getTensor(dxDesc);
			MemoryDescriptor &cpu_dxMem = getMemory(dxMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_poolingBackward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
							cpu_dxMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_poolingBackward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
							cpu_dxMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_poolingBackward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
							cpu_dxMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_poolingBackward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
							cpu_dxMem);
				case SimdLevel::NONE:
					return ns_none::cpu_poolingBackward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
							cpu_dxMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_poolingBackward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dxDesc,
					cpu_dxMem);
#endif
		}

		/*
		 *
		 * Convolutions.
		 *
		 */

		avStatus_t cpuIm2Row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t rowDesc,
				avMemoryDescriptor_t rowMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_filterDesc = getTensor(filterDesc);
			const TensorDescriptor &cpu_srcDesc = getTensor(srcDesc);
			const MemoryDescriptor &cpu_srcMem = getMemory(srcMem);
			const TensorDescriptor &cpu_rowDesc = getTensor(rowDesc);
			MemoryDescriptor &cpu_rowMem = getMemory(rowMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_im2row(cpu_context, cpu_config, cpu_filterDesc, cpu_srcDesc, cpu_srcMem, cpu_rowDesc, cpu_rowMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_im2row(cpu_context, cpu_config, cpu_filterDesc, cpu_srcDesc, cpu_srcMem, cpu_rowDesc, cpu_rowMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_im2row(cpu_context, cpu_config, cpu_filterDesc, cpu_srcDesc, cpu_srcMem, cpu_rowDesc, cpu_rowMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_im2row(cpu_context, cpu_config, cpu_filterDesc, cpu_srcDesc, cpu_srcMem, cpu_rowDesc, cpu_rowMem);
				case SimdLevel::NONE:
					return ns_none::cpu_im2row(cpu_context, cpu_config, cpu_filterDesc, cpu_srcDesc, cpu_srcMem, cpu_rowDesc, cpu_rowMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_im2row(cpu_context, cpu_config, cpu_filterDesc, cpu_srcDesc, cpu_srcMem, cpu_rowDesc, cpu_rowMem);
#endif
		}

		avStatus_t cpuConvolutionImplicitGemmForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				avActivationType_t activation)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_zDesc = getTensor(zDesc);
			const MemoryDescriptor &cpu_zMem = getMemory(zMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_convolutionImplicitGemmForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolutionImplicitGemmForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolutionImplicitGemmForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolutionImplicitGemmForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_convolutionImplicitGemmForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_convolutionImplicitGemmForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
					cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
#endif
		}
		avStatus_t cpuConvolutionWinogradFusedForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				avActivationType_t activation)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_zDesc = getTensor(zDesc);
			const MemoryDescriptor &cpu_zMem = getMemory(zMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_convolutionWinogradFusedForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_convolutionWinogradFusedForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_convolutionWinogradFusedForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_convolutionWinogradFusedForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_convolutionWinogradFusedForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_convolutionWinogradFusedForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
					cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation);
#endif
		}
		avStatus_t cpuWinogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc,
				avMemoryDescriptor_t matricesMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
			const TensorDescriptor &cpu_matricesDesc = getTensor(matricesDesc);
			MemoryDescriptor &cpu_matricesMem = getMemory(matricesMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradWeightTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_wMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradWeightTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_wMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradWeightTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_wMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradWeightTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_wMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradWeightTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_wMem, cpu_matricesDesc, cpu_matricesMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_winogradWeightTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_wMem, cpu_matricesDesc,
					cpu_matricesMem);
#endif
		}
		avStatus_t cpuWinogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem,
				const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
			const TensorDescriptor &cpu_matricesDesc = getTensor(matricesDesc);
			MemoryDescriptor &cpu_matricesMem = getMemory(matricesMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradInputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_xDesc, cpu_xMem, cpu_matricesDesc,
							cpu_matricesMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradInputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_xDesc, cpu_xMem, cpu_matricesDesc,
							cpu_matricesMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradInputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_xDesc, cpu_xMem, cpu_matricesDesc,
							cpu_matricesMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradInputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_xDesc, cpu_xMem, cpu_matricesDesc,
							cpu_matricesMem);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradInputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_xDesc, cpu_xMem, cpu_matricesDesc,
							cpu_matricesMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_winogradInputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_xDesc, cpu_xMem,
					cpu_matricesDesc, cpu_matricesMem);
#endif
		}
		avStatus_t cpuWinogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const void *alpha1, const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *alpha2, const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta,
				avActivationType_t activation)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_matricesDesc = getTensor(matricesDesc);
			const MemoryDescriptor &cpu_matricesMem = getMemory(matricesMem);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
			const TensorDescriptor &cpu_zDesc = getTensor(zDesc);
			const MemoryDescriptor &cpu_zMem = getMemory(zMem);
			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
			MemoryDescriptor &cpu_yMem = getMemory(yMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradOutputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, alpha1, cpu_matricesDesc, cpu_matricesMem,
							cpu_yDesc, cpu_yMem, cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, activation);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradOutputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, alpha1, cpu_matricesDesc, cpu_matricesMem,
							cpu_yDesc, cpu_yMem, cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, activation);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradOutputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, alpha1, cpu_matricesDesc, cpu_matricesMem,
							cpu_yDesc, cpu_yMem, cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, activation);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradOutputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, alpha1, cpu_matricesDesc, cpu_matricesMem,
							cpu_yDesc, cpu_yMem, cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, activation);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradOutputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, alpha1, cpu_matricesDesc, cpu_matricesMem,
							cpu_yDesc, cpu_yMem, cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, activation);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_winogradOutputTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, alpha1, cpu_matricesDesc,
					cpu_matricesMem, cpu_yDesc, cpu_yMem, cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, activation);

#endif
		}
		avStatus_t cpuWinogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem,
				const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
			const TensorDescriptor &cpu_matricesDesc = getTensor(matricesDesc);
			MemoryDescriptor &cpu_matricesMem = getMemory(matricesMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradGradientTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_dyDesc, cpu_dyMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradGradientTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_dyDesc, cpu_dyMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradGradientTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_dyDesc, cpu_dyMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradGradientTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_dyDesc, cpu_dyMem, cpu_matricesDesc, cpu_matricesMem);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradGradientTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_dyDesc, cpu_dyMem, cpu_matricesDesc, cpu_matricesMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_winogradGradientTransform(cpu_context, cpu_config, transformSize, cpu_wDesc, cpu_dyDesc, cpu_dyMem,
					cpu_matricesDesc, cpu_matricesMem);
#endif
		}
		avStatus_t cpuWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const void *alpha, const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta,
				const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const ConvolutionDescriptor &cpu_config = getConvolution(config);
			const TensorDescriptor &cpu_matricesDesc = getTensor(matricesDesc);
			const MemoryDescriptor &cpu_matricesMem = getMemory(matricesMem);
			const TensorDescriptor &cpu_dwDesc = getTensor(dwDesc);
			MemoryDescriptor &cpu_dwMem = getMemory(dwMem);

#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_winogradUpdateTransform(cpu_context, cpu_config, transformSize, alpha, cpu_matricesDesc, cpu_matricesMem, beta, cpu_dwDesc, cpu_dwMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_winogradUpdateTransform(cpu_context, cpu_config, transformSize, alpha, cpu_matricesDesc, cpu_matricesMem, beta, cpu_dwDesc, cpu_dwMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_winogradUpdateTransform(cpu_context, cpu_config, transformSize, alpha, cpu_matricesDesc, cpu_matricesMem, beta, cpu_dwDesc, cpu_dwMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_winogradUpdateTransform(cpu_context, cpu_config, transformSize, alpha, cpu_matricesDesc, cpu_matricesMem, beta, cpu_dwDesc, cpu_dwMem);
				case SimdLevel::NONE:
					return ns_none::cpu_winogradUpdateTransform(cpu_context, cpu_config, transformSize, alpha, cpu_matricesDesc, cpu_matricesMem, beta, cpu_dwDesc, cpu_dwMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_winogradUpdateTransform(cpu_context, cpu_config, transformSize, alpha, cpu_matricesDesc, cpu_matricesMem, beta,
					cpu_dwDesc, cpu_dwMem);
#endif
		}

//		avStatus_t cpuGetConvolutionWorkspaceSize(const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
//				const avTensorDescriptor_t wDesc, bool inferenceOnly, av_int64 *result)
//		{
//			const ConvolutionDescriptor &cpu_config = const_getConvolution(config);
//			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
//			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
//
//			return cpu_getConvolutionWorkspaceSize(cpu_config, cpu_xDesc, cpu_wDesc, inferenceOnly, result);
//		}
//		avStatus_t cpuConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
//				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
//				const avActivationType_t activation, avMemoryDescriptor_t workspaceMem)
//		{
//			const ContextDescriptor &cpu_context = getContext(context);
//			const ConvolutionDescriptor &cpu_config = getConvolution(config);
//			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
//			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
//			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
//			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
//			const TensorDescriptor &cpu_bDesc = getTensor(bDesc);
//			const MemoryDescriptor &cpu_bMem = getMemory(bMem);
//			const TensorDescriptor &cpu_zDesc = getTensor(zDesc);
//			const MemoryDescriptor &cpu_zMem = getMemory(zMem);
//			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
//			MemoryDescriptor &cpu_yMem = getMemory(yMem);
//			MemoryDescriptor &cpu_workspaceMem = getMemory(workspaceMem);
//#if DYNAMIC_ARCH
//			switch (getSimdSupport())
//			{
//				case SimdLevel::AVX2:
//					return ns_avx2::cpu_convolutionBiasActivationForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
//							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation, cpu_workspaceMem);
//				case SimdLevel::AVX:
//					return ns_avx::cpu_convolutionBiasActivationForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
//							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation, cpu_workspaceMem);
//				case SimdLevel::SSE41:
//					return ns_sse41::cpu_convolutionBiasActivationForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
//							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation, cpu_workspaceMem);
//				case SimdLevel::SSE2:
//					return ns_sse2::cpu_convolutionBiasActivationForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
//							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation, cpu_workspaceMem);
//				case SimdLevel::NONE:
//					return ns_none::cpu_convolutionBiasActivationForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
//							cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation, cpu_workspaceMem);
//				default:
//					return AVOCADO_STATUS_NOT_SUPPORTED;
//			}
//#else
//			return SIMD_NAMESPACE::cpu_convolutionBiasActivationForward(cpu_context, cpu_config, alpha1, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem,
//					cpu_bDesc, cpu_bMem, alpha2, cpu_zDesc, cpu_zMem, beta, cpu_yDesc, cpu_yMem, activation, cpu_workspaceMem);
//#endif
//		}
//		avStatus_t cpuConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t workspaceMem)
//		{
//			const ContextDescriptor &cpu_context = getContext(context);
//			const ConvolutionDescriptor &cpu_config = getConvolution(config);
//			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
//			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
//			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
//			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
//			const TensorDescriptor &cpu_yDesc = getTensor(yDesc);
//			MemoryDescriptor &cpu_yMem = getMemory(yMem);
//			MemoryDescriptor &cpu_workspaceMem = getMemory(workspaceMem);
//#if DYNAMIC_ARCH
//			switch (getSimdSupport())
//			{
//				case SimdLevel::AVX2:
//					return ns_avx2::cpu_convolutionForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem, beta, cpu_yDesc,
//							cpu_yMem, cpu_workspaceMem);
//				case SimdLevel::AVX:
//					return ns_avx::cpu_convolutionForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem, beta, cpu_yDesc,
//							cpu_yMem, cpu_workspaceMem);
//				case SimdLevel::SSE41:
//					return ns_sse41::cpu_convolutionForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem, beta, cpu_yDesc,
//							cpu_yMem, cpu_workspaceMem);
//				case SimdLevel::SSE2:
//					return ns_sse2::cpu_convolutionForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem, beta, cpu_yDesc,
//							cpu_yMem, cpu_workspaceMem);
//				case SimdLevel::NONE:
//					return ns_none::cpu_convolutionForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem, beta, cpu_yDesc,
//							cpu_yMem, cpu_workspaceMem);
//				default:
//					return AVOCADO_STATUS_NOT_SUPPORTED;
//			}
//#else
//			return SIMD_NAMESPACE::cpu_convolutionForward(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_wDesc, cpu_wMem, beta, cpu_yDesc,
//					cpu_yMem, cpu_workspaceMem);
//#endif
//		}
//		avStatus_t cpuConvolutionBackward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const void *beta, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, avMemoryDescriptor_t workspaceMem)
//		{
//			const ContextDescriptor &cpu_context = getContext(context);
//			const ConvolutionDescriptor &cpu_config = getConvolution(config);
//			const TensorDescriptor &cpu_dxDesc = getTensor(dxDesc);
//			MemoryDescriptor &cpu_dxMem = getMemory(dxMem);
//			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
//			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
//			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
//			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
//			MemoryDescriptor &cpu_workspaceMem = getMemory(workspaceMem);
//#if DYNAMIC_ARCH
//			switch (getSimdSupport())
//			{
//				case SimdLevel::AVX2:
//					return ns_avx2::cpu_convolutionBackward(cpu_context, cpu_config, alpha, cpu_dxDesc, cpu_dxMem, cpu_wDesc, cpu_wMem, beta,
//							cpu_dyDesc, cpu_dyMem, cpu_workspaceMem);
//				case SimdLevel::AVX:
//					return ns_avx::cpu_convolutionBackward(cpu_context, cpu_config, alpha, cpu_dxDesc, cpu_dxMem, cpu_wDesc, cpu_wMem, beta,
//							cpu_dyDesc, cpu_dyMem, cpu_workspaceMem);
//				case SimdLevel::SSE41:
//					return ns_sse41::cpu_convolutionBackward(cpu_context, cpu_config, alpha, cpu_dxDesc, cpu_dxMem, cpu_wDesc, cpu_wMem, beta,
//							cpu_dyDesc, cpu_dyMem, cpu_workspaceMem);
//				case SimdLevel::SSE2:
//					return ns_sse2::cpu_convolutionBackward(cpu_context, cpu_config, alpha, cpu_dxDesc, cpu_dxMem, cpu_wDesc, cpu_wMem, beta,
//							cpu_dyDesc, cpu_dyMem, cpu_workspaceMem);
//				case SimdLevel::NONE:
//					return ns_none::cpu_convolutionBackward(cpu_context, cpu_config, alpha, cpu_dxDesc, cpu_dxMem, cpu_wDesc, cpu_wMem, beta,
//							cpu_dyDesc, cpu_dyMem, cpu_workspaceMem);
//				default:
//					return AVOCADO_STATUS_NOT_SUPPORTED;
//			}
//#else
//			return SIMD_NAMESPACE::cpu_convolutionBackward(cpu_context, cpu_config, alpha, cpu_dxDesc, cpu_dxMem, cpu_wDesc, cpu_wMem, beta,
//					cpu_dyDesc, cpu_dyMem, cpu_workspaceMem);
//#endif
//		}
//		avStatus_t cpuConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
//				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem,
//				avMemoryDescriptor_t workspaceMem)
//		{
//			const ContextDescriptor &cpu_context = getContext(context);
//			const ConvolutionDescriptor &cpu_config = getConvolution(config);
//			const TensorDescriptor &cpu_xDesc = getTensor(xDesc);
//			const MemoryDescriptor &cpu_xMem = getMemory(xMem);
//			const TensorDescriptor &cpu_dyDesc = getTensor(dyDesc);
//			const MemoryDescriptor &cpu_dyMem = getMemory(dyMem);
//			const TensorDescriptor &cpu_dwDesc = getTensor(dwDesc);
//			MemoryDescriptor &cpu_dwMem = getMemory(dwMem);
//			MemoryDescriptor &cpu_workspaceMem = getMemory(workspaceMem);
//#if DYNAMIC_ARCH
//			switch (getSimdSupport())
//			{
//				case SimdLevel::AVX2:
//					return ns_avx2::cpu_convolutionUpdate(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dwDesc,
//							cpu_dwMem, cpu_workspaceMem);
//				case SimdLevel::AVX:
//					return ns_avx::cpu_convolutionUpdate(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dwDesc,
//							cpu_dwMem, cpu_workspaceMem);
//				case SimdLevel::SSE41:
//					return ns_sse41::cpu_convolutionUpdate(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dwDesc,
//							cpu_dwMem, cpu_workspaceMem);
//				case SimdLevel::SSE2:
//					return ns_sse2::cpu_convolutionUpdate(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dwDesc,
//							cpu_dwMem, cpu_workspaceMem);
//				case SimdLevel::NONE:
//					return ns_none::cpu_convolutionUpdate(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dwDesc,
//							cpu_dwMem, cpu_workspaceMem);
//				default:
//					return AVOCADO_STATUS_NOT_SUPPORTED;
//			}
//#else
//			return SIMD_NAMESPACE::cpu_convolutionUpdate(cpu_context, cpu_config, alpha, cpu_xDesc, cpu_xMem, cpu_dyDesc, cpu_dyMem, beta, cpu_dwDesc,
//					cpu_dwMem, cpu_workspaceMem);
//#endif
//		}

		/*
		 *
		 * Training operations.
		 *
		 */

		avStatus_t cpuMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_outputDesc = getTensor(outputDesc);
			const MemoryDescriptor &cpu_outputMem = getMemory(outputMem);
			const TensorDescriptor &cpu_targetDesc = getTensor(targetDesc);
			const MemoryDescriptor &cpu_targetMem = getMemory(targetMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_metricFunction(cpu_context, metricType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::AVX:
					return ns_avx::cpu_metricFunction(cpu_context, metricType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_metricFunction(cpu_context, metricType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_metricFunction(cpu_context, metricType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::NONE:
					return ns_none::cpu_metricFunction(cpu_context, metricType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_metricFunction(cpu_context, metricType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
#endif
		}
		avStatus_t cpuLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_outputDesc = getTensor(outputDesc);
			const MemoryDescriptor &cpu_outputMem = getMemory(outputMem);
			const TensorDescriptor &cpu_targetDesc = getTensor(targetDesc);
			const MemoryDescriptor &cpu_targetMem = getMemory(targetMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_lossFunction(cpu_context, lossType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::AVX:
					return ns_avx::cpu_lossFunction(cpu_context, lossType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_lossFunction(cpu_context, lossType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_lossFunction(cpu_context, lossType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				case SimdLevel::NONE:
					return ns_none::cpu_lossFunction(cpu_context, lossType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_lossFunction(cpu_context, lossType, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, result);
#endif
		}
		avStatus_t cpuLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_outputDesc = getTensor(outputDesc);
			const MemoryDescriptor &cpu_outputMem = getMemory(outputMem);
			const TensorDescriptor &cpu_targetDesc = getTensor(targetDesc);
			const MemoryDescriptor &cpu_targetMem = getMemory(targetMem);
			const TensorDescriptor &cpu_gradientDesc = getTensor(gradientDesc);
			MemoryDescriptor &cpu_gradientMem = getMemory(gradientMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_lossGradient(cpu_context, lossType, alpha, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, beta,
							cpu_gradientDesc, cpu_gradientMem, isFused);
				case SimdLevel::AVX:
					return ns_avx::cpu_lossGradient(cpu_context, lossType, alpha, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, beta,
							cpu_gradientDesc, cpu_gradientMem, isFused);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_lossGradient(cpu_context, lossType, alpha, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, beta,
							cpu_gradientDesc, cpu_gradientMem, isFused);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_lossGradient(cpu_context, lossType, alpha, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, beta,
							cpu_gradientDesc, cpu_gradientMem, isFused);
				case SimdLevel::NONE:
					return ns_none::cpu_lossGradient(cpu_context, lossType, alpha, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, beta,
							cpu_gradientDesc, cpu_gradientMem, isFused);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_lossGradient(cpu_context, lossType, alpha, cpu_outputDesc, cpu_outputMem, cpu_targetDesc, cpu_targetMem, beta,
					cpu_gradientDesc, cpu_gradientMem, isFused);
#endif
		}
		avStatus_t cpuOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem, const void *beta, const avTensorDescriptor_t wDesc,
				avMemoryDescriptor_t wMem, avMemoryDescriptor_t workspaceMem)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			OptimizerDescriptor &cpu_config = getOptimizer(config);
			const TensorDescriptor &cpu_dwDesc = getTensor(dwDesc);
			const MemoryDescriptor &cpu_dwMem = getMemory(dwMem);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			MemoryDescriptor &cpu_wMem = getMemory(wMem);
			MemoryDescriptor &cpu_workspaceMem = getMemory(workspaceMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_optimizerLearn(cpu_context, cpu_config, alpha, cpu_dwDesc, cpu_dwMem, beta, cpu_wDesc, cpu_wMem, cpu_workspaceMem);
				case SimdLevel::AVX:
					return ns_avx::cpu_optimizerLearn(cpu_context, cpu_config, alpha, cpu_dwDesc, cpu_dwMem, beta, cpu_wDesc, cpu_wMem, cpu_workspaceMem);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_optimizerLearn(cpu_context, cpu_config, alpha, cpu_dwDesc, cpu_dwMem, beta, cpu_wDesc, cpu_wMem, cpu_workspaceMem);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_optimizerLearn(cpu_context, cpu_config, alpha, cpu_dwDesc, cpu_dwMem, beta, cpu_wDesc, cpu_wMem, cpu_workspaceMem);
				case SimdLevel::NONE:
					return ns_none::cpu_optimizerLearn(cpu_context, cpu_config, alpha, cpu_dwDesc, cpu_dwMem, beta, cpu_wDesc, cpu_wMem, cpu_workspaceMem);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_optimizerLearn(cpu_context, cpu_config, alpha, cpu_dwDesc, cpu_dwMem, beta, cpu_wDesc, cpu_wMem,
					cpu_workspaceMem);
#endif
		}
		avStatus_t cpuRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const void *coefficient, const void *offset, void *loss)
		{
			const ContextDescriptor &cpu_context = getContext(context);
			const TensorDescriptor &cpu_dwDesc = getTensor(dwDesc);
			MemoryDescriptor &cpu_dwMem = getMemory(dwMem);
			const TensorDescriptor &cpu_wDesc = getTensor(wDesc);
			const MemoryDescriptor &cpu_wMem = getMemory(wMem);
#if DYNAMIC_ARCH
			switch (getSimdSupport())
			{
				case SimdLevel::AVX2:
					return ns_avx2::cpu_regularizerL2(cpu_context, cpu_dwDesc, cpu_dwMem, cpu_wDesc, cpu_wMem, coefficient, offset, loss);
				case SimdLevel::AVX:
					return ns_avx::cpu_regularizerL2(cpu_context, cpu_dwDesc, cpu_dwMem, cpu_wDesc, cpu_wMem, coefficient, offset, loss);
				case SimdLevel::SSE41:
					return ns_sse41::cpu_regularizerL2(cpu_context, cpu_dwDesc, cpu_dwMem, cpu_wDesc, cpu_wMem, coefficient, offset, loss);
				case SimdLevel::SSE2:
					return ns_sse2::cpu_regularizerL2(cpu_context, cpu_dwDesc, cpu_dwMem, cpu_wDesc, cpu_wMem, coefficient, offset, loss);
				case SimdLevel::NONE:
					return ns_none::cpu_regularizerL2(cpu_context, cpu_dwDesc, cpu_dwMem, cpu_wDesc, cpu_wMem, coefficient, offset, loss);
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
#else
			return SIMD_NAMESPACE::cpu_regularizerL2(cpu_context, cpu_dwDesc, cpu_dwMem, cpu_wDesc, cpu_wMem, coefficient, offset, loss);
#endif
		}

	} /* namespace backend */
} /* namespace avocado */

