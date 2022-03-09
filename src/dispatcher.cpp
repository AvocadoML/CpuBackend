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
		/*
		 *
		 * Tensor operations.
		 *
		 */
		avStatus_t cpuChangeTypeHost(avContextDescriptor_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType,
				av_int64 elements)
		{
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::MemoryDescriptor &cpu_src = cpu::getMemory(src);
			cpu::MemoryDescriptor &cpu_dst = cpu::getMemory(dst);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
			std::vector<const cpu::TensorDescriptor*> cpu_aDesc(nbTensors);
			std::vector<const cpu::MemoryDescriptor*> cpu_aMem(nbTensors);
			for (int i = 0; i < nbTensors; i++)
			{
				cpu_aDesc[i] = &(cpu::getTensor(aDesc[i]));
				cpu_aMem[i] = &(cpu::getMemory(aMem[i]));
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
			std::vector<const cpu::TensorDescriptor*> cpu_cDesc(nbTensors);
			std::vector<cpu::MemoryDescriptor*> cpu_cMem(nbTensors);
			for (int i = 0; i < nbTensors; i++)
			{
				cpu_cDesc[i] = &(cpu::getTensor(cDesc[i]));
				cpu_cMem[i] = &(cpu::getMemory(cMem[i]));
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);

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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);

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
		avStatus_t cpuAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem,
				const void *scalar, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
			const cpu::MemoryDescriptor &cpu_zMem = cpu::getMemory(zMem);

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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);

			return cpu_gemm(cpu_context, aOp, bOp, alpha, cpu_aDesc, cpu_aMem, cpu_bDesc, cpu_bMem, beta, cpu_cDesc, cpu_cMem);
		}
		avStatus_t cpuGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_aDesc = cpu::getTensor(aDesc);
			const cpu::MemoryDescriptor &cpu_aMem = cpu::getMemory(aMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_cDesc = cpu::getTensor(cDesc);
			cpu::MemoryDescriptor &cpu_cMem = cpu::getMemory(cMem);

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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			const cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
			const cpu::TensorDescriptor &cpu_dxDesc = cpu::getTensor(dxDesc);
			cpu::MemoryDescriptor &cpu_dxMem = cpu::getMemory(dxMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			const cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
			const cpu::TensorDescriptor &cpu_dxDesc = cpu::getTensor(dxDesc);
			cpu::MemoryDescriptor &cpu_dxMem = cpu::getMemory(dxMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_scaleBiasMeanVarDesc = cpu::getTensor(scaleBiasMeanVarDesc);
			const cpu::MemoryDescriptor &cpu_scaleMem = cpu::getMemory(scaleMem);
			const cpu::MemoryDescriptor &cpu_biasMem = cpu::getMemory(biasMem);
			const cpu::MemoryDescriptor &cpu_meanMem = cpu::getMemory(meanMem);
			const cpu::MemoryDescriptor &cpu_varianceMem = cpu::getMemory(varianceMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_scaleBiasMeanVarDesc = cpu::getTensor(scaleBiasMeanVarDesc);
			const cpu::MemoryDescriptor &cpu_scaleMem = cpu::getMemory(scaleMem);
			const cpu::MemoryDescriptor &cpu_biasMem = cpu::getMemory(biasMem);
			cpu::MemoryDescriptor &cpu_meanMem = cpu::getMemory(meanMem);
			cpu::MemoryDescriptor &cpu_varianceMem = cpu::getMemory(varianceMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			const cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
			const cpu::TensorDescriptor &cpu_dxDesc = cpu::getTensor(dxDesc);
			cpu::MemoryDescriptor &cpu_dxMem = cpu::getMemory(dxMem);
			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
			cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
			const cpu::TensorDescriptor &cpu_scaleMeanVarDesc = cpu::getTensor(scaleMeanVarDesc);
			const cpu::MemoryDescriptor &cpu_scaleMem = cpu::getMemory(scaleMem);
			const cpu::MemoryDescriptor &cpu_meanMem = cpu::getMemory(meanMem);
			const cpu::MemoryDescriptor &cpu_varianceMem = cpu::getMemory(varianceMem);
			cpu::MemoryDescriptor &cpu_scaleUpdateMem = cpu::getMemory(scaleUpdateMem);
			cpu::MemoryDescriptor &cpu_biasUpdateMem = cpu::getMemory(biasUpdateMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::DropoutDescriptor &cpu_config = cpu::getDropout(config);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
			cpu::MemoryDescriptor &cpu_states = cpu::getMemory(states);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::DropoutDescriptor &cpu_config = cpu::getDropout(config);
			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
			const cpu::TensorDescriptor &cpu_dxDesc = cpu::getTensor(dxDesc);
			cpu::MemoryDescriptor &cpu_dxMem = cpu::getMemory(dxMem);
			const cpu::MemoryDescriptor &cpu_states = cpu::getMemory(states);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::PoolingDescriptor &cpu_config = cpu::getPooling(config);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::PoolingDescriptor &cpu_config = cpu::getPooling(config);
			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_dxDesc = cpu::getTensor(dxDesc);
			cpu::MemoryDescriptor &cpu_dxMem = cpu::getMemory(dxMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_filterDesc = cpu::getTensor(filterDesc);
			const cpu::TensorDescriptor &cpu_srcDesc = cpu::getTensor(srcDesc);
			const cpu::MemoryDescriptor &cpu_srcMem = cpu::getMemory(srcMem);
			const cpu::TensorDescriptor &cpu_rowDesc = cpu::getTensor(rowDesc);
			cpu::MemoryDescriptor &cpu_rowMem = cpu::getMemory(rowMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_zDesc = cpu::getTensor(zDesc);
			const cpu::MemoryDescriptor &cpu_zMem = cpu::getMemory(zMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_zDesc = cpu::getTensor(zDesc);
			const cpu::MemoryDescriptor &cpu_zMem = cpu::getMemory(zMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
			const cpu::TensorDescriptor &cpu_matricesDesc = cpu::getTensor(matricesDesc);
			cpu::MemoryDescriptor &cpu_matricesMem = cpu::getMemory(matricesMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
			const cpu::TensorDescriptor &cpu_matricesDesc = cpu::getTensor(matricesDesc);
			cpu::MemoryDescriptor &cpu_matricesMem = cpu::getMemory(matricesMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_matricesDesc = cpu::getTensor(matricesDesc);
			const cpu::MemoryDescriptor &cpu_matricesMem = cpu::getMemory(matricesMem);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
			const cpu::TensorDescriptor &cpu_zDesc = cpu::getTensor(zDesc);
			const cpu::MemoryDescriptor &cpu_zMem = cpu::getMemory(zMem);
			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
			const cpu::TensorDescriptor &cpu_matricesDesc = cpu::getTensor(matricesDesc);
			cpu::MemoryDescriptor &cpu_matricesMem = cpu::getMemory(matricesMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
			const cpu::TensorDescriptor &cpu_matricesDesc = cpu::getTensor(matricesDesc);
			const cpu::MemoryDescriptor &cpu_matricesMem = cpu::getMemory(matricesMem);
			const cpu::TensorDescriptor &cpu_dwDesc = cpu::getTensor(dwDesc);
			cpu::MemoryDescriptor &cpu_dwMem = cpu::getMemory(dwMem);

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
//			const cpu::ConvolutionDescriptor &cpu_config = cpu::const_getConvolution(config);
//			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
//			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
//
//			return cpu_getConvolutionWorkspaceSize(cpu_config, cpu_xDesc, cpu_wDesc, inferenceOnly, result);
//		}
//		avStatus_t cpuConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
//				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
//				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
//				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
//				const avActivationType_t activation, avMemoryDescriptor_t workspaceMem)
//		{
//			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
//			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
//			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
//			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
//			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
//			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
//			const cpu::TensorDescriptor &cpu_bDesc = cpu::getTensor(bDesc);
//			const cpu::MemoryDescriptor &cpu_bMem = cpu::getMemory(bMem);
//			const cpu::TensorDescriptor &cpu_zDesc = cpu::getTensor(zDesc);
//			const cpu::MemoryDescriptor &cpu_zMem = cpu::getMemory(zMem);
//			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
//			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
//			cpu::MemoryDescriptor &cpu_workspaceMem = cpu::getMemory(workspaceMem);
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
//			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
//			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
//			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
//			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
//			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
//			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
//			const cpu::TensorDescriptor &cpu_yDesc = cpu::getTensor(yDesc);
//			cpu::MemoryDescriptor &cpu_yMem = cpu::getMemory(yMem);
//			cpu::MemoryDescriptor &cpu_workspaceMem = cpu::getMemory(workspaceMem);
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
//			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
//			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
//			const cpu::TensorDescriptor &cpu_dxDesc = cpu::getTensor(dxDesc);
//			cpu::MemoryDescriptor &cpu_dxMem = cpu::getMemory(dxMem);
//			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
//			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
//			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
//			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
//			cpu::MemoryDescriptor &cpu_workspaceMem = cpu::getMemory(workspaceMem);
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
//			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
//			const cpu::ConvolutionDescriptor &cpu_config = cpu::getConvolution(config);
//			const cpu::TensorDescriptor &cpu_xDesc = cpu::getTensor(xDesc);
//			const cpu::MemoryDescriptor &cpu_xMem = cpu::getMemory(xMem);
//			const cpu::TensorDescriptor &cpu_dyDesc = cpu::getTensor(dyDesc);
//			const cpu::MemoryDescriptor &cpu_dyMem = cpu::getMemory(dyMem);
//			const cpu::TensorDescriptor &cpu_dwDesc = cpu::getTensor(dwDesc);
//			cpu::MemoryDescriptor &cpu_dwMem = cpu::getMemory(dwMem);
//			cpu::MemoryDescriptor &cpu_workspaceMem = cpu::getMemory(workspaceMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_outputDesc = cpu::getTensor(outputDesc);
			const cpu::MemoryDescriptor &cpu_outputMem = cpu::getMemory(outputMem);
			const cpu::TensorDescriptor &cpu_targetDesc = cpu::getTensor(targetDesc);
			const cpu::MemoryDescriptor &cpu_targetMem = cpu::getMemory(targetMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_outputDesc = cpu::getTensor(outputDesc);
			const cpu::MemoryDescriptor &cpu_outputMem = cpu::getMemory(outputMem);
			const cpu::TensorDescriptor &cpu_targetDesc = cpu::getTensor(targetDesc);
			const cpu::MemoryDescriptor &cpu_targetMem = cpu::getMemory(targetMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_outputDesc = cpu::getTensor(outputDesc);
			const cpu::MemoryDescriptor &cpu_outputMem = cpu::getMemory(outputMem);
			const cpu::TensorDescriptor &cpu_targetDesc = cpu::getTensor(targetDesc);
			const cpu::MemoryDescriptor &cpu_targetMem = cpu::getMemory(targetMem);
			const cpu::TensorDescriptor &cpu_gradientDesc = cpu::getTensor(gradientDesc);
			cpu::MemoryDescriptor &cpu_gradientMem = cpu::getMemory(gradientMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::OptimizerDescriptor &cpu_config = cpu::getOptimizer(config);
			const cpu::TensorDescriptor &cpu_dwDesc = cpu::getTensor(dwDesc);
			const cpu::MemoryDescriptor &cpu_dwMem = cpu::getMemory(dwMem);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
			cpu::MemoryDescriptor &cpu_workspaceMem = cpu::getMemory(workspaceMem);
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
			const cpu::ContextDescriptor &cpu_context = cpu::getContext(context);
			const cpu::TensorDescriptor &cpu_dwDesc = cpu::getTensor(dwDesc);
			cpu::MemoryDescriptor &cpu_dwMem = cpu::getMemory(dwMem);
			const cpu::TensorDescriptor &cpu_wDesc = cpu::getTensor(wDesc);
			const cpu::MemoryDescriptor &cpu_wMem = cpu::getMemory(wMem);
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

