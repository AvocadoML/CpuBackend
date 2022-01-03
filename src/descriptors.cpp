/*
 * descriptors.cpp
 *
 *  Created on: Dec 20, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cpu_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include <omp.h>

#if USE_BLIS
#  ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#    include "../../../extern/blis/blis.h"
#    pragma GCC diagnostic pop
#  else
#    include "../../../extern/blis/blis.h"
#  endif
#endif

#if USE_OPENBLAS
#    ifdef __linux__
#      include <cblas.h>
#    else
#      include <openblas/cblas.h>
#    endif
#endif

namespace
{
	thread_local int number_of_threads = omp_get_num_procs();
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cpuSetNumberOfThreads(int threads)
		{
#if USE_BLIS
			bli_thread_set_num_threads(threads);
#endif
#if USE_OPENBLAS
			openblas_set_num_threads(threads);
#endif
			omp_set_num_threads(threads);
			number_of_threads = threads;
			return AVOCADO_STATUS_SUCCESS;
		}

		int cpuGetNumberOfThreads()
		{
			return number_of_threads;
		}

		avStatus_t cpuCreateMemoryDescriptor(avMemoryDescriptor_t *result, avSize_t sizeInBytes)
		{
			return internal::create<MemoryDescriptor>(result, sizeInBytes);
		}
		avStatus_t cpuCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t sizeInBytes, avSize_t offsetInBytes)
		{
			return internal::create<MemoryDescriptor>(result, getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t cpuDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return internal::destroy<MemoryDescriptor>(desc);
		}
		avStatus_t cpuSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstSize, const void *pattern, avSize_t patternSize)
		{
			if (getPointer(dst) == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if (pattern == nullptr)
			{
				std::memset(getPointer(dst), 0, dstSize);
				return AVOCADO_STATUS_SUCCESS;
			}

			if (dstSize % patternSize != 0)
				return AVOCADO_STATUS_BAD_PARAM;

			// buffer size must be divisible by pattern size, using about 256 bytes, but not less than then the actual pattern size
			const avSize_t buffer_size = patternSize * std::max(1ull, (256ull / patternSize)); // bytes
			if (dstSize >= 4 * buffer_size)
			{
				uint8_t buffer[buffer_size];
				for (avSize_t i = 0; i < buffer_size; i += patternSize)
					std::memcpy(buffer + i, pattern, patternSize);
				for (avSize_t i = 0; i < dstSize; i += buffer_size)
					std::memcpy(getPointer<uint8_t>(dst) + i, buffer, std::min(buffer_size, dstSize - i));
			}
			else
			{
				for (avSize_t i = 0; i < dstSize; i += patternSize)
					std::memcpy(getPointer<uint8_t>(dst) + i, pattern, patternSize);
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, const avMemoryDescriptor_t src, avSize_t count)
		{
			std::memcpy(getPointer(dst), getPointer(src), count);
			return AVOCADO_STATUS_SUCCESS;
		}
		void* cpuGetMemoryPointer(avMemoryDescriptor_t mem)
		{
			return getPointer(mem);
		}

		avStatus_t cpuCreateContextDescriptor(avContextDescriptor_t *result)
		{
			return internal::create<ContextDescriptor>(result);
		}
		avStatus_t cpuDestroyContextDescriptor(avContextDescriptor_t desc)
		{
			return internal::destroy<ContextDescriptor>(desc);
		}
		avStatus_t cpuSynchronizeWithContext(avContextDescriptor_t context)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuIsContextReady(avContextDescriptor_t context, bool *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			result[0] = true;
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cpuCreateTensorDescriptor(avTensorDescriptor_t *result)
		{
			return internal::create<TensorDescriptor>(result);
		}
		avStatus_t cpuDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			return internal::destroy<TensorDescriptor>(desc);
		}
		avStatus_t cpuSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
		{
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS or dimensions == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getTensor(desc).set(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[])
		{
			try
			{
				getTensor(desc).get(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cpuCreateConvolutionDescriptor(avConvolutionDescriptor_t *result)
		{
			return internal::create<ConvolutionDescriptor>(result);
		}
		avStatus_t cpuDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
		{
			return internal::destroy<ConvolutionDescriptor>(desc);
		}
		avStatus_t cpuSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue)
		{
			try
			{
				getConvolution(desc).set(mode, nbDims, strides, padding, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[],
				int dilation[], int *groups, void *paddingValue)
		{
			try
			{
				getConvolution(desc).get(mode, nbDims, strides, padding, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cpuCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
		{
			return internal::create<OptimizerDescriptor>(result);
		}
		avStatus_t cpuDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
		{
			return internal::destroy<OptimizerDescriptor>(desc);
		}
		avStatus_t cpuSetOptimizerSGD(avOptimizerDescriptor_t desc, double learningRate, bool useMomentum, bool useNesterov, double beta1)
		{
			try
			{
				getOptimizer(desc).set_sgd(learningRate, useMomentum, useNesterov, beta1);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerSGD(avOptimizerDescriptor_t desc, double *learningRate, bool *useMomentum, bool *useNesterov, double *beta1)
		{
			try
			{
				getOptimizer(desc).get_sgd(learningRate, useMomentum, useNesterov, beta1);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuSetOptimizerADAM(avOptimizerDescriptor_t desc, double learningRate, double beta1, double beta2)
		{
			try
			{
				getOptimizer(desc).set_adam(learningRate, beta1, beta2);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerADAM(avOptimizerDescriptor_t desc, double *learningRate, double *beta1, double *beta2)
		{
			try
			{
				getOptimizer(desc).get_adam(learningRate, beta1, beta2);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerType(avOptimizerDescriptor_t desc, avOptimizerType_t *type)
		{
			if (type == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getOptimizer(desc).get_type(type);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, int *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getOptimizer(desc).get_workspace_size(result, getTensor(wDesc));
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

