/*
 * descriptors.cpp
 *
 *  Created on: Dec 20, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CpuBackend/cpu_backend.h>
#include <backend_descriptors.hpp>

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
	using namespace avocado::backend;
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
			return cpu::create<cpu::MemoryDescriptor>(result, sizeInBytes);
		}
		avStatus_t cpuCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t sizeInBytes, avSize_t offsetInBytes)
		{
			return cpu::create<cpu::MemoryDescriptor>(result, cpu::getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t cpuDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return cpu::destroy<cpu::MemoryDescriptor>(desc);
		}
		avStatus_t cpuSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstOffset, avSize_t dstSize, const void *pattern,
				avSize_t patternSize)
		{
			if (not cpu::same_device_type(context, dst))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;
			if (cpu::getPointer(dst) == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if (pattern == nullptr)
			{
				std::memset(cpu::getPointer<int8_t>(dst) + dstOffset, 0, dstSize);
				return AVOCADO_STATUS_SUCCESS;
			}

			if (dstSize % patternSize != 0 or dstOffset % patternSize != 0)
				return AVOCADO_STATUS_BAD_PARAM;

			// buffer size must be divisible by pattern size, using about 256 bytes, but not less than then the actual pattern size
			const avSize_t buffer_size = patternSize * std::max(1ull, (256ull / patternSize)); // bytes
			if (dstSize >= 4 * buffer_size)
			{
				uint8_t buffer[buffer_size];
				for (avSize_t i = 0; i < buffer_size; i += patternSize)
					std::memcpy(buffer + i, pattern, patternSize);
				for (avSize_t i = 0; i < dstSize; i += buffer_size)
					std::memcpy(cpu::getPointer<uint8_t>(dst) + dstOffset + i, buffer, std::min(buffer_size, dstSize - i));
			}
			else
			{
				for (avSize_t i = 0; i < dstSize; i += patternSize)
					std::memcpy(cpu::getPointer<uint8_t>(dst) + dstOffset + i, pattern, patternSize);
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstOffset, const avMemoryDescriptor_t src,
				avSize_t srcOffset, avSize_t count)
		{
			if (not cpu::same_device_type(context, dst, src))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;
			std::memcpy(cpu::getPointer<int8_t>(dst) + dstOffset, cpu::getPointer<int8_t>(src) + srcOffset, count);
			return AVOCADO_STATUS_SUCCESS;
		}
		void* cpuGetMemoryPointer(avMemoryDescriptor_t mem)
		{
			return cpu::getPointer(mem);
		}

		avStatus_t cpuCreateContextDescriptor(avContextDescriptor_t *result)
		{
			return cpu::create<cpu::ContextDescriptor>(result);
		}
		avStatus_t cpuDestroyContextDescriptor(avContextDescriptor_t desc)
		{
			if (cpu::isDefault(desc))
				return AVOCADO_STATUS_BAD_PARAM;
			return cpu::destroy<cpu::ContextDescriptor>(desc);
		}
		avContextDescriptor_t cpuGetDefaultContext()
		{
			return cpu::create_descriptor(0, cpu::ContextDescriptor::descriptor_type);
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
			return cpu::create<cpu::TensorDescriptor>(result);
		}
		avStatus_t cpuDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			return cpu::destroy<cpu::TensorDescriptor>(desc);
		}
		avStatus_t cpuSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
		{
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS or dimensions == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				cpu::getTensor(desc).set(dtype, nbDims, dimensions);
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
				cpu::getTensor(desc).get(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cpuCreateConvolutionDescriptor(avConvolutionDescriptor_t *result)
		{
			return cpu::create<cpu::ConvolutionDescriptor>(result);
		}
		avStatus_t cpuDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
		{
			return cpu::destroy<cpu::ConvolutionDescriptor>(desc);
		}
		avStatus_t cpuSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue)
		{
			try
			{
				cpu::getConvolution(desc).set(mode, nbDims, padding, strides, dilation, groups, paddingValue);
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
				cpu::getConvolution(desc).get(mode, nbDims, padding, strides, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cpuCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
		{
			return cpu::create<cpu::OptimizerDescriptor>(result);
		}
		avStatus_t cpuDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
		{
			return cpu::destroy<cpu::OptimizerDescriptor>(desc);
		}
		avStatus_t cpuSetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t type, double learningRate, const double coefficients[],
				const bool flags[])
		{
			try
			{
				cpu::getOptimizer(desc).set(type, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t *type, double *learningRate, double coefficients[],
				bool flags[])
		{
			try
			{
				cpu::getOptimizer(desc).get(type, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, avSize_t *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				cpu::getOptimizer(desc).get_workspace_size(result, cpu::getTensor(wDesc));
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

