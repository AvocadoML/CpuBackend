/*
 * descriptors.cpp
 *
 *  Created on: Dec 20, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cpu_backend.h>
#include <Avocado/backend_descriptors.hpp>

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
		using namespace avocado::backend::BACKEND_NAMESPACE;

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

		avStatus_t cpuCreateMemoryDescriptor(avMemoryDescriptor_t *result, av_int64 sizeInBytes)
		{
			return create_descriptor<MemoryDescriptor>(result, sizeInBytes, 0);
		}
		avStatus_t cpuCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, av_int64 sizeInBytes, av_int64 offsetInBytes)
		{
//			std::cout << __FUNCTION__ << "() owning memory = " << desc << ", view size = " << sizeInBytes << ", offset = " << offsetInBytes << '\n';
			return create_descriptor<MemoryDescriptor>(result, getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t cpuDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return destroy_descriptor<MemoryDescriptor>(desc);
		}
		avStatus_t cpuSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, av_int64 dstSize, const void *pattern,
				av_int64 patternSize)
		{
			if (not same_device_type(context, dst))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;
			if (getPointer(dst) == nullptr)
			{
				if (dstSize != 0)
					return AVOCADO_STATUS_BAD_PARAM;
				else
					return AVOCADO_STATUS_SUCCESS;
			}
			if (pattern == nullptr)
			{
				std::memset(getPointer<int8_t>(dst) + dstOffset, 0, dstSize);
				return AVOCADO_STATUS_SUCCESS;
			}

			if (dstSize % patternSize != 0 or dstOffset % patternSize != 0)
				return AVOCADO_STATUS_BAD_PARAM;

			// buffer size must be divisible by pattern size, using about 256 bytes, but not less than then the actual pattern size
			const av_int64 buffer_size = patternSize * std::max(1ull, (256ull / patternSize)); // bytes
			if (dstSize >= 4 * buffer_size)
			{
				uint8_t buffer[buffer_size];
				for (av_int64 i = 0; i < buffer_size; i += patternSize)
					std::memcpy(buffer + i, pattern, patternSize);
				for (av_int64 i = 0; i < dstSize; i += buffer_size)
					std::memcpy(getPointer<uint8_t>(dst) + dstOffset + i, buffer, std::min(buffer_size, dstSize - i));
			}
			else
			{
				for (av_int64 i = 0; i < dstSize; i += patternSize)
					std::memcpy(getPointer<uint8_t>(dst) + dstOffset + i, pattern, patternSize);
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, const avMemoryDescriptor_t src,
				av_int64 srcOffset, av_int64 count)
		{
			if (not same_device_type(context, dst, src))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;
//			std::cout << "tutaj\n";
//			std::cout << dst << " " << src << '\n';
//			std::cout << getPointer<int8_t>(dst) << " " << dstOffset << " " << getMemory(dst).size() << '\n';
//			std::cout << getPointer<int8_t>(src) << " " << srcOffset << " " << getMemory(src).size() << '\n';
			std::memcpy(getPointer<int8_t>(dst) + dstOffset, getPointer<int8_t>(src) + srcOffset, count);
			return AVOCADO_STATUS_SUCCESS;
		}
		void* cpuGetMemoryPointer(avMemoryDescriptor_t mem)
		{
			return getPointer(mem);
		}

		avStatus_t cpuCreateContextDescriptor(avContextDescriptor_t *result)
		{
			return create_descriptor<ContextDescriptor>(result, 0);
		}
		avStatus_t cpuDestroyContextDescriptor(avContextDescriptor_t desc)
		{
			return destroy_descriptor<ContextDescriptor>(desc);
		}
		avContextDescriptor_t cpuGetDefaultContext()
		{
			return createDescriptor(AVOCADO_NULL_DESCRIPTOR, ContextDescriptor::descriptor_type);
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
			return create_descriptor<TensorDescriptor>(result, std::initializer_list<int> { }, AVOCADO_DTYPE_UNKNOWN);
		}
		avStatus_t cpuDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			return destroy_descriptor<TensorDescriptor>(desc);
		}
		avStatus_t cpuSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
		{
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS)
				return AVOCADO_STATUS_BAD_PARAM;
			if (dimensions == nullptr and nbDims != 0)
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
			return create_descriptor<ConvolutionDescriptor>(result);
		}
		avStatus_t cpuDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
		{
			if (desc == AVOCADO_NULL_DESCRIPTOR)
				return AVOCADO_STATUS_BAD_PARAM;
			return destroy_descriptor<ConvolutionDescriptor>(desc);
		}
		avStatus_t cpuSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue)
		{
			try
			{
				getConvolution(desc).set(mode, nbDims, padding, strides, dilation, groups, paddingValue);
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
				getConvolution(desc).get(mode, nbDims, padding, strides, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cpuCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
		{
			return create_descriptor<OptimizerDescriptor>(result);
		}
		avStatus_t cpuDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
		{
			if (desc == AVOCADO_NULL_DESCRIPTOR)
				return AVOCADO_STATUS_BAD_PARAM;
			return destroy_descriptor<OptimizerDescriptor>(desc);
		}
		avStatus_t cpuSetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t type, av_int64 steps, double learningRate,
				const double coefficients[], const bool flags[])
		{
			try
			{
				getOptimizer(desc).set(type, steps, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t *type, av_int64 *steps, double *learningRate,
				double coefficients[], bool flags[])
		{
			try
			{
				getOptimizer(desc).get(type, steps, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cpuGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, av_int64 *result)
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

