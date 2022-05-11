/*
 * softmax.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	struct limits
	{
			static constexpr T min_value = std::numeric_limits<T>::lowest();
	};
	template<>
	struct limits<float16>
	{
			static constexpr float16 min_value { 0xFbffu }; // -65504
	};
	template<>
	struct limits<bfloat16>
	{
			static constexpr bfloat16 min_value { 0xFf7fu }; //  approx. -3.402 × 10^38
	};

	template<typename T, typename U = T>
	void kernel_softmax_forward(U alpha, const T *input, U beta, T *output, int first_dim, int last_dim, T *workspace)
	{
		assert(input != nullptr);
		assert(output != nullptr);
#pragma omp parallel
		{
			T *thread_workspace = workspace + omp_get_thread_num() * last_dim;
#pragma omp for
			for (int i = 0; i < first_dim; i++)
			{
				SIMD<T> max_value(limits<T>::min_value);
				for (int j = 0; j < last_dim; j += SIMD<T>::length)
				{
					const int elements_left = std::min(last_dim - j, SIMD<T>::length);
					SIMD<T> loaded(input + i * last_dim + j, elements_left);
					loaded.cutoff(elements_left, limits<T>::min_value);
					max_value = max(max_value, loaded);
				}
				max_value = horizontal_max(max_value);

				SIMD<T> sum_value = SIMD<T>::zero();
				for (int j = 0; j < last_dim; j += SIMD<T>::length)
				{
					const int elements_left = std::min(last_dim - j, SIMD<T>::length);
					SIMD<T> loaded(input + i * last_dim + j, elements_left);
					loaded = exp(loaded - max_value);
					loaded.cutoff(elements_left);
					sum_value += loaded;
					loaded.store(thread_workspace + j, elements_left);
				}
				T tmp = horizontal_add(sum_value);

				if (tmp == SIMD<T>::scalar_zero())
				{
					sum_value = alpha / static_cast<float>(last_dim);
					if (beta == scalar::zero<U>())
					{
						for (int j = 0; j < last_dim; j += SIMD<T>::length)
						{
							const int elements_left = std::min(last_dim - j, SIMD<T>::length);
							sum_value.store(output + i * last_dim + j, elements_left);
						}
					}
					else
					{
						for (int j = 0; j < last_dim; j += SIMD<T>::length)
						{
							const int elements_left = std::min(last_dim - j, SIMD<T>::length);
							SIMD<T> loaded_output(output + i * last_dim + j, elements_left);
							loaded_output = sum_value + beta * loaded_output;
							loaded_output.store(output + i * last_dim + j, elements_left);
						}
					}
				}
				else
				{
					sum_value = alpha / SIMD<T>(tmp);
					if (beta == scalar::zero<U>())
					{
						for (int j = 0; j < last_dim; j += SIMD<T>::length)
						{
							const int elements_left = std::min(last_dim - j, SIMD<T>::length);
							SIMD<T> loaded_workspace(thread_workspace + j, elements_left);

							loaded_workspace = sum_value * loaded_workspace;
							loaded_workspace.store(output + i * last_dim + j, elements_left);
						}
					}
					else
					{
						for (int j = 0; j < last_dim; j += SIMD<T>::length)
						{
							const int elements_left = std::min(last_dim - j, SIMD<T>::length);
							SIMD<T> loaded_workspace(thread_workspace + j, elements_left);
							SIMD<T> loaded_output(output + i * last_dim + j, elements_left);

							loaded_workspace = sum_value * loaded_workspace + beta * loaded_output;
							loaded_workspace.store(output + i * last_dim + j, elements_left);
						}
					}
				}
			}
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_softmaxForward(const ContextDescriptor &context, avSoftmaxMode_t mode, const void *alpha, const TensorDescriptor &xDesc,
			const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem)
	{
		int first_dim, last_dim;
		if (mode == AVOCADO_SOFTMAX_MODE_CHANNEL)
		{
			first_dim = xDesc.volumeWithoutLastDim();
			last_dim = xDesc.lastDim();
		}
		else
		{
			first_dim = xDesc.firstDim();
			last_dim = xDesc.volumeWithoutFirstDim();
		}

		const int required_workspace_size = cpuGetNumberOfThreads() * last_dim * dataTypeSize(xDesc.dtype());
		if (context.getWorkspace().sizeInBytes() < required_workspace_size)
			return AVOCADO_STATUS_INTERNAL_ERROR; // not enough workspace

		switch (xDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				kernel_softmax_forward(getAlphaValue(alpha), xMem.data<float16>(), getBetaValue(beta), yMem.data<float16>(), first_dim, last_dim,
						context.getWorkspace().data<float16>());
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_softmax_forward(getAlphaValue(alpha), xMem.data<bfloat16>(), getBetaValue(beta), yMem.data<bfloat16>(), first_dim, last_dim,
						context.getWorkspace().data<bfloat16>());
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_softmax_forward(getAlphaValue(alpha), xMem.data<float>(), getBetaValue(beta), yMem.data<float>(), first_dim, last_dim,
						context.getWorkspace().data<float>());
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_softmax_forward(getAlphaValue<double>(alpha), xMem.data<double>(), getBetaValue<double>(beta), yMem.data<double>(), first_dim,
						last_dim, context.getWorkspace().data<double>());
				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_softmaxBackward(const ContextDescriptor &context, avSoftmaxMode_t mode, const void *alpha, const TensorDescriptor &yDesc,
			const MemoryDescriptor &yMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const void *beta,
			const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem)
	{
		return SIMD_NAMESPACE::cpu_activationBackward(context, AVOCADO_ACTIVATION_SIGMOID, alpha, yDesc, yMem, dyDesc, dyMem, beta, dxDesc, dxMem);
	}

} /* namespace SIMD_NAMESPACE */

