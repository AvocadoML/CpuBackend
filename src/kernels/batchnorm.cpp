/*
 * batchnorm.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>
#include "activation.hpp"
#include "array_utils.hpp"

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<typename T, typename U>
	void kernel_affine_forward(U alpha, U beta, const T *xMem, T *yMem, const T *wMem, const T *bMem, cpu::BroadcastedDimensions dims,
			avActivationType_t activation)
	{
#pragma omp parallel for
		for (int i = 0; i < dims.first; i++)
			for (int j = 0; j < dims.last; j++)
			{
				const int elements_left = std::min(dims.last - j, SIMD<T>::length);
				SIMD<T> input(xMem + i * dims.last + j, elements_left);
				SIMD<T> weights(wMem + j, elements_left);
				SIMD<T> bias(bMem + j, elements_left);
				input = alpha * activation_forward(activation, weights * input + bias);

				if (beta != scalar::zero<U>())
				{
					SIMD<T> output(yMem + i * dims.last + j, elements_left);
					input += beta * output;
				}

				input.store(yMem + i * dims.last + j, elements_left);
			}
	}

	template<typename T>
	void prepare_scaling_factors(T *workspace, const T *scaleMem, const T *biasMem, const T *meanMem, const T *varianceMem, int last_dim, T epsilon)
	{
		for (int j = 0; j < last_dim; j += SIMD<T>::length)
		{
			const int elements_left = std::min(last_dim - j, SIMD<T>::length);
			SIMD<T> scale(scaleMem + j, elements_left);
			SIMD<T> shift(biasMem + j, elements_left);
			SIMD<T> mean(meanMem + j, elements_left);
			SIMD<T> variance(varianceMem + j, elements_left);

			scale = scale * rsqrt(epsilon + variance);
			shift = shift - mean * scale;

			scale.store(workspace + j, elements_left);
			shift.store(workspace + j + last_dim, elements_left);
		}
	}

	template<typename T>
	void kernel_batchnorm_inference(T alpha, T beta, const T *xMem, T *yMem, const T *scaleMem, const T *biasMem, const T *meanMem,
			const T *varianceMem, T epsilon, cpu::BroadcastedDimensions dims, avActivationType_t activation, T *workspace)
	{
		prepare_scaling_factors(workspace, scaleMem, biasMem, meanMem, varianceMem, dims.last, epsilon);
#pragma omp parallel for
		for (int i = 0; i < dims.first; i++)
			for (int j = 0; j < dims.last; j += SIMD<T>::length)
			{
				const int elements_left = std::min(dims.last - j, SIMD<T>::length);
				SIMD<T> input(xMem + i * dims.last + j, elements_left);
				SIMD<T> scale(workspace + j, elements_left);
				SIMD<T> shift(workspace + j + dims.last, elements_left);

				input = alpha * activation_forward(activation, scale * input + shift);

				if (beta != scalar::zero<T>())
				{
					SIMD<T> output(yMem + i * dims.last + j, elements_left);
					input += beta * output;
				}

				input.store(yMem + i * dims.last + j, elements_left);
			}
	}
	template<typename T>
	void kernel_batchnorm_forward(T alpha, T beta, const T *xMem, T *yMem, const T *scaleMem, const T *biasMem, T *meanMem, T *varianceMem, T epsilon,
			cpu::BroadcastedDimensions dims, avActivationType_t activation, T *workspace)
	{
#pragma omp parallel
		{
			T *thread_workspace = workspace + omp_get_thread_num() * dims.last;
#pragma omp single
			{
				clear_array(meanMem, dims.last);
				clear_array(varianceMem, dims.last);
			} // here is implicit synchronization barrier

			clear_array(thread_workspace, dims.last);
#pragma omp for nowait
			for (int i = 0; i < dims.first; i++)
				add_arrays(thread_workspace, xMem + i * dims.last, dims.last);
#pragma omp critical
			{
				add_arrays(meanMem, thread_workspace, dims.last);
			}
#pragma omp barrier

#pragma omp single
			{
				T m = scalar::one<T>() / static_cast<T>(dims.first);
				scale_array(meanMem, m, dims.last);
			} // here is implicit synchronization barrier

			clear_array(thread_workspace, dims.last);
#pragma omp for nowait
			for (int i = 0; i < dims.first; i++)
				for (int j = 0; j < dims.last; j += SIMD<T>::length)
				{
					const int elements_left = std::min(dims.last - j, SIMD<T>::length);
					SIMD<T> input(xMem + i * dims.last + j, elements_left);
					SIMD<T> mean(meanMem + j, elements_left);
					SIMD<T> variance(thread_workspace + j, elements_left);

					variance += square(input - mean);
					variance.store(thread_workspace + j, elements_left);
				}
#pragma omp critical
			{
				add_arrays(varianceMem, thread_workspace, dims.last);
			}
#pragma omp barrier

#pragma omp single
			{
				T m = scalar::one<T>() / static_cast<T>(dims.first);
				scale_array(varianceMem, m, dims.last);
				prepare_scaling_factors(workspace, scaleMem, biasMem, meanMem, varianceMem, dims.last, epsilon);
			} // here is implicit synchronization barrier

#pragma omp for nowait
			for (int i = 0; i < dims.first; i++)
				for (int j = 0; j < dims.last; j += SIMD<T>::length)
				{
					const int elements_left = std::min(dims.last - j, SIMD<T>::length);
					SIMD<T> input(xMem + i * dims.last + j, elements_left);
					SIMD<T> scale(workspace + j, elements_left);
					SIMD<T> shift(workspace + j + dims.last, elements_left);

					input = alpha * activation_forward(activation, scale * input + shift);

					if (beta != scalar::zero<T>())
					{
						SIMD<T> output(yMem + i * dims.last + j, elements_left);
						input += beta * output;
					}

					input.store(yMem + i * dims.last + j, elements_left);
				}
		}
	}
	template<typename T>
	void kernel_batchnorm_backward(T alpha1, T beta1, const T *xMem, const T *yMem, T *dxMem, T *dyMem, const T *scaleMem, const T *meanMem,
			const T *varianceMem, T epsilon, cpu::BroadcastedDimensions dims, T alpha2, T beta2, T *dwMem, T *dbMem, avActivationType_t activation,
			T *workspace)
	{
		T *d_sigma = workspace + 0 * 2 * dims.last;
		T *d_mu = workspace + (0 * 2 + 1) * dims.last;
		clear_array(d_sigma, dims.last);
		clear_array(d_mu, dims.last);
#pragma omp parallel
		{
			T *thread_d_sigma = workspace + (1 + omp_get_thread_num()) * 2 * dims.last;
			T *thread_d_mu = workspace + ((1 + omp_get_thread_num()) * 2 + 1) * dims.last;
			clear_array(thread_d_sigma, dims.last);
			clear_array(thread_d_mu, dims.last);

#pragma omp for nowait
			for (int i = 0; i < dims.first; i++)
				for (int j = 0; j < dims.last; j += SIMD<T>::length)
				{
					const int elements_left = std::min(dims.last - j, SIMD<T>::length);
					SIMD<T> gradient_next(dyMem + i * dims.last + j, elements_left);
					SIMD<T> output(yMem + i * dims.last + j, elements_left);

					gradient_next = activation_backward(activation, gradient_next, output);
					gradient_next.store(dyMem + i * dims.last + j, elements_left);

					SIMD<T> mean(meanMem + j, elements_left);
					SIMD<T> variance(varianceMem + j, elements_left);
					SIMD<T> input(xMem + i * dims.last + j, elements_left);

					input = (input - mean) * rsqrt(epsilon + variance);

					SIMD<T> tmp_d_sigma(thread_d_sigma + j, elements_left);
					SIMD<T> tmp_d_mu(thread_d_mu + j, elements_left);
					tmp_d_sigma += gradient_next * input;
					tmp_d_mu += gradient_next;
					tmp_d_sigma.store(thread_d_sigma + j, elements_left);
					tmp_d_mu.store(thread_d_mu + j, elements_left);
				}

#pragma omp critical
			{
				add_arrays(d_sigma, thread_d_sigma, dims.last);
				add_arrays(d_mu, thread_d_mu, dims.last);
			}
#pragma omp barrier
#pragma omp single
			{
				add_arrays(dwMem, d_sigma, alpha2, beta2, dims.last);
				add_arrays(dbMem, d_mu, alpha2, beta2, dims.last);
			} // here is implicit synchronization

			const SIMD<T> inv_m = SIMD<T>::one() / static_cast<T>(dims.first);
#pragma omp for
			for (int i = 0; i < dims.first; i++)
				for (int j = 0; j < dims.last; j += SIMD<T>::length)
				{
					const int elements_left = std::min(dims.last - j, SIMD<T>::length);
					SIMD<T> gradient_next(dyMem + i * dims.last + j, elements_left);
					SIMD<T> input(xMem + i * dims.last + j, elements_left);

					SIMD<T> mean(meanMem + j, elements_left);
					SIMD<T> variance(varianceMem + j, elements_left);
					SIMD<T> scale(scaleMem + j, elements_left);
					SIMD<T> tmp_d_sigma(d_sigma + j, elements_left);
					SIMD<T> tmp_d_mu(d_mu + j, elements_left);

					SIMD<T> inv_stddev = rsqrt(epsilon + variance);
					input = (input - mean) * inv_stddev;
					SIMD<T> tmp = alpha1 * scale * inv_stddev * (gradient_next - (tmp_d_sigma * input + tmp_d_mu) * inv_m);

					if (beta1 != scalar::zero<T>())
					{
						SIMD<T> gradient_prev(dxMem + i * dims.last + j, elements_left);
						tmp += beta1 * gradient_prev;
					}
					tmp.store(dxMem + i * dims.last + j, elements_left);
				}
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_affineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
			const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
			const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
			avMemoryDescriptor_t yMem)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(cpu::getTensor(yDesc), cpu::getTensor(xDesc));
		switch (cpu::getTensor(xDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				kernel_affine_forward<float16>(cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), cpu::getPointer<float16>(xMem),
						cpu::getPointer<float16>(yMem), cpu::getPointer<float16>(wMem), cpu::getPointer<float16>(bMem), dimensions, activation);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_affine_forward(cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), cpu::getPointer<bfloat16>(xMem),
						cpu::getPointer<bfloat16>(yMem), cpu::getPointer<bfloat16>(wMem), cpu::getPointer<bfloat16>(bMem), dimensions, activation);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_affine_forward(cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), cpu::getPointer<float>(xMem), cpu::getPointer<float>(yMem),
						cpu::getPointer<float>(wMem), cpu::getPointer<float>(bMem), dimensions, activation);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_affine_forward(cpu::getAlphaValue<double>(alpha), cpu::getBetaValue<double>(beta), cpu::getPointer<double>(xMem),
						cpu::getPointer<double>(yMem), cpu::getPointer<double>(wMem), cpu::getPointer<double>(bMem), dimensions, activation);
				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_batchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
			const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
			avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
			const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(cpu::getTensor(xDesc), cpu::getTensor(scaleBiasMeanVarDesc));

		const int required_workspace_size = 2 * dimensions.last * cpu::dataTypeSize(cpu::getTensor(xDesc).dtype());
		if (cpu::getContext(context).getWorkspace().size() < required_workspace_size)
			return AVOCADO_STATUS_INTERNAL_ERROR; // not enough workspace

		switch (cpu::getTensor(xDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_batchnorm_inference<float>(cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), cpu::getPointer<float>(xMem),
						cpu::getPointer<float>(yMem), cpu::getPointer<float>(scaleMem), cpu::getPointer<float>(biasMem),
						cpu::getPointer<float>(meanMem), cpu::getPointer<float>(varianceMem), epsilon, dimensions, activation,
						cpu::getContext(context).getWorkspace().data<float>());
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_batchnorm_inference<double>(cpu::getAlphaValue<double>(alpha), cpu::getBetaValue<double>(beta), cpu::getPointer<double>(xMem),
						cpu::getPointer<double>(yMem), cpu::getPointer<double>(scaleMem), cpu::getPointer<double>(biasMem),
						cpu::getPointer<double>(meanMem), cpu::getPointer<double>(varianceMem), epsilon, dimensions, activation,
						cpu::getContext(context).getWorkspace().data<double>());
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_batchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
			const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
			const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
			avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(cpu::getTensor(xDesc), cpu::getTensor(scaleBiasMeanVarDesc));

		const int required_workspace_size = std::min(2, cpuGetNumberOfThreads()) * dimensions.last * cpu::dataTypeSize(cpu::getTensor(xDesc).dtype());
		if (cpu::getContext(context).getWorkspace().size() < required_workspace_size)
			return AVOCADO_STATUS_INTERNAL_ERROR; // not enough workspace

		switch (cpu::getTensor(xDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_batchnorm_forward<float>(cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), cpu::getPointer<float>(xMem),
						cpu::getPointer<float>(yMem), cpu::getPointer<float>(scaleMem), cpu::getPointer<float>(biasMem),
						cpu::getPointer<float>(meanMem), cpu::getPointer<float>(varianceMem), epsilon, dimensions, activation,
						cpu::getContext(context).getWorkspace().data<float>());
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_batchnorm_forward<double>(cpu::getAlphaValue<double>(alpha), cpu::getBetaValue<double>(beta), cpu::getPointer<double>(xMem),
						cpu::getPointer<double>(yMem), cpu::getPointer<double>(scaleMem), cpu::getPointer<double>(biasMem),
						cpu::getPointer<double>(meanMem), cpu::getPointer<double>(varianceMem), epsilon, dimensions, activation,
						cpu::getContext(context).getWorkspace().data<double>());
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_batchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
			const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
			const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
			avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
			const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, const void *alpha2, const void *beta2,
			avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, double epsilon)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(cpu::getTensor(xDesc), cpu::getTensor(scaleMeanVarDesc));
		const int required_workspace_size = (1 + cpuGetNumberOfThreads()) * 2 * dimensions.last * cpu::dataTypeSize(cpu::getTensor(xDesc).dtype());
		if (cpu::getContext(context).getWorkspace().size() < required_workspace_size)
			return AVOCADO_STATUS_INTERNAL_ERROR; // not enough workspace

		switch (cpu::getTensor(xDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_batchnorm_backward<float>(cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), cpu::getPointer<float>(xMem),
						cpu::getPointer<float>(yMem), cpu::getPointer<float>(dxMem), cpu::getPointer<float>(dyMem), cpu::getPointer<float>(scaleMem),
						cpu::getPointer<float>(meanMem), cpu::getPointer<float>(varianceMem), epsilon, dimensions, cpu::getAlphaValue(alpha2),
						cpu::getBetaValue(beta2), cpu::getPointer<float>(scaleUpdateMem), cpu::getPointer<float>(biasUpdateMem), activation,
						cpu::getContext(context).getWorkspace().data<float>());
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_batchnorm_backward<double>(cpu::getAlphaValue<double>(alpha), cpu::getBetaValue<double>(beta), cpu::getPointer<double>(xMem),
						cpu::getPointer<double>(yMem), cpu::getPointer<double>(dxMem), cpu::getPointer<double>(dyMem),
						cpu::getPointer<double>(scaleMem), cpu::getPointer<double>(meanMem), cpu::getPointer<double>(varianceMem), epsilon,
						dimensions, cpu::getAlphaValue<double>(alpha2), cpu::getBetaValue<double>(beta2), cpu::getPointer<double>(scaleUpdateMem),
						cpu::getPointer<double>(biasUpdateMem), activation, cpu::getContext(context).getWorkspace().data<double>());
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

