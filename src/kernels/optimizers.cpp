/*
 * optimizers.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>
#include "array_utils.hpp"

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"
#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	SIMD<T> round_small_to_zero(SIMD<T> x) noexcept
	{
		return select(abs(x) < SIMD<T>::epsilon(), SIMD<T>::zero(), x);
	}

	template<typename T, bool UseMomentum, bool UseNesterov>
	void kernel_learn_sgd(T *wMem, const T *dwMem, T *momentumMem, int elements, T learning_rate, T beta1, T alpha, T beta)
	{
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> update(dwMem + i, elements_left);

			SIMD<T> result;
			if constexpr (UseMomentum)
			{
				SIMD<T> momentum(momentumMem + i, elements_left);
				momentum = beta1 * momentum - learning_rate * update;
				momentum.store(momentumMem + i, elements_left);
				if constexpr (UseNesterov)
					result = beta1 * momentum - learning_rate * update;
				else
					result = momentum;
			}
			else
				result = -learning_rate * update;

			result *= alpha;
			if (beta != scalar::zero<T>())
			{
				SIMD<T> dst(wMem + i, elements_left);
				result += beta * dst;
			}
			result = round_small_to_zero(result);
			result.store(wMem + i, elements_left);
		}
	}
	template<typename T>
	void kernel_learn_adam(T *wMem, const T *dwMem, T *momentumMem, T *varianceMem, int elements, T learning_rate, T beta1, T beta2, T alpha, T beta)
	{
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> update(dwMem + i, elements_left);
			SIMD<T> momentum(momentumMem + i, elements_left);
			SIMD<T> variance(varianceMem + i, elements_left);

			momentum = momentum * beta1 + update * (SIMD<T>::one() - beta1);
			variance = variance * beta2 + square(update) * (SIMD<T>::one() - beta2);
			momentum.store(momentumMem + i, elements_left);
			variance.store(varianceMem + i, elements_left);
			SIMD<T> result = -momentum * learning_rate * rsqrt(variance + SIMD<T>::epsilon());

			result *= alpha;
			if (beta != scalar::zero<T>())
			{
				SIMD<T> dst(wMem + i, elements_left);
				result += beta * dst;
			}
			result = round_small_to_zero(result);
			result.store(wMem + i, elements_left);
		}
	}

	template<typename T>
	avStatus_t launcher_optimizer(const cpu::OptimizerDescriptor &optimizer, const cpu::TensorDescriptor &wDesc, T *weight, const T *update,
			cpu::MemoryDescriptor &workspace, T alpha, T beta)
	{
		const int elements = wDesc.volume();
		switch (optimizer.type)
		{
			case AVOCADO_OPTIMIZER_SGD:
			{
				bool use_momentum = optimizer.flags[0];
				if (use_momentum and workspace.size() < elements * cpu::dataTypeSize(wDesc.dtype()))
					return AVOCADO_STATUS_INTERNAL_ERROR;

				bool use_nesterov = optimizer.flags[1];
				T beta1 = optimizer.coef[0];
				T learning_rate = optimizer.learning_rate;
				T *momentum = workspace.data<T>();
				if (use_momentum)
				{
					if (use_nesterov)
						kernel_learn_sgd<T, true, true>(weight, update, momentum, elements, learning_rate, beta1, alpha, beta);
					else
						kernel_learn_sgd<T, true, false>(weight, update, momentum, elements, learning_rate, beta1, alpha, beta);
				}
				else
					kernel_learn_sgd<T, false, false>(weight, update, momentum, elements, learning_rate, beta1, alpha, beta);
				return AVOCADO_STATUS_SUCCESS;
			}
			case AVOCADO_OPTIMIZER_ADAM:
			{
				if (workspace.size() < 2 * elements * cpu::dataTypeSize(wDesc.dtype()))
					return AVOCADO_STATUS_INTERNAL_ERROR;
				T beta1 = optimizer.coef[0];
				T beta2 = optimizer.coef[1];
				T learning_rate = optimizer.learning_rate;
				T *momentum = workspace.data<T>();
				T *variance = workspace.data<T>() + elements;
				kernel_learn_adam(weight, update, momentum, variance, elements, learning_rate, beta1, beta2, alpha, beta);
				return AVOCADO_STATUS_SUCCESS;
			}
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
	}

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_optimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const void *alpha,
			const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem, const void *beta, const avTensorDescriptor_t wDesc,
			avMemoryDescriptor_t wMem, avMemoryDescriptor_t workspace)
	{
		switch (cpu::getTensor(wDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
				return launcher_optimizer(cpu::getOptimizer(config), cpu::getTensor(wDesc), cpu::getPointer<float>(wMem),
						cpu::getPointer<float>(dwMem), cpu::getMemory(workspace), cpu::getAlphaValue(alpha), cpu::getBetaValue(beta));
			case AVOCADO_DTYPE_FLOAT64:
				return launcher_optimizer(cpu::getOptimizer(config), cpu::getTensor(wDesc), cpu::getPointer<double>(wMem),
						cpu::getPointer<double>(dwMem), cpu::getMemory(workspace), cpu::getAlphaValue<double>(alpha), cpu::getBetaValue<double>(beta));
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

