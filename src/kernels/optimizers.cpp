/*
 * optimizers.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>
#include "array_utils.hpp"

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"
#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	SIMD<T> round_small_to_zero(SIMD<T> x) noexcept
	{
		return select(abs(x) < SIMD<T>::epsilon(), SIMD<T>::zero(), x);
	}

	template<typename T, bool UseMomentum, bool UseNesterov>
	void kernel_learn_sgd(T *wMem, const T *dwMem, T *momentumMem, int elements, T learningRate, T beta1, T alpha, T beta)
	{
		assert(wMem != nullptr);
		assert(dwMem != nullptr);
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> update(dwMem + i, elements_left);

			SIMD<T> result;
			if constexpr (UseMomentum)
			{
				SIMD<T> momentum(momentumMem + i, elements_left);
				momentum = beta1 * momentum - learningRate * update;
				momentum.store(momentumMem + i, elements_left);
				if constexpr (UseNesterov)
					result = beta1 * momentum - learningRate * update;
				else
					result = momentum;
			}
			else
				result = -learningRate * update;

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
	void kernel_learn_adam(T *wMem, const T *dwMem, T *momentumMem, T *varianceMem, int elements, T learningRate, T beta1, T beta2, T alpha, T beta)
	{
		assert(wMem != nullptr);
		assert(dwMem != nullptr);
		assert(momentumMem != nullptr);
		assert(varianceMem != nullptr);
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
			SIMD<T> result = -momentum * learningRate * rsqrt(variance + SIMD<T>::epsilon());

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
	avStatus_t launcher_optimizer(OptimizerDescriptor &optimizer, const TensorDescriptor &wDesc, T *weight, const T *update,
			MemoryDescriptor &workspace, T alpha, T beta)
	{
		const int elements = wDesc.volume();
		switch (optimizer.type)
		{
			case AVOCADO_OPTIMIZER_SGD:
			{
				bool use_momentum = optimizer.flags[0];
				if (use_momentum and workspace.sizeInBytes() < elements * dataTypeSize(wDesc.dtype()))
					return AVOCADO_STATUS_INTERNAL_ERROR;

				optimizer.steps++;
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
				if (workspace.sizeInBytes() < 2 * elements * dataTypeSize(wDesc.dtype()))
					return AVOCADO_STATUS_INTERNAL_ERROR;
				optimizer.steps++;
				T beta1 = optimizer.coef[0];
				T beta2 = optimizer.coef[1];
				T learning_rate = optimizer.learning_rate;
				if (optimizer.steps < 10000)
					learning_rate *= std::sqrt(1.0 - std::pow(beta2, optimizer.steps)) / (1.0 - std::pow(beta1, optimizer.steps));
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
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_optimizerLearn(const ContextDescriptor &context, OptimizerDescriptor &config, const void *alpha, const TensorDescriptor &dwDesc,
			const MemoryDescriptor &dwMem, const void *beta, const TensorDescriptor &wDesc, MemoryDescriptor &wMem, MemoryDescriptor &workspace)
	{
		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
				return launcher_optimizer(config, wDesc, wMem.data<float>(), dwMem.data<float>(), workspace, getAlphaValue(alpha), getBetaValue(beta));
			case AVOCADO_DTYPE_FLOAT64:
				return launcher_optimizer(config, wDesc, wMem.data<double>(), dwMem.data<double>(), workspace, getAlphaValue<double>(alpha),
						getBetaValue<double>(beta));
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

