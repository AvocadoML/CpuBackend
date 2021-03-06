/*
 * losses.cpp
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
	struct LossMSE
	{
			SIMD<T> loss(SIMD<T> output, SIMD<T> target) const noexcept
			{
				return static_cast<T>(0.5) * square(output - target);
			}
			SIMD<T> gradient(SIMD<T> output, SIMD<T> target) const noexcept
			{
				return output - target;
			}
	};
	template<typename T, bool Fused = false>
	struct LossCE
	{
			SIMD<T> loss(SIMD<T> output, SIMD<T> target) const noexcept
			{
				return -target * log(SIMD<T>::epsilon() + output) - (SIMD<T>::one() - target) * log(SIMD<T>::epsilon() + SIMD<T>::one() - output);
			}
			SIMD<T> gradient(SIMD<T> output, SIMD<T> target) const noexcept
			{
				if (Fused)
					return output - target;
				else
					return (output - target) / (SIMD<T>::epsilon() + output * (SIMD<T>::one() - output));
			}
	};
	template<typename T, bool Fused = false>
	struct LossKLD
	{
			SIMD<T> loss(SIMD<T> output, SIMD<T> target) const noexcept
			{
				LossCE<T, Fused> tmp;
				return tmp.loss(output, target) - tmp.loss(target, target);
			}
			SIMD<T> gradient(SIMD<T> output, SIMD<T> target) const noexcept
			{
				return LossCE<T, Fused>().gradient(output, target);
			}
	};

	template<class LossFunction, typename T>
	T kernel_loss(const T *outputMem, const T *targetMem, int elements) noexcept
	{
		assert(outputMem != nullptr);
		assert(targetMem != nullptr);
		LossFunction loss_function;
		SIMD<T> result = SIMD<T>::zero();
#pragma omp parallel
		{
			SIMD<T> thread_acc = SIMD<T>::zero();
#pragma omp for nowait
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> output(outputMem + i, elements_left);
				SIMD<T> target(targetMem + i, elements_left);
				thread_acc += loss_function.loss(output, target);
			}
#pragma omp critical
			{
				result += thread_acc;
			}
		}
		return horizontal_add(result);
	}
	template<class LossFunction, typename T>
	void kernel_gradient(T *gradientMem, const T *outputMem, const T *targetMem, int elements, T alpha, T beta) noexcept
	{
		assert(gradientMem != nullptr);
		assert(targetMem != nullptr);
		assert(outputMem != nullptr);
		LossFunction loss_function;
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> output(outputMem + i, elements_left);
			SIMD<T> target(targetMem + i, elements_left);
			SIMD<T> gradient = alpha * loss_function.gradient(output, target);
			if (beta != scalar::zero<T>())
			{
				SIMD<T> dst(gradientMem + i, elements_left);
				gradient += beta * dst;
			}
			gradient.store(gradientMem + i, elements_left);
		}
	}

	template<typename T>
	T launcher_loss(avLossType_t lossType, const T *output, const T *target, int elements) noexcept
	{
		switch (lossType)
		{
			case AVOCADO_MEAN_SQUARE_LOSS:
				return kernel_loss<LossMSE<T>, T>(output, target, elements);
			case AVOCADO_CROSS_ENTROPY_LOSS:
				return kernel_loss<LossCE<T>, T>(output, target, elements);
			case AVOCADO_KL_DIVERGENCE_LOSS:
				return kernel_loss<LossKLD<T>, T>(output, target, elements);
			default:
				return scalar::zero<T>();
		}
	}
	template<typename T>
	void launcher_gradient(avLossType_t lossType, T *gradient, const T *output, const T *target, int elements, T alpha, T beta, bool fused) noexcept
	{
		switch (lossType)
		{
			case AVOCADO_MEAN_SQUARE_LOSS:
				kernel_gradient<LossMSE<T>, T>(gradient, output, target, elements, alpha, beta);
				break;
			case AVOCADO_CROSS_ENTROPY_LOSS:
			{
				if (fused)
					kernel_gradient<LossCE<T, true>, T>(gradient, output, target, elements, alpha, beta);
				else
					kernel_gradient<LossCE<T, false>, T>(gradient, output, target, elements, alpha, beta);
				break;
			}
			case AVOCADO_KL_DIVERGENCE_LOSS:
			{
				if (fused)
					kernel_gradient<LossKLD<T, true>, T>(gradient, output, target, elements, alpha, beta);
				else
					kernel_gradient<LossKLD<T, false>, T>(gradient, output, target, elements, alpha, beta);
				break;
			}
		}
	}

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_lossFunction(const ContextDescriptor &context, avLossType_t lossType, const TensorDescriptor &outputDesc,
			const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, void *result)
	{
		const int elements = outputDesc.volume();
		switch (outputDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float loss = launcher_loss(lossType, outputMem.data<float>(), targetMem.data<float>(), elements);
				std::memcpy(result, &loss, sizeof(float));
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double loss = launcher_loss(lossType, outputMem.data<double>(), targetMem.data<double>(), elements);
				std::memcpy(result, &loss, sizeof(double));
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	avStatus_t cpu_lossGradient(const ContextDescriptor &context, avLossType_t lossType, const void *alpha, const TensorDescriptor &outputDesc,
			const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, const void *beta,
			const TensorDescriptor &gradientDesc, MemoryDescriptor &gradientMem, bool isFused)
	{
		const int elements = outputDesc.volume();
		switch (outputDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				launcher_gradient(lossType, gradientMem.data<float>(), outputMem.data<float>(), targetMem.data<float>(), elements,
						getAlphaValue(alpha), getBetaValue(beta), isFused);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				launcher_gradient(lossType, gradientMem.data<double>(), outputMem.data<double>(), targetMem.data<double>(), elements,
						getAlphaValue<double>(alpha), getBetaValue<double>(beta), isFused);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

