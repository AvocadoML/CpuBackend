/*
 * activation.hpp
 *
 *  Created on: Jan 4, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef KERNELS_ACTIVATION_HPP_
#define KERNELS_ACTIVATION_HPP_

#include "../vectors/simd_vectors.hpp"

namespace SIMD_NAMESPACE
{
	template<typename T>
	struct ActivationLinear
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return input;
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient;
			}
	};
	template<typename T>
	struct ActivationSigmoid
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return SIMD<T>::one() / (SIMD<T>::one() + exp(-input));
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient * (SIMD<T>::one() - output) * output;
			}
	};
	template<typename T>
	struct ActivationTanh
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return tanh(input);
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient * (SIMD<T>::one() - output) * (SIMD<T>::one() + output);;
			}
	};
	template<typename T>
	struct ActivationRelu
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return max(SIMD<T>::zero(), input);
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return select(output > SIMD<T>::zero(), gradient, SIMD<T>::zero());
			}
	};
	template<typename T>
	struct ActivationSelu
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return SIMD<T>(1.05070098f) * select(input >= SIMD<T>::zero(), input, SIMD<T>(1.67326324f) * expm1(input));
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return SIMD<T>(1.05070098f) * gradient
						* select(output >= SIMD<T>::zero(), SIMD<T>::one(), SIMD<T>(1.67326324f) * (output + SIMD<T>::one()));
			}
	};
	template<typename T>
	struct ActivationElu
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return select(input >= SIMD<T>::zero(), input, expm1(input));
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient * select(output >= SIMD<T>::zero(), SIMD<T>::one(), (output + SIMD<T>::one()));
			}
	};
	template<typename T>
	struct ActivationExponential
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return exp(input);
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient * output;
			}
	};
	template<typename T>
	struct ActivationSoftplus
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return log1p(exp(input));
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient * expm1(output) * exp(-output);
			}
	};
	template<typename T>
	struct ActivationSoftsign
	{
			SIMD<T> forward(SIMD<T> input) const noexcept
			{
				return input / (abs(input) + SIMD<T>::one());
			}
			SIMD<T> backward(SIMD<T> gradient, SIMD<T> output) const noexcept
			{
				return gradient / square(abs(output / (SIMD<T>::one() - sgn(output) * output)) + SIMD<T>::one());
			}
	};

	namespace internal
	{
		template<class Activation, typename T>
		void kernel_forward(T *input, const int elements) noexcept
		{
			Activation activation;
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> x(input + i, elements_left);
				x = activation.forward(x);
				x.storeu(input + i, elements_left);
			}
		}
	} /* namespace internal */

	template<typename T>
	void activation_forward(avocado::backend::avActivationType_t activation, T *input, const int elements)
	{
		switch (activation)
		{
			case avocado::backend::AVOCADO_ACTIVATION_LINEAR:
				break;
			case avocado::backend::AVOCADO_ACTIVATION_SIGMOID:
				internal::kernel_forward<ActivationSigmoid<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_TANH:
				internal::kernel_forward<ActivationTanh<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_RELU:
				internal::kernel_forward<ActivationRelu<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_SELU:
				internal::kernel_forward<ActivationSelu<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_ELU:
				internal::kernel_forward<ActivationElu<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_EXPONENTIAL:
				internal::kernel_forward<ActivationExponential<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_SOFTPLUS:
				internal::kernel_forward<ActivationSoftplus<T>, T>(input, elements);
				break;
			case avocado::backend::AVOCADO_ACTIVATION_SOFTSIGN:
				internal::kernel_forward<ActivationSoftsign<T>, T>(input, elements);
				break;
		}
	}
}

#endif /* KERNELS_ACTIVATION_HPP_ */
