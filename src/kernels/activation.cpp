/*
 * activation.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

#include "activation.hpp"

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<class Act, typename T, typename U = T>
	void kernel_activation_forward(U alpha, const T *xMem, U beta, T *yMem, int elements) noexcept
	{
		Act activation;
		if (beta == scalar::zero<U>())
		{
#pragma omp parallel for
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> x(xMem + i, elements_left);

				x = alpha * activation.forward(x);
				x.store(yMem + i, elements_left);
			}
		}
		else
		{
#pragma omp parallel for
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> x(xMem + i, elements_left);
				SIMD<T> y(yMem + i, elements_left);

				x = alpha * activation.forward(x) + beta * y;
				x.store(yMem + i, elements_left);
			}
		}
	}
	template<class Act, typename T, typename U = T>
	void kernel_activation_backward(U alpha, T *dxMem, U beta, const T *dyMem, const T *yMem, int elements) noexcept
	{
		Act activation;
		if (beta == scalar::zero<U>())
		{
#pragma omp parallel for
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> dy(dyMem + i, elements_left);
				SIMD<T> y(yMem + i, elements_left);

				dy = alpha * activation.backward(dy, y);
				dy.store(dxMem + i, elements_left);
			}
		}
		else
		{
#pragma omp parallel for
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> dy(dyMem + i, elements_left);
				SIMD<T> y(yMem + i, elements_left);
				SIMD<T> dx(dxMem + i, elements_left);

				dy = alpha * activation.backward(dy, y) + beta * dx;
				dy.store(dxMem + i, elements_left);
			}
		}
	}

	template<typename T, typename U = T>
	avStatus_t launcher_activation_forward(avActivationType_t activation, U alpha, const T *xMem, U beta, T *yMem, int elements) noexcept
	{
		switch (activation)
		{
			case AVOCADO_ACTIVATION_LINEAR:
				kernel_activation_forward<ActivationLinear<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_activation_forward<ActivationSigmoid<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_activation_forward<ActivationTanh<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_activation_forward<ActivationRelu<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_activation_forward<ActivationSelu<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_activation_forward<ActivationElu<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_activation_forward<ActivationExponential<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_activation_forward<ActivationSoftplus<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_activation_forward<ActivationSoftsign<T>, T, U>(alpha, xMem, beta, yMem, elements);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	template<typename T, typename U = T>
	avStatus_t launcher_activation_backward(avActivationType_t activation, U alpha, T *dxMem, U beta, const T *dyMem, const T *yMem,
			int elements) noexcept
	{
		switch (activation)
		{
			case AVOCADO_ACTIVATION_LINEAR:
				kernel_activation_backward<ActivationLinear<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_activation_backward<ActivationSigmoid<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_activation_backward<ActivationTanh<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_activation_backward<ActivationRelu<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_activation_backward<ActivationSelu<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_activation_backward<ActivationElu<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_activation_backward<ActivationExponential<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_activation_backward<ActivationSoftplus<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_activation_backward<ActivationSoftsign<T>, T, U>(alpha, dxMem, beta, dyMem, yMem, elements);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_activationForward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem)
	{
		const int elements = xDesc.volume();
		switch (xDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				return launcher_activation_forward<float16, float>(activation, getAlphaValue(alpha), xMem.data<float16>(), getBetaValue(beta),
						yMem.data<float16>(), elements);
			case AVOCADO_DTYPE_BFLOAT16:
				return launcher_activation_forward<bfloat16, float>(activation, getAlphaValue(alpha), xMem.data<bfloat16>(), getBetaValue(beta),
						yMem.data<bfloat16>(), elements);
			case AVOCADO_DTYPE_FLOAT32:
				return launcher_activation_forward<float>(activation, getAlphaValue(alpha), xMem.data<float>(), getBetaValue(beta),
						yMem.data<float>(), elements);
			case AVOCADO_DTYPE_FLOAT64:
				return launcher_activation_forward<double>(activation, getAlphaValue<double>(alpha), xMem.data<double>(), getBetaValue<double>(beta),
						yMem.data<double>(), elements);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
	}

	avStatus_t cpu_activationBackward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
			const TensorDescriptor &yDesc, const MemoryDescriptor &yMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
			const void *beta, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem)
	{
		const int elements = yDesc.volume();
		switch (yDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
				return launcher_activation_backward<float>(activation, getAlphaValue(alpha), dxMem.data<float>(), getBetaValue(beta),
						dyMem.data<float>(), yMem.data<float>(), elements);
			case AVOCADO_DTYPE_FLOAT64:
				return launcher_activation_backward<double>(activation, getAlphaValue<double>(alpha), dxMem.data<double>(),
						getBetaValue<double>(beta), dyMem.data<double>(), yMem.data<double>(), elements);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
	}

} /* namespace SIMD_NAMESPACE */

