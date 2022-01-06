/*
 * activation.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cpu_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

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
	avStatus_t launcher_activation_backward(avActivationType_t activation, U alpha, T *dxMem, U beta, const T *dyMem, const T *yMem, int elements)
	noexcept
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

namespace avocado
{
	namespace backend
	{
		avStatus_t activationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			const int elements = getTensor(xDesc).volume();
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					return launcher_activation_forward<float16, float>(activation, getAlphaValue(alpha), getPointer<float16>(xMem),
							getBetaValue(beta), getPointer<float16>(yMem), elements);
				case AVOCADO_DTYPE_BFLOAT16:
					return launcher_activation_forward<bfloat16, float>(activation, getAlphaValue(alpha), getPointer<bfloat16>(xMem),
							getBetaValue(beta), getPointer<bfloat16>(yMem), elements);
				case AVOCADO_DTYPE_FLOAT32:
					return launcher_activation_forward<float>(activation, getAlphaValue(alpha), getPointer<float>(xMem), getBetaValue(beta),
							getPointer<float>(yMem), elements);
				case AVOCADO_DTYPE_FLOAT64:
					return launcher_activation_forward<double>(activation, getAlphaValue<double>(alpha), getPointer<double>(xMem),
							getBetaValue<double>(beta), getPointer<double>(yMem), elements);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

		avStatus_t activationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const int elements = getTensor(yDesc).volume();
			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					return launcher_activation_backward<float>(activation, getAlphaValue(alpha), getPointer<float>(dxMem), getBetaValue(beta),
							getPointer<float>(dyMem), getPointer<float>(yMem), elements);
				case AVOCADO_DTYPE_FLOAT64:
					return launcher_activation_backward<double>(activation, getAlphaValue<double>(alpha), getPointer<double>(dxMem),
							getBetaValue<double>(beta), getPointer<double>(dyMem), getPointer<double>(yMem), elements);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

	} /* namespace backend */
} /* namespace avocado */

