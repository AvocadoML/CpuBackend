/*
 * regularizers.cpp
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
	void kernel_regularizer_l2(T *gradient, const T *param, T coefficient, T offset, int elements)
	{
		assert(gradient != nullptr);
		assert(param != nullptr);
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> grad(gradient + i, elements_left);
			SIMD<T> w(param + i, elements_left);

			grad += coefficient * (w - offset);
			grad.store(gradient + i, elements_left);
		}
	}
	template<typename T>
	T kernel_loss_l2(const T *param, T coefficient, T offset, int elements)
	{
		assert(param != nullptr);
		SIMD<T> result = SIMD<T>::zero();
#pragma omp parallel
		{
			SIMD<T> tmp = SIMD<T>::zero();
#pragma omp for
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> w(param + i, elements_left);
				w -= offset;
				w.cutoff(elements_left);
				tmp += square(w);
			}
#pragma omp critical
			{
				result += tmp;
			}
		}
		return static_cast<T>(0.5) * coefficient * horizontal_add(result);
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_regularizerL2(const ContextDescriptor &context, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem,
			const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *coefficient, const void *offset, void *loss)
	{
		const int elements = dwDesc.volume();
		switch (dwDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_regularizer_l2(dwMem.data<float>(), wMem.data<float>(), getScalarValue<float>(coefficient), getScalarValue<float>(offset),
						elements);
				if (loss != nullptr)
				{
					float l2_loss = kernel_loss_l2(wMem.data<float>(), getScalarValue<float>(coefficient), getScalarValue<float>(offset), elements);
					setScalarValue(loss, l2_loss);
				}
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_regularizer_l2(dwMem.data<double>(), wMem.data<double>(), getScalarValue<double>(coefficient), getScalarValue<double>(offset),
						elements);
				if (loss != nullptr)
				{
					double l2_loss = kernel_loss_l2(wMem.data<double>(), getScalarValue<double>(coefficient), getScalarValue<double>(offset),
							elements);
					setScalarValue(loss, l2_loss);
				}
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

