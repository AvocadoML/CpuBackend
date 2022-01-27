/*
 * regularizers.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	void kernel_regularizer_l2(T *gradient, const T *param, T coefficient, T offset, int elements)
	{
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

	avStatus_t cpu_regularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem,
			const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const void *coefficient, const void *offset, void *loss)
	{
		const int elements = cpu::getTensor(dwDesc).volume();
		switch (cpu::getTensor(dwDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_regularizer_l2(cpu::getPointer<float>(dwMem), cpu::getPointer<float>(wMem), cpu::getScalarValue<float>(coefficient),
						cpu::getScalarValue<float>(offset), elements);
				if (loss != nullptr)
				{
					float l2_loss = kernel_loss_l2(cpu::getPointer<float>(wMem), cpu::getScalarValue<float>(coefficient),
							cpu::getScalarValue<float>(offset), elements);
					cpu::setScalarValue(loss, l2_loss);
				}
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_regularizer_l2(cpu::getPointer<double>(dwMem), cpu::getPointer<double>(wMem), cpu::getScalarValue<double>(coefficient),
						cpu::getScalarValue<double>(offset), elements);
				if (loss != nullptr)
				{
					double l2_loss = kernel_loss_l2(cpu::getPointer<double>(wMem), cpu::getScalarValue<double>(coefficient),
							cpu::getScalarValue<double>(offset), elements);
					std::cout << "loss = " << l2_loss << '\n';
					cpu::setScalarValue(loss, l2_loss);
				}
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

