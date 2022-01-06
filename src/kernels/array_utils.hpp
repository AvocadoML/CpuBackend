/*
 * array_utils.hpp
 *
 *  Created on: Jan 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef KERNELS_ARRAY_UTILS_HPP_
#define KERNELS_ARRAY_UTILS_HPP_

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <cstring>

namespace SIMD_NAMESPACE
{
	template<typename T>
	void clear_array(T *ptr, int elements) noexcept
	{
		std::memset(ptr, 0, sizeof(T) * elements);
	}
	template<typename T>
	void set_array(T *ptr, T value, int elements) noexcept
	{
		for (int i = 0; i < elements; i++)
			ptr[i] = value;
	}
	template<typename T>
	void set_array(T *ptr, SIMD<T> value, int elements) noexcept
	{
		for (int i = 0; i < elements; i += value.length)
		{
			const int elements_left = std::min(elements - i, value.length);
			value.storeu(ptr + i, elements_left);
		}
	}

	/*
	 * \brief Calculates dst[i] += src[i]
	 */
	template<typename T>
	void add_arrays(T *dst, const T *src, int elements) noexcept
	{
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> loaded_lhs(dst + i, elements_left);
			SIMD<T> loaded_rhs(src + i, elements_left);

			loaded_lhs += loaded_rhs;
			loaded_lhs.storeu(dst + i, elements_left);
		}
	}
	/*
	 * \brief Calculates dst[i] = alpha * src[i] + beta * dst[i]
	 */
	template<typename T>
	void add_arrays(T *dst, const T *src, T alpha, T beta, int elements) noexcept
	{
		if (beta == scalar::zero<T>())
		{
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> loaded(src + i, elements_left);
				loaded *= alpha;
				loaded.storeu(dst + i, elements_left);
			}
		}
		else
		{
			for (int i = 0; i < elements; i += SIMD<T>::length)
			{
				const int elements_left = std::min(elements - i, SIMD<T>::length);
				SIMD<T> loaded_src(src + i, elements_left);
				SIMD<T> loaded_dst(dst + i, elements_left);
				loaded_src = alpha * loaded_src + beta * loaded_dst;
				loaded_src.storeu(dst + i, elements_left);
			}
		}
	}
	/*
	 * \brief Calculates ptr[i] += value
	 */
	template<typename T>
	void scale_array(T *ptr, T value, int elements) noexcept
	{
		for (int i = 0; i < elements; i++)
			ptr[i] *= value;
	}
	/*
	 * \brief Calculates ptr[i] += value
	 */
	template<typename T>
	void scale_array(T *ptr, SIMD<T> value, int elements) noexcept
	{
		for (int i = 0; i < elements; i += value.length)
		{
			const int elements_left = std::min(elements - i, value.length);
			SIMD<T> loaded(ptr + i, elements_left);
			loaded *= value;
			loaded.storeu(ptr + i, elements_left);
		}
	}
}

#endif /* KERNELS_ARRAY_UTILS_HPP_ */
