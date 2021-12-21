/*
 * simd_functions.hpp
 *
 *  Created on: Dec 19, 2021
 *      Author: maciek
 */

#ifndef VECTORS_SIMD_FUNCTIONS_HPP_
#define VECTORS_SIMD_FUNCTIONS_HPP_

#include "generic_simd.hpp"

namespace SIMD_NAMESPACE
{
	template<typename T>
	static inline SIMD<T> square(SIMD<T> x) noexcept
	{
		return x * x;
	}

	template<typename T>
	static inline SIMD<T> mod(SIMD<T> x, SIMD<T> y) noexcept
	{
		T tmp_x[SIMD<T>::length];
		T tmp_y[SIMD<T>::length];
		x.storeu(tmp_x);
		y.storeu(tmp_y);
		for (size_t i = 0; i < SIMD<T>::length; i++)
			tmp_x[i] = std::modf(tmp_x[i], tmp_y[i]);
		return SIMD<T>(tmp_x);
	}
	template<typename T>
	static inline SIMD<T> exp(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (size_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::exp(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> log(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (size_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::log(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> tanh(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (size_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::tanh(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> expm1(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (size_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::expm1(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> log1p(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (size_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::log1p(tmp[i]);
		return SIMD<T>(tmp);
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_SIMD_FUNCTIONS_HPP_ */
