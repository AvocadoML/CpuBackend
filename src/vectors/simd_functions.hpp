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

	static inline SIMD<float> to_float(SIMD<int32_t> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cvtepi32_ps(x);
#elif SUPPORTS_SSE2
		return _mm_cvtepi32_ps(x);
#else
		return SIMD<float>(static_cast<float>(static_cast<int32_t>(x)));
#endif
	}
	template<typename T>
	static inline T horizontal_max(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		T result = tmp[0];
		for (size_t i = 1; i < SIMD<T>::length; i++)
			result = std::max(result, tmp[i]);
		return result;
	}
	template<typename T>
	static inline T horizontal_min(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		T result = tmp[0];
		for (size_t i = 1; i < SIMD<T>::length; i++)
			result = std::min(result, tmp[i]);
		return result;
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
