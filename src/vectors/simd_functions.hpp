/*
 * simd_functions.hpp
 *
 *  Created on: Dec 19, 2021
 *      Author: maciek
 */

#ifndef VECTORS_SIMD_FUNCTIONS_HPP_
#define VECTORS_SIMD_FUNCTIONS_HPP_

#include "generic_simd.hpp"
#include "fp16_simd.hpp"
#include "bf16_simd.hpp"
#include "fp32_simd.hpp"

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
		for (int64_t i = 1; i < SIMD<T>::length; i++)
			result = std::max(result, tmp[i]);
		return result;
	}
	template<typename T>
	static inline T horizontal_min(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		T result = tmp[0];
		for (int64_t i = 1; i < SIMD<T>::length; i++)
			result = std::min(result, tmp[i]);
		return result;
	}

	template<typename T>
	static inline SIMD<T> pow(SIMD<T> x, SIMD<T> y) noexcept
	{
		T tmp_x[SIMD<T>::length];
		T tmp_y[SIMD<T>::length];
		x.storeu(tmp_x);
		y.storeu(tmp_y);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp_x[i] = std::pow(tmp_x[i], tmp_y[i]);
		return SIMD<T>(tmp_x);
	}
	template<typename T>
	static inline SIMD<T> mod(SIMD<T> x, SIMD<T> y) noexcept
	{
		T tmp_x[SIMD<T>::length];
		T tmp_y[SIMD<T>::length];
		x.storeu(tmp_x);
		y.storeu(tmp_y);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp_x[i] = std::fmod(tmp_x[i], tmp_y[i]);
		return SIMD<T>(tmp_x);
	}
	template<typename T>
	static inline SIMD<T> exp(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::exp(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> log(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::log(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> tanh(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::tanh(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> expm1(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::expm1(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> log1p(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::log1p(tmp[i]);
		return SIMD<T>(tmp);
	}

	template<typename T>
	static inline SIMD<T> sin(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::sin(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> cos(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::cos(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> tan(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.storeu(tmp);
		for (int64_t i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::tan(tmp[i]);
		return SIMD<T>(tmp);
	}

	static inline float horizontal_max(SIMD<avocado::backend::float16> x) noexcept
	{
		return horizontal_max(static_cast<SIMD<float>>(x));
	}
	static inline float horizontal_min(SIMD<avocado::backend::float16> x) noexcept
	{
		return horizontal_min(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> mod(SIMD<avocado::backend::float16> x, SIMD<avocado::backend::float16> y) noexcept
	{
		return SIMD<avocado::backend::float16>();
	}
	static inline SIMD<avocado::backend::float16> exp(SIMD<avocado::backend::float16> x) noexcept
	{
		return exp(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> log(SIMD<avocado::backend::float16> x) noexcept
	{
		return log(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> tanh(SIMD<avocado::backend::float16> x) noexcept
	{
		return tanh(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> expm1(SIMD<avocado::backend::float16> x) noexcept
	{
		return expm1(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> log1p(SIMD<avocado::backend::float16> x) noexcept
	{
		return log1p(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> sin(SIMD<avocado::backend::float16> x) noexcept
	{
		return sin(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> cos(SIMD<avocado::backend::float16> x) noexcept
	{
		return cos(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::float16> tan(SIMD<avocado::backend::float16> x) noexcept
	{
		return tan(static_cast<SIMD<float>>(x));
	}

	static inline float horizontal_max(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return horizontal_max(static_cast<SIMD<float>>(x));
	}
	static inline float horizontal_min(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return horizontal_min(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> mod(SIMD<avocado::backend::bfloat16> x, SIMD<avocado::backend::bfloat16> y) noexcept
	{
		return SIMD<avocado::backend::bfloat16>();
	}
	static inline SIMD<avocado::backend::bfloat16> exp(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return exp(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> log(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return log(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> tanh(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return tanh(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> expm1(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return expm1(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> log1p(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return log1p(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> sin(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return sin(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> cos(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return cos(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<avocado::backend::bfloat16> tan(SIMD<avocado::backend::bfloat16> x) noexcept
	{
		return tan(static_cast<SIMD<float>>(x));
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_SIMD_FUNCTIONS_HPP_ */
