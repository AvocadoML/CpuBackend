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
	using avocado::backend::float16;
	using avocado::backend::bfloat16;

	template<typename T>
	static inline SIMD<T> square(SIMD<T> x) noexcept
	{
		return x * x;
	}

	template<typename T>
	static inline SIMD<T> pow(SIMD<T> x, SIMD<T> y) noexcept
	{
		T tmp_x[SIMD<T>::length];
		T tmp_y[SIMD<T>::length];
		x.store(tmp_x);
		y.store(tmp_y);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp_x[i] = std::pow(tmp_x[i], tmp_y[i]);
		return SIMD<T>(tmp_x);
	}
	template<typename T>
	static inline SIMD<T> mod(SIMD<T> x, SIMD<T> y) noexcept
	{
		T tmp_x[SIMD<T>::length];
		T tmp_y[SIMD<T>::length];
		x.store(tmp_x);
		y.store(tmp_y);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp_x[i] = std::fmod(tmp_x[i], tmp_y[i]);
		return SIMD<T>(tmp_x);
	}
	template<typename T>
	static inline SIMD<T> exp(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::exp(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> log(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::log(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> tanh(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::tanh(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> expm1(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::expm1(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> log1p(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::log1p(tmp[i]);
		return SIMD<T>(tmp);
	}

	template<typename T>
	static inline SIMD<T> sin(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::sin(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> cos(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::cos(tmp[i]);
		return SIMD<T>(tmp);
	}
	template<typename T>
	static inline SIMD<T> tan(SIMD<T> x) noexcept
	{
		T tmp[SIMD<T>::length];
		x.store(tmp);
		for (int i = 0; i < SIMD<T>::length; i++)
			tmp[i] = std::tan(tmp[i]);
		return SIMD<T>(tmp);
	}

	static inline SIMD<float16> pow(SIMD<float16> x, SIMD<float16> y) noexcept
	{
		return pow(static_cast<SIMD<float>>(x), static_cast<SIMD<float>>(y));
	}
	static inline SIMD<float16> mod(SIMD<float16> x, SIMD<float16> y) noexcept
	{
		return mod(static_cast<SIMD<float>>(x), static_cast<SIMD<float>>(y));
	}
	static inline SIMD<float16> exp(SIMD<float16> x) noexcept
	{
		return exp(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> log(SIMD<float16> x) noexcept
	{
		return log(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> tanh(SIMD<float16> x) noexcept
	{
		return tanh(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> expm1(SIMD<float16> x) noexcept
	{
		return expm1(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> log1p(SIMD<float16> x) noexcept
	{
		return log1p(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> sin(SIMD<float16> x) noexcept
	{
		return sin(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> cos(SIMD<float16> x) noexcept
	{
		return cos(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> tan(SIMD<float16> x) noexcept
	{
		return tan(static_cast<SIMD<float>>(x));
	}

	static inline SIMD<bfloat16> pow(SIMD<bfloat16> x, SIMD<bfloat16> y) noexcept
	{
		return pow(static_cast<SIMD<float>>(x), static_cast<SIMD<float>>(y));
	}
	static inline SIMD<bfloat16> mod(SIMD<bfloat16> x, SIMD<bfloat16> y) noexcept
	{
		return mod(static_cast<SIMD<float>>(x), static_cast<SIMD<float>>(y));
	}
	static inline SIMD<bfloat16> exp(SIMD<bfloat16> x) noexcept
	{
		return exp(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> log(SIMD<bfloat16> x) noexcept
	{
		return log(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> tanh(SIMD<bfloat16> x) noexcept
	{
		return tanh(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> expm1(SIMD<bfloat16> x) noexcept
	{
		return expm1(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> log1p(SIMD<bfloat16> x) noexcept
	{
		return log1p(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> sin(SIMD<bfloat16> x) noexcept
	{
		return sin(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> cos(SIMD<bfloat16> x) noexcept
	{
		return cos(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> tan(SIMD<bfloat16> x) noexcept
	{
		return tan(static_cast<SIMD<float>>(x));
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_SIMD_FUNCTIONS_HPP_ */
