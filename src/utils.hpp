/*
 * utils.hpp
 *
 *  Created on: Jul 29, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cmath>
#include <limits>

namespace scalar
{
	template<typename T>
	constexpr T zero() noexcept
	{
		return static_cast<T>(0);
	}
	template<typename T>
	constexpr T one() noexcept
	{
		return static_cast<T>(1);
	}
	template<typename T>
	constexpr T eps() noexcept
	{
		return std::numeric_limits<T>::epsilon();
	}

	template<typename T>
	constexpr T square(T x) noexcept
	{
		return x * x;
	}
	template<typename T>
	constexpr T sgn(T x) noexcept
	{
		return (zero<T>() < x) - (x < zero<T>());
	}

	/**
	 * @brief Computes log(epsilon + x) making sure that logarithm of 0 never occurs.
	 */
	template<typename T>
	constexpr T safe_log(T x) noexcept
	{
		return std::log(eps<T>() + x);
	}
} /* namespace scalar */

namespace avocado
{
	namespace backend
	{
		enum class SimdLevel
		{
			NONE,
			SSE,
			SSE2,
			SSE3,
			SSSE3,
			SSE41,
			SSE42,
			AVX,
			F16C,
			AVX2,
			AVX512F,
			AVX512VL_BW_DQ
		};

		std::string toString(SimdLevel sl);
		SimdLevel simdLevelFromString(const std::string &str);

		SimdLevel getSimdSupport() noexcept;

	} /* namespace backend */
} /* namespace avocado */

#endif /* UTILS_HPP_ */
