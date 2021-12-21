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

		inline bool supPorts_simd(avDeviceProperty_t prop)
		{
			bool result = false;
			cpuGetDeviceProperty(prop, &result);
			return result;
		}

		inline SimdLevel getSimdSupport() noexcept
		{
			static const SimdLevel supported_simd_level = []()
			{
//				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ))
//					return SimdLevel::AVX512VL_BW_DQ;
//				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_AVX512_F))
//					return SimdLevel::AVX512F;
				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_AVX2))
					return SimdLevel::AVX2;
				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION))
					return SimdLevel::F16C;
				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_AVX))
					return SimdLevel::AVX;
//				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_SSE42))
//					return SimdLevel::SSE42;
				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_SSE41))
					return SimdLevel::SSE41;
//				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_SSSE3))
//					return SimdLevel::SSSE3;
//				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_SSE3))
//					return SimdLevel::SSE3;
				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_SSE2))
					return SimdLevel::SSE2;
//				if (supPorts_simd (AVOCADO_DEVICE_SUPPORTS_SSE))
//					return SimdLevel::SSE;
				return SimdLevel::NONE;
			}();
			return supported_simd_level;
		}
	} /* namespace backend */
} /* namespace avocado */

#endif /* UTILS_HPP_ */
