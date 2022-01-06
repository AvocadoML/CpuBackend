/*
 * simd_utils.hpp
 *
 *  Created on: Jan 6, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_SIMD_UTILS_HPP_
#define VECTORS_SIMD_UTILS_HPP_

#include "simd_macros.hpp"

#include <cstring>
#include <cmath>
#include <inttypes.h>
#include <x86intrin.h>

namespace avocado
{
	namespace backend
	{
		struct bfloat16
		{
				uint16_t m_data;

				friend bool operator==(bfloat16 lhs, bfloat16 rhs) noexcept
				{
					return lhs.m_data == rhs.m_data;
				}
				friend bool operator!=(bfloat16 lhs, bfloat16 rhs) noexcept
				{
					return lhs.m_data != rhs.m_data;
				}
		};

		struct float16
		{
				uint16_t m_data;

				friend bool operator==(float16 lhs, float16 rhs) noexcept
				{
					return lhs.m_data == rhs.m_data;
				}
				friend bool operator!=(float16 lhs, float16 rhs) noexcept
				{
					return lhs.m_data != rhs.m_data;
				}
		};
	}
}

namespace SIMD_NAMESPACE
{
#if SUPPORTS_AVX
	static inline __m128 get_low(__m256 reg) noexcept
	{
		return _mm256_castps256_ps128(reg);
	}
	static inline __m128 get_high(__m256 reg) noexcept
	{
		return _mm256_extractf128_ps(reg, 1);
	}

	static inline __m128d get_low(__m256d reg) noexcept
	{
		return _mm256_castpd256_pd128(reg);
	}
	static inline __m128d get_high(__m256d reg) noexcept
	{
		return _mm256_extractf128_pd(reg, 1);
	}

	static inline __m128i get_low(__m256i reg) noexcept
	{
		return _mm256_castsi256_si128(reg);
	}
	static inline __m128i get_high(__m256i reg) noexcept
	{
		return _mm256_extractf128_si256(reg, 1);
	}

	static inline __m256 combine(__m128 low, __m128 high = _mm_setzero_ps()) noexcept
	{
		return _mm256_setr_m128(low, high);
	}
	static inline __m256d combine(__m128d low, __m128d high = _mm_setzero_pd()) noexcept
	{
		return _mm256_setr_m128d(low, high);
	}
	static inline __m256i combine(__m128i low, __m128i high = _mm_setzero_si128()) noexcept
	{
		return _mm256_setr_m128i(low, high);
	}
#endif

#if SUPPORTS_AVX
	template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7>
	inline __m256i constant() noexcept
	{
		return _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
	}
	template<uint32_t i0, uint32_t i1>
	inline __m256i constant() noexcept
	{
		return constant<i0, i1, i0, i1, i0, i1, i0, i1>();
	}
	template<uint32_t i>
	inline __m256i constant() noexcept
	{
		return constant<i, i, i, i, i, i, i, i>();
	}
#elif SUPPORTS_SSE2
	template<uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3>
	inline __m128i constant() noexcept
	{
		return _mm_setr_epi32(i0, i1, i2, i3);
	}
	template<uint32_t i0, uint32_t i1>
	inline __m128i constant() noexcept
	{
		return constant<i0, i1, i0, i1>();
	}
	template<uint32_t i>
	inline __m128i constant() noexcept
	{
		return constant<i, i, i, i>();
	}
#endif

	template<typename T, typename U>
	inline constexpr T bitwise_cast(U x) noexcept
	{
		static_assert(sizeof(T) == sizeof(U));
		T result;
		std::memcpy(&result, &x, sizeof(T));
		return result;
	}
}

#endif /* VECTORS_SIMD_UTILS_HPP_ */
