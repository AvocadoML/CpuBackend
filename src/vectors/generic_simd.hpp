/*
 * generic_simd.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_GENERIC_SIMD_HPP_
#define VECTORS_GENERIC_SIMD_HPP_

#include "simd_macros.hpp"

#include <cstring>
#include <cmath>
#include <inttypes.h>
#include <cassert>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
	template<typename T, class dummy = T>
	class SIMD;

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
	constexpr T bitwise_cast(U x) noexcept
	{
		static_assert(sizeof(T) == sizeof(U));
		T result;
		std::memcpy(&result, &x, sizeof(T));
		return result;
	}

#if SUPPORTS_SSE2
	static inline __m128d partial_load(const double *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 2);
		switch (num)
		{
			default:
			case 0:
				return _mm_setzero_pd();
			case 1:
				return _mm_load_sd(ptr);
			case 2:
				return _mm_loadu_pd(ptr);
		}
	}
	static inline void partial_store(__m128d reg, double *ptr, const int num) noexcept
	{
		assert(num > 0 && num <= 2);
		switch (num)
		{
			case 1:
				_mm_store_sd(ptr, reg);
				break;
			case 2:
				_mm_store_pd(ptr, reg);
				break;
		}
	}

	static inline __m128 partial_load(const float *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 4);
		switch (num)
		{
			default:
			case 0:
				return _mm_setzero_ps();
			case 1:
				return _mm_load_ss(ptr);
			case 2:
				return _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(ptr)));
			case 3:
			{
				__m128 tmp1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(ptr)));
				__m128 tmp2 = _mm_load_ss(ptr + 2);
				return _mm_movelh_ps(tmp1, tmp2);
			}
			case 4:
				return _mm_loadu_ps(ptr);
		}
	}
	static inline void partial_store(__m128 reg, float *ptr, const int num) noexcept
	{
		assert(num > 0 && num <= 4);
		switch (num)
		{
			case 1:
				_mm_store_ss(ptr, reg);
				break;
			case 2:
				_mm_store_sd(reinterpret_cast<double*>(ptr), _mm_castps_pd(reg));
				break;
			case 3:
			{
				_mm_store_sd(reinterpret_cast<double*>(ptr), _mm_castps_pd(reg));
				__m128 tmp = _mm_movehl_ps(reg, reg);
				_mm_store_ss(ptr + 2, tmp);
				break;
			}
			case 4:
				_mm_storeu_ps(ptr, reg);
				break;
		}
	}

#endif
#if SUPPORTS_SSE2 // _mm_castps_si128() requires sse2
	static inline __m128i partial_load(const void *ptr, const int bytes) noexcept
	{
		assert(num >= 0 && num <= 16);
		switch (bytes)
		{
			case 0:
				return _mm_setzero_si128();
			case 4:
				return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
			case 8:
				return _mm_loadu_si64(ptr);
			case 12:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(ptr) + 2);
				return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
				break;
			}
			case 16:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
			default:
			{
				int32_t tmp[4];
				std::memcpy(tmp, ptr, bytes);
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
			}
		}
	}
	static inline void partial_store(__m128i reg, void *ptr, const int bytes) noexcept
	{
		assert(num >= 0 && num <= 16);
		switch (bytes)
		{
			case 0:
				break;
			case 4:
				_mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(reg));
				break;
			case 8:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 12:
			{
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(reg), _mm_castsi128_ps(reg));
				_mm_store_ss(reinterpret_cast<float*>(ptr) + 2, tmp);
				break;
			}
			case 16:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			default:
			{
				int32_t tmp[4];
				_mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), reg);
				std::memcpy(ptr, tmp, bytes);
				break;
			}
		}
	}
#endif

	/*
	 * Bitwise operations.
	 */
	template<typename T, typename U = T>
	static inline SIMD<T>& operator&(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs & SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator&(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) & rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator&=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs & rhs);
		return lhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T>& operator|(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs | SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator|(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) | rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator|=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs | rhs);
		return lhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T>& operator^(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs ^ SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator^(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) ^ rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator^=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs ^ rhs);
		return lhs;
	}

	/*
	 * Bitwise shifts.
	 */
	template<typename T>
	static inline SIMD<T>& operator>>=(SIMD<T> &lhs, T rhs) noexcept
	{
		lhs = (lhs >> rhs);
		return lhs;
	}
	template<typename T>
	static inline SIMD<T>& operator<<=(SIMD<T> &lhs, T rhs) noexcept
	{
		lhs = (lhs << rhs);
		return lhs;
	}

	/*
	 * Compare operations.
	 */
	template<typename T, typename U = T>
	static inline SIMD<T> operator<(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs < SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator<(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) < rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator<=(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs <= SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator<=(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) <= rhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T> operator>(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator>(U lhs, SIMD<T> rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator>=(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs < rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator>=(U lhs, SIMD<T> rhs) noexcept
	{
		return lhs < rhs;
	}

	template<typename T>
	static inline SIMD<T> operator>(SIMD<T> lhs, SIMD<T> rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T>
	static inline SIMD<T> operator>=(SIMD<T> lhs, SIMD<T> rhs) noexcept
	{
		return lhs < rhs;
	}

	/*
	 * Arithmetic operations
	 */
	template<typename T, typename U = T>
	static inline SIMD<T> operator+(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs + SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator+(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) + rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator+=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator+=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T>
	static inline SIMD<T> operator+(SIMD<T> x) noexcept
	{
		return x;
	}

	template<typename T, typename U = T>
	static inline SIMD<T> operator-(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs - SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator-(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) - rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator-=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator-=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T> operator*(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs * SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator*(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) * rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator*=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator*=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T, typename U>
	static inline SIMD<T> operator/(SIMD<U> lhs, T rhs) noexcept
	{
		return lhs / SIMD<U>(rhs);
	}
	template<typename T, typename U>
	static inline SIMD<T> operator/(T lhs, SIMD<U> rhs) noexcept
	{
		return SIMD<U>(lhs) / rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator/=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator/=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_GENERIC_SIMD_HPP_ */
