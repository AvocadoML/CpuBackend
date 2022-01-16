/*
 * simd_load_store.hpp
 *
 *  Created on: Jan 5, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_SIMD_LOAD_STORE_HPP_
#define VECTORS_SIMD_LOAD_STORE_HPP_

#include "simd_macros.hpp"
#include "simd_utils.hpp"
#include "simd_length.hpp"

#include <cassert>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
	using avocado::backend::float16;
	using avocado::backend::bfloat16;

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
		assert(bytes >= 0 && bytes <= 16);
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
			}
			case 16:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
			default:
			{
				int32_t tmp[4] = { 0, 0, 0, 0 };
				std::memcpy(tmp, ptr, bytes);
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
			}
		}
	}
	static inline __m128i partial_load(const int16_t *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 8);
		switch (num)
		{
			default:
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_setr_epi16(ptr[0], 0u, 0u, 0u, 0u, 0u, 0u, 0u);
			case 2:
				return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
			case 3:
				return _mm_setr_epi16(ptr[0], ptr[1], ptr[2], 0u, 0u, 0u, 0u, 0u);
			case 4:
				return _mm_loadu_si64(ptr);
			case 5:
				return _mm_setr_epi16(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], 0u, 0u, 0u);
			case 6:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(ptr) + 2);
				return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
			}
			case 7:
				return _mm_setr_epi16(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], 0u);
			case 8:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	}
	static inline __m128i partial_load(const int32_t *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 4);
		switch (num)
		{
			default:
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
			case 2:
				return _mm_loadu_si64(ptr);
			case 3:
			{
				__m128 tmp1 = _mm_castsi128_ps(_mm_loadu_si64(ptr));
				__m128 tmp2 = _mm_load_ss(reinterpret_cast<const float*>(ptr) + 2);
				return _mm_castps_si128(_mm_movelh_ps(tmp1, tmp2));
			}
			case 4:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	}
	static inline __m128i partial_load(const int64_t *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 2);
		switch (num)
		{
			default:
			case 0:
				return _mm_setzero_si128();
			case 1:
				return _mm_loadu_si64(ptr);
			case 2:
				return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	}
	static inline void partial_store(__m128i reg, void *ptr, const int bytes) noexcept
	{
		assert(bytes >= 0 && bytes <= 16);
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
	static inline void partial_store(__m128i reg, int16_t *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 8);
		switch (num)
		{
			case 0:
				break;
			case 1:
				ptr[0] = _mm_extract_epi16(reg, 1);
				break;
			case 2:
			case 3:
				_mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(reg));
				if (num == 3)
					ptr[3] = _mm_extract_epi16(reg, 3);
				break;
			case 4:
			case 5:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				if (num == 5)
					ptr[5] = _mm_extract_epi16(reg, 5);
				break;
			case 6:
			case 7:
			{
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(reg), _mm_castsi128_ps(reg));
				_mm_store_ss(reinterpret_cast<float*>(ptr) + 2, tmp);
				if (num == 7)
					ptr[7] = _mm_extract_epi16(reg, 7);
				break;
			}
			case 8:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
		}
	}
	static inline void partial_store(__m128i reg, int32_t *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 4);
		switch (num)
		{
			case 0:
				break;
			case 1:
				_mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(reg));
				break;
			case 2:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 3:
			{
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				__m128 tmp = _mm_movehl_ps(_mm_castsi128_ps(reg), _mm_castsi128_ps(reg));
				_mm_store_ss(reinterpret_cast<float*>(ptr) + 2, tmp);
				break;
			}
			case 4:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
		}
	}
	static inline void partial_store(__m128i reg, int64_t *ptr, const int num) noexcept
	{
		assert(num >= 0 && num <= 2);
		switch (num)
		{
			case 0:
				break;
			case 1:
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), reg);
				break;
			case 2:
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), reg);
				break;
		}
	}
#endif

#if SUPPORTS_AVX

	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline __m256i simd_load(const T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= simd_length<T>());
		if (num == simd_length<T>())
			return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
		else
		{
			if (num > simd_length<T>() / 2)
				return combine(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)),
						partial_load(ptr + (simd_length<T>() / 2), sizeof(T) * (num - simd_length<T>() / 2)));
			else
				return combine(partial_load(ptr, sizeof(T) * num));
		}
	}
	static inline __m128i simd_load(const bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 8); // TODO AVX512 adds full support for bfloat16 data
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	static inline __m128i simd_load(const float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 8); // TODO AVX512 adds full support for float16 data
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	static inline __m256 simd_load(const float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 8);
		if (num == 8)
			return _mm256_loadu_ps(ptr);
		else
		{
			if (num > 4)
				return combine(_mm_loadu_ps(ptr), partial_load(ptr + 4, num - 4));
			else
				return combine(partial_load(ptr, num));
		}
	}
	static inline __m256d simd_load(const double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4);
		if (num == 4)
			return _mm256_loadu_pd(ptr);
		else
		{
			if (num > 2)
				return combine(_mm_loadu_pd(ptr), partial_load(ptr + 2, num - 2));
			else
				return combine(partial_load(ptr, num));
		}
	}

	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline void simd_store(__m256i x, T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= simd_length<T>());
		if (num == simd_length<T>())
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), x);
		else
		{
			if (num > simd_length<T>() / 2)
			{
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), get_low(x));
				partial_store(get_high(x), ptr + simd_length<T>() / 2, sizeof(T) * (num - simd_length<T>() / 2));
			}
			else
				partial_store(get_low(x), ptr, sizeof(T) * num);
		}
	}
	static inline void simd_store(__m128i x, bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 8);
		partial_store(x, reinterpret_cast<int16_t*>(ptr), num);
	}
	static inline void simd_store(__m128i x, float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 8);
		partial_store(x, reinterpret_cast<int16_t*>(ptr), num);
	}
	static inline void simd_store(__m256 x, float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 8);
		if (num == simd_length<float>())
			_mm256_storeu_ps(ptr, x);
		else
		{
			if (num > simd_length<float>() / 2)
			{
				_mm_storeu_ps(ptr, get_low(x));
				partial_store(get_high(x), ptr + simd_length<float>() / 2, num - simd_length<float>() / 2);
			}
			else
				partial_store(get_low(x), ptr, num);
		}
	}
	static inline void simd_store(__m256d x, double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4);
		if (num == 4)
			_mm256_storeu_pd(ptr, x);
		else
		{
			if (num > 4 / 2)
			{
				_mm_storeu_pd(ptr, get_low(x));
				partial_store(get_high(x), ptr + 4 / 2, num - 4 / 2);
			}
			else
				partial_store(get_low(x), ptr, num);
		}
	}

#elif SUPPORTS_SSE2

	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline __m128i simd_load(const T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= simd_length<T>());
		return partial_load(ptr, sizeof(T) * num);
	}
	static inline __m128i simd_load(const bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4); // TODO AVX512 adds full support for bfloat16 data
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	static inline __m128i simd_load(const float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4); // TODO AVX512 adds full support for float16 data
		return partial_load(reinterpret_cast<const int16_t*>(ptr), num);
	}
	static inline __m128 simd_load(const float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4);
		return partial_load(ptr, num);
	}
	static inline __m128d simd_load(const double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 2);
		return partial_load(ptr, num);
	}

	template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
	static inline void simd_store(__m128i x, T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= simd_length<T>());
		partial_store(x, ptr, sizeof(T) * num);
	}
	static inline void simd_store(__m128i x, bfloat16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4);
		partial_store(x, reinterpret_cast<int16_t*>(ptr), num);
	}
	static inline void simd_store(__m128i x, float16 *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4);
		partial_store(x, reinterpret_cast<int16_t*>(ptr), num);
	}
	static inline void simd_store(__m128 x, float *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 4);
		partial_store(x, ptr, num);
	}
	static inline void simd_store(__m128d x, double *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 2);
		partial_store(x, ptr, num);
	}

#else

	template<typename T>
	static inline T simd_load(const T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 1);
		return ptr[0];
	}
	template<typename T>
	static inline void simd_store(T x, T *ptr, int num) noexcept
	{
		assert(ptr != nullptr);
		assert(num >= 0 && num <= 1);
		ptr[0] = x;
	}
#endif

} /* SIMD_NAMESPACE */

#endif /* VECTORS_SIMD_LOAD_STORE_HPP_ */
