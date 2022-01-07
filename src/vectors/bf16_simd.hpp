/*
 * bf16_simd.hpp
 *
 *  Created on: Nov 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_BF16_SIMD_HPP_
#define VECTORS_BF16_SIMD_HPP_

#include "fp32_simd.hpp"
#include "generic_simd.hpp"
#include "simd_length.hpp"
#include "simd_utils.hpp"
#include "simd_load_store.hpp"

#include <cstring>

namespace scalar
{
	using avocado::backend::bfloat16;
	static inline float bfloat16_to_float(bfloat16 x) noexcept
	{
		uint16_t tmp[2] = { 0u, x.m_data };
		float result;
		std::memcpy(&result, tmp, sizeof(float));
		return result;
	}
	static inline bfloat16 float_to_bfloat16(float x) noexcept
	{
		uint16_t tmp[2];
		std::memcpy(&tmp, &x, sizeof(float));
		return bfloat16 { tmp[1] };
	}
}

namespace SIMD_NAMESPACE
{
	using avocado::backend::bfloat16;

#if SUPPORTS_AVX

	static inline __m256 bfloat16_to_float(__m128i x) noexcept
	{
#  if SUPPORTS_AVX2
		__m256i tmp = _mm256_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
		tmp = _mm256_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
#  else
		__m128i tmp_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
		__m128i tmp_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), x); // pad upper half with zeros
		__m256i tmp = _mm256_setr_m128i(tmp_lo, tmp_hi); // combine two halves
#  endif
		return _mm256_castsi256_ps(tmp);
	}
	static inline __m128i float_to_bfloat16(__m256 x) noexcept
	{
#  if SUPPORTS_AVX2
		__m256i tmp = _mm256_srli_epi32(_mm256_castps_si256(x), 16); // shift right by 16 bits while shifting in zeros
		return _mm_packs_epi32(get_low(tmp), get_high(tmp)); // pack 32 bits into 16 bits
#  else
		__m128i tmp_lo = _mm_srli_epi32(_mm_castps_si128(get_low(x)), 16); // shift right by 16 bits while shifting in zeros
		__m128i tmp_hi = _mm_srli_epi32(_mm_castps_si128(get_high(x)), 16); // shift right by 16 bits while shifting in zeros
		return _mm_packs_epi32(tmp_lo, tmp_hi); // pack 32 bits into 16 bits
#  endif
	}

#elif SUPPORTS_SSE2

	static inline __m128 bfloat16_to_float(__m128i x) noexcept
	{
#  if SUPPORTS_SSE41
		__m128i tmp = _mm_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
		tmp = _mm_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
#  else
		__m128i tmp = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
#  endif
		return _mm_castsi128_ps(tmp);
	}
	static inline __m128i float_to_bfloat16(__m128 x) noexcept
	{
		__m128i tmp = _mm_srli_epi32(_mm_castps_si128(x), 16); // shift right by 16 bits while shifting in zeros
		return _mm_packs_epi32(tmp, _mm_setzero_si128()); // pack 32 bits into 16 bits
	}

#else

	static inline float bfloat16_to_float(bfloat16 x) noexcept
	{
		return scalar::bfloat16_to_float(x);
	}
	static inline bfloat16 float_to_bfloat16(float x) noexcept
	{
		return scalar::float_to_bfloat16(x);
	}

#endif

	template<>
	class SIMD<bfloat16>
	{
		private:
#if SUPPORTS_AVX
			__m256 m_data;
#elif SUPPORTS_SSE2
			__m128 m_data;
#else
			float m_data;
#endif
		public:
			static constexpr int length = simd_length<bfloat16>();

			SIMD() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			SIMD(const float *ptr, int num = length) noexcept :
					m_data(simd_load(ptr, num))
			{
			}
			SIMD(const bfloat16 *ptr, int num = length) noexcept :
					m_data(bfloat16_to_float(simd_load(ptr, num)))
			{
			}
			SIMD(SIMD<float> x) noexcept :
					m_data(x)
			{
			}
			SIMD(float x) noexcept
			{
#if SUPPORTS_AVX
				m_data = _mm256_set1_ps(x);
#elif SUPPORTS_SSE2
				m_data = _mm_set1_ps(x);
#else
				m_data = x;
#endif
			}
			SIMD(bfloat16 x) noexcept
			{
#if SUPPORTS_SSE2
				m_data = bfloat16_to_float(_mm_set1_epi16(x.m_data));
#else
				m_data = scalar::bfloat16_to_float(x);
#endif
			}
			SIMD(double x) noexcept :
					SIMD(static_cast<float>(x))
			{
			}
			operator SIMD<float>() const noexcept
			{
				return SIMD<float>(m_data);
			}
			void load(const float *ptr, int num) noexcept
			{
				m_data = simd_load(ptr, num);
			}
			void load(const bfloat16 *ptr, int num = length) noexcept
			{
				m_data = bfloat16_to_float(simd_load(ptr, num));
			}
			void store(float *ptr, int num = length) const noexcept
			{
				simd_store(m_data, ptr, num);
			}
			void store(bfloat16 *ptr, int num = length) const noexcept
			{
				simd_store(float_to_bfloat16(m_data), ptr, num);
			}
			void insert(float value, int index) noexcept
			{
//				m_data.insert(value, index); // FIXME
			}
			float extract(int index) const noexcept
			{
				return 0.0f; //FIXME
//				return m_data.extract(index);
			}
			float operator[](int index) const noexcept
			{
				return extract(index);
			}

			static constexpr bfloat16 scalar_zero() noexcept
			{
				return bfloat16 { 0x0000u };
			}
			static constexpr bfloat16 scalar_one() noexcept
			{
				return bfloat16 { 0x3f80u };
			}
			static constexpr bfloat16 scalar_epsilon() noexcept
			{
				return bfloat16 { 0x0800u };
			}

			static SIMD<bfloat16> zero() noexcept
			{
				return SIMD<bfloat16>(scalar_zero());
			}
			static SIMD<bfloat16> one() noexcept
			{
				return SIMD<bfloat16>(scalar_one());
			}
			static SIMD<bfloat16> epsilon() noexcept
			{
				return SIMD<bfloat16>(scalar_epsilon());
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of half floats, either 0x0000 (0.0f) for false, or 0xFFFF (-nan) for true.
	 */
	static inline SIMD<bfloat16> operator==(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) == static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator!=(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) != static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator<(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) < static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator<=(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) <= static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator&(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) & static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator|(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) | static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator^(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) ^ static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator~(SIMD<bfloat16> x) noexcept
	{
		return ~static_cast<SIMD<float>>(x);
	}
	static inline SIMD<bfloat16> operator!(SIMD<bfloat16> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline SIMD<bfloat16> operator+(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) + static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator-(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) - static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator-(SIMD<bfloat16> x) noexcept
	{
		return -static_cast<SIMD<float>>(x);
	}
	static inline SIMD<bfloat16> operator*(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) * static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator/(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) / static_cast<SIMD<float>>(rhs);
	}

	/*
	 * Mixed precision arithmetics
	 */
	static inline SIMD<bfloat16> operator+(SIMD<bfloat16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) + rhs;
	}
	static inline SIMD<bfloat16> operator+(SIMD<float> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return lhs + static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator-(SIMD<bfloat16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) - rhs;
	}
	static inline SIMD<bfloat16> operator-(SIMD<float> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return lhs - static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator*(SIMD<bfloat16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) * rhs;
	}
	static inline SIMD<bfloat16> operator*(SIMD<float> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return lhs * static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<bfloat16> operator/(SIMD<bfloat16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) / rhs;
	}
	static inline SIMD<bfloat16> operator/(SIMD<float> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return lhs / static_cast<SIMD<float>>(rhs);
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline SIMD<bfloat16> select(SIMD<bfloat16> mask, SIMD<bfloat16> x, SIMD<bfloat16> y)
	{
		return select(static_cast<SIMD<float>>(mask), static_cast<SIMD<float>>(x), static_cast<SIMD<float>>(y));
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline SIMD<bfloat16> mul_add(SIMD<bfloat16> a, SIMD<bfloat16> b, SIMD<bfloat16> c) noexcept
	{
		return SIMD<bfloat16>(mul_add(static_cast<SIMD<float>>(a), static_cast<SIMD<float>>(b), static_cast<SIMD<float>>(c)));
	}
	/* Calculates a * b - c */
	static inline SIMD<bfloat16> mul_sub(SIMD<bfloat16> a, SIMD<bfloat16> b, SIMD<bfloat16> c) noexcept
	{
		return SIMD<bfloat16>(mul_sub(static_cast<SIMD<float>>(a), static_cast<SIMD<float>>(b), static_cast<SIMD<float>>(c)));
	}
	/* Calculates - a * b + c */
	static inline SIMD<bfloat16> neg_mul_add(SIMD<bfloat16> a, SIMD<bfloat16> b, SIMD<bfloat16> c) noexcept
	{
		return SIMD<bfloat16>(neg_mul_add(static_cast<SIMD<float>>(a), static_cast<SIMD<float>>(b), static_cast<SIMD<float>>(c)));
	}
	/* Calculates - a * b - c */
	static inline SIMD<bfloat16> neg_mul_sub(SIMD<bfloat16> a, SIMD<bfloat16> b, SIMD<bfloat16> c) noexcept
	{
		return SIMD<bfloat16>(neg_mul_sub(static_cast<SIMD<float>>(a), static_cast<SIMD<float>>(b), static_cast<SIMD<float>>(c)));
	}

	/*
	 * Float vector functions
	 */
	static inline SIMD<bfloat16> max(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return max(static_cast<SIMD<float>>(lhs), static_cast<SIMD<float>>(rhs));
	}
	static inline SIMD<bfloat16> min(SIMD<bfloat16> lhs, SIMD<bfloat16> rhs) noexcept
	{
		return min(static_cast<SIMD<float>>(lhs), static_cast<SIMD<float>>(rhs));
	}
	static inline SIMD<bfloat16> abs(SIMD<bfloat16> x) noexcept
	{
		return abs(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> sqrt(SIMD<bfloat16> x) noexcept
	{
		return sqrt(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> rsqrt(SIMD<bfloat16> x) noexcept
	{
		return rsqrt(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> rcp(SIMD<bfloat16> x) noexcept
	{
		return rcp(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> sgn(SIMD<bfloat16> x) noexcept
	{
		return sgn(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> floor(SIMD<bfloat16> x) noexcept
	{
		return floor(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<bfloat16> ceil(SIMD<bfloat16> x) noexcept
	{
		return ceil(static_cast<SIMD<float>>(x));
	}

	/*
	 * Horizontal functions
	 */
	static inline bfloat16 horizontal_add(SIMD<bfloat16> x) noexcept
	{
		return scalar::float_to_bfloat16(horizontal_add(static_cast<SIMD<float>>(x)));
	}
	static inline bfloat16 horizontal_mul(SIMD<bfloat16> x) noexcept
	{
		return scalar::float_to_bfloat16(horizontal_mul(static_cast<SIMD<float>>(x)));
	}
	static inline bfloat16 horizontal_min(SIMD<bfloat16> x) noexcept
	{
		return scalar::float_to_bfloat16(horizontal_min(static_cast<SIMD<float>>(x)));
	}
	static inline bfloat16 horizontal_max(SIMD<bfloat16> x) noexcept
	{
		return scalar::float_to_bfloat16(horizontal_max(static_cast<SIMD<float>>(x)));
	}

}

#endif /* VECTORS_BF16_SIMD_HPP_ */
