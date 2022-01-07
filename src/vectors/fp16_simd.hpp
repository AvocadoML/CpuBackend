/*
 * fp16_simd.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP16_SIMD_HPP_
#define VECTORS_FP16_SIMD_HPP_

#include "generic_simd.hpp"
#include "fp32_simd.hpp"
#include "simd_length.hpp"
#include "simd_utils.hpp"
#include "simd_load_store.hpp"

namespace scalar
{
	using avocado::backend::float16;
	static inline float float16_to_float(float16 x) noexcept
	{
#if SUPPORTS_FP16
		return _cvtsh_ss(x.m_data);
#else
		return 0.0f;
#endif
	}
	static inline float16 float_to_float16(float x) noexcept
	{
#if SUPPORTS_FP16
		return float16 { _cvtss_sh(x, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)) };
#else
		return float16 { 0u };
#endif
	}
}

namespace SIMD_NAMESPACE
{
	using avocado::backend::float16;

#if SUPPORTS_AVX

	static inline __m256 float16_to_float(__m128i x) noexcept
	{
#  if SUPPORTS_FP16
		return _mm256_cvtph_ps(x);
#  else
		return _mm256_setzeros_ps();
#  endif
	}
	static inline __m128i float_to_float16(__m256 x) noexcept
	{
#  if SUPPORTS_FP16
		return _mm256_cvtps_ph(x, _MM_FROUND_NO_EXC);
#  else
		return _mm256_setzeros_si256();
#  endif

	}

#elif SUPPORTS_SSE2 /* if __AVX__ is not defined */

	static inline __m128 float16_to_float(__m128i x) noexcept
	{
#  if SUPPORTS_FP16
		return _mm_cvtph_ps(x);
#  else
		return _mm_setzeros_ps();
#  endif
	}
	static inline __m128i float_to_float16(__m128 x) noexcept
	{
#  if SUPPORTS_FP16
		return _mm_cvtps_ph(x, _MM_FROUND_NO_EXC);
#  else
		return _mm_setzeros_si128();
#  endif
	}

#else /* if __SSE2__ is not defined */

	static inline float float16_to_floatbfloat16 x) noexcept
	{
		return scalar::float16_to_float(x);
	}
	static inline float16 float_to_float16(float x) noexcept
	{
		return scalar::float_to_float16(x);
	}

#endif

	template<>
	class SIMD<float16>
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
			static constexpr int length = simd_length<float16>();

			SIMD() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			SIMD(const float *ptr, int num = length) noexcept :
					m_data(simd_load(ptr, num))
			{
			}
			SIMD(const float16 *ptr, int num = length) noexcept :
					m_data(float16_to_float(simd_load(ptr, num)))
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
			SIMD(float16 x) noexcept :
					SIMD(scalar::float16_to_float(x))
			{
			}
			SIMD(double x) noexcept :
					SIMD(static_cast<float>(x))
			{
			}
			operator SIMD<float>() const noexcept
			{
				return SIMD<float>(m_data);
			}
			void load(const float *ptr, int num = length) noexcept
			{
				m_data = simd_load(ptr, num);
			}
			void load(const float16 *ptr, int num = length) noexcept
			{
				m_data = float16_to_float(simd_load(ptr, num));
			}
			void store(float *ptr, int num = length) const noexcept
			{
				simd_store(m_data, ptr, num);
			}
			void store(float16 *ptr, int num = length) const noexcept
			{
				simd_store(float_to_float16(m_data), ptr, num);
			}
			void insert(float value, int index) noexcept
			{
//				m_data.insert(value, index);  // FIXME
			}
			float extract(int index) const noexcept
			{
				return 0.0f; // FIXME
//				return m_data.extract(index);
			}
			float operator[](int index) const noexcept
			{
				return extract(index);
			}

			static constexpr float16 scalar_zero() noexcept
			{
				return float16 { 0x0000u };
			}
			static constexpr float16 scalar_one() noexcept
			{
				return float16 { 0x3c00u };
			}
			static constexpr float16 scalar_epsilon() noexcept
			{
				return float16 { 0x0400u };
			}

			static SIMD<float16> zero() noexcept
			{
				return SIMD<float16>(scalar_zero());
			}
			static SIMD<float16> one() noexcept
			{
				return SIMD<float16>(scalar_one());
			}
			static SIMD<float16> epsilon() noexcept
			{
				return SIMD<float16>(scalar_epsilon());
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of half floats, either 0x0000 (0.0f) for false, or 0xFFFF (-nan) for true.
	 */
	static inline SIMD<float16> operator==(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) == static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator!=(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) != static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator<(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) < static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator<=(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) <= static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator&(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) & static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator|(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) | static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator^(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) ^ static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator~(SIMD<float16> x) noexcept
	{
		return ~static_cast<SIMD<float>>(x);
	}
	static inline SIMD<float16> operator!(SIMD<float16> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline SIMD<float16> operator+(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) + static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator-(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) - static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator-(SIMD<float16> x) noexcept
	{
		return -static_cast<SIMD<float>>(x);
	}
	static inline SIMD<float16> operator*(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) * static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator/(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) / static_cast<SIMD<float>>(rhs);
	}

	/*
	 * Mixed precision arithmetics
	 */
	static inline SIMD<float16> operator+(SIMD<float16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) + rhs;
	}
	static inline SIMD<float16> operator+(SIMD<float> lhs, SIMD<float16> rhs) noexcept
	{
		return lhs + static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator-(SIMD<float16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) - rhs;
	}
	static inline SIMD<float16> operator-(SIMD<float> lhs, SIMD<float16> rhs) noexcept
	{
		return lhs - static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator*(SIMD<float16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) * rhs;
	}
	static inline SIMD<float16> operator*(SIMD<float> lhs, SIMD<float16> rhs) noexcept
	{
		return lhs * static_cast<SIMD<float>>(rhs);
	}
	static inline SIMD<float16> operator/(SIMD<float16> lhs, SIMD<float> rhs) noexcept
	{
		return static_cast<SIMD<float>>(lhs) / rhs;
	}
	static inline SIMD<float16> operator/(SIMD<float> lhs, SIMD<float16> rhs) noexcept
	{
		return lhs / static_cast<SIMD<float>>(rhs);
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline SIMD<float16> select(SIMD<float16> mask, SIMD<float16> x, SIMD<float16> y)
	{
		return select(static_cast<SIMD<float>>(mask), static_cast<SIMD<float>>(x), static_cast<SIMD<float>>(y));
	}

	/*
	 * Float vector functions
	 */
	static inline SIMD<float16> max(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return max(static_cast<SIMD<float>>(lhs), static_cast<SIMD<float>>(rhs));
	}
	static inline SIMD<float16> min(SIMD<float16> lhs, SIMD<float16> rhs) noexcept
	{
		return min(static_cast<SIMD<float>>(lhs), static_cast<SIMD<float>>(rhs));
	}
	static inline SIMD<float16> abs(SIMD<float16> x) noexcept
	{
		return abs(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> sqrt(SIMD<float16> x) noexcept
	{
		return sqrt(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> rsqrt(SIMD<float16> x) noexcept
	{
		return rsqrt(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> rcp(SIMD<float16> x) noexcept
	{
		return rcp(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> sgn(SIMD<float16> x) noexcept
	{
		return sgn(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> floor(SIMD<float16> x) noexcept
	{
		return floor(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> ceil(SIMD<float16> x) noexcept
	{
		return ceil(static_cast<SIMD<float>>(x));
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline SIMD<float16> mul_add(SIMD<float16> a, SIMD<float16> b, SIMD<float16> c) noexcept
	{
#if SUPPORTS_AVX and defined(__FMA__)
		return SIMD<float16>(_mm256_fmadd_ps(static_cast<SIMD<float>>(a), static_cast<SIMD<float>>(b), static_cast<SIMD<float>>(c)));
#else
		return a * b + c;
#endif
	}
	/* Calculates a * b - c */
	static inline SIMD<float16> mul_sub(SIMD<float16> a, SIMD<float16> b, SIMD<float16> c) noexcept
	{
#if SUPPORTS_AVX and defined(__FMA__)
		return SIMD<float16>(_mm256_fmsub_ps(static_cast<SIMD<float>>(a), static_cast<SIMD<float>>(b), static_cast<SIMD<float>>(c)));
#else
		return a * b - c;
#endif
	}

	/*
	 * Horizontal functions
	 */
	static inline float16 horizontal_add(SIMD<float16> x) noexcept
	{
		return scalar::float_to_float16(horizontal_add(static_cast<SIMD<float>>(x)));
	}
	static inline float16 horizontal_mul(SIMD<float16> x) noexcept
	{
		return scalar::float_to_float16(horizontal_mul(static_cast<SIMD<float>>(x)));
	}
	static inline float16 horizontal_min(SIMD<float16> x) noexcept
	{
		return scalar::float_to_float16(horizontal_min(static_cast<SIMD<float>>(x)));
	}
	static inline float16 horizontal_max(SIMD<float16> x) noexcept
	{
		return scalar::float_to_float16(horizontal_max(static_cast<SIMD<float>>(x)));
	}
}

#endif /* VECTORS_FP16_SIMD_HPP_ */
