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

namespace avocado
{
	namespace backend
	{
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

namespace scalar
{
	static inline float float16_to_float(avocado::backend::float16 x) noexcept
	{
#if SUPPORTS_FP16
		return _cvtsh_ss(x.m_data);
#else
		return 0.0f;
#endif
	}
	static inline avocado::backend::float16 float_to_float16(float x) noexcept
	{
#if SUPPORTS_FP16
		return avocado::backend::float16 { _cvtss_sh(x, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)) };
#else
		return avocado::backend::float16 { 0u };
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
			SIMD<float> m_data;
		public:
			static constexpr int length = SIMD<float>::length;

			SIMD() noexcept
			{
			}
			SIMD(const float *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const float16 *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const float *ptr, int num) noexcept
			{
				loadu(ptr, num);
			}
			SIMD(const float16 *ptr, int num) noexcept
			{
				loadu(ptr, num);
			}
			SIMD(SIMD<float> x) noexcept :
					m_data(x)
			{
			}
			SIMD(float x) noexcept :
					m_data(x)
			{
			}
			SIMD(float16 x) noexcept :
					m_data(scalar::float16_to_float(x))
			{
			}
			operator SIMD<float>() const noexcept
			{
				return m_data;
			}
			void loadu(const float *ptr) noexcept
			{
				m_data.loadu(ptr);
			}
			void loadu(const float16 *ptr) noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX and SUPPORTS_FP16
				__m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
				m_data = _mm256_cvtph_ps(tmp);
#elif SUPPORTS_SSE2 and SUPPORTS_FP16
				__m128i tmp = _mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr));
				m_data = _mm_cvtph_ps(tmp);
#else
				m_data = float16_to_float(ptr[0]);
#endif
			}
			void loadu(const float *ptr, int num) noexcept
			{
				m_data.loadu(ptr, num);
			}
			void loadu(const float16 *ptr, int num) noexcept
			{
				assert(ptr != nullptr);
				assert(num >= 0 && num <= length);
#if SUPPORTS_AVX and SUPPORTS_FP16
				__m128i tmp = partial_load(ptr, sizeof(float16) * num);
				m_data = SIMD<float>(_mm256_cvtph_ps(tmp));
#elif SUPPORTS_SSE2 and SUPPORTS_FP16
				__m128i tmp = partial_load(ptr, sizeof(float16) * num);
				m_data = _mm_cvtph_ps(tmp);
#else
				m_data = float16_to_float(ptr[0]);
#endif
			}
			void storeu(float *ptr) const noexcept
			{
				m_data.storeu(ptr);
			}
			void storeu(float16 *ptr) const noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX and SUPPORTS_FP16
				__m128i tmp = _mm256_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), tmp);
#elif SUPPORTS_SSE2 and SUPPORTS_FP16
				__m128i tmp = _mm_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), tmp);
#else
				ptr[0] = float_to_float16(m_data);
#endif
			}
			void storeu(float *ptr, int num) const noexcept
			{
				m_data.storeu(ptr, num);
			}
			void storeu(float16 *ptr, int num) const noexcept
			{
				assert(ptr != nullptr);
				assert(num >= 0 && num <= length);
#if SUPPORTS_AVX and SUPPORTS_FP16
				__m128i tmp = _mm256_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				partial_store(tmp, ptr, num * sizeof(float16));
#elif SUPPORTS_SSE2 and SUPPORTS_FP16
				__m128i tmp = _mm_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				partial_store(tmp, ptr, num * sizeof(float16));
#else
				ptr[0] = float_to_float16(m_data);
#endif
			}
			void insert(float value, int index) noexcept
			{
				m_data.insert(value, index);
			}
			float extract(int index) const noexcept
			{
				return m_data.extract(index);
			}
			float operator[](int index) const noexcept
			{
				return extract(index);
			}

			static float16 scalar_zero() noexcept
			{
				return float16 { 0x0000u };
			}
			static float16 scalar_one() noexcept
			{
				return float16 { 0x3c00u };
			}
			static float16 scalar_epsilon() noexcept
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
	static inline SIMD<float16> approx_recipr(SIMD<float16> x) noexcept
	{
		return approx_recipr(static_cast<SIMD<float>>(x));
	}
	static inline SIMD<float16> approx_rsqrt(SIMD<float16> x) noexcept
	{
		return approx_rsqrt(static_cast<SIMD<float>>(x));
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
