/*
 * fp64_simd.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP64_SSIMDHPP_
#define VECTORS_FP64_SSIMDHPP_

#include "generic_simd.hpp"
#include "simd_length.hpp"
#include "simd_utils.hpp"
#include "simd_load_store.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
	template<>
	class SIMD<double>
	{
		private:
#if SUPPORTS_AVX
			__m256d m_data;
#elif SUPPORTS_SSE2
			__m128d m_data;
#else
			double m_data;
#endif
		public:
			static constexpr int length = simd_length<double>();

			SIMD() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			SIMD(const double *ptr, int num = length) noexcept :
					m_data(simd_load(ptr, num))
			{
			}
			SIMD(double x) noexcept
			{
#if SUPPORTS_AVX
				m_data = _mm256_set1_pd(x);
#elif SUPPORTS_SSE2
				m_data = _mm_set1_pd(x);
#else
				m_data = x;
#endif
			}
			SIMD(float x) noexcept : // @suppress("Class members should be properly initialized")
					SIMD(static_cast<double>(x))
			{
			}
#if SUPPORTS_AVX
			SIMD(__m256d x) noexcept :
					m_data(x)
			{
			}
			SIMD(__m128d low) noexcept :
					m_data(_mm256_setr_m128d(low, _mm_setzero_pd()))
			{
			}
			SIMD(__m128d low, __m128d high) noexcept :
					m_data(_mm256_setr_m128d(low, high))
			{
			}
			SIMD<double>& operator=(__m256d x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m256d() const noexcept
			{
				return m_data;
			}
#elif SUPPORTS_SSE2
			SIMD(__m128d x) noexcept :
					m_data(x)
			{
			}
			SIMD<double>& operator=(__m128d x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m128d() const noexcept
			{
				return m_data;
			}
#else
			operator double() const noexcept
			{
				return m_data;
			}
#endif
			void load(const double *ptr, int num = length) noexcept
			{
				m_data = simd_load(ptr, num);
			}
			void store(double *ptr, int num = length) const noexcept
			{
				simd_store(m_data, ptr, num);
			}
			void insert(double value, int index) noexcept
			{
				assert(index >= 0 && index < length);
#if SUPPORTS_AVX
				__m256d tmp = _mm256_broadcast_sd(&value);
				switch (index)
				{
					case 0:
						m_data = _mm256_blend_pd(m_data, tmp, 1);
						break;
					case 1:
						m_data = _mm256_blend_pd(m_data, tmp, 2);
						break;
					case 2:
						m_data = _mm256_blend_pd(m_data, tmp, 4);
						break;
					case 3:
						m_data = _mm256_blend_pd(m_data, tmp, 8);
						break;
				}
#elif SUPPORTS_SSE2
				double tmp[2];
				store(tmp);
				tmp[index] = value;
				load(tmp);
#else
				m_data = value;
#endif
			}
			double extract(int index) const noexcept
			{
				assert(index >= 0 && index < length);
				double tmp[length];
				store(tmp);
				return tmp[index];
			}
			double operator[](int index) const noexcept
			{
				return extract(index);
			}
			void cutoff(const int num, SIMD<double> value = zero()) noexcept
			{
#if SUPPORTS_AVX
				switch(num)
				{
					case 0:
						m_data = value.m_data;
						break;
					case 1:
						m_data = _mm256_blend_pd(value, m_data, 1);
						break;
					case 2:
						m_data = _mm256_blend_pd(value, m_data, 3);
						break;
					case 3:
						m_data = _mm256_blend_pd(value, m_data, 7);
						break;
					default:
					case 4:
						m_data = _mm256_blend_pd(value, m_data, 15);
						break;
				}
#elif SUPPORTS_SSE41
				switch(num)
				{
					case 0:
						m_data = value.m_data;
						break;
					case 1:
						m_data = _mm_blend_pd(value, m_data, 1);
						break;
					default:
					case 2:
						m_data = _mm_blend_pd(value, m_data, 3);
						break;
				}
#elif SUPPORTS_SSE2
				__m128d mask = get_cutoff_mask_pd(num);
				m_data = _mm_or_pd(_mm_and_pd(mask, m_data), _mm_andnot_pd(mask, value));
#else
				if(num == 0)
					m_data = value.m_data;
#endif
			}

			static constexpr double scalar_zero() noexcept
			{
				return 0.0;
			}
			static constexpr double scalar_one() noexcept
			{
				return 1.0;
			}
			static constexpr double scalar_epsilon() noexcept
			{
				return std::numeric_limits<double>::epsilon();
			}

			static SIMD<double> zero() noexcept
			{
				return SIMD<double>(scalar_zero());
			}
			static SIMD<double> one() noexcept
			{
				return SIMD<double>(scalar_one());
			}
			static SIMD<double> epsilon() noexcept
			{
				return SIMD<double>(scalar_epsilon());
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of floats, either 0x0000000000000000 (0.0) for false, or 0xFFFFFFFFFFFFFFFF (-nan) for true.
	 */
	static inline SIMD<double> operator==(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_pd(lhs, rhs, 0);
#elif SUPPORTS_SSE2
		return _mm_cmpeq_pd(lhs, rhs);
#else
		return bitwise_cast<double>(static_cast<double>(lhs) == static_cast<double>(rhs) ? 0xFFFFFFFFFFFFFFFFu : 0x0000000000000000u);
#endif
	}
	static inline SIMD<double> operator!=(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_pd(lhs, rhs, 4);
#elif SUPPORTS_SSE2
		return _mm_cmpneq_pd(lhs, rhs);
#else
		return bitwise_cast<double>(static_cast<double>(lhs) != static_cast<double>(rhs) ? 0xFFFFFFFFFFFFFFFFu : 0x0000000000000000u);
#endif
	}
	static inline SIMD<double> operator<(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_pd(lhs, rhs, 1);
#elif SUPPORTS_SSE2
		return _mm_cmplt_pd(lhs, rhs);
#else
		return bitwise_cast<double>(static_cast<double>(lhs) < static_cast<double>(rhs) ? 0xFFFFFFFFFFFFFFFFu : 0x0000000000000000u);
#endif
	}
	static inline SIMD<double> operator<=(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_pd(lhs, rhs, 2);
#elif SUPPORTS_SSE2
		return _mm_cmple_pd(lhs, rhs);
#else
		return bitwise_cast<double>(static_cast<double>(lhs) <= static_cast<double>(rhs) ? 0xFFFFFFFFFFFFFFFFu : 0x0000000000000000u);
#endif
	}
	static inline SIMD<double> operator&(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_and_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_and_pd(lhs, rhs);
#else
		return bitwise_cast<double>(bitwise_cast<uint64_t>(lhs) & bitwise_cast<uint64_t>(rhs));
#endif
	}
	static inline SIMD<double> operator|(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_or_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_or_pd(lhs, rhs);
#else
		return bitwise_cast<double>(bitwise_cast<uint64_t>(lhs) | bitwise_cast<uint64_t>(rhs));
#endif
	}
	static inline SIMD<double> operator^(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_xor_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_xor_pd(lhs, rhs);
#else
		return bitwise_cast<double>(bitwise_cast<uint64_t>(lhs) ^ bitwise_cast<uint64_t>(rhs));
#endif
	}
	static inline SIMD<double> operator~(SIMD<double> x) noexcept
	{
		return x == SIMD<double>(0.0);
	}
	static inline SIMD<double> operator!(SIMD<double> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics
	 */
	static inline SIMD<double> operator+(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_add_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_add_pd(lhs, rhs);
#else
		return static_cast<double>(lhs) + static_cast<double>(rhs);
#endif
	}

	static inline SIMD<double> operator-(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_sub_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_sub_pd(lhs, rhs);
#else
		return static_cast<double>(lhs) - static_cast<double>(rhs);
#endif
	}
	static inline SIMD<double> operator-(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_xor_pd(x, SIMD<double>(-0.0));
#elif SUPPORTS_SSE2
		return _mm_xor_pd(x, SIMD<double>(-0.0));
#else
		return -x;
#endif
	}

	static inline SIMD<double> operator*(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_mul_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_mul_pd(lhs, rhs);
#else
		return static_cast<double>(lhs) * static_cast<double>(rhs);
#endif
	}

	static inline SIMD<double> operator/(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_div_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_div_pd(lhs, rhs);
#else
		return static_cast<double>(lhs) / static_cast<double>(rhs);
#endif
	}

	/* Float vector functions */

	/**
	 * result = (mask == 0xFFFFFFFFFFFFFFFF) ? x : y
	 */
	static inline SIMD<double> select(SIMD<double> mask, SIMD<double> x, SIMD<double> y)
	{
#if SUPPORTS_AVX
		return _mm256_blendv_pd(y, x, mask);
#elif SUPPORTS_SSE41
			return _mm_blendv_pd(y, x, mask);
#elif SUPPORTS_SSE2
		return _mm_or_pd(_mm_and_pd(mask, x), _mm_andnot_pd(mask, y));
#else
		return (bitwise_cast<uint64_t>(static_cast<double>(mask)) == 0xFFFFFFFFFFFFFFFFu) ? x : y;
#endif
	}

	static inline SIMD<double> max(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_max_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_max_pd(lhs, rhs);
#else
		return std::max(static_cast<double>(lhs), static_cast<double>(rhs));
#endif
	}
	static inline SIMD<double> min(SIMD<double> lhs, SIMD<double> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_min_pd(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_min_pd(lhs, rhs);
#else
		return std::min(static_cast<double>(lhs), static_cast<double>(rhs));
#endif
	}
	static inline SIMD<double> abs(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_and_pd(x, _mm256_castsi256_pd(constant<0xFFFFFFFFu, 0x7FFFFFFFu>()));
#elif SUPPORTS_SSE2
		return _mm_and_pd(x, _mm_castsi128_pd(constant<0xFFFFFFFFu, 0x7FFFFFFFu>()));
#else
		return std::fabs(static_cast<double>(x));
#endif
	}
	static inline SIMD<double> sqrt(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_sqrt_pd(x);
#elif SUPPORTS_SSE2
		return _mm_sqrt_pd(x);
#else
		return std::sqrt(static_cast<double>(x));
#endif
	}
	static inline SIMD<double> rsqrt(SIMD<double> x) noexcept
	{
		return SIMD<double>::one() / sqrt(x);
	}
	static inline SIMD<double> rcp(SIMD<double> x) noexcept
	{
		return SIMD<double>::one() / x;
	}
	static inline SIMD<double> sgn(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		__m256d zero = _mm256_setzero_pd();
		__m256d positive = _mm256_and_pd(_mm256_cmp_pd(zero, x, 1), _mm256_set1_pd(1.0));
		__m256d negative = _mm256_and_pd(_mm256_cmp_pd(x, zero, 1), _mm256_set1_pd(-1.0));
		return _mm256_or_pd(positive, negative);
#elif SUPPORTS_SSE2
		__m128d zero = _mm_setzero_pd();
		__m128d positive = _mm_and_pd(_mm_cmpgt_pd(x, zero), _mm_set1_pd(1.0));
		__m128d negative = _mm_and_pd(_mm_cmplt_pd(x, zero), _mm_set1_pd(-1.0));
		return _mm_or_pd(positive, negative);
#else
		return static_cast<double>((static_cast<double>(x) > 0.0) - (static_cast<double>(x) < 0.0));
#endif
	}
	static inline SIMD<double> floor(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_floor_pd(x);
#elif SUPPORTS_SSE41
		return _mm_floor_pd(x);
#elif SUPPORTS_SSE2
		double tmp[2];
		x.store(tmp);
		tmp[0] = std::floor(tmp[0]);
		tmp[1] = std::floor(tmp[1]);
		return SIMD<double>(tmp);
#else
		return std::floor(static_cast<double>(x));
#endif
	}
	static inline SIMD<double> ceil(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_ceil_pd(x);
#elif SUPPORTS_SSE41
		return _mm_ceil_pd(x);
#elif SUPPORTS_SSE2
		double tmp[2];
		x.store(tmp);
		tmp[0] = std::ceil(tmp[0]);
		tmp[1] = std::ceil(tmp[1]);
		return SIMD<double>(tmp);
#else
		return std::ceil(static_cast<double>(x));
#endif
	}

	/*
	 * Fused multiply accumulate
	 */

	/* Calculates a * b + c */
	static inline SIMD<double> mul_add(SIMD<double> a, SIMD<double> b, SIMD<double> c) noexcept
	{
#if SUPPORTS_AVX and defined(__FMA__)
		return _mm256_fmadd_pd(a, b, c);
#else
		return a * b + c;
#endif
	}
	/* Calculates a * b - c */
	static inline SIMD<double> mul_sub(SIMD<double> a, SIMD<double> b, SIMD<double> c) noexcept
	{
#if SUPPORTS_AVX and defined(__FMA__)
		return _mm256_fmsub_pd(a, b, c);
#else
		return a * b - c;
#endif
	}
	/* Calculates - a * b + c */
	static inline SIMD<double> neg_mul_add(SIMD<double> a, SIMD<double> b, SIMD<double> c) noexcept
	{
#if SUPPORTS_AVX and defined(__FMA__)
		return _mm256_fnmadd_pd(a, b, c);
#else
		return -a * b + c;
#endif
	}
	/* Calculates - a * b - c */
	static inline SIMD<double> neg_mul_sub(SIMD<double> a, SIMD<double> b, SIMD<double> c) noexcept
	{
#if SUPPORTS_AVX and defined(__FMA__)
		return _mm256_fnmsub_pd(a, b, c);
#else
		return -a * b - c;
#endif
	}

	/*
	 * Horizontal functions
	 */

	static inline float horizontal_add(SIMD<double> x) noexcept
	{
#if SUPPORTS_SSE2
#  if SUPPORTS_AVX
		__m128d y = _mm_add_pd(get_low(x), get_high(x));
#  else
		__m128d y = x;
#  endif
		__m128d t1 = _mm_unpackhi_pd(y, y);
		__m128d t2 = _mm_add_pd(y, t1);
		return _mm_cvtsd_f64(t2);
#else
		return static_cast<double>(x);
#endif
	}
	static inline float horizontal_mul(SIMD<double> x) noexcept
	{
#if SUPPORTS_SSE2
#  if SUPPORTS_AVX
		__m128d y = _mm_mul_pd(get_low(x), get_high(x));
#  else
		__m128d y = x;
#  endif
		__m128d t1 = _mm_unpackhi_pd(y, y);
		__m128d t2 = _mm_mul_pd(y, t1);
		return _mm_cvtsd_f64(t2);
#else
		return static_cast<double>(x);
#endif
	}
	static inline float horizontal_min(SIMD<double> x) noexcept
	{
#if SUPPORTS_SSE2
#  if SUPPORTS_AVX
		__m128d y = _mm_min_pd(get_low(x), get_high(x));
#  else
		__m128d y = x;
#  endif
		__m128d t1 = _mm_unpackhi_pd(y, y);
		__m128d t2 = _mm_min_pd(y, t1);
		return _mm_cvtsd_f64(t2);
#else
		return static_cast<double>(x);
#endif
	}
	static inline float horizontal_max(SIMD<double> x) noexcept
	{
#if SUPPORTS_SSE2
#  if SUPPORTS_AVX
		__m128d y = _mm_max_pd(get_low(x), get_high(x));
#  else
		__m128d y = x;
#  endif
		__m128d t1 = _mm_unpackhi_pd(y, y);
		__m128d t2 = _mm_max_pd(y, t1);
		return _mm_cvtsd_f64(t2);
#else
		return static_cast<double>(x);
#endif
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP64_SSIMDHPP_ */
