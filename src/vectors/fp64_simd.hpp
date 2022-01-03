/*
 * fp64_simd.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP64_SSIMDHPP_
#define VECTORS_FP64_SSIMDHPP_

#include "generic_simd.hpp"

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

#if SUPPORTS_AVX
			static constexpr size_t length = 4;
#elif SUPPORTS_SSE2
			static constexpr size_t length = 2;
#else
			static constexpr size_t length = 1;
#endif

			SIMD() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			SIMD(const double *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const double *ptr, size_t num) noexcept
			{
				loadu(ptr, num);
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
#if SUPPORTS_AVX
			SIMD(__m256d x) noexcept
			{
				m_data = x;
			}
			SIMD(__m128d low, __m128d high) noexcept
			{
				m_data = _mm256_setr_m128d(low, high);
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
			SIMD(__m128d x) noexcept
			{
				m_data = x;
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
			void loadu(const double *ptr) noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				m_data = _mm256_loadu_pd(ptr);
#elif SUPPORTS_SSE2
				m_data = _mm_loadu_pd(ptr);
#else
				m_data = ptr[0];
#endif
			}
			void loadu(const double *ptr, size_t num) noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);
#if SUPPORTS_AVX
				if (num == length)
					m_data = _mm256_loadu_pd(ptr);
				else
				{
					if (num > length / 2)
						*this = SIMD<double>(_mm_loadu_pd(ptr), partial_load(ptr + length / 2, num - length / 2));
					else
						*this = SIMD<double>(partial_load(ptr, num), _mm_setzero_pd());
				}
#elif SUPPORTS_SSE2
				m_data = partial_load(ptr, num);
#else
				m_data = ptr[0];
#endif
			}
			void storeu(double *ptr) const noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				_mm256_storeu_pd(ptr, m_data);
#elif SUPPORTS_SSE2
				_mm_storeu_pd(ptr, m_data);
#else
				ptr[0] = m_data;
#endif
			}
			void storeu(double *ptr, size_t num) const noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);

#if SUPPORTS_AVX
				if (num == length)
					_mm256_storeu_pd(ptr, m_data);
				else
				{
					if (num > length / 2)
					{
						_mm_storeu_pd(ptr, get_low(m_data));
						partial_store(get_high(m_data), ptr + length / 2, num - length / 2);
					}
					else
						partial_store(get_low(m_data), ptr, num);
				}
#elif SUPPORTS_SSE2
				partial_store(m_data, ptr, num);
#else
				ptr[0] = m_data;
#endif
			}
			void insert(double value, size_t index) noexcept
			{
				assert(index < length);
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
				storeu(tmp);
				tmp[index] = value;
				loadu(tmp);
#else
				m_data = value;
#endif
			}
			double extract(size_t index) const noexcept
			{
				assert(index < length);
				double tmp[length];
				storeu(tmp);
				return tmp[index];
			}
			double operator[](size_t index) const noexcept
			{
				return extract(index);
			}

			static SIMD<double> zero() noexcept
			{
				return SIMD<double>(0.0);
			}
			static SIMD<double> one() noexcept
			{
				return SIMD<double>(1.0);
			}
			static SIMD<double> epsilon() noexcept
			{
				return SIMD<double>(std::numeric_limits<double>::epsilon());
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
	static inline SIMD<double> operator!(SIMD<double> x) noexcept
	{
		return x == SIMD<double>(0.0);
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
		return bitwise_cast<uint64_t>(static_cast<double>(mask) == 0xFFFFFFFFFFFFFFFFu ? x : y);
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
	static inline SIMD<double> approx_recipr(SIMD<double> x) noexcept
	{
		return SIMD<double>::one() / x;
	}
	static inline SIMD<double> approx_rsqrt(SIMD<double> x) noexcept
	{
		return SIMD<double>::one() / sqrt(x);
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
		return (static_cast<double>(x) > 0.0) - (static_cast<double>(x) < 0.0);
#endif
	}
	static inline SIMD<double> floor(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_floor_pd(x);
#elif SUPPORTS_SSE2
		return _mm_floor_pd(x);
#else
		return std::floor(static_cast<double>(x));
#endif
	}
	static inline SIMD<double> ceil(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_ceil_pd(x);
#elif SUPPORTS_SSE2
		return _mm_ceil_pd(x);
#else
		return std::ceil(static_cast<double>(x));
#endif
	}

	/* Horizontal functions */
	static inline double horizontal_add(SIMD<double> x) noexcept
	{
#if SUPPORTS_AVX
		__m128d t1 = get_low(x) + get_high(x);
		__m128d t2 = _mm_unpackhi_pd(t1, t1);
		__m128d t3 = _mm_add_pd(t1, t2);
		return _mm_cvtsd_f64(t3);
#elif SUPPORTS_SSE2
		__m128d t1 = _mm_unpackhi_pd(x, x);
		__m128d t2 = _mm_add_pd(x, t1);
		return _mm_cvtsd_f64(t2);
#else
		return static_cast<double>(x);
#endif
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP64_SSIMDHPP_ */
