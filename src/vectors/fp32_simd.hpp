/*
 * fp32_simd.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_FP32_SIMD_HPP_
#define VECTORS_FP32_SIMD_HPP_

#include "generic_simd.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
	template<>
	class SIMD<float>
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

#if SUPPORTS_AVX
			static constexpr int64_t length = 8;
#elif SUPPORTS_SSE2
			static constexpr int64_t length = 4;
#else
			static constexpr int64_t length = 1;
#endif

			SIMD() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			SIMD(const float *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const float *ptr, size_t num) noexcept
			{
				loadu(ptr, num);
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
#if SUPPORTS_AVX
			SIMD(__m256 x) noexcept
			{
				m_data = x;
			}
			SIMD(__m128 low, __m128 high) noexcept
			{
				m_data = _mm256_setr_m128(low, high);
			}
			SIMD<float>& operator=(__m256 x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m256() const noexcept
			{
				return m_data;
			}
#elif SUPPORTS_SSE2
			SIMD(__m128 x) noexcept
			{
				m_data = x;
			}
			SIMD<float>& operator=(__m128 x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m128() const noexcept
			{
				return m_data;
			}
#else
			operator float() const noexcept
			{
				return m_data;
			}
#endif
			void loadu(const float *ptr) noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				m_data = _mm256_loadu_ps(ptr);
#elif SUPPORTS_SSE2
				m_data = _mm_loadu_ps(ptr);
#else
				m_data = ptr[0];
#endif
			}
			void loadu(const float *ptr, size_t num) noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);
#if SUPPORTS_AVX
				if (num == length)
					m_data = _mm256_loadu_ps(ptr);
				else
				{
					if (num > length / 2)
						*this = SIMD<float>(_mm_loadu_ps(ptr), partial_load(ptr + length / 2, num - length / 2));
					else
						*this = SIMD<float>(partial_load(ptr, num), _mm_setzero_ps());
				}
#elif SUPPORTS_SSE2
				m_data = partial_load(ptr, num);
#else
				m_data = ptr[0];
#endif
			}
			void storeu(float *ptr) const noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				_mm256_storeu_ps(ptr, m_data);
#elif SUPPORTS_SSE2
				_mm_storeu_ps(ptr, m_data);
#else
				ptr[0] = m_data;
#endif
			}
			void storeu(float *ptr, size_t num) const noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);

#if SUPPORTS_AVX
				if (num == length)
					_mm256_storeu_ps(ptr, m_data);
				else
				{
					if (num > length / 2)
					{
						_mm_storeu_ps(ptr, get_low(m_data));
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
			void insert(float value, size_t index) noexcept
			{
				assert(index < length);
#if SUPPORTS_AVX
				__m256 tmp = _mm256_broadcast_ss(&value);
				switch (index)
				{
					case 0:
						m_data = _mm256_blend_ps(m_data, tmp, 1);
						break;
					case 1:
						m_data = _mm256_blend_ps(m_data, tmp, 2);
						break;
					case 2:
						m_data = _mm256_blend_ps(m_data, tmp, 4);
						break;
					case 3:
						m_data = _mm256_blend_ps(m_data, tmp, 8);
						break;
					case 4:
						m_data = _mm256_blend_ps(m_data, tmp, 16);
						break;
					case 5:
						m_data = _mm256_blend_ps(m_data, tmp, 32);
						break;
					case 6:
						m_data = _mm256_blend_ps(m_data, tmp, 64);
						break;
					default:
						m_data = _mm256_blend_ps(m_data, tmp, 128);
						break;
				}
#elif SUPPORTS_SSE2
				float tmp[4];
				storeu(tmp);
				tmp[index] = value;
				loadu(tmp);
#else
				m_data = value;
#endif
			}
			float extract(size_t index) const noexcept
			{
				assert(index < length);
				float tmp[length];
				storeu(tmp);
				return tmp[index];
			}
			float operator[](size_t index) const noexcept
			{
				return extract(index);
			}

			static SIMD<float> zero() noexcept
			{
				return SIMD<float>(0.0f);
			}
			static SIMD<float> one() noexcept
			{
				return SIMD<float>(1.0f);
			}
			static SIMD<float> epsilon() noexcept
			{
				return SIMD<float>(std::numeric_limits<float>::epsilon());
			}
	};

	/*
	 * Float vector logical operations.
	 * Return vector of floats, either 0x00000000 (0.0f) for false, or 0xFFFFFFFF (-nan) for true.
	 */
	static inline SIMD<float> operator==(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_ps(lhs, rhs, 0);
#elif SUPPORTS_SSE2
		return _mm_cmpeq_ps(lhs, rhs);
#else
		return bitwise_cast<float>(static_cast<float>(lhs) == static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
#endif
	}
	static inline SIMD<float> operator!=(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_ps(lhs, rhs, 4);
#elif SUPPORTS_SSE2
		return _mm_cmpneq_ps(lhs, rhs);
#else
		return bitwise_cast<float>(static_cast<float>(lhs) != static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
#endif
	}
	static inline SIMD<float> operator<(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_ps(lhs, rhs, 1);
#elif SUPPORTS_SSE2
		return _mm_cmplt_ps(lhs, rhs);
#else
		return bitwise_cast<float>(static_cast<float>(lhs) < static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
#endif
	}
	static inline SIMD<float> operator<=(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmp_ps(lhs, rhs, 2);
#elif SUPPORTS_SSE2
		return _mm_cmple_ps(lhs, rhs);
#else
		return bitwise_cast<float>(static_cast<float>(lhs) <= static_cast<float>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
#endif
	}
	static inline SIMD<float> operator&(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_and_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_and_ps(lhs, rhs);
#else
		return bitwise_cast<float>(bitwise_cast<uint32_t>(lhs) & bitwise_cast<uint32_t>(rhs));
#endif
	}
	static inline SIMD<float> operator|(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_or_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_or_ps(lhs, rhs);
#else
		return bitwise_cast<float>(bitwise_cast<uint32_t>(lhs) | bitwise_cast<uint32_t>(rhs));
#endif
	}
	static inline SIMD<float> operator^(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_xor_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_xor_ps(lhs, rhs);
#else
		return bitwise_cast<float>(bitwise_cast<uint32_t>(lhs) ^ bitwise_cast<uint32_t>(rhs));
#endif
	}
	static inline SIMD<float> operator~(SIMD<float> x) noexcept
	{
		return x == SIMD<float>(0.0f);
	}
	static inline SIMD<float> operator!(SIMD<float> x) noexcept
	{
		return ~x;
	}

	/*
	 * Float vector arithmetics.
	 */
	static inline SIMD<float> operator+(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_add_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_add_ps(lhs, rhs);
#else
		return static_cast<float>(lhs) + static_cast<float>(rhs);
#endif
	}

	static inline SIMD<float> operator-(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_sub_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_sub_ps(lhs, rhs);
#else
		return static_cast<float>(lhs) - static_cast<float>(rhs);
#endif
	}
	static inline SIMD<float> operator-(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_xor_ps(x, SIMD<float>(-0.0f));
#elif SUPPORTS_SSE2
		return _mm_xor_ps(x, SIMD<float>(-0.0f));
#else
		return -x;
#endif
	}

	static inline SIMD<float> operator*(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_mul_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_mul_ps(lhs, rhs);
#else
		return static_cast<float>(lhs) * static_cast<float>(rhs);
#endif
	}

	static inline SIMD<float> operator/(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_div_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_div_ps(lhs, rhs);
#else
		return static_cast<float>(lhs) / static_cast<float>(rhs);
#endif
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline SIMD<float> select(SIMD<float> mask, SIMD<float> x, SIMD<float> y)
	{
#if SUPPORTS_AVX
		return _mm256_blendv_ps(y, x, mask);
#elif SUPPORTS_SSE41
			return _mm_blendv_ps(y, x, mask);
#elif SUPPORTS_SSE2
		return _mm_or_ps(_mm_and_ps(mask, x), _mm_andnot_ps(mask, y));
#else
		return bitwise_cast<uint32_t>(static_cast<float>(mask) == 0xFFFFFFFFu ? x : y);
#endif
	}

	/* Float vector functions */
	static inline SIMD<float> max(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_max_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_max_ps(lhs, rhs);
#else
		return std::max(static_cast<float>(lhs), static_cast<float>(rhs));
#endif
	}
	static inline SIMD<float> min(SIMD<float> lhs, SIMD<float> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_min_ps(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_min_ps(lhs, rhs);
#else
		return std::min(static_cast<float>(lhs), static_cast<float>(rhs));
#endif
	}
	static inline SIMD<float> abs(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_and_ps(x, _mm256_castsi256_ps(constant<0x7FFFFFFFu>()));
#elif SUPPORTS_SSE2
		return _mm_and_ps(x, _mm_castsi128_ps(constant<0x7FFFFFFFu>()));
#else
		return std::fabs(static_cast<float>(x));
#endif
	}
	static inline SIMD<float> sqrt(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_sqrt_ps(x);
#elif SUPPORTS_SSE2
		return _mm_sqrt_ps(x);
#else
		return std::sqrt(static_cast<float>(x));
#endif
	}
	static inline SIMD<float> approx_recipr(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_rcp_ps(x);
#elif SUPPORTS_SSE2
		return _mm_rcp_ps(x);
#else
		return 1.0f / static_cast<float>(x);
#endif
	}
	static inline SIMD<float> approx_rsqrt(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_rsqrt_ps(x);
#elif SUPPORTS_SSE2
		return _mm_rsqrt_ps(x);
#else
		return 1.0f / std::sqrt(static_cast<float>(x));
#endif
	}
	static inline SIMD<float> sgn(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		__m256 zero = _mm256_setzero_ps();
		__m256 positive = _mm256_and_ps(_mm256_cmp_ps(zero, x, 1), _mm256_set1_ps(1.0f));
		__m256 negative = _mm256_and_ps(_mm256_cmp_ps(x, zero, 1), _mm256_set1_ps(-1.0f));
		return _mm256_or_ps(positive, negative);
#elif SUPPORTS_SSE2
		__m128 zero = _mm_setzero_ps();
		__m128 positive = _mm_and_ps(_mm_cmpgt_ps(x, zero), _mm_set1_ps(1.0f));
		__m128 negative = _mm_and_ps(_mm_cmplt_ps(x, zero), _mm_set1_ps(-1.0f));
		return _mm_or_ps(positive, negative);
#else
		return (static_cast<float>(x) > 0.0f) - (static_cast<float>(x) < 0.0f);
#endif
	}
	static inline SIMD<float> floor(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_floor_ps(x);
#elif SUPPORTS_SSE2
		return _mm_floor_ps(x);
#else
		return std::floorf(static_cast<float>(x));
#endif
	}
	static inline SIMD<float> ceil(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_ceil_ps(x);
#elif SUPPORTS_SSE2
		return _mm_ceil_ps(x);
#else
		return std::ceilf(static_cast<float>(x));
#endif
	}

	/* Horizontal functions */

	static inline float horizontal_add(SIMD<float> x) noexcept
	{
#if SUPPORTS_AVX
		__m128 t1 = get_low(x) + get_high(x);
		__m128 t2 = _mm_hadd_ps(t1, t1);
		__m128 t3 = _mm_hadd_ps(t2, t2);
		return _mm_cvtss_f32(t3);
#elif  SUPPORTS_SSE3
		__m128 t1 = _mm_hadd_ps(x, x);
		__m128 t2 = _mm_hadd_ps(t1, t1);
		return _mm_cvtss_f32(t2);
#elif SUPPORTS_SSE2
		__m128 t1 = _mm_movehl_ps(x, x);
		__m128 t2 = _mm_add_ps(x, t1);
		__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
		__m128 t4 = _mm_add_ss(t2, t3);
		return _mm_cvtss_f32(t4);
#else
		return static_cast<float>(x);
#endif
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_FP32_SIMD_HPP_ */
