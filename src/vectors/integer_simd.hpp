/*
 * integer_simd.hpp
 *
 *  Created on: Nov 17, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_INTEGER_SIMD_HPP_
#define VECTORS_INTEGER_SIMD_HPP_

#include "generic_simd.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
#if SUPPORTS_AVX
	template<typename T>
	static inline __m256i set1(T x) noexcept
	{
		switch (sizeof(T))
		{
			case 1:
				return _mm256_set1_epi8(x);
			case 2:
				return _mm256_set1_epi16(x);
			case 4:
				return _mm256_set1_epi32(x);
			case 8:
				return _mm256_set1_epi64x(x);
			default:
				return _mm256_setzero_si256();
		}
	}
#elif SUPPORTS_SSE2
	template<typename T>
	static inline __m128i set1(T x) noexcept
	{
		switch (sizeof(T))
		{
			case 1:
				return _mm_set1_epi8(x);
			case 2:
				return _mm_set1_epi16(x);
			case 4:
				return _mm_set1_epi32(x);
			case 8:
				return _mm_set1_epi64x(x);
			default:
				return _mm_setzero_si128();
		}
	}
#else
	template<typename T>
	static inline T set1(T x) noexcept
	{
		return x;
	}
#endif

	template<typename T>
	class SIMD<T, typename std::enable_if<std::is_integral<T>::value, T>::type>
	{
		private:
#if SUPPORTS_AVX
			__m256i m_data;
#elif SUPPORTS_SSE2
			__m128i m_data;
#else
			T m_data;
#endif
		public:
			static constexpr int length = sizeof(m_data) / sizeof(T);

			SIMD() noexcept // @suppress("Class members should be properly initialized")
			{
			}
			SIMD(T x) noexcept :
					m_data(set1(x))
			{
			}
			SIMD(const T *ptr) noexcept // @suppress("Class members should be properly initialized")
			{
				loadu(ptr);
			}
			SIMD(const T *ptr, int num) noexcept // @suppress("Class members should be properly initialized")
			{
				loadu(ptr, num);
			}
#if SUPPORTS_AVX
			SIMD(__m256i x) noexcept :
					m_data(x)
			{
			}
			SIMD(__m128i low) noexcept :
					m_data(_mm256_setr_m128i(low, _mm_setzero_si128()))
			{
			}
			SIMD(__m128i low, __m128i high) noexcept :
					m_data(_mm256_setr_m128i(low, high))
			{
			}
			SIMD& operator=(__m256i x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m256i() const noexcept
			{
				return m_data;
			}
#elif SUPPORTS_SSE2
			SIMD(__m128i x) noexcept:
					m_data(x)
			{
			}
			SIMD& operator=(__m128i x) noexcept
			{
				m_data = x;
				return *this;
			}
			operator __m128i() const noexcept
			{
				return m_data;
			}
#else
			operator T() const noexcept
			{
				return m_data;
			}
#endif
			void loadu(const T *ptr) noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				m_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
#elif SUPPORTS_SSE2
				m_data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
#else
				m_data = ptr[0];
#endif
			}
			void loadu(const T *ptr, int num) noexcept
			{
				assert(ptr != nullptr);
				assert(num >= 0 && num <= length);
#if SUPPORTS_AVX
				if (num == length)
					m_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
				else
				{
					if (num > length / 2)
						*this = SIMD<T>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)),
								partial_load(ptr + (length / 2), sizeof(T) * (num - length / 2)));
					else
						*this = SIMD<T>(partial_load(ptr, sizeof(T) * num));
				}
#elif SUPPORTS_SSE2
				m_data = partial_load(ptr, sizeof(T) * num);
#else
				m_data = ptr[0];
#endif
			}
			void storeu(T *ptr) const noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), m_data);
#elif SUPPORTS_SSE2
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), m_data);
#else
				ptr[0] = m_data;
#endif
			}
			void storeu(T *ptr, int num) const noexcept
			{
				assert(ptr != nullptr);
				assert(num >= 0 && num <= length);
#if SUPPORTS_AVX
				if (num == length)
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), m_data);
				else
				{
					if (num > length / 2)
					{
						_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), get_low(m_data));
						partial_store(get_high(m_data), ptr + length / 2, sizeof(T) * (num - length / 2));
					}
					else
						partial_store(get_low(m_data), ptr, sizeof(T) * num);
				}
#elif SUPPORTS_SSE2
				partial_store(m_data, ptr, sizeof(T) * num);
#else
				ptr[0] = m_data;
#endif
			}
			void insert(T value, int index) noexcept
			{
				assert(index >= 0 && index < length);
				T tmp[length];
				storeu(tmp);
				tmp[index] = value;
				loadu(tmp);
			}
			T extract(int index) const noexcept
			{
				assert(index >= 0 && index < length);
				T tmp[length];
				storeu(tmp);
				return tmp[index];
			}
			T operator[](int index) const noexcept
			{
				return extract(index);
			}

			static SIMD<T> zero() noexcept
			{
				return SIMD<T>(0);
			}
			static SIMD<T> one() noexcept
			{
				return SIMD<T>(1);
			}
			static SIMD<T> epsilon() noexcept
			{
				return zero();
			}
	};

	/*
	 * Int32 vector logical operations.
	 * Return vector of int32, either 0x00000000 for false, or 0xFFFFFFFF for true.
	 */
	static inline SIMD<int32_t> operator&(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_and_si256(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_and_si128(lhs, rhs);
#else
		return SIMD<int32_t>(bitwise_cast<uint32_t>(lhs) & bitwise_cast<uint32_t>(rhs));
#endif
	}
	static inline SIMD<int32_t> operator|(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_or_si256(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_or_si128(lhs, rhs);
#else
		return SIMD<int32_t>(bitwise_cast<uint32_t>(lhs) | bitwise_cast<uint32_t>(rhs));
#endif
	}
	static inline SIMD<int32_t> operator^(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_xor_si256(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_xor_si128(lhs, rhs);
#else
		return SIMD<int32_t>(bitwise_cast<uint32_t>(lhs) ^ bitwise_cast<uint32_t>(rhs));
#endif
	}
	static inline SIMD<int32_t> operator~(SIMD<int32_t> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_xor_si256(x, constant<0xFFFFFFFFu>());
#elif SUPPORTS_SSE2
		return _mm_xor_si128(x, constant<0xFFFFFFFFu>());
#else
		return SIMD<int32_t>(~static_cast<uint32_t>(x));
#endif
	}
	static inline SIMD<int32_t> operator!(SIMD<int32_t> x) noexcept
	{
		return ~x;
	}
	static inline SIMD<int32_t> operator==(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmpeq_epi32(lhs, rhs);
#elif SUPPORTS_SSE2
		return _mm_cmpeq_epi32(lhs, rhs);
#else
		return SIMD<int32_t>(static_cast<int32_t>(lhs) == static_cast<int32_t>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
#endif
	}
	static inline SIMD<int32_t> operator!=(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
		return !(lhs == rhs);
	}
	static inline SIMD<int32_t> operator<(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cmpgt_epi32(rhs, lhs);
#elif SUPPORTS_SSE2
		return _mm_cmpgt_epi32(rhs, lhs);
#else
		return SIMD<int32_t>(static_cast<int32_t>(lhs) < static_cast<int32_t>(rhs) ? 0xFFFFFFFFu : 0x00000000u);
#endif
	}
	static inline SIMD<int32_t> operator<=(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
		return (lhs < rhs) | (lhs == rhs);
	}

	/* 32 bit integers arithmetics */
	static inline SIMD<int32_t> operator+(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_add_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_add_epi32(get_low(lhs), get_low(rhs)), _mm_add_epi32(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_add_epi32(lhs, rhs);
#else
		return static_cast<int32_t>(lhs) + static_cast<int32_t>(rhs);
#endif
	}
	static inline SIMD<int32_t> operator-(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_sub_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_sub_epi32(get_low(lhs), get_low(rhs)), _mm_sub_epi32(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_sub_epi32(lhs, rhs);
#else
		return static_cast<int32_t>(lhs) - static_cast<int32_t>(rhs);
#endif
	}
	static inline SIMD<int32_t> operator-(SIMD<int32_t> x) noexcept
	{
		return SIMD<int32_t>::zero() - x;
	}
	static inline SIMD<int32_t> operator*(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_mullo_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_mullo_epi32(get_low(lhs), get_low(rhs)), _mm_mullo_epi32(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE41
		return _mm_mullo_epi32(lhs, rhs);
#elif SUPPORTS_SSE2
		int32_t lhs_tmp[lhs.length];
		int32_t rhs_tmp[rhs.length];
		lhs.storeu(lhs_tmp);
		rhs.storeu(rhs_tmp);
		for (int i = 0; i < lhs.length; i++)
			lhs_tmp[i] *= rhs_tmp[i];
		return SIMD<int32_t>(lhs_tmp);
#else
		return static_cast<int32_t>(lhs) * static_cast<int32_t>(rhs);
#endif
	}
	static inline SIMD<int32_t> operator/(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
		exit(-1); // integer division is not supported right now
	}

	/* Bit shifts */
	static inline SIMD<int32_t> operator>>(SIMD<int32_t> lhs, int32_t rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_slli_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_slli_epi32(get_low(lhs), rhs), _mm_slli_epi32(get_high(lhs), rhs));
#elif SUPPORTS_SSE2
		return _mm_slli_epi32(lhs, rhs);
#else
		return static_cast<int32_t>(lhs) >> rhs;
#endif
	}
	static inline SIMD<int32_t> operator<<(SIMD<int32_t> lhs, int32_t rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_srli_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_srli_epi32(get_low(lhs), rhs), _mm_srli_epi32(get_high(lhs), rhs));
#elif SUPPORTS_SSE2
		return _mm_srli_epi32(lhs, rhs);
#else
		return static_cast<int32_t>(lhs) << rhs;
#endif
	}

	/**
	 * result = (mask == 0xFFFFFFFF) ? x : y
	 */
	static inline SIMD<int32_t> select(SIMD<int32_t> mask, SIMD<int32_t> x, SIMD<int32_t> y)
	{
#if SUPPORTS_AVX
		return _mm256_blendv_epi8(y, x, mask);
#elif SUPPORTS_SSE41
		return _mm_blendv_epi8(y, x, mask);
#elif SUPPORTS_SSE2
		return _mm_or_ps(_mm_and_ps(mask, x), _mm_andnot_ps(mask, y));
#else
		return bitwise_cast<uint32_t>(static_cast<float>(mask) == 0xFFFFFFFFu ? x : y);
#endif
	}
	static inline SIMD<int32_t> max(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_max_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_max_epi32(get_low(lhs), get_low(rhs)), _mm_max_epi32(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_max_epi32(lhs, rhs);
#else
		return std::max(static_cast<int32_t>(lhs), static_cast<int32_t>(rhs));
#endif
	}
	static inline SIMD<int32_t> min(SIMD<int32_t> lhs, SIMD<int32_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_min_epi32(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_min_epi32(get_low(lhs), get_low(rhs)), _mm_min_epi32(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_min_epi32(lhs, rhs);
#else
		return std::min(static_cast<int32_t>(lhs), static_cast<int32_t>(rhs));
#endif
	}
	static inline SIMD<int32_t> abs(SIMD<int32_t> x) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_abs_epi32(x);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_abs_epi32(get_low(x)), _mm_abs_epi32(get_high(x)));
#elif SUPPORTS_SSE2
		return _mm_abs_epi32(x);
#else
		return std::abs(static_cast<int32_t>(x));
#endif
	}
	static inline SIMD<int32_t> sgn(SIMD<int32_t> x) noexcept
	{
#if SUPPORTS_AVX2
		__m256i zero = _mm256_setzero_si256();
		__m256i positive = _mm256_and_si256(_mm256_cmpgt_epi32(x, zero), _mm256_set1_epi32(1));
		__m256i negative = _mm256_and_si256(_mm256_cmpgt_epi32(zero, x), _mm256_set1_epi32(-1));
		return _mm256_or_si256(positive, negative);
#elif SUPPORTS_AVX
		__m128i zero = _mm_setzero_si128();
		__m128i xlo = get_low(x);
		__m128i xhi = get_high(x);
		__m256i positive = _mm256_setr_m128i(_mm_cmpgt_epi32(xlo, zero), _mm_cmpgt_epi32(xhi, zero));
		__m256i negative = _mm256_setr_m128i(_mm_cmpgt_epi32(zero, xlo), _mm_cmpgt_epi32(zero, xhi));
		positive = _mm256_and_si256(positive, _mm256_set1_epi32(1));
		negative = _mm256_and_si256(negative, _mm256_set1_epi32(-1));
		return _mm256_or_si256(positive, negative);
#elif SUPPORTS_SSE2
		__m128i zero = _mm_setzero_si128();
		__m128i positive = _mm_and_si128(_mm_cmpgt_epi32(x, zero), _mm_set1_epi32(1));
		__m128i negative = _mm_and_si128(_mm_cmpgt_epi32(zero, x), _mm_set1_epi32(-1));
		return _mm_or_si128(positive, negative);
#else
		return (static_cast<int32_t>(x) > 0) - (static_cast<int32_t>(x) < 0);
#endif
	}

	/* 8 bit integers arithmetics */
	static inline SIMD<int8_t> operator+(SIMD<int8_t> lhs, SIMD<int8_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_add_epi8(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int8_t>(_mm_add_epi8(get_low(lhs), get_low(rhs)), _mm_add_epi8(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_add_epi8(lhs, rhs);
#else
		return static_cast<int8_t>(lhs) + static_cast<int8_t>(rhs);
#endif
	}
	static inline SIMD<int8_t> operator-(SIMD<int8_t> lhs, SIMD<int8_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_sub_epi8(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int8_t>(_mm_sub_epi8(get_low(lhs), get_low(rhs)), _mm_sub_epi8(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_sub_epi8(lhs, rhs);
#else
		return static_cast<int8_t>(lhs) - static_cast<int8_t>(rhs);
#endif
	}
	static inline SIMD<int8_t> operator-(SIMD<int8_t> x) noexcept
	{
		return SIMD<int8_t>(static_cast<int8_t>(0)) - x;
	}

	/**
	 * result = (mask == 0xFF) ? x : y
	 */
	static inline SIMD<int8_t> select(SIMD<int8_t> mask, SIMD<int8_t> x, SIMD<int8_t> y)
	{
#if SUPPORTS_AVX
		return _mm256_blendv_epi8(y, x, mask);
#elif SUPPORTS_SSE41
		return _mm_blendv_epi8(y, x, mask);
#elif SUPPORTS_SSE2
		return _mm_or_ps(_mm_and_ps(mask, x), _mm_andnot_ps(mask, y));
#else
		return bitwise_cast<uint8_t>(static_cast<float>(mask) == 0xFFu ? x : y);
#endif
	}
	static inline SIMD<int8_t> max(SIMD<int8_t> lhs, SIMD<int8_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_max_epi8(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int8_t>(_mm_max_epi8(get_low(lhs), get_low(rhs)), _mm_max_epi8(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_max_epi8(lhs, rhs);
#else
		return std::max(static_cast<int8_t>(lhs), static_cast<int8_t>(rhs));
#endif
	}
	static inline SIMD<int8_t> min(SIMD<int8_t> lhs, SIMD<int8_t> rhs) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_min_epi8(lhs, rhs);
#elif SUPPORTS_AVX
		return SIMD<int8_t>(_mm_min_epi8(get_low(lhs), get_low(rhs)), _mm_min_epi8(get_high(lhs), get_high(rhs)));
#elif SUPPORTS_SSE2
		return _mm_min_epi8(lhs, rhs);
#else
		return std::min(static_cast<int8_t>(lhs), static_cast<int8_t>(rhs));
#endif
	}
	static inline SIMD<int8_t> abs(SIMD<int8_t> x) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_abs_epi8(x);
#elif SUPPORTS_AVX
		return SIMD<int8_t>(_mm_abs_epi8(get_low(x)), _mm_abs_epi8(get_high(x)));
#elif SUPPORTS_SSE2
		return _mm_abs_epi8(x);
#else
		return std::abs(static_cast<int8_t>(x));
#endif
	}
	static inline SIMD<int8_t> sgn(SIMD<int8_t> x) noexcept
	{
#if SUPPORTS_AVX2
		__m256i zero = _mm256_setzero_si256();
		__m256i positive = _mm256_and_si256(_mm256_cmpgt_epi8(x, zero), _mm256_set1_epi8(1));
		__m256i negative = _mm256_and_si256(_mm256_cmpgt_epi8(zero, x), _mm256_set1_epi8(-1));
		return _mm256_or_si256(positive, negative);
#elif SUPPORTS_AVX
		__m128i zero = _mm_setzero_si128();
		__m128i xlo = get_low(x);
		__m128i xhi = get_high(x);
		__m256i positive = _mm256_setr_m128i(_mm_cmpgt_epi8(xlo, zero), _mm_cmpgt_epi8(xhi, zero));
		__m256i negative = _mm256_setr_m128i(_mm_cmpgt_epi8(zero, xlo), _mm_cmpgt_epi8(zero, xhi));
		positive = _mm256_and_si256(positive, _mm256_set1_epi8(1));
		negative = _mm256_and_si256(negative, _mm256_set1_epi8(-1));
		return _mm256_or_si256(positive, negative);
#elif SUPPORTS_SSE2
		__m128i zero = _mm_setzero_si128();
		__m128i positive = _mm_and_si128(_mm_cmpgt_epi8(x, zero), _mm_set1_epi8(1));
		__m128i negative = _mm_and_si128(_mm_cmpgt_epi8(zero, x), _mm_set1_epi8(-1));
		return _mm_or_si128(positive, negative);
#else
		return (static_cast<int8_t>(x) > 0) - (static_cast<int8_t>(x) < 0);
#endif
	}

	static inline SIMD<int32_t> dp4a(SIMD<int8_t> x, SIMD<int8_t> y) noexcept
	{
#if SUPPORTS_AVX2
		return _mm256_madd_epi16(x, y);
#elif SUPPORTS_AVX
		return SIMD<int32_t>(_mm_madd_epi16(get_low(x), get_low(y)), _mm_madd_epi16(get_high(x), get_high(y)));
#elif SUPPORTS_SSE2
		return _mm_madd_epi16(x, y);
#else
		return static_cast<int32_t>(static_cast<int8_t>(x)) * static_cast<int32_t>(static_cast<int8_t>(y));
#endif
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_INTEGER_SIMD_HPP_ */
