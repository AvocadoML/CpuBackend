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

#include <cstring>

namespace avocado
{
	namespace backend
	{
		struct bfloat16
		{
				uint16_t m_data;
		};
	}
}

namespace SIMD_NAMESPACE
{
#if SUPPORTS_AVX

	static inline __m256 bfloat16_to_float(__m128i x) noexcept
	{
#if SUPPORTS_AVX2
		__m256i tmp = _mm256_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
		tmp = _mm256_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
#else
		__m128i tmp_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
		__m128i tmp_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), x); // pad upper half with zeros
		__m256i tmp = _mm256_setr_m128i(tmp_lo, tmp_hi); // combine two halves
#endif /* SUPPORTS_AVX2 */
		return _mm256_castsi256_ps(tmp);
	}
	static inline __m128i float_to_bfloat16(__m256 x) noexcept
	{
#if SUPPORTS_AVX2
		__m256i tmp = _mm256_srli_epi32(_mm256_castps_si256(x), 16); // shift right by 16 bits while shifting in zeros
		return _mm_packs_epi32(get_low(tmp), get_high(tmp)); // pack 32 bits into 16 bits
#else
		__m128i tmp_lo = _mm_srli_epi32(_mm_castps_si128(get_low(x)), 16); // shift right by 16 bits while shifting in zeros
		__m128i tmp_hi = _mm_srli_epi32(_mm_castps_si128(get_high(x)), 16); // shift right by 16 bits while shifting in zeros
		return _mm_packs_epi32(tmp_lo, tmp_hi); // pack 32 bits into 16 bits
#endif

	}

#elif SUPPORTS_SSE2 /* if __AVX__ is not defined */

	static inline __m128 bfloat16_to_float(__m128i x) noexcept
	{
#if SUPPORTS_SSE41
			__m128i tmp = _mm_cvtepu16_epi32(x); // extend 16 bits with zeros to 32 bits
			tmp = _mm_slli_epi32(tmp, 16); // shift left by 16 bits while shifting in zeros
#else
		__m128i tmp = _mm_unpacklo_epi16(_mm_setzero_si128(), x); // pad lower half with zeros
#endif /* defined(__SSE4_1__) */
		return _mm_castsi128_ps(tmp);
	}
	static inline __m128i float_to_bfloat16(__m128 x) noexcept
	{
		__m128i tmp = _mm_srli_epi32(_mm_castps_si128(x), 16); // shift right by 16 bits while shifting in zeros
		return _mm_packs_epi32(tmp, _mm_setzero_si128()); // pack 32 bits into 16 bits
	}

#else /* if __SSE2__ is not defined */

		static inline float bfloat16_to_float(avocado::backend::bfloat16 x) noexcept
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
#endif

	template<>
	class SIMD<avocado::backend::bfloat16>
	{
		private:
			SIMD<float> m_data;
		public:
			static constexpr int64_t length = SIMD<float>::length;

			SIMD() noexcept
			{
			}
			SIMD(const float *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const avocado::backend::bfloat16 *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const float *ptr, size_t num) noexcept
			{
				loadu(ptr, num);
			}
			SIMD(const avocado::backend::bfloat16 *ptr, size_t num) noexcept
			{
				loadu(ptr, num);
			}
			SIMD(SIMD<float> x) noexcept :
					m_data(x)
			{
			}
			SIMD(float x) noexcept
			{
				m_data = SIMD<float>(x);
			}
			operator SIMD<float>() const noexcept
			{
				return m_data;
			}
			void loadu(const float *ptr) noexcept
			{
				m_data.loadu(ptr);
			}
			void loadu(const avocado::backend::bfloat16 *ptr) noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				__m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
				m_data = bfloat16_to_float(tmp);
#elif SUPPORTS_SSE2
				__m128i tmp = _mm_loadu_si64(ptr);
				m_data = bfloat16_to_float(tmp);
#else
					m_data = bfloat16_to_float(ptr[0]);
#endif
			}
			void loadu(const float *ptr, size_t num) noexcept
			{
				m_data.loadu(ptr, num);
			}
			void loadu(const avocado::backend::bfloat16 *ptr, size_t num) noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);
#if SUPPORTS_SSE2
				__m128i tmp = partial_load(ptr, num * sizeof(avocado::backend::bfloat16));
				m_data = bfloat16_to_float(tmp);
#else
					m_data = bfloat16_to_float(ptr[0]);
#endif
			}
			void storeu(float *ptr) const noexcept
			{
				m_data.storeu(ptr);
			}
			void storeu(avocado::backend::bfloat16 *ptr) const noexcept
			{
				assert(ptr != nullptr);
#if SUPPORTS_AVX
				_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), float_to_bfloat16(m_data));
#elif SUPPORTS_SSE2
				_mm_storeu_si64(reinterpret_cast<__m128i*>(ptr), float_to_bfloat16(m_data));
#else
					ptr[0] = float_to_bfloat16(static_cast<float>(m_data));
#endif
			}
			void storeu(float *ptr, size_t num) const noexcept
			{
				m_data.storeu(ptr, num);
			}
			void storeu(avocado::backend::bfloat16 *ptr, size_t num) const noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);
#if SUPPORTS_AVX
				partial_store(float_to_bfloat16(m_data), ptr, num * sizeof(avocado::backend::bfloat16));
#elif SUPPORTS_SSE2
				partial_store(float_to_bfloat16(m_data), ptr, num * sizeof(avocado::backend::bfloat16));
#else
					ptr[0] = float_to_bfloat16(static_cast<float>(m_data));
#endif
			}
			void insert(float value, size_t index) noexcept
			{
				m_data.insert(value, index);
			}
			float extract(size_t index) const noexcept
			{
				return m_data.extract(index);
			}
			float operator[](size_t index) const noexcept
			{
				return extract(index);
			}
	};

}

#endif /* VECTORS_BF16_SIMD_HPP_ */
