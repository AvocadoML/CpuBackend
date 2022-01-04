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
		};
	}
}

namespace SIMD_NAMESPACE
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
		return avocado::backend::float16();
#endif
	}

	template<>
	class SIMD<avocado::backend::float16>
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
			SIMD(const avocado::backend::float16 *ptr) noexcept
			{
				loadu(ptr);
			}
			SIMD(const float *ptr, size_t num) noexcept
			{
				loadu(ptr, num);
			}
			SIMD(const avocado::backend::float16 *ptr, size_t num) noexcept
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
			void loadu(const avocado::backend::float16 *ptr) noexcept
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
			void loadu(const float *ptr, size_t num) noexcept
			{
				m_data.loadu(ptr, num);
			}
			void loadu(const avocado::backend::float16 *ptr, size_t num) noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);
#if SUPPORTS_AVX and SUPPORTS_FP16
				__m128i tmp = partial_load(ptr, num * sizeof(avocado::backend::float16));
				m_data = SIMD<float>(_mm256_cvtph_ps(tmp));
#elif SUPPORTS_SSE2 and SUPPORTS_FP16
				__m128i tmp = partial_load(ptr, num * sizeof(avocado::backend::float16));
				m_data = _mm_cvtph_ps(tmp);
#else
					m_data = float16_to_float(ptr[0]);
#endif
			}
			void storeu(float *ptr) const noexcept
			{
				m_data.storeu(ptr);
			}
			void storeu(avocado::backend::float16 *ptr) const noexcept
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
			void storeu(float *ptr, size_t num) const noexcept
			{
				m_data.storeu(ptr, num);
			}
			void storeu(avocado::backend::float16 *ptr, size_t num) const noexcept
			{
				assert(ptr != nullptr);
				assert(num <= length);
#if SUPPORTS_AVX and SUPPORTS_FP16
				__m128i tmp = _mm256_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				partial_store(tmp, ptr, num * sizeof(avocado::backend::float16));
#elif SUPPORTS_SSE2 and SUPPORTS_FP16
				__m128i tmp = _mm_cvtps_ph(m_data, _MM_FROUND_NO_EXC);
				partial_store(tmp, ptr, num * sizeof(avocado::backend::float16));
#else
				ptr[0] = float_to_float16(m_data);
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

#endif /* VECTORS_FP16_SIMD_HPP_ */
