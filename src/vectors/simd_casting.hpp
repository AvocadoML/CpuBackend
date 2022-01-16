/*
 * simd_casting.hpp
 *
 *  Created on: Jan 4, 2022
 *      Author: maciek
 */

#ifndef VECTORS_SIMD_CASTING_HPP_
#define VECTORS_SIMD_CASTING_HPP_

#include "generic_simd.hpp"
#include "fp16_simd.hpp"
#include "bf16_simd.hpp"
#include "fp32_simd.hpp"
#include "fp64_simd.hpp"

namespace SIMD_NAMESPACE
{
	static inline SIMD<float> to_float(SIMD<int8_t> x) noexcept
	{
#if SUPPORTS_AVX
		return _mm256_cvtepi32_ps(x);
#elif SUPPORTS_SSE2
		return _mm_cvtepi32_ps(x);
#else
		return SIMD<float>(static_cast<float>(static_cast<int32_t>(x)));
#endif
	}
	static inline SIMD<float> to_float(SIMD<int32_t> x) noexcept
	{
//#if SUPPORTS_AVX
//		return _mm256_cvtepi32_ps(x);
//#elif SUPPORTS_SSE2
//		return _mm_cvtepi32_ps(x);
//#else
//		return SIMD<float>(static_cast<float>(static_cast<int32_t>(x)));
//#endif
	}
	static inline SIMD<float> to_float(SIMD<avocado::backend::float16> x) noexcept
	{
//#if SUPPORTS_AVX
//		return _mm256_cvtepi32_ps(x);
//#elif SUPPORTS_SSE2
//		return _mm_cvtepi32_ps(x);
//#else
//		return SIMD<float>(static_cast<float>(static_cast<int32_t>(x)));
//#endif
	}
	static inline SIMD<float> to_float(SIMD<avocado::backend::bfloat16> x) noexcept
	{
//#if SUPPORTS_AVX
//		return _mm256_cvtepi32_ps(x);
//#elif SUPPORTS_SSE2
//		return _mm_cvtepi32_ps(x);
//#else
//		return SIMD<float>(static_cast<float>(static_cast<int32_t>(x)));
//#endif
	}
	static inline SIMD<float> to_float(SIMD<double> x) noexcept
	{
//#if SUPPORTS_AVX
//		return _mm256_cvtepi32_ps(x);
//#elif SUPPORTS_SSE2
//		return _mm_cvtepi32_ps(x);
//#else
//		return SIMD<float>(static_cast<float>(static_cast<int32_t>(x)));
//#endif
	}
}

#endif /* VECTORS_SIMD_CASTING_HPP_ */
