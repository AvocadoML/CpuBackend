/*
 * conversions.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <avocado/backend/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <complex>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<typename T, typename U>
	struct Converter
	{
			static T convert(U x) noexcept
			{
				return static_cast<T>(x);
			}
	};

	template<typename U>
	struct Converter<float16, U>
	{
			static float16 convert(U x) noexcept
			{
				return scalar::float_to_float16(static_cast<float>(x));
			}
	};
	template<typename T>
	struct Converter<T, float16>
	{
			static T convert(float16 x) noexcept
			{
				return static_cast<T>(scalar::float16_to_float(x));
			}
	};
	template<>
	struct Converter<float16, float16>
	{
			static float16 convert(float16 x) noexcept
			{
				return x;
			}
	};
	template<>
	struct Converter<float16, std::complex<float>>
	{
			static float16 convert(std::complex<float> x) noexcept
			{
				return Converter<float16, float>::convert(x.real());
			}
	};
	template<>
	struct Converter<float16, std::complex<double>>
	{
			static float16 convert(std::complex<double> x) noexcept
			{
				return Converter<float16, double>::convert(x.real());
			}
	};

	template<typename U>
	struct Converter<bfloat16, U>
	{
			static bfloat16 convert(U x) noexcept
			{
				return scalar::float_to_bfloat16(static_cast<float>(x));
			}
	};
	template<typename T>
	struct Converter<T, bfloat16>
	{
			static T convert(bfloat16 x) noexcept
			{
				return static_cast<T>(scalar::bfloat16_to_float(x));
			}
	};
	template<>
	struct Converter<bfloat16, bfloat16>
	{
			static bfloat16 convert(bfloat16 x) noexcept
			{
				return x;
			}
	};
	template<>
	struct Converter<bfloat16, std::complex<float>>
	{
			static bfloat16 convert(std::complex<float> x) noexcept
			{
				return Converter<bfloat16, float>::convert(x.real());
			}
	};
	template<>
	struct Converter<bfloat16, std::complex<double>>
	{
			static bfloat16 convert(std::complex<double> x) noexcept
			{
				return Converter<bfloat16, double>::convert(x.real());
			}
	};

	template<>
	struct Converter<float16, bfloat16>
	{
			static float16 convert(bfloat16 x) noexcept
			{
				return scalar::float_to_float16(scalar::bfloat16_to_float(x));
			}
	};
	template<>
	struct Converter<bfloat16, float16>
	{
			static bfloat16 convert(float16 x) noexcept
			{
				return scalar::float_to_bfloat16(scalar::float16_to_float(x));
			}
	};

	template<typename T>
	struct Converter<T, std::complex<float>>
	{
			static T convert(std::complex<float> x) noexcept
			{
				return Converter<T, float>::convert(x.real());
			}
	};
	template<typename T>
	struct Converter<T, std::complex<double>>
	{
			static T convert(std::complex<double> x) noexcept
			{
				return Converter<T, double>::convert(x.real());
			}
	};

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t changeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
			avDataType_t srcType, avSize_t elements)
	{
		return AVOCADO_STATUS_SUCCESS;
	}
} /* namespace SIMD_NAMESPACE */

