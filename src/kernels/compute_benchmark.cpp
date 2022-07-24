/*
 * compute_benchmark.cpp
 *
 *  Created on: May 18, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/testing/utils.hpp>
#include <Avocado/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;
	using namespace SIMD_NAMESPACE;

	template<typename T, int N>
	struct PerformCalculation
	{
			double operator()(size_t iterations)
			{
				SIMD<T> data[N];
				for (int n = 0; n < N; n++)
					data[n] = static_cast<float>(1 + n);

				Timer timer;
				timer++;
				for (size_t i = 0; i < iterations; i++)
					for (int n = 0; n < N; n++)
						data[n] = mul_add(data[n], data[n], data[n]);
				const double time = timer.get();

				if (data[0][0] == 0.0f)
					std::cout << "will never be printed\n";

				return iterations * N * SIMD<T>::length / time;
			}
	};

	template<int N>
	struct PerformCalculation<int8_t, N>
	{
			double operator()(size_t iterations)
			{
				SIMD<int8_t> data_a = 1;
				SIMD<int8_t> data_b = 2;
				SIMD<int32_t> data_c[N];

				for (int n = 0; n < N; n++)
					data_c[n] = static_cast<int32_t>(1 + n);

				Timer timer;
				timer++;
				for (size_t i = 0; i < iterations; i++)
					for (int n = 0; n < N; n++)
						data_c[n] += dp4a(data_a, data_b);
				const double time = timer.get();

				if (data_c[0][0] == 0)
					std::cout << "will never be printed\n";

				return iterations * N * (SIMD<int8_t>::length / 2) / time;
			}
	};

	template<typename T>
	double launch_benchmark(size_t iterations)
	{
		switch (getSimdSupport())
		{
			default:
			case SimdLevel::NONE:
				return PerformCalculation<T, 1>()(iterations);
			case SimdLevel::SSE:
			case SimdLevel::SSE2:
			case SimdLevel::SSE3:
			case SimdLevel::SSSE3:
			case SimdLevel::SSE41:
			case SimdLevel::SSE42:
				return PerformCalculation<T, 8>()(iterations);
			case SimdLevel::AVX:
			case SimdLevel::F16C:
			case SimdLevel::AVX2:
				return PerformCalculation<T, 16>()(iterations);
			case SimdLevel::AVX512F:
			case SimdLevel::AVX512VL_BW_DQ:
				return PerformCalculation<T, 32>()(iterations);
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	double cpu_compute_banchmark(avDataType_t dtype, size_t iterations)
	{
		switch (dtype)
		{
			case AVOCADO_DTYPE_INT8:
				return launch_benchmark<int8_t>(iterations);
			case AVOCADO_DTYPE_FLOAT16:
				return launch_benchmark<float16>(iterations);
			case AVOCADO_DTYPE_BFLOAT16:
				return launch_benchmark<bfloat16>(iterations);
			case AVOCADO_DTYPE_FLOAT32:
				return launch_benchmark<float>(iterations);
			case AVOCADO_DTYPE_FLOAT64:
				return launch_benchmark<double>(iterations);
			default:
				return 0.0;
		}
	}

} /* namespace SIMD_NAMESPACE */

