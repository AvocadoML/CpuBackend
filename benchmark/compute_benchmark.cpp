/*
 * compute_benchmark.cpp
 *
 *  Created on: May 18, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/testing/utils.hpp>
#include "../src/kernel_definitions.hpp"
#include "../src/utils.hpp"
#include <iostream>

using namespace avocado::backend;

int main(int argc, char *argv[])
{
	if (argc != 3)
		return -1;
	const double max_time = std::stod(argv[1]);
	const avDataType_t dtype = dtypeFromString(argv[2]);
	const SimdLevel simd_level = simdLevelFromString(argv[3]);

#if DYNAMIC_ARCH
	switch (simd_level)
	{
		case SimdLevel::AVX2:
			std::cout << ns_avx2::cpu_compute_banchmark(dtype, 1000000000ull) << std::endl;
			break;
		case SimdLevel::AVX:
			std::cout << ns_avx::cpu_compute_banchmark(dtype, 1000000000ull) << std::endl;
			break;
		case SimdLevel::SSE41:
			std::cout << ns_sse41::cpu_compute_banchmark(dtype, 1000000000ull) << std::endl;
			break;
		case SimdLevel::SSE2:
			std::cout << ns_sse2::cpu_compute_banchmark(dtype, 1000000000ull) << std::endl;
			break;
		case SimdLevel::NONE:
			std::cout << ns_none::cpu_compute_banchmark(dtype, 1000000000ull) << std::endl;
			break;
		default:
			std::cout << "0" << std::endl;
			return -1;
	}
#else
	std::cout << SIMD_NAMESPACE::cpu_compute_banchmark(dtype, 1000000000ull) << std::endl;
#endif

	return 0;
}

