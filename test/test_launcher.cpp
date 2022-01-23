/*
 * test_launcher.cpp
 *
 *  Created on: Jan 21, 2022
 *      Author: Maciej Kozarzewski
 */

//#include <gtest/gtest.h>
#include "../../Avocado/include/Avocado/backend/testing/wrappers.hpp"
#include "../../Avocado/include/Avocado/backend/testing/testing_helpers.hpp"
#include <CpuBackend/cpu_backend.h>
#include <iostream>
#include <bitset>
#include <x86intrin.h>

#include "../src/vectors/simd_vectors.hpp"

using namespace avocado::backend;
using namespace SIMD_NAMESPACE;

int main(int argc, char *argv[])
{
//	cpuSetNumberOfThreads(1);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();

	UnaryOpTester data(0, AVOCADO_UNARY_OP_LOGICAL_NOT, { 8 }, AVOCADO_DTYPE_FLOAT64);
	double alpha = 1.1, beta = 0.1;
	double diff = data.getDifference(&alpha, &beta);
	std::cout << "diff = " << diff << '\n';

	std::cout << "END" << std::endl;
	return 0;
}

