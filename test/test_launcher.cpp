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
#include <cstring>

using namespace avocado::backend;

int main(int argc, char *argv[])
{
	cpuSetNumberOfThreads(1);
//	::testing::InitGoogleTest(&argc, argv);
//	return RUN_ALL_TESTS();

	UnaryOpTester data(0, AVOCADO_UNARY_OP_LOGICAL_NOT, { 13, 15 }, AVOCADO_DTYPE_FLOAT32);
	float alpha = 1.0f, beta = 0.0f;
	double diff = data.getDifference(&alpha, &beta);
	std::cout << "diff = " << diff << '\n';

	float f = 2.0f;
	int tmp;
	std::memcpy(&tmp, &f, 4);
	std::cout << tmp << '\n';
	tmp = ~tmp;
	std::cout << tmp << '\n';
	std::memcpy(&f, &tmp, 4);
	std::cout << f << '\n';

	std::cout << "END" << std::endl;
	return 0;
}

