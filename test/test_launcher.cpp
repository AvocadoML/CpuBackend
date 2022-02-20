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

#include <gtest/gtest.h>

using namespace avocado::backend;

int main(int argc, char *argv[])
{
	cpuSetNumberOfThreads(1);
//	::testing::InitGoogleTest(&argc, argv);
//	return RUN_ALL_TESTS();

	WinogradTest data(0, { 12, 4, 4, 13 }, { 12, 5, 5, 13 }, AVOCADO_DTYPE_FLOAT32, 2);
	float padding[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	data.set(AVOCADO_CONVOLUTION_MODE, { -1, -1, 0 }, { 1, 1, 0 }, 1, padding);
	double alpha = 1.1, beta = 0.1;
//	double diff = data.getDifferenceWeight();
//	double diff = data.getDifferenceInput();

	double diff = data.getDifferenceGradient();
//	double diff = data.getDifferenceBackward(&alpha, &beta);
//	double diff = data.getDifferenceUpdate(&alpha, &beta);
	std::cout << "diff = " << diff << '\n';

	std::cout << "END" << std::endl;
	return 0;
}

