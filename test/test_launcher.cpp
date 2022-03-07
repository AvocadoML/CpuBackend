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
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();

	WinogradTest data( { 12, 13, 14, 15 }, { 21, 5, 5, 15 }, AVOCADO_DTYPE_FLOAT16, 2);
	float padding[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	data.set(AVOCADO_CONVOLUTION_MODE, { -1, -1, 0 }, { 1, 1, 0 }, 1, padding);
	float alpha1 = 1.1, alpha2 = 1.2, beta = 0.1;
//	double diff = data.getDifferenceWeight();
	double diff = data.getDifferenceInput();
//	double diff = data.getDifferenceOutput(&alpha1, &alpha2, &beta);
//	double diff = data.getDifferenceGradient();
//	double diff = data.getDifferenceUpdate(&alpha, &beta);
//	double diff = data.getDifferenceBackward(&alpha, &beta);
//	double diff = data.getDifferenceUpdate(&alpha, &beta);
//	double diff = data.getDifferenceUpdate(&alpha, &beta);
	std::cout << "diff = " << diff << '\n';

	std::cout << "END" << std::endl;
	return 0;
}

