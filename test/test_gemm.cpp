/*
 * test_gemm.cpp
 *
 *  Created on: Jan 22, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace avocado
{
	namespace backend
	{
		TEST(TestGemm, float32_AB)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_N, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT32));
		}
		TEST(TestGemm, float32_ABT)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT32));
		}
		TEST(TestGemm, float32_ATB)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_N, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT32));
		}
		TEST(TestGemm, float32_ATBT)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_T, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT32));
		}

		TEST(TestGemm, float64_AB)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_N, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT64));
		}
		TEST(TestGemm, float64_ABT)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT64));
		}
		TEST(TestGemm, float64_ATB)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_N, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT64));
		}
		TEST(TestGemm, float64_ATBT)
		{
			GemmTester data(23, 45, 67, AVOCADO_GEMM_OPERATION_T, AVOCADO_GEMM_OPERATION_T, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			double diff = data.getDifference(&alpha, &beta);
			EXPECT_LT(diff, epsilonForTest(AVOCADO_DTYPE_FLOAT64));
		}
	} /* namespace backend */
} /* namespace avocado */

