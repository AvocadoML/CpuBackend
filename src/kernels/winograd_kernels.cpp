/*
 * winograd_kernels.cpp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include "../vectors/simd_vectors.hpp"
#include <avocado/backend/backend_descriptors.hpp>

#include <omp.h>
#include <iostream>

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

//	template<typename T, int kernel_size, int tile_size>
//	struct WinogradKernels
//	{
//			static void weight_transform(const TensorDescriptor *weights, TensorDescriptor *matrices, bool invert_kernel);
//			static void cpuWinogradWeightTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t weights,
//					avTensor_t matrices);
//			static void cpuWinogradInputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t input,
//					avTensor_t matrices, const avTensor_t bias, const avActivationType_t activation);
//			static void cpuWinogradOutputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
//					const avScalar_t beta, const avTensor_t matrices, avTensor_t output);
//			static void cpuWinogradGradientTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t gradient,
//					avTensor_t matrices);
//			static void cpuWinogradUpdateTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
//					const avScalar_t beta, const avTensor_t matrices, avTensor_t update);
//	};
//
//	template<typename T>
//	struct WinogradKernels<T, 3, 4>
//	{
//			static void weight_transform(const TensorDescriptor &weights, TensorDescriptor &matrices, bool invert_kernel)
//			{
//				const int filtersOut = firstDim(weights);
//				const size_t filtersIn = lastDim(weights);
//
//#pragma omp parallel
//				{
//					const T *ptr_in[9];
//					T *ptr_out[36];
//					SIMD<T> storage[24];
//#pragma omp for
//					for (int out = 0; out < filtersOut; out++)
//					{
//						for (int k = 0; k < 9; k++)
//						{
//							if (invert_kernel)
//								ptr_in[8 - k] = data < T > (weights) + (out * 9 + k) * filtersIn; //(out, k / 3, k % 3, 0);
//							else
//								ptr_in[k] = data < T > (weights) + (out * 9 + k) * filtersIn; //(out, k / 3, k % 3, 0);
//						}
//						for (int k = 0; k < 36; k++)
//							ptr_out[k] = data < T > (matrices) + (k * filtersOut + out) * filtersIn; //(k, out, 0);
//
//							//Transform matrix
//							// 1.0  0.0  0.0
//							// 2/3  2/3  2/3
//							// 2/3 -2/3  2/3
//							// 1/3  2/3  4/3
//							// 1/3 -2/3  4/3
//							// 0.0  0.0  2.0
//						const SIMD<T> c23 = 2.0f / 3.0f;
//						const SIMD<T> c13 = 1.0f / 3.0f;
//						for (size_t f = 0; f < filtersIn; f += SIMD<T>::length)
//						{
//							const size_t items = std::min(SIMD<T>::length, filtersIn - f);
//							for (int k = 0; k < 3; k++)
//							{
//								SIMD<T> load0(ptr_in[k + 0 * 3] + f, items);
//								SIMD<T> load1(ptr_in[k + 1 * 3] + f, items);
//								SIMD<T> load2(ptr_in[k + 2 * 3] + f, items);
//
//								storage[k + 0 * 3] = load0;
//								storage[k + 1 * 3] = c23 * (load0 + load1 + load2);
//								storage[k + 2 * 3] = c23 * (load0 - load1 + load2);
//								storage[k + 3 * 3] = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
//								storage[k + 4 * 3] = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
//								storage[k + 5 * 3] = 2.0f * load2;
//							}
//
//							for (int k = 0; k < 6; k++)
//							{
//								SIMD<T> load0 = storage[k * 3 + 0];
//								SIMD<T> load1 = storage[k * 3 + 1];
//								SIMD<T> load2 = storage[k * 3 + 2];
//
//								SIMD<T> tmp0 = load0;
//								SIMD<T> tmp1 = c23 * (load0 + load1 + load2);
//								SIMD<T> tmp2 = c23 * (load0 - load1 + load2);
//								SIMD<T> tmp3 = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
//								SIMD<T> tmp4 = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
//								SIMD<T> tmp5 = 2.0f * load2;
//
//								tmp0.storeu(ptr_out[6 * k + 0] + f, items);
//								tmp1.storeu(ptr_out[6 * k + 1] + f, items);
//								tmp2.storeu(ptr_out[6 * k + 2] + f, items);
//								tmp3.storeu(ptr_out[6 * k + 3] + f, items);
//								tmp4.storeu(ptr_out[6 * k + 4] + f, items);
//								tmp5.storeu(ptr_out[6 * k + 5] + f, items);
//							}
//						}
//					}
//				}
//			}
//	};
//
//	avStatus_t winograd_3x3_4x4_weight_transform(const TensorDescriptor *weights, TensorDescriptor *matrices, bool invert_kernel)
//	{
//		switch (weights->dtype)
//		{
//			case AVOCADO_DTYPE_BFLOAT16:
//				WinogradKernels<bfloat16, 3, 4>::weight_transform(weights, matrices, invert_kernel);
//				return AVOCADO_STATUS_SUCCESS;
//			case AVOCADO_DTYPE_FLOAT16:
//				WinogradKernels<float16, 3, 4>::weight_transform(weights, matrices, invert_kernel);
//				return AVOCADO_STATUS_SUCCESS;
//			case AVOCADO_DTYPE_FLOAT32:
//				WinogradKernels<float, 3, 4>::weight_transform(weights, matrices, invert_kernel);
//				return AVOCADO_STATUS_SUCCESS;
//			case AVOCADO_DTYPE_FLOAT64:
//				WinogradKernels<double, 3, 4>::weight_transform(weights, matrices, invert_kernel);
//				return AVOCADO_STATUS_SUCCESS;
//			default:
//				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//		}
//	}

} /* namespace SIMD_NAMESPACE */
