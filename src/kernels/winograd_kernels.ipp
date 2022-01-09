/*
 * winograd_kernels.ipp
 *
 *  Created on: Nov 22, 2021
 *      Author: Maciej Kozarzewski
 */

// you can uncomment the lines below for developing
//#include <avocado/cpu_backend.h>
//#include <avocado/backend/tensor_helpers.hpp>
//#include <vectors/fp32_simd.hpp>
//#include <omp.h>
//#include <utils.hpp>
//using namespace avocado::backend;

//template<typename T, int kernel_size, int tile_size>
//struct WinogradKernels
//{
//		void weight_transform(const TensorDescriptor *weights, TensorDescriptor *matrices);
//};
//
//template<typename T>
//struct WinogradKernels<T, 3, 4>
//{
//		void weight_transform(const TensorDescriptor *weights, TensorDescriptor *matrices, bool invert_kernel)
//		{
//			const int filtersOut = firstDim(weights);
//			const size_t filtersIn = lastDim(weights);
//
//#pragma omp parallel
//			{
//				const T *ptr_in[9];
//				T *ptr_out[36];
//				SIMD<T> storage[24];
//#pragma omp for
//				for (int out = 0; out < filtersOut; out++)
//				{
//					for (int k = 0; k < 9; k++)
//					{
//						if (invert_kernel)
//							ptr_in[8 - k] = data<T>(weights) + (out * 9 + k) * filtersIn; //(out, k / 3, k % 3, 0);
//						else
//							ptr_in[k] = data<T>(weights) + (out * 9 + k) * filtersIn; //(out, k / 3, k % 3, 0);
//					}
//					for (int k = 0; k < 36; k++)
//						ptr_out[k] = data<T>(matrices) + (k * filtersOut + out) * filtersIn; //(k, out, 0);
//						//Transform matrix
//						// 1.0  0.0  0.0
//						// 2/3  2/3  2/3
//						// 2/3 -2/3  2/3
//						// 1/3  2/3  4/3
//						// 1/3 -2/3  4/3
//						// 0.0  0.0  2.0
//					const SIMD<T> c23 = 2.0f / 3.0f;
//					const SIMD<T> c13 = 1.0f / 3.0f;
//					for (size_t f = 0; f < filtersIn; f += SIMD<T>::length)
//					{
//						const size_t items = std::min(SIMD<T>::length, filtersIn - f);
//						for (int k = 0; k < 3; k++)
//						{
//							SIMD<T> load0(ptr_in[k + 0 * 3] + f, items);
//							SIMD<T> load1(ptr_in[k + 1 * 3] + f, items);
//							SIMD<T> load2(ptr_in[k + 2 * 3] + f, items);
//
//							storage[k + 0 * 3] = load0;
//							storage[k + 1 * 3] = c23 * (load0 + load1 + load2);
//							storage[k + 2 * 3] = c23 * (load0 - load1 + load2);
//							storage[k + 3 * 3] = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
//							storage[k + 4 * 3] = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
//							storage[k + 5 * 3] = 2.0f * load2;
//						}
//
//						for (int k = 0; k < 6; k++)
//						{
//							SIMD<T> load0 = storage[k * 3 + 0];
//							SIMD<T> load1 = storage[k * 3 + 1];
//							SIMD<T> load2 = storage[k * 3 + 2];
//
//							SIMD<T> tmp0 = load0;
//							SIMD<T> tmp1 = c23 * (load0 + load1 + load2);
//							SIMD<T> tmp2 = c23 * (load0 - load1 + load2);
//							SIMD<T> tmp3 = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
//							SIMD<T> tmp4 = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
//							SIMD<T> tmp5 = 2.0f * load2;
//
//							tmp0.storeu(ptr_out[6 * k + 0] + f, items);
//							tmp1.storeu(ptr_out[6 * k + 1] + f, items);
//							tmp2.storeu(ptr_out[6 * k + 2] + f, items);
//							tmp3.storeu(ptr_out[6 * k + 3] + f, items);
//							tmp4.storeu(ptr_out[6 * k + 4] + f, items);
//							tmp5.storeu(ptr_out[6 * k + 5] + f, items);
//						}
//					}
//				}
//			}
//
//		}
////		void input_transform(const T **ptr_in, T **ptr_out, SIMD<T> *storage, const size_t filters)
////		{
////			//Transform matrix
////			// 1.0  0.0 -1.25  0.0   0.25  0.0
////			// 0.0  1.0  1.0  -0.25 -0.25  0.0
////			// 0.0 -1.0  1.0   0.25 -0.25  0.0
////			// 0.0 -1.0 -0.5   1.0   0.5   0.0
////			// 0.0  1.0 -0.5  -1.0   0.5   0.0
////			// 0.0  1.0  0.0  -1.25  0.0   0.25
////
////			const SIMD<T> c025(0.25);
////			const SIMD<T> c05(0.5);
////			for (size_t f = 0; f < filters; f += SIMD<T>::length)
////			{
////				const size_t elements = std::min(SIMD<T>::length, filters - f);
////				for (int l = 0; l < 36; l += 6)
////				{
////					SIMD<T> load0(ptr_in[l + 0] + f, elements);
////					SIMD<T> load1(ptr_in[l + 1] + f, elements);
////					SIMD<T> load2(ptr_in[l + 2] + f, elements);
////					SIMD<T> load3(ptr_in[l + 3] + f, elements);
////					SIMD<T> load4(ptr_in[l + 4] + f, elements);
////					SIMD<T> load5(ptr_in[l + 5] + f, elements);
////					storage[l + 0] = load0 - load2 + c025 * (load4 - load2);
////					storage[l + 1] = load1 + load2 - c025 * (load3 + load4);
////					storage[l + 2] = load2 - load1 + c025 * (load3 - load4);
////					storage[l + 3] = load3 - load1 + c05 * (load4 - load2);
////					storage[l + 4] = load1 - load3 + c05 * (load4 - load2);
////					storage[l + 5] = load1 - load3 + c025 * (load5 - load3);
////				}
////				for (int l = 0; l < 6; l++)
////				{
////					SIMD<T> load0 = storage[0 * 6 + l];
////					SIMD<T> load1 = storage[1 * 6 + l];
////					SIMD<T> load2 = storage[2 * 6 + l];
////					SIMD<T> load3 = storage[3 * 6 + l];
////					SIMD<T> load4 = storage[4 * 6 + l];
////					SIMD<T> load5 = storage[5 * 6 + l];
////
////					load0 = load0 - load2 + c025 * (load4 - load2);
////					SIMD<T> tmp1 = load1 + load2 - c025 * (load3 + load4);
////					SIMD<T> tmp2 = load2 - load1 + c025 * (load3 - load4);
////					SIMD<T> tmp3 = load3 - load1 + c05 * (load4 - load2);
////					SIMD<T> tmp4 = load1 - load3 + c05 * (load4 - load2);
////					load5 = load1 - load3 + c025 * (load5 - load3);
////
////					load0.storeu(ptr_out[0 + l] + f, elements);
////					tmp1.storeu(ptr_out[6 + l] + f, elements);
////					tmp2.storeu(ptr_out[12 + l] + f, elements);
////					tmp3.storeu(ptr_out[18 + l] + f, elements);
////					tmp4.storeu(ptr_out[24 + l] + f, elements);
////					load5.storeu(ptr_out[30 + l] + f, elements);
////				}
////			}
////		}
//};

//template<typename T>

//void def_transform_output_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters, const float *bias, const float **ptr_add)
//{
//	//Transform matrix
//	// 1.0 1.0  1.0 0.25 0.25 0.0
//	// 0.0 1.0 -1.0 0.5 -0.5  0.0
//	// 0.0 1.0  1.0 1.0  1.0  0.0
//	// 0.0 1.0 -1.0 2.0 -2.0  2.0
//	for (int out = 0; out < filters; out++)
//	{
//		const float _bias = (bias == nullptr) ? 0.0f : bias[out];
//		for (int l = 0; l < 6; l++)
//		{
//			int tmp_index = 6 * l;
//			float load0 = ptr_in[tmp_index][out];
//			float load1 = ptr_in[tmp_index + 1][out];
//			float load2 = ptr_in[tmp_index + 2][out];
//			float load3 = ptr_in[tmp_index + 3][out];
//			float load4 = ptr_in[tmp_index + 4][out];
//			float load5 = ptr_in[tmp_index + 5][out];
//
//			tmp_index = 4 * l;
//			storage[tmp_index] = load0 + load1 + load2 + 0.25f * (load3 + load4);
//			storage[tmp_index + 1] = load1 - load2 + 0.5f * (load3 - load4);
//			storage[tmp_index + 2] = load1 + load2 + load3 + load4;
//			storage[tmp_index + 3] = load1 - load2 + 2.0f * (load3 - load4 + load5);
//		}
//
//		for (int l = 0; l < 4; l++)
//		{
//			float load0 = storage[l];
//			float load1 = storage[4 + l];
//			float load2 = storage[8 + l];
//			float load3 = storage[12 + l];
//			float load4 = storage[16 + l];
//			float load5 = storage[20 + l];
//
//			float tmp0 = load0 + load1 + load2 + 0.25f * (load3 + load4) + _bias;
//			float tmp1 = load1 - load2 + 0.5f * (load3 - load4) + _bias;
//			float tmp2 = load1 + load2 + load3 + load4 + _bias;
//			float tmp3 = load1 - load2 + 2.0f * (load3 - load4 + load5) + _bias;
//
//			if (ptr_add != nullptr)
//			{
//				tmp0 += ptr_add[0 + l][out];
//				tmp1 += ptr_add[4 + l][out];
//				tmp2 += ptr_add[8 + l][out];
//				tmp3 += ptr_add[12 + l][out];
//			}
//
//			ptr_out[0 + l][out] = tmp0;
//			ptr_out[4 + l][out] = tmp1;
//			ptr_out[8 + l][out] = tmp2;
//			ptr_out[12 + l][out] = tmp3;
//		}
//	}
//}
//void def_transform_gradient_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
//{
//	//Transform matrix
//	// 1.0  0.0  0.0  0.0
//	// 2/3  2/3  2/3  2/3
//	// 2/3 -2/3  2/3 -2/3
//	// 1/3  2/3  4/3  8/3
//	// 1/3 -2/3  4/3 -8/3
//	// 0.0  0.0  0.0  2.0
//
//	const float c0_33 = 1.0f / 3.0f;
//	const float c0_66 = 2.0f / 3.0f;
//	for (int out = 0; out < filters; out++)
//	{
//		for (int l = 0; l < 4; l++)
//		{
//			float load0 = ptr_in[l][out];
//			float load1 = ptr_in[l + 4][out];
//			float load2 = ptr_in[l + 8][out];
//			float load3 = ptr_in[l + 12][out];
//
//			storage[l] = load0;
//			storage[4 + l] = c0_66 * (load0 + load1 + load2 + load3);
//			storage[8 + l] = c0_66 * (load0 - load1 + load2 - load3);
//			storage[12 + l] = c0_33 * (load0 + load2) + load2 + (c0_66 * (load1 + load3) + 2.0f * load3);
//			storage[16 + l] = c0_33 * (load0 + load2) + load2 - (c0_66 * (load1 + load3) + 2.0f * load3);
//			storage[20 + l] = 2.0f * load3;
//		}
//		for (int l = 0; l < 6; l++)
//		{
//			int tmp_index = 4 * l;
//			float load0 = storage[tmp_index];
//			float load1 = storage[tmp_index + 1];
//			float load2 = storage[tmp_index + 2];
//			float load3 = storage[tmp_index + 3];
//
//			tmp_index = 6 * l;
//			ptr_out[tmp_index + 0][out] = load0;
//			ptr_out[tmp_index + 1][out] = c0_66 * (load0 + load1 + load2 + load3);
//			ptr_out[tmp_index + 2][out] = c0_66 * (load0 - load1 + load2 - load3);
//			ptr_out[tmp_index + 3][out] = c0_33 * (load0 + load2) + load2 + (c0_66 * (load1 + load3) + 2.0f * load3);
//			ptr_out[tmp_index + 4][out] = c0_33 * (load0 + load2) + load2 - (c0_66 * (load1 + load3) + 2.0f * load3);
//			ptr_out[tmp_index + 5][out] = 2.0f * load3;
//		}
//	}
//}
//void def_transform_update_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
//{
//	//Transform matrix
//	// 1.0  1.0  1.0  0.25 0.25 0.0
//	// 0.0  1.0 -1.0  0.5 -0.5  0.0
//	// 0.0  1.0  1.0  1.0  1.0  2.0
//	for (int in = 0; in < filters; in++)
//	{
//		for (int l = 0; l < 6; l++)
//		{
//			float load0 = ptr_in[l][in];
//			float load1 = ptr_in[6 + l][in];
//			float load2 = ptr_in[12 + l][in];
//			float load3 = ptr_in[18 + l][in];
//			float load4 = ptr_in[24 + l][in];
//			float load5 = ptr_in[30 + l][in];
//
//			storage[l] = load0 + load1 + load2 + 0.25f * (load3 + load4);
//			storage[6 + l] = load1 - load2 + 0.5f * (load3 - load4);
//			storage[12 + l] = load1 + load2 + load3 + load4 + 2.0f * load5;
//		}
//
//		for (int l = 0; l < 3; l++)
//		{
//			int tmp_index = 6 * l;
//			float load0 = storage[tmp_index];
//			float load1 = storage[tmp_index + 1];
//			float load2 = storage[tmp_index + 2];
//			float load3 = storage[tmp_index + 3];
//			float load4 = storage[tmp_index + 4];
//			float load5 = storage[tmp_index + 5];
//
//			ptr_out[l * 3 + 0][in] += load0 + load1 + load2 + 0.25f * (load3 + load4);
//			ptr_out[l * 3 + 1][in] += load1 - load2 + 0.5f * (load3 - load4);
//			ptr_out[l * 3 + 2][in] += load1 + load2 + load3 + load4 + 2.0f * load5;
//		}
//	}
//}

//Conv2D 3x3 4x4 Winograd transforms
//template<typename T, bool invert_kernel>
//void cpu_winograd3x3_4x4_transform_weight(const T *weight, T *matrices, size_t filtersIn, size_t filtersOut)
//{
//#pragma omp parallel
//	{
//		const T *ptr_in[9];
//		T *ptr_out[36];
//		SIMD<T> storage[24];
//#pragma omp for
//		for (int out = 0; out < filtersOut; out++)
//		{
//			for (int k = 0; k < 9; k++)
//			{
//				if (invert_kernel)
//					ptr_in[8 - k] = weight + (out * 9 + k) * filtersIn; //(out, k / 3, k % 3, 0);
//				else
//					ptr_in[k] = weight + (out * 9 + k) * filtersIn; //(out, k / 3, k % 3, 0);
//			}
//			for (int k = 0; k < 36; k++)
//				ptr_out[k] = matrices->data + (k * filtersOut + out) * filtersIn; //(k, out, 0);
//
//				//Transform matrix
//				// 1.0  0.0  0.0
//				// 2/3  2/3  2/3
//				// 2/3 -2/3  2/3
//				// 1/3  2/3  4/3
//				// 1/3 -2/3  4/3
//				// 0.0  0.0  2.0
//			const SIMD<T> c23 = 2.0f / 3.0f;
//			const SIMD<T> c13 = 1.0f / 3.0f;
//			for (size_t f = 0; f < filtersIn; f += SIMD<T>::length)
//			{
//				const size_t items = std::min(SIMD<T>::length, filtersIn - f);
//				for (int k = 0; k < 3; k++)
//				{
//					SIMD<T> load0(ptr_in[k + 0 * 3] + f, items);
//					SIMD<T> load1(ptr_in[k + 1 * 3] + f, items);
//					SIMD<T> load2(ptr_in[k + 2 * 3] + f, items);
//
//					storage[k + 0 * 3] = load0;
//					storage[k + 1 * 3] = c23 * (load0 + load1 + load2);
//					storage[k + 2 * 3] = c23 * (load0 - load1 + load2);
//					storage[k + 3 * 3] = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
//					storage[k + 4 * 3] = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
//					storage[k + 5 * 3] = 2.0f * load2;
//				}
//
//				for (int k = 0; k < 6; k++)
//				{
//					SIMD<T> load0 = storage[k * 3 + 0];
//					SIMD<T> load1 = storage[k * 3 + 1];
//					SIMD<T> load2 = storage[k * 3 + 2];
//
//					SIMD<T> tmp0 = load0;
//					SIMD<T> tmp1 = c23 * (load0 + load1 + load2);
//					SIMD<T> tmp2 = c23 * (load0 - load1 + load2);
//					SIMD<T> tmp3 = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
//					SIMD<T> tmp4 = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
//					SIMD<T> tmp5 = 2.0f * load2;
//
//					tmp0.storeu(ptr_out[6 * k + 0] + f, items);
//					tmp1.storeu(ptr_out[6 * k + 1] + f, items);
//					tmp2.storeu(ptr_out[6 * k + 2] + f, items);
//					tmp3.storeu(ptr_out[6 * k + 3] + f, items);
//					tmp4.storeu(ptr_out[6 * k + 4] + f, items);
//					tmp5.storeu(ptr_out[6 * k + 5] + f, items);
//				}
//			}
//
//		}
//	}
//}
//int cpu_winograd3x3_4x4_transform_input(SIMD_TYPE simd, ConstTensorDescriptor *input, TensorDescriptor *matrices)
//{
//	const int batch_size = input->shape[0];
//	const int height = input->shape[1];
//	const int width = input->shape[2];
//	const int filters = input->shape[3];
//
//	const int tile_h = (height + 3) / 4;
//	const int tile_w = (width + 3) / 4;
//	const int number_of_tiles = tile_h * tile_w;
//	const int nb_of_tiles = batch_size * number_of_tiles;
//
//	float zero_line[filters];
//	std::memset(zero_line, 0, filters * sizeof(float));
//
//#pragma omp parallel
//	{
//		float tmp_storage[288];
//		const float *ptr_in[36];
//		float *ptr_out[36];
//#pragma omp for
//		for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
//		{
//			int b = tile_idx / number_of_tiles;
//			int x = 4 * ((tile_idx % number_of_tiles) / tile_w);
//			int y = 4 * ((tile_idx % number_of_tiles) % tile_w);
//			int tmp_idx = 0;
//			for (int j = -1; j < 4 + 1; j++)
//				for (int k = -1; k < 4 + 1; k++, tmp_idx++)
//				{
//					ptr_out[tmp_idx] = reinterpret_cast<float*>(matrices->data) + (tmp_idx * matrices->shape[1] + tile_idx) * filters; //(tmp_idx, tile_idx, 0);
//					if ((x + j) >= 0 && (x + j) < height && (y + k) >= 0 && (y + k) < width)
//						ptr_in[tmp_idx] = reinterpret_cast<float*>(input->data) + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
//					else
//						ptr_in[tmp_idx] = zero_line;
//				}
//			if (simd >= SIMD_AVX)
//				avx_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
//			else
//			{
//				if (simd >= SIMD_SSE)
//					sse_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
//				else
//					def_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
//			}
//		}
//	}
//	return 0;
//}
//int cpu_winograd3x3_4x4_transform_output(SIMD_TYPE simd, TensorDescriptor *output, ConstTensorDescriptor *matrices, ConstTensorDescriptor *bias,
//		ConstTensorDescriptor *add, ACTIVATION_TYPE act)
//{
//	const int batch_size = output->shape[0];
//	const int height = output->shape[1];
//	const int width = output->shape[2];
//	const int filters = output->shape[3];
//
//	const int tile_h = (height + 3) / 4;
//	const int tile_w = (width + 3) / 4;
//	const int number_of_tiles = tile_h * tile_w;
//	const int nb_of_tiles = batch_size * number_of_tiles;
//
//	const float *bias_ptr = reinterpret_cast<float*>(bias->data);
//	float zero_line[filters];
//	std::memset(zero_line, 0, filters * sizeof(float));
//
//#pragma omp parallel
//	{
//		float fake_storage[filters];
//
//		float tmp_storage[192];
//		const float *ptr_in[36];
//		float *ptr_out[16];
//		const float *ptr_add[16];
//
//#pragma omp for
//		for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
//		{
//			int b = tile_idx / number_of_tiles;
//			int x = 4 * ((tile_idx % number_of_tiles) / tile_w);
//			int y = 4 * ((tile_idx % number_of_tiles) % tile_w);
//			for (int j = 0; j < 36; j++)
//				ptr_in[j] = reinterpret_cast<float*>(matrices->data) + (j * matrices->shape[1] + tile_idx) * filters;
//
//			int tmp_idx = 0;
//			for (int j = 0; j < 4; j++)
//				for (int k = 0; k < 4; k++, tmp_idx++)
//					if ((x + j) < height && (y + k) < width)
//						ptr_out[tmp_idx] = reinterpret_cast<float*>(output->data) + ((b * height + x + j) * width + y + k) * filters;
//					else
//						ptr_out[tmp_idx] = fake_storage;
//
//			if (add->data != nullptr)
//			{
//				tmp_idx = 0;
//				for (int j = 0; j < 4; j++)
//					for (int k = 0; k < 4; k++, tmp_idx++)
//						if ((x + j) < height && (y + k) < width)
//							ptr_add[tmp_idx] = reinterpret_cast<float*>(add->data) + ((b * height + x + j) * width + y + k) * filters;
//						else
//							ptr_add[tmp_idx] = zero_line;
//			}
//			const float **tmp_ptr = (add->data == nullptr) ? nullptr : ptr_add;
//			if (simd >= SIMD_AVX)
//				avx_transform_output_4x4(ptr_in, ptr_out, tmp_storage, filters, bias_ptr, tmp_ptr);
//			else
//			{
//				if (simd >= SIMD_SSE)
//					sse_transform_output_4x4(ptr_in, ptr_out, tmp_storage, filters, bias_ptr, tmp_ptr);
//				else
//					def_transform_output_4x4(ptr_in, ptr_out, tmp_storage, filters, bias_ptr, tmp_ptr);
//			}
//			for (int i = 0; i < 16; i++)
//				cpu_act_forward_range_in_place(act, ptr_out[i], filters);
//		}
//	}
//	return 0;
//}
//int cpu_winograd3x3_4x4_transform_gradient(SIMD_TYPE simd, ConstTensorDescriptor *gradient_next, TensorDescriptor *matrices)
//{
//	const int batch_size = gradient_next->shape[0];
//	const int height = gradient_next->shape[1];
//	const int width = gradient_next->shape[2];
//	const int filters = gradient_next->shape[3];
//
//	const int tile_h = (height + 3) / 4;
//	const int tile_w = (width + 3) / 4;
//	const int number_of_tiles = tile_h * tile_w;
//	const int nb_of_tiles = batch_size * number_of_tiles;
//
//	float zero_line[filters];
//	for (int i = 0; i < filters; i++)
//		zero_line[i] = 0.0f;
//#pragma omp parallel
//	{
//		float tmp_storage[288];
//		const float *ptr_in[16];
//		float *ptr_out[36];
//#pragma omp for
//		for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
//		{
//			int b = tile_idx / number_of_tiles;
//			int x = 4 * ((tile_idx % number_of_tiles) / tile_w);
//			int y = 4 * ((tile_idx % number_of_tiles) % tile_w);
//			int tmp_idx = 0;
//			for (int j = 0; j < 4; j++)
//				for (int k = 0; k < 4; k++, tmp_idx++)
//					if ((x + j) < height && (y + k) < width)
//						ptr_in[tmp_idx] = reinterpret_cast<float*>(gradient_next->data) + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
//					else
//						ptr_in[tmp_idx] = zero_line;
//
//			for (int j = 0; j < 36; j++)
//				ptr_out[j] = reinterpret_cast<float*>(matrices->data) + (j * matrices->shape[1] + tile_idx) * filters; //(j, tile_idx, 0);
//
//			if (simd >= SIMD_AVX)
//				avx_transform_gradient_4x4(ptr_in, ptr_out, tmp_storage, filters);
//			else
//			{
//				if (simd >= SIMD_SSE)
//					sse_transform_gradient_4x4(ptr_in, ptr_out, tmp_storage, filters);
//				else
//					def_transform_gradient_4x4(ptr_in, ptr_out, tmp_storage, filters);
//			}
//		}
//	}
//	return 0;
//}
//int cpu_winograd3x3_4x4_transform_update(SIMD_TYPE simd, TensorDescriptor *update, ConstTensorDescriptor *matrices)
//{
//	const int filtersOut = update->shape[0];
//	const int filtersIn = update->shape[3];
//
//#pragma omp parallel
//	{
//		float tmp_storage[288];
//		const float *ptr_in[36];
//		float *ptr_out[9];
//#pragma omp for
//		for (int f = 0; f < filtersOut; f++)
//		{
//			for (int j = 0; j < 36; j++)
//				ptr_in[j] = reinterpret_cast<float*>(matrices->data) + (j * filtersOut + f) * filtersIn; //(j, f, 0);
//			for (int j = 0; j < 9; j++)
//				ptr_out[j] = reinterpret_cast<float*>(update->data) + (f * 9 + j) * filtersIn; //(f, j / 3, j % 3, 0);
//
//			if (simd >= SIMD_AVX)
//				avx_transform_update_4x4(ptr_in, ptr_out, tmp_storage, filtersIn);
//			else
//			{
//				if (simd >= SIMD_SSE)
//					sse_transform_update_4x4(ptr_in, ptr_out, tmp_storage, filtersIn);
//				else
//					def_transform_update_4x4(ptr_in, ptr_out, tmp_storage, filtersIn);
//			}
//		}
//	}
//	return 0;
//}

