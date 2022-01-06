//============================================================================
// Name        : CpuBackend.cpp
// Author      : Maciej Kozarzewski
//============================================================================

#include <avocado/backend/backend_defs.h>
#include <avocado/backend/backend_descriptors.hpp>

#include <type_traits>
#include <iostream>
#include <omp.h>
#include <memory>
#include <thread>
#include <chrono>
#include <x86intrin.h>

#include "../src/vectors/simd_vectors.hpp"

using namespace avocado::backend;

void print_defined_flags()
{
	std::cout << "defined flags =";
#if SUPPORTS_SSE2__
	std::cout << " sse sse2";
#endif
#if SUPPORTS_SSE41
	std::cout << " sse3 ssse3 sse4.1";
#endif
#if SUPPORTS_AVX
	std::cout << " sse4.1 avx";
#endif
#if SUPPORTS_FP16
	std::cout << " f16c";
#endif
#if SUPPORTS_AVX2
	std::cout << " avx2";
#endif
	std::cout << '\n';
}

void def_transform_input_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
{
	//Transform matrix
	// 1.0  0.0 -1.25  0.0   0.25  0.0
	// 0.0  1.0  1.0  -0.25 -0.25  0.0
	// 0.0 -1.0  1.0   0.25 -0.25  0.0
	// 0.0 -1.0 -0.5   1.0   0.5   0.0
	// 0.0  1.0 -0.5  -1.0   0.5   0.0
	// 0.0  1.0  0.0  -1.25  0.0   0.25
	for (int in = 0; in < filters; in++)
	{
		for (int l = 0; l < 6; l++)
		{
			int tmp_index = 6 * l;
			float load0 = ptr_in[tmp_index][in];
			float load1 = ptr_in[tmp_index + 1][in];
			float load2 = ptr_in[tmp_index + 2][in];
			float load3 = ptr_in[tmp_index + 3][in];
			float load4 = ptr_in[tmp_index + 4][in];
			float load5 = ptr_in[tmp_index + 5][in];

			storage[tmp_index] = load0 - 1.25f * load2 + 0.25f * load4;
			storage[tmp_index + 1] = load1 + load2 - 0.25f * (load3 + load4);
			storage[tmp_index + 2] = load2 - load1 + 0.25f * (load3 - load4);
			storage[tmp_index + 3] = load3 - load1 + 0.5f * (load4 - load2);
			storage[tmp_index + 4] = load1 - load3 + 0.5f * (load4 - load2);
			storage[tmp_index + 5] = load1 - 1.25f * load3 + 0.25f * load5;
		}
		for (int l = 0; l < 6; l++)
		{
			float load0 = storage[0 + l];
			float load1 = storage[6 + l];
			float load2 = storage[12 + l];
			float load3 = storage[18 + l];
			float load4 = storage[24 + l];
			float load5 = storage[30 + l];

			ptr_out[0 + l][in] = load0 - 1.25f * load2 + 0.25f * load4;
			ptr_out[6 + l][in] = load1 + load2 - 0.25f * (load3 + load4);
			ptr_out[12 + l][in] = load2 - load1 + 0.25f * (load3 - load4);
			ptr_out[18 + l][in] = load3 - load1 + 0.5f * (load4 - load2);
			ptr_out[24 + l][in] = load1 - load3 + 0.5f * (load4 - load2);
			ptr_out[30 + l][in] = load1 - 1.25f * load3 + 0.25f * load5;
		}
	}
}

#if defined(__AVX__)
void avx_transform_input_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
{
	//Transform matrix
	// 1.0  0.0 -1.25  0.0   0.25  0.0
	// 0.0  1.0  1.0  -0.25 -0.25  0.0
	// 0.0 -1.0  1.0   0.25 -0.25  0.0
	// 0.0 -1.0 -0.5   1.0   0.5   0.0
	// 0.0  1.0 -0.5  -1.0   0.5   0.0
	// 0.0  1.0  0.0  -1.25  0.0   0.25
	const __m256 c05 = _mm256_set1_ps(0.5f);
	const __m256 c025 = _mm256_set1_ps(0.25f);
	if (filters % 8 != 0)
		def_transform_input_4x4(ptr_in, ptr_out, storage, filters % 8);
	for (int in = filters % 8; in < filters; in += 8)
	{
		for (int l = 0; l < 6; l++)
		{
#define AVX_WINOGRAD_4x4_INPUT_TRANSFORM\
	__m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;\
	tmp0 = _mm256_sub_ps(load4, load2);\
	load5 = _mm256_mul_ps(load5, c025);\
	tmp2 = _mm256_mul_ps(c025, load2);\
	tmp3 = _mm256_mul_ps(c025, load3);\
	tmp4 = _mm256_mul_ps(c025, load4);\
	tmp0 = _mm256_mul_ps(tmp0, c05);\
	tmp1 = _mm256_sub_ps(load3, load1);\
	load5 = _mm256_sub_ps(load5, load3);\
	load0 = _mm256_sub_ps(load0, tmp2);\
	load1 = _mm256_sub_ps(load1, tmp3);\
	load2 = _mm256_sub_ps(load2, tmp4);\
	tmp4 = _mm256_add_ps(tmp0, tmp1);\
	tmp5 = _mm256_sub_ps(tmp0, tmp1);\
	load5 = _mm256_add_ps(load5, load1);\
	load0 = _mm256_sub_ps(load0, load2);\
	tmp3 = _mm256_sub_ps(load2, load1);\
	tmp2 = _mm256_add_ps(load2, load1);

			int tmp_index = 6 * l;
			__m256 load0 = _mm256_loadu_ps(&ptr_in[tmp_index][in]);
			__m256 load1 = _mm256_loadu_ps(&ptr_in[tmp_index + 1][in]);
			__m256 load2 = _mm256_loadu_ps(&ptr_in[tmp_index + 2][in]);
			__m256 load3 = _mm256_loadu_ps(&ptr_in[tmp_index + 3][in]);
			__m256 load4 = _mm256_loadu_ps(&ptr_in[tmp_index + 4][in]);
			__m256 load5 = _mm256_loadu_ps(&ptr_in[tmp_index + 5][in]);
			AVX_WINOGRAD_4x4_INPUT_TRANSFORM
			tmp_index = 48 * l;
			_mm256_storeu_ps(&storage[tmp_index], load0);
			_mm256_storeu_ps(&storage[tmp_index + 8], tmp2);
			_mm256_storeu_ps(&storage[tmp_index + 16], tmp3);
			_mm256_storeu_ps(&storage[tmp_index + 24], tmp4);
			_mm256_storeu_ps(&storage[tmp_index + 32], tmp5);
			_mm256_storeu_ps(&storage[tmp_index + 40], load5);
		}
		for (int l = 0; l < 6; l++)
		{
			int tmp_index = 8 * l;
			__m256 load0 = _mm256_loadu_ps(&storage[0 + tmp_index]);
			__m256 load1 = _mm256_loadu_ps(&storage[48 + tmp_index]);
			__m256 load2 = _mm256_loadu_ps(&storage[96 + tmp_index]);
			__m256 load3 = _mm256_loadu_ps(&storage[144 + tmp_index]);
			__m256 load4 = _mm256_loadu_ps(&storage[192 + tmp_index]);
			__m256 load5 = _mm256_loadu_ps(&storage[240 + tmp_index]);
			AVX_WINOGRAD_4x4_INPUT_TRANSFORM
			tmp_index = 8 * in;
			_mm256_storeu_ps(&ptr_out[0 + l][in], load0);
			_mm256_storeu_ps(&ptr_out[6 + l][in], tmp2);
			_mm256_storeu_ps(&ptr_out[12 + l][in], tmp3);
			_mm256_storeu_ps(&ptr_out[18 + l][in], tmp4);
			_mm256_storeu_ps(&ptr_out[24 + l][in], tmp5);
			_mm256_storeu_ps(&ptr_out[30 + l][in], load5);
		}
	}
}
#elif defined(__SSE__)
void sse_transform_input_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
{
	//Transform matrix
	// 1.0  0.0 -1.25  0.0   0.25  0.0
	// 0.0  1.0  1.0  -0.25 -0.25  0.0
	// 0.0 -1.0  1.0   0.25 -0.25  0.0
	// 0.0 -1.0 -0.5   1.0   0.5   0.0
	// 0.0  1.0 -0.5  -1.0   0.5   0.0
	// 0.0  1.0  0.0  -1.25  0.0   0.25
	if (filters % 4 != 0)
		def_transform_input_4x4(ptr_in, ptr_out, storage, filters % 4);
	const __m128 c05 = _mm_set1_ps(0.5f);
	const __m128 c025 = _mm_set1_ps(0.25f);
	for (int in = filters % 4; in < filters; in += 4)
	{
		for (int l = 0; l < 6; l++)
		{
#define SSE_WINOGRAD_4x4_INPUT_TRANSFORM\
	__m128 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;\
	tmp0 = _mm_sub_ps(load4, load2);\
	load5 = _mm_mul_ps(load5, c025);\
	tmp2 = _mm_mul_ps(c025, load2);\
	tmp3 = _mm_mul_ps(c025, load3);\
	tmp4 = _mm_mul_ps(c025, load4);\
	tmp0 = _mm_mul_ps(tmp0, c05);\
	tmp1 = _mm_sub_ps(load3, load1);\
	load5 = _mm_sub_ps(load5, load3);\
	load0 = _mm_sub_ps(load0, tmp2);\
	load1 = _mm_sub_ps(load1, tmp3);\
	load2 = _mm_sub_ps(load2, tmp4);\
	tmp4 = _mm_add_ps(tmp0, tmp1);\
	tmp5 = _mm_sub_ps(tmp0, tmp1);\
	load5 = _mm_add_ps(load5, load1);\
	load0 = _mm_sub_ps(load0, load2);\
	tmp3 = _mm_sub_ps(load2, load1);\
	tmp2 = _mm_add_ps(load2, load1);

			int tmp_index = 6 * l;
			__m128 load0 = _mm_loadu_ps(&ptr_in[tmp_index][in]);
			__m128 load1 = _mm_loadu_ps(&ptr_in[tmp_index + 1][in]);
			__m128 load2 = _mm_loadu_ps(&ptr_in[tmp_index + 2][in]);
			__m128 load3 = _mm_loadu_ps(&ptr_in[tmp_index + 3][in]);
			__m128 load4 = _mm_loadu_ps(&ptr_in[tmp_index + 4][in]);
			__m128 load5 = _mm_loadu_ps(&ptr_in[tmp_index + 5][in]);
			SSE_WINOGRAD_4x4_INPUT_TRANSFORM
			tmp_index = 24 * l;
			_mm_storeu_ps(&storage[tmp_index], load0);
			_mm_storeu_ps(&storage[tmp_index + 4], tmp2);
			_mm_storeu_ps(&storage[tmp_index + 8], tmp3);
			_mm_storeu_ps(&storage[tmp_index + 12], tmp4);
			_mm_storeu_ps(&storage[tmp_index + 16], tmp5);
			_mm_storeu_ps(&storage[tmp_index + 20], load5);
		}
		for (int l = 0; l < 6; l++)
		{
			int tmp_index = 4 * l;
			__m128 load0 = _mm_loadu_ps(&storage[0 + tmp_index]);
			__m128 load1 = _mm_loadu_ps(&storage[24 + tmp_index]);
			__m128 load2 = _mm_loadu_ps(&storage[48 + tmp_index]);
			__m128 load3 = _mm_loadu_ps(&storage[72 + tmp_index]);
			__m128 load4 = _mm_loadu_ps(&storage[96 + tmp_index]);
			__m128 load5 = _mm_loadu_ps(&storage[120 + tmp_index]);
			SSE_WINOGRAD_4x4_INPUT_TRANSFORM
			tmp_index = 4 * in;
			_mm_storeu_ps(&ptr_out[0 + l][in], load0);
			_mm_storeu_ps(&ptr_out[6 + l][in], tmp2);
			_mm_storeu_ps(&ptr_out[12 + l][in], tmp3);
			_mm_storeu_ps(&ptr_out[18 + l][in], tmp4);
			_mm_storeu_ps(&ptr_out[24 + l][in], tmp5);
			_mm_storeu_ps(&ptr_out[30 + l][in], load5);
		}
	}
}
#endif

int cpu_winograd3x3_4x4_transform_input(const TensorDescriptor &inputDesc, const float *inputMem, TensorDescriptor &matricesDesc, float *matricesMem)
{
//	const int batch_size = dimension(input, 0);
//	const int height = dimension(input, 1);
//	const int width = dimension(input, 2);
//	const int filters = dimension(input, 3);
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
//					ptr_out[tmp_idx] = data<float>(matrices) + (tmp_idx * dimension(matrices, 1) + tile_idx) * filters; //(tmp_idx, tile_idx, 0);
//					if ((x + j) >= 0 && (x + j) < height && (y + k) >= 0 && (y + k) < width)
//						ptr_in[tmp_idx] = data<float>(input) + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
//					else
//						ptr_in[tmp_idx] = zero_line;
//				}
//#if defined(__AVX__)
//			avx_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
//#elif defined(__SSE__)
//			sse_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
//#else
//			def_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
//#endif
//		}
//	}
	return 0;
}
//template<typename T>
//int vec_winograd3x3_4x4_transform_input(const TensorDescriptor *input, TensorDescriptor *matrices)
//{
//	const int batch_size = dimension(input, 0);
//	const int height = dimension(input, 1);
//	const int width = dimension(input, 2);
//	const size_t filters = dimension(input, 3);
//
//	const int tile_h = (height + 3) / 4;
//	const int tile_w = (width + 3) / 4;
//	const int number_of_tiles = tile_h * tile_w;
//	const int nb_of_tiles = batch_size * number_of_tiles;
//
//	T zero_line[filters];
//	std::memset(zero_line, 0, filters * sizeof(T));
//
//#pragma omp parallel
//	{
//		SIMD<T> storage[36];
////		float tmp_storage[288];
//		const T *ptr_in[36];
//		T *ptr_out[36];
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
//					ptr_out[tmp_idx] = data<T>(matrices) + (tmp_idx * dimension(matrices, 1) + tile_idx) * filters; //(tmp_idx, tile_idx, 0);
//					if ((x + j) >= 0 && (x + j) < height && (y + k) >= 0 && (y + k) < width)
//						ptr_in[tmp_idx] = data<T>(input) + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
//					else
//						ptr_in[tmp_idx] = zero_line;
//				}
//
//			const SIMD<T> c025(0.25);
//			const SIMD<T> c05(0.5);
//			for (size_t f = 0; f < filters; f += SIMD<T>::length)
//			{
//				const size_t elements = std::min(SIMD<T>::length, filters - f);
//				for (int l = 0; l < 36; l += 6)
//				{
//					SIMD<T> load0(ptr_in[l + 0] + f, elements);
//					SIMD<T> load1(ptr_in[l + 1] + f, elements);
//					SIMD<T> load2(ptr_in[l + 2] + f, elements);
//					SIMD<T> load3(ptr_in[l + 3] + f, elements);
//					SIMD<T> load4(ptr_in[l + 4] + f, elements);
//					SIMD<T> load5(ptr_in[l + 5] + f, elements);
//					storage[l + 0] = load0 - load2 + c025 * (load4 - load2);
//					storage[l + 1] = load1 + load2 - c025 * (load3 + load4);
//					storage[l + 2] = load2 - load1 + c025 * (load3 - load4);
//					storage[l + 3] = load3 - load1 + c05 * (load4 - load2);
//					storage[l + 4] = load1 - load3 + c05 * (load4 - load2);
//					storage[l + 5] = load1 - load3 + c025 * (load5 - load3);
//				}
//				for (int l = 0; l < 6; l++)
//				{
//					SIMD<T> load0 = storage[0 * 6 + l];
//					SIMD<T> load1 = storage[1 * 6 + l];
//					SIMD<T> load2 = storage[2 * 6 + l];
//					SIMD<T> load3 = storage[3 * 6 + l];
//					SIMD<T> load4 = storage[4 * 6 + l];
//					SIMD<T> load5 = storage[5 * 6 + l];
//
//					load0 = load0 - load2 + c025 * (load4 - load2);
//					SIMD<T> tmp1 = load1 + load2 - c025 * (load3 + load4);
//					SIMD<T> tmp2 = load2 - load1 + c025 * (load3 - load4);
//					SIMD<T> tmp3 = load3 - load1 + c05 * (load4 - load2);
//					SIMD<T> tmp4 = load1 - load3 + c05 * (load4 - load2);
//					load5 = load1 - load3 + c025 * (load5 - load3);
//
//					load0.storeu(ptr_out[0 + l] + f, elements);
//					tmp1.storeu(ptr_out[6 + l] + f, elements);
//					tmp2.storeu(ptr_out[12 + l] + f, elements);
//					tmp3.storeu(ptr_out[18 + l] + f, elements);
//					tmp4.storeu(ptr_out[24 + l] + f, elements);
//					load5.storeu(ptr_out[30 + l] + f, elements);
//				}
//			}
//		}
//	}
//	return 0;
//}

//template<typename T>
//TensorDescriptor createTensor(std::initializer_list<int> shape)
//{
//	TensorDescriptor result;
//	result.shape = createShapeDescriptor(shape);
//	result.dtype = typeOf<T>();
//	result.data = new int8_t[volume(result.shape) * dataTypeSize(result.dtype)];
//	return result;
//}
//template<typename T>
//void initTensor(TensorDescriptor &tensor)
//{
//	for (int i = 0; i < volume(tensor.shape); i++)
//		reinterpret_cast<T*>(tensor.data)[i] = std::sin(i / static_cast<T>(1234));
//}
//template<typename T>
//void setTensor(TensorDescriptor &tensor, T value)
//{
//	for (int i = 0; i < volume(tensor.shape); i++)
//		reinterpret_cast<T*>(tensor.data)[i] = value;
//}
//template<typename T>
//double diff(const TensorDescriptor &lhs, const TensorDescriptor &rhs)
//{
//	assert(volume(lhs.shape) == volume(rhs.shape));
//	double result = 0.0;
//	for (int i = 0; i < volume(lhs.shape); i++)
//		result += std::abs(reinterpret_cast<T*>(lhs.data)[i] - reinterpret_cast<T*>(rhs.data)[i]);
//	return result / volume(lhs.shape);
//}
//void destroyTensor(TensorDescriptor &tensor)
//{
//	delete[] reinterpret_cast<int8_t*>(tensor.data);
//}

void measure_time(int batch, int filters)
{
//	TensorDescriptor input = createTensor<float>( { batch, 20, 20, filters });
//	initTensor<float>(input);
//
//	TensorDescriptor matrix = createTensor<float>( { 36, batch * 5 * 5, filters });
//	int repeats = 10000 / filters;
//	double start = omp_get_wtime();
//	for (int i = 0; i < repeats; i++)
//		cpu_winograd3x3_4x4_transform_input(&input, &matrix);
//	double stop = omp_get_wtime();
//	std::cout << filters << " " << (1000.0 / repeats) * (stop - start) << " ";
//
//	TensorDescriptor matrix2 = createTensor<float>( { 36, batch * 5 * 5, filters });
////	repeats = 50000 / filters;
//	start = omp_get_wtime();
////	for (int i = 0; i < repeats; i++)
////		vec_winograd3x3_4x4_transform_input<float>(&input, &matrix2);
//	stop = omp_get_wtime();
//	std::cout << (1000.0 / repeats) * (stop - start) << "\n";
//
////	std::cout << diff<float>(matrix, matrix2) << '\n';
//
//	destroyTensor(input);
//	destroyTensor(matrix);
//	destroyTensor(matrix2);
}

int main()
{
	omp_set_num_threads(1);
	print_defined_flags();

//	float tmp[8] = { -65, 0, 23, 5, 0, 1, -255, 256 };
	float tmp2[8] = { -0.65f, 0.10f, 0.0223f, 0.5f, 0.0f, 0.11f, -0.0255f, 0.1256f };

	int integer[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	SIMD_NAMESPACE::SIMD<int32_t> asdf;
	std::cout << SIMD_NAMESPACE::SIMD<int32_t>::length << '\n';
	asdf.load(integer, 8);

	std::unique_ptr<float[]> data = std::make_unique<float[]>(1000000);
	SIMD_NAMESPACE::SIMD<float> lhs(tmp2);

	int repeats = 10000;

	SIMD_NAMESPACE::SIMD<float> sum;

	double start = omp_get_wtime();
	for (int i = 0; i < repeats; i++)
	{
		for (int j = 0; j < 1000000; j += 8)
		{
			lhs.load(data.get() + j, 8);
			sum += lhs;
		}
	}
	double stop = omp_get_wtime();
	std::cout << (stop - start) << "s \n\n";
	std::cout << sum[0] << '\n';

//	SIMD<float> rhs(tmp2);
//	SIMD<float> sign = sgn(lhs);
//	SIMD<float> asdf(0.0);

//	SIMD<float> eps(1.0e-1f);
//	rhs = select((lhs < eps) & (lhs > -eps), SIMD<float>(0.0f), rhs);

//	rhs = ((lhs > SIMD<float>(0.0f)) & SIMD<float>(1.0f)) | ((lhs < SIMD<float>(0.0f)) & SIMD<float>(-1.0f));

//	for (int i = 0; i < lhs.length; i++)
//		std::cout << lhs[i] << " == " << rhs[i] << " = " << sign[i] << '\n';
//		std::cout << lhs[i] << " = " << sign[i] << " " << rhs[i] << '\n';
//		std::cout << rhs[i] << " " << ((lhs < eps) & (lhs > -eps))[i] << '\n';

//	for (int i = 1; i <= 128; i += 1)
//		measure_time(128, i);

//	std::cout << "avx2   " << COMPILE_WITH_AVX2 << '\n';
//	std::cout << "avx    " << COMPILE_WITH_AVX << '\n';
//	std::cout << "sse4.1 " << COMPILE_WITH_SSE41 << '\n';
//	std::cout << "sse2   " << COMPILE_WITH_SSE2 << '\n';
//	std::cout << "fp16   " << COMPILE_WITH_HALF_FLOATS << '\n';

//	const int length = SIMD<float16>::length;
//	float array[length];
//	for (int i = 0; i < length; i++)
//		array[i] = 0.1f * (1 + i);
//
//	SIMD<float16> simd(array);
//
//	float16 fp16_array[length];
//	std::memset(fp16_array, 0, sizeof(fp16_array));
//	simd.storeu(fp16_array, 2);
//
//	simd = SIMD<float16>(0.0f);
//	simd.loadu(fp16_array);
//	for (int i = 0; i < length; i++)
//		std::cout << i << " : " << simd[i] << '\n';

//	for (int i = 2; i < 256; i += 2)
//		measure_time(32, i);

//	bfloat16 src[8];
//	for (int i = 0; i < 8; i++)
//		src[i] = float_to_bfloat16(1 + i + 0.35f);
//
//	SIMD<bfloat16> simd(src);
//
//	float dst[8];
//	simd.storeu(dst);
//
//	for (int i = 0; i < 8; i++)
//		std::cout << i << " : " << dst[i] << '\n';

//	float src[8];
//	for (int i = 0; i < 8; i++)
//		src[i] = 1 + i + 0.34f;
//
//	SIMD<bfloat16> simd(src);
//
//	bfloat16 dst[8];
//	simd.storeu(dst);
//
//	for (int i = 0; i < 8; i++)
//		std::cout << i << " : " << dst[i].m_data << " " << bfloat16_to_float(dst[i]) << '\n';
//
//	uint32_t a[4] = { 10, 11, 12, 13 };
//	uint16_t c[4];
//
//	__m128i reg_a = _mm_loadu_si128((__m128i*) a);
//	__m128i mask = _mm_set1_epi32(0x0000FFFFu);
//	__m128i tmp1 = _mm_and_si128(reg_a, mask);
//	__m128i tmp2 = _mm_packs_epi16(tmp1, tmp1);
//	_mm_storeu_si64((__m128i*) c, tmp2);

//	__m128i reg_a = _mm_loadu_si64(a);
//	__m128i reg_c = _mm_unpacklo_epi16(reg_a, _mm_setzero_si128());
//	_mm_storeu_si128((__m128i*) c, reg_c);
//
//	std::cout << "a = ";
//	for (int i = 0; i < 4; i++)
//		std::cout << a[i] << ' ';
//	std::cout << '\n';

//	std::cout << "c = ";
//	for (int i = 0; i < 4; i++)
//		std::cout << c[i] << ' ';
//	std::cout << '\n';

//	uint16_t b[16] = { 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215 };
//	uint16_t d[16];
//	__m128i reg_b = _mm_loadu_si128((__m128i*) b);
//	__m128i reg_lo = _mm_unpacklo_epi16(reg_b, _mm_setzero_si128());
//	__m128i reg_hi = _mm_unpackhi_epi16(reg_b, _mm_setzero_si128());
//	__m256i reg_d = _mm256_setr_m128i(reg_lo,  reg_hi);
//
//	_mm256_storeu_si256((__m256i*) d, reg_d);
//	std::cout << "d = ";
//	for (int i = 0; i < 16; i++)
//		std::cout << d[i] << ' ';
//	std::cout << '\n';
//	for (int i = 0; i < 8; i++)
//		std::cout << reinterpret_cast<uint32_t*>(d)[i] << '\n';

	std::cout << "END" << std::endl;
	return 0;
}
