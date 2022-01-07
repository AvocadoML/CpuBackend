//============================================================================
// Name        : CpuBackend.cpp
// Author      : Maciej Kozarzewski
//============================================================================

#include <avocado/cpu_backend.h>
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
#if __FMA__
	std::cout << " fma";
#endif
	std::cout << '\n';
}
class TensorWrapper
{
	private:
		avTensorDescriptor_t desc;
		avMemoryDescriptor_t mem;
	public:
		TensorWrapper(std::initializer_list<int> dimensions, avDataType_t dtype)
		{
			cpuCreateTensorDescriptor(&desc);
			cpuSetTensorDescriptor(desc, dtype, dimensions.size(), dimensions.begin());

			avSize_t size_in_bytes = getTensor(desc).sizeInBytes();
			cpuCreateMemoryDescriptor(&mem, size_in_bytes);
			cpuSetMemory(cpuGetDefaultContext(), mem, size_in_bytes, nullptr, 0);
		}
		~TensorWrapper()
		{
			cpuDestroyTensorDescriptor(desc);
			cpuDestroyMemoryDescriptor(mem);
		}
		int dimension(int idx) const noexcept
		{
			return getTensor(desc).dimension(idx);
		}
		template<typename T>
		void fill(T value)
		{
			assert(typeOf<T>() == getTensor(desc).dtype());
			for (int i = 0; i < getTensor(desc).volume(); i++)
				getPointer<T>(mem)[i] = value;
		}
		template<typename T>
		void set(T value, std::initializer_list<int> idx)
		{
			assert(typeOf<T>() == getTensor(desc).dtype());
			getPointer<T>(mem)[getTensor(desc).getIndex(idx)] = value;
		}
		template<typename T>
		T get(std::initializer_list<int> idx) const
		{
			assert(typeOf<T>() == getTensor(desc).dtype());
			return getPointer<T>(mem)[getTensor(desc).getIndex(idx)];
		}
		const TensorDescriptor& tensor() const noexcept
		{
			return getTensor(desc);
		}
		avTensorDescriptor_t getDesc() const noexcept
		{
			return desc;
		}
		avMemoryDescriptor_t getMem() const noexcept
		{
			return mem;
		}
		template<typename T = void>
		T* data() noexcept
		{
			return getPointer<T>(mem);
		}
		template<typename T = void>
		const T* data() const noexcept
		{
			return getPointer<T>(mem);
		}
		int volume() const noexcept
		{
			return getTensor(desc).volume();
		}
		int sizeIntBytes() const noexcept
		{
			return volume() * dataTypeSize(getTensor(desc).dtype());
		}
		avDataType_t dtype() const noexcept
		{
			return getTensor(desc).dtype();
		}
};

class ContextWrapper
{
	private:
		avContextDescriptor_t desc;
	public:
		ContextWrapper()
		{
			cpuCreateContextDescriptor(&desc);
		}
		~ContextWrapper()
		{
			cpuDestroyContextDescriptor(desc);
		}
		operator avContextDescriptor_t() noexcept
		{
			return desc;
		}
};

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

int cpu_winograd3x3_4x4_transform_input(const TensorWrapper &input, TensorWrapper &matrices)
{
	const int batch_size = input.dimension(0);
	const int height = input.dimension(1);
	const int width = input.dimension(2);
	const int filters = input.dimension(3);

	const int tiles = matrices.dimension(1);

	const int tile_h = (height + 3) / 4;
	const int tile_w = (width + 3) / 4;
	const int number_of_tiles = tile_h * tile_w;
	const int nb_of_tiles = batch_size * number_of_tiles;

	float zero_line[filters];
	std::memset(zero_line, 0, filters * sizeof(float));

//#pragma omp parallel
	{
		float tmp_storage[288];
		const float *ptr_in[36];
		float *ptr_out[36];
//#pragma omp for
		for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
		{
			int b = tile_idx / number_of_tiles;
			int x = 4 * ((tile_idx % number_of_tiles) / tile_w);
			int y = 4 * ((tile_idx % number_of_tiles) % tile_w);
			int tmp_idx = 0;
			for (int j = -1; j < 4 + 1; j++)
				for (int k = -1; k < 4 + 1; k++, tmp_idx++)
				{
					ptr_out[tmp_idx] = matrices.data<float>() + (tmp_idx * tiles + tile_idx) * filters; //(tmp_idx, tile_idx, 0);
					if ((x + j) >= 0 && (x + j) < height && (y + k) >= 0 && (y + k) < width)
						ptr_in[tmp_idx] = input.data<float>() + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
					else
						ptr_in[tmp_idx] = zero_line;
				}
#if defined(__AVX__)
			avx_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
#elif defined(__SSE__)
			sse_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
#else
			def_transform_input_4x4(ptr_in, ptr_out, tmp_storage, filters);
#endif
		}
	}
	return 0;
}
template<typename T>
int vec_winograd3x3_4x4_transform_input(const TensorWrapper &input, TensorWrapper &matrices)
{
	const int batch_size = input.dimension(0);
	const int height = input.dimension(1);
	const int width = input.dimension(2);
	const int filters = input.dimension(3);

	const int tiles = matrices.dimension(1);

	const int tile_h = (height + 3) / 4;
	const int tile_w = (width + 3) / 4;
	const int number_of_tiles = tile_h * tile_w;
	const int nb_of_tiles = batch_size * number_of_tiles;

	T zero_line[filters];
	std::memset(zero_line, 0, filters * sizeof(T));

//#pragma omp parallel
	{
		SIMD_NAMESPACE::SIMD<T> storage[36];
//		float tmp_storage[288];
		const T *ptr_in[36];
		T *ptr_out[36];
//#pragma omp for
		for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
		{
			int b = tile_idx / number_of_tiles;
			int x = 4 * ((tile_idx % number_of_tiles) / tile_w);
			int y = 4 * ((tile_idx % number_of_tiles) % tile_w);
			int tmp_idx = 0;
			for (int j = -1; j < 4 + 1; j++)
				for (int k = -1; k < 4 + 1; k++, tmp_idx++)
				{
					ptr_out[tmp_idx] = matrices.data<T>() + (tmp_idx * tiles + tile_idx) * filters; //(tmp_idx, tile_idx, 0);
					if ((x + j) >= 0 && (x + j) < height && (y + k) >= 0 && (y + k) < width)
						ptr_in[tmp_idx] = input.data<T>() + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
					else
						ptr_in[tmp_idx] = zero_line;
				}
			const SIMD_NAMESPACE::SIMD<T> c025(0.25);
			const SIMD_NAMESPACE::SIMD<T> c05(0.5);
			for (int f = 0; f < filters; f += SIMD_NAMESPACE::SIMD<T>::length)
			{
				const int elements = std::min(SIMD_NAMESPACE::SIMD<T>::length, filters - f);
				for (int l = 0; l < 36; l += 6)
				{
					SIMD_NAMESPACE::SIMD<T> load0(ptr_in[l + 0] + f, elements);
					SIMD_NAMESPACE::SIMD<T> load1(ptr_in[l + 1] + f, elements);
					SIMD_NAMESPACE::SIMD<T> load2(ptr_in[l + 2] + f, elements);
					SIMD_NAMESPACE::SIMD<T> load3(ptr_in[l + 3] + f, elements);
					SIMD_NAMESPACE::SIMD<T> load4(ptr_in[l + 4] + f, elements);
					SIMD_NAMESPACE::SIMD<T> load5(ptr_in[l + 5] + f, elements);
					storage[l + 0] = load0 - load2 + c025 * (load4 - load2);
					storage[l + 1] = load1 + load2 - c025 * (load3 + load4);
					storage[l + 2] = load2 - load1 + c025 * (load3 - load4);
					storage[l + 3] = load3 - load1 + c05 * (load4 - load2);
					storage[l + 4] = load1 - load3 + c05 * (load4 - load2);
					storage[l + 5] = load1 - load3 + c025 * (load5 - load3);
				}
				for (int l = 0; l < 6; l++)
				{
					SIMD_NAMESPACE::SIMD<T> load0 = storage[0 * 6 + l];
					SIMD_NAMESPACE::SIMD<T> load1 = storage[1 * 6 + l];
					SIMD_NAMESPACE::SIMD<T> load2 = storage[2 * 6 + l];
					SIMD_NAMESPACE::SIMD<T> load3 = storage[3 * 6 + l];
					SIMD_NAMESPACE::SIMD<T> load4 = storage[4 * 6 + l];
					SIMD_NAMESPACE::SIMD<T> load5 = storage[5 * 6 + l];

					load0 = load0 - load2 + c025 * (load4 - load2);
					SIMD_NAMESPACE::SIMD<T> tmp1 = load1 + load2 - c025 * (load3 + load4);
					SIMD_NAMESPACE::SIMD<T> tmp2 = load2 - load1 + c025 * (load3 - load4);
					SIMD_NAMESPACE::SIMD<T> tmp3 = load3 - load1 + c05 * (load4 - load2);
					SIMD_NAMESPACE::SIMD<T> tmp4 = load1 - load3 + c05 * (load4 - load2);
					load5 = load1 - load3 + c025 * (load5 - load3);

					load0.store(ptr_out[0 + l] + f, elements);
					tmp1.store(ptr_out[6 + l] + f, elements);
					tmp2.store(ptr_out[12 + l] + f, elements);
					tmp3.store(ptr_out[18 + l] + f, elements);
					tmp4.store(ptr_out[24 + l] + f, elements);
					load5.store(ptr_out[30 + l] + f, elements);
				}
			}
		}
	}
	return 0;
}

struct int2
{
		int x, y;
};

template<typename T, int Length>
struct Line
{
	private:
		SIMD_NAMESPACE::SIMD<T> data[Length];
	public:
		inline void load_column(const T **ptr, const int col, const int offset, const int num, const int columns) noexcept
		{
			for (int i = 0; i < Length; i++)
				data[i].load(ptr[i * columns + col] + offset, num);
		}
		inline void load_row(const SIMD_NAMESPACE::SIMD<T> *ptr, const int row, const int columns) noexcept
		{
			for (int i = 0; i < Length; i++)
				data[i] = ptr[row * columns + i];
		}

		inline void store_row(T **ptr, const int row, const int offset, const int num, const int columns) const noexcept
		{
			for (int i = 0; i < Length; i++)
				data[i].store(ptr[row * columns + i] + offset, num);
		}
		inline void store_column(SIMD_NAMESPACE::SIMD<T> *ptr, const int col, const int columns) const noexcept
		{
			for (int i = 0; i < Length; i++)
				ptr[i * columns + col] = data[i];
		}

		inline SIMD_NAMESPACE::SIMD<T>& operator[](int index) noexcept
		{
			assert(index >= 0 && index < Length);
			return data[index];
		}
		inline SIMD_NAMESPACE::SIMD<T> operator[](int index) const noexcept
		{
			assert(index >= 0 && index < Length);
			return data[index];
		}
};

template<typename T, int TransformSize, int KernelSize>
struct InputTransform
{
};
template<typename T>
struct InputTransform<T, 4, 3>
{
		inline Line<T, 6> operator()(Line<T, 6> line) const noexcept
		{
			const SIMD_NAMESPACE::SIMD<T> c025(0.25);
			const SIMD_NAMESPACE::SIMD<T> c05(0.5);

			Line<T, 6> result;
			result[0] = line[0] - line[2] + c025 * (line[4] - line[2]);
			result[1] = line[1] + line[2] - c025 * (line[3] + line[4]);
			result[2] = line[2] - line[1] + c025 * (line[3] - line[4]);
			result[3] = line[3] - line[1] + c05 * (line[4] - line[2]);
			result[4] = line[1] - line[3] + c05 * (line[4] - line[2]);
			result[5] = line[1] - line[3] + c025 * (line[5] - line[3]);
			return result;
		}
};

template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
void winograd_input_transform(const TensorDescriptor &xDesc, const T *xMem, const TensorDescriptor &mDesc, T *mMem, int2 padding, T *workspace)
{
	const int batch_size = xDesc.dimension(0);
	const int height = xDesc.dimension(1);
	const int width = xDesc.dimension(2);
	const int filters = xDesc.dimension(3);

	const int tile_h = (height + TransformSize - 1) / TransformSize;
	const int tile_w = (width + TransformSize - 1) / TransformSize;
	const int number_of_tiles = tile_h * tile_w;
	const int nb_of_tiles = batch_size * number_of_tiles;

	T *zero_line = workspace;
	std::memset(zero_line, 0, filters * sizeof(T));

//#pragma omp parallel
	{
		const T *ptr_in[TileSize * TileSize];
		T *ptr_out[TileSize * TileSize];
		SIMD_NAMESPACE::SIMD<T> storage[TileSize * TileSize];
//#pragma omp for
		for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
		{
			int batch = tile_idx / number_of_tiles;
			int tile_x = ((tile_idx % number_of_tiles) / tile_w);
			int tile_y = ((tile_idx % number_of_tiles) % tile_w);

			int matrix_idx = 0;
			for (int i = 0; i < TileSize; i++)
				for (int j = 0; j < TileSize; j++, matrix_idx++)
				{
					int x = padding.x + TransformSize * tile_x + i;
					int y = padding.y + TransformSize * tile_y + j;
					ptr_out[matrix_idx] = mMem + mDesc.getIndex( { matrix_idx, tile_idx, 0 });
					if (x >= 0 and x < height and y >= 0 and y < width)
						ptr_in[matrix_idx] = xMem + xDesc.getIndex( { batch, x, y, 0 });
					else
						ptr_in[matrix_idx] = zero_line;
				}

			InputTransform<T, TransformSize, KernelSize> transform;
			for (int in = 0; in < filters; in += SIMD_NAMESPACE::SIMD<T>::length)
			{
				const int elements_left = std::min(filters - in, SIMD_NAMESPACE::SIMD<T>::length);
				for (int col = 0; col < TileSize; col++)
				{
					Line<T, TileSize> column;
					column.load_column(ptr_in, col, in, elements_left, TileSize);
					Line<T, TileSize> transformed = transform(column);
					transformed.store_column(storage, col, TileSize);
				}

				for (int col = 0; col < TileSize; col++)
				{
					Line<T, TileSize> column;
					column.load_row(storage, col, TileSize);
					Line<T, TileSize> transformed = transform(column);
					transformed.store_row(ptr_out, col, in, elements_left, TileSize);
				}
			}
		}
	}
}

template<typename T>
void initTensor(TensorWrapper &tensor)
{
	if (typeOf<T>() != tensor.dtype())
		throw std::logic_error("initTensor() : data type mismatch");

	for (int i = 0; i < tensor.volume(); i++)
		tensor.data<T>()[i] = std::sin(i / static_cast<T>(1234));
}
template<typename T>
void setTensor(TensorWrapper &tensor, T value)
{
	if (typeOf<T>() != tensor.dtype())
		throw std::logic_error("setTensor() : data type mismatch");

	for (int i = 0; i < tensor.volume(); i++)
		tensor.data<T>()[i] = value;
}
template<typename T>
double diff(const TensorWrapper &lhs, const TensorWrapper &rhs)
{
	if (typeOf<T>() != lhs.dtype() or lhs.dtype() != rhs.dtype())
		throw std::logic_error("diff() : data type mismatch");

	assert(volume(lhs.shape) == volume(rhs.shape));
	double result = 0.0;
	for (int i = 0; i < lhs.volume(); i++)
		result += std::abs(lhs.data<T>()[i] - rhs.data<T>()[i]);
	return result / lhs.volume();
}

void measure_time(int batch, int filters)
{
	avDataType_t dtype = AVOCADO_DTYPE_BFLOAT16;
	TensorWrapper workspace( { 1000000 }, dtype);

	TensorWrapper input( { batch, 20, 20, filters }, dtype);
//	initTensor<float>(input);

	TensorWrapper matrix( { 36, batch * 5 * 5, filters }, dtype);
	TensorWrapper matrix2( { 36, batch * 5 * 5, filters }, dtype);
	TensorWrapper matrix3( { 36, batch * 5 * 5, filters }, dtype);
	int repeats = 10000 / filters;
	double start, stop;

	std::cout << filters << " ";

	start = omp_get_wtime();
	for (int i = 0; i < repeats; i++)
		winograd_input_transform<bfloat16, 4, 3>(input.tensor(), input.data<bfloat16>(), matrix3.tensor(), matrix3.data<bfloat16>(), { -1, -1 },
				workspace.data<bfloat16>());
	stop = omp_get_wtime();
//	std::cout << (1000.0 / repeats) * (stop - start) << " ";

//	start = omp_get_wtime();
//	for (int i = 0; i < repeats; i++)
//		cpu_winograd3x3_4x4_transform_input(input, matrix);
//	stop = omp_get_wtime();
//	std::cout << (1000.0 / repeats) * (stop - start) << " ";
//
//	start = omp_get_wtime();
//	for (int i = 0; i < repeats; i++)
//		vec_winograd3x3_4x4_transform_input<float>(input, matrix2);
//	stop = omp_get_wtime();
	std::cout << (1000.0 / repeats) * (stop - start) << "\n";

//	start = omp_get_wtime();
//	for (int i = 0; i < repeats; i++)
//		winograd_input_transform<float, 4, 3>(input.tensor(), input.data<float>(), matrix3.tensor(), matrix3.data<float>(), { -1, -1 },
//				workspace.data<float>());
//	stop = omp_get_wtime();
//	std::cout << (1000.0 / repeats) * (stop - start) << "\n";

//	std::cout << diff<float>(matrix, matrix2) << '\n';
//	std::cout << diff<float>(matrix, matrix3) << '\n';
}

int main()
{
//	float float_data[8] = { 1.0f / 3.0f, 0.65f, -0.34f, -1.23f, 45.0f, 10.1f, 3.34f, 0.1f };
//	float16 float16_data[8];
//	SIMD_NAMESPACE::SIMD<float16> asdf(float_data);
//	std::cout << "loaded\n";
//	asdf.store(float16_data);
//	std::cout << "stored\n";
//
//	SIMD_NAMESPACE::SIMD<float16> asdf2(float16_data);
//
//	float dst[8];
//	asdf2.store(dst);
//	for (int i = 0; i < 8; i++)
//		std::cout << float_data[i] << " " << dst[i] << '\n';
	std::cout << "filters v3 old v2\n";
	for (int i = 1; i <= 256; i += 1)
		measure_time(128, i);
	return 0;

	char result[256];
	std::memset(result, 0, sizeof(result));
	cpuGetDeviceProperty(AVOCADO_DEVICE_NAME, result);
	std::cout << result << '\n';
	print_defined_flags();

	ContextWrapper context;

	TensorWrapper tensor1( { 1000000, 100 }, AVOCADO_DTYPE_FLOAT32);
	TensorWrapper tensor2( { 100 }, AVOCADO_DTYPE_FLOAT32);
	TensorWrapper tensor3( { 1000000, 100 }, AVOCADO_DTYPE_FLOAT32);

	SIMD_NAMESPACE::SIMD<float> res1, res2, res3, res4, res5, res6, res7, res8, res9, res10;
	SIMD_NAMESPACE::SIMD<float> data1(1.0f);
	SIMD_NAMESPACE::SIMD<float> data2(1.0f);

	int repeats = 3000;

	float alpha = 2.0f;
	float alpha2 = 1.0f;
	float beta = 0.0f;
	cpuSetNumberOfThreads(1);
	double total_time = 0.0;
	for (int i = 0; i < repeats; i++)
	{
		double start = omp_get_wtime();
//		avStatus_t status = cpuReduceTensor(context, AVOCADO_REDUCE_MAX, &alpha, tensor1.getDesc(), tensor1.getMem(), &beta, tensor2.getDesc(),
//				tensor2.getMem());

		for (int j = 0; j < 1000000; j++)
		{
			res1 = SIMD_NAMESPACE::mul_add(data1, data2, res1);
			res2 = SIMD_NAMESPACE::mul_add(data2, data2, res2);
			res3 = SIMD_NAMESPACE::mul_add(data2, data2, res3);
			res4 = SIMD_NAMESPACE::mul_add(data1, data1, res4);
			res5 = SIMD_NAMESPACE::mul_add(data1, data2, res5);
			res6 = SIMD_NAMESPACE::mul_add(data1, data1, res6);
			res7 = SIMD_NAMESPACE::mul_add(data2, data2, res7);
			res8 = SIMD_NAMESPACE::mul_add(data2, data1, res8);
			res9 = SIMD_NAMESPACE::mul_add(data2, data1, res9);
			res10 = SIMD_NAMESPACE::mul_add(data1, data2, res10);
		}
//		avStatus_t status = cpuBinaryOp(context, AVOCADO_BINARY_OP_ADD_SQUARE, &alpha, tensor1.getDesc(), tensor1.getMem(), &alpha2,
//				tensor2.getDesc(), tensor2.getMem(), &beta, tensor3.getDesc(), tensor3.getMem());

//		avStatus_t status = cpuScaleTensor(context, tensor1.getDesc(), tensor1.getMem(), &alpha);
//		std::memcpy(tensor1.data(), tensor2.data(), tensor1.sizeIntBytes());
		double stop = omp_get_wtime();
		total_time += (stop - start);

//		if (status != AVOCADO_STATUS_SUCCESS)
//		{
//			std::cout << "error : " << status << "\n";
//			break;
//		}
	}
	std::cout << (res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10)[0] << '\n';

	double bandwidth = 0.0; //static_cast<double>(repeats) * (tensor1.sizeIntBytes() + tensor2.sizeIntBytes() + tensor3.sizeIntBytes()) / total_time;
	double flops = repeats * (1000000.0 * res1.length * 10) / total_time;
	std::cout << 1000.0 * total_time / repeats << "ms \n\n";
	std::cout << bandwidth * 1.0e-9 << " GB/s\n";
	std::cout << flops * 1.0e-9 << " GFLOPS\n";

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
