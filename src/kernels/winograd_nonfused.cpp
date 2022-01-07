/*
 * winograd_nonfused.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <avocado/backend/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	struct int2
	{
			int x, y;
	};

	template<typename T, int Length>
	struct Line
	{
		private:
			SIMD<T> data[Length];
		public:
			template<int Columns>
			inline void load_column(const T **ptr, const int col, const int offset, const int num) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[i * Columns + col] + offset, num);
			}
			template<int Columns>
			inline void load_column(const SIMD<T> *ptr, const int col) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[i * Columns + col];
			}

			template<int Columns>
			inline void store_row(T **ptr, const int row, const int offset, const int num) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * Columns + i] + offset, num);
			}
			template<int Columns>
			inline void store_row(SIMD<T> *ptr, const int row) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[row * Columns + i];
			}

			inline SIMD<T>& operator[](int index) noexcept
			{
				assert(index >= 0 && index < Length);
				return data[index];
			}
			inline SIMD<T> operator[](int index) const noexcept
			{
				assert(index >= 0 && index < Length);
				return data[index];
			}
	};

	template<typename T, int TransformSize, int KernelSize>
	struct WeightTransform
	{
	};
	template<typename T>
	struct WeightTransform<T, 3, 4>
	{
			inline Line<T, 6> operator()(Line<T, 3> line) const noexcept
			{
				const SIMD<T> c13(1.0 / 3.0);
				const SIMD<T> c23(2.0 / 3.0);
				const SIMD<T> c2(2.0);
				const SIMD<T> c4(4.0);

				Line<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2]);
				result[2] = c23 * (line[0] - line[1] + line[2]);
				result[3] = c13 * (line[0] + c2 * line[1] + c4 * line[2]);
				result[4] = c13 * (line[0] - c2 * line[1] + c4 * line[2]);
				result[5] = c2 * line[2];
				return result;
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
				const SIMD<T> c025(0.25);
				const SIMD<T> c05(0.5);

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

	template<typename T, int TransformSize, int KernelSize>
	struct OutputTransform
	{
	};
	template<typename T>
	struct OutputTransform<T, 4, 3>
	{
			inline Line<T, 4> operator()(Line<T, 6> line) const noexcept
			{
				const SIMD<T> c025(0.25);
				const SIMD<T> c05(0.5);
				const SIMD<T> c2(2.0);

				Line<T, 4> result;
//				result[0] = line[0] + line[1] + load2 + 0.25f * (load3 + load4);
//				result[1] = line[1] - load2 + 0.5f * (load3 - load4);
//				result[2] = line[1] + load2 + load3 + load4;
//				result[3] = line[1] - load2 + 2.0f * (load3 - load4 + load5);
//				result[0] = line[0] - line[2] + c025 * (line[4] - line[2]);
//				result[1] = line[1] + line[2] - c025 * (line[3] + line[4]);
//				result[2] = line[2] - line[1] + c025 * (line[3] - line[4]);
//				result[3] = line[3] - line[1] + c05 * (line[4] - line[2]);
//				result[4] = line[1] - line[3] + c05 * (line[4] - line[2]);
//				result[5] = line[1] - line[3] + c025 * (line[5] - line[3]);
				return result;
			}
	};

	//default transforms
	void def_transform_weight_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
	{
		//Transform matrix
		// 1.0  0.0  0.0
		// 2/3  2/3  2/3
		// 2/3 -2/3  2/3
		// 1/3  2/3  4/3
		// 1/3 -2/3  4/3
		// 0.0  0.0  2.0
		const float c23 = 2.0f / 3.0f;
		const float c13 = 1.0f / 3.0f;
		for (int in = 0; in < filters; in++)
		{
			for (int k = 0; k < 3; k++)
			{
				float load0 = ptr_in[k + 0 * 3][in];
				float load1 = ptr_in[k + 1 * 3][in];
				float load2 = ptr_in[k + 2 * 3][in];

				storage[k + 0 * 3] = load0;
				storage[k + 1 * 3] = c23 * (load0 + load1 + load2);
				storage[k + 2 * 3] = c23 * (load0 - load1 + load2);
				storage[k + 3 * 3] = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
				storage[k + 4 * 3] = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
				storage[k + 5 * 3] = 2.0f * load2;
			}

			for (int k = 0; k < 6; k++)
			{
				float load0 = storage[k * 3 + 0];
				float load1 = storage[k * 3 + 1];
				float load2 = storage[k * 3 + 2];

				ptr_out[6 * k + 0][in] = load0;
				ptr_out[6 * k + 1][in] = c23 * (load0 + load1 + load2);
				ptr_out[6 * k + 2][in] = c23 * (load0 - load1 + load2);
				ptr_out[6 * k + 3][in] = c13 * (load0 + 2.0f * load1 + 4.0f * load2);
				ptr_out[6 * k + 4][in] = c13 * (load0 - 2.0f * load1 + 4.0f * load2);
				ptr_out[6 * k + 5][in] = 2.0f * load2;
			}
		}
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
	void def_transform_output_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters, const float *bias, const float **ptr_add)
	{
		//Transform matrix
		// 1.0 1.0  1.0 0.25 0.25 0.0
		// 0.0 1.0 -1.0 0.5 -0.5  0.0
		// 0.0 1.0  1.0 1.0  1.0  0.0
		// 0.0 1.0 -1.0 2.0 -2.0  2.0
		for (int out = 0; out < filters; out++)
		{
			const float _bias = (bias == nullptr) ? 0.0f : bias[out];
			for (int l = 0; l < 6; l++)
			{
				int tmp_index = 6 * l;
				float load0 = ptr_in[tmp_index][out];
				float load1 = ptr_in[tmp_index + 1][out];
				float load2 = ptr_in[tmp_index + 2][out];
				float load3 = ptr_in[tmp_index + 3][out];
				float load4 = ptr_in[tmp_index + 4][out];
				float load5 = ptr_in[tmp_index + 5][out];

				tmp_index = 4 * l;
				storage[tmp_index] = load0 + load1 + load2 + 0.25f * (load3 + load4);
				storage[tmp_index + 1] = load1 - load2 + 0.5f * (load3 - load4);
				storage[tmp_index + 2] = load1 + load2 + load3 + load4;
				storage[tmp_index + 3] = load1 - load2 + 2.0f * (load3 - load4 + load5);
			}

			for (int l = 0; l < 4; l++)
			{
				float load0 = storage[l];
				float load1 = storage[4 + l];
				float load2 = storage[8 + l];
				float load3 = storage[12 + l];
				float load4 = storage[16 + l];
				float load5 = storage[20 + l];

				float tmp0 = load0 + load1 + load2 + 0.25f * (load3 + load4) + _bias;
				float tmp1 = load1 - load2 + 0.5f * (load3 - load4) + _bias;
				float tmp2 = load1 + load2 + load3 + load4 + _bias;
				float tmp3 = load1 - load2 + 2.0f * (load3 - load4 + load5) + _bias;

				if (ptr_add != nullptr)
				{
					tmp0 += ptr_add[0 + l][out];
					tmp1 += ptr_add[4 + l][out];
					tmp2 += ptr_add[8 + l][out];
					tmp3 += ptr_add[12 + l][out];
				}

				ptr_out[0 + l][out] = tmp0;
				ptr_out[4 + l][out] = tmp1;
				ptr_out[8 + l][out] = tmp2;
				ptr_out[12 + l][out] = tmp3;
			}
		}
	}
	void def_transform_gradient_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
	{
		//Transform matrix
		// 1.0  0.0  0.0  0.0
		// 2/3  2/3  2/3  2/3
		// 2/3 -2/3  2/3 -2/3
		// 1/3  2/3  4/3  8/3
		// 1/3 -2/3  4/3 -8/3
		// 0.0  0.0  0.0  2.0

		const float c0_33 = 1.0f / 3.0f;
		const float c0_66 = 2.0f / 3.0f;
		for (int out = 0; out < filters; out++)
		{
			for (int l = 0; l < 4; l++)
			{
				float load0 = ptr_in[l][out];
				float load1 = ptr_in[l + 4][out];
				float load2 = ptr_in[l + 8][out];
				float load3 = ptr_in[l + 12][out];

				storage[l] = load0;
				storage[4 + l] = c0_66 * (load0 + load1 + load2 + load3);
				storage[8 + l] = c0_66 * (load0 - load1 + load2 - load3);
				storage[12 + l] = c0_33 * (load0 + load2) + load2 + (c0_66 * (load1 + load3) + 2.0f * load3);
				storage[16 + l] = c0_33 * (load0 + load2) + load2 - (c0_66 * (load1 + load3) + 2.0f * load3);
				storage[20 + l] = 2.0f * load3;
			}
			for (int l = 0; l < 6; l++)
			{
				int tmp_index = 4 * l;
				float load0 = storage[tmp_index];
				float load1 = storage[tmp_index + 1];
				float load2 = storage[tmp_index + 2];
				float load3 = storage[tmp_index + 3];

				tmp_index = 6 * l;
				ptr_out[tmp_index + 0][out] = load0;
				ptr_out[tmp_index + 1][out] = c0_66 * (load0 + load1 + load2 + load3);
				ptr_out[tmp_index + 2][out] = c0_66 * (load0 - load1 + load2 - load3);
				ptr_out[tmp_index + 3][out] = c0_33 * (load0 + load2) + load2 + (c0_66 * (load1 + load3) + 2.0f * load3);
				ptr_out[tmp_index + 4][out] = c0_33 * (load0 + load2) + load2 - (c0_66 * (load1 + load3) + 2.0f * load3);
				ptr_out[tmp_index + 5][out] = 2.0f * load3;
			}
		}
	}
	void def_transform_update_4x4(const float **ptr_in, float **ptr_out, float *storage, const int filters)
	{
		//Transform matrix
		// 1.0  1.0  1.0  0.25 0.25 0.0
		// 0.0  1.0 -1.0  0.5 -0.5  0.0
		// 0.0  1.0  1.0  1.0  1.0  2.0
		for (int in = 0; in < filters; in++)
		{
			for (int l = 0; l < 6; l++)
			{
				float load0 = ptr_in[l][in];
				float load1 = ptr_in[6 + l][in];
				float load2 = ptr_in[12 + l][in];
				float load3 = ptr_in[18 + l][in];
				float load4 = ptr_in[24 + l][in];
				float load5 = ptr_in[30 + l][in];

				storage[l] = load0 + load1 + load2 + 0.25f * (load3 + load4);
				storage[6 + l] = load1 - load2 + 0.5f * (load3 - load4);
				storage[12 + l] = load1 + load2 + load3 + load4 + 2.0f * load5;
			}

			for (int l = 0; l < 3; l++)
			{
				int tmp_index = 6 * l;
				float load0 = storage[tmp_index];
				float load1 = storage[tmp_index + 1];
				float load2 = storage[tmp_index + 2];
				float load3 = storage[tmp_index + 3];
				float load4 = storage[tmp_index + 4];
				float load5 = storage[tmp_index + 5];

				ptr_out[l * 3 + 0][in] += load0 + load1 + load2 + 0.25f * (load3 + load4);
				ptr_out[l * 3 + 1][in] += load1 - load2 + 0.5f * (load3 - load4);
				ptr_out[l * 3 + 2][in] += load1 + load2 + load3 + load4 + 2.0f * load5;
			}
		}
	}

	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void winograd_weight_transform(const TensorDescriptor &wDesc, const T *wMem, const TensorDescriptor &mDesc, T *mMem, bool invert)
	{
		const int filtersOut = wDesc.firstDim();
		const int filtersIn = wDesc.lastDim();

#pragma omp parallel
		{
			const T *ptr_in[KernelSize * KernelSize];
			T *ptr_out[TileSize * TileSize];
			SIMD<T> storage[TileSize * KernelSize];
#pragma omp for
			for (int out = 0; out < filtersOut; out++)
			{
				for (int i = 0; i < KernelSize; i++)
					for (int j = 0; j < KernelSize; j++)
					{
						int tmp = i * KernelSize + j;
						if (invert)
							ptr_in[KernelSize * KernelSize - 1 - tmp] = wMem + wDesc.getIndex( { out, i, j, 0 });
						else
							ptr_in[tmp] = wMem + wDesc.getIndex( { out, i, j, 0 });
					}
				for (int k = 0; k < TileSize * TileSize; k++)
					ptr_out[k] = mMem + mDesc.getIndex( { k, out, 0 });

				WeightTransform<T, TransformSize, KernelSize> transform;
				for (int in = 0; in < filtersIn; in += SIMD<T>::length)
				{
					const int elements_left = std::min(filtersIn - in, SIMD<T>::length);
					for (int col = 0; col < KernelSize; col++)
					{
						Line<T, KernelSize> column;
						column.load_column<KernelSize>(ptr_in, col, in, elements_left);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_row<TileSize>(storage, col);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<T, KernelSize> column;
						column.load_column<KernelSize>(storage, col);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_row<TileSize>(ptr_out, col, in, elements_left);
					}
				}
			}
		}
	}
	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void winograd_input_transform(const TensorDescriptor &xDesc, const T *xMem, const TensorDescriptor &mDesc, T *mMem, int2 padding, T *workspace)
	{
		const int batch_size = xDesc.dimension(0);
		const int height = xDesc.dimension(1);
		const int width = xDesc.dimension(2);
		const int filters = xDesc.dimension(3);

		const int tiles = mDesc.dimension(1);

		const int tile_h = (height + 3) / 4;
		const int tile_w = (width + 3) / 4;
		const int number_of_tiles = tile_h * tile_w;
		const int nb_of_tiles = batch_size * number_of_tiles;

		T *zero_line = workspace;
		std::memset(zero_line, 0, filters * sizeof(T));

#pragma omp parallel
		{
			const T *ptr_in[TileSize * TileSize];
			T *ptr_out[TileSize * TileSize];
			SIMD<T> storage[TileSize * TileSize];
#pragma omp for
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
				for (int in = 0; in < filters; in += SIMD<T>::length)
				{
					const int elements_left = std::min(filters - in, SIMD<T>::length);
					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TileSize> column;
						column.load_column<TileSize>(ptr_in, col, in, elements_left);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_row<TileSize>(storage, col);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TileSize> column;
						column.load_column<TileSize>(storage, col);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_row<TileSize>(ptr_out, col, in, elements_left);
					}
				}
			}
		}
	}
	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void winograd_output_transform(const TensorDescriptor &yDesc, T *yMem, const TensorDescriptor &mDesc, const T *mMem, const T *bMem, const T *zMem,
			avActivationType_t activation, T alpha1, T alpha2, T beta, T *workspace)
	{
		const int batch_size = yDesc.dimension(0);
		const int height = yDesc.dimension(1);
		const int width = yDesc.dimension(2);
		const int filters = yDesc.dimension(3);

		const int tiles = mDesc.dimension(1);

		const int tile_h = (height + 3) / 4;
		const int tile_w = (width + 3) / 4;
		const int number_of_tiles = tile_h * tile_w;
		const int nb_of_tiles = batch_size * number_of_tiles;

		T *zero_line = workspace;
		std::memset(zero_line, 0, filters * sizeof(float));

#pragma omp parallel
		{
			T *fake_storage = workspace + (1 + omp_get_thread_num()) * filters;

			SIMD<T> tmp_storage[TransformSize * TileSize];
			const T *ptr_in[36];
			T *ptr_out[16];
			const T *ptr_add[16];

#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				for (int j = 0; j < 36; j++)
					ptr_in[j] = mMem + (j * tiles + tile_idx) * filters;

				int batch = tile_idx / number_of_tiles;
				int tile_x = (tile_idx % number_of_tiles) / tile_w;
				int tile_y = (tile_idx % number_of_tiles) % tile_w;
				int output_idx = 0;
				for (int i = 0; i < TransformSize; i++)
					for (int j = 0; j < TransformSize; j++, output_idx++)
					{
						int x = tile_x * TransformSize + i;
						int y = tile_y * TransformSize + j;
						if (x < height and y < width)
							ptr_out[output_idx] = yMem + yDesc.getIndex( { batch, x, y, 0 });
						else
							ptr_out[output_idx] = fake_storage;
					}

				if (zMem != nullptr)
				{
					output_idx = 0;
					for (int i = 0; i < TransformSize; i++)
						for (int j = 0; j < TransformSize; j++, output_idx++)
						{
							int x = tile_x * TransformSize + i;
							int y = tile_y * TransformSize + j;
							if (x < height and y < width)
								ptr_add[output_idx] = zMem + yDesc.getIndex( { batch, x, y, 0 });
							else
								ptr_add[output_idx] = fake_storage;
						}
				}
//				const float **tmp_ptr = (add->data == nullptr) ? nullptr : ptr_add;
//				if (simd >= SIMD_AVX)
//					avx_transform_output_4x4(ptr_in, ptr_out, tmp_storage, filters, bias_ptr, tmp_ptr);
//				else
//				{
//					if (simd >= SIMD_SSE)
//						sse_transform_output_4x4(ptr_in, ptr_out, tmp_storage, filters, bias_ptr, tmp_ptr);
//					else
//						def_transform_output_4x4(ptr_in, ptr_out, tmp_storage, filters, bias_ptr, tmp_ptr);
//				}
//				for (int i = 0; i < 16; i++)
//					cpu_act_forward_range_in_place(act, ptr_out[i], filters);
			}
		}
	}
//	int cpu_winograd3x3_4x4_transform_gradient(SIMD_TYPE simd, ConstTensorDescriptor *gradient_next, TensorDescriptor *matrices)
//	{
//		const int batch_size = gradient_next->shape[0];
//		const int height = gradient_next->shape[1];
//		const int width = gradient_next->shape[2];
//		const int filters = gradient_next->shape[3];
//
//		const int tile_h = (height + 3) / 4;
//		const int tile_w = (width + 3) / 4;
//		const int number_of_tiles = tile_h * tile_w;
//		const int nb_of_tiles = batch_size * number_of_tiles;
//
//		float zero_line[filters];
//		for (int i = 0; i < filters; i++)
//			zero_line[i] = 0.0f;
//#pragma omp parallel
//		{
//			float tmp_storage[288];
//			const float *ptr_in[16];
//			float *ptr_out[36];
//#pragma omp for
//			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
//			{
//				int b = tile_idx / number_of_tiles;
//				int x = 4 * ((tile_idx % number_of_tiles) / tile_w);
//				int y = 4 * ((tile_idx % number_of_tiles) % tile_w);
//				int tmp_idx = 0;
//				for (int j = 0; j < 4; j++)
//					for (int k = 0; k < 4; k++, tmp_idx++)
//						if ((x + j) < height && (y + k) < width)
//							ptr_in[tmp_idx] = reinterpret_cast<float*>(gradient_next->data) + ((b * height + x + j) * width + y + k) * filters; //(b, x + j, y + k, 0);
//						else
//							ptr_in[tmp_idx] = zero_line;
//
//				for (int j = 0; j < 36; j++)
//					ptr_out[j] = reinterpret_cast<float*>(matrices->data) + (j * matrices->shape[1] + tile_idx) * filters; //(j, tile_idx, 0);
//
//				if (simd >= SIMD_AVX)
//					avx_transform_gradient_4x4(ptr_in, ptr_out, tmp_storage, filters);
//				else
//				{
//					if (simd >= SIMD_SSE)
//						sse_transform_gradient_4x4(ptr_in, ptr_out, tmp_storage, filters);
//					else
//						def_transform_gradient_4x4(ptr_in, ptr_out, tmp_storage, filters);
//				}
//			}
//		}
//		return 0;
//	}
//	int cpu_winograd3x3_4x4_transform_update(SIMD_TYPE simd, TensorDescriptor *update, ConstTensorDescriptor *matrices)
//	{
//		const int filtersOut = update->shape[0];
//		const int filtersIn = update->shape[3];
//
//#pragma omp parallel
//		{
//			float tmp_storage[288];
//			const float *ptr_in[36];
//			float *ptr_out[9];
//#pragma omp for
//			for (int f = 0; f < filtersOut; f++)
//			{
//				for (int j = 0; j < 36; j++)
//					ptr_in[j] = reinterpret_cast<float*>(matrices->data) + (j * filtersOut + f) * filtersIn; //(j, f, 0);
//				for (int j = 0; j < 9; j++)
//					ptr_out[j] = reinterpret_cast<float*>(update->data) + (f * 9 + j) * filtersIn; //(f, j / 3, j % 3, 0);
//
//				if (simd >= SIMD_AVX)
//					avx_transform_update_4x4(ptr_in, ptr_out, tmp_storage, filtersIn);
//				else
//				{
//					if (simd >= SIMD_SSE)
//						sse_transform_update_4x4(ptr_in, ptr_out, tmp_storage, filtersIn);
//					else
//						def_transform_update_4x4(ptr_in, ptr_out, tmp_storage, filtersIn);
//				}
//			}
//		}
//		return 0;
//	}

}

namespace avocado
{
	namespace backend
	{

	} /* namespace backend */
} /* namespace avocado */

