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
			inline void load_column(const T **ptr, const int col, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[i * columns + col] + offset, num);
			}
			inline void load_column(const SIMD<T> *ptr, const int col, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[i * columns + col];
			}
			inline void load_row(const T **ptr, const int row, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * columns + i] + offset, num);
			}

			inline void store_row(T **ptr, const int row, const int offset, const int num, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * columns + i] + offset, num);
			}
			inline void store_row(SIMD<T> *ptr, const int row, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[row * columns + i];
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

			void scale_add_bias(T scale, SIMD<T> bias) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = mul_add(scale, data[i], bias);
			}
			void add_scaled_data(T scale, Line<T, Length> other) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = mul_add(scale, other[i], data[i]);
			}
	};

	template<typename T, int TransformSize, int KernelSize>
	struct WeightTransform
	{
	};
	template<typename T>
	struct WeightTransform<T, 2, 3>
	{
			inline Line<T, 4> operator()(Line<T, 3> line) const noexcept
			{
				Line<T, 6> result;
				result[0] = line[0];
				result[1] = line[0] + line[1] + line[2];
				result[2] = line[0] - line[1] + line[2];
				result[3] = line[2];
				return result;
			}
	};
	template<typename T>
	struct WeightTransform<T, 4, 3>
	{
			inline Line<T, 6> operator()(Line<T, 3> line) const noexcept
			{
				const SIMD<T> c13(1.0 / 3.0);
				const SIMD<T> c23(2.0 / 3.0);
				const SIMD<T> c2(2.0);

				Line<T, 6> result;
				result[0] = line[0];
				result[1] = mul_add(c23, line[0] + line[2], c23 * line[1]);
				result[2] = mul_sub(c23, line[0] + line[2], c23 * line[1]);
				result[3] = mul_add(c13, line[0] + line[2], c23 * line[1]);
				result[4] = mul_sub(c13, line[0] + line[2], c23 * line[1]);
				result[5] = c2 * line[2];
				return result;
			}
	};
	template<typename T>
	struct WeightTransform<T, 2, 5>
	{
			inline Line<T, 6> operator()(Line<T, 5> line) const noexcept
			{
				const SIMD<T> c13(1.0 / 3.0);
				const SIMD<T> c23(2.0 / 3.0);
				const SIMD<T> c2(2.0);

				Line<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3] + line[4]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3] + line[4]);
//				result[3] = c13 * (line[0] + c2 * line[1] + c4 * line[2]); TODO
//				result[4] = c13 * (line[0] - c2 * line[1] + c4 * line[2]); TODO
				result[5] = c2 * line[4];
				return result;
			}
	};

	template<typename T, int TransformSize, int KernelSize>
	struct InputTransform
	{
	};
	template<typename T>
	struct InputTransform<T, 2, 3>
	{
			inline Line<T, 4> operator()(Line<T, 4> line) const noexcept
			{
				Line<T, 4> result;
				result[0] = line[0] - line[2];
				result[1] = line[1] + line[2];
				result[2] = line[2] - line[1];
				result[3] = line[3] - line[1];
				return result;
			}
	};
	template<typename T>
	struct InputTransform<T, 4, 3>
	{
			inline Line<T, 6> operator()(Line<T, 6> line) const noexcept
			{
				const SIMD<T> c025(0.25);
				const SIMD<T> c05(0.5);

				Line<T, 6> result;
				result[0] = mul_add(c025, line[4] - line[2], line[0] - line[2]);
				result[1] = neg_mul_add(c025, line[3] + line[4], line[1] + line[2]);
				result[2] = mul_add(c025, line[3] - line[4], line[2] - line[1]);
				result[3] = mul_sub(c05, line[4] - line[2], line[1] - line[3]);
				result[4] = mul_add(c05, line[4] - line[2], line[1] - line[3]);
				result[5] = mul_add(c025, line[5] - line[3], line[1] - line[3]);
				return result;
			}
	};
	template<typename T>
	struct InputTransform<T, 2, 5>
	{
			inline Line<T, 6> operator()(Line<T, 6> line) const noexcept
			{
				return InputTransform<T, 4, 3>()(line); // it turns out that those two transforms are the same
			}
	};

	template<typename T, int TransformSize, int KernelSize>
	struct OutputTransform
	{
	};
	template<typename T>
	struct OutputTransform<T, 2, 3>
	{
			inline Line<T, 2> operator()(Line<T, 4> line) const noexcept
			{
				const SIMD<T> c05(0.5);

				Line<T, 2> result;
				result[0] = mul_add(c05, line[1] + line[2], line[0]);
				result[1] = mul_add(c05, line[1] - line[2], line[3]);
				return result;
			}
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
				result[0] = mul_add(c025, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c05, line[3] - line[4], line[1] - line[2]);
				result[2] = line[1] + line[2] + line[3] + line[4];
				result[3] = mul_add(c2, line[3] - line[4] + line[5], line[1] - line[2]);
				return result;
			}
	};
	template<typename T>
	struct OutputTransform<T, 2, 5>
	{
			inline Line<T, 2> operator()(Line<T, 6> line) const noexcept
			{
				const SIMD<T> c05(0.5);
				const SIMD<T> c2(2.0);

				Line<T, 2> result;
				result[0] = mul_add(c05, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c2, line[5], line[1] - line[2] + line[3] - line[4]);
				return result;
			}
	};

	template<typename T, int TransformSize, int KernelSize>
	struct GradientTransform
	{
	};
	template<typename T>
	struct GradientTransform<T, 2, 3>
	{
			inline Line<T, 4> operator()(Line<T, 2> line) const noexcept
			{
				Line<T, 4> result;
				result[0] = line[0];
				result[1] = line[0] + line[1];
				result[2] = line[0] - line[1];
				result[3] = line[1];
				return result;
			}
	};
	template<typename T>
	struct GradientTransform<T, 4, 3>
	{
			inline Line<T, 6> operator()(Line<T, 4> line) const noexcept
			{
				const SIMD<T> c13(1.0 / 3.0);
				const SIMD<T> c23(2.0 / 3.0);
				const SIMD<T> c2(2.0);

				Line<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3]);
				result[3] = mul_add(c13, line[0] + line[2], line[2]) + mul_add(c23, line[1] + line[3], c2 * line[3]);
				result[4] = mul_add(c13, line[0] + line[2], line[2]) - mul_add(c23, line[1] + line[3], c2 * line[3]);
				result[5] = c2 * line[3];
				return result;
			}
	};
	template<typename T>
	struct GradientTransform<T, 2, 5>
	{
			inline Line<T, 6> operator()(Line<T, 2> line) const noexcept
			{
				const SIMD<T> c13(1.0 / 3.0);
				const SIMD<T> c23(2.0 / 3.0);

				Line<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1]);
				result[2] = c23 * (line[0] - line[1]);
				result[3] = c13 * line[0] + c23 * line[1];
				result[4] = c13 * line[0] - c23 * line[1];
				result[5] = line[1];
				return result;
			}
	};

	template<typename T, int TransformSize, int KernelSize>
	struct UpdateTransform
	{
	};
	template<typename T>
	struct UpdateTransform<T, 2, 3>
	{
			inline Line<T, 3> operator()(Line<T, 4> line) const noexcept
			{
				const SIMD<T> c05(0.5);

				Line<T, 3> result;
				result[0] = mul_add(c05, line[1] + line[2], line[0]);
				result[1] = c05 * (line[1] - line[2]);
				result[2] = mul_add(c05, line[1] + line[2], line[3]);
				return result;
			}
	};
	template<typename T>
	struct UpdateTransform<T, 4, 3>
	{
			inline Line<T, 3> operator()(Line<T, 6> line) const noexcept
			{
				const SIMD<T> c025(0.25);
				const SIMD<T> c05(0.5);
				const SIMD<T> c2(2.0);

				Line<T, 3> result;
				result[0] = mul_add(c025, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c05, line[3] - line[4], line[1] - line[2]);
				result[2] = mul_add(c2, line[5], line[1] + line[2] + line[3] + line[4]);
				return result;
			}
	};
	template<typename T>
	struct UpdateTransform<T, 2, 5>
	{
			inline Line<T, 5> operator()(Line<T, 6> line) const noexcept
			{
				const SIMD<T> c025(0.25);
				const SIMD<T> c05(0.5);
				const SIMD<T> c2(2.0);
				const SIMD<T> c4(4.0);

				Line<T, 5> result;
				result[0] = mul_add(c025, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c05, line[3] - line[4], line[1] - line[2]);
				result[2] = line[1] + line[2] + line[3] + line[4];
				result[3] = mul_add(c2, line[3] - line[4], line[1] - line[2]);
				result[4] = mul_add(c4, line[3] + line[4] + line[5], line[1] + line[2]);
				return result;
			}
	};

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
						column.load_column(ptr_in, col, in, elements_left, KernelSize);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_column(storage, col, KernelSize);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<T, KernelSize> column;
						column.load_row(storage, col, KernelSize);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_row(ptr_out, col, in, elements_left, TileSize);
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

		const int tiles_h = (height + TransformSize - 1) / TransformSize;
		const int tiles_w = (width + TransformSize - 1) / TransformSize;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;

		T *zero_line = workspace;
		std::memset(zero_line, 0, sizeof(T) * filters);

#pragma omp parallel
		{
			const T *ptr_in[TileSize * TileSize];
			T *ptr_out[TileSize * TileSize];
			SIMD<T> storage[TileSize * TileSize];
#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				int batch = tile_idx / tiles_per_image;
				int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

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
	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void winograd_output_transform(const TensorDescriptor &yDesc, T *yMem, const TensorDescriptor &mDesc, const T *mMem, const T *bMem, const T *zMem,
			avActivationType_t activation, T alpha1, T alpha2, T beta, T *workspace)
	{
		const int batch_size = yDesc.dimension(0);
		const int height = yDesc.dimension(1);
		const int width = yDesc.dimension(2);
		const int filters = yDesc.dimension(3);

		const int tiles = mDesc.dimension(1);

		const int tiles_h = (height + TransformSize - 1) / TransformSize;
		const int tiles_w = (width + TransformSize - 1) / TransformSize;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;

		T *zero_line = workspace;
		std::memset(zero_line, 0, sizeof(T) * filters);

#pragma omp parallel
		{
			T *fake_storage = workspace + (1 + omp_get_thread_num()) * filters;

			SIMD<T> storage[TransformSize * TileSize];
			const T *ptr_in[TileSize * TileSize];
			T *ptr_out[TransformSize * TransformSize];
			const T *ptr_add[TransformSize * TransformSize];

#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				for (int j = 0; j < 36; j++)
					ptr_in[j] = mMem + mDesc.getIndex( { j, tile_idx, 0 });

				int batch = tile_idx / tiles_per_image;
				int tile_x = (tile_idx % tiles_per_image) / tiles_w;
				int tile_y = (tile_idx % tiles_per_image) % tiles_w;
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
							if (x < height and y < width and zMem != nullptr)
								ptr_add[output_idx] = zMem + yDesc.getIndex( { batch, x, y, 0 });
							else
								ptr_add[output_idx] = zero_line;
						}
				}

				OutputTransform<T, TransformSize, KernelSize> transform;
				for (int out = 0; out < filters; out += SIMD<T>::length)
				{
					const int elements_left = std::min(filters - out, SIMD<T>::length);
					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TileSize> column;
						column.load_column(ptr_in, col, out, elements_left, TileSize);
						Line<T, TransformSize> transformed = transform(column);
						transformed.store_column(storage, col, TileSize);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TileSize> column;
						column.load_row(storage, col, TileSize);
						Line<T, TransformSize> transformed = transform(column);

						SIMD<T> bias;
						if (bMem != nullptr)
							(bMem + out, elements_left);
						else
							bias = SIMD<T>::zero();

						transformed.scale_add_bias(alpha1, bias);

						if (zMem != nullptr)
						{
							Line<T, TransformSize> z_line;
							z_line.load_row(ptr_add, col, out, elements_left, TransformSize);
							transformed.add_scaled_data(alpha2, z_line);
						}
						transformed = activation_forward(activation, transformed);

						if (beta != scalar::zero<T>())
						{
							Line<T, TransformSize> dst_line;
							dst_line.load_row(ptr_out, col, out, elements_left, TransformSize);
							transformed.add_scaled_data(beta, dst_line);
						}

						transformed.store_row(ptr_out, col, out, elements_left, TransformSize);
					}
				}
			}
		}
	}
	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void winograd_gradient_transform(const TensorDescriptor &dyDesc, const T *dyMem, const TensorDescriptor &mDesc, T *mMem, T *workspace)
	{
		const int batch_size = dyDesc.dimension(0);
		const int height = dyDesc.dimension(1);
		const int width = dyDesc.dimension(2);
		const int filters = dyDesc.dimension(3);

		const int tiles = mDesc.dimension(1);

		const int tiles_h = (height + TransformSize - 1) / TransformSize;
		const int tiles_w = (width + TransformSize - 1) / TransformSize;
		const int tiles_per_image = tiles_h * tiles_w;
		const int nb_of_tiles = batch_size * tiles_per_image;

		T *zero_line = workspace;
		std::memset(zero_line, 0, sizeof(T) * filters);

#pragma omp parallel
		{
			const T *ptr_in[TileSize * TileSize];
			T *ptr_out[TileSize * TileSize];
			SIMD<T> storage[TileSize * TileSize];
#pragma omp for
			for (int tile_idx = 0; tile_idx < nb_of_tiles; tile_idx++)
			{
				int batch = tile_idx / tiles_per_image;
				int tile_x = ((tile_idx % tiles_per_image) / tiles_w);
				int tile_y = ((tile_idx % tiles_per_image) % tiles_w);

				int matrix_idx = 0;
				for (int i = 0; i < TileSize; i++)
					for (int j = 0; j < TileSize; j++, matrix_idx++)
					{
						int x = TransformSize * tile_x + i;
						int y = TransformSize * tile_y + j;
						ptr_out[matrix_idx] = mMem + mDesc.getIndex( { matrix_idx, tile_idx, 0 });
						if (x < height and y < width)
							ptr_in[matrix_idx] = dyMem + dyDesc.getIndex( { batch, x, y, 0 });
						else
							ptr_in[matrix_idx] = zero_line;
					}

				GradientTransform<T, TransformSize, KernelSize> transform;
				for (int out = 0; out < filters; out += SIMD<T>::length)
				{
					const int elements_left = std::min(filters - out, SIMD<T>::length);
					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TransformSize> column;
						column.load_column(ptr_in, col, out, elements_left, TransformSize);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_column(storage, col, TransformSize);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TransformSize> column;
						column.load_row(storage, col, TransformSize);
						Line<T, TileSize> transformed = transform(column);
						transformed.store_row(ptr_out, col, out, elements_left, TileSize);
					}
				}
			}
		}
	}
	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void winograd_update_transform(const TensorDescriptor &wDesc, const T *wMem, const TensorDescriptor &mDesc, T *mMem, T alpha, T beta)
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
				int matrix_index = 0;
				for (int i = 0; i < KernelSize; i++)
					for (int j = 0; j < KernelSize; j++, matrix_index++)
						ptr_in[matrix_index] = wMem + wDesc.getIndex( { out, i, j, 0 });

				for (int k = 0; k < TileSize * TileSize; k++)
					ptr_out[k] = mMem + mDesc.getIndex( { k, out, 0 });

				UpdateTransform<T, TransformSize, KernelSize> transform;
				for (int in = 0; in < filtersIn; in += SIMD<T>::length)
				{
					const int elements_left = std::min(filtersIn - in, SIMD<T>::length);
					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TileSize> column;
						column.load_column(ptr_in, col, in, elements_left, TileSize);
						Line<T, KernelSize> transformed = transform(column);
						transformed.store_column(storage, col, TileSize);
					}

					for (int col = 0; col < TileSize; col++)
					{
						Line<T, TileSize> column;
						column.load_row(storage, col, TileSize);
						Line<T, KernelSize> transformed = transform(column);
						transformed.store_row(ptr_out, col, in, elements_left, KernelSize);
					}
				}
			}
		}
	}

}

namespace avocado
{
	namespace backend
	{

	} /* namespace backend */
} /* namespace avocado */

