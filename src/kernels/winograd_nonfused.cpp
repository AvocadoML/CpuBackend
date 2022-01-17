/*
 * winograd_nonfused.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "activation.hpp"
#include "../vectors/simd_vectors.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<typename T, int Length>
	struct Line
	{
		private:
			SIMD<T> data[Length];
		public:
			inline void load_row(T **ptr, const int row, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * columns + i] + offset, num);
			}
			inline void load_row(const T **ptr, const int row, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * columns + i] + offset, num);
			}
			inline void store_row(T **ptr, const int row, const int offset, const int num, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].store(ptr[row * columns + i] + offset, num);
			}
			inline void load_column(const T **ptr, const int col, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[i * columns + col] + offset, num);
			}
			inline void store_column(T **ptr, const int col, const int offset, const int num, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].store(ptr[i * columns + col] + offset, num);
			}

			inline void load_row(const SIMD<T> *ptr, const int row, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[row * columns + i];
			}
			inline void store_row(SIMD<T> *ptr, const int row, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					ptr[row * columns + i] = data[i];
			}
			inline void load_column(const SIMD<T> *ptr, const int col, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[i * columns + col];
			}
			inline void store_column(SIMD<T> *ptr, const int col, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					ptr[i * columns + col] = data[i];
			}

			inline int length() const noexcept
			{
				return Length;
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
	struct WeightTransform<T, 2, 3>
	{
			inline Line<T, 4> operator()(Line<T, 3> line) const noexcept
			{
				Line<T, 4> result;
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
				const SIMD<T> c16(1.0 / 6.0);
				const SIMD<T> c13(1.0 / 3.0);
				const SIMD<T> c23(2.0 / 3.0);
				const SIMD<T> c2(2.0);

				Line<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3] + line[4]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3] + line[4]);
				result[3] = c16 * line[0] + mul_add(c13, line[1] + line[3], line[3]) + mul_add(c13, line[2] + line[4], c2 * line[4]);
				result[4] = c16 * line[0] - mul_add(c13, line[1] + line[3], line[3]) + mul_add(c13, line[2] + line[4], c2 * line[4]);
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
	void kernel_weight_transform(const cpu::TensorDescriptor &wDesc, const T *wMem, const cpu::TensorDescriptor &mDesc, T *mMem, bool invert)
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
	void kernel_input_transform(const cpu::TensorDescriptor &xDesc, const T *xMem, const cpu::TensorDescriptor &mDesc, T *mMem,
			std::array<int, 3> padding, T *workspace)
	{
		const int batch_size = xDesc.dimension(0);
		const int height = xDesc.dimension(1);
		const int width = xDesc.dimension(2);
		const int filters = xDesc.dimension(3);

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
						int x = padding[0] + TransformSize * tile_x + i;
						int y = padding[1] + TransformSize * tile_y + j;
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
	template<typename T, typename U, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void kernel_output_transform(const cpu::TensorDescriptor &yDesc, T *yMem, const cpu::TensorDescriptor &mDesc, const T *mMem, const T *bMem,
			const T *zMem, avActivationType_t activation, U alpha1, U alpha2, U beta, T *workspace)
	{
		const int batch_size = yDesc.dimension(0);
		const int height = yDesc.dimension(1);
		const int width = yDesc.dimension(2);
		const int filters = yDesc.dimension(3);

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
							bias.load(bMem + out, elements_left);
						else
							bias = SIMD<T>::zero();

						for (int i = 0; i < transformed.length(); i++)
							transformed[i] = mul_add(alpha1, transformed[i], bias);

						if (zMem != nullptr)
						{
							Line<T, TransformSize> z_line;
							z_line.load_row(ptr_add, col, out, elements_left, TransformSize);
							for (int i = 0; i < transformed.length(); i++)
								transformed[i] = mul_add(alpha2, z_line[i], transformed[i]);
						}
						for (int i = 0; i < transformed.length(); i++)
							transformed[i] = activation_forward(activation, transformed[i]);

						if (beta != scalar::zero<U>())
						{
							Line<T, TransformSize> dst_line;
							dst_line.load_row(ptr_out, col, out, elements_left, TransformSize);
							for (int i = 0; i < transformed.length(); i++)
								transformed[i] = mul_add(beta, dst_line[i], transformed[i]);
						}

						transformed.store_row(ptr_out, col, out, elements_left, TransformSize);
					}
				}
			}
		}
	}
	template<typename T, int TransformSize, int KernelSize, int TileSize = TransformSize + KernelSize - 1>
	void kernel_gradient_transform(const cpu::TensorDescriptor &dyDesc, const T *dyMem, const cpu::TensorDescriptor &mDesc, T *mMem, T *workspace)
	{
		const int batch_size = dyDesc.dimension(0);
		const int height = dyDesc.dimension(1);
		const int width = dyDesc.dimension(2);
		const int filters = dyDesc.dimension(3);

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
	void kernel_update_transform(const cpu::TensorDescriptor &dwDesc, T *dwMem, const cpu::TensorDescriptor &mDesc, const T *mMem, T alpha, T beta)
	{
		const int filtersOut = dwDesc.firstDim();
		const int filtersIn = dwDesc.lastDim();

#pragma omp parallel
		{
			const T *ptr_in[TileSize * TileSize];
			T *ptr_out[KernelSize * KernelSize];
			SIMD<T> storage[TileSize * KernelSize];
#pragma omp for
			for (int out = 0; out < filtersOut; out++)
			{
				int matrix_index = 0;
				for (int i = 0; i < KernelSize; i++)
					for (int j = 0; j < KernelSize; j++, matrix_index++)
						ptr_out[matrix_index] = dwMem + dwDesc.getIndex( { out, i, j, 0 });

				for (int k = 0; k < TileSize * TileSize; k++)
					ptr_in[k] = mMem + mDesc.getIndex( { k, out, 0 });

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

					for (int col = 0; col < KernelSize; col++)
					{
						Line<T, TileSize> column;
						column.load_row(storage, col, TileSize);
						Line<T, KernelSize> transformed = transform(column);

						for (int i = 0; i < KernelSize; i++)
							transformed[i] *= alpha;

						if (beta != scalar::zero<T>())
						{
							Line<T, KernelSize> dst_line;
							dst_line.load_row(ptr_out, col, out, elements_left, KernelSize);
							for (int i = 0; i < transformed.length(); i++)
								transformed[i] = mul_add(beta, dst_line[i], transformed[i]);
						}

						transformed.store_row(ptr_out, col, in, elements_left, KernelSize);
					}
				}
			}
		}
	}

	int getWinogradWorkspace(int transformSize, const cpu::TensorDescriptor &xDesc, const cpu::TensorDescriptor &yDesc,
			const cpu::TensorDescriptor &wDesc)
	{
		assert(wDesc.dimension(1) == wDesc.dimension(2));
		assert(wDesc.nbDims() == 3 || wDesc.nbDims() == 4 || wDesc.nbDims() == 5);
		int tile_size = transformSize + wDesc.dimension(1) - 1;
		int nb_of_matrices = tile_size * tile_size;

		int nb_of_tiles = xDesc.firstDim();
		for (int i = 0; i < wDesc.nbDims() - 2; i++)
			nb_of_tiles *= (xDesc.dimension(1 + i) + tile_size - 1) / tile_size;

		int weight_matrix_size = cpu::dataTypeSize(wDesc.dtype()) * wDesc.firstDim() * wDesc.lastDim();
		int input_matrix_size = cpu::dataTypeSize(xDesc.dtype()) * nb_of_tiles * xDesc.lastDim();
		int output_matrix_size = cpu::dataTypeSize(yDesc.dtype()) * nb_of_tiles * yDesc.lastDim();

		return nb_of_matrices * (weight_matrix_size + input_matrix_size + output_matrix_size);
	}

	bool is_conv(int expectedSize, const cpu::TensorDescriptor &wDesc)
	{
		for (int i = 0; i < wDesc.nbDims() - 2; i++)
			if (wDesc.dimension(1 + i) != expectedSize)
				return false;
		return true;
	}

	template<typename T>
	avStatus_t launch_weight_transform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
			const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
	{
		bool invert = cpu::getConvolution(config).mode == AVOCADO_CROSS_CORRELATION_MODE;
		if (is_conv(3, cpu::getTensor(wDesc)))
		{
			kernel_weight_transform<T, 4, 3>(cpu::getTensor(wDesc), cpu::getPointer<T>(wMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), invert);
			return AVOCADO_STATUS_SUCCESS;
		}
		if (is_conv(5, cpu::getTensor(wDesc)))
		{
			kernel_weight_transform<T, 2, 5>(cpu::getTensor(wDesc), cpu::getPointer<T>(wMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), invert);
			return AVOCADO_STATUS_SUCCESS;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
	template<typename T>
	avStatus_t launch_input_transform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
			const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
			const avTensorDescriptor_t wDesc)
	{
		const int input_filters = cpu::getTensor(wDesc).lastDim();
		if (cpu::getContext(context).getWorkspace().size() < static_cast<int>(sizeof(T)) * input_filters)
			return AVOCADO_STATUS_INTERNAL_ERROR;

		if (is_conv(3, cpu::getTensor(wDesc)))
		{
			kernel_input_transform<T, 4, 3>(cpu::getTensor(xDesc), cpu::getPointer<T>(xMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getConvolution(config).padding, cpu::getContext(context).getWorkspace().data<T>());
			return AVOCADO_STATUS_SUCCESS;
		}
		if (is_conv(5, cpu::getTensor(wDesc)))
		{
			kernel_input_transform<T, 2, 5>(cpu::getTensor(xDesc), cpu::getPointer<T>(xMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getConvolution(config).padding, cpu::getContext(context).getWorkspace().data<T>());
			return AVOCADO_STATUS_SUCCESS;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
	template<typename T, typename U = T>
	avStatus_t launch_output_transform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
			const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc,
			avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2,
			const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation,
			const avTensorDescriptor_t wDesc)
	{
		const int input_filters = cpu::getTensor(wDesc).lastDim();
		if (cpu::getContext(context).getWorkspace().size() < static_cast<int>(sizeof(T)) * input_filters * (1 + cpuGetNumberOfThreads()))
			return AVOCADO_STATUS_INTERNAL_ERROR;

		if (is_conv(3, cpu::getTensor(wDesc)))
		{
			kernel_output_transform<T, U, 4, 3>(cpu::getTensor(yDesc), cpu::getPointer<T>(yMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getPointer<T>(bMem), cpu::getPointer<T>(zMem), activation, cpu::getAlphaValue<U>(alpha1),
					cpu::getAlphaValue<U>(alpha2), cpu::getBetaValue<U>(beta), cpu::getContext(context).getWorkspace().data<T>());
			return AVOCADO_STATUS_SUCCESS;
		}
		if (is_conv(5, cpu::getTensor(wDesc)))
		{
			kernel_output_transform<T, U, 2, 5>(cpu::getTensor(yDesc), cpu::getPointer<T>(yMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getPointer<T>(bMem), cpu::getPointer<T>(zMem), activation, cpu::getAlphaValue<U>(alpha1),
					cpu::getAlphaValue<U>(alpha2), cpu::getBetaValue<U>(beta), cpu::getContext(context).getWorkspace().data<T>());
			return AVOCADO_STATUS_SUCCESS;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
	template<typename T>
	avStatus_t launch_gradient_transform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
			const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
			const avTensorDescriptor_t wDesc)
	{
		const int input_filters = cpu::getTensor(wDesc).lastDim();
		if (cpu::getContext(context).getWorkspace().size() < static_cast<int>(sizeof(T)) * input_filters)
			return AVOCADO_STATUS_INTERNAL_ERROR;

		if (is_conv(3, cpu::getTensor(wDesc)))
		{
			kernel_gradient_transform<T, 4, 3>(cpu::getTensor(dyDesc), cpu::getPointer<T>(dyMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getContext(context).getWorkspace().data<T>());
			return AVOCADO_STATUS_SUCCESS;
		}
		if (is_conv(5, cpu::getTensor(wDesc)))
		{
			kernel_gradient_transform<T, 2, 5>(cpu::getTensor(dyDesc), cpu::getPointer<T>(dyMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getContext(context).getWorkspace().data<T>());
			return AVOCADO_STATUS_SUCCESS;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
	template<typename T>
	avStatus_t launch_update_transform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
			const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
			avMemoryDescriptor_t dwMem)
	{
		if (is_conv(3, cpu::getTensor(dwDesc)))
		{
			kernel_update_transform<T, 4, 3>(cpu::getTensor(dwDesc), cpu::getPointer<T>(dwMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getAlphaValue<T>(alpha), cpu::getBetaValue<T>(beta));
			return AVOCADO_STATUS_SUCCESS;
		}
		if (is_conv(5, cpu::getTensor(dwDesc)))
		{
			kernel_update_transform<T, 2, 5>(cpu::getTensor(dwDesc), cpu::getPointer<T>(dwMem), cpu::getTensor(matricesDesc),
					cpu::getPointer<T>(matricesMem), cpu::getAlphaValue<T>(alpha), cpu::getBetaValue<T>(beta));
			return AVOCADO_STATUS_SUCCESS;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
}

#if DYNAMIC_ARCH == 0 or (DYNAMIC_ARCH == 1 and defined(COMPILE_COMMON_CODE))
namespace avocado
{
	namespace backend
	{
		avSize_t winogradGetWorkspaceSize(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &xDesc,
				const cpu::TensorDescriptor &yDesc, const cpu::TensorDescriptor &wDesc)
		{
			if (config.algorithm == AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD)
			{
				switch (wDesc.dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
					case AVOCADO_DTYPE_BFLOAT16:
					case AVOCADO_DTYPE_FLOAT32:
					case AVOCADO_DTYPE_FLOAT64:
					{
						if (is_conv(3, wDesc))
							return getWinogradWorkspace(4, xDesc, yDesc, wDesc);
						if (is_conv(5, wDesc))
							return getWinogradWorkspace(2, xDesc, yDesc, wDesc);
						return 0;
					}
					default:
						return 0;
				}
			}
			return 0;
		}
	} /* namespace backend */
} /* namespace avocado */
#endif

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_winogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
			const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
	{
		switch (cpu::getTensor(wDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				return launch_weight_transform<float16>(context, config, wDesc, wMem, matricesDesc, matricesMem);
			case AVOCADO_DTYPE_BFLOAT16:
				return launch_weight_transform<bfloat16>(context, config, wDesc, wMem, matricesDesc, matricesMem);
			case AVOCADO_DTYPE_FLOAT32:
				return launch_weight_transform<float>(context, config, wDesc, wMem, matricesDesc, matricesMem);
			case AVOCADO_DTYPE_FLOAT64:
				return launch_weight_transform<double>(context, config, wDesc, wMem, matricesDesc, matricesMem);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
	avStatus_t cpu_winogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
			const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
			const avTensorDescriptor_t wDesc)
	{
		switch (cpu::getTensor(xDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				return launch_input_transform<float16>(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
			case AVOCADO_DTYPE_BFLOAT16:
				return launch_input_transform<bfloat16>(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
			case AVOCADO_DTYPE_FLOAT32:
				return launch_input_transform<float>(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
			case AVOCADO_DTYPE_FLOAT64:
				return launch_input_transform<double>(context, config, xDesc, xMem, matricesDesc, matricesMem, wDesc);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	avStatus_t cpu_winogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
			const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc,
			avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2,
			const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation,
			const avTensorDescriptor_t wDesc)
	{
		switch (cpu::getTensor(yDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				return launch_output_transform<float16, float>(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2,
						zDesc, zMem, beta, activation, wDesc);
			case AVOCADO_DTYPE_BFLOAT16:
				return launch_output_transform<bfloat16, float>(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2,
						zDesc, zMem, beta, activation, wDesc);
			case AVOCADO_DTYPE_FLOAT32:
				return launch_output_transform<float>(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc,
						zMem, beta, activation, wDesc);
			case AVOCADO_DTYPE_FLOAT64:
				return launch_output_transform<double>(context, config, alpha1, matricesDesc, matricesMem, yDesc, yMem, bDesc, bMem, alpha2, zDesc,
						zMem, beta, activation, wDesc);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	avStatus_t cpu_winogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
			const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem,
			const avTensorDescriptor_t wDesc)
	{
		switch (cpu::getTensor(dyDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
				return launch_gradient_transform<float>(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
			case AVOCADO_DTYPE_FLOAT64:
				return launch_gradient_transform<double>(context, config, dyDesc, dyMem, matricesDesc, matricesMem, wDesc);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	avStatus_t cpu_winogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
			const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
			avMemoryDescriptor_t dwMem)
	{
		switch (cpu::getTensor(dwDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
				return launch_update_transform<float>(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
			case AVOCADO_DTYPE_FLOAT64:
				return launch_update_transform<double>(context, config, alpha, matricesDesc, matricesMem, beta, dwDesc, dwMem);
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

