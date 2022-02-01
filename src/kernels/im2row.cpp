/*
 * im2row.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "../vectors/simd_macros.hpp"

namespace
{
	using namespace avocado::backend;

	void kernel_im2row_2d(const cpu::ConvolutionDescriptor &config, const cpu::TensorDescriptor &rowDesc, cpu::MemoryDescriptor &rowMem,
			const cpu::TensorDescriptor &srcDesc, const cpu::MemoryDescriptor &srcMem, const cpu::TensorDescriptor &filterDesc,
			cpu::MemoryDescriptor &workspace)
	{
		assert(rowMem.data() != nullptr);
		assert(srcMem.data() != nullptr);

		const int batch_size = srcDesc.dimension(0);
		const int input_height = srcDesc.dimension(1);
		const int input_width = srcDesc.dimension(2);

		const int filter_height = filterDesc.dimension(1);
		const int filter_width = filterDesc.dimension(2);
		const int input_filters = filterDesc.lastDim();

		const int padding_h = config.padding[0];
		const int padding_w = config.padding[1];

		const int stride_h = config.stride[0];
		const int stride_w = config.stride[1];

		const int dilation_h = config.dilation[0];
		const int dilation_w = config.dilation[1];

		const bool no_dilation = (dilation_h == 1) and (dilation_w == 1);

		const int dtype_size = cpu::dataTypeSize(srcDesc.dtype());
		const bool zero_padding = config.paddingWithZeros();

		const int filter_size_in_bytes = input_filters * dtype_size;
		const int row_size_in_bytes = filter_width * filter_size_in_bytes;

		if (not zero_padding)
		{
			for (int i = 0; i < row_size_in_bytes; i += dtype_size)
				std::memcpy(workspace.data<uint8_t>() + i, config.padding_value.data(), dtype_size);
		}

		cpu::TensorDescriptor output_shape = config.getOutputShape(srcDesc, filterDesc);

#pragma omp parallel for
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < output_shape.dimension(1); h++)
				for (int w = 0; w < output_shape.dimension(2); w++)
				{
					int tile_idx = (b * output_shape.dimension(1) + h) * output_shape.dimension(2) + w;
					uint8_t *dst_ptr = rowMem.data<uint8_t>() + rowDesc.getIndex( { tile_idx, 0 }) * dtype_size;
					for (int i = 0; i < filter_height; i++)
					{
						int x, y;
						if (config.mode == AVOCADO_CONVOLUTION_MODE)
						{
							x = padding_h + i * dilation_h + h * stride_h;
							y = padding_w + 0 * dilation_w + w * stride_w;
						}
						else
						{
							x = padding_h + (filter_height - 1 - i) * dilation_h + h * stride_h;
							y = padding_w + (filter_width - 1 - 0) * dilation_w + w * stride_w;
						}

						if (x >= 0 and x < input_height)
						{
							const uint8_t *src_ptr = srcMem.data<uint8_t>();

							if (y >= 0 and (y + filter_width) <= input_width and config.mode == AVOCADO_CONVOLUTION_MODE and no_dilation)
							{ // copy entire row at once
								std::memcpy(dst_ptr, src_ptr + srcDesc.getIndex( { b, x, y, 0 }) * dtype_size, row_size_in_bytes);
								dst_ptr += row_size_in_bytes;
							}
							else
							{ // copy each point separately
								for (int j = 0; j < filter_width; j++)
								{
									if (y >= 0 and y < input_width)
										std::memcpy(dst_ptr, src_ptr + srcDesc.getIndex( { b, x, y, 0 }) * dtype_size, filter_size_in_bytes);
									else
									{
										if (zero_padding)
											std::memset(dst_ptr, 0, filter_size_in_bytes);
										else
											std::memcpy(dst_ptr, workspace.data(), filter_size_in_bytes);
									}
									dst_ptr += input_filters * dtype_size;
									if (config.mode == AVOCADO_CONVOLUTION_MODE)
										y += dilation_w;
									else
										y -= dilation_w;
								}
							}
						}
						else
						{ // set entire row with padding value
							if (zero_padding)
								std::memset(dst_ptr, 0, row_size_in_bytes);
							else
								std::memcpy(dst_ptr, workspace.data(), row_size_in_bytes);
							dst_ptr += row_size_in_bytes;
						}
					}
				}
	}

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_im2row(const ContextDescriptor &context, const ConvolutionDescriptor &config, const TensorDescriptor &filterDesc,
			const TensorDescriptor &srcDesc, const MemoryDescriptor &srcMem, const TensorDescriptor &rowDesc, MemoryDescriptor &rowMem)
	{
		switch (filterDesc.nbDims())
		{
//			case 3: // 1D convolution
//				kernel_im2row_2d(config, rowDesc, rowMem, srcDesc, srcMem, filterDesc, context.getWorkspace());
//				return AVOCADO_STATUS_SUCCESS;
			case 4: // 2D convolution
				kernel_im2row_2d(config, rowDesc, rowMem, srcDesc, srcMem, filterDesc, context.getWorkspace());
				return AVOCADO_STATUS_SUCCESS;
			case 5: // 3D convolution
				return AVOCADO_STATUS_NOT_SUPPORTED; // TODO
			default:
				return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

