/*
 * basic_math.cpp
 *
 *  Created on: Nov 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cpu_backend.h>

#include <vector_types.h>

namespace
{
	using namespace avocado::backend;

//	template<typename T, int dim>
//	void kernel_transpose(T *dst, const T *src, const int *src_shape, const int *ordering)
//	{
//		int volume = 1;
//		for (int i = 0; i < dim; i++)
//			volume *= src_shape[i];
//
//		int src_stride[dim];
//		int dst_stride[dim];
//		int tmp_src = 1, tmp_dst = 1;
//		for (int i = dim - 1; i >= 0; i--)
//		{
//			src_stride[i] = tmp_src;
//			dst_stride[ordering[i]] = tmp_dst;
//			tmp_src *= src_shape[i];
//			tmp_dst *= src_shape[ordering[i]];
//		}
//
//		for (int i = 0; i < volume; i++)
//		{
//			int tmp = i, dst_idx = 0;
//			for (int j = 0; j < dim; j++)
//			{
//				int idx = tmp / src_stride[j];
//				dst_idx += idx * dst_stride[j];
//				tmp -= idx * src_stride[j];
//			}
//			dst[dst_idx] = src[i];
//		}
//	}
//	template<int dim>
//	void transpose_helper(avTensor_t dst, const avTensor_t src, const int *ordering)
//	{
//		switch (dataTypeSize(dst->dtype))
//		{
//			case 1:
//				kernel_transpose<int8_t, dim>(data<int8_t>(dst), data<int8_t>(src), src->shape.dim, ordering);
//				break;
//			case 2:
//				kernel_transpose<int16_t, dim>(data<int16_t>(dst), data<int16_t>(src), src->shape.dim, ordering);
//				break;
//			case 4:
//				kernel_transpose<int32_t, dim>(data<int32_t>(dst), data<int32_t>(src), src->shape.dim, ordering);
//				break;
//			case 8:
//				kernel_transpose<int64_t, dim>(data<int64_t>(dst), data<int64_t>(src), src->shape.dim, ordering);
//				break;
//			case 16:
//				kernel_transpose<int4, dim>(data<int4>(dst), data<int4>(src), src->shape.dim, ordering);
//				break;
//		}
//	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t cpuConcatTensors(avContext_t context, avTensor_t dst, const avTensor_t src, avSize_t lastDimOffsetInBytes)
//		{
//			assert(context != nullptr);
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			const avSize_t type_size = dataTypeSize(dst->dtype);
//			const avSize_t first_dim = volumeWithoutLastDim(dst);
//			const avSize_t last_dim = lastDim(dst);
//			for (avSize_t i = 0; i < first_dim; i++)
//			{
//				char *ptr_dst = data<char>(dst) + (i * last_dim + lastDimOffsetInBytes) * type_size;
//				const char *ptr_src = data<char>(src) + i * lastDim(src) * type_size;
//				std::memcpy(ptr_dst, ptr_src, lastDim(src) * type_size);
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t cpuSplitTensors(avContext_t context, avTensor_t dst, const avTensor_t src, avSize_t lastDimOffsetInBytes)
//		{
//			assert(context != nullptr);
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			const avSize_t type_size = dataTypeSize(src->dtype);
//			const avSize_t first_dim = volumeWithoutLastDim(src);
//			const avSize_t last_dim = lastDim(src);
//			for (avSize_t i = 0; i < first_dim; i++)
//			{
//				char *ptr_dst = data<char>(dst) + i * lastDim(dst) * type_size;
//				const char *ptr_src = data<char>(src) + (i * last_dim + lastDimOffsetInBytes) * type_size;
//				std::memcpy(ptr_dst, ptr_src, lastDim(dst) * type_size);
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t cpuTranspose(avContext_t context, avTensor_t dst, const avTensor_t src, const int *order)
//		{
//			assert(context != nullptr);
//			assert(dst != nullptr);
//			assert(src != nullptr);
//			assert(order != nullptr);
//			switch (numberOfDimensions(dst))
//			{
//				case 2:
//					transpose_helper<2>(dst, src, order);
//					break;
//				case 3:
//					transpose_helper<3>(dst, src, order);
//					break;
//				case 4:
//					transpose_helper<4>(dst, src, order);
//					break;
//				case 5:
//					transpose_helper<5>(dst, src, order);
//					break;
//				case 6:
//					transpose_helper<6>(dst, src, order);
//					break;
//				case 7:
//					transpose_helper<7>(dst, src, order);
//					break;
//				case 8:
//					transpose_helper<8>(dst, src, order);
//					break;
//				default:
//					return AVOCADO_STATUS_NOT_SUPPORTED;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
	} /* namespace backend */
} /* namespace avocado */

