/*
 * tensor_op.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <avocado/backend/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	struct int4
	{
			int x, y, z, w;
	};

	void kernel_concat_tensors(const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc[],
			const avMemoryDescriptor_t aMem[], int nbTensors)
	{
		const int64_t dtype_size = dataTypeSize(getTensor(cDesc).dtype());
		const int64_t first_dim = getTensor(cDesc).volumeWithoutLastDim();

		const int64_t dst_last_dim = getTensor(cDesc).lastDim() * dtype_size;

#pragma omp parallel
		{
			int64_t last_dim_offset = 0;
			for (int k = 0; k < nbTensors; k++)
			{
				const int64_t src_last_dim = getTensor(aDesc[k]).lastDim() * dtype_size;

#pragma omp for nowait
				for (int64_t i = 0; i < first_dim; i++)
					std::memcpy(getPointer<uint8_t>(cMem) + i * dst_last_dim, getPointer<uint8_t>(aMem[k]) + i * src_last_dim, src_last_dim);
				last_dim_offset += src_last_dim;
			}
		}
	}

	void kernel_split_tensors(const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[], const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, int nbTensors)
	{
		const int64_t dtype_size = dataTypeSize(getTensor(aDesc).dtype());
		const avSize_t first_dim = getTensor(aDesc).volumeWithoutLastDim();

		const avSize_t src_last_dim = getTensor(aDesc).lastDim() * dtype_size;

#pragma omp parallel
		{
			int64_t last_dim_offset = 0;
			for (int k = 0; k < nbTensors; k++)
			{
				const int64_t dst_last_dim = getTensor(cDesc[k]).lastDim() * dtype_size;

#pragma omp for nowait
				for (int64_t i = 0; i < first_dim; i++)
					std::memcpy(getPointer<uint8_t>(cMem[k]) + i * dst_last_dim, getPointer<uint8_t>(aMem) + i * src_last_dim, dst_last_dim);
				last_dim_offset += dst_last_dim;
			}
		}
	}

	template<typename T>
	void kernel_transpose(T *dst, const T *src, const TensorDescriptor &src_shape, const int ordering[])
	{
		const int src_volume = src_shape.volume();
		const int dimension = src_shape.nbDims();

		std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> src_stride;
		std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> dst_stride;

		int tmp_src = 1, tmp_dst = 1;
		for (int i = dimension - 1; i >= 0; i--)
		{
			src_stride[i] = tmp_src;
			dst_stride[ordering[i]] = tmp_dst;
			tmp_src *= src_shape.dimension(i);
			tmp_dst *= src_shape.dimension(ordering[i]);
		}

#pragma omp parallel for
		for (int i = 0; i < src_volume; i++)
		{
			int src_idx = i, dst_idx = 0;
			for (int j = 0; j < dimension; j++)
			{
				int tmp = src_idx / src_stride[j];
				dst_idx += tmp * dst_stride[j];
				src_idx -= tmp * src_stride[j];
			}
			dst[dst_idx] = src[i];
		}
	}

	template<typename T, typename U>
	void kernel_scale_tensor(T *dst, U value, int elements) noexcept
	{
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> data(dst + i, elements_left);
			data = data * value;
			data.store(dst + i, elements_left);
		}
	}
	template<typename T, typename U>
	void kernel_add_scalar_to_tensor(T *dst, U scalar, int elements) noexcept
	{
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> data(dst + i, elements_left);
			data = data + scalar;
			data.store(dst + i, elements_left);
		}
	}

	template<typename T, typename U, typename V>
	void kernel_add_bias(T *dst, U alpha3, U alpha1, const V *src1, U alpha2, const U *src2, U beta, BroadcastedDimensions dims,
			avActivationType_t type) noexcept
	{
//		if (beta == zero<U>())
//			clear(dst, volume(dims));
//
//		for (int64_t i = 0; i < dims.first; i++)
//			for (int64_t j = 0; j < dims.last; j++)
//			{
//				U lhs = alpha1 * static_cast<U>(src1[i * dims.last + j]);
//				U rhs = alpha2 * static_cast<U>(src2[j]);
//				U tmp = activation_forward(type, lhs + rhs);
//				dst[i * dims.last + j] = Store<T, U>::store(alpha3 * tmp + beta * static_cast<U>(dst[i * dims.last + j]));
//			}
	}

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t concatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
			const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
	{
		switch (dataTypeSize(getTensor(cDesc).dtype()))
		{
			case 1:
			case 2:
			case 4:
			case 8:
			case 16:
				kernel_concat_tensors(cDesc, cMem, aDesc, aMem, nbTensors);
				return AVOCADO_STATUS_SUCCESS;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
	}

	avStatus_t splitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
			const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
	{
		switch (dataTypeSize(getTensor(aDesc).dtype()))
		{
			case 1:
			case 2:
			case 4:
			case 8:
			case 16:
				kernel_split_tensors(cDesc, cMem, aDesc, aMem, nbTensors);
				return AVOCADO_STATUS_SUCCESS;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
	}

	avStatus_t transpose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, const int newDimOrder[])
	{
		switch (dataTypeSize(getTensor(aDesc).dtype()))
		{
			case 1:
				kernel_transpose<int8_t>(getPointer<int8_t>(cMem), getPointer<int8_t>(aMem), getTensor(aDesc), newDimOrder);
				break;
			case 2:
				kernel_transpose<int16_t>(getPointer<int16_t>(cMem), getPointer<int16_t>(aMem), getTensor(aDesc), newDimOrder);
				break;
			case 4:
				kernel_transpose<int32_t>(getPointer<int32_t>(cMem), getPointer<int32_t>(aMem), getTensor(aDesc), newDimOrder);
				break;
			case 8:
				kernel_transpose<int64_t>(getPointer<int64_t>(cMem), getPointer<int64_t>(aMem), getTensor(aDesc), newDimOrder);
				break;
			case 16:
				kernel_transpose<int4>(getPointer<int4>(cMem), getPointer<int4>(aMem), getTensor(aDesc), newDimOrder);
				break;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t scaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha)
	{
		const avSize_t elements = getTensor(cDesc).volume();
		switch (getTensor(cDesc).dtype())
		{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_scale_tensor(getPointer<uint8_t>(cMem), getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_scale_tensor(getPointer<int8_t>(cMem), getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_scale_tensor(getPointer<int16_t>(cMem), getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_scale_tensor(getPointer<int32_t>(cMem), getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_scale_tensor(getPointer<int64_t>(cMem), getAlphaValue(alpha), elements);
//					break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_scale_tensor(getPointer<float16>(cMem), getAlphaValue(alpha), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_scale_tensor(getPointer<bfloat16>(cMem), getAlphaValue(alpha), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_scale_tensor(getPointer<float>(cMem), getAlphaValue(alpha), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_scale_tensor(getPointer<double>(cMem), getAlphaValue<double>(alpha), elements);
				break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_scale_tensor(getPointer<std::complex<float>>(cMem), getAlphaValue<std::complex<float>>(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_scale_tensor(getPointer<std::complex<double>>(cMem), getAlphaValue<std::complex<double>>(alpha), elements);
//					break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t addScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *scalar)
	{
		const avSize_t elements = getTensor(cDesc).volume();
		switch (getTensor(cDesc).dtype())
		{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_add_scalar_to_tensor(getPointer<uint8_t>(cMem), scalar, elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_add_scalar_to_tensor(getPointer<int8_t>(cMem), scalar, elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_add_scalar_to_tensor(getPointer<int16_t>(cMem), scalar, elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_add_scalar_to_tensor(getPointer<int32_t>(cMem), scalar, elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_add_scalar_to_tensor(getPointer<int64_t>(cMem), scalar, elements);
//					break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_add_scalar_to_tensor(getPointer<float16>(cMem), getScalarValue<float>(scalar), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_add_scalar_to_tensor(getPointer<bfloat16>(cMem), getScalarValue<float>(scalar), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_add_scalar_to_tensor(getPointer<float>(cMem), getScalarValue<float>(scalar), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_add_scalar_to_tensor(getPointer<double>(cMem), getScalarValue<double>(scalar), elements);
				break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_add_scalar_to_tensor(getPointer<std::complex<float>>(cMem), scalar, elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_add_scalar_to_tensor(getPointer<std::complex<double>>(cMem), scalar, elements);
//					break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t addBias(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *beta,
			const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

