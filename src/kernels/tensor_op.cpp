/*
 * tensor_op.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"
#include "activation.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	struct int4
	{
			int x, y, z, w;
	};

	void kernel_concat_tensors(const TensorDescriptor &cDesc, MemoryDescriptor &cMem, const std::vector<const TensorDescriptor*> &aDesc,
			const std::vector<const MemoryDescriptor*> &aMem)
	{
		const int dtype_size = cpu::dataTypeSize(cDesc.dtype());
		const int first_dim = cDesc.volumeWithoutLastDim();

		const int dst_last_dim = cDesc.lastDim() * dtype_size;

#pragma omp parallel
		{
			int last_dim_offset = 0;
			for (size_t k = 0; k < aMem.size(); k++)
			{
				assert(aDesc[k] != nullptr);
				assert(aMem[k] != nullptr);
				const int src_last_dim = aDesc[k]->lastDim() * dtype_size;

#pragma omp for nowait
				for (int i = 0; i < first_dim; i++)
					std::memcpy(cMem.data<uint8_t>() + i * dst_last_dim + last_dim_offset, aMem[k]->data<uint8_t>() + i * src_last_dim, src_last_dim);
				last_dim_offset += src_last_dim;
			}
		}
	}

	void kernel_split_tensors(const std::vector<const TensorDescriptor*> &cDesc, const std::vector<MemoryDescriptor*> &cMem,
			const TensorDescriptor &aDesc, const MemoryDescriptor &aMem)
	{
		const int dtype_size = cpu::dataTypeSize(aDesc.dtype());
		const int first_dim = aDesc.volumeWithoutLastDim();

		const int src_last_dim = aDesc.lastDim() * dtype_size;

#pragma omp parallel
		{
			int last_dim_offset = 0;
			for (size_t k = 0; k < cDesc.size(); k++)
			{
				assert(cDesc[k] != nullptr);
				assert(cDesc[k] != nullptr);
				const int dst_last_dim = cDesc[k]->lastDim() * dtype_size;

#pragma omp for nowait
				for (int i = 0; i < first_dim; i++)
					std::memcpy(cMem[k]->data<uint8_t>() + i * dst_last_dim, aMem.data<uint8_t>() + i * src_last_dim + last_dim_offset, dst_last_dim);
				last_dim_offset += dst_last_dim;
			}
		}
	}

	template<typename T>
	void kernel_transpose(T *dst, const T *src, const cpu::TensorDescriptor &src_shape, const int ordering[])
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		assert(ordering != nullptr);
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
	void kernel_scale_tensor(T *dst, const T *src, U value, int elements) noexcept
	{
		assert(dst != nullptr);
		assert(src != nullptr);
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> data(src + i, elements_left);
			data = data * value;
			data.store(dst + i, elements_left);
		}
	}
	template<typename T, typename U>
	void kernel_add_scalar_to_tensor(T *dst, const T *src, U scalar, int elements) noexcept
	{
		assert(dst != nullptr);
		assert(src != nullptr);
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> data(src + i, elements_left);
			data = data + scalar;
			data.store(dst + i, elements_left);
		}
	}

	template<class Activation, typename T, typename U, typename V>
	void kernel_add_bias(T *dst, U alpha3, U alpha1, const V *src, U alpha2, const U *bias, U beta, cpu::BroadcastedDimensions dims) noexcept
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		assert(bias != nullptr);
		Activation activation;
#pragma omp parallel for
		for (int i = 0; i < dims.first; i++)
			for (int j = 0; j < dims.last; j += SIMD<T>::length)
			{
				const int elements_left = std::min(dims.last - j, SIMD<T>::length);
				SIMD<T> lhs(src + i * dims.last + j, elements_left);
				SIMD<T> rhs(bias + j, elements_left);
				SIMD<T> result = alpha3 * activation.forward(lhs * alpha1 + rhs * alpha2);
				if (beta != scalar::zero<U>())
				{
					SIMD<T> tmp(dst + i * dims.last + j, elements_left);
					result += beta * tmp;
				}
				result.store(dst + i * dims.last + j, elements_left);
			}
	}

	template<typename T, typename U, typename V>
	avStatus_t launcher_add_bias(T *dst, U alpha3, U alpha1, const V *src, U alpha2, const U *bias, U beta, cpu::BroadcastedDimensions dims,
			avActivationType_t activation) noexcept
	{
		switch (activation)
		{
			case AVOCADO_ACTIVATION_LINEAR:
				kernel_add_bias<ActivationLinear<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_add_bias<ActivationSigmoid<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_add_bias<ActivationTanh<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_add_bias<ActivationRelu<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_add_bias<ActivationSelu<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_add_bias<ActivationElu<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_add_bias<ActivationExponential<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_add_bias<ActivationSoftplus<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_add_bias<ActivationSoftsign<U>, T, U, V>(dst, alpha3, alpha1, src, alpha2, bias, beta, dims);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_concatTensors(const ContextDescriptor &context, const TensorDescriptor &cDesc, MemoryDescriptor &cMem,
			const std::vector<const TensorDescriptor*> &aDesc, const std::vector<const MemoryDescriptor*> &aMem)
	{
		switch (cpu::dataTypeSize(cDesc.dtype()))
		{
			case 1:
			case 2:
			case 4:
			case 8:
			case 16:
				kernel_concat_tensors(cDesc, cMem, aDesc, aMem);
				return AVOCADO_STATUS_SUCCESS;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
	}

	avStatus_t cpu_splitTensors(const ContextDescriptor &context, const std::vector<const TensorDescriptor*> &cDesc,
			std::vector<MemoryDescriptor*> &cMem, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem)
	{
		switch (cpu::dataTypeSize(aDesc.dtype()))
		{
			case 1:
			case 2:
			case 4:
			case 8:
			case 16:
				kernel_split_tensors(cDesc, cMem, aDesc, aMem);
				return AVOCADO_STATUS_SUCCESS;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
	}

	avStatus_t cpu_transpose(const ContextDescriptor &context, const TensorDescriptor &cDesc, MemoryDescriptor &cMem, const TensorDescriptor &aDesc,
			const MemoryDescriptor &aMem, const int newDimOrder[])
	{
		switch (cpu::dataTypeSize(aDesc.dtype()))
		{
			case 1:
				kernel_transpose(cMem.data<int8_t>(), aMem.data<int8_t>(), aDesc, newDimOrder);
				break;
			case 2:
				kernel_transpose(cMem.data<int16_t>(), aMem.data<int16_t>(), aDesc, newDimOrder);
				break;
			case 4:
				kernel_transpose(cMem.data<int32_t>(), aMem.data<int32_t>(), aDesc, newDimOrder);
				break;
			case 8:
				kernel_transpose(cMem.data<int64_t>(), aMem.data<int64_t>(), aDesc, newDimOrder);
				break;
			case 16:
				kernel_transpose(cMem.data<int4>(), aMem.data<int4>(), aDesc, newDimOrder);
				break;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_scaleTensor(const ContextDescriptor &context, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const void *alpha,
			const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
	{
		const avSize_t elements = cDesc.volume();
		switch (cDesc.dtype())
		{
//			case AVOCADO_DTYPE_UINT8:
//				kernel_scale_tensor(cpu::getPointer<uint8_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT8:
//				kernel_scale_tensor(cpu::getPointer<int8_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT16:
//				kernel_scale_tensor(cpu::getPointer<int16_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT32:
//				kernel_scale_tensor(cpu::getPointer<int32_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT64:
//				kernel_scale_tensor(cpu::getPointer<int64_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_scale_tensor(cMem.data<float16>(), aMem.data<float16>(), cpu::getAlphaValue(alpha), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_scale_tensor(cMem.data<bfloat16>(), aMem.data<bfloat16>(), cpu::getAlphaValue(alpha), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_scale_tensor(cMem.data<float>(), aMem.data<float>(), cpu::getAlphaValue(alpha), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_scale_tensor(cMem.data<double>(), aMem.data<double>(), cpu::getAlphaValue<double>(alpha), elements);
				break;
//			case AVOCADO_DTYPE_COMPLEX32:
//				kernel_scale_tensor(cpu::getPointer<std::complex<float>>(cMem), cpu::getAlphaValue<std::complex<float>>(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_COMPLEX64:
//				kernel_scale_tensor(cpu::getPointer<std::complex<double>>(cMem), cpu::getAlphaValue<std::complex<double>>(alpha), elements);
//				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_addScalarToTensor(const ContextDescriptor &context, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem,
			const void *scalar, const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
	{
		const avSize_t elements = cDesc.volume();
		switch (cDesc.dtype())
		{
//			case AVOCADO_DTYPE_UINT8:
//				kernel_add_scalar_to_tensor(getPointer<uint8_t>(cMem), scalar, elements);
//				break;
//			case AVOCADO_DTYPE_INT8:
//				kernel_add_scalar_to_tensor(getPointer<int8_t>(cMem), scalar, elements);
//				break;
//			case AVOCADO_DTYPE_INT16:
//				kernel_add_scalar_to_tensor(getPointer<int16_t>(cMem), scalar, elements);
//				break;
//			case AVOCADO_DTYPE_INT32:
//				kernel_add_scalar_to_tensor(getPointer<int32_t>(cMem), scalar, elements);
//				break;
//			case AVOCADO_DTYPE_INT64:
//				kernel_add_scalar_to_tensor(getPointer<int64_t>(cMem), scalar, elements);
//				break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_add_scalar_to_tensor(cMem.data<float16>(), aMem.data<float16>(), cpu::getScalarValue<float>(scalar), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_add_scalar_to_tensor(cMem.data<bfloat16>(), aMem.data<bfloat16>(), cpu::getScalarValue<float>(scalar), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_add_scalar_to_tensor(cMem.data<float>(), aMem.data<float>(), cpu::getScalarValue<float>(scalar), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_add_scalar_to_tensor(cMem.data<double>(), aMem.data<double>(), cpu::getScalarValue<double>(scalar), elements);
				break;
//			case AVOCADO_DTYPE_COMPLEX32:
//				kernel_add_scalar_to_tensor(getPointer<std::complex<float>>(cMem), scalar, elements);
//				break;
//			case AVOCADO_DTYPE_COMPLEX64:
//				kernel_add_scalar_to_tensor(getPointer<std::complex<double>>(cMem), scalar, elements);
//				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

	avStatus_t cpu_addBias(const ContextDescriptor &context, const void *alpha3, const void *alpha1, const TensorDescriptor &aDesc,
			const MemoryDescriptor &aMem, const void *alpha2, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *beta,
			const TensorDescriptor &cDesc, MemoryDescriptor &cMem, avActivationType_t activation)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(aDesc, bDesc);
		switch (cDesc.dtype())
		{
//			case AVOCADO_DTYPE_UINT8:
//				kernel_scale_tensor(cpu::getPointer<uint8_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT8:
//				kernel_scale_tensor(cpu::getPointer<int8_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT16:
//				kernel_scale_tensor(cpu::getPointer<int16_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT32:
//				kernel_scale_tensor(cpu::getPointer<int32_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_INT64:
//				kernel_scale_tensor(cpu::getPointer<int64_t>(cMem), cpu::getAlphaValue(alpha), elements);
//				break;
			case AVOCADO_DTYPE_FLOAT16:
				return launcher_add_bias(cMem.data<float16>(), cpu::getAlphaValue(alpha3), cpu::getAlphaValue(alpha1), aMem.data<float16>(),
						cpu::getAlphaValue(alpha2), bMem.data<float>(), cpu::getBetaValue(beta), dimensions, activation);
			case AVOCADO_DTYPE_BFLOAT16:
				return launcher_add_bias(cMem.data<bfloat16>(), cpu::getAlphaValue(alpha3), cpu::getAlphaValue(alpha1), aMem.data<bfloat16>(),
						cpu::getAlphaValue(alpha2), bMem.data<float>(), cpu::getBetaValue(beta), dimensions, activation);
			case AVOCADO_DTYPE_FLOAT32:
				return launcher_add_bias(cMem.data<float>(), cpu::getAlphaValue(alpha3), cpu::getAlphaValue(alpha1), aMem.data<float>(),
						cpu::getAlphaValue(alpha2), bMem.data<float>(), cpu::getBetaValue(beta), dimensions, activation);
			case AVOCADO_DTYPE_FLOAT64:
				return launcher_add_bias(cMem.data<double>(), cpu::getAlphaValue<double>(alpha3), cpu::getAlphaValue<double>(alpha1),
						aMem.data<double>(), cpu::getAlphaValue<double>(alpha2), bMem.data<double>(), cpu::getBetaValue<double>(beta), dimensions,
						activation);
//			case AVOCADO_DTYPE_COMPLEX32:
//				kernel_scale_tensor(cpu::getPointer<std::complex<float>>(cMem), cpu::getAlphaValue<std::complex<float>>(alpha), elements);
//				break;
//			case AVOCADO_DTYPE_COMPLEX64:
//				kernel_scale_tensor(cpu::getPointer<std::complex<double>>(cMem), cpu::getAlphaValue<std::complex<double>>(alpha), elements);
//				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

