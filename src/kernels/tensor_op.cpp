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
		src_stride.fill(0);
		dst_stride.fill(0);

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

	template<typename T, typename U>
	void kernel_add_tensors(T *dst, const T *src, U alpha, U beta, cpu::BroadcastedDimensions dims) noexcept
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		if (dims.first == 1)
		{
#pragma omp parallel for
			for (int i = 0; i < dims.last; i += SIMD<T>::length)
			{
				const int elements_left = std::min(dims.last - i, SIMD<T>::length);
				SIMD<T> tmp = alpha * SIMD<T>(src + i, elements_left);
				if (beta != scalar::zero<U>())
					tmp += beta * SIMD<T>(dst + i, elements_left);
				tmp.store(dst + i, elements_left);
			}
		}
		else
		{
#pragma omp parallel for
			for (int i = 0; i < dims.first; i++)
				for (int j = 0; j < dims.last; j += SIMD<T>::length)
				{
					const int elements_left = std::min(dims.last - j, SIMD<T>::length);
					SIMD<T> tmp = alpha * SIMD<T>(src + i * dims.last + j, elements_left);
					if (beta != scalar::zero<U>())
						tmp += beta * SIMD<T>(dst + i * dims.last + j, elements_left);
					tmp.store(dst + i * dims.last + j, elements_left);
				}
		}
	}

	template<class Activation, typename dstT, typename srcT, typename biasT>
	void kernel_add_bias(dstT *yMem, biasT alpha1, biasT alpha2, const srcT *xMem, const biasT *bMem, const dstT *zMem, biasT beta1, biasT beta2,
			biasT beta3, cpu::BroadcastedDimensions dims) noexcept
	{
		Activation activation;
#pragma omp parallel for
		for (int i = 0; i < dims.first; i++)
			for (int j = 0; j < dims.last; j += SIMD<dstT>::length)
			{
				const int elements_left = std::min(dims.last - j, SIMD<dstT>::length);
				SIMD<dstT> input(xMem + i * dims.last + j, elements_left);
				SIMD<dstT> bias(bMem + j, elements_left);

				SIMD<dstT> tmp = alpha2 * input + bias;
				if (beta1 != scalar::zero<biasT>() or beta2 != scalar::zero<biasT>())
				{
					SIMD<dstT> ext(zMem + i * dims.last + j, elements_left);
					tmp = alpha1 * activation.forward(tmp + beta1 * ext) + beta2 * ext;
				}
				else
					tmp = alpha1 * activation.forward(tmp);
				if (beta3 != scalar::zero<biasT>())
					tmp += beta3 * SIMD<dstT>(yMem + i * dims.last + j, elements_left);
				tmp.store(yMem + i * dims.last + j, elements_left);
			}
	}

	template<typename dstT, typename srcT, typename biasT>
	avStatus_t launcher_add_bias(dstT *yMem, biasT alpha1, biasT alpha2, const srcT *xMem, const biasT *bMem, const dstT *zMem, biasT beta1,
			biasT beta2, biasT beta3, cpu::BroadcastedDimensions dims, avActivationType_t activation) noexcept
	{
		switch (activation)
		{
			case AVOCADO_ACTIVATION_LINEAR:
				kernel_add_bias<ActivationLinear<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_add_bias<ActivationSigmoid<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_add_bias<ActivationTanh<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_add_bias<ActivationRelu<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_add_bias<ActivationSelu<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_add_bias<ActivationElu<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_add_bias<ActivationExponential<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_add_bias<ActivationSoftplus<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_add_bias<ActivationSoftsign<dstT>, dstT, srcT, biasT>(yMem, alpha1, alpha2, xMem, bMem, zMem, beta1, beta2, beta3, dims);
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
		const av_int64 elements = cDesc.volume();
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
		const av_int64 elements = cDesc.volume();
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

	avStatus_t cpu_addTensors(const ContextDescriptor &context, const void *alpha, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem,
			const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(aDesc, cDesc);
		const av_int64 elements = cDesc.volume();
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
				kernel_add_tensors(cMem.data<float16>(), aMem.data<float16>(), cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), dimensions);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_add_tensors(cMem.data<bfloat16>(), aMem.data<bfloat16>(), cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), dimensions);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_add_tensors(cMem.data<float>(), aMem.data<float>(), cpu::getAlphaValue(alpha), cpu::getBetaValue(beta), dimensions);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_add_tensors(cMem.data<double>(), aMem.data<double>(), cpu::getAlphaValue<double>(alpha), cpu::getBetaValue<double>(beta),
						dimensions);
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

	avStatus_t cpu_addBias(const ContextDescriptor &context, const void *alpha1, const void *alpha2, const TensorDescriptor &xDesc,
			const MemoryDescriptor &xMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const TensorDescriptor &yDesc,
			MemoryDescriptor &yMem, const void *beta1, const void *beta2, const void *beta3, const MemoryDescriptor &zMem,
			avActivationType_t activation)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(xDesc, bDesc);
		switch (yDesc.dtype())
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
				return launcher_add_bias(yMem.data<float16>(), cpu::getAlphaValue(alpha1), cpu::getAlphaValue(alpha2), xMem.data<float16>(),
						bMem.data<float>(), zMem.data<float16>(), cpu::getBetaValue(beta1), cpu::getBetaValue(beta2), cpu::getBetaValue(beta3),
						dimensions, activation);
			case AVOCADO_DTYPE_BFLOAT16:
				return launcher_add_bias(yMem.data<bfloat16>(), cpu::getAlphaValue(alpha1), cpu::getAlphaValue(alpha2), xMem.data<bfloat16>(),
						bMem.data<float>(), zMem.data<bfloat16>(), cpu::getBetaValue(beta1), cpu::getBetaValue(beta2), cpu::getBetaValue(beta3),
						dimensions, activation);
			case AVOCADO_DTYPE_FLOAT32:
				return launcher_add_bias(yMem.data<float>(), cpu::getAlphaValue(alpha1), cpu::getAlphaValue(alpha2), xMem.data<float>(),
						bMem.data<float>(), zMem.data<float>(), cpu::getBetaValue(beta1), cpu::getBetaValue(beta2), cpu::getBetaValue(beta3),
						dimensions, activation);
			case AVOCADO_DTYPE_FLOAT64:
				return launcher_add_bias(yMem.data<double>(), cpu::getAlphaValue<double>(alpha1), cpu::getAlphaValue<double>(alpha2),
						xMem.data<double>(), bMem.data<double>(), zMem.data<double>(), cpu::getBetaValue<double>(beta1),
						cpu::getBetaValue<double>(beta2), cpu::getBetaValue<double>(beta3), dimensions, activation);
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

