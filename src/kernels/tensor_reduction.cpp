/*
 * tensor_reduction.cpp
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

	template<typename T>
	struct limits
	{
			static constexpr float max_value = std::numeric_limits<T>::max();
	};
	template<>
	struct limits<float16>
	{
			static constexpr float16 max_value { 0x7bffu }; // 65504
	};
	template<>
	struct limits<bfloat16>
	{
			static constexpr bfloat16 max_value { 0x7f7fu }; //  approx. 3.402 × 10^38
	};

	template<typename T>
	struct ReduceAdd
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>::zero();
			}
			SIMD<T> accumulate(SIMD<T> &acc, SIMD<T> x) const noexcept
			{
				return acc + x;
			}
			SIMD<T> combine_partial(SIMD<T> &acc, SIMD<T> x) const noexcept
			{
				return accumulate(acc, x);
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_add(acc);
			}
	};
	template<typename T>
	struct ReduceMul
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>::one();
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return acc * x;
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return accumulate(acc, x);
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_mul(acc);
			}
	};
	template<typename T>
	struct ReduceMin
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>(limits<T>::max_value);
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return min(acc, x);
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return accumulate(acc, x);
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_min(acc);
			}
	};
	template<typename T>
	struct ReduceMax
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>(-limits<T>::max_value);
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return max(acc, x);
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return accumulate(acc, x);
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_max(acc);
			}
	};
	template<typename T>
	struct ReduceAmax
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>::zero();
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return max(acc, abs(x));
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return max(acc, x);
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_max(acc);
			}
	};
	template<typename T>
	struct ReduceNorm1
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>::zero();
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return acc + abs(x);
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return acc + x;
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_add(acc);
			}
	};
	template<typename T>
	struct ReduceNorm2
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>::zero();
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return acc + square(x);
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return acc + x;
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return sqrt(acc);
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return std::sqrt(horizontal_add(acc));
			}
	};
	template<typename T>
	struct ReduceMulNoZeros
	{
			SIMD<T> init() const noexcept
			{
				return SIMD<T>::one();
			}
			SIMD<T> accumulate(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return acc * select(x == SIMD<T>::zero(), SIMD<T>::one(), x);
			}
			SIMD<T> combine_partial(SIMD<T> acc, SIMD<T> x) const noexcept
			{
				return accumulate(acc, x);
			}
			SIMD<T> final_action_vector(SIMD<T> acc) const noexcept
			{
				return acc;
			}
			T final_action_scalar(SIMD<T> acc) const noexcept
			{
				return horizontal_mul(acc);
			}
	};

	template<class Op, typename T, typename U>
	void kernel_reduce_tensor(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dimensions, T *workspace) noexcept
	{
		Op reduction;
		if (dimensions.last == 1) // reduce into single element
		{
			SIMD<T> result = reduction.init();
#pragma omp parallel
			{
				SIMD<T> acc = reduction.init();
#pragma omp for nowait
				for (int i = 0; i < dimensions.first; i += SIMD<T>::length)
				{
					const int elements_left = std::min(dimensions.first - i, SIMD<T>::length);
					SIMD<T> data(src + i, elements_left);
					data.cutoff(elements_left, reduction.init());
					acc = reduction.accumulate(acc, data);
				}
#pragma omp critical
				{
					reduction.combine_partial(result, acc);
				}
			}
			T tmp = alpha * reduction.final_action_scalar(result);
			if (beta != scalar::zero<U>())
				tmp += beta * dst[0];
			dst[0] = tmp;
		}
		else
		{
			T *master_workspace = workspace + 0;
#pragma omp parallel
			{
				T *thread_workspace = workspace + (1 + omp_get_thread_num()) * dimensions.last;
				for (int j = 0; j < dimensions.last; j += SIMD<T>::length)
				{
					const int elements_left = std::min(dimensions.last - j, SIMD<T>::length);
					SIMD<T> tmp = reduction.init();
					tmp.store(thread_workspace + j, elements_left);
				}
#pragma omp for nowait
				for (int i = 0; i < dimensions.first; i++)
					for (int j = 0; j < dimensions.last; j += SIMD<T>::length)
					{
						const int elements_left = std::min(dimensions.last - j, SIMD<T>::length);
						SIMD<T> acc(thread_workspace + j, elements_left);
						SIMD<T> data(src + (i * dimensions.last + j), elements_left);
						acc = reduction.accumulate(acc, data);
						acc.store(thread_workspace + j, elements_left);
					}
#pragma omp critical
				{
					for (int j = 0; j < dimensions.last; j += SIMD<T>::length)
					{
						const int elements_left = std::min(dimensions.last - j, SIMD<T>::length);
						SIMD<T> master_acc(master_workspace + j, elements_left);
						SIMD<T> thread_acc(thread_workspace + j, elements_left);

						master_acc = reduction.combine_partial(master_acc, thread_acc);
						master_acc.store(master_workspace + j, elements_left);
					}
				}
			}
			for (int j = 0; j < dimensions.last; j += SIMD<T>::length)
			{
				const int elements_left = std::min(dimensions.last - j, SIMD<T>::length);
				SIMD<T> master_acc(master_workspace + j, elements_left);

				SIMD<T> result = alpha * reduction.final_action_vector(master_acc);
				if (beta != scalar::zero<U>())
				{
					SIMD<T> loaded_dst(dst + j, elements_left);
					result = result + beta * loaded_dst;
				}
				result.store(dst + j, elements_left);
			}
		}
	}
	template<typename T, typename U>
	void launcher_tensor_reduction(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dimensions, avReduceOp_t operation, T *workspace)
	{
		switch (operation)
		{
			case AVOCADO_REDUCE_ADD:
				kernel_reduce_tensor<ReduceAdd<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MUL:
				kernel_reduce_tensor<ReduceMul<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MIN:
				kernel_reduce_tensor<ReduceMin<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MAX:
				kernel_reduce_tensor<ReduceMax<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_AMAX:
				kernel_reduce_tensor<ReduceAmax<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_AVG:
				kernel_reduce_tensor<ReduceAdd<T>, T, U>(dst, src, alpha / dimensions.first, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_NORM1:
				kernel_reduce_tensor<ReduceNorm1<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_NORM2:
				kernel_reduce_tensor<ReduceNorm2<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MUL_NO_ZEROS:
				kernel_reduce_tensor<ReduceMulNoZeros<T>, T, U>(dst, src, alpha, beta, dimensions, workspace);
				break;
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t reduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
	{
		BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(cDesc));

		const int required_workspace_size = (1 + cpuGetNumberOfThreads()) * dimensions.last * dataTypeSize(getTensor(aDesc).dtype());
		if (getContext(context).getWorkspace().size() < required_workspace_size)
			return AVOCADO_STATUS_INTERNAL_ERROR; // not enough workspace

		switch (getTensor(aDesc).dtype())
		{
//			case AVOCADO_DTYPE_FLOAT16:
//				launcher_tensor_reduction(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions,
//						operation, getContext(context).getWorkspace().data<float16>());
//				break;
//			case AVOCADO_DTYPE_BFLOAT16:
//				launcher_tensor_reduction(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta),
//						dimensions, operation, getContext(context).getWorkspace().data<bfloat16>());
//				break;
			case AVOCADO_DTYPE_FLOAT32:
				launcher_tensor_reduction(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions,
						operation, getContext(context).getWorkspace().data<float>());
				break;
			case AVOCADO_DTYPE_FLOAT64:
				launcher_tensor_reduction(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha),
						getBetaValue<double>(beta), dimensions, operation, getContext(context).getWorkspace().data<double>());
				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

