/*
 * tensor_binary_op.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	struct BinaryOpAdd
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs + rhs;
			}
	};
	template<typename T>
	struct BinaryOpAddSquare
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs + square(rhs);
			}
	};
	template<typename T>
	struct BinaryOpSub
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs - rhs;
			}
	};
	template<typename T>
	struct BinaryOpMul
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs * rhs;
			}
	};
	template<typename T>
	struct BinaryOpDiv
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs / rhs;
			}
	};
	template<typename T>
	struct BinaryOpMod
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return mod(lhs, rhs);
			}
	};
	template<typename T>
	struct BinaryOpPow
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return pow(lhs, rhs);
			}
	};
	template<typename T>
	struct BinaryOpMin
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return min(lhs, rhs);
			}
	};
	template<typename T>
	struct BinaryOpMax
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return max(lhs, rhs);
			}
	};
	template<typename T>
	struct BinaryOpCompareEq
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs == rhs;
			}
	};
	template<typename T>
	struct BinaryOpCompareNeq
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs != rhs;
			}
	};
	template<typename T>
	struct BinaryOpCompareGt
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs > rhs;
			}
	};
	template<typename T>
	struct BinaryOpCompareGe
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs >= rhs;
			}
	};
	template<typename T>
	struct BinaryOpCompareLt
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs < rhs;
			}
	};
	template<typename T>
	struct BinaryOpCompareLe
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs <= rhs;
			}
	};
	template<typename T>
	struct BinaryOpLogicalAnd
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs & rhs;
			}
	};
	template<typename T>
	struct BinaryOpLogicalOr
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs | rhs;
			}
	};
	template<typename T>
	struct BinaryOpLogicalXor
	{
			SIMD<T> operator()(SIMD<T> lhs, SIMD<T> rhs) const noexcept
			{
				return lhs ^ rhs;
			}
	};

	template<class Op, typename T, typename U, bool ZeroBeta, bool LogicalOp = false>
	void kernel_binary_op(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, cpu::BroadcastedDimensions dimensions) noexcept
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);
		Op operation;
		if (dimensions.first == 1) // both src1 and src2 have the same shape
		{
#pragma omp parallel for
			for (int i = 0; i < dimensions.last; i += SIMD<T>::length)
			{
				const int elements_left = std::min(dimensions.last - i, SIMD<T>::length);
				SIMD<T> lhs(src1 + i, elements_left);
				SIMD<T> rhs(src2 + i, elements_left);

				SIMD<T> result;
				if (LogicalOp)
					result = operation(lhs, rhs);
				else
				{
					lhs *= alpha1;
					rhs *= alpha2;
					result = operation(lhs, rhs);
					if (not ZeroBeta)
					{
						SIMD<T> loaded_dst(dst + i, elements_left);
						result = result + beta * loaded_dst;
					}
				}
				result.store(dst + i, elements_left);
			}
		}
		else
		{
			if (dimensions.last == 1) // src2 is a single element
			{
				SIMD<T> rhs(src2[0]);
				if (not LogicalOp)
					rhs = alpha2 * rhs;
#pragma omp parallel for
				for (int i = 0; i < dimensions.first; i += SIMD<T>::length)
				{
					const int elements_left = std::min(dimensions.first - i, SIMD<T>::length);
					SIMD<T> lhs(src1 + i, elements_left);

					SIMD<T> result;
					if (LogicalOp)
						result = operation(lhs, rhs);
					else
					{
						result = operation(alpha1 * lhs, rhs);
						if (not ZeroBeta)
						{
							SIMD<T> loaded_dst(dst + i, elements_left);
							result = result + beta * loaded_dst;
						}
					}
					result.store(dst + i, elements_left);
				}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < dimensions.first; i++)
					for (int j = 0; j < dimensions.last; j += SIMD<T>::length)
					{
						const int elements_left = std::min(dimensions.last - j, SIMD<T>::length);
						SIMD<T> lhs(src1 + (i * dimensions.last + j), elements_left);
						SIMD<T> rhs(src2 + j, elements_left);

						SIMD<T> result;
						if (LogicalOp)
							result = operation(lhs, rhs);
						else
						{
							result = operation(alpha1 * lhs, alpha2 * rhs);
							if (not ZeroBeta)
							{
								SIMD<T> loaded_dst(dst + (i * dimensions.last + j), elements_left);
								result = result + beta * loaded_dst;
							}
						}
						result.store(dst + (i * dimensions.last + j), elements_left);
					}
			}
		}

	}
	template<typename T, typename U, bool ZeroBeta>
	void launcher_binary_op(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, cpu::BroadcastedDimensions dimensions,
			avBinaryOp_t operation)
	{
		switch (operation)
		{
			case AVOCADO_BINARY_OP_ADD:
				kernel_binary_op<BinaryOpAdd<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_ADD_SQUARE:
				kernel_binary_op<BinaryOpAddSquare<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_SUB:
				kernel_binary_op<BinaryOpSub<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MUL:
				kernel_binary_op<BinaryOpMul<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_DIV:
				kernel_binary_op<BinaryOpDiv<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MOD:
				kernel_binary_op<BinaryOpMod<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_POW:
				kernel_binary_op<BinaryOpPow<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MIN:
				kernel_binary_op<BinaryOpMin<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MAX:
				kernel_binary_op<BinaryOpMax<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_EQ:
				kernel_binary_op<BinaryOpCompareEq<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_NEQ:
				kernel_binary_op<BinaryOpCompareNeq<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_GT:
				kernel_binary_op<BinaryOpCompareGt<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_GE:
				kernel_binary_op<BinaryOpCompareGe<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_LT:
				kernel_binary_op<BinaryOpCompareLt<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_LE:
				kernel_binary_op<BinaryOpCompareLe<T>, T, U, ZeroBeta>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_LOGICAL_AND:
				kernel_binary_op<BinaryOpLogicalAnd<T>, T, U, ZeroBeta, true>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_LOGICAL_OR:
				kernel_binary_op<BinaryOpLogicalOr<T>, T, U, ZeroBeta, true>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_LOGICAL_XOR:
				kernel_binary_op<BinaryOpLogicalXor<T>, T, U, ZeroBeta, true>(dst, src1, src2, alpha1, alpha2, beta, dimensions);
				break;
		}
	}
	template<typename T, typename U>
	void helper_binary_op(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, cpu::BroadcastedDimensions dimensions,
			avBinaryOp_t operation)
	{
		if (beta == scalar::zero<U>())
			launcher_binary_op<T, U, true>(dst, src1, src2, alpha1, alpha2, beta, dimensions, operation);
		else
			launcher_binary_op<T, U, false>(dst, src1, src2, alpha1, alpha2, beta, dimensions, operation);
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;

	avStatus_t cpu_binaryOp(const ContextDescriptor &context, avBinaryOp_t operation, const void *alpha1, const TensorDescriptor &aDesc,
			const MemoryDescriptor &aMem, const void *alpha2, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *beta,
			const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
	{
		cpu::BroadcastedDimensions dimensions = cpu::getBroadcastDimensions(aDesc, bDesc);
		switch (cDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				helper_binary_op(cMem.data<float16>(), aMem.data<float16>(), bMem.data<float16>(), cpu::getAlphaValue(alpha1),
						cpu::getAlphaValue(alpha2), cpu::getBetaValue(beta), dimensions, operation);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				helper_binary_op(cMem.data<bfloat16>(), aMem.data<bfloat16>(), bMem.data<bfloat16>(), cpu::getAlphaValue(alpha1),
						cpu::getAlphaValue(alpha2), cpu::getBetaValue(beta), dimensions, operation);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				helper_binary_op(cMem.data<float>(), aMem.data<float>(), bMem.data<float>(), cpu::getAlphaValue(alpha1), cpu::getAlphaValue(alpha2),
						cpu::getBetaValue(beta), dimensions, operation);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				helper_binary_op(cMem.data<double>(), aMem.data<double>(), bMem.data<double>(), cpu::getAlphaValue<double>(alpha1),
						cpu::getAlphaValue<double>(alpha2), cpu::getBetaValue<double>(beta), dimensions, operation);
				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}

		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

