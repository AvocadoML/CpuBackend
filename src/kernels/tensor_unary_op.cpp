/*
 * tensor_unary_op.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

#include "../vectors/simd_vectors.hpp"
#include "../utils.hpp"

#include <omp.h>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	struct UnaryOpAbs
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return abs(x);
			}
	};
	template<typename T>
	struct UnaryOpCeil
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return ceil(x);
			}
	};
	template<typename T>
	struct UnaryOpCos
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return cos(x);
			}
	};
	template<typename T>
	struct UnaryOpExp
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return exp(x);
			}
	};
	template<typename T>
	struct UnaryOpFloor
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return floor(x);
			}
	};
	template<typename T>
	struct UnaryOpLn
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return log(x);
			}
	};
	template<typename T>
	struct UnaryOpNeg
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return -x;
			}
	};
	template<typename T>
	struct UnaryOpRcp
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return rcp(x);
			}
	};
	template<typename T>
	struct UnaryOpRsqrt
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return rsqrt(x);
			}
	};
	template<typename T>
	struct UnaryOpSin
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return sin(x);
			}
	};
	template<typename T>
	struct UnaryOpSquare
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return square(x);
			}
	};
	template<typename T>
	struct UnaryOpSqrt
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return sqrt(x);
			}
	};
	template<typename T>
	struct UnaryOpTan
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return tan(x);
			}
	};
	template<typename T>
	struct UnaryOpLogicalNot
	{
			SIMD<T> operator()(SIMD<T> x) const noexcept
			{
				return ~x;
			}
	};

	/**
	 * \brief
	 */
	template<class Op, typename T, typename U, bool ZeroBeta, bool LogicalOp = false>
	void kernel_unary_op(T *dst, const T *src, U alpha, U beta, int elements) noexcept
	{
		assert(dst != nullptr);
		assert(src != nullptr);

		Op operation;
#pragma omp parallel for
		for (int i = 0; i < elements; i += SIMD<T>::length)
		{
			const int elements_left = std::min(elements - i, SIMD<T>::length);
			SIMD<T> value(src + i, elements_left);
			if (LogicalOp)
				value = operation(value);
			else
			{
				value = operation(alpha * value);
				if (not ZeroBeta)
				{
					SIMD<T> loaded_dst(dst + i, elements_left);
					value = value + beta * loaded_dst;
				}
			}
			value.store(dst + i, elements_left);
		}
	}
	template<typename T, typename U, bool ZeroBeta>
	void launcher_unary_op(T *dst, const T *src, U alpha, U beta, av_int64 elements, avUnaryOp_t operation)
	{
		switch (operation)
		{
			case AVOCADO_UNARY_OP_ABS:
				kernel_unary_op<UnaryOpAbs<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_CEIL:
				kernel_unary_op<UnaryOpCeil<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_COS:
				kernel_unary_op<UnaryOpCos<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_EXP:
				kernel_unary_op<UnaryOpExp<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_FLOOR:
				kernel_unary_op<UnaryOpFloor<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_LN:
				kernel_unary_op<UnaryOpLn<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_NEG:
				kernel_unary_op<UnaryOpNeg<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_RCP:
				kernel_unary_op<UnaryOpRcp<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_RSQRT:
				kernel_unary_op<UnaryOpRsqrt<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SIN:
				kernel_unary_op<UnaryOpSin<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SQUARE:
				kernel_unary_op<UnaryOpSquare<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SQRT:
				kernel_unary_op<UnaryOpSqrt<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_TAN:
				kernel_unary_op<UnaryOpTan<T>, T, U, ZeroBeta>(dst, src, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_LOGICAL_NOT:
				kernel_unary_op<UnaryOpLogicalNot<T>, T, U, ZeroBeta, true>(dst, src, alpha, beta, elements);
				break;
		}
	}
	template<typename T, typename U>
	void helper_unary_op(T *dst, const T *src, U alpha, U beta, av_int64 elements, avUnaryOp_t operation)
	{
		if (beta == scalar::zero<U>())
			launcher_unary_op<T, U, true>(dst, src, alpha, beta, elements, operation);
		else
			launcher_unary_op<T, U, false>(dst, src, alpha, beta, elements, operation);
	}
}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_unaryOp(const ContextDescriptor &context, avUnaryOp_t operation, const void *alpha, const TensorDescriptor &aDesc,
			const MemoryDescriptor &aMem, const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
	{
		const av_int64 elements = aDesc.volume();
		switch (cDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT16:
				helper_unary_op(cMem.data<float16>(), aMem.data<float16>(), getAlphaValue(alpha), getBetaValue(beta), elements, operation);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				helper_unary_op(cMem.data<bfloat16>(), aMem.data<bfloat16>(), getAlphaValue(alpha), getBetaValue(beta), elements, operation);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				helper_unary_op(cMem.data<float>(), aMem.data<float>(), getAlphaValue(alpha), getBetaValue(beta), elements, operation);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				helper_unary_op(cMem.data<double>(), aMem.data<double>(), getAlphaValue<double>(alpha), getBetaValue<double>(beta), elements,
						operation);
				break;
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}

} /* namespace SIMD_NAMESPACE */

