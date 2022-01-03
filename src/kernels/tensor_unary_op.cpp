/*
 * tensor_unary_op.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cpu_backend.h>

#include <vector_types.h>

namespace
{
	using namespace avocado::backend;

}

namespace avocado
{
	namespace backend
	{
		avStatus_t unaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
		}
	} /* namespace backend */
} /* namespace avocado */

