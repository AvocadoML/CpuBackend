/*
 * im2row.cpp
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

		avStatus_t im2row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc,
				avMemoryDescriptor_t colMem)
		{
		}

	} /* namespace backend */
} /* namespace avocado */

