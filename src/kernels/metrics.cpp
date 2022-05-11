/*
 * metrics.cpp
 *
 *  Created on: Jan 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

#include "../vectors/simd_macros.hpp"

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

}

namespace SIMD_NAMESPACE
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	avStatus_t cpu_metricFunction(const ContextDescriptor &context, avMetricType_t metricType, const TensorDescriptor &outputDesc,
			const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, void *result)
	{
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

} /* namespace SIMD_NAMESPACE */

