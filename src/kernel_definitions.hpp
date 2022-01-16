/*
 * kernel_definitions.hpp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef KERNELS_KERNEL_DEFINITIONS_HPP_
#define KERNELS_KERNEL_DEFINITIONS_HPP_

#include <CpuBackend/cpu_backend.h>

#include "vectors/simd_macros.hpp"
#include "utils.hpp"

#if DYNAMIC_ARCH

namespace NAMESPACE_AVX2
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_AVX
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_SSE41
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_SSE2
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_NO_SIMD
{
#include "kernels/kernel_definitions.ipp"
}

#else

namespace SIMD_NAMESPACE
{
#include "kernels/kernel_definitions.ipp"
}

#endif /* DYNAMIC_ARCH */

#endif /* KERNELS_KERNEL_DEFINITIONS_HPP_ */
