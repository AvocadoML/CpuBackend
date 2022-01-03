/*
 * kernel_definitions.hpp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef KERNELS_KERNEL_DEFINITIONS_HPP_
#define KERNELS_KERNEL_DEFINITIONS_HPP_

#include <avocado/cpu_backend.h>

#include "vectors/simd_macros.hpp"
#include "utils.hpp"

namespace NAMESPACE_AVX2
{
#include "kernels/kernel_definitions.ipp"
}

namespace NAMESPACE_AVX
{
#include "kernels/kernel_definitions.ipp"
}

namespace NAMESPACE_F16C
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

namespace NAMESPACE_NONE
{
#include "kernels/kernel_definitions.ipp"
}

#endif /* KERNELS_KERNEL_DEFINITIONS_HPP_ */
