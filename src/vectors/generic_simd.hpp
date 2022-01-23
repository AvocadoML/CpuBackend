/*
 * generic_simd.hpp
 *
 *  Created on: Nov 14, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_GENERIC_SIMD_HPP_
#define VECTORS_GENERIC_SIMD_HPP_

#include "simd_macros.hpp"

#include <cstring>
#include <cmath>
#include <inttypes.h>
#include <cassert>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{
	template<typename T, class dummy = T>
	class SIMD;

	/*
	 * Bitwise operations.
	 */
	template<typename T, typename U = T>
	static inline SIMD<T>& operator&(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs & SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator&(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) & rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator&=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs & rhs);
		return lhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T>& operator|(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs | SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator|(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) | rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator|=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs | rhs);
		return lhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T>& operator^(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs ^ SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator^(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) ^ rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator^=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = (lhs ^ rhs);
		return lhs;
	}

	/*
	 * Bitwise shifts.
	 */
	template<typename T>
	static inline SIMD<T>& operator>>=(SIMD<T> &lhs, T rhs) noexcept
	{
		lhs = (lhs >> rhs);
		return lhs;
	}
	template<typename T>
	static inline SIMD<T>& operator<<=(SIMD<T> &lhs, T rhs) noexcept
	{
		lhs = (lhs << rhs);
		return lhs;
	}

	/*
	 * Compare operations.
	 */
	template<typename T, typename U = T>
	static inline SIMD<T> operator<(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs < SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator<(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) < rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator<=(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs <= SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator<=(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) <= rhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T> operator>(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator>(U lhs, SIMD<T> rhs) noexcept
	{
		return lhs <= rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator>=(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs < rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator>=(U lhs, SIMD<T> rhs) noexcept
	{
		return lhs < rhs;
	}

	/*
	 * Arithmetic operations
	 */
	template<typename T, typename U = T>
	static inline SIMD<T> operator+(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs + SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator+(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) + rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator+=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator+=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T>
	static inline SIMD<T> operator+(SIMD<T> x) noexcept
	{
		return x;
	}

	template<typename T, typename U = T>
	static inline SIMD<T> operator-(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs - SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator-(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) - rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator-=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator-=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T, typename U = T>
	static inline SIMD<T> operator*(SIMD<T> lhs, U rhs) noexcept
	{
		return lhs * SIMD<U>(rhs);
	}
	template<typename T, typename U = T>
	static inline SIMD<T> operator*(U lhs, SIMD<T> rhs) noexcept
	{
		return SIMD<U>(lhs) * rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator*=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator*=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T, typename U>
	static inline SIMD<T> operator/(SIMD<U> lhs, T rhs) noexcept
	{
		return lhs / SIMD<U>(rhs);
	}
	template<typename T, typename U>
	static inline SIMD<T> operator/(T lhs, SIMD<U> rhs) noexcept
	{
		return SIMD<U>(lhs) / rhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator/=(SIMD<T> &lhs, SIMD<U> rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T, typename U = T>
	static inline SIMD<T>& operator/=(SIMD<T> &lhs, U rhs) noexcept
	{
		lhs = lhs / rhs;
		return lhs;
	}

} /* namespace SIMD_NAMESPACE */

#endif /* VECTORS_GENERIC_SIMD_HPP_ */
