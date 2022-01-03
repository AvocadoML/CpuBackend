/*
 * cpu_properties.cpp
 *
 *  Created on: May 12, 2020
 *      Author: Alexander J. Yee
 *      modified by Maciej Kozarzewski
 */

#include <avocado/cpu_backend.h>

#include <string>
#include <cstring>
#include <algorithm>
#include <cassert>

#ifdef _WIN32
#  include <windows.h>
#elif MACOS
#  include <sys/param.h>
#  include <sys/sysctl.h>
#else
#  include <unistd.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#  ifdef _WIN32
#    include <Windows.h>
#    include <intrin.h>
#  elif (defined(__GNUC__) || defined(__clang__))
#    include <cpuid.h>
#  else
#    // error
#  endif
#endif

namespace
{

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#  if _WIN32
	void cpuid(uint32_t out[4], uint32_t eax, uint32_t ecx)
	{
		__cpuidex(out, eax, ecx);
	}
	int64_t xgetbv(uint32_t x)
	{
		return _xgetbv(x);
	}
	//  Detect 64-bit - Note that this snippet of code for detecting 64-bit has been copied from MSDN.
	typedef BOOL (WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
	BOOL IsWow64()
	{
		BOOL bIsWow64 = FALSE;

		LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS) GetProcAddress(GetModuleHandle(TEXT("kernel32")),
				"IsWow64Process");

		if (NULL != fnIsWow64Process)
		{
			if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64))
			{
				printf("Error Detecting Operating System.\n");
				printf("Defaulting to 32-bit OS.\n\n");
				bIsWow64 = FALSE;
			}
		}
		return bIsWow64;
	}
	bool detect_OS_x64()
	{
#    ifdef _M_X64
	    return true;
#    else
		return IsWow64() != 0;
#    endif // _M_X64
	}
#    ifndef _XCR_XFEATURE_ENABLED_MASK
#      define _XCR_XFEATURE_ENABLED_MASK  0
#    endif

#  elif (defined(__GNUC__) || defined(__clang__))
	void cpuid(uint32_t out[4], uint32_t eax, uint32_t ecx)
	{
		__cpuid_count(eax, ecx, out[0], out[1], out[2], out[3]);
	}
	uint64_t xgetbv(uint32_t index)
	{
		uint32_t eax, edx;
		__asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
		return (static_cast<uint64_t>(edx) << 32) | eax;
	}
#    define _XCR_XFEATURE_ENABLED_MASK  0
//  Detect 64-bit
	bool detect_OS_x64()
	{
		//  We only support x64 on Linux.
		return true;
	}
#  endif /* _WIN32 */
#endif

	uint64_t get_total_system_memory()
	{
#if defined(_WIN32)
		MEMORYSTATUSEX status;
		status.dwLength = sizeof(status);
		GlobalMemoryStatusEx(&status);
		return status.ullTotalPhys;
#else
		uint64_t pages = sysconf(_SC_PHYS_PAGES);
		uint64_t page_size = sysconf(_SC_PAGE_SIZE);
		return pages * page_size;
#endif /* defined(_WIN32) */
	}
	bool detect_OS_AVX()
	{
		// Copied from: http://stackoverflow.com/a/22521619/922184

		bool avxSupported = false;

		uint32_t info[4];
		cpuid(info, 1, 0);

		bool osUsesXSAVE_XRSTORE = (info[2] & (1 << 27)) != 0;
		bool cpuAVXSuport = (info[2] & (1 << 28)) != 0;

		if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
		{
			uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
			avxSupported = (xcrFeatureMask & 0x6) == 0x6;
		}

		return avxSupported;
	}
	bool detect_OS_AVX512()
	{
		if (!detect_OS_AVX())
			return false;

		uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
		return (xcrFeatureMask & 0xe6) == 0xe6;
	}
	std::string get_vendor_string()
	{
		uint32_t info[4];
		char name[13];

		cpuid(info, 0, 0);
		memcpy(name + 0, &info[1], 4);
		memcpy(name + 4, &info[3], 4);
		memcpy(name + 8, &info[2], 4);
		name[12] = '\0';

		return name;
	}
}

struct cpu_x86
{
		std::string vendor_name = "";

		int64_t memory = 0; //RAM [bytes]

		int cores = 0;
		bool HYPER_THREADING = false;

		bool OS_x64 = false;
		bool OS_AVX = false;
		bool OS_AVX512 = false;

		bool HW_MMX = false;
		bool HW_x64 = false;
		bool HW_ABM = false;
		bool HW_RDRAND = false;
		bool HW_RDSEED = false;
		bool HW_BMI1 = false;
		bool HW_BMI2 = false;
		bool HW_ADX = false;
		bool HW_MPX = false;
		bool HW_PREFETCHW = false;
		bool HW_PREFETCHWT1 = false;
		bool HW_RDPID = false;
		bool HW_F16C = false;
		bool HW_POPCOUNT = false;

		//  SIMD: 128-bit
		bool HW_SSE = false;
		bool HW_SSE2 = false;
		bool HW_SSE3 = false;
		bool HW_SSSE3 = false;
		bool HW_SSE41 = false;
		bool HW_SSE42 = false;
		bool HW_SSE4a = false;
		bool HW_AES = false;
		bool HW_SHA = false;

		//  SIMD: 256-bit
		bool HW_AVX = false;
		bool HW_XOP = false;
		bool HW_FMA3 = false;
		bool HW_FMA4 = false;
		bool HW_AVX2 = false;

		// SIMD: 512-bit
		bool HW_AVX512_F = false;
		bool HW_AVX512_CD = false;

		//  Knights Landing
		bool HW_AVX512_PF = false;
		bool HW_AVX512_ER = false;

		//  Skylake Purley
		bool HW_AVX512_VL = false;
		bool HW_AVX512_BW = false;
		bool HW_AVX512_DQ = false;

		//  Cannon Lake
		bool HW_AVX512_IFMA = false;
		bool HW_AVX512_VBMI = false;

		//  Knights Mill
		bool HW_AVX512_VPOPCNTDQ = false;
		bool HW_AVX512_4FMAPS = false;
		bool HW_AVX512_4VNNIW = false;

		//  Cascade Lake
		bool HW_AVX512_VNNI = false;

		//  Cooper Lake
		bool HW_AVX512_BF16 = false;

		//  Ice Lake
		bool HW_AVX512_VBMI2 = false;
		bool HW_GFNI = false;
		bool HW_VAES = false;
		bool HW_AVX512_VPCLMUL = false;
		bool HW_AVX512_BITALG = false;

	public:
		cpu_x86()
		{
			//  OS Features
			OS_x64 = detect_OS_x64();
			OS_AVX = detect_OS_AVX();
			OS_AVX512 = detect_OS_AVX512();

			// RAM
			memory = get_total_system_memory();

			//  Vendor
			vendor_name = get_vendor_string();

			uint32_t info[4];
			cpuid(info, 0, 0);
			int32_t nIds = info[0];

			cpuid(info, 0x80000000, 0);
			uint32_t nExIds = info[0];

			//  Detect Features
			if (nIds >= 0x00000001)
			{
				cpuid(info, 0x00000001, 0);
				HW_MMX = (info[3] & (1 << 23)) != 0;
				HW_SSE = (info[3] & (1 << 25)) != 0;
				HW_SSE2 = (info[3] & (1 << 26)) != 0;
				HW_SSE3 = (info[2] & (1 << 0)) != 0;

				HW_SSSE3 = (info[2] & (1 << 9)) != 0;
				HW_SSE41 = (info[2] & (1 << 19)) != 0;
				HW_SSE42 = (info[2] & (1 << 20)) != 0;
				HW_AES = (info[2] & (1 << 25)) != 0;

				HW_F16C = (info[2] & (1 << 29)) != 0;
				HW_POPCOUNT = (info[2] & (1 << 23)) != 0;
				HW_AVX = (info[2] & (1 << 28)) != 0;
				HW_FMA3 = (info[2] & (1 << 12)) != 0;

				HW_RDRAND = (info[2] & (1 << 30)) != 0;

				HYPER_THREADING = (info[3] & (1 << 28)) != 0;
#if defined(_WIN32)
				SYSTEM_INFO systeminfo;
				GetSystemInfo(&systeminfo);
				cores = systeminfo.dwNumberOfProcessors;
#else
				cores = sysconf( _SC_NPROCESSORS_ONLN);
#endif // defined(_WIN32)

//			unsigned int ncores = 0, nthreads = 0;
//			asm volatile("cpuid": "=a" (ncores), "=b" (nthreads) : "a" (0xb), "c" (0x1) : );
			}
			if (nIds >= 0x00000007)
			{
				cpuid(info, 0x00000007, 0);
				HW_AVX2 = (info[1] & (1 << 5)) != 0;

				HW_BMI1 = (info[1] & (1 << 3)) != 0;
				HW_BMI2 = (info[1] & (1 << 8)) != 0;
				HW_ADX = (info[1] & (1 << 19)) != 0;
				HW_MPX = (info[1] & (1 << 14)) != 0;
				HW_SHA = (info[1] & (1 << 29)) != 0;
				HW_RDSEED = (info[1] & (1 << 18)) != 0;
				HW_PREFETCHWT1 = (info[2] & (1 << 0)) != 0;
				HW_RDPID = (info[2] & (1 << 22)) != 0;

				HW_AVX512_F = (info[1] & (1 << 16)) != 0;
				HW_AVX512_CD = (info[1] & (1 << 28)) != 0;
				HW_AVX512_PF = (info[1] & (1 << 26)) != 0;
				HW_AVX512_ER = (info[1] & (1 << 27)) != 0;

				HW_AVX512_VL = (info[1] & (1u << 31)) != 0;
				HW_AVX512_BW = (info[1] & (1 << 30)) != 0;
				HW_AVX512_DQ = (info[1] & (1 << 17)) != 0;

				HW_AVX512_IFMA = (info[1] & (1 << 21)) != 0;
				HW_AVX512_VBMI = (info[2] & (1 << 1)) != 0;

				HW_AVX512_VPOPCNTDQ = (info[2] & (1 << 14)) != 0;
				HW_AVX512_4FMAPS = (info[3] & (1 << 2)) != 0;
				HW_AVX512_4VNNIW = (info[3] & (1 << 3)) != 0;

				HW_AVX512_VNNI = (info[2] & (1 << 11)) != 0;

				HW_AVX512_VBMI2 = (info[2] & (1 << 6)) != 0;
				HW_GFNI = (info[2] & (1 << 8)) != 0;
				HW_VAES = (info[2] & (1 << 9)) != 0;
				HW_AVX512_VPCLMUL = (info[2] & (1 << 10)) != 0;
				HW_AVX512_BITALG = (info[2] & (1 << 12)) != 0;

				cpuid(info, 0x00000007, 1);
				HW_AVX512_BF16 = (info[0] & (1 << 5)) != 0;
			}
			if (nExIds >= 0x80000001)
			{
				cpuid(info, 0x80000001, 0);
				HW_x64 = (info[3] & (1 << 29)) != 0;
				HW_ABM = (info[2] & (1 << 5)) != 0;
				HW_SSE4a = (info[2] & (1 << 6)) != 0;
				HW_FMA4 = (info[2] & (1 << 16)) != 0;
				HW_XOP = (info[2] & (1 << 11)) != 0;
			}
		}
};

namespace avocado
{
	namespace backend
	{
		avStatus_t cpuGetDeviceProperty(avDeviceProperty_t propertyName, void *result)
		{
			static const cpu_x86 features;

			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;

			switch (propertyName)
			{
				case AVOCADO_DEVICE_NAME:
					std::memcpy(result, features.vendor_name.data(), features.vendor_name.length());
					break;
				case AVOCADO_DEVICE_PROCESSOR_COUNT:
					reinterpret_cast<int32_t*>(result)[0] = features.cores;
					break;
				case AVOCADO_DEVICE_MEMORY:
					reinterpret_cast<int64_t*>(result)[0] = features.memory;
					break;
				case AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION:
					reinterpret_cast<bool*>(result)[0] = features.OS_AVX and features.HW_F16C;
					break;
				case AVOCADO_DEVICE_SUPPORTS_BFLOAT16:
				case AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION:
				case AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION:
					reinterpret_cast<bool*>(result)[0] = true;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSE:
					reinterpret_cast<bool*>(result)[0] = features.HW_SSE;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSE2:
					reinterpret_cast<bool*>(result)[0] = features.HW_SSE2;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSE3:
					reinterpret_cast<bool*>(result)[0] = features.HW_SSE3;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSSE3:
					reinterpret_cast<bool*>(result)[0] = features.HW_SSSE3;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSE41:
					reinterpret_cast<bool*>(result)[0] = features.HW_SSE41;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSE42:
					reinterpret_cast<bool*>(result)[0] = features.HW_SSE42;
					break;
				case AVOCADO_DEVICE_SUPPORTS_AVX:
					reinterpret_cast<bool*>(result)[0] = features.OS_AVX and features.HW_AVX;
					break;
				case AVOCADO_DEVICE_SUPPORTS_AVX2:
					reinterpret_cast<bool*>(result)[0] = features.OS_AVX and features.HW_AVX2;
					break;
				case AVOCADO_DEVICE_SUPPORTS_AVX512_F:
					reinterpret_cast<bool*>(result)[0] = features.OS_AVX512 and features.HW_AVX512_F;
					break;
				case AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ:
					reinterpret_cast<bool*>(result)[0] = features.OS_AVX512 and features.HW_AVX512_VL and features.HW_AVX512_BW
							and features.HW_AVX512_DQ;
					break;
				case AVOCADO_DEVICE_SUPPORTS_DP4A:
					reinterpret_cast<bool*>(result)[0] = false;
					break;
				case AVOCADO_DEVICE_CUDA_ARCH_MAJOR:
				case AVOCADO_DEVICE_CUDA_ARCH_MINOR:
					reinterpret_cast<int32_t*>(result)[0] = 0;
					break;
				case AVOCADO_DEVICE_SUPPORTS_TENSOR_CORES:
					reinterpret_cast<bool*>(result)[0] = false;
					break;
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	}
/* namespace backend */
} /* namespace avocado */

