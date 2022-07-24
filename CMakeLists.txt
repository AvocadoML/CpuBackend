cmake_minimum_required(VERSION 3.18)
project(AvocadoCpuBackend VERSION 1.0.0 DESCRIPTION "CPU backend library" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_WITH_OPENBLAS "Use OpenBlas for linear algebra on CPU?" ON)
option(DYNAMIC_ARCH "Use runtime dispatching of simd code" OFF)
option(ENABLE_FAST_MATH "Use fast arithmetic functions in few places" OFF)
option(BUILD_SHARED_LIBS "Build shared library?" OFF)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose build type" FORCE)
set(PATH_TO_COMMON_UTILS "${PROJECT_SOURCE_DIR}/../CommonBackendUtils" CACHE STRING "Path to the common backend utils project")

set(commonsrc 	${PATH_TO_COMMON_UTILS}/src/descriptors/ContextDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/ConvolutionDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/DropoutDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/MemoryDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/OptimizerDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/PoolingDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/TensorDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/backend_utils.cpp)

set(libsrc 	src/cpu_properties.cpp
			src/descriptors.cpp
			src/dispatcher.cpp
			src/gemms.cpp
			src/utils.cpp)

if(BUILD_SHARED_LIBS)
  set(LIBRARY_NAME "AvocadoCpuBackend")
  add_library(${LIBRARY_NAME} SHARED ${libsrc} ${commonsrc})
  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(LIBRARY_NAME "AvocadoCpuBackend_static")
  add_library(${LIBRARY_NAME} STATIC ${libsrc} ${commonsrc})
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
target_compile_definitions(${LIBRARY_NAME} PRIVATE CPU_BACKEND)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
else()
	message(FATAL_ERROR "no OpenMP")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_OPENBLAS)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_OPENBLAS=1)
	target_link_libraries(${LIBRARY_NAME} PUBLIC openblas)
endif()

set(kernelsrc 	src/kernels/activation.cpp
				src/kernels/batchnorm.cpp
				src/kernels/conv2d_implicit_gemm.cpp
				src/kernels/conv2d_winograd_fused.cpp
				src/kernels/conv2d_winograd_nonfused.cpp
				src/kernels/conversion.cpp
				src/kernels/convolution.cpp
				src/kernels/dropout.cpp
				src/kernels/im2row.cpp
				src/kernels/losses.cpp
				src/kernels/metrics.cpp
				src/kernels/optimizers.cpp
				src/kernels/pooling.cpp
				src/kernels/regularizers.cpp
				src/kernels/softmax.cpp
				src/kernels/tensor_binary_op.cpp
				src/kernels/tensor_op.cpp
				src/kernels/tensor_reduction.cpp
				src/kernels/tensor_unary_op.cpp)
				
if(DYNAMIC_ARCH)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE DYNAMIC_ARCH=1)
	add_library(base_backend OBJECT ${kernelsrc})
	target_compile_definitions(base_backend PRIVATE COMPILE_COMMON_CODE)
	target_compile_definitions(base_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(base_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(base_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
	
	add_library(sse2_backend OBJECT ${kernelsrc})
	target_compile_options(sse2_backend PRIVATE -msse2)
	target_compile_definitions(sse2_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(sse2_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(sse2_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
	
	add_library(sse41_backend OBJECT ${kernelsrc})
	target_compile_options(sse41_backend PRIVATE -msse4.1)
	target_compile_definitions(sse41_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(sse41_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(sse41_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
	
	add_library(avx_backend OBJECT ${kernelsrc})
	target_compile_options(avx_backend PRIVATE -mavx -mf16c)
	target_compile_definitions(avx_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(avx_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(avx_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
	
	add_library(avx2_backend OBJECT ${kernelsrc})
	target_compile_options(avx2_backend PRIVATE -mavx2 -mfma -mf16c)
	target_compile_definitions(avx2_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(avx2_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(avx2_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
	
	if(BUILD_SHARED_LIBS)
		set_target_properties(base_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
		set_target_properties(sse2_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
		set_target_properties(sse41_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
		set_target_properties(avx_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
		set_target_properties(avx2_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
	endif()
	
	target_link_libraries(${LIBRARY_NAME} PRIVATE base_backend sse2_backend sse41_backend avx_backend avx2_backend)
	
else()
	target_compile_definitions(${LIBRARY_NAME} PRIVATE DYNAMIC_ARCH=0)
	target_sources(${LIBRARY_NAME} PRIVATE ${kernelsrc})
	if(MSVC)
		
	else()
		target_compile_options(${LIBRARY_NAME} PRIVATE -march=native)
#		target_compile_options(${LIBRARY_NAME} PRIVATE -fsanitize=address -fsanitize=undefined)
	endif()
endif()


install(TARGETS ${LIBRARY_NAME})
install(FILES "include/Avocado/cpu_backend.h" DESTINATION include/Avocado)
