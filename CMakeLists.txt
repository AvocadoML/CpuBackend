cmake_minimum_required(VERSION 3.18)
project(AvocadoCpuBackend VERSION 1.0.0 DESCRIPTION "CPU backend library" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_WITH_OPENBLAS "Use OpenBlas for linear algebra on CPU?" ON)
option(BUILD_WITH_BLIS "Use BLIS for linear algebra on CPU?" OFF)
option(DYNAMIC_ARCH "Use runtime dispatching of simd code" ON)
option(BUILD_SHARED_LIBS "Build shared library?" OFF)
set(PATH_TO_MAIN_LIB "${PROJECT_SOURCE_DIR}/../Avocado" CACHE STRING "Path to the main Avocado project")

set(libsrc 	src/cpu_properties.cpp
			src/descriptors.cpp
			src/dispatcher.cpp
			src/gemms.cpp)

if(BUILD_SHARED_LIBS)
  set(LIBRARY_NAME "AvocadoCpuBackend")
  add_library(${LIBRARY_NAME} SHARED ${libsrc})
  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(LIBRARY_NAME "AvocadoCpuBackend_static")
  add_library(${LIBRARY_NAME} STATIC ${libsrc})
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
target_include_directories(${LIBRARY_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/extern")
target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_CPU=1 USE_CUDA=0 USE_OPENCL=0)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
else()
	message(FATAL_ERROR "no OpenMP")
endif()

if(BUILD_WITH_OPENBLAS AND BUILD_WITH_BLIS)
	message(FATAL_ERROR "Flags BUILD_WITH_OPENBLAS and BUILD_WITH_BLIS are mutually exclusive.")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_OPENBLAS)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_OPENBLAS=1)
	target_link_libraries(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/extern/openblas/lib/libopenblas.a")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_BLIS)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_BLIS=1)
	target_link_libraries(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/extern/blis/libblis.a")
endif()

set(kernelsrc 	src/kernels/activation.cpp
				src/kernels/batchnorm.cpp
				src/kernels/conv2d_implicit_gemm.cpp
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
				src/kernels/tensor_unary_op.cpp
				src/kernels/winograd_fused.cpp
				src/kernels/winograd_nonfused.cpp)
				
if(DYNAMIC_ARCH)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE DYNAMIC_ARCH=1)
	add_library(base_backend OBJECT ${kernelsrc})
	target_compile_definitions(base_backend PRIVATE COMPILE_COMMON_CODE)
	target_compile_definitions(base_backend PRIVATE USE_CPU=1 USE_CUDA=0 USE_OPENCL=0 DYNAMIC_ARCH=1)
	set_target_properties(base_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(base_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
	
	add_library(sse2_backend OBJECT ${kernelsrc})
	target_compile_options(sse2_backend PRIVATE -msse2)
	target_compile_definitions(sse2_backend PRIVATE USE_CPU=1 USE_CUDA=0 USE_OPENCL=0 DYNAMIC_ARCH=1)
	set_target_properties(sse2_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(sse2_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
	
	add_library(sse41_backend OBJECT ${kernelsrc})
	target_compile_options(sse41_backend PRIVATE -msse4.1)
	target_compile_definitions(sse41_backend PRIVATE USE_CPU=1 USE_CUDA=0 USE_OPENCL=0 DYNAMIC_ARCH=1)
	set_target_properties(sse41_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(sse41_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
	
	add_library(avx_backend OBJECT ${kernelsrc})
	target_compile_options(avx_backend PRIVATE -mavx)
	target_compile_definitions(avx_backend PRIVATE USE_CPU=1 USE_CUDA=0 USE_OPENCL=0 DYNAMIC_ARCH=1)
	set_target_properties(avx_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(avx_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
	
	add_library(avx2_backend OBJECT ${kernelsrc})
	target_compile_options(avx2_backend PRIVATE -mavx2 -mfma -mf16c)
	target_compile_definitions(avx2_backend PRIVATE USE_CPU=1 USE_CUDA=0 USE_OPENCL=0 DYNAMIC_ARCH=1)
	set_target_properties(avx2_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(avx2_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
	
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
	endif()
endif()


install(TARGETS ${LIBRARY_NAME})
install(FILES "include/CpuBackend/cuda_backend.h" DESTINATION include/CpuBackend)
install(FILES "${PATH_TO_MAIN_LIB}/include/Avocado/backend/backend_defs.h" DESTINATION include/CpuBackend)
