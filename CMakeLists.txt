cmake_minimum_required(VERSION 3.18)
project(AvocadoCpuBackend VERSION 1.0.0 DESCRIPTION "CPU backend library" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_WITH_OPENBLAS "Use OpenBlas for linear algebra on CPU?" ON)
option(BUILD_WITH_BLIS "Use BLIS for linear algebra on CPU?" OFF)
set(PATH_TO_MAIN_LIB "Path to the main Avocado project" "${PROJECT_SOURCE_DIR}/../Avocado")

add_library(AvocadoCpuBackend STATIC)
set_target_properties(AvocadoCpuBackend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(AvocadoCpuBackend PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(AvocadoCpuBackend PRIVATE "${PROJECT_SOURCE_DIR}/extern")
target_include_directories(AvocadoCpuBackend PUBLIC "${PATH_TO_MAIN_LIB}/include")
target_compile_definitions(AvocadoCpuBackend PRIVATE USE_CUDA=0)
target_compile_definitions(AvocadoCpuBackend PRIVATE USE_OPENCL=0)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(AvocadoCpuBackend PUBLIC OpenMP::OpenMP_CXX)
else()
	message(FATAL_ERROR "no OpenMP")
endif()

if(BUILD_WITH_OPENBLAS AND BUILD_WITH_BLIS)
	message(FATAL_ERROR "Flags BUILD_WITH_OPENBLAS and BUILD_WITH_BLIS are mutually exclusive.")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_OPENBLAS)
	target_compile_definitions(AvocadoCpuBackend PRIVATE USE_OPENBLAS=1)
	target_link_libraries(AvocadoCpuBackend PUBLIC openblas)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_BLIS)
	target_compile_definitions(AvocadoCpuBackend PRIVATE USE_BLIS=1)
	target_link_libraries(AvocadoCpuBackend PUBLIC "${PROJECT_SOURCE_DIR}/extern/blis/libblis.a")
endif()

add_subdirectory(src)

