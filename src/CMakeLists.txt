target_sources(AvocadoCpuBackend PRIVATE	cpu_features.cpp
											descriptors.cpp
											dispatcher.cpp
											gemms.cpp)

if(WIN32 AND NOT MSVC)
	set_source_files_properties(cpu_features.cpp PROPERTIES COMPILE_FLAGS -mxsave)
endif()

add_subdirectory(kernels)