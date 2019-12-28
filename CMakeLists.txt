cmake_minimum_required(VERSION 2.8)
project(wurstframework)

if(UNIX AND NOT APPLE)
	set(LINUX TRUE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Embree
	set(EMBREE_ISA_AVX OFF CACHE BOOL "Enables AVX ISA" FORCE)
	set(EMBREE_ISA_AVX2 ON CACHE BOOL "Enables AVX2 ISA" FORCE)
	set(EMBREE_ISA_SSE2 OFF CACHE BOOL "Enables SSE ISA" FORCE)
	set(EMBREE_ISA_SSE42 OFF CACHE BOOL "Enables SSE4.2 ISA" FORCE)
	set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "Build Embree with support for ISPC applications." FORCE)
	set(EMBREE_GEOMETRY_CURVE OFF CACHE BOOL "Enables support for curve geometries" FORCE)
	set(EMBREE_GEOMETRY_GRID OFF CACHE BOOL "Enables support for grid geometries" FORCE)
	set(EMBREE_GEOMETRY_INSTANCE OFF CACHE BOOL "Enables support for instances" FORCE)
	set(EMBREE_GEOMETRY_POINT OFF CACHE BOOL "Enables support for point geometries" FORCE)
	set(EMBREE_GEOMETRY_QUAD OFF CACHE BOOL "Enables support for quad geometries" FORCE)
	set(EMBREE_GEOMETRY_SUBDIVISION OFF CACHE BOOL "Enables support for subdiv geometries" FORCE)
	set(EMBREE_RAY_PACKETS OFF CACHE BOOL "Enabled support for ray packets" FORCE)
	set(EMBREE_STAT_COUNTERS OFF CACHE BOOL "Enables statistic counters." FORCE)
	set(EMBREE_STATIC_LIB ON CACHE BOOL "Build Embree as a static library." FORCE)
	set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "Selects tasking system" FORCE)
	set(EMBREE_TESTING_INTENSITY "0" CACHE BOOL "Intensity of testing (0 = no testing, 1 = verify and tutorials, 2 = light testing, 3 = intensive testing" FORCE)
	set(EMBREE_TUTORIALS OFF CACHE BOOL "Enable to build Embree tutorials" FORCE)
	add_subdirectory(dependencies/embree)
	include_directories(dependencies/embree/include)

# CUDA (for Optix and Enoki)
	enable_language(CUDA)
	find_package(CUDA)
	include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Optix prime
	set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.5.0")
	find_package(OptiX)
	include_directories("${OptiX_INCLUDE}")

# Enoki
	set(ENOKI_CUDA ON CACHE BOOL "Build Enoki CUDA library?" FORCE)
	set(ENOKI_AUTODIFF ON CACHE BOOL "Build Enoki automatic differentation library?" FORCE)
	set(ENOKI_PYTHON OFF CACHE BOOL "Build pybind11 interface to CUDA & automatic differentiation libraries?" FORCE)
	add_subdirectory(dependencies/enoki)
	include_directories(dependencies/enoki/include)

#[[
# pbrtParser
	add_subdirectory(dependencies/pbrt-parser EXCLUDE_FROM_ALL)
	include_directories(dependencies/pbrt-parser/pbrtParser/include)
]]

# Worst Framework Ever
	# C++ Mode
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAVX2")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

	# output directory
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/lib")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/lib")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sandbox")

	# add the wurst sourcecode
	file(GLOB_RECURSE wurst_src
		"src/*.h"
		"src/*.hpp"
		"src/*.cpp"
		"src/*.cc")
	include_directories(src)
	add_executable(wurstframework "${wurst_src}")
	add_dependencies(wurstframework optixshaders)
	target_link_libraries(wurstframework embree)
	target_link_libraries(wurstframework enoki-cuda enoki-autodiff)
	target_link_libraries(wurstframework ${CUDA_LIBRARIES})
	target_link_libraries(wurstframework optix_prime)
	#target_link_libraries(wurstframework pbrtParser_static)

	# Windows requires copying dlls of: enoki, optix
	add_custom_command(TARGET wurstframework POST_BUILD
		# copy enoki dlls
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/dependencies/enoki/enoki-cuda.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/dependencies/enoki/enoki-autodiff.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		
		# copy optix dlls
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.5.0/bin64/optix.6.5.0.dll" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.5.0/bin64/optix_prime.6.5.0.dll" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# OpenGL and X11 for the viewer
	find_package(OpenGL)
	if (OpenGL_FOUND)
		if (LINUX)
			find_package(X11)
			if (X11_FOUND)
				include_directories($(X11_INCLUDE_DIR))
				add_definitions(-DWURST_OPENGL)
				target_link_libraries(wurstframework ${OPENGL_gl_LIBRARY})
				target_link_libraries(wurstframework ${X11_LIBRARIES} GL)
			endif()
		elseif (WIN32)
			add_definitions(-DWURST_OPENGL)
			target_link_libraries(wurstframework ${OPENGL_gl_LIBRARY})
		endif()
	endif()

# for me
set(CMAKE_VERBOSE_MAKEFILE ON)
