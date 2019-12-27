cmake_minimum_required(VERSION 2.8)
project(wurstframework)

if(UNIX AND NOT APPLE)
	set(LINUX TRUE)
endif()

# Worst Framework Ever
	# C++ Mode
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -DAVX2 -march=native")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

	# output directory
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/archive")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/library")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sandbox")

	# add the wurst sourcecode
	file(GLOB_RECURSE wurst_src
		"src/*.h"
		"src/*.hpp"
		"src/*.cpp"
		"src/*.cc")
	include_directories(src)
	add_executable(wurstframework "${wurst_src}")

	# some std libraries to fix problems
	# target_link_libraries(wurstframework stdc++fs)

#[[
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
	target_link_libraries(wurstframework embree)
]]

# Enoki
	set(ENOKI_CUDA "Build Enoki CUDA library?" FORCE)
	set(ENOKI_AUTODIFF "Build Enoki automatic differentation library?" FORCE)
	set(ENOKI_PYTHON "Build pybind11 interface to CUDA & automatic differentiation libraries?" FORCE)
	add_subdirectory(dependencies/enoki)
	include_directories(dependencies/enoki/include)

#[[
# pbrtParser
	add_subdirectory(dependencies/pbrt-parser EXCLUDE_FROM_ALL)
	include_directories(dependencies/pbrt-parser/pbrtParser/include)
	target_link_libraries(wurstframework pbrtParser_static )
]]

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
