#pragma once

#include "enoki_entry.h"
#include "ray.h"
#include "add_math.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <nvrtc.h>

#include <fstream>
#include <sstream>

template <typename Int_, typename Real_>
struct TriangleHitInfo
{
	using RealT = Real_;
	using IntT = Int_;
	using Real2T = Array<RealT, 2, false>;
	using Real3T = Array<RealT, 3, false>;

	TriangleHitInfo()
	{
	}

	TriangleHitInfo(const RealT & t, const IntT & tri_id, const Real2T & barycentric,
					const Real3T & position, const Real3T & geometry_normal) :
		m_t(t),
		m_tri_id(tri_id),
		m_barycentric(barycentric),
		m_position(position),
		m_geometry_normal(geometry_normal)
	{
	}

	RealT	m_t;
	IntT	m_tri_id;
	Real2T	m_barycentric;
	Real3T	m_position;
	Real3T	m_geometry_normal;
	//Real3T	m_shading_normal;
};
using TriangleHitInfoC = TriangleHitInfo<IntC, RealC>;
using TriangleHitInfoS = TriangleHitInfo<Int, Real>;

bool readSourceFile(std::string & str, const std::string & filename)
{
	// Try to open file
	std::ifstream file(filename.c_str());
	if (file.good())
	{
		// Found usable source file
		std::stringstream source_buffer;
		source_buffer << file.rdbuf();
		str = source_buffer.str();
		return true;
	}
	return false;
}

#define CUDA_NVRTC_OPTIONS  \
  "-arch", \
  "compute_30", \
  "-use_fast_math", \
  "-lineinfo", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64",

struct OptixBackend
{
	OptixBackend():
		m_triangles(empty<Int3C>()),
		m_vertices(empty<Real3C>())
	{
		optixInit();
		OptixDeviceContextOptions optix_options = {};
		CUcontext cu_context = 0;
		optixDeviceContextCreate(cu_context, &optix_options, &m_optix_context);
	}

	void set_triangles_soup(const int3 * triangles_host, const size_t num_triangles, const float3 * vertices_host, const size_t num_vertices)
	{
		m_triangles = IntC::copy(triangles_host, 3 * num_triangles);
		m_vertices = RealC::copy(vertices_host, 3 * num_vertices);
		CUdeviceptr test2 = reinterpret_cast<CUdeviceptr>(m_triangles.data()->data());
		CUdeviceptr test = reinterpret_cast<CUdeviceptr>(m_vertices.data()->data());

		// Specify options for the build. We use default options for simplicity.
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// Allocate and copy device memory for our input triangle vertices
		const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangle_input = {};

		// Populate the build input struct with our triangle data as well as
		// information about the sizes and types of our data
		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices = num_vertices;
		triangle_input.triangleArray.vertexBuffers = &test;
		triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangle_input.triangleArray.numIndexTriplets = num_triangles;
		triangle_input.triangleArray.indexBuffer = test2;
		triangle_input.triangleArray.flags = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;

		OptixAccelBufferSizes gas_buffer_size;
		optixAccelComputeMemoryUsage(m_optix_context, &accel_options, &triangle_input, 1, &gas_buffer_size);

		CUdeviceptr d_temp_buffer_gas, d_gas_output_buffer;
		cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_size.tempSizeInBytes);
		cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer), gas_buffer_size.outputSizeInBytes);

		OptixTraversableHandle gas_handle = 0;
		optixAccelBuild(m_optix_context,
						0,
						&accel_options,
						&triangle_input,
						1,
						d_temp_buffer_gas,
						gas_buffer_size.tempSizeInBytes,
						d_gas_output_buffer,
						gas_buffer_size.outputSizeInBytes,
						&gas_handle,
						nullptr,
						0);
		cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas));

		// Pipeline options must be consistent for all modules used in a
// single pipeline
		OptixPipelineCompileOptions pipeline_compile_options = {};
		pipeline_compile_options.usesMotionBlur = false;

		// This option is important to ensure we compile code which is optimal
		// for our scene hierarchy. We use a single GAS – no instancing or
		// multi-level hierarchies
		pipeline_compile_options.traversableGraphFlags =
			OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

		// Our device code uses 3 payload registers (r,g,b output value)
		pipeline_compile_options.numPayloadValues = 3;

		// This is the name of the param struct variable in our device code
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
	}

	TriangleHitInfoC intersect(const Ray3C & rays)
	{

		OptixAccelBuildOptions optix_accel_options = {};
		TriangleHitInfoC result;
		return result;
	}

	Int3C					m_triangles;
	Real3C					m_vertices;
	OptixDeviceContext		m_optix_context;
};