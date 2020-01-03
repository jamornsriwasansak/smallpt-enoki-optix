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

#include "optixdata.h"

// SBT record with an appropriately aligned and sized data block
template<typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT)
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

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
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		// Our device code uses 3 payload registers (r,g,b output value)
		pipeline_compile_options.numPayloadValues = 4;
		pipeline_compile_options.numAttributeValues = 0;
		// This is the name of the param struct variable in our device code
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		std::string ptx;
		readSourceFile(ptx, "ptxfiles/src/wavefront_isect.ptx");

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixModule module = nullptr; // The output module
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		optixModuleCreateFromPTX(
			m_optix_context,
			&module_compile_options,
			&pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&module);


		// Create program groups.
		OptixProgramGroup raygen_prog_group = nullptr;
		OptixProgramGroup miss_prog_group = nullptr;
		OptixProgramGroup hitgroup_prog_group = nullptr;

		OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

		OptixProgramGroupDesc raygen_prog_group_desc = {}; //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		optixProgramGroupCreate(
			m_optix_context,
			&raygen_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&raygen_prog_group
		);

		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		optixProgramGroupCreate(
			m_optix_context,
			&miss_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&miss_prog_group
		);

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		optixProgramGroupCreate(
			m_optix_context,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group
		);


		// Link pipeline.
		OptixPipeline pipeline = nullptr;
		{
			OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

			OptixPipelineLinkOptions pipeline_link_options = {};
			pipeline_link_options.maxTraceDepth = 5;
			pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
			pipeline_link_options.overrideUsesMotionBlur = false;
			char log[2048]; size_t sizeof_log = sizeof(log);
			optixPipelineCreate(
				m_optix_context,
				&pipeline_compile_options,
				&pipeline_link_options,
				program_groups,
				sizeof(program_groups) / sizeof(program_groups[0]),
				log,
				&sizeof_log,
				&pipeline
			);
		}

		// Set up shader binding table.
		OptixShaderBindingTable sbt = {};
		{
			CUdeviceptr raygen_record;
			const size_t raygen_record_size = sizeof(RayGenSbtRecord);
			cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size);
			RayGenSbtRecord rg_sbt;
			rg_sbt.data = {};
			optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt);
			cudaMemcpy(
				reinterpret_cast<void *>(raygen_record),
				&rg_sbt,
				raygen_record_size,
				cudaMemcpyHostToDevice
			);

			CUdeviceptr miss_record;
			size_t      miss_record_size = sizeof(MissSbtRecord);
			cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size);
			MissSbtRecord ms_sbt;
			optixSbtRecordPackHeader(miss_prog_group, &ms_sbt);
			cudaMemcpy(
				reinterpret_cast<void *>(miss_record),
				&ms_sbt,
				miss_record_size,
				cudaMemcpyHostToDevice
			);

			CUdeviceptr hitgroup_record;
			size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
			cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size);
			HitGroupSbtRecord hg_sbt;
			optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt);
			cudaMemcpy(
				reinterpret_cast<void *>(hitgroup_record),
				&hg_sbt,
				hitgroup_record_size,
				cudaMemcpyHostToDevice
			);

			sbt.raygenRecord = raygen_record;
			sbt.missRecordBase = miss_record;
			sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
			sbt.missRecordCount = 1;
			sbt.hitgroupRecordBase = hitgroup_record;
			sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
			sbt.hitgroupRecordCount = 1;
		}

		OptixRayResult * result;
		cudaMalloc(&result, sizeof(OptixRayResult) * 1024 * 768);

		Params param;
		param.handle = gas_handle;
		param.closest = true;
		param.results = result;

		CUdeviceptr d_param;
		cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params));
		cudaMemcpy(reinterpret_cast<void *>(d_param), &param, sizeof(Params), cudaMemcpyHostToDevice);

		// Launch now, passing in our pipeline, launch params, and SBT
		optixLaunch(pipeline,
					0,   // Default CUDA stream
					d_param,
					sizeof(Params),
					&sbt,
					1024,
					768,
					1); // depth

		// check if hit
		OptixRayResult * testtt = new OptixRayResult[1024 * 768];
		cudaMemcpy(testtt, result, sizeof(OptixRayResult) * 1024 * 768, cudaMemcpyDeviceToHost);

		std::cout << "t:" << testtt->id << std::endl;
		std::cout << "id:" << testtt->t << std::endl;
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