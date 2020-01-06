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

std::string createMessage(OptixResult res, const char * msg)
{
	std::ostringstream out;
	out << optixGetErrorName(res) << ": " << msg;
	return out.str();
}

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw std::runtime_error( createMessage(res, ss.str().c_str()) );  \
        }                                                                      \
    } while( 0 )


#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log > sizeof( log ) ? "<TRUNCATED>" : "" )          \
               << "\n";                                                        \
            throw std::runtime_error( createMessage(res, ss.str().c_str()) );  \
        }                                                                      \
    } while( 0 )

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw std::runtime_error( ss.str().c_str() );                      \
        }                                                                      \
    } while( 0 )


#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw std::runtime_error( ss.str().c_str() );                      \
        }                                                                      \
    } while( 0 )



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

	TriangleHitInfo(const IntT & tri_id,
					const RealT & t,
					const Real2T & barycentric,
					const Real3T & position,
					const Real3T & geometry_normal):
		m_tri_id(tri_id),
		m_t(t),
		m_barycentric(barycentric),
		m_position(position),
		m_geometry_normal(geometry_normal)
	{
	}

	IntT	m_tri_id;
	Real2T	m_barycentric;
	Real3T	m_position;
	Real3T	m_geometry_normal;
	RealT	m_t;
	//Real3T	m_shading_normal;
};

using TriangleHitInfoC = TriangleHitInfo<IntC, RealC>;
using TriangleHitInfoS = TriangleHitInfo<Int, Real>;

bool read_source_file(std::string & str, const std::string & filename)
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
		m_triangles_aos(empty<IntC>()),
		m_vertices(empty<Real3C>()),
		m_vertices_aos(empty<RealC>())
	{
		optixInit();
		OptixDeviceContextOptions optix_options = {};
		CUcontext cu_context = 0;
		optixDeviceContextCreate(cu_context, &optix_options, &m_optix_context);
	}

	void set_triangles_soup(const int3 * triangles_host, const size_t num_triangles, const float3 * vertices_host, const size_t num_vertices)
	{
		m_triangles_aos = IntC::copy(triangles_host, 3 * num_triangles);
		m_vertices_aos = RealC::copy(vertices_host, 3 * num_vertices);
		cuda_eval();

		CUdeviceptr test2 = reinterpret_cast<CUdeviceptr>(m_triangles_aos.data());
		CUdeviceptr test = reinterpret_cast<CUdeviceptr>(m_vertices_aos.data());

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

		CUdeviceptr temp_buffer_gas, gas_output_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp_buffer_gas), gas_buffer_size.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gas_output_buffer), gas_buffer_size.outputSizeInBytes));

		m_optix_gas_handle = 0;
		OPTIX_CHECK(optixAccelBuild(m_optix_context, 0, &accel_options, &triangle_input, 1, temp_buffer_gas,
						gas_buffer_size.tempSizeInBytes, gas_output_buffer, gas_buffer_size.outputSizeInBytes,
						&m_optix_gas_handle, nullptr, 0));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(temp_buffer_gas)));

		// Pipeline options must be consistent for all modules used in a
		// single pipeline
		OptixPipelineCompileOptions pipeline_compile_options = {};
		pipeline_compile_options.usesMotionBlur = false;

		// for our scene hierarchy. We use a single GAS – no instancing or multi-level hierarchies
		// Our device code uses 4 payload registers (triangle_id, t, uv.x, uv.y)
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.numPayloadValues = 4;
		pipeline_compile_options.numAttributeValues = 0;
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		std::string ptx;
		read_source_file(ptx, "ptxfiles/wavefront_isect.ptx");

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixModule module = nullptr; // The output module
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m_optix_context, &module_compile_options, &pipeline_compile_options, ptx.c_str(), ptx.size(), log, &sizeof_log, &module));

		// Create program groups.
		OptixProgramGroup raygen_prog_group = nullptr;
		OptixProgramGroup miss_prog_group = nullptr;
		OptixProgramGroup hitgroup_prog_group = nullptr;
		OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
		OptixProgramGroupDesc raygen_prog_group_desc = {}; //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &raygen_prog_group_desc, 1, &program_group_options, log, &sizeof_log, &raygen_prog_group));

		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &miss_prog_group_desc, 1, &program_group_options, log, &sizeof_log, &miss_prog_group));

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &hitgroup_prog_group_desc, 1, &program_group_options, log, &sizeof_log, &hitgroup_prog_group));

		// Link pipeline.
		m_optix_pipeline = nullptr;
		OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };
		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = 5;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		pipeline_link_options.overrideUsesMotionBlur = false;
		OPTIX_CHECK_LOG(optixPipelineCreate(m_optix_context, &pipeline_compile_options, &pipeline_link_options, program_groups,
							sizeof(program_groups) / sizeof(program_groups[0]), log, &sizeof_log, &m_optix_pipeline));

		// Set up shader binding table.
		CUdeviceptr raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
		RayGenSbtRecord rg_sbt;
		rg_sbt.data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

		CUdeviceptr miss_record;
		size_t      miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
		MissSbtRecord ms_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

		CUdeviceptr hitgroup_record;
		size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
		HitGroupSbtRecord hg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

		m_optix_sbt = {};
		m_optix_sbt.raygenRecord = raygen_record;
		m_optix_sbt.missRecordBase = miss_record;
		m_optix_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		m_optix_sbt.missRecordCount = 1;
		m_optix_sbt.hitgroupRecordBase = hitgroup_record;
		m_optix_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		m_optix_sbt.hitgroupRecordCount = 1;

		// convert AoS to SoA for storing triangles internally
		const IntC t_indices = arange<IntC>(num_triangles);
		const IntC t0 = gather<IntC>(m_triangles_aos, t_indices * 3 + 0);
		const IntC t1 = gather<IntC>(m_triangles_aos, t_indices * 3 + 1);
		const IntC t2 = gather<IntC>(m_triangles_aos, t_indices * 3 + 2);
		m_triangles = Int3C(t0, t1, t2);
		const IntC p_indices = arange<IntC>(num_vertices);
		const RealC p0 = gather<RealC>(m_vertices_aos, p_indices * 3 + 0);
		const RealC p1 = gather<RealC>(m_vertices_aos, p_indices * 3 + 1);
		const RealC p2 = gather<RealC>(m_vertices_aos, p_indices * 3 + 2);
		m_vertices = Real3C(p0, p1, p2);

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
	}

	TriangleHitInfoC intersect(const Ray3C & rays)
	{
		// make sure that enoki makes the rays variable ready
		cuda_eval();

		IntC tri_id = empty<IntC>(rays.m_dir.z().size());
		RealC barycentric_u = empty<RealC>(rays.m_dir.z().size());
		RealC barycentric_v = empty<RealC>(rays.m_dir.z().size());
		RealC t = empty<RealC>(rays.m_dir.z().size());

		Params param;
		param.m_optix_handle = m_optix_gas_handle;
		param.m_do_closest = true;
		param.m_ray_origin_x = rays.m_origin.x().data();
		param.m_ray_origin_y = rays.m_origin.y().data();
		param.m_ray_origin_z = rays.m_origin.z().data();
		param.m_ray_dir_x = rays.m_dir.x().data();
		param.m_ray_dir_y = rays.m_dir.y().data();
		param.m_ray_dir_z = rays.m_dir.z().data();
		param.m_ray_tmin = rays.m_tmin.data();
		param.m_ray_tmax = rays.m_tmax.data();
		param.m_result_tri_id = tri_id.data();
		param.m_result_t = t.data();
		param.m_result_barycentric_u = barycentric_u.data();
		param.m_result_barycentric_v = barycentric_v.data();

		cudaMemcpy(reinterpret_cast<void *>(d_param), &param, sizeof(Params), cudaMemcpyHostToDevice);

		// Launch now, passing in our pipeline, launch params, and SBT
		optixLaunch(m_optix_pipeline,
					0,   // Default CUDA stream
					d_param,
					sizeof(Params),
					&m_optix_sbt,
					rays.m_origin.x().size(), // width (number of all rays)
					1,  // height
					1); // depth

		const Real2C barycentric = Real2C(barycentric_u, barycentric_v);
		const Int3C vertex_indices = gather<Int3C>(m_triangles, tri_id);
		const Real3C p = select(eq(tri_id, -1), zero<RealC>(), rays.m_origin + t * rays.m_dir);
		const Real3C p0 = gather<Real3C>(m_vertices, vertex_indices.x());
		const Real3C p1 = gather<Real3C>(m_vertices, vertex_indices.y());
		const Real3C p2 = gather<Real3C>(m_vertices, vertex_indices.z());
		const Real3C geometry_normal = compute_geometry_normal(p0, p1, p2);

		return TriangleHitInfoC(tri_id, t, barycentric, p, geometry_normal);
	}

	CUdeviceptr d_param;

	Int3C					m_triangles;
	IntC					m_triangles_aos;
	Real3C					m_vertices;
	RealC					m_vertices_aos;
	OptixDeviceContext		m_optix_context;
	OptixPipeline			m_optix_pipeline;
	OptixTraversableHandle	m_optix_gas_handle;
	OptixShaderBindingTable m_optix_sbt;
};