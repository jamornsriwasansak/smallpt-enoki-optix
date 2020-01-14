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

std::string createMessage(OptixResult res,
						  const char * msg)
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
					const Real3T & geometry_normal,
					const Real3T & shading_normal,
					const Real2T & texcoord):
		m_tri_id(tri_id),
		m_t(t),
		m_barycentric(barycentric),
		m_position(position),
		m_geometry_normal(geometry_normal),
		m_shading_normal(shading_normal),
		m_texcoord(texcoord)
	{
	}

	IntT	m_tri_id;
	Real2T	m_barycentric;
	Real3T	m_position;
	Real3T	m_geometry_normal;
	Real3T	m_shading_normal;
	Real2T	m_texcoord;
	RealT	m_t;
};

using TriangleHitInfoC = TriangleHitInfo<IntC, RealC>;
using TriangleHitInfoS = TriangleHitInfo<Int, Real>;

template <typename Array_>
Array_ soa_from_aos(const value_t<Array_> & array_aos, const BoolC & mask = true)
{
	Array_ result = empty<Array_>();
	IntC indices = arange<IntC>(0, array_aos.size(), Array_::Size);
	for (size_t i = 0; i < Array_::Size; i++)
	{
		result[i] = gather<value_t<Array_>>(array_aos, indices + i, mask);
	}
	return result;
}

Int3C soa_from_aos_x(const IntC & array_aos, const BoolC & mask = true)
{
	Int3C result = empty<Int3C>();
	IntC indices = arange<IntC>(0, array_aos.size(), 3);
	result.x() = gather<IntC>(array_aos, indices + 0, mask);
	//result.y() = gather<IntC>(array_aos, indices + 1, mask);
	//result.z() = gather<IntC>(array_aos, indices + 2, mask);
	return result;
}

bool read_source_file(std::string & str,
					  const std::string & filename)
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
		m_positions_triplets_aos(empty<IntC>()),
		m_positions_aos(empty<RealC>()),
		m_texcoords_soa(empty<Real2C>()),
		m_texcoord_triplets_soa(empty<Int3C>()),
		m_shading_normals_soa(empty<Real3C>()),
		m_shading_normal_triplets_soa(empty<Real3C>())
	{
		optixInit();
		OptixDeviceContextOptions optix_options = {};
		CUcontext cu_context = 0;
		optixDeviceContextCreate(cu_context, &optix_options, &m_optix_context);
		setup_shaders_and_program();
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
	}

	~OptixBackend()
	{
		cudaFree(reinterpret_cast<void *>(m_optix_raygen_record));
		cudaFree(reinterpret_cast<void *>(m_optix_miss_record));
		cudaFree(reinterpret_cast<void *>(m_optix_hitgroup_record));
		cudaFree(reinterpret_cast<void *>(m_optix_compacted_gas_buffer));
	}

	void setup_shaders_and_program()
	{
		// get ptx
		std::string ptx;
		read_source_file(ptx, "ptxfiles/wavefront_isect.ptx");

		// create log
		char log[2048];
		size_t sizeof_log = sizeof(log);

		// setup pipeline
		OptixPipelineCompileOptions pipeline_compile_options = {};
		pipeline_compile_options.usesMotionBlur = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.numPayloadValues = 4;
		pipeline_compile_options.numAttributeValues = 0;
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		OptixModule module = nullptr; // The output module
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m_optix_context, &module_compile_options, &pipeline_compile_options, ptx.c_str(), ptx.size(), log, &sizeof_log, &module));

		// create program groups
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

		// link pipeline
		m_optix_pipeline = nullptr;
		OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };
		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = 1;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		pipeline_link_options.overrideUsesMotionBlur = false;
		OPTIX_CHECK_LOG(optixPipelineCreate(m_optix_context, &pipeline_compile_options, &pipeline_link_options, program_groups,
											sizeof(program_groups) / sizeof(program_groups[0]), log, &sizeof_log, &m_optix_pipeline));

		// ray gen record
		RayGenSbtRecord rg_sbt;
		rg_sbt.data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

		const size_t sizeof_RayGenSbtRecord = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_optix_raygen_record), sizeof_RayGenSbtRecord));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_optix_raygen_record), &rg_sbt, sizeof_RayGenSbtRecord, cudaMemcpyHostToDevice));

		// miss record
		MissSbtRecord ms_sbt;
		ms_sbt.data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));

		const size_t sizeof_MissSbtRecord = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_optix_miss_record), sizeof_MissSbtRecord));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_optix_miss_record), &ms_sbt, sizeof_MissSbtRecord, cudaMemcpyHostToDevice));

		// hit record
		HitGroupSbtRecord hg_sbt;
		hg_sbt.data = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));

		const size_t sizeof_HitGroupSbtRecord = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_optix_hitgroup_record), sizeof_HitGroupSbtRecord));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_optix_hitgroup_record), &hg_sbt, sizeof_HitGroupSbtRecord, cudaMemcpyHostToDevice));

		m_optix_sbt = {};
		m_optix_sbt.raygenRecord = m_optix_raygen_record;
		m_optix_sbt.missRecordBase = m_optix_miss_record;
		m_optix_sbt.missRecordStrideInBytes = sizeof_MissSbtRecord;
		m_optix_sbt.missRecordCount = 1;
		m_optix_sbt.hitgroupRecordBase = m_optix_hitgroup_record;
		m_optix_sbt.hitgroupRecordStrideInBytes = sizeof_HitGroupSbtRecord;
		m_optix_sbt.hitgroupRecordCount = 1;
	}

	void set_triangles_soup(std::vector<int3> position_triplets,
							std::vector<float3> positions_host,
							std::vector<int3> shading_normal_triplets,
							std::vector<float3> shading_normals_host,
							std::vector<int3> per_face_texcoord_indices_host,
							std::vector<float2> texcoords_host)
	{
		// copy traingles and vertices to GPU
		m_positions_triplets_aos = IntC::copy(position_triplets.data(), 3 * position_triplets.size());
		m_positions_aos = RealC::copy(positions_host.data(), 3 * positions_host.size());

		{
			const IntC shading_normal_triplets_aos = IntC::copy(shading_normal_triplets.data(), 3 * shading_normal_triplets.size());
			m_shading_normal_triplets_soa = soa_from_aos<Real3C>(shading_normal_triplets_aos);
		}

		{
			const RealC shading_normals_aos = RealC::copy(shading_normals_host.data(), 3 * shading_normals_host.size());
			m_shading_normals_soa = soa_from_aos<Real3C>(shading_normals_aos);
		}

		{
			const IntC texcoord_triplets_aos = IntC::copy(per_face_texcoord_indices_host.data(), 3 * per_face_texcoord_indices_host.size());
			m_texcoord_triplets_soa = soa_from_aos<Int3C>(texcoord_triplets_aos);
		}

		{
			const RealC texcoords_aos = RealC::copy(texcoords_host.data(), 2 * texcoords_host.size());
			m_texcoords_soa = soa_from_aos<Real2C>(texcoords_aos);
		}

		cuda_eval();

		const unsigned int num_meshes = 1;
		CUdeviceptr cu_traingles_aos = reinterpret_cast<CUdeviceptr>(m_positions_triplets_aos.data());
		CUdeviceptr cu_vertices_aos = reinterpret_cast<CUdeviceptr>(m_positions_aos.data());

		// specify options for the build
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// allocate and copy triangles
		const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangle_input = {};
		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices = (unsigned int)(positions_host.size());
		triangle_input.triangleArray.vertexBuffers = &cu_vertices_aos;
		triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangle_input.triangleArray.numIndexTriplets = (unsigned int)(position_triplets.size());
		triangle_input.triangleArray.indexBuffer = cu_traingles_aos;
		triangle_input.triangleArray.flags = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;

		// setup accel emit description
		CUdeviceptr compacted_size_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&compacted_size_buffer), sizeof(uint64_t)));
		OptixAccelEmitDesc emit_property = {};
		emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emit_property.result = compacted_size_buffer;
		const unsigned int num_emit_property = 1;

		// compute size of temp_buffer and output_buffer needed for constructing GAS
		OptixAccelBufferSizes gas_buffer_size;
		optixAccelComputeMemoryUsage(m_optix_context, &accel_options, &triangle_input, num_meshes, &gas_buffer_size);

		// create temp_buffer for optixAccelBuild
		CUdeviceptr temp_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temp_buffer), gas_buffer_size.tempSizeInBytes));

		// create output_buffer for optixAccelBuild
		CUdeviceptr output_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_buffer), gas_buffer_size.outputSizeInBytes));

		// build data structure
		OPTIX_CHECK(optixAccelBuild(m_optix_context, 0, &accel_options, &triangle_input, num_meshes,
									temp_buffer, gas_buffer_size.tempSizeInBytes,
									output_buffer, gas_buffer_size.outputSizeInBytes,
									&m_optix_gas_handle, &emit_property, num_emit_property));

		// get GAS' compacted size
		uint64_t compacted_size = 0;
		CUDA_CHECK(cudaMemcpy(&compacted_size, reinterpret_cast<void *>(compacted_size_buffer), sizeof(uint64_t), cudaMemcpyDeviceToHost));

		// perform compaction
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_optix_compacted_gas_buffer), compacted_size));
		OPTIX_CHECK(optixAccelCompact(m_optix_context, 0, m_optix_gas_handle, m_optix_compacted_gas_buffer, compacted_size, &m_optix_gas_handle));

		// free!
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(compacted_size_buffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(temp_buffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(output_buffer)));
	}

	Int3C get_position_triplet(const IntC & indices,
							   const BoolC & mask = true) const
	{
		const IntC t0 = gather<IntC>(m_positions_triplets_aos, indices * 3 + 0, mask);
		const IntC t1 = gather<IntC>(m_positions_triplets_aos, indices * 3 + 1, mask);
		const IntC t2 = gather<IntC>(m_positions_triplets_aos, indices * 3 + 2, mask);
		return Int3C(t0, t1, t2);
	}

	Real3C get_position(const IntC & indices,
						const BoolC & mask = true) const
	{
		const RealC px = gather<RealC>(m_positions_aos, indices * 3 + 0, mask);
		const RealC py = gather<RealC>(m_positions_aos, indices * 3 + 1, mask);
		const RealC pz = gather<RealC>(m_positions_aos, indices * 3 + 2, mask);
		return Real3C(px, py, pz);
	}

	Real2C get_interpolated_texcoord(const Int3C & indices,
									 const Real2C & barycentric_coords,
									 const BoolC & mask = true) const
	{
		const Real2C t0 = gather<Real2C>(m_texcoords_soa, indices.x(), mask);
		const Real2C t1 = gather<Real2C>(m_texcoords_soa, indices.y(), mask);
		const Real2C t2 = gather<Real2C>(m_texcoords_soa, indices.z(), mask);
		return barycentric_interpolate(t0, t1, t2, barycentric_coords);
	}

	Real3C get_interpolated_shading_normal(const Int3C & indices,
										   const Real2C & barycentric_coords,
										   const BoolC & mask = true) const
	{
		const Real3C s0 = gather<Real3C>(m_shading_normals_soa, indices.x(), mask);
		const Real3C s1 = gather<Real3C>(m_shading_normals_soa, indices.y(), mask);
		const Real3C s2 = gather<Real3C>(m_shading_normals_soa, indices.z(), mask);

		return barycentric_interpolate(s0, s1, s2, barycentric_coords);
	}

	std::tuple<TriangleHitInfoC, BoolC> intersect(const Ray3C & rays,
												  const BoolC & mask = true) const
	{
		const size_t num_rays = rays.m_origin.x().size();

		// make sure that enoki makes the rays variable ready
		cuda_eval();

		// prepare result
		IntC tri_id = empty<IntC>(num_rays);
		RealC barycentric_u = empty<RealC>(num_rays);
		RealC barycentric_v = empty<RealC>(num_rays);
		RealC t = empty<RealC>(num_rays);

		// setup param
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
		param.m_ray_mask = mask.data();
		param.m_ray_mask_size = Uint(mask.size());
		param.m_result_tri_id = tri_id.data();
		param.m_result_t = t.data();
		param.m_result_barycentric_u = barycentric_u.data();
		param.m_result_barycentric_v = barycentric_v.data();

		// update param
		cudaMemcpy(reinterpret_cast<void *>(d_param), &param, sizeof(Params), cudaMemcpyHostToDevice);

		// launch optix
		const Uint sizeof_Params = sizeof(Params);
		optixLaunch(m_optix_pipeline, 0, d_param, sizeof_Params, &m_optix_sbt, Uint(num_rays), 1,  1); 

		// active mask
		const BoolC active = neq(tri_id, -1) && mask;

		// barycentric coord
		const Real2C barycentric = Real2C(barycentric_u, barycentric_v);

		// position
		const Real3C p = rays.m_origin + t * rays.m_dir;

		// geometry normal
		const Int3C position_triplet = get_position_triplet(tri_id, active);
		const Real3C p0 = get_position(position_triplet.x(), active);
		const Real3C p1 = get_position(position_triplet.y(), active);
		const Real3C p2 = get_position(position_triplet.z(), active);
		const Real3C geometry_normal = compute_geometry_normal(p0, p1, p2);

		// texcoord
		const Int3C texcoord_triplet = gather<Int3C>(m_texcoord_triplets_soa, tri_id, active);
		const Real2C texcoord = get_interpolated_texcoord(texcoord_triplet, barycentric, active);
		
		// shading normal
		const Int3C shading_normal_triplet = gather<Int3C>(m_shading_normal_triplets_soa, tri_id, active);
		const Real3C shading_normal = get_interpolated_shading_normal(shading_normal_triplet, barycentric, active);

		return std::make_tuple(TriangleHitInfoC(tri_id, t, barycentric, p, geometry_normal, shading_normal, texcoord), active);
	}

	CUdeviceptr				d_param;

	Real2C					m_texcoords_soa;
	Int3C					m_texcoord_triplets_soa;
	Real3C					m_shading_normals_soa;
	Int3C					m_shading_normal_triplets_soa;

	// these two needed to be Array Of Structure(AoS) since it is shared with optix
	IntC					m_positions_triplets_aos;
	RealC					m_positions_aos;

	CUdeviceptr				m_optix_raygen_record, m_optix_miss_record, m_optix_hitgroup_record;
	CUdeviceptr				m_optix_compacted_gas_buffer;
	OptixDeviceContext		m_optix_context;
	OptixPipeline			m_optix_pipeline;
	OptixTraversableHandle	m_optix_gas_handle;
	OptixShaderBindingTable m_optix_sbt;
};