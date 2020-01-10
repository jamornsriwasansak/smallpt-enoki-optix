#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <iostream>

#include <cuda_runtime.h>

#include "accel/optix_backend.h"
#include "ray.h"
#include "enoki_entry.h"
#include "fimage.h"
#include "pinhole.h"
#include "mapping.h"
#include "coordframe.h"

#include "stopwatch.h"

#include <tiny_obj_loader.h>

struct Bsdf
{
	virtual std::tuple<SpectrumC, Real3C> sample(const Real3C & incoming,
												 const Real3C & texture_coord,
												 const Real2C & sample,
												 const BoolC & mask = true) const = 0;
};

ENOKI_CALL_SUPPORT_BEGIN(Bsdf)
ENOKI_CALL_SUPPORT_METHOD(sample)
ENOKI_CALL_SUPPORT_END(Bsdf)

struct LambertBsdf : public Bsdf
{
	LambertBsdf(const SpectrumC & reflectance): m_reflectance(reflectance)
	{
	}

	std::tuple<SpectrumC, Real3C> sample(const Real3C & incoming,
										 const Real3C & texture_coord,
										 const Real2C & sample,
										 const BoolC & mask = true) const override
	{
		SpectrumC outgoing = select(mask, cosine_weighted_hemisphere_from_square(sample), empty<SpectrumC>());
		return std::make_tuple(m_reflectance, outgoing);
	}

	SpectrumC m_reflectance;
};

template <typename T>
std::vector<T *> raw_ptrs(const std::vector<std::shared_ptr<T>> & shared_ptrs)
{
	std::vector<T *> result(shared_ptrs.size());
	for (size_t i = 0; i < shared_ptrs.size(); i++)
	{
		result[i] = shared_ptrs[i].get();
	}
	return result;
}

std::tuple<std::vector<int3>, std::vector<int>, std::vector<float3>, std::vector<std::shared_ptr<Bsdf>>> load_meshes(const std::string & path)
{
	// load mesh
	tinyobj::ObjReader obj_reader;
	tinyobj::ObjReaderConfig obj_reader_config;
	obj_reader_config.triangulate = true;
	obj_reader_config.vertex_color = false;
	bool ret = obj_reader.ParseFromFile(path, obj_reader_config);

	// compute num_all_triangles
	const size_t num_shapes = obj_reader.GetShapes().size();
	size_t num_all_triangles = 0;
	for (size_t i_shape = 0; i_shape < num_shapes; i_shape++)
	{
		const tinyobj::shape_t & tiny_shape = obj_reader.GetShapes()[i_shape];
		num_all_triangles += tiny_shape.mesh.num_face_vertices.size();
	}

	// get triangles and material_id
	std::vector<int3> triangles(num_all_triangles);
	std::vector<int> per_face_material_id(num_all_triangles);
	const std::vector<tinyobj::material_t> & tiny_mat = obj_reader.GetMaterials();
	size_t i_face_offset = 0;
	for (size_t i_shape = 0; i_shape < num_shapes; i_shape++)
	{
		const tinyobj::shape_t & tiny_shape = obj_reader.GetShapes()[i_shape];
		const size_t num_triangles = tiny_shape.mesh.num_face_vertices.size();
		for (size_t i_face = 0; i_face < num_triangles; i_face++)
		{
			// copy triangles
			triangles[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].vertex_index;
			triangles[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].vertex_index;
			triangles[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].vertex_index;

			// copy per triangle material id
			per_face_material_id[i_face + i_face_offset] = tiny_shape.mesh.material_ids[i_face] + 1;
		}
		i_face_offset += num_triangles;
	}

	// get vertices
	const tinyobj::attrib_t & tiny_attrib = obj_reader.GetAttrib();
	const size_t num_vertices = tiny_attrib.vertices.size() / 3;
	std::vector<float3> vertices(num_vertices);
	for (size_t i_vertex = 0; i_vertex < num_vertices; i_vertex++)
	{
		// copy vertices
		vertices[i_vertex].x = tiny_attrib.vertices[i_vertex * 3 + 0];
		vertices[i_vertex].y = tiny_attrib.vertices[i_vertex * 3 + 1];
		vertices[i_vertex].z = tiny_attrib.vertices[i_vertex * 3 + 2];
	}

	// get materials
	const size_t num_materials = tiny_mat.size();
	std::vector<std::shared_ptr<Bsdf>> materials(num_materials + 1);
	materials[0] = std::make_shared<LambertBsdf>(SpectrumC(1.0_f));
	for (size_t i_material = 0; i_material < num_materials; i_material++)
	{
		materials[i_material + 1] = std::make_shared<LambertBsdf>(SpectrumC(1.0_f));
	}

	return std::make_tuple(triangles, per_face_material_id, vertices, materials);
}

int main()
{
	auto [triangles_host, material_ids_host, vertices_host, materials_host] = load_meshes("mitsuba.obj");
	IntC material_id = IntC::copy(material_ids_host.data(), material_ids_host.size());
	CUDAArray<Bsdf *> materials = CUDAArray<Bsdf *>::copy(raw_ptrs(materials_host).data(), materials_host.size());

	OptixBackend optix_backend;
	optix_backend.set_triangles_soup(triangles_host.data(), triangles_host.size(), vertices_host.data(), vertices_host.size());
	int width = 1920;
	int height = 1080;

	SpectrumC film = zero<SpectrumC>(width * height);

	PCG32<RealC> rng(PCG32_DEFAULT_STATE, arange<RealC>(width * height));
	int num_samples = 50;
	StopWatch sw;
	sw.reset();
	for (int i = 0; i < num_samples; i++)
	{
		std::cout << i << std::endl;
		int num_pixels = width * height;
		const IntC pixel_index = arange<IntC>(num_pixels);
		const IntC y = pixel_index / width;
		const IntC x = pixel_index % width;
		const Int2C pixel(x, y);
		const ThinlensCamera thinlens(Real3(0.0_f, 3.03_f, 5.0_f), Real3(0.0_f, 0.03_f, 0.0_f), Real3(0.0_f, 1.0_f, 0.0_f), 0.00_f, 1.0_f, 70.0_f / 180.0_f * M_PI_f);
		const Real3C origin = thinlens.sample_pos(Real2C(rng.next_float32(), rng.next_float32()));
		const Real3C direction = thinlens.sample_dir(pixel, origin, Int2(width, height), Real2C(rng.next_float32(), rng.next_float32()));
		const Ray3C rays(origin, direction);
		const auto [hit_info, hit_mask] = optix_backend.intersect(rays);

		const Frame3C coord_frame(hit_info.m_geometry_normal);
		const IntC mat_index = gather<IntC>(material_id, hit_info.m_tri_id, hit_mask);
		const CUDAArray<Bsdf *> bsdf = gather<CUDAArray<Bsdf *>>(materials, mat_index);

		const Real2C second_sample = Real2C(rng.next_float32(), rng.next_float32());
		const Real3C incoming_local = coord_frame.to_local(-direction);
		auto [bsdf_contrib, outgoing_local] = bsdf->sample(incoming_local, hit_info.m_position, second_sample, hit_mask);
		const Real3C outgoing = coord_frame.to_world(outgoing_local);

		// sample out going direction
		const Ray3C second_rays(hit_info.m_position, outgoing);
		const auto [second_hit_info, second_hit_mask] = optix_backend.intersect(second_rays, hit_mask);
		film += bsdf_contrib * select(neq(hit_info.m_tri_id, -1) && eq(second_hit_info.m_tri_id, -1), full<RealC>(1.0_f, num_pixels), full<RealC>(0.0_f, num_pixels));
	}

	film /= Real(num_samples);

	std::cout << sw.timeMilliSec() << std::endl;

	// image write
	float * film_host_r = new float[width * height];
	float * film_host_g = new float[width * height];
	float * film_host_b = new float[width * height];
	cuda_fetch_element(film_host_r, film.x().index_(), 0, sizeof(int) * width * height);
	cuda_fetch_element(film_host_g, film.y().index_(), 0, sizeof(int) * width * height);
	cuda_fetch_element(film_host_b, film.z().index_(), 0, sizeof(int) * width * height);

	std::cout << "writing image" << std::endl;
	Fimage::save_pfm(film_host_r, film_host_g, film_host_b, width, height, "test.pfm");

	std::cout << "done" << std::endl;
	return 0;
}