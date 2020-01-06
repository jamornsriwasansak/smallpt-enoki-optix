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

#include <tiny_obj_loader.h>

struct Bsdf
{
	virtual std::tuple<SpectrumC, Real3C> sample(const Real3C & incoming,
												 const Real3C & texture_coord,
												 const Real2C & sample) const = 0;
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
										 const Real2C & sample) const override
	{
		SpectrumC outgoing = cosine_weighted_hemisphere_from_square(sample);
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

int main()
{
	// load mesh
	tinyobj::ObjReader obj_reader;
	tinyobj::ObjReaderConfig obj_reader_config;
	obj_reader_config.triangulate = true;
	obj_reader_config.vertex_color = false;
	bool ret = obj_reader.ParseFromFile("cow.obj", obj_reader_config);

	const tinyobj::attrib_t & tiny_attrib = obj_reader.GetAttrib();
	const tinyobj::shape_t & tiny_shape = obj_reader.GetShapes()[0];
	const std::vector<tinyobj::material_t> & tiny_mat = obj_reader.GetMaterials();

	const size_t num_triangles = tiny_shape.mesh.num_face_vertices.size();
	const size_t num_vertices = tiny_attrib.vertices.size() / 3;
	const size_t num_materials = tiny_mat.size();

	int3 * triangles_host = new int3[num_triangles];
	int * material_id_host = new int[num_triangles];
	float3 * vertices_host = new float3[num_vertices];

	for (size_t i_face = 0; i_face < num_triangles; i_face++)
	{
		// copy triangles
		triangles_host[i_face].x = tiny_shape.mesh.indices[i_face * 3 + 0].vertex_index;
		triangles_host[i_face].y = tiny_shape.mesh.indices[i_face * 3 + 1].vertex_index;
		triangles_host[i_face].z = tiny_shape.mesh.indices[i_face * 3 + 2].vertex_index;

		// copy per triangle material id
		material_id_host[i_face] = tiny_shape.mesh.material_ids[i_face] + 1;
	}

	for (size_t i_vertex = 0; i_vertex < num_vertices; i_vertex++)
	{
		// copy vertices
		vertices_host[i_vertex].x = tiny_attrib.vertices[i_vertex * 3 + 0];
		vertices_host[i_vertex].y = tiny_attrib.vertices[i_vertex * 3 + 1];
		vertices_host[i_vertex].z = tiny_attrib.vertices[i_vertex * 3 + 2];
	}

	IntC material_id = IntC::copy(material_id_host, num_triangles);

	std::vector<std::shared_ptr<Bsdf>> materials_host;
	materials_host.push_back(std::make_shared<LambertBsdf>(SpectrumC(0.5_f)));
	for (size_t i_material = 0; i_material < num_materials; i_material++)
	{
		materials_host.push_back(std::make_shared<LambertBsdf>(SpectrumC(0.5_f)));
	}
	CUDAArray<Bsdf *> materials = CUDAArray<Bsdf *>::copy(raw_ptrs(materials_host).data(), materials_host.size());

	OptixBackend optix_backend;
	optix_backend.set_triangles_soup(triangles_host, num_triangles, vertices_host, num_vertices);
	int width = 1920;
	int height = 1080;

	SpectrumC film = zero<SpectrumC>(width * height);

	PCG32<RealC> rng(PCG32_DEFAULT_STATE, arange<RealC>(width * height));
	int num_samples = 100;
	for (int i = 0; i < num_samples; i++)
	{
		std::cout << i << std::endl;
		int num_pixels = width * height;
		const IntC pixel_index = arange<IntC>(num_pixels);
		const IntC y = pixel_index / width;
		const IntC x = pixel_index % width;
		const Int2C pixel(x, y);
		ThinlensCamera thinlens(Real3(0.0_f, 0.03_f, 0.2_f), Real3(0.0_f, 0.03_f, 0.0_f), Real3(0.0_f, 1.0_f, 0.0_f), 0.00_f, 1.0_f, 70.0_f / 180.0_f * M_PI_f);
		const Real3C origin = thinlens.sample_pos(Real2C(rng.next_float32(), rng.next_float32()));
		const Real3C direction = thinlens.sample_dir(pixel, origin, Int2(width, height), Real2C(rng.next_float32(), rng.next_float32()));
		Ray3C rays(origin, direction, zero<RealC>(origin.x().size()) + 0.0001_f, zero<RealC>(origin.x().size()) + 1e20_f);
		const TriangleHitInfoC hit_info = optix_backend.intersect(rays);

		const Frame3C coord_frame(hit_info.m_geometry_normal);
		const IntC mat_index = gather<IntC>(material_id, hit_info.m_tri_id, hit_info.m_tri_id >= 0);
		const CUDAArray<Bsdf *> bsdf = gather<CUDAArray<Bsdf *>>(materials, mat_index);

		const Real2C second_sample = Real2C(rng.next_float32(), rng.next_float32());
		const Real3C incoming_local = coord_frame.to_local(-direction);
		auto [bsdf_contrib, outgoing_local] = bsdf->sample(incoming_local, hit_info.m_position, second_sample);
		const Real3C outgoing = coord_frame.to_world(outgoing_local);

		// sample out going direction
		Ray3C second_rays(hit_info.m_position, outgoing, zero<RealC>(origin.x().size()) + 0.0001_f, zero<RealC>(origin.x().size()) + 1e20_f);
		const TriangleHitInfoC second_hit_info = optix_backend.intersect(second_rays);
		film += bsdf_contrib * select(neq(hit_info.m_tri_id, -1) && eq(second_hit_info.m_tri_id, -1), zero<RealC>(num_pixels) + 1.0_f, zero<RealC>(num_pixels) + 0.0_f);
	}

	film /= Real(num_samples);

	// image write
	float * film_host = new float[width * height];
	cuda_fetch_element(film_host, film.data()->index_(), 0, sizeof(int) * width * height);

	std::cout << "writing image" << std::endl;
	Fimage::save_pfm_mono(film_host, width, height, "test.pfm");

	std::cout << "done" << std::endl;
	return 0;
}