#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <iostream>
#include <filesystem>

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

struct Texture
{
	virtual SpectrumC eval(const Real3C & uvw, const BoolC & mask = true) const = 0;
};

ENOKI_CALL_SUPPORT_BEGIN(Texture)
ENOKI_CALL_SUPPORT_METHOD(eval)
ENOKI_CALL_SUPPORT_END(Texture)

struct ImageTexture : public Texture
{
	std::shared_ptr<ImageTexture> load(const std::filesystem::path & path)
	{
	}

	ImageTexture(const Spectrum & color):
		m_image(color),
		m_image_size(1, 1)
	{
	}

	SpectrumC texel_fetch(const Int2C & xy) const
	{
		assert(all(xy.x() >= 0));
		assert(all(xy.y() >= 0));
		assert(all(xy.x() < m_image_size.x()));
		assert(all(xy.y() < m_image_size.y()));
		const IntC pixel_index = xy.y() * m_image_size.y() + xy.x();
		return gather<SpectrumC>(m_image, pixel_index);
	}

	SpectrumC texel_fetch_wrap(const Int2C & xy) const
	{
		// TODO:: support more than clamping
		return texel_fetch(clamp(xy, Int2(0, 0), m_image_size - Int2(1, 1)));
	}

	SpectrumC eval(const Real3C & uvw, const BoolC & mask = true) const override
	{
		// scale uv to image size
		const Real2C scaled_uv = Real2C(uvw.x(), uvw.y()) * m_image_size - Real2C(0.5_f);
		const IntC x_pos = enoki::floor2int<IntC>(scaled_uv.x());
		const IntC y_pos = enoki::floor2int<IntC>(scaled_uv.y());
		const RealC dx1 = scaled_uv.x() - x_pos;
		const RealC dy1 = scaled_uv.y() - y_pos;
		const RealC dx2 = 1.0_f - dx1;
		const RealC dy2 = 1.0_f - dy1;
		return texel_fetch_wrap(Int2C(x_pos, y_pos)) * dx2 * dy2
			+ texel_fetch_wrap(Int2C(x_pos, y_pos + 1)) * dx2 * dy1
			+ texel_fetch_wrap(Int2C(x_pos + 1, y_pos)) * dx1 * dy2
			+ texel_fetch_wrap(Int2C(x_pos + 1, y_pos + 1)) * dx1 * dy1;
	}

	SpectrumC m_image;
	Int2 m_image_size;
};

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
	LambertBsdf(const Spectrum & reflectance):
		m_reflectance(std::make_shared<ImageTexture>(reflectance))
	{
	}

	std::tuple<SpectrumC, Real3C> sample(const Real3C & incoming,
										 const Real3C & texture_coord,
										 const Real2C & sample,
										 const BoolC & mask = true) const override
	{
		SpectrumC outgoing = cosine_weighted_hemisphere_from_square(sample);
		SpectrumC contrib = m_reflectance->eval(texture_coord, mask);
		return std::make_tuple(contrib, outgoing);
	}

	std::shared_ptr<ImageTexture> m_reflectance;
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

std::tuple<
	std::vector<int3>, /* per face position indices */
	std::vector<float3>, /* position */
	std::vector<int3>, /* per face normal indices */
	std::vector<float3>, /* normal */
	std::vector<int3>, /* per face texcoord indices */
	std::vector<float2>, /* texcoord */
	std::vector<int>, /* per triangle material id */
	std::vector<std::shared_ptr<LambertBsdf>>> load_meshes(const std::string & path)
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
	std::vector<int3> per_face_position_indices(num_all_triangles);
	std::vector<int3> per_face_shading_normal_indices(num_all_triangles);
	std::vector<int3> per_face_texcoord_indices(num_all_triangles);
	std::vector<int> per_face_material_id(num_all_triangles);
	const std::vector<tinyobj::material_t> & tiny_mat = obj_reader.GetMaterials();
	size_t i_face_offset = 0;
	for (size_t i_shape = 0; i_shape < num_shapes; i_shape++)
	{
		const tinyobj::shape_t & tiny_shape = obj_reader.GetShapes()[i_shape];
		const size_t num_triangles = tiny_shape.mesh.num_face_vertices.size();
		for (size_t i_face = 0; i_face < num_triangles; i_face++)
		{
			per_face_position_indices[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].vertex_index;
			per_face_position_indices[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].vertex_index;
			per_face_position_indices[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].vertex_index;

			per_face_shading_normal_indices[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].normal_index;
			per_face_shading_normal_indices[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].normal_index;
			per_face_shading_normal_indices[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].normal_index;

			per_face_texcoord_indices[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].texcoord_index;
			per_face_texcoord_indices[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].texcoord_index;
			per_face_texcoord_indices[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].texcoord_index;

			// copy per triangle material id
			per_face_material_id[i_face + i_face_offset] = tiny_shape.mesh.material_ids[i_face] + 1;
		}
		i_face_offset += num_triangles;
	}

	// get vertices information
	const tinyobj::attrib_t & tiny_attrib = obj_reader.GetAttrib();

	// vertices
	const size_t num_positions = tiny_attrib.vertices.size() / 3;
	std::vector<float3> positions(num_positions);
	for (size_t i_vertex = 0; i_vertex < num_positions; i_vertex++)
	{
		// copy vertices
		positions[i_vertex].x = tiny_attrib.vertices[i_vertex * 3 + 0];
		positions[i_vertex].y = tiny_attrib.vertices[i_vertex * 3 + 1];
		positions[i_vertex].z = tiny_attrib.vertices[i_vertex * 3 + 2];
	}

	// normals
	const size_t num_shading_normals = tiny_attrib.normals.size() / 3;
	std::vector<float3> shading_normals(num_shading_normals);
	for (size_t i_normal = 0; i_normal < num_shading_normals; i_normal++)
	{
		// copy normals
		shading_normals[i_normal].x = tiny_attrib.normals[i_normal * 3 + 0];
		shading_normals[i_normal].y = tiny_attrib.normals[i_normal * 3 + 1];
		shading_normals[i_normal].z = tiny_attrib.normals[i_normal * 3 + 2];
	}

	// texcoords
	const size_t num_texcoords = tiny_attrib.texcoords.size() / 2;
	std::vector<float2> texcoords(num_texcoords);
	for (size_t i_texcoord = 0; i_texcoord < num_texcoords; i_texcoord++)
	{
		// copy texture coords
		texcoords[i_texcoord].x = tiny_attrib.texcoords[i_texcoord * 2 + 0];
		texcoords[i_texcoord].y = tiny_attrib.texcoords[i_texcoord * 2 + 1];
	}

	// get materials
	const size_t num_materials = tiny_mat.size();
	
	// init materials vector
	std::vector<std::shared_ptr<LambertBsdf>> materials(num_materials + 1);

	// start with default material at 0
	materials[0] = std::make_shared<LambertBsdf>(Spectrum(0.5_f));

	// push the rest of materials
	for (size_t i_material = 0; i_material < num_materials; i_material++)
	{
		const Real r = tiny_mat[i_material].diffuse[0];
		const Real g = tiny_mat[i_material].diffuse[1];
		const Real b = tiny_mat[i_material].diffuse[2];
		materials[i_material + 1] = std::make_shared<LambertBsdf>(Spectrum(r, g, b));
	}

	return std::make_tuple(per_face_position_indices, positions,
						   per_face_shading_normal_indices, shading_normals,
						   per_face_texcoord_indices, texcoords,
						   per_face_material_id, materials);
}

int main()
{
	// all const
	const int width = 1920;
	const int height = 1080;
	const int num_pixels = width * height;
	const int num_samples = 50;
	const int num_bounces = 3;

	// load meshes and materials
	auto [per_face_position_indices, positions,
		per_face_shading_normal_indices, shading_normals,
		per_face_texcoord_indices, texcoords,
		per_face_material_id, materials_host] = load_meshes("mitsuba.obj");
	IntC material_id = IntC::copy(per_face_material_id.data(), per_face_material_id.size());
	CUDAArray<LambertBsdf *> materials = CUDAArray<LambertBsdf *>::copy(raw_ptrs(materials_host).data(), materials_host.size());

	// setup optix back end
	OptixBackend optix_backend;
	optix_backend.set_triangles_soup(per_face_position_indices, positions,
									 per_face_shading_normal_indices, shading_normals,
									 per_face_texcoord_indices, texcoords);

	// init film
	SpectrumC film = zero<SpectrumC>(width * height);

	// init timer
	StopWatch sw;
	sw.reset();

	// init RNG
	PCG32<RealC> rng(PCG32_DEFAULT_STATE, arange<RealC>(num_pixels));

	// sample thinlens
	const IntC pixel_index = arange<IntC>(num_pixels);
	const IntC y = pixel_index / width;
	const IntC x = pixel_index % width;
	const Int2C pixel(x, y);
	const ThinlensCamera thinlens(Real3(0.0_f, 3.03_f, 5.0_f), Real3(0.0_f, 0.03_f, 0.0_f), Real3(0.0_f, 1.0_f, 0.0_f), 0.00_f, 1.0_f, 70.0_f / 180.0_f * M_PI_f);

	for (int i = 0; i < num_samples; i++)
	{
		// print iter
		std::cout << i << std::endl;

		// init per iteration path contrib and active path
		SpectrumC contrib = full<SpectrumC>(1.0_f, num_pixels);
		BoolC active = true;

		// sample origin and direction from thinlens
		Real3C origin = thinlens.sample_pos(Real2C(rng.next_float32(), rng.next_float32()));
		Real3C direction = thinlens.sample_dir(pixel, origin, Int2(width, height), Real2C(rng.next_float32(), rng.next_float32()));

		// for all bounces
		for (int j = 0; j < num_bounces; j++)
		{
			// make ray
			const Ray3C rays(origin, direction);

			// intersection test
			auto [hit_info, is_intersect] = optix_backend.intersect(rays, active);

			// check if hit lightsource
			film += select(!is_intersect && active, contrib, full<SpectrumC>(0.0_f, num_pixels));

			// check which path still active
			active &= is_intersect;

			// create coord frame
			const Real2C scatter_sample = Real2C(rng.next_float32(), rng.next_float32());
			const Frame3C coord_frame(hit_info.m_shading_normal);
#if 1
			// fetch bsdf
			const IntC mat_index = gather<IntC>(material_id, hit_info.m_tri_id, active);
			const CUDAArray<Bsdf *> bsdf = gather<CUDAArray<Bsdf *>>(materials, mat_index, active);

			// sample outgoing direction
			const Real3C incoming_local = coord_frame.to_local(-direction, active);
			auto [bsdf_contrib, outgoing_local] = bsdf->sample(incoming_local, hit_info.m_position, scatter_sample, active);
			const Real3C outgoing = coord_frame.to_world(outgoing_local, active);
#else
			const Real3C outgoing_local = cosine_weighted_hemisphere_from_square(scatter_sample);
			const Real3C outgoing = coord_frame.to_world(outgoing_local, active);
			const SpectrumC bsdf_contrib = 0.5_f;
#endif

			// update variables for next bounce
			contrib *= bsdf_contrib;
			origin = hit_info.m_position;
			direction = outgoing;
		}
	}

	// normalize the film by number of samples
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