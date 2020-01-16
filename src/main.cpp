#define _CRT_SECURE_NO_WARNINGS
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

struct Light
{
};

using LightPtrC = CUDAArray<Light *>;

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

using BsdfPtrC = CUDAArray<Bsdf *>;

struct LambertBsdf : public Bsdf
{
	LambertBsdf(const Spectrum & reflectance):
		m_reflectance(std::make_shared<ImageTexture>(reflectance))
	{
	}

	std::tuple<SpectrumC, Real3C> sample(const Real3C & incoming,
										 const Real3C & texcoord,
										 const Real2C & sample,
										 const BoolC & mask = true) const override
	{
		const Real3C outgoing = cosine_weighted_hemisphere_from_square(sample);
		const SpectrumC contrib = m_reflectance->eval(texcoord, mask);
		return std::make_tuple(contrib, outgoing);
	}

	std::shared_ptr<ImageTexture> m_reflectance;
};

RealC schlick_weight(const RealC & cos_theta)
{
	return pow(1.0_f - cos_theta, 5);
}

RealC schlick_approximation(const RealC & f0, const RealC & cos_theta)
{
	return lerp(schlick_weight(cos_theta), 1.0_f, f0);
}

SpectrumC schlick_approx(const RealC & cos_theta_i,
						 const RealC & eta,
						 const BoolC & mask = 0)
{
	const RealC f0 = sqr((1.0_f - eta) / (1.0_f + eta));
	const RealC sqr_cos_theta_t = 1.0_f - (1.0_f - sqr(cos_theta_i)) / sqr(eta);
	return select(mask && (sqr_cos_theta_t > 0.0_f),
				  schlick_approximation(f0, cos_theta_i),
				  1.0_f);
}

struct SpecBsdf : public Bsdf
{
	SpecBsdf(const SpectrumC & reflectance)
	{
	}

	SpectrumC eval(const Real3C & v,
				   const Real3C & l,
				   const Real3C & texcoord,
				   const BoolC & mask = true) const
	{
		const Real3C h = normalize(v + l);

		// compute F - fresnel reflection coefficient
		const RealC eta = 1.0_f;
		const SpectrumC f_term = schlick_approx(v.y(), eta, mask);

		// compute G - geometric distribution / shadowing factor


		// compute D - microfacet distribution term

		return f_term;
	}

	std::tuple<SpectrumC, Real3C> sample(const Real3C & in,
										 const Real3C & texcoord,
										 const Real2C & sample,
										 const BoolC & mask = true) const override
	{
		const Real3C outgoing = cosine_weighted_hemisphere_from_square(sample);
		const SpectrumC contrib = eval(in, outgoing, texcoord, mask) * M_PI_f;
		return std::make_tuple(contrib, outgoing);
	}
};

struct DiffuseBsdf : public Bsdf
{
	DiffuseBsdf(const SpectrumC & reflectance)
	{
	}

	SpectrumC eval(const Real3C & v,
				   const Real3C & l,
				   const Real3C & texcoord,
				   const BoolC & mask = true) const
	{
		const SpectrumC base_color = 1.0_f;
		const SpectrumC roughness = 1.0_f;

		// Brent Burley 2015 eq.4 
		const Real3C half = normalize(v + l);
		const RealC cos_theta_d = dot(half, v);
		const SpectrumC rr = 2.0_f * roughness * sqr(cos_theta_d);
		const SpectrumC fv = schlick_weight(abs(v.y()));
		const SpectrumC fl = schlick_weight(abs(l.y()));
		const SpectrumC f_lambert = base_color * M_1_PI_f * (1.0_f - 0.5_f * fl) * (1.0_f - 0.5_f * fv);
		const SpectrumC f_retro_reflect = base_color * M_1_PI_f * rr * (fl + fv + fl * fv * (rr - 1.0_f));
		const SpectrumC f_d = f_lambert + f_retro_reflect;
		return f_d;
	}

	std::tuple<SpectrumC, Real3C> sample(const Real3C & v,
										 const Real3C & texcoord,
										 const Real2C & sample,
										 const BoolC & mask = true) const override
	{
		const Real3C outgoing = cosine_weighted_hemisphere_from_square(sample);
		const SpectrumC contrib = eval(v, outgoing, texcoord, mask) * M_PI_f;
		return std::make_tuple(contrib, outgoing);
	}
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
	std::vector<std::shared_ptr<Bsdf>>> load_meshes(const std::filesystem::path & path)
{
	// load mesh
	tinyobj::ObjReader obj_reader;
	tinyobj::ObjReaderConfig obj_reader_config;
	obj_reader_config.triangulate = true;
	obj_reader_config.vertex_color = false;
	bool ret = obj_reader.ParseFromFile(path.string(), obj_reader_config);

	// compute num_all_triangles
	const size_t num_shapes = obj_reader.GetShapes().size();
	size_t num_all_triangles = 0;
	for (size_t i_shape = 0; i_shape < num_shapes; i_shape++)
	{
		const tinyobj::shape_t & tiny_shape = obj_reader.GetShapes()[i_shape];
		num_all_triangles += tiny_shape.mesh.num_face_vertices.size();
	}

	// get triangles and material_id
	std::vector<int3> position_triplets(num_all_triangles);
	std::vector<int3> shading_normal_triplets(num_all_triangles);
	std::vector<int3> texcoord_triplets(num_all_triangles);
	std::vector<int> per_face_material_id(num_all_triangles);
	const std::vector<tinyobj::material_t> & tiny_mat = obj_reader.GetMaterials();
	size_t i_face_offset = 0;
	for (size_t i_shape = 0; i_shape < num_shapes; i_shape++)
	{
		const tinyobj::shape_t & tiny_shape = obj_reader.GetShapes()[i_shape];
		const size_t num_triangles = tiny_shape.mesh.num_face_vertices.size();
		for (size_t i_face = 0; i_face < num_triangles; i_face++)
		{
			position_triplets[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].vertex_index;
			position_triplets[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].vertex_index;
			position_triplets[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].vertex_index;

			shading_normal_triplets[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].normal_index;
			shading_normal_triplets[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].normal_index;
			shading_normal_triplets[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].normal_index;

			texcoord_triplets[i_face + i_face_offset].x = tiny_shape.mesh.indices[i_face * 3 + 0].texcoord_index;
			texcoord_triplets[i_face + i_face_offset].y = tiny_shape.mesh.indices[i_face * 3 + 1].texcoord_index;
			texcoord_triplets[i_face + i_face_offset].z = tiny_shape.mesh.indices[i_face * 3 + 2].texcoord_index;

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
	std::vector<std::shared_ptr<Bsdf>> materials(num_materials + 1);

	// start with default material at 0
	//materials[0] = std::make_shared<LambertBsdf>(Spectrum(0.5_f));
	//materials[0] = std::make_shared<DiffuseBsdf>(Spectrum(0.5_f));
	materials[0] = std::make_shared<DiffuseBsdf>(Spectrum(1.0_f));

	// push the rest of materials
	for (size_t i_material = 0; i_material < num_materials; i_material++)
	{
		const Real r = tiny_mat[i_material].diffuse[0];
		const Real g = tiny_mat[i_material].diffuse[1];
		const Real b = tiny_mat[i_material].diffuse[2];
		//materials[i_material + 1] = std::make_shared<LambertBsdf>(Spectrum(r, g, b));
		//materials[i_material + 1] = std::make_shared<DiffuseBsdf>(Spectrum(0.5_f));
		materials[i_material + 1] = std::make_shared<DiffuseBsdf>(Spectrum(1.0_f));
	}

	return std::make_tuple(position_triplets, positions,
						   shading_normal_triplets, shading_normals,
						   texcoord_triplets, texcoords,
						   per_face_material_id, materials);
}

#include "enoki_entry.h"

struct Interaction
{
	Interaction(const Real3C & position,
				const Real3C & shading_normal,
				const Real3C & geometry_normal,
				const Real3C & texcoord,
				const BoolC & is_in_medium,
				const BsdfPtrC & scatter,
				const BoolC & has_light,
				const LightPtrC & light):
		m_position(position),
		m_shading_normal(shading_normal),
		m_geometry_normal(geometry_normal),
		m_texcoord(texcoord),
		m_is_in_medium(is_in_medium),
		m_scatter(scatter),
		m_has_light(has_light),
		m_light(light)
	{
	}

	Real3C			m_position;
	Real3C			m_shading_normal;
	Real3C			m_geometry_normal;
	Real3C			m_texcoord;
	BoolC			m_is_in_medium;
	BsdfPtrC		m_scatter;
	BoolC			m_has_light;
	LightPtrC		m_light;
};

struct Scene
{
	void add_triangle_mesh(const std::filesystem::path & obj_filepath)
	{
		auto [position_triplets_host, positions_host,
			shading_normal_triplets_host, shading_normals_host,
			texcoord_triplets_host, texcoords_host,
			per_face_material_id, materials_host] = load_meshes(obj_filepath);

		// copy position CPU -> GPU
		m_position_triplets = IntC::copy(position_triplets_host.data(), 3 * position_triplets_host.size());
		m_positions = RealC::copy(positions_host.data(), 3 * positions_host.size());

		// copy shading normal CPU -> GPU
		m_shading_normal_triplets = IntC::copy(shading_normal_triplets_host.data(), 3 * shading_normal_triplets_host.size());
		m_shading_normals = RealC::copy(shading_normals_host.data(), 3 * shading_normals_host.size());

		// copy texcoords CPU -> GPU
		m_texcoord_triplets = IntC::copy(texcoord_triplets_host.data(), 3 * texcoord_triplets_host.size());
		m_texcoords = RealC::copy(texcoords_host.data(), 2 * texcoords_host.size());

		m_materials_host = materials_host;
		m_material_id = IntC::copy(per_face_material_id.data(), per_face_material_id.size());
		m_materials = CUDAArray<Bsdf *>::copy(raw_ptrs(m_materials_host).data(), m_materials_host.size());
	}

	void commit()
	{
		m_optix_backend.init();
		m_optix_backend.set_triangles_soup(&m_position_triplets, &m_positions,
										   &m_shading_normal_triplets, &m_shading_normals,
										   &m_texcoord_triplets, &m_texcoords);
	}

	std::tuple<Interaction, BoolC> intersect(const Ray3C & ray, const BoolC & active)
	{
		auto [hit_info, is_intersect] = m_optix_backend.intersect(ray, active);

		// fetch bsdf
		const IntC mat_index = gather<IntC>(m_material_id, hit_info.m_tri_id, active);
		const BsdfPtrC bsdf = gather<CUDAArray<Bsdf *>>(m_materials, mat_index, active);

		// make result
		Interaction result(hit_info.m_position,
						   hit_info.m_shading_normal,
						   hit_info.m_geometry_normal,
						   Real3C(hit_info.m_texcoord.x(), hit_info.m_texcoord.y(), 0.0_f),
						   false,
						   bsdf,
						   false,
						   nullptr);

		// return
		return std::make_tuple(result, is_intersect);
	}

	IntC								m_position_triplets;
	RealC								m_positions;
	RealC								m_shading_normals;
	IntC								m_shading_normal_triplets;
	RealC								m_texcoords;
	IntC								m_texcoord_triplets;
	IntC								m_material_id;
	BsdfPtrC							m_materials;
	std::vector<std::shared_ptr<Bsdf>>	m_materials_host;
	OptixBackend						m_optix_backend;
};

int main()
{
	// all const
	const int width = 512;
	const int height = 512;
	const int num_pixels = width * height;
	const int num_samples = 100;
	const int num_bounces = 2;

	// load meshes and materials
	Scene scene;
	scene.add_triangle_mesh("mitsuba.obj");
	scene.commit();

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
	const ThinlensCamera thinlens(Real3(0.0_f, 3.03_f, 5.0_f), Real3(0.0_f, 0.03_f, 0.0_f), Real3(0.0_f, 1.0_f, 0.0_f), 0.00_f, 1.0_f, 40.0_f / 180.0_f * M_PI_f);

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
			const Ray3C ray(origin, direction);

			auto [interaction, is_intersect] = scene.intersect(ray, active);

			// check if hit lightsource
			film += select(!is_intersect && active, contrib, full<SpectrumC>(0.0_f, num_pixels));

			// check which path still active
			active &= is_intersect;

			// create coord frame
			const Real2C scatter_sample = Real2C(rng.next_float32(), rng.next_float32());
			const Frame3C coord_frame(interaction.m_shading_normal);

			// sample outgoing direction
			const Real3C incoming_local = coord_frame.to_local(-direction, active);
			auto [bsdf_contrib, outgoing_local] = interaction.m_scatter->sample(incoming_local, interaction.m_texcoord, scatter_sample, active);
			const Real3C outgoing = coord_frame.to_world(outgoing_local, active);

			// update variables for next bounce
			contrib *= bsdf_contrib;
			origin = interaction.m_position;
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
	Fimage::save_pfm(film_host_r, film_host_g, film_host_b, width, height, "wurst.pfm");

	std::cout << "done" << std::endl;
	return 0;
}