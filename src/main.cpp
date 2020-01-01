#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <iostream>

#include <cuda_runtime.h>

#include "accel/optix_prime_backend.h"
#include "ray.h"
#include "enoki_entry.h"
#include "fimage.h"
#include "pinhole.h"
#include "mapping.h"
#include "coordframe.h"

#include <optix.h>
#include <optix_prime/optix_prime.h>
#include <optix_prime/optix_prime_declarations.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

#include <tiny_obj_loader.h>

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

	const size_t num_triangles = tiny_shape.mesh.num_face_vertices.size();
	const size_t num_vertices = tiny_attrib.vertices.size() / 3;

	int3 * triangles = new int3[num_triangles];
	float3 * vertices = new float3[num_vertices];

	// copy triangles
	for (size_t i_face = 0; i_face < num_triangles; i_face++)
	{
		triangles[i_face].x = tiny_shape.mesh.indices[i_face * 3 + 0].vertex_index;
		triangles[i_face].y = tiny_shape.mesh.indices[i_face * 3 + 1].vertex_index;
		triangles[i_face].z = tiny_shape.mesh.indices[i_face * 3 + 2].vertex_index;
	}

	// copy vertices
	for (size_t i_vertex = 0; i_vertex < num_vertices; i_vertex++)
	{
		vertices[i_vertex].x = tiny_attrib.vertices[i_vertex * 3 + 0];
		vertices[i_vertex].y = tiny_attrib.vertices[i_vertex * 3 + 1];
		vertices[i_vertex].z = tiny_attrib.vertices[i_vertex * 3 + 2];
	}

	OptixPrimeBackend prime_backend;
	prime_backend.set_triangles_soup(triangles, num_triangles, vertices, num_vertices);
	int width = 1920;
	int height = 1080;

	RealC film = zero<RealC>(width * height);

	PCG32<RealC> rng(PCG32_DEFAULT_STATE, arange<RealC>(width * height));
	for (int i = 0; i < 100; i++)
	{
		int num_pixels = width * height;
		const IntC pixel_index = arange<IntC>(num_pixels);
		const IntC y = pixel_index / width;
		const IntC x = pixel_index % width;
		const Int2C pixel(x, y);
		ThinlensCamera thinlens(Real3(0.0_f, 0.03_f, 0.2_f), Real3(0.0_f, 0.03_f, 0.0_f), Real3(0.0_f, 1.0_f, 0.0_f), 70.0_f / 180.0_f * M_PI);
		const Real3C origin = thinlens.m_origin + zero<Real3C>(width * height);
		const Real3C direction = thinlens.sample(pixel, Int2(width, height), rng.next_float32(), rng.next_float32());
		const Ray3C rays(origin, direction, 0.0_f, 1e20_f);
		const TriangleHitInfoC hit_info = prime_backend.intersect(rays);
		const CoordFrame3C coord_frame(hit_info.m_geometry_normal);

		// sample out going direction
		const Real3C second_direction = coord_frame.to_world(cosine_weighted_hemisphere_from_square(rng.next_float32(), rng.next_float32()));
		const Ray3C second_rays(hit_info.m_position, second_direction, 0.01_f, 1e20_f);
		const TriangleHitInfoC second_hit_info = prime_backend.intersect(second_rays);

		film += select(eq(second_hit_info.m_tri_id, -1), zero<RealC>(num_pixels) + 1.0_f, zero<RealC>(num_pixels) + 0.0_f);
	}

	film /= 100.0_f;

	// image write
	float * film_host = new float[width * height];
	cuda_fetch_element(film_host, film.index_(), 0, sizeof(int) * width * height);

	std::cout << "writing image" << std::endl;
	Fimage::save_pfm(film_host, width, height, "test.pfm");

	std::cout << "done" << std::endl;
	return 0;
}