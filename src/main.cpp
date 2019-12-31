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

	PCG32<RealC> rng(PCG32_DEFAULT_STATE, arange<RealC>(width * height));
	int num_pixels = width * height;
	const IntC pixel_index = arange<IntC>(num_pixels);
	const IntC y = pixel_index / width;
	const IntC x = pixel_index % width;
	const Int2C pixel(x, y);
	ThinlensCamera thinlens(Real3(0.0_f, 0.03_f, 0.2_f), Real3(0.0_f, 0.03_f, 0.0_f), Real3(0.0_f, 1.0_f, 0.0_f), 70.0_f / 180.0_f * M_PI);
	Real3C origin = thinlens.m_origin + zero<Real3C>(width * height);
	Real3C direction = thinlens.sample(pixel, Int2(width, height), rng.next_float32(), rng.next_float32());
	Ray3C rays(origin, direction, 0.0_f, 1e20_f);
	TriangleHitInfoC hit_info = prime_backend.intersect(rays);

	// position
	float * image = new float[width * height * 3];
	float * p = new float[width * height * 3];
	int * tri_id = new int[width * height];
	cuda_fetch_element(p, hit_info.m_position.x().index_(), 0, sizeof(int) * width * height);

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			/*
			if (tri_id[x + y * width] != -1)
			{
				image[x + y * width] = 1;
			}
			else
			{
				image[x + y * width] = 0;
			}
			*/
			image[x + y * width] = p[x + y * width];
		}

	Fimage::save_pfm(image, width, height, "test.pfm");

	return 0;
}