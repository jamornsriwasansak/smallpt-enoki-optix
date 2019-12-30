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

#include <optix.h>
#include <optix_prime/optix_prime.h>
#include <optix_prime/optix_prime_declarations.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

#include <tiny_obj_loader.h>

inline int idivCeil(int x, int y)
{
	return (x + y - 1) / y;
}

Ray3C enokiCreateRaysOrtho(int width, int * height, const float3 & bbmin, const float3 & bbmax, const float margin)
{
	float3 bbspan = bbmax - bbmin;

	// set height according to aspect ratio of bounding box    
	//*rays = new PrimeRay[width * *height];
	*height = (int)(width * bbspan.y / bbspan.x);

	float dx = bbspan.x * (1 + 2 * margin) / width;
	float dy = bbspan.y * (1 + 2 * margin) / *height;
	float x0 = bbmin.x - bbspan.x * margin + dx / 2;
	float y0 = bbmin.y - bbspan.y * margin + dy / 2;
	float z = bbmin.z - std::max(bbspan.z, 1.0f) * .001f;
	int rows = idivCeil((*height - 0), 1);

	// get pixel index
	int num_pixels = width * (*height);
	auto pixel_index = arange<IntC>(num_pixels);

	IntC y = pixel_index / width;
	IntC x = pixel_index % width;

	Vec3C origin(x0 + x * dx, y0 + y * dy, z);
	Vec3C direction(0.0_f + x*0, 0.0_f + x*0, 1.0_f + x*0);

	Ray3C result(origin, direction, 0.0, 1e34);
	return result;
}

RealC primeRaysFromRays(const Ray3C & rays)
{
	RealC prime_rays = zero<RealC>(rays.m_origin.x().size() * 8);
	auto indices = arange<IntC>(rays.m_origin.x().size()) * 8;
	scatter(prime_rays, rays.m_origin.x(), indices + 0);
	scatter(prime_rays, rays.m_origin.y(), indices + 1);
	scatter(prime_rays, rays.m_origin.z(), indices + 2);
	scatter(prime_rays, rays.m_tmin, indices + 3);
	scatter(prime_rays, rays.m_dir.x(), indices + 4);
	scatter(prime_rays, rays.m_dir.y(), indices + 5);
	scatter(prime_rays, rays.m_dir.z(), indices + 6);
	scatter(prime_rays, rays.m_tmax, indices + 7);
	return prime_rays;
}

struct Test
{
	float x;
};

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
	int width = 640;
	int height;
	Ray3C enoki_rays = enokiCreateRaysOrtho(width, &height, make_float3(-0.080734, -0.002271, -0.026525), make_float3(0.080813, 0.095336, 0.025957), 0.05f);
	prime_backend.intersect(enoki_rays);

	return 0;
}