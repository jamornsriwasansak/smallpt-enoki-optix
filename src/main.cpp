#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <iostream>

#include <cuda_runtime.h>

#include <enoki/cuda.h>
#include <enoki/array.h>
#include <enoki/autodiff.h>
#include <enoki/cuda.h>
#include <enoki/stl.h>

#include <optix.h>
#include <optix_prime/optix_prime.h>
#include <optix_prime/optix_prime_declarations.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

#include <tiny_obj_loader.h>

struct Ray
{
	static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

	float3 origin;
	float  tmin;
	float3 dir;
	float  tmax;
};

struct Hit
{
	static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

	float t;
	int   triId;
	float u;
	float v;
};

template <typename Value> Value srgb_gamma(Value x) {
	return enoki::select(
		x <= 0.0031308f,
		x * 12.92f,
		enoki::pow(x * 1.055f, 1.f / 2.4f) - 0.055f
	);
}

inline int idivCeil(int x, int y)
{
	return (x + y - 1) / y;
}


void createRaysOrtho(Ray ** rays, int width, int * height, const float3 & bbmin, const float3 & bbmax, float margin)
{
	float3 bbspan = bbmax - bbmin;

	// set height according to aspect ratio of bounding box    
	*height = (int)(width * bbspan.y / bbspan.x);

	*rays = new Ray[width * *height];

	float dx = bbspan.x * (1 + 2 * margin) / width;
	float dy = bbspan.y * (1 + 2 * margin) / *height;
	float x0 = bbmin.x - bbspan.x * margin + dx / 2;
	float y0 = bbmin.y - bbspan.y * margin + dy / 2;
	float z = bbmin.z - std::max(bbspan.z, 1.0f) * .001f;
	int rows = idivCeil((*height - 0), 1);

	float y = y0;
	size_t idx = 0;
	for (int iy = 0; iy < *height; iy += 1)
	{
		float x = x0;
		for (int ix = 0; ix < width; ix++)
		{
			float tminOrMask = 0.0f;
			Ray r = { make_float3(x,y,z), tminOrMask, make_float3(0,0,1), 1e34f };
			(*rays)[idx++] = r;
			x += dx;
		}
		y += dy * 1;
	}
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

#if 1
	RTPcontexttype context_type = RTP_CONTEXT_TYPE_CUDA;
	RTPbuffertype buffer_type = RTP_BUFFER_TYPE_CUDA_LINEAR;
#else
	RTPcontexttype contextType = RTP_CONTEXT_TYPE_CPU;
	RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
#endif

	//try
	{
		optix::prime::Context context = optix::prime::Context::create(context_type);
		unsigned int device = 0;
		context->setCudaDeviceNumbers(1, &device);

		// model
		optix::prime::Model model = context->createModel();
		model->setTriangles(num_triangles, RTP_BUFFER_TYPE_HOST, triangles,
							num_vertices,  RTP_BUFFER_TYPE_HOST, vertices);
		model->update(RTPqueryhint::RTP_QUERY_HINT_NONE);

		// query
		// TODO:: initialize hits and rays


		int width = 640;

#if 0
		Hit * hits;
		Ray * rays;
		cudaMalloc(&rays, sizeof(Ray) * 100);
		cudaMalloc(&hits, sizeof(Hit) * 100);
		query->setRays(100, Ray::format, buffer_type, rays);
		query->setHits(100, Hit::format, buffer_type, hits);
		query->execute(RTPqueryhint::RTP_QUERY_HINT_NONE);
#else
		Hit * hits = nullptr;
		Ray * rays = nullptr;
		//rays = new Ray[width * height];
		int height;
		createRaysOrtho(&rays, width, &height, make_float3(-0.080734, -0.002271, -0.026525), make_float3(0.080813, 0.095336, 0.025957), 0.05f);
		hits = new Hit[width * height];
		optix::prime::Query query = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
		query->setRays(width * height, Ray::format, RTP_BUFFER_TYPE_HOST, rays);
		query->setHits(width * height, Hit::format, RTP_BUFFER_TYPE_HOST, hits);
		query->execute(0);

		for (int i = 0; i < width * height; i++)
		{
			if (hits[i].triId != -1)
				std::cout << hits[i].triId << std::endl;
		}
#endif
	}
	//catch (const std::exception & e)
	{
		//std::cout << e.what() << std::endl;
	}

	using FloatC = enoki::CUDAArray<float>;
	using FloatD = enoki::DiffArray<FloatC>;
	using Color3fD = enoki::Array<FloatD, 3>;

	FloatC f(1.0f);
	FloatC g(1.0f);
	std::cout << "f + g : " << f + g << std::endl;
	std::cout << f.data() << std::endl;
	std::cout << "f + g : " << f + g << std::endl;
	std::cout << f.data() << std::endl;
	std::cout << "f + g : " << f + g << std::endl;
	std::cout << f.data() << std::endl;
	std::cout << "f + g : " << f + g << std::endl;
	std::cout << f.data() << std::endl;

	Color3fD input = Color3fD(0.5f, 1.0f, 2.0f);
	enoki::set_requires_gradient(input);

	Color3fD output = srgb_gamma(input);

	std::cout << "test1" << std::endl;
	FloatD loss = enoki::norm(output - Color3fD(.1f, .2f, .3f));
	enoki::backward(loss);
	std::cout << "test2" << std::endl;
	std::cout << enoki::gradient(input) << std::endl;
	std::cout << "test3" << std::endl;
	return 0;
}