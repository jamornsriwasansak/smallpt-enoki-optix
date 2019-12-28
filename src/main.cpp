#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <iostream>
#include "enoki/cuda.h"
#include "enoki/array.h"
#include "enoki/autodiff.h"
#include "enoki/dynamic.h"
#include "enoki/cuda.h"
#include "enoki/stl.h"

#include <optix.h>
#include <optix_prime/optix_prime.h>
#include <optix_prime/optix_prime_declarations.h>
#include <optix_prime/optix_primepp.h>
#include <cuda_runtime.h>

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

struct HitInstancing
{
	static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V;

	float t;
	int   triId;
	int   instId;
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

int main()
{
#if 1
	RTPcontexttype context_type = RTP_CONTEXT_TYPE_CUDA;
	RTPbuffertype buffer_type = RTP_BUFFER_TYPE_CUDA_LINEAR;
#else
	RTPcontexttype contextType = RTP_CONTEXT_TYPE_CPU;
	RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
#endif

	try
	{
		optix::prime::Context context = optix::prime::Context::create(context_type);
		unsigned int device = 0;
		context->setCudaDeviceNumbers(1, &device);

		int num_triangles = 0;
		int num_vertices = 0;
		int3 *triangles;
		float3 * vertices;

		// model
		optix::prime::Model model = context->createModel();
		model->setTriangles(num_triangles, RTP_BUFFER_TYPE_HOST, triangles,
							num_vertices,  RTP_BUFFER_TYPE_HOST, vertices);
		model->update(RTPqueryhint::RTP_QUERY_HINT_NONE);

		// query
		optix::prime::Query query = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
		// TODO:: initialize hits and rays
		Hit * hits;
		Ray * rays;
		cudaMalloc(&rays, sizeof(Ray) * 100);
		cudaMalloc(&hits, sizeof(Hit) * 100);
		optix::prime::BufferDesc buffer_desc;
		query->setRays(100, Ray::format, buffer_type, rays);
		query->setHits(100, Hit::format, buffer_type, hits);
		query->execute(RTPqueryhint::RTP_QUERY_HINT_NONE);

	}
	catch (const std::exception & e)
	{
		std::cout << e.what() << std::endl;
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