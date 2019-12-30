#pragma once

#include "ray.h"

#include <optix.h>
#include <optix_prime/optix_prime.h>
#include <optix_prime/optix_prime_declarations.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

struct PrimeRay
{
	static const RTPbufferformat Format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;
	static const size_t SizeInFloats = 8;

	float	m_origin_x;
	float	m_origin_y;
	float	m_origin_z;
	float	m_tmin;
	float	m_dir_x;
	float	m_dir_y;
	float	m_dir_z;
	float	m_tmax;
};

struct PrimeHit
{
	static const RTPbufferformat Format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;
	static const size_t SizeInFloats = 4;

	float	m_t;
	int		m_tri_id;
	float	m_barycentric_u;
	float	m_barycentric_v;
};

struct TriangleHitInfo
{
	TriangleHitInfo(const RealC & t, const IntC & tri_id, const Vec2C & barycentric):
		m_t(t),
		m_tri_id(tri_id),
		m_barycentric(barycentric)
	{
	}

	RealC	m_t;
	IntC	m_tri_id;
	Vec2C	m_barycentric;
};

struct OptixPrimeBackend
{
	static const RTPcontexttype context_type = RTP_CONTEXT_TYPE_CUDA;
	static const RTPbuffertype buffer_type = RTP_BUFFER_TYPE_CUDA_LINEAR;

	OptixPrimeBackend()
	{
		m_context = optix::prime::Context::create(context_type);
		unsigned int device = 0;
		m_context->setCudaDeviceNumbers(1, &device);
	}

	void set_triangles_soup(const int3 * triangles, const size_t num_triangles, const float3 * vertices, const size_t num_vertices)
	{
		// create model
		m_model = m_context->createModel();
		m_model->setTriangles(num_triangles, RTP_BUFFER_TYPE_HOST, triangles,
							  num_vertices,  RTP_BUFFER_TYPE_HOST, vertices);
		m_model->update(RTPqueryhint::RTP_QUERY_HINT_NONE);
		// create query
		m_query = m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
	}

	RealC prime_rays_from_rays(const Ray3C & rays)
	{
		const size_t num_rays = rays.m_origin.x().size();
		const IntC indices = arange<IntC>(num_rays) * PrimeRay::SizeInFloats;
		RealC prime_rays = zero<RealC>(num_rays * PrimeRay::SizeInFloats);
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

	static TriangleHitInfo hit_info_from_prime_hit(const RealC & hits)
	{
		const size_t num_hits = hits.size() / PrimeHit::SizeInFloats;
		const IntC indices = arange<IntC>(num_hits) * PrimeHit::SizeInFloats;
		const RealC t = gather<RealC>(hits, indices + 0);
		const IntC tri_id = gather<IntC>(hits, indices + 1);
		const RealC barycentric_u = gather<RealC>(hits, indices + 2);
		const RealC barycentric_v = gather<RealC>(hits, indices + 3);
		Vec2C barycentric(barycentric_u, barycentric_v);
		TriangleHitInfo result(t, tri_id, barycentric);
		return result;
	}

	TriangleHitInfo intersect(const Ray3C & rays)
	{
		RealC prime_rays = prime_rays_from_rays(rays);
		size_t num_rays = rays.m_origin.x().size();
		RealC prime_hits = zero<RealC>(num_rays * PrimeHit::SizeInFloats);
		// cuda_eval is needed here to make sure that prime_rays and prime_hits are ready.
		cuda_eval();
		m_query->setRays(num_rays, PrimeRay::Format, RTP_BUFFER_TYPE_CUDA_LINEAR, prime_rays.data());
		m_query->setHits(num_rays, PrimeHit::Format, RTP_BUFFER_TYPE_CUDA_LINEAR, prime_hits.data());
		m_query->execute(0);
		TriangleHitInfo result = hit_info_from_prime_hit(prime_hits);
		return result;
	}

	optix::prime::Context	m_context;
	optix::prime::Query		m_query;
	optix::prime::Model		m_model;
};