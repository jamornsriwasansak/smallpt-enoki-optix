#pragma once

#include "enoki_entry.h"
#include "ray.h"
#include "add_math.h"

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

template <typename Int_, typename Real_>
struct TriangleHitInfo
{
	using RealT = Real_;
	using IntT = Int_;
	using Real2T = Array<RealT, 2, false>;
	using Real3T = Array<RealT, 3, false>;

	TriangleHitInfo(const RealT & t, const IntT & tri_id, const Real2T & barycentric, const Real3T & position):
		m_t(t),
		m_tri_id(tri_id),
		m_barycentric(barycentric),
		m_position(position)
	{
	}

	RealT	m_t;
	IntT	m_tri_id;
	Real2T	m_barycentric;
	Real3T	m_position;
};
using TriangleHitInfoC = TriangleHitInfo<IntC, RealC>;

struct OptixPrimeBackend
{
	static const RTPcontexttype context_type = RTP_CONTEXT_TYPE_CUDA;
	static const RTPbuffertype buffer_type = RTP_BUFFER_TYPE_CUDA_LINEAR;

	OptixPrimeBackend():
		m_context(optix::prime::Context::create(context_type)),
		m_model(),
		m_query(),
		m_triangles(empty<Int3C>()),
		m_vertices(empty<Real3C>())
	{
		unsigned int device = 0;
		m_context->setCudaDeviceNumbers(1, &device);
	}

	void set_triangles_soup(const int3 * triangles_host, const size_t num_triangles, const float3 * vertices_host, const size_t num_vertices)
	{
		IntC triangles = IntC::copy(triangles_host, 3 * num_triangles);
		RealC vertices = RealC::copy(vertices_host, 3 * num_vertices);

		// create model
		m_model = m_context->createModel();
		m_model->setTriangles(num_triangles, RTP_BUFFER_TYPE_CUDA_LINEAR, triangles.data(),
							  num_vertices,  RTP_BUFFER_TYPE_CUDA_LINEAR, vertices.data());
		m_model->update(RTPqueryhint::RTP_QUERY_HINT_NONE);

		// create query
		m_query = m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);

		// convert AoS to SoA for storing triangles internally
		const IntC t_indices = arange<IntC>(num_triangles);
		const IntC t0 = gather<IntC>(triangles, t_indices * 3 + 0);
		const IntC t1 = gather<IntC>(triangles, t_indices * 3 + 1);
		const IntC t2 = gather<IntC>(triangles, t_indices * 3 + 2);
		m_triangles = Int3C(t0, t1, t2);
		const IntC p_indices = arange<IntC>(num_vertices);
		const RealC p0 = gather<RealC>(vertices, p_indices * 3 + 0);
		const RealC p1 = gather<RealC>(vertices, p_indices * 3 + 1);
		const RealC p2 = gather<RealC>(vertices, p_indices * 3 + 2);
		m_vertices = Real3C(p0, p1, p2);
	}

	RealC prime_rays_from_rays(const Ray3C & rays)
	{
		const size_t num_rays = rays.m_dir.x().size();
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

	TriangleHitInfoC hit_info_from_prime_hit(const Ray3C & rays, const RealC & hits)
	{
		const size_t num_hits = hits.size() / PrimeHit::SizeInFloats;
		const IntC indices = arange<IntC>(num_hits) * PrimeHit::SizeInFloats;
		const RealC t = gather<RealC>(hits, indices + 0);
		const IntC tri_id = gather<IntC>(hits, indices + 1);
		const RealC barycentric_w = gather<RealC>(hits, indices + 2);
		const RealC barycentric_u = gather<RealC>(hits, indices + 3);
		const RealC barycentric_v = 1.0_f - barycentric_w - barycentric_u;
		const Real2C barycentric(barycentric_u, barycentric_v);

		// compute position
		/*
		const Int3C tri_indices = gather<Int3C>(m_triangles, tri_id);
		const Real3C p0 = gather<Real3C>(m_vertices, tri_indices.x());
		const Real3C p1 = gather<Real3C>(m_vertices, tri_indices.y());
		const Real3C p2 = gather<Real3C>(m_vertices, tri_indices.z());
		const Real3C p = barycentric_interpolate(p0, p1, p2, barycentric);
		*/
		const Real3C p = rays.m_origin + rays.m_dir * t;
		TriangleHitInfoC result(t, tri_id, barycentric, p);
		return result;
	}

	TriangleHitInfoC intersect(const Ray3C & rays)
	{
		RealC prime_rays = prime_rays_from_rays(rays);
		size_t num_rays = rays.m_origin.x().size();
		RealC prime_hits = zero<RealC>(num_rays * PrimeHit::SizeInFloats);
		// cuda_eval is needed here to make sure that prime_rays and prime_hits are ready.
		cuda_eval();
		m_query->setRays(num_rays, PrimeRay::Format, RTP_BUFFER_TYPE_CUDA_LINEAR, prime_rays.data());
		m_query->setHits(num_rays, PrimeHit::Format, RTP_BUFFER_TYPE_CUDA_LINEAR, prime_hits.data());
		m_query->execute(0);
		TriangleHitInfoC result = hit_info_from_prime_hit(rays, prime_hits);
		return result;
	}

	Int3C					m_triangles;
	Real3C					m_vertices;
	optix::prime::Context	m_context;
	optix::prime::Query		m_query;
	optix::prime::Model		m_model;
};