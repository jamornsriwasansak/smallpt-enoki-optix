#pragma once

#include "enoki_entry.h"

template <typename RealNT_>
struct Ray
{
	using RealNT = RealNT_;
	using RealT = value_t<RealNT>;

	Ray()
	{
	}

	Ray(const RealNT & origin, const RealNT & dir):
		Ray(origin, dir, 0.001_f, 1e20_f)
	{
	}

	Ray(const RealNT & origin, const RealNT & dir, RealT tmin, RealT tmax) :
		m_origin(origin),
		m_dir(dir),
		m_tmin(zero<RealT>(origin.x().size()) + tmin),
		m_tmax(zero<RealT>(origin.x().size()) + tmax)
	{
	}

	RealNT	m_origin;
	RealNT	m_dir;
	RealT	m_tmin;
	RealT	m_tmax;
};

using Ray3S = Ray<Real3>;
using Ray3P = Ray<Real3P>;
using Ray3C = Ray<Real3C>;
