#pragma once

#include "enoki_entry.h"

template <typename RealNT_>
struct Ray
{
	using RealNT = RealNT_;
	using RealT = value_t<RealNT>;

	Ray():
		m_origin(RealNT::zero_()),
		m_dir(RealNT::zero_()),
		m_tmin(0.0_f),
		m_tmax(1e34_f)
	{
	}

	Ray(const RealNT & origin, const RealNT & dir):
		m_origin(origin),
		m_dir(dir),
		m_tmin(0.0_f),
		m_tmax(1e34_f)
	{
	}

	Ray(const RealNT & origin, const RealNT & dir, RealT tmin, RealT tmax) :
		m_origin(origin),
		m_dir(dir),
		m_tmin(tmin),
		m_tmax(tmax)
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
