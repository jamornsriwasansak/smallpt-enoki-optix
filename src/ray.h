#pragma once

#include "enoki_entry.h"

template <typename Vector_>
struct Ray
{
	using Vector = Vector_;
	using Value = value_t<Vector>;

	Ray():
		m_origin(Vector::zero_()),
		m_dir(Vector::zero_()),
		m_tmin(0),
		m_tmax(1e34)
	{
	}

	Ray(const Vector & origin, const Vector & dir, Value tmin, Value tmax) :
		m_origin(origin),
		m_dir(dir),
		m_tmin(tmin),
		m_tmax(tmax)
	{
	}

	Vector	m_origin;
	Vector	m_dir;
	Value	m_tmin;
	Value	m_tmax;
};

using Ray3 = Ray<Vec3>;
using Ray3P = Ray<Vec3P>;
using Ray3C = Ray<Vec3C>;
